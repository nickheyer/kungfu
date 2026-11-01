"""Fits draft heads on the features kungfu drafter-data wrote and measures how many tokens each accepts per round.

Usage: python3 tools/drafter_study.py DIR [DIR ...] [--epochs N] [--eval-seqs N] [--device cuda]

Each DIR holds tokens.bin, features.bin, index.jsonl, embd.bin, norm0.bin, meta.json from one drafter-data run. The
first DIR's tables serve every run. Heads: a ridge regression from [f_t, e(x_{t+1})] to f_{t+1}, a two layer MLP with
the same input trained on the feature regression loss plus cross entropy through the frozen head, and a one layer
attention head over a window of the last features. Acceptance is measured by chaining each head on held out sequences
the model wrote greedily, so the text token at every position is the model's own argmax.
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_run(d):
    meta = json.load(open(os.path.join(d, "meta.json")))
    n_embd = meta["n_embd"]
    tokens = np.fromfile(os.path.join(d, "tokens.bin"), dtype=np.uint32)
    feats = np.memmap(os.path.join(d, "features.bin"), dtype=np.float16, mode="r").reshape(-1, n_embd)
    index = [json.loads(l) for l in open(os.path.join(d, "index.jsonl")) if l.strip()]
    n = min(len(tokens), feats.shape[0])
    index = [s for s in index if s["offset"] + s["n"] <= n]
    return meta, tokens, feats, index


def rms_norm(x, w, eps):
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps) * w


class Tables:
    def __init__(self, d, meta, device):
        n_embd, n_vocab = meta["n_embd"], meta["n_vocab"]
        e = np.fromfile(os.path.join(d, "embd.bin"), dtype=np.float16).reshape(n_vocab, n_embd)
        self.embd = torch.tensor(e, device=device, dtype=torch.float16)
        hp = os.path.join(d, "head.bin")
        self.head = torch.tensor(np.fromfile(hp, dtype=np.float16).reshape(n_vocab, n_embd), device=device, dtype=torch.float16) if os.path.exists(hp) else self.embd
        self.norm = torch.tensor(np.fromfile(os.path.join(d, "norm0.bin"), dtype=np.float32), device=device)
        self.eps = meta["rms_eps"]
        self.n_embd, self.n_vocab = n_embd, n_vocab

    def logits(self, f):
        return rms_norm(f.float(), self.norm, self.eps).half() @ self.head.t()


class Linear(nn.Module):
    """Ridge regression f_{t+1} ~ W [f_t, e_{t+1}, 1], solved in closed form"""

    def __init__(self, n_embd):
        super().__init__()
        self.W = nn.Parameter(torch.zeros(2 * n_embd + 1, n_embd), requires_grad=False)

    def forward(self, f, e):
        x = torch.cat([f, e, torch.ones_like(f[..., :1])], -1)
        return x @ self.W

    def fit(self, loader, device, lam=1e-2):
        d = self.W.shape[0]
        xtx = torch.zeros(d, d, device=device, dtype=torch.float64)
        xty = torch.zeros(d, self.W.shape[1], device=device, dtype=torch.float64)
        for f, e, y in loader:
            x = torch.cat([f, e, torch.ones_like(f[..., :1])], -1).double()
            xtx += x.t() @ x
            xty += x.t() @ y.double()
        xtx += lam * torch.eye(d, device=device, dtype=torch.float64) * xtx.diagonal().mean()
        self.W.data = torch.linalg.solve(xtx, xty).float()


class Mlp(nn.Module):
    """f_{t+1} ~ f_t + MLP([f_t, e_{t+1}]), a gated two layer MLP"""

    def __init__(self, n_embd, hidden):
        super().__init__()
        self.inp = nn.Linear(2 * n_embd, hidden, bias=False)
        self.gate = nn.Linear(2 * n_embd, hidden, bias=False)
        self.out = nn.Linear(hidden, n_embd, bias=False)
        self.norm = nn.RMSNorm(2 * n_embd)

    def forward(self, f, e):
        x = self.norm(torch.cat([f, e], -1))
        return f + self.out(F.silu(self.gate(x)) * self.inp(x))


class AttnHead(nn.Module):
    """fc([f_t, e_{t+1}]) then one decoder layer attending over the last window features of the same sequence"""

    def __init__(self, n_embd, n_head, window):
        super().__init__()
        self.fc = nn.Linear(2 * n_embd, n_embd, bias=False)
        self.n1 = nn.RMSNorm(n_embd)
        self.q = nn.Linear(n_embd, n_embd, bias=False)
        self.k = nn.Linear(n_embd, n_embd, bias=False)
        self.v = nn.Linear(n_embd, n_embd, bias=False)
        self.o = nn.Linear(n_embd, n_embd, bias=False)
        self.n2 = nn.RMSNorm(n_embd)
        self.up = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.gate = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.down = nn.Linear(4 * n_embd, n_embd, bias=False)
        self.n_head, self.window = n_head, window

    def forward(self, f, e):
        # f, e: [B, T, D], causal attention within the window
        x = self.fc(torch.cat([f, e], -1))
        B, T, D = x.shape
        h = self.n1(x)
        q = self.q(h).view(B, T, self.n_head, -1).transpose(1, 2)
        k = self.k(h).view(B, T, self.n_head, -1).transpose(1, 2)
        v = self.v(h).view(B, T, self.n_head, -1).transpose(1, 2)
        i = torch.arange(T, device=x.device)
        mask = (i[None, :] <= i[:, None]) & (i[None, :] > i[:, None] - self.window)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.o(a.transpose(1, 2).reshape(B, T, D))
        h = self.n2(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


def batches(runs, seqs, tables, device, bsz, T):
    """Yields (f_t, e_{t+1}, f_{t+1}) windows of T positions from response regions, as [B, T, D] tensors"""
    items = []
    for ri, s in seqs:
        meta, tokens, feats, index = runs[ri]
        o, n, np_ = s["offset"], s["n"], s["n_prompt"]
        # position t in [np_-1, n-2]: f_t predicts, token t+1 is known, target f_{t+1}
        lo, hi = max(np_ - 1, 0), n - 2
        for t0 in range(lo, hi, T):
            if t0 + T <= hi:
                items.append((ri, o, t0))
    rng = np.random.default_rng(0)
    rng.shuffle(items)
    for b0 in range(0, len(items), bsz):
        chunk = items[b0:b0 + bsz]
        fs, es, ys = [], [], []
        for ri, o, t0 in chunk:
            _, tokens, feats, _ = runs[ri]
            f = torch.tensor(np.asarray(feats[o + t0:o + t0 + T + 1]), device=device)
            tk = torch.tensor(tokens[o + t0 + 1:o + t0 + T + 1].astype(np.int64), device=device)
            fs.append(f[:-1]); ys.append(f[1:]); es.append(tables.embd[tk])
        yield torch.stack(fs).float(), torch.stack(es).float(), torch.stack(ys).float()


@torch.no_grad()
def acceptance(head, runs, seqs, tables, device, depth=5, topk=(1, 2, 4), seq_mode=False):
    """Chained greedy acceptance: at position t the head predicts f'_{t+1} from (f_t, e(x_{t+1})), drafts argmax, and
    continues from its own prediction while the drafts match the text; returns the mean accepted length per round and
    the per depth rank statistics for tree sizing"""
    accepted = []
    hit = {k: np.zeros(depth) for k in topk}
    tried = np.zeros(depth)
    rounds = 0
    for ri, s in seqs:
        _, tokens, feats, _ = runs[ri]
        o, n, np_ = s["offset"], s["n"], s["n_prompt"]
        f_all = torch.tensor(np.asarray(feats[o:o + n]), device=device).float()
        tk = torch.tensor(tokens[o:o + n].astype(np.int64), device=device)
        t = max(np_ - 1, 1)
        while t + depth + 1 < n:
            rounds += 1
            f = f_all[:t + 1].clone()
            ok = 0
            for d in range(depth):
                nxt_tok = tk[t + 1 + d]
                e = tables.embd[nxt_tok].float()
                if seq_mode:
                    es = tables.embd[tk[1:t + 2 + d]].float()
                    pred = head(f[None, :t + 1 + d], es[None, -f.shape[0]:] if False else torch.cat([es[None]], 0))[0, -1]
                else:
                    pred = head(f[t + d][None], e[None])[0]
                logits = tables.logits(pred[None])[0].float()
                truth = tk[t + 2 + d]
                order = torch.argsort(logits, descending=True)
                rank = (order == truth).nonzero()[0].item()
                tried[d] += 1
                for k in topk:
                    if rank < k:
                        hit[k][d] += 1
                if rank == 0:
                    ok += 1
                    if t + 1 + d < f.shape[0]:
                        pass
                    f = torch.cat([f, pred[None]], 0) if f.shape[0] <= t + 1 + d else f
                    f[t + 1 + d] = pred
                else:
                    break
            accepted.append(ok)
            t += ok + 1
    hits = {k: (hit[k] / np.maximum(tried, 1)).round(3).tolist() for k in topk}
    return float(np.mean(accepted)), hits, rounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--eval-seqs", type=int, default=60)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--heads", default="linear,mlp,attn")
    ap.add_argument("--no-kl", action="store_true", help="train on the feature regression alone, as the runtime does")
    args = ap.parse_args()
    device = torch.device(args.device)
    runs = [load_run(d) for d in args.dirs]
    meta = runs[0][0]
    tables = Tables(args.dirs[0], meta, device)
    seqs = [(ri, s) for ri, r in enumerate(runs) for s in r[3]]
    rng = np.random.default_rng(1)
    rng.shuffle(seqs)
    eval_seqs, train_seqs = seqs[: args.eval_seqs], seqs[args.eval_seqs :]
    n_train = sum(s["n"] - s["n_prompt"] for _, s in train_seqs)
    print(f"{len(train_seqs)} training sequences, {n_train} response tokens, {len(eval_seqs)} held out", flush=True)
    D = meta["n_embd"]
    results = {}
    for name in args.heads.split(","):
        t0 = time.time()
        if name == "linear":
            head = Linear(D).to(device)
            head.fit(((f.reshape(-1, D), e.reshape(-1, D), y.reshape(-1, D)) for f, e, y in batches(runs, train_seqs, tables, device, 64, 32)), device)
            fwd = lambda f, e: head(f, e)
            seq_mode = False
        else:
            head = (Mlp(D, args.hidden) if name == "mlp" else AttnHead(D, 16, args.window)).to(device)
            opt = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=0.01)
            params = sum(p.numel() for p in head.parameters())
            steps = 0
            for ep in range(args.epochs):
                for f, e, y in batches(runs, train_seqs, tables, device, 16, 64):
                    pred = head(f, e)
                    reg = F.smooth_l1_loss(pred, y)
                    logits = tables.logits(pred.reshape(-1, D)).float()
                    truth_logits = tables.logits(y.reshape(-1, D)).float()
                    ce = F.kl_div(F.log_softmax(logits, -1), F.softmax(truth_logits, -1), reduction="batchmean")
                    loss = reg if args.no_kl else reg + 0.1 * ce
                    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step()
                    steps += 1
                    if steps % 200 == 0:
                        print(f"  {name} step {steps} reg {reg.item():.4f} kl {ce.item():.4f}", flush=True)
            print(f"  {name}: {params / 1e6:.1f} M parameters, {steps} steps", flush=True)
            head.eval()
            if name == "mlp":
                fwd = lambda f, e: head(f, e)
                seq_mode = False
            else:
                fwd = head
                seq_mode = True
        mean_acc, hits, rounds = acceptance(fwd, runs, eval_seqs, tables, device, seq_mode=seq_mode)
        results[name] = (mean_acc, hits)
        print(f"{name}: {mean_acc:.3f} tokens accepted per round over {rounds} rounds, top k hit rate by depth {hits}, {time.time() - t0:.0f} s", flush=True)
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()

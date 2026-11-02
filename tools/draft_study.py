"""Trains the fused draft block of crates/kf-backend-cpu/src/draft.rs on drafter-data runs and measures tokens accepted per round"""
import argparse
import json
import math
import os
import queue
import threading
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TREE_ROWS = 64
DRAFT_VOCAB = 32768
TREE_FANOUT = (4, 3, 2, 2)
CHAIN_FANOUT = (1, 1, 1, 1)
HIT_DEPTHS = 5
KEEPS = (8, 16, 32, 64)
BASE_CHUNK = 64
VARIANTS = {
    "fin": dict(use_mid=False, steps=3, window=64, chunk=64),
    "k1": dict(use_mid=True, steps=1, window=64, chunk=64),
    "full": dict(use_mid=True, steps=3, window=64, chunk=64),
    "w256": dict(use_mid=True, steps=3, window=256, chunk=256),
    "t256": dict(use_mid=True, steps=3, window=64, chunk=256),
}
PARAM_ORDER = ("fuse", "fc_g", "fc_e", "wq", "wk", "wv", "wo", "n1", "wg", "wu", "wd", "n2")


class Run:
    # Opens one drafter-data directory through memory maps and checks every file covers the same rows
    def __init__(self, path):
        self.path = path
        self.meta = json.load(open(os.path.join(path, "meta.json")))
        d, nc = self.meta["n_embd"], self.meta["cands"]
        self.tokens = np.fromfile(os.path.join(path, "tokens.bin"), dtype=np.uint32)
        n = len(self.tokens)
        self.fin = np.memmap(os.path.join(path, "features.bin"), dtype=np.float16, mode="r").reshape(-1, d)
        self.mid = np.memmap(os.path.join(path, "features_mid.bin"), dtype=np.float16, mode="r").reshape(-1, d)
        cand = np.dtype([("ids", "<u4", (nc,)), ("logits", "<f4", (nc,)), ("lse", "<f4")])
        self.cands = np.memmap(os.path.join(path, "cands.bin"), dtype=cand, mode="r")
        self.index = [json.loads(l) for l in open(os.path.join(path, "index.jsonl")) if l.strip()]
        if not (self.fin.shape[0] == n and self.mid.shape[0] == n and self.cands.shape[0] == n):
            raise SystemExit(f"{path} holds {n} tokens, {self.fin.shape[0]} final rows, {self.mid.shape[0]} mid rows, {self.cands.shape[0]} candidate rows")
        if any(s["offset"] + s["n"] > n for s in self.index):
            raise SystemExit(f"{path} indexes rows past its {n} tokens")


class Tables:
    # Loads the embedding table, the head rows of the draft vocabulary, and the final norm onto the device
    def __init__(self, run, vocab, dev):
        m = run.meta
        d, nv = m["n_embd"], m["n_vocab"]
        embd = np.fromfile(os.path.join(run.path, "embd.bin"), dtype=np.float16).reshape(nv, d)
        head = embd if m["head_tied"] else np.fromfile(os.path.join(run.path, "head.bin"), dtype=np.float16).reshape(nv, d)
        self.embd = torch.from_numpy(embd).to(dev)
        self.ids = torch.from_numpy(vocab.astype(np.int64)).to(dev)
        self.head = torch.from_numpy(head[vocab].astype(np.float32)).to(dev)
        self.head_low = self.head.to(torch.bfloat16)
        self.norm = torch.from_numpy(np.fromfile(os.path.join(run.path, "norm0.bin"), dtype=np.float32)).to(dev)
        self.eps = m["rms_eps"]
        self.col = torch.full((nv,), -1, dtype=torch.int64, device=dev)
        self.col[self.ids] = torch.arange(len(vocab), device=dev)

    # Embedding rows of token ids as float32
    def embed(self, tokens):
        return self.embd[tokens].float()

    # Logits over the draft vocabulary of output rows through the frozen final norm and head rows
    def logits(self, out, low=False):
        n = rms(out, self.norm, self.eps)
        return n.to(torch.bfloat16) @ self.head_low.t() if low else n @ self.head.t()

    # Column of each token in the draft vocabulary or minus one when it is outside
    def column(self, tokens):
        return torch.where(tokens >= 0, self.col[tokens.clamp_min(0)], -1)


# Rows scaled to unit root mean square in float32 then by the weight
def rms(x, w, eps):
    xf = x.float()
    return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w


class Block(nn.Module):
    # Builds the fusion as identity on the final half, uniform matrices scaled by fan in, and unit norms
    def __init__(self, d, heads, ff, eps, use_mid, seed):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)

        def mat(i, o):
            return nn.Parameter((torch.rand(i, o, generator=gen) * 2 - 1) / math.sqrt(i))

        self.d, self.heads, self.eps = d, heads, eps
        self.fuse_mid = nn.Parameter(torch.zeros(d, d)) if use_mid else None
        self.fuse_fin = nn.Parameter(torch.eye(d))
        self.fc_g, self.fc_e = mat(d, d), mat(d, d)
        self.wq, self.wk, self.wv, self.wo = mat(d, d), mat(d, d), mat(d, d), mat(d, d)
        self.n1 = nn.Parameter(torch.ones(d))
        self.wg, self.wu, self.wd = mat(d, ff), mat(d, ff), mat(ff, d)
        self.n2 = nn.Parameter(torch.ones(d))

    # Feature rows g from mid and final rows, the mid half absent when the block reads the final row alone
    def fuse(self, mid, fin):
        g = (fin @ self.fuse_fin).float()
        return g + (mid @ self.fuse_mid).float() if self.fuse_mid is not None else g

    # Residual x0 with queries, keys, and values of rows from g and the next token embeddings
    def layer_in(self, g, e):
        x0 = (g @ self.fc_g).float() + (e @ self.fc_e).float()
        h1 = rms(x0, self.n1, self.eps)
        return x0, h1 @ self.wq, h1 @ self.wk, h1 @ self.wv

    # Output rows from the residual x0 and the attention output a
    def layer_out(self, x0, a):
        x1 = x0 + (a @ self.wo).float()
        h2 = rms(x1, self.n2, self.eps)
        return x1 + ((F.silu(h2 @ self.wg) * (h2 @ self.wu)) @ self.wd).float()

    # Parameters in the order and layout of DraftConfig param_shapes, the fusion as mid rows over final rows
    def export(self):
        fuse_mid = self.fuse_mid if self.fuse_mid is not None else torch.zeros_like(self.fuse_fin)
        p = dict(fuse=torch.cat([fuse_mid, self.fuse_fin], 0), fc_g=self.fc_g, fc_e=self.fc_e, wq=self.wq, wk=self.wk, wv=self.wv, wo=self.wo, n1=self.n1[None], wg=self.wg, wu=self.wu, wd=self.wd, n2=self.n2[None])
        return {k: p[k].detach().cpu().float() for k in PARAM_ORDER}


# Heads split out of rows as [batch, heads, rows, head width]
def split_heads(x, h):
    return x.view(*x.shape[:-1], h, -1).transpose(-3, -2)


# Visibility of key j to query i within one sequence, causal and within the window
def window_mask(n, w, dev):
    i = torch.arange(n, device=dev)
    return (i[None, :] <= i[:, None]) & (i[:, None] - i[None, :] < w)


# Masks of each chained step over stacked keys, step zero keys by window and later step keys by row
def step_masks(t, steps, w, dev):
    eye = torch.eye(t, dtype=torch.bool, device=dev)
    return [torch.cat([window_mask(t, w, dev)] + [eye] * k, 1) for k in range(steps)]


# Chunk starts of response rows for one epoch, each sequence tiled from a random offset, in shuffled order
def epoch_chunks(runs, seqs, t, steps, rng):
    items = []
    for ri, s in seqs:
        lo, hi = max(s["n_prompt"] - 1, 0), s["n"] - 2 - steps
        span = hi - lo + 1
        if span < t:
            continue
        off = int(rng.integers(0, span % t + 1)) if rng is not None else 0
        items.extend((ri, s["offset"] + c) for c in range(lo + off, hi - t + 2, t))
    if rng is not None:
        rng.shuffle(items)
    return items


# Features, next tokens, and candidate rows of one chunk starting at global row s
def read_chunk(run, s, t, steps):
    c = run.cands[s + 1:s + t + steps]
    return np.array(run.mid[s:s + t]), np.array(run.fin[s:s + t]), run.tokens[s + 1:s + 1 + t + steps].astype(np.int64), c["ids"].astype(np.int64), np.array(c["logits"]), np.array(c["lse"])


# Chunks read on a background thread in the given order
def chunk_stream(runs, items, t, steps):
    q = queue.Queue(maxsize=64)

    def work():
        for ri, s in items:
            q.put(read_chunk(runs[ri], s, t, steps))
        q.put(None)

    threading.Thread(target=work, daemon=True).start()
    while (x := q.get()) is not None:
        yield x


# Chunks stacked on the device with soft targets restricted to the draft vocabulary and renormalized
def to_device(batch, tables, dev):
    mid, fin, toks, ids, logits, lse = (torch.from_numpy(np.stack(x)).to(dev) for x in zip(*batch))
    cols = tables.col[ids]
    p = torch.exp(logits - lse[..., None]) * (cols >= 0)
    mass = p.sum(-1)
    return mid.float(), fin.float(), toks, cols.clamp_min(0), p / mass.clamp_min(1e-30)[..., None], mass > 0


# Summed soft cross entropy over chained steps, rows counted, and the step zero top four hit rate
def chained_loss(block, tables, b, masks):
    mid, fin, toks, cols, probs, has = b
    nb, t, d = fin.shape
    g = block.fuse(mid, fin)
    ks, vs = [], []
    total, counted, hit = 0.0, 0, 0
    for k in range(len(masks)):
        x0, q, kk, vv = block.layer_in(g, tables.embed(toks[:, k:k + t]))
        ks.append(split_heads(kk, block.heads))
        vs.append(split_heads(vv, block.heads))
        a = F.scaled_dot_product_attention(split_heads(q, block.heads), torch.cat(ks, 2), torch.cat(vs, 2), attn_mask=masks[k])
        out = block.layer_out(x0, a.transpose(1, 2).reshape(nb, t, d))
        z = tables.logits(out, low=True).float()
        row = torch.logsumexp(z, -1) - (probs[:, k:k + t] * z.gather(-1, cols[:, k:k + t])).sum(-1)
        m = has[:, k:k + t]
        total = total + row[m].sum()
        counted += int(m.sum())
        if k == 0:
            nxt = tables.column(toks[:, 1:t + 1])
            hit = ((z.topk(TREE_FANOUT[0], -1).indices == nxt[..., None]).any(-1) & (nxt >= 0)).float().mean().item()
        g = out
    if counted == 0:
        raise RuntimeError("no row of the batch has a target in the draft vocabulary")
    return total, counted, hit


# Mean chained loss and step zero hit without updating the block, halving the batch on device memory errors
@torch.no_grad()
def heldout_loss(block, tables, runs, items, t, steps, masks, nb, dev):
    total, counted, hits, n = 0.0, 0, 0.0, 0
    stream = chunk_stream(runs, items, t, steps)
    pending = []
    while True:
        while len(pending) < nb and (c := next(stream, None)) is not None:
            pending.append(c)
        if not pending:
            break
        oom = False
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tl, cn, h = chained_loss(block, tables, to_device(pending[:nb], tables, dev), masks)
        except torch.cuda.OutOfMemoryError:
            oom = True
        if oom:
            torch.cuda.empty_cache()
            if nb == 1:
                raise RuntimeError("out of device memory at a held out batch of one chunk")
            nb //= 2
            print(f"  out of memory, held out batch halved to {nb} chunks", flush=True)
            continue
        k = min(nb, len(pending))
        total, counted, hits, n = total + float(tl), counted + cn, hits + h * k, n + k
        pending = pending[nb:]
    return total / counted, hits / n


# Trains the block for the epochs and returns the loss summary, halving the batch on device memory errors
def train(block, tables, runs, train_seqs, eval_seqs, v, args, dev):
    t, steps, w = v["chunk"], v["steps"], v["window"]
    nb = max(1, args.batch * BASE_CHUNK // t)
    opt = torch.optim.AdamW(block.parameters(), lr=3e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    masks = step_masks(t, steps, w, dev)
    rng = np.random.default_rng(args.seed)
    held_items = epoch_chunks(runs, eval_seqs, t, steps, None)
    losses, hits, epochs = [], [], []
    start = time.time()
    for ep in range(args.epochs):
        items = epoch_chunks(runs, train_seqs, t, steps, rng)
        stream = chunk_stream(runs, items, t, steps)
        pending, first = [], len(losses)
        while True:
            while len(pending) < nb and (c := next(stream, None)) is not None:
                pending.append(c)
            if not pending:
                break
            oom = False
            try:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    total, counted, hit = chained_loss(block, tables, to_device(pending[:nb], tables, dev), masks)
                loss = total / counted
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(block.parameters(), 1.0)
                opt.step()
                losses.append(loss.item())
                hits.append(hit)
            except torch.cuda.OutOfMemoryError:
                oom = True
            if oom:
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                if nb == 1:
                    raise RuntimeError("out of device memory at a batch of one chunk")
                nb //= 2
                print(f"  out of memory, batch halved to {nb} chunks", flush=True)
                continue
            pending = pending[nb:]
            if len(losses) % 200 == 0:
                print(f"  epoch {ep + 1} step {len(losses)} loss {np.mean(losses[-200:]):.4f} hit {np.mean(hits[-200:]):.3f} {time.time() - start:.0f} s", flush=True)
        hl, hh = heldout_loss(block, tables, runs, held_items, t, steps, masks, nb, dev)
        ep_losses = losses[first:]
        epochs.append(dict(epoch=ep + 1, steps=len(ep_losses), chunks=len(items), mean_loss=float(np.mean(ep_losses)), last200_loss=float(np.mean(ep_losses[-200:])), last200_hit=float(np.mean(hits[first:][-200:])), heldout_loss=hl, heldout_hit=hh))
        print(f"  epoch {ep + 1} done, {len(ep_losses)} steps, mean loss {epochs[-1]['mean_loss']:.4f}, last 200 {epochs[-1]['last200_loss']:.4f}, held out loss {hl:.4f} hit {hh:.3f}, {time.time() - start:.0f} s", flush=True)
    return dict(first_loss=losses[0], first50_loss=float(np.mean(losses[:50])), first50_hit=float(np.mean(hits[:50])), epochs=epochs, final_batch_chunks=nb, seconds=time.time() - start)


class Context:
    # Teacher forced step zero keys, values, and outputs of every held out row with the root positions of every round
    def __init__(self, block, tables, runs, seqs, w, dev):
        ks, vs, outs, roots, bases, truth, spans = [], [], [], [], [], [], []
        base = 0
        for ri, s in seqs:
            run, o, n, npr = runs[ri], s["offset"], s["n"], s["n_prompt"]
            mid = torch.from_numpy(np.array(run.mid[o:o + n - 1])).to(dev).float()
            fin = torch.from_numpy(np.array(run.fin[o:o + n - 1])).to(dev).float()
            tk = run.tokens[o:o + n].astype(np.int64)
            x0, q, k, v = block.layer_in(block.fuse(mid, fin), tables.embed(torch.from_numpy(tk[1:]).to(dev)))
            a = F.scaled_dot_product_attention(split_heads(q, block.heads)[None], split_heads(k, block.heads)[None], split_heads(v, block.heads)[None], attn_mask=window_mask(n - 1, w, dev))
            outs.append(block.layer_out(x0, a[0].transpose(0, 1).reshape(n - 1, -1)))
            ks.append(k)
            vs.append(v)
            lo, count = max(npr - 1, 0), n - 5 - max(npr - 1, 0)
            spans.append((len(roots), max(count, 0)))
            for tt in range(lo, lo + max(count, 0)):
                roots.append(base + tt)
                bases.append(base)
                truth.append([tk[tt + 1 + j] if tt + 1 + j < n else -1 for j in range(1, HIT_DEPTHS + 1)])
            base += n - 1
        self.k, self.v, self.out = torch.cat(ks), torch.cat(vs), torch.cat(outs)
        self.roots = torch.tensor(roots, device=dev)
        self.bases = torch.tensor(bases, device=dev)
        self.truth = torch.tensor(truth, device=dev)
        self.spans, self.w, self.dev = spans, w, dev


# Outputs, keys, and values of chained rows at depth k over the true window, their ancestors, and themselves
def chained_rows(block, tables, ctx, r, b, k, g, token, anc, chain):
    p, rows, d = g.shape
    h = block.heads
    hd = d // h
    x0, q, kk, vv = block.layer_in(g, tables.embed(token))
    span = ctx.w - k
    pos = r[:, None] + (k + 1 - ctx.w) + torch.arange(span, device=ctx.dev)
    seen = pos >= b[:, None]
    kt = ctx.k[pos.clamp_min(0)].view(p, span, h, hd)
    vt = ctx.v[pos.clamp_min(0)].view(p, span, h, hd)
    kc = torch.stack([chain[m][0].gather(1, anc[..., m, None].expand(-1, -1, d)) for m in range(k - 1)] + [kk], 2).view(p, rows, k, h, hd)
    vc = torch.stack([chain[m][1].gather(1, anc[..., m, None].expand(-1, -1, d)) for m in range(k - 1)] + [vv], 2).view(p, rows, k, h, hd)
    qh = q.view(p, rows, h, hd) / math.sqrt(hd)
    st = torch.einsum("prhd,plhd->prhl", qh, kt).masked_fill(~seen[:, None, None, :], float("-inf"))
    sc = torch.einsum("prhd,prchd->prhc", qh, kc)
    att = torch.softmax(torch.cat([st, sc], -1), -1)
    a = torch.einsum("prhl,plhd->prhd", att[..., :span], vt) + torch.einsum("prhc,prchd->prhd", att[..., span:], vc)
    return block.layer_out(x0, a.reshape(p, rows, d)), kk, vv


# Accepted length and tree rows at each root, grown, kept, and capped as CpuDraft forest grows one root
@torch.no_grad()
def grow(block, tables, ctx, idx, fanout, keep, cap):
    r, b, tr = ctx.roots[idx], ctx.bases[idx], ctx.truth[idx]
    p, dev = len(idx), ctx.dev
    out = ctx.out[r][:, None]
    weight = torch.ones(p, 1, device=dev)
    valid = torch.ones(p, 1, dtype=torch.bool, device=dev)
    intree, matched = valid.clone(), valid.clone()
    size = torch.ones(p, dtype=torch.int64, device=dev)
    acc = torch.zeros(p, dtype=torch.int64, device=dev)
    anc = torch.zeros(p, 1, 0, dtype=torch.int64, device=dev)
    chain = []
    for k in range(1, len(fanout) + 1):
        rows, f = out.shape[1], fanout[k - 1]
        z = tables.logits(out.reshape(p * rows, -1)).view(p, rows, -1)
        top, cols = z.topk(f, -1)
        q = torch.exp(top - torch.logsumexp(z, -1, keepdim=True))
        del z
        cw = torch.where((valid & (weight > 0))[..., None], weight[..., None] * q, float("-inf")).view(p, rows * f)
        order = torch.sort(cw, dim=-1, descending=True, stable=True).indices[:, :min(keep, rows * f)]
        weight = cw.gather(1, order)
        parent = order // f
        token = tables.ids[cols.view(p, rows * f).gather(1, order)]
        valid = weight > float("-inf")
        elig = valid & intree.gather(1, parent)
        intree = elig & (size[:, None] + elig.long().cumsum(1) <= cap)
        size = size + intree.sum(1)
        matched = intree & matched.gather(1, parent) & (token == tr[:, k - 1, None])
        acc += matched.any(1)
        if k == len(fanout):
            break
        g = out.gather(1, parent[..., None].expand(-1, -1, out.shape[-1]))
        anc = anc.gather(1, parent[..., None].expand(-1, -1, anc.shape[-1]))
        if k >= 2:
            anc = torch.cat([anc, parent[..., None]], -1)
        out, kk, vv = chained_rows(block, tables, ctx, r, b, k, g, token, anc, chain)
        chain.append((kk, vv))
    return acc, size


# Top one and top four hits at each chain depth with the true tokens fed along the chain
@torch.no_grad()
def chain_hits(block, tables, ctx, idx):
    r, b, tr = ctx.roots[idx], ctx.bases[idx], ctx.truth[idx]
    p = len(idx)
    out = ctx.out[r][:, None]
    anc = torch.zeros(p, 1, 0, dtype=torch.int64, device=ctx.dev)
    chain, h1, h4 = [], [], []
    for j in range(HIT_DEPTHS):
        tc = tables.column(tr[:, j])
        hit = (tables.logits(out[:, 0]).topk(4, -1).indices == tc[:, None]) & (tc[:, None] >= 0)
        h1.append(hit[:, 0])
        h4.append(hit.any(1))
        if j == HIT_DEPTHS - 1:
            break
        k = j + 1
        if k >= 2:
            anc = torch.cat([anc, torch.zeros(p, 1, 1, dtype=torch.int64, device=ctx.dev)], -1)
        out, kk, vv = chained_rows(block, tables, ctx, r, b, k, out, tr[:, k - 1, None], anc, chain)
        chain.append((kk, vv))
    return torch.stack(h1, 1), torch.stack(h4, 1)


# Results of fn over chunks of root indices, halving the chunk on device memory errors
def chunked(fn, idx, pc):
    outs, i = [], 0
    while i < len(idx):
        oom = False
        try:
            outs.append(fn(idx[i:i + pc]))
            i += pc
        except torch.cuda.OutOfMemoryError:
            oom = True
        if oom:
            torch.cuda.empty_cache()
            if pc == 1:
                raise RuntimeError("out of device memory at an evaluation chunk of one root")
            pc //= 2
            print(f"  out of memory, evaluation chunk halved to {pc} roots", flush=True)
    return [torch.cat(x) for x in zip(*outs)], pc


# Roots each sequence visits when every round starts after the last accepted token plus the sampled one
def visited(acc, spans):
    out = []
    for lo, count in spans:
        j = 0
        while j < count:
            out.append(lo + j)
            j += int(acc[lo + j]) + 1
    return np.array(out, dtype=np.int64)


# Tokens accepted per round for the chain and each keep, chain hit rates by depth, and timing
@torch.no_grad()
def evaluate(block, tables, runs, seqs, w, pc, dev):
    block.eval()
    t0 = time.time()
    ctx = Context(block, tables, runs, seqs, w, dev)
    torch.cuda.synchronize()
    res = dict(roots=len(ctx.roots), teacher_seconds=time.time() - t0)
    all_idx = torch.arange(len(ctx.roots), device=dev)
    configs = [("chain", CHAIN_FANOUT, 1, TREE_ROWS)] + [(f"keep{k}", TREE_FANOUT, k, TREE_ROWS) for k in KEEPS] + [("keep64_nocap", TREE_FANOUT, 64, 1 << 30)]
    for name, fanout, keep, cap in configs:
        torch.cuda.synchronize()
        t1 = time.time()
        (acc, size), pc = chunked(lambda i: grow(block, tables, ctx, i, fanout, keep, cap), all_idx, pc)
        torch.cuda.synchronize()
        sec = time.time() - t1
        acc, size = acc.cpu().numpy(), size.cpu().numpy()
        vis = visited(acc, ctx.spans)
        res[name] = dict(accepted=float(acc[vis].mean()), rounds=len(vis), tree_rows=float(size[vis].mean()), accepted_every_root=float(acc.mean()), batched_ms_per_tree=1000 * sec / len(acc))
        print(f"  {name}: {res[name]['accepted']:.3f} accepted per round over {len(vis)} rounds, {res[name]['tree_rows']:.1f} tree rows, {res[name]['batched_ms_per_tree']:.3f} ms per tree batched", flush=True)
    single = all_idx[:100]
    torch.cuda.synchronize()
    t1 = time.time()
    for i in range(len(single)):
        grow(block, tables, ctx, single[i:i + 1], TREE_FANOUT, 32, TREE_ROWS)
    torch.cuda.synchronize()
    res["single_tree_ms_keep32"] = 1000 * (time.time() - t1) / len(single)
    full = all_idx[ctx.truth[:, HIT_DEPTHS - 1] >= 0]
    (h1, h4), _ = chunked(lambda i: chain_hits(block, tables, ctx, i), full, pc)
    res["hit_positions"] = len(full)
    res["top1_by_depth"] = h1.float().mean(0).tolist()
    res["top4_by_depth"] = h4.float().mean(0).tolist()
    res["vocab_coverage_depth1"] = float((tables.column(ctx.truth[:, 0]) >= 0).float().mean())
    print(f"  chain hits top1 {np.round(res['top1_by_depth'], 3).tolist()} top4 {np.round(res['top4_by_depth'], 3).tolist()}, single tree {res['single_tree_ms_keep32']:.2f} ms", flush=True)
    return res


# Plain text table of accepted per round, rounds, chain hit rates, tree rows, and timing for every variant
def table(results):
    names = ["chain"] + [f"keep{k}" for k in KEEPS] + ["keep64_nocap"]
    head = f"{'variant':<8} " + " ".join(f"{n:>13}" for n in names)
    lines = ["accepted tokens per round under greedy acceptance on held out response rows, rounds walked in brackets", head]
    for v, r in results.items():
        e = r["eval"]
        lines.append(f"{v:<8} " + " ".join(f"{e[n]['accepted']:>6.3f} ({e[n]['rounds']:>5})" for n in names))
    lines += ["", "mean tree rows per round including the root", head]
    for v, r in results.items():
        e = r["eval"]
        lines.append(f"{v:<8} " + " ".join(f"{e[n]['tree_rows']:>13.1f}" for n in names))
    lines += ["", "chain hit rate at depths 1 to 5 with the true tokens fed along the chain"]
    for v, r in results.items():
        e = r["eval"]
        lines.append(f"{v:<8} top1 " + " ".join(f"{x:.3f}" for x in e["top1_by_depth"]) + "   top4 " + " ".join(f"{x:.3f}" for x in e["top4_by_depth"]))
    lines += ["", "wall clock of this implementation, keep 32"]
    for v, r in results.items():
        e = r["eval"]
        lines.append(f"{v:<8} batched {e['keep32']['batched_ms_per_tree']:.3f} ms per tree   one tree at a time {e['single_tree_ms_keep32']:.2f} ms per tree")
    return "\n".join(lines)


# Trains and evaluates every requested variant on the pooled runs and writes the results as they finish
def main():
    ap = argparse.ArgumentParser(description="Trains the fused draft block on drafter-data runs and measures tokens accepted per round")
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--variants", default="fin,k1,full,w256")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8, help="chunks of 64 rows per step, longer chunks taking proportionally fewer")
    ap.add_argument("--eval-seqs", type=int, default=60)
    ap.add_argument("--train-seqs", type=int, default=0, help="training sequences used, every one when zero")
    ap.add_argument("--eval-chunk", type=int, default=128, help="root positions grown together")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mem-gib", type=float, default=8.5, help="device memory the allocator may hold")
    ap.add_argument("--out", required=True, help="results json path")
    ap.add_argument("--save", help="path the full variant's weights are written to")
    args = ap.parse_args()
    variants = args.variants.split(",")
    if any(v not in VARIANTS for v in variants):
        raise SystemExit(f"variants are {', '.join(VARIANTS)}")
    if args.save and "full" not in variants:
        raise SystemExit("--save writes the full variant, which is not among the variants")
    dev = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(min(1.0, args.mem_gib * 2**30 / torch.cuda.get_device_properties(dev).total_memory))
    runs = [Run(d) for d in args.dirs]
    meta = runs[0].meta
    for r in runs[1:]:
        if any(r.meta[k] != meta[k] for k in ("n_embd", "n_vocab", "tap", "rms_eps", "head_tied", "cands", "name")):
            raise SystemExit(f"{r.path} was written for a different model than {runs[0].path}")
    seqs = [(ri, s) for ri, r in enumerate(runs) for s in r.index]
    np.random.default_rng(1).shuffle(seqs)
    eval_seqs, train_seqs = seqs[:args.eval_seqs], seqs[args.eval_seqs:]
    if args.train_seqs:
        train_seqs = train_seqs[:args.train_seqs]
    counts = np.zeros(meta["n_vocab"], dtype=np.int64)
    for ri, s in train_seqs:
        counts += np.bincount(runs[ri].tokens[s["offset"]:s["offset"] + s["n"]], minlength=meta["n_vocab"])
    vocab = np.sort(np.argsort(-counts, kind="stable")[:DRAFT_VOCAB])
    tables = Tables(runs[0], vocab, dev)
    d = meta["n_embd"]
    heads = d // 128 if d % 128 == 0 else 16
    data = dict(dirs=args.dirs, tap=meta["tap"], train_seqs=len(train_seqs), eval_seqs=len(eval_seqs), train_response_rows=int(sum(s["n"] - s["n_prompt"] for _, s in train_seqs)), eval_response_rows=int(sum(s["n"] - s["n_prompt"] for _, s in eval_seqs)), draft_vocab=len(vocab), vocab_train_token_share=float(counts[vocab].sum() / counts.sum()))
    print(json.dumps(data), flush=True)
    results = {}
    out = dict(data=data, results=results)
    for name in variants:
        v = VARIANTS[name]
        torch.cuda.empty_cache()
        block = Block(d, heads, 2 * d, 1e-5, v["use_mid"], args.seed).to(dev)
        nb = max(1, args.batch * BASE_CHUNK // v["chunk"])
        cfg = dict(v, d=d, heads=heads, ff=2 * d, eps=1e-5, batch_chunks=nb, rows_per_step=nb * v["chunk"], epochs=args.epochs, lr=3e-4, weight_decay=0.01, betas=[0.9, 0.999], adam_eps=1e-8, grad_clip=1.0, precision="bf16 autocast for matmuls and attention, float32 master weights, residual stream, norms, and loss", eval_precision="float32", eval_depth=len(TREE_FANOUT), tree_fanout=list(TREE_FANOUT), tree_rows=TREE_ROWS, params=sum(p.numel() for p in block.parameters()))
        print(f"{name}: {json.dumps(cfg)}", flush=True)
        tr = train(block, tables, runs, train_seqs, eval_seqs, v, args, dev)
        if name == "full" and args.save:
            torch.save(dict(config=cfg, vocab=torch.from_numpy(vocab.astype(np.int64)), params=block.export()), args.save)
        torch.cuda.empty_cache()
        ev = evaluate(block, tables, runs, eval_seqs, v["window"], args.eval_chunk, dev)
        results[name] = dict(config=cfg, train=tr, eval=ev)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
        del block
    print(table(results), flush=True)


if __name__ == "__main__":
    main()

//! The draft head on the host, the reference the device head is checked against, see kf_core::HeadConfig
use crate::train;
use anyhow::{Result, ensure};
use kf_core::{HeadConfig, TrainStep};
use std::collections::HashMap;

/// Deterministic uniform noise in a range from a seed
pub fn init_values(n: usize, scale: f32, seed: &mut u64) -> Vec<f32> {
    (0..n)
        .map(|_| {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 7;
            *seed ^= *seed << 17;
            ((*seed >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
        })
        .collect()
}

/// Initial parameters of a head, products scaled by their fan in and norms at one
pub fn init_params(cfg: &HeadConfig) -> Vec<Vec<f32>> {
    let mut seed = cfg.seed.max(1);
    cfg.param_shapes()
        .iter()
        .map(|(name, rows, cols)| if name.starts_with('n') { vec![1.0; rows * cols] } else { init_values(rows * cols, 1.0 / (*rows as f32).sqrt(), &mut seed) })
        .collect()
}

/// A head with its parameters, gradients, and optimizer moments on the host
pub struct CpuHead {
    pub cfg: HeadConfig,
    pub params: Vec<Vec<f32>>,
    pub grads: Vec<Vec<f32>>,
    m: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    pub step: i32,
}

/// Every intermediate of one forward pass the backward pass reads
pub struct Trace {
    pub n: usize,
    pub x0: Vec<f32>,
    pub h1: Vec<f32>,
    pub inv1: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub a: Vec<f32>,
    pub probs: Vec<f32>,
    pub x1: Vec<f32>,
    pub h2: Vec<f32>,
    pub inv2: Vec<f32>,
    pub g: Vec<f32>,
    pub u: Vec<f32>,
    pub mm: Vec<f32>,
    pub pred: Vec<f32>,
}

const FC_F: usize = 0;
const FC_E: usize = 1;
const WQ: usize = 2;
const WK: usize = 3;
const WV: usize = 4;
const WO: usize = 5;
const N1: usize = 6;
const WG: usize = 7;
const WU: usize = 8;
const WD: usize = 9;
const N2: usize = 10;

fn mm(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize, ta: bool, tb: bool, beta: f32) {
    let (lda, ldb) = (if ta { m } else { k }, if tb { k } else { n });
    train::gemm(a, b, c, m, n, k, lda, ldb, n, ta, tb, 1.0, beta);
}

impl CpuHead {
    pub fn new(cfg: HeadConfig) -> Result<Self> {
        Self::with_params(cfg, init_params(&cfg))
    }

    /// A head over given parameters in the configuration's order
    pub fn with_params(cfg: HeadConfig, params: Vec<Vec<f32>>) -> Result<Self> {
        ensure!(cfg.d.is_multiple_of(cfg.heads) && cfg.heads > 0, "a head width of {} over {} heads", cfg.d, cfg.heads);
        let shapes = cfg.param_shapes();
        ensure!(params.len() == shapes.len() && params.iter().zip(&shapes).all(|(p, (_, r, c))| p.len() == r * c), "parameters of the wrong shapes for the head");
        let zeros: Vec<Vec<f32>> = params.iter().map(|p| vec![0.0; p.len()]).collect();
        Ok(Self { cfg, grads: zeros.clone(), m: zeros.clone(), v: zeros, params, step: 0 })
    }

    /// Predicts the next rows of b chunks of t positions from rows f and the next tokens' embeddings e, both [b t, d]
    pub fn forward(&self, f: &[f32], e: &[f32], b: usize, t: usize) -> Trace {
        let (d, ff, h) = (self.cfg.d, self.cfg.ff, self.cfg.heads);
        let n = b * t;
        let p = &self.params;
        let mut x0 = vec![0.0; n * d];
        mm(f, &p[FC_F], &mut x0, n, d, d, false, false, 0.0);
        mm(e, &p[FC_E], &mut x0, n, d, d, false, false, 1.0);
        let mut h1 = vec![0.0; n * d];
        let inv1 = train::rms_fwd(&x0, &p[N1], &mut h1, d, self.cfg.eps);
        let (mut q, mut k, mut v) = (vec![0.0; n * d], vec![0.0; n * d], vec![0.0; n * d]);
        mm(&h1, &p[WQ], &mut q, n, d, d, false, false, 0.0);
        mm(&h1, &p[WK], &mut k, n, d, d, false, false, 0.0);
        mm(&h1, &p[WV], &mut v, n, d, d, false, false, 0.0);
        let mut a = vec![0.0; n * d];
        let probs = train::attn_fwd(&q, &k, &v, &mut a, b, t, h, d / h, self.cfg.window, 1.0 / (self.cfg.head_dim() as f32).sqrt());
        let mut x1 = x0.clone();
        mm(&a, &p[WO], &mut x1, n, d, d, false, false, 1.0);
        let mut h2 = vec![0.0; n * d];
        let inv2 = train::rms_fwd(&x1, &p[N2], &mut h2, d, self.cfg.eps);
        let (mut g, mut u) = (vec![0.0; n * ff], vec![0.0; n * ff]);
        mm(&h2, &p[WG], &mut g, n, ff, d, false, false, 0.0);
        mm(&h2, &p[WU], &mut u, n, ff, d, false, false, 0.0);
        let mut mmv = vec![0.0; n * ff];
        train::silu_gate_fwd(&g, &u, &mut mmv);
        let mut pred = x1.clone();
        mm(&mmv, &p[WD], &mut pred, n, d, ff, false, false, 1.0);
        Trace { n, x0, h1, inv1, q, k, v, a, probs, x1, h2, inv2, g, u, mm: mmv, pred }
    }

    /// The mean smooth L1 loss against the target rows, the gradients of every parameter added into grads
    pub fn backward(&mut self, f: &[f32], e: &[f32], target: &[f32], tr: &Trace, b: usize, t: usize) -> f32 {
        let (d, ff, h) = (self.cfg.d, self.cfg.ff, self.cfg.heads);
        let n = tr.n;
        let mut dx2 = vec![0.0; n * d];
        let loss = train::smooth_l1(&tr.pred, target, &mut dx2, 1.0);
        let p = &self.params;
        let gr = &mut self.grads;
        mm(&tr.mm, &dx2, &mut gr[WD], ff, d, n, true, false, 1.0);
        let mut dm = vec![0.0; n * ff];
        mm(&dx2, &p[WD], &mut dm, n, ff, d, false, true, 0.0);
        let (mut dg, mut du) = (vec![0.0; n * ff], vec![0.0; n * ff]);
        train::silu_gate_bwd(&tr.g, &tr.u, &dm, &mut dg, &mut du);
        mm(&tr.h2, &dg, &mut gr[WG], d, ff, n, true, false, 1.0);
        mm(&tr.h2, &du, &mut gr[WU], d, ff, n, true, false, 1.0);
        let mut dh2 = vec![0.0; n * d];
        mm(&dg, &p[WG], &mut dh2, n, d, ff, false, true, 0.0);
        mm(&du, &p[WU], &mut dh2, n, d, ff, false, true, 1.0);
        let mut dx1 = dx2.clone();
        let mut dx1n = vec![0.0; n * d];
        train::rms_bwd(&tr.x1, &p[N2], &tr.inv2, &dh2, &mut dx1n, &mut gr[N2], d);
        dx1.iter_mut().zip(&dx1n).for_each(|(a, b)| *a += b);
        mm(&tr.a, &dx1, &mut gr[WO], d, d, n, true, false, 1.0);
        let mut da = vec![0.0; n * d];
        mm(&dx1, &p[WO], &mut da, n, d, d, false, true, 0.0);
        let (mut dq, mut dk, mut dv) = (vec![0.0; n * d], vec![0.0; n * d], vec![0.0; n * d]);
        train::attn_bwd(&tr.q, &tr.k, &tr.v, &tr.probs, &da, &mut dq, &mut dk, &mut dv, b, t, h, d / h, 1.0 / (self.cfg.head_dim() as f32).sqrt());
        mm(&tr.h1, &dq, &mut gr[WQ], d, d, n, true, false, 1.0);
        mm(&tr.h1, &dk, &mut gr[WK], d, d, n, true, false, 1.0);
        mm(&tr.h1, &dv, &mut gr[WV], d, d, n, true, false, 1.0);
        let mut dh1 = vec![0.0; n * d];
        mm(&dq, &p[WQ], &mut dh1, n, d, d, false, true, 0.0);
        mm(&dk, &p[WK], &mut dh1, n, d, d, false, true, 1.0);
        mm(&dv, &p[WV], &mut dh1, n, d, d, false, true, 1.0);
        let mut dx0 = dx1.clone();
        let mut dx0n = vec![0.0; n * d];
        train::rms_bwd(&tr.x0, &p[N1], &tr.inv1, &dh1, &mut dx0n, &mut gr[N1], d);
        dx0.iter_mut().zip(&dx0n).for_each(|(a, b)| *a += b);
        mm(f, &dx0, &mut gr[FC_F], d, d, n, true, false, 1.0);
        mm(e, &dx0, &mut gr[FC_E], d, d, n, true, false, 1.0);
        loss
    }

    /// One AdamW step over every parameter, the gradients cleared
    pub fn step(&mut self) {
        self.step += 1;
        for i in 0..self.params.len() {
            train::adamw(&mut self.params[i], &self.grads[i], &mut self.m[i], &mut self.v[i], self.cfg.lr, 0.9, 0.999, 1e-8, self.cfg.weight_decay, self.step);
            self.grads[i].iter_mut().for_each(|g| *g = 0.0);
        }
    }

    /// One training step over the batch, returning the loss before the update
    pub fn train(&mut self, f: &[f32], e: &[f32], target: &[f32], b: usize, t: usize) -> f32 {
        let tr = self.forward(f, e, b, t);
        let loss = self.backward(f, e, target, &tr, b, t);
        self.step();
        loss
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The head's backward agrees with central differences of its loss on every parameter
    #[test]
    fn head_gradients_match_finite_differences() {
        let cfg = HeadConfig { d: 8, heads: 2, ff: 12, window: 3, eps: 1e-5, lr: 1e-3, weight_decay: 0.0, seed: 9 };
        let (b, t) = (2usize, 4usize);
        let n = b * t;
        let mut seed = 77u64;
        let f = init_values(n * cfg.d, 1.0, &mut seed);
        let e = init_values(n * cfg.d, 1.0, &mut seed);
        let y = init_values(n * cfg.d, 1.0, &mut seed);
        let mut head = CpuHead::new(cfg).expect("head");
        let tr = head.forward(&f, &e, b, t);
        head.backward(&f, &e, &y, &tr, b, t);
        let eps = 1e-3f32;
        for pi in 0..head.params.len() {
            for i in (0..head.params[pi].len()).step_by(5) {
                let saved = head.params[pi][i];
                head.params[pi][i] = saved + eps;
                let up = {
                    let tr = head.forward(&f, &e, b, t);
                    let mut g = vec![0.0; n * cfg.d];
                    train::smooth_l1(&tr.pred, &y, &mut g, 1.0)
                };
                head.params[pi][i] = saved - eps;
                let down = {
                    let tr = head.forward(&f, &e, b, t);
                    let mut g = vec![0.0; n * cfg.d];
                    train::smooth_l1(&tr.pred, &y, &mut g, 1.0)
                };
                head.params[pi][i] = saved;
                let fd = (f64::from(up) - f64::from(down)) / (2.0 * f64::from(eps));
                let an = f64::from(head.grads[pi][i]);
                assert!((fd - an).abs() < 2e-3 * (1.0 + an.abs()), "parameter {pi} entry {i}: analytic {an} against difference {fd}");
            }
        }
    }
}

/// Children per depth of a drafted tree, the first depth widest, as the device drafter grows them
pub const DRAFT_FANOUT: [usize; 5] = [4, 2, 2, 1, 1];

/// A scored row kept for the head: its slot, position, token, epoch, and hidden values
#[derive(Clone, Debug, Default)]
pub struct Row {
    pub seq: u32,
    pub pos: u32,
    pub token: u32,
    pub epoch: u32,
    pub feat: Vec<f32>,
}

/// The head at work on the host: the rows passes left, the trainer, per slot windows of pairs, held rows, and the
/// draft vocabulary with the model's head rows dequantized
pub struct CpuDrafter {
    pub head: CpuHead,
    pub rows: Vec<Row>,
    cap: usize,
    /// Rows appended since the start, those since dropped included
    pub appended: u64,
    epoch: Vec<u32>,
    /// Per slot the pairs the window holds, oldest first, each the head's input row f and the next token
    windows: Vec<Vec<(Vec<f32>, u32)>>,
    /// Per slot the position after the last pair in the window and the epoch the window follows
    next_pos: Vec<u32>,
    epoch_seen: Vec<u32>,
    held: Vec<Option<(Vec<f32>, u32)>>,
    pub vocab: Vec<u32>,
    vocab_rows: Vec<f32>,
    embd: Vec<f32>,
    n_vocab: usize,
    draft_vocab: usize,
    batch: usize,
    rng: u64,
    pub steps: usize,
}

impl CpuDrafter {
    pub fn new(cfg: HeadConfig, n_seq: usize, embd: Vec<f32>, head_rows: Vec<f32>, n_vocab: usize, draft_vocab: usize, batch: usize, cap: usize) -> Result<Self> {
        ensure!(embd.len() == n_vocab * cfg.d && head_rows.len() == n_vocab * cfg.d, "embedding and head tables of {} and {} values for {n_vocab} tokens of {}", embd.len(), head_rows.len(), cfg.d);
        let n = n_seq.max(1);
        let mut s = Self { head: CpuHead::new(cfg)?, rows: Vec::new(), cap, appended: 0, epoch: vec![0; n], windows: vec![Vec::new(); n], next_pos: vec![0; n], epoch_seen: vec![u32::MAX; n], held: vec![None; n], vocab: Vec::new(), vocab_rows: head_rows, embd, n_vocab, draft_vocab: draft_vocab.clamp(1, n_vocab), batch: batch.max(1), rng: cfg.seed.max(1), steps: 0 };
        s.refresh_vocab();
        Ok(s)
    }

    pub fn roll_back(&mut self, seq: usize) {
        if let Some(e) = self.epoch.get_mut(seq) {
            *e += 1;
        }
        if seq < self.windows.len() {
            self.windows[seq].clear();
            self.next_pos[seq] = 0;
            self.epoch_seen[seq] = u32::MAX;
            self.held[seq] = None;
        }
    }

    /// Appends a scored row of a slot at a position with the token fed there
    pub fn append(&mut self, seq: usize, pos: u32, token: u32, feat: Vec<f32>) {
        let epoch = self.epoch.get(seq).copied().unwrap_or(0);
        if self.rows.len() >= self.cap {
            self.rows.remove(0);
        }
        self.rows.push(Row { seq: seq as u32, pos, token, epoch, feat });
        self.appended += 1;
    }

    /// Runs of rows of one slot and epoch at consecutive positions, as row indices
    fn runs(&self) -> Vec<Vec<usize>> {
        let mut order: Vec<usize> = (0..self.rows.len()).collect();
        order.sort_by_key(|i| (self.rows[*i].seq, self.rows[*i].epoch, self.rows[*i].pos));
        let mut runs: Vec<Vec<usize>> = Vec::new();
        for (k, i) in order.iter().enumerate() {
            let joins = k > 0 && {
                let (a, b) = (&self.rows[order[k - 1]], &self.rows[*i]);
                a.seq == b.seq && a.epoch == b.epoch && a.pos + 1 == b.pos
            };
            if joins {
                runs.last_mut().expect("a run").push(*i);
            } else {
                runs.push(vec![*i]);
            }
        }
        runs
    }

    fn next(&mut self) -> u64 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        self.rng
    }

    /// One training step over batch chunks of t consecutive rows, the loss before the step and the share of the
    /// target rows whose following token stood among the children the first depth drafts, none while no run holds
    /// t plus two rows
    pub fn step(&mut self, t: usize) -> Option<TrainStep> {
        let d = self.head.cfg.d;
        let runs = self.runs();
        let spans: Vec<(usize, usize)> = runs.iter().enumerate().filter(|(_, r)| r.len() > t + 1).map(|(i, r)| (i, r.len() - t - 1)).collect();
        let total: usize = spans.iter().map(|(_, n)| *n).sum();
        if total == 0 {
            return None;
        }
        let (mut f, mut e, mut y, mut truth) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for _ in 0..self.batch {
            let mut pick = (self.next() % total as u64) as usize;
            let mut chosen = (0, 0);
            for (i, n) in &spans {
                if pick < *n {
                    chosen = (*i, pick);
                    break;
                }
                pick -= n;
            }
            let run = &runs[chosen.0];
            for k in 0..t {
                let (a, b, c) = (&self.rows[run[chosen.1 + k]], &self.rows[run[chosen.1 + k + 1]], &self.rows[run[chosen.1 + k + 2]]);
                f.extend_from_slice(&a.feat);
                e.extend_from_slice(&self.embd[b.token as usize * d..(b.token as usize + 1) * d]);
                y.extend_from_slice(&b.feat);
                truth.push(c.token);
            }
        }
        let tr = self.head.forward(&f, &e, self.batch, t);
        let loss = self.head.backward(&f, &e, &y, &tr, self.batch, t);
        let n = self.batch * t;
        let hits = (0..n).filter(|j| self.children(&tr.pred[j * d..(j + 1) * d], DRAFT_FANOUT[0]).contains(&truth[*j])).count();
        self.head.step();
        self.steps += 1;
        Some(TrainStep { loss, hit: hits as f32 / n as f32 })
    }

    pub fn hold(&mut self, seq: usize, feat: Vec<f32>, pos: u32) {
        if seq < self.held.len() {
            self.held[seq] = Some((feat, pos));
        }
    }

    /// Brings a slot's window up to its rows: the window follows the slot's latest run of consecutive positions in
    /// its current epoch, starts over when a new run or epoch begins, and adds every pair of that run it lacks
    pub fn sync(&mut self, seq: usize) {
        if seq >= self.windows.len() {
            return;
        }
        let epoch = self.epoch[seq];
        let by_pos: HashMap<u32, usize> = self.rows.iter().enumerate().filter(|(_, r)| r.seq as usize == seq && r.epoch == epoch).map(|(i, r)| (r.pos, i)).collect();
        let Some(hi) = by_pos.keys().copied().max() else { return };
        let mut lo = hi;
        while lo > 0 && by_pos.contains_key(&(lo - 1)) {
            lo -= 1;
        }
        if self.epoch_seen[seq] != epoch || self.next_pos[seq] < lo || self.next_pos[seq] > hi + 1 {
            self.epoch_seen[seq] = epoch;
            self.next_pos[seq] = lo;
            self.windows[seq].clear();
        }
        let keep = self.head.cfg.window - 1;
        for pos in self.next_pos[seq]..hi {
            let (cur, nxt) = (&self.rows[by_pos[&pos]], &self.rows[by_pos[&(pos + 1)]]);
            self.windows[seq].push((cur.feat.clone(), nxt.token));
            if self.windows[seq].len() > keep {
                self.windows[seq].remove(0);
            }
            self.next_pos[seq] = pos + 1;
        }
    }

    /// The draft vocabulary as the most frequent tokens of the rows kept, in rising id order
    pub fn refresh_vocab(&mut self) {
        let mut counts = vec![0u64; self.n_vocab];
        for r in &self.rows {
            if (r.token as usize) < self.n_vocab {
                counts[r.token as usize] += 1;
            }
        }
        let mut ids: Vec<u32> = (0..self.n_vocab as u32).collect();
        ids.sort_by(|a, b| counts[*b as usize].cmp(&counts[*a as usize]).then(a.cmp(b)));
        ids.truncate(self.draft_vocab);
        ids.sort_unstable();
        self.vocab = ids;
    }

    /// The prediction of the head for a row f with next token tok after the pairs given, attending over them and itself
    fn predict(&self, pairs: &[(Vec<f32>, u32)], f: &[f32], tok: u32) -> Vec<f32> {
        let d = self.head.cfg.d;
        let t = pairs.len() + 1;
        let (mut ff, mut ee) = (Vec::with_capacity(t * d), Vec::with_capacity(t * d));
        for (pf, ptok) in pairs {
            ff.extend_from_slice(pf);
            ee.extend_from_slice(&self.embd[*ptok as usize * d..(*ptok as usize + 1) * d]);
        }
        ff.extend_from_slice(f);
        ee.extend_from_slice(&self.embd[tok as usize * d..(tok as usize + 1) * d]);
        let tr = self.head.forward(&ff, &ee, 1, t);
        tr.pred[(t - 1) * d..t * d].to_vec()
    }

    /// The k most likely draft vocabulary tokens of a predicted row under the model's head, best first
    fn children(&self, pred: &[f32], k: usize) -> Vec<u32> {
        let d = self.head.cfg.d;
        let mut scored: Vec<(f32, u32)> = self.vocab.iter().map(|tok| ((0..d).map(|c| pred[c] * self.vocab_rows[*tok as usize * d + c]).sum::<f32>(), *tok)).collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        scored.iter().take(k).map(|(_, t)| *t).collect()
    }

    /// A tree rooted at own of at most budget rows, each depth DRAFT_FANOUT children wide, every node predicting from
    /// its ancestors' predictions over the slot's window, the window read only while its pairs run up to the held row
    pub fn draft(&self, seq: usize, own: u32, budget: usize) -> kf_core::Tree {
        let mut tree = kf_core::Tree::root(own);
        let Some((held, pos)) = self.held.get(seq).and_then(|h| h.as_ref()) else { return tree };
        let budget = budget.min(kf_core::TREE_ROWS);
        if budget <= 1 || self.vocab.is_empty() {
            return tree;
        }
        let window: &[(Vec<f32>, u32)] = if self.next_pos[seq] == *pos { &self.windows[seq] } else { &[] };
        // Every node's pairs: the window, then the ancestors as (prediction of the parent, node token)
        let mut node_pairs: Vec<Vec<(Vec<f32>, u32)>> = vec![window.to_vec()];
        let mut node_pred: Vec<Vec<f32>> = vec![self.predict(&node_pairs[0], held, own)];
        let mut node_f: Vec<Vec<f32>> = vec![held.clone()];
        let mut frontier = vec![0usize];
        let mut depth = 0;
        while !frontier.is_empty() && tree.len() < budget && depth < DRAFT_FANOUT.len() {
            let fan = DRAFT_FANOUT[depth].min(budget - tree.len()).max(1);
            let mut next = Vec::new();
            for n in &frontier {
                for tok in self.children(&node_pred[*n], fan) {
                    if tree.len() >= budget {
                        break;
                    }
                    if tree.has_child(*n, tok) {
                        continue;
                    }
                    let Some(row) = tree.push(*n, tok) else { break };
                    let mut pairs = node_pairs[*n].clone();
                    pairs.push((node_f[*n].clone(), tree.tokens[*n]));
                    let f = node_pred[*n].clone();
                    let pred = self.predict(&pairs, &f, tok);
                    node_pairs.push(pairs);
                    node_f.push(f);
                    node_pred.push(pred);
                    next.push(row);
                }
            }
            frontier = next;
            depth += 1;
        }
        tree
    }
}

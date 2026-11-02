//! The draft block, one decoder layer kungfu teaches itself to run ahead of the model, see docs/drafter.md
use anyhow::{Result, bail, ensure};

/// Most rows one tree of drafts holds, the keys one ancestor mask covers
pub const DRAFT_ROWS: usize = super::TREE_ROWS;

/// Consecutive rows one chunk of the block's training holds
pub const DRAFT_CHUNK: usize = 64;

/// Candidates a pass keeps per row, the soft target the block learns from
pub const DRAFT_CANDS: usize = 64;

/// Scored rows the ring keeps for the block to learn from
pub const DRAFT_RING_ROWS: usize = 16384;

/// Most positions one pass of the block drafts, the fanout table's length
pub const DRAFT_DEPTH: usize = 8;

/// Most learners an ensemble trains side by side
pub const DRAFT_LEARNERS: usize = 4;

/// Rows of one layer representative the tap selection averages token blocks into
pub const DRAFT_CAL_BLOCK: usize = 32;

/// Share of a pass's rows the tap selection keeps, the highest predictive entropy ones
pub const DRAFT_CAL_KEEP: f32 = 0.3;

/// What the block's training minimizes on the rows that followed the model's own path
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Objective {
    /// Cross entropy against the model's candidates at every drafted position
    Ce,
    /// Expected decoding rounds, every position's rejection cost weighted by the chance the round reaches it
    Edr,
    /// Log of the expected accepted length under block verification, the vocabulary integrated form
    Bv,
}

impl Objective {
    pub fn name(self) -> &'static str {
        match self {
            Objective::Ce => "ce",
            Objective::Edr => "edr",
            Objective::Bv => "bv",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "ce" => Some(Objective::Ce),
            "edr" => Some(Objective::Edr),
            "bv" => Some(Objective::Bv),
            _ => None,
        }
    }
}

/// Shape and schedule of a draft block, the fused tap rows with the next token's embedding through one attention
/// layer and a gated MLP, read through the model's own head over a draft vocabulary, the block's mask rows drafting
/// the positions after the first in the same pass
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DraftConfig {
    /// Width of the model's residual rows and of the block
    pub d: usize,
    pub heads: usize,
    pub ff: usize,
    /// Positions a row attends back over, its own included
    pub window: usize,
    /// Layers whose residual rows the block fuses
    pub taps: usize,
    /// Positions one pass drafts, the root row and depth minus one mask rows
    pub depth: usize,
    /// Children scored per position of the block, the root's first
    pub fanout: [usize; DRAFT_DEPTH],
    /// Learners trained side by side at geometrically spaced rates
    pub learners: usize,
    pub objective: Objective,
    /// Share of a batch's targets the training keeps, the widest gaps between the learner and the reference first
    pub keep_frac: f32,
    /// Weight the reference keeps of its own parameters each step
    pub ema: f32,
    /// Rate of the hedge over the learners' running losses
    pub hedge: f32,
    pub eps: f32,
    pub lr: f32,
    pub weight_decay: f32,
    pub seed: u64,
}

impl DraftConfig {
    /// A block for a model of width d with heads of 128 when the width allows and sixteen otherwise, a gated MLP
    /// twice as wide, a window of 64, three taps, five drafted positions, three learners
    pub fn for_width(d: usize) -> Self {
        let heads = if d.is_multiple_of(128) { d / 128 } else { 16 };
        Self { d, heads, ff: 2 * d, window: 64, taps: 3, depth: 5, fanout: [4, 3, 2, 2, 2, 1, 1, 1], learners: 3, objective: Objective::Edr, keep_frac: 0.4, ema: 0.999, hedge: 10.0, eps: 1e-5, lr: 3e-4, weight_decay: 0.01, seed: 0x5eed }
    }

    pub fn head_dim(&self) -> usize {
        self.d / self.heads
    }

    /// Parameters in order, the fusion of the tap rows, fc of g and of e, q k v o, norm one, gate up down, norm two,
    /// the mask rows of the drafted positions after the first
    pub fn param_shapes(&self) -> Vec<(&'static str, usize, usize)> {
        let (d, ff) = (self.d, self.ff);
        vec![("fuse", self.taps * d, d), ("fc_g", d, d), ("fc_e", d, d), ("wq", d, d), ("wk", d, d), ("wv", d, d), ("wo", d, d), ("n1", 1, d), ("wg", d, ff), ("wu", d, ff), ("wd", ff, d), ("n2", 1, d), ("mask", self.depth - 1, d)]
    }

    pub fn n_params(&self) -> usize {
        self.param_shapes().iter().map(|(_, r, c)| r * c).sum()
    }

    /// Rows one training chunk reads from a run, the chunk, the drafted positions, and the last target's row
    pub fn chunk_rows(&self) -> usize {
        DRAFT_CHUNK + self.depth + 1
    }

    /// Learning rate of learner i, the base rate spread by factors of two around the middle learner
    pub fn learner_lr(&self, i: usize) -> f32 {
        self.lr * 2f32.powf(i as f32 - (self.learners as f32 - 1.0) / 2.0)
    }

    /// Device bytes the block takes to train and draft, every buffer the backends allocate counted at its size,
    /// a tenth over and 256 MiB for the allocator's rounding and the kernel modules, every learner's parameters with gradients, moments, and
    /// last directions, the mixture and the reference, the int8 snapshot with its activation scratch, the training
    /// work of batch chunks through both query groups at every drafted position with their logits, the drafting
    /// work, the ring, every slot's history and last rows, the vocabulary's head rows in native and single precision,
    /// and the tap statistics of n_layers layers
    pub fn device_bytes(&self, batch: usize, n_seq: usize, ring_rows: usize, draft_vocab: usize, n_vocab: usize, n_layers: usize) -> u64 {
        let (d, ff, h, v, k) = (self.d as u64, self.ff as u64, self.heads as u64, draft_vocab as u64, self.depth as u64);
        let (td, n, chunk, steps) = (self.taps as u64 * d, (batch * DRAFT_CHUNK) as u64, DRAFT_CHUNK as u64, 2 * self.depth as u64);
        let (cand_w, fan, rows, anc) = (2 * DRAFT_CANDS as u64 + 1, self.fanout.iter().take(self.depth).copied().max().unwrap_or(1) as u64, 512u64, 16u64);
        let np = self.n_params() as u64;
        let slices = v.div_ceil(8192);
        let params = (5 * self.learners as u64 + 2) * 4 * np + np * 34 / 32 + 32 * td.max(ff) * 2;
        let probs = batch as u64 * h * chunk * (chunk + k);
        let work = 4 * (n * td + 2 * n * d + 10 * steps * n * d + 3 * steps * n + 3 * steps * n * ff + n * d.max(ff) + steps * probs + 2 * steps * n * v + 2 * n * d + steps * n * d + 3 * n * ff + 9 * n * d + 4 * steps * n * d + d + 1 + 2 * steps * n + steps * n * cand_w + 4 * k * n + n * (2 * fan + 1) + n * slices * fan * 2 + n * slices * 2)
            + 4 * (1 + 2 * n + (1 + steps) * n + (k - 1) * n + k * n);
        let drafting = 4 * (rows * td + 2 * rows * d + 12 * rows * d + 2 * rows + 3 * rows * ff + rows * d.max(ff) + 2 * DRAFT_DEPTH as u64 * d + DRAFT_DEPTH as u64 * (1 + v + 2 * fan + 1 + slices * fan * 2 + slices * 2)) + 4 * rows * (anc + 3);
        let ring = 4 * (ring_rows as u64 * td + (ring_rows as u64 + 1) * cand_w);
        let slots = n_seq as u64 * (4 * (2 * 2 * self.window as u64 * d + rows * td) + 4 * 2 * self.window as u64);
        let vocab = v * d * 4 + v * d * 2 + n_vocab as u64 * 4 + d * 4;
        let taps = n_layers as u64 * DRAFT_CAL_BLOCK as u64 * d * 4;
        (params + work + drafting + ring + slots + vocab + taps) * 11 / 10 + (256 << 20)
    }

    pub fn check(&self) -> Result<()> {
        ensure!(self.d > 0 && self.heads > 0 && self.d.is_multiple_of(self.heads), "a block width of {} over {} heads", self.d, self.heads);
        ensure!(self.d.is_multiple_of(32) && self.ff.is_multiple_of(32), "a block of width {} and feed forward {} for int8 rows of 32", self.d, self.ff);
        ensure!(self.window >= 2 && self.taps >= 1 && (2..=DRAFT_DEPTH).contains(&self.depth), "a window of {}, {} taps, {} depths", self.window, self.taps, self.depth);
        ensure!(self.fanout.iter().take(self.depth).all(|f| *f >= 1), "a fanout of {:?} over {} positions", self.fanout, self.depth);
        ensure!((1..=DRAFT_LEARNERS).contains(&self.learners), "{} learners of at most {DRAFT_LEARNERS}", self.learners);
        ensure!(self.keep_frac > 0.0 && self.keep_frac <= 1.0 && self.ema >= 0.0 && self.ema < 1.0 && self.hedge > 0.0, "a kept share of {}, a reference weight of {}, a hedge rate of {}", self.keep_frac, self.ema, self.hedge);
        Ok(())
    }
}

/// The model's distribution at one row as a pass keeps it, its most likely tokens with their logits and the row's log sum exp
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Soft {
    pub ids: Vec<u32>,
    pub logits: Vec<f32>,
    pub lse: f32,
}

impl Soft {
    /// From the packed row a top k op writes, k ids as floats, k logits, and the log sum exp
    pub fn from_packed(packed: &[f32]) -> Result<Self> {
        ensure!(packed.len() >= 3 && !packed.len().is_multiple_of(2), "{} packed values for a candidate row", packed.len());
        let k = (packed.len() - 1) / 2;
        Ok(Self { ids: packed[..k].iter().map(|v| *v as u32).collect(), logits: packed[k..2 * k].to_vec(), lse: packed[2 * k] })
    }

    /// Probability of each candidate under the model
    pub fn probs(&self) -> Vec<f32> {
        self.logits.iter().map(|l| (l - self.lse).exp()).collect()
    }

    /// Probability of one token, zero when it is not a candidate
    pub fn prob(&self, token: u32) -> f32 {
        self.ids.iter().position(|t| *t == token).map_or(0.0, |i| (self.logits[i] - self.lse).exp())
    }
}

/// One row of a scored pass as the drafter reads it, its slot, position, token, and the fused tap rows side by side
#[derive(Clone, Debug, Default)]
pub struct Scored {
    pub seq: u32,
    pub pos: u32,
    pub token: u32,
    pub taps: Vec<f32>,
    pub soft: Soft,
}

/// The saved form of a block, a magic word, the configuration, the tap layers, the vocabulary, and every parameter
/// in order, little endian
pub fn draft_save(cfg: &DraftConfig, tap_layers: &[usize], vocab: &[u32], params: &[Vec<f32>]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"KFDRFT03");
    for v in [cfg.d, cfg.heads, cfg.ff, cfg.window, cfg.taps, cfg.depth] {
        out.extend_from_slice(&(v as u32).to_le_bytes());
    }
    out.extend_from_slice(&cfg.eps.to_le_bytes());
    out.extend_from_slice(&(tap_layers.len() as u32).to_le_bytes());
    for l in tap_layers {
        out.extend_from_slice(&(*l as u32).to_le_bytes());
    }
    out.extend_from_slice(&(vocab.len() as u32).to_le_bytes());
    for v in vocab {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(&(params.len() as u32).to_le_bytes());
    for p in params {
        out.extend_from_slice(&(p.len() as u32).to_le_bytes());
        for v in p {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    out
}

/// What a saved block holds, its tap layers, its vocabulary, and its parameters in order
pub type SavedBlock = (Vec<usize>, Vec<u32>, Vec<Vec<f32>>);

/// The tap layers, vocabulary, and parameters of a saved block, checked against the configuration they load into
pub fn draft_load(cfg: &DraftConfig, bytes: &[u8]) -> Result<SavedBlock> {
    let mut at = 0usize;
    let mut take = |n: usize| -> Result<&[u8]> {
        ensure!(at + n <= bytes.len(), "a saved block of {} bytes ends early", bytes.len());
        at += n;
        Ok(&bytes[at - n..at])
    };
    let u32_of = |b: &[u8]| u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    if take(8)? != b"KFDRFT03" {
        bail!("not a saved draft block of this form");
    }
    let dims: Vec<usize> = (0..6).map(|_| take(4).map(|b| u32_of(b) as usize)).collect::<Result<_>>()?;
    let want = [cfg.d, cfg.heads, cfg.ff, cfg.window, cfg.taps, cfg.depth];
    ensure!(dims == want, "a saved block of shape {dims:?} for a block of {want:?}");
    let eps = f32::from_le_bytes(take(4)?.try_into().expect("four bytes"));
    ensure!(eps == cfg.eps, "a saved block with eps {eps} for a block with {}", cfg.eps);
    let nt = u32_of(take(4)?) as usize;
    ensure!(nt == cfg.taps, "a saved block tapping {nt} layers for a block of {} taps", cfg.taps);
    let taps: Vec<usize> = (0..nt).map(|_| take(4).map(|b| u32_of(b) as usize)).collect::<Result<_>>()?;
    let nv = u32_of(take(4)?) as usize;
    let vocab: Vec<u32> = (0..nv).map(|_| take(4).map(u32_of)).collect::<Result<_>>()?;
    let np = u32_of(take(4)?) as usize;
    let shapes = cfg.param_shapes();
    ensure!(np == shapes.len(), "a saved block of {np} parameter tensors for a block of {}", shapes.len());
    let mut params = Vec::with_capacity(np);
    for (_, r, c) in &shapes {
        let n = u32_of(take(4)?) as usize;
        ensure!(n == r * c, "a saved parameter of {n} values for a tensor of {r} by {c}");
        let raw = take(4 * n)?;
        params.push(raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect());
    }
    Ok((taps, vocab, params))
}

/// One scored row's place in the ring, its slot, position, token, epoch, whether it stands on the model's own
/// path, its parent among the ring's rows, and the rows below it
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Entry {
    pub seq: u32,
    pub pos: u32,
    pub token: u32,
    pub epoch: u32,
    pub live: bool,
    pub main: bool,
    pub parent: i32,
    pub kids: Vec<u32>,
}

/// What one training chunk reads from the ring, the rows whose features it takes and, per row, the main group's
/// first token with the rows of its targets and realized tokens along the model's own path, and the branch
/// group's first token with the rows of its targets along a rejected branch, minus one where none
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ChunkIdx {
    pub rows: Vec<u32>,
    pub main_tokens: Vec<u32>,
    pub main_targets: Vec<Vec<i32>>,
    pub main_truth: Vec<Vec<i32>>,
    pub branch_tokens: Vec<i32>,
    pub branch_targets: Vec<Vec<i32>>,
}

/// The bookkeeping of a ring of scored rows, shared by every backend so all sample the same chunks, every row of
/// every pass with its parent, each slot's rows joining its current epoch, which rises whenever the slot rolls
/// back so no chunk crosses a rollback
#[derive(Clone, Debug)]
pub struct RingMeta {
    entries: Vec<Entry>,
    pub cap: usize,
    head: usize,
    epoch: Vec<u32>,
    /// Rows appended since the start, those since dropped included
    pub appended: u64,
    rng: u64,
}

impl RingMeta {
    pub fn new(cap: usize, n_seq: usize, seed: u64) -> Self {
        let cap = cap.max(1);
        Self { entries: vec![Entry::default(); cap], cap, head: 0, epoch: vec![0; n_seq.max(1)], appended: 0, rng: seed.max(1) }
    }

    pub fn roll_back(&mut self, seq: usize) {
        if let Some(e) = self.epoch.get_mut(seq) {
            *e += 1;
        }
    }

    pub fn entry(&self, i: u32) -> &Entry {
        &self.entries[i as usize]
    }

    /// The index the next appended row takes, the ones after it following modulo the capacity
    pub fn next_index(&self) -> u32 {
        self.head as u32
    }

    /// Appends a row below a parent, on the model's own path when main, returning its ring index, the row the
    /// slot it takes held dropped with its links
    pub fn append(&mut self, seq: u32, pos: u32, token: u32, main: bool, parent: i32) -> u32 {
        let at = self.head;
        let old = std::mem::take(&mut self.entries[at]);
        if old.live {
            for k in old.kids {
                self.entries[k as usize].parent = -1;
            }
            if old.parent >= 0 {
                self.entries[old.parent as usize].kids.retain(|k| *k != at as u32);
            }
        }
        let parent = if parent >= 0 && self.entries[parent as usize].live { parent } else { -1 };
        let epoch = self.epoch.get(seq as usize).copied().unwrap_or(0);
        self.entries[at] = Entry { seq, pos, token, epoch, live: true, main, parent, kids: Vec::new() };
        if parent >= 0 {
            self.entries[parent as usize].kids.push(at as u32);
        }
        self.head = (self.head + 1) % self.cap;
        self.appended += 1;
        at as u32
    }

    /// Puts a row on the model's own path at a position, the compaction of a tree pass
    pub fn promote(&mut self, i: u32, pos: u32) {
        let e = &mut self.entries[i as usize];
        if e.live {
            e.main = true;
            e.pos = pos;
        }
    }

    pub fn len(&self) -> usize {
        self.entries.iter().filter(|e| e.live).count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Runs of rows on the model's own path of one slot and epoch at consecutive positions, as ring indices in position order
    pub fn runs(&self) -> Vec<Vec<u32>> {
        let mut live: Vec<(u32, u32, u32, u32)> = self.entries.iter().enumerate().filter(|(_, e)| e.live && e.main).map(|(i, e)| (e.seq, e.epoch, e.pos, i as u32)).collect();
        live.sort_unstable();
        let mut runs: Vec<Vec<u32>> = Vec::new();
        for (k, e) in live.iter().enumerate() {
            let joins = k > 0 && live[k - 1].0 == e.0 && live[k - 1].1 == e.1 && live[k - 1].2 + 1 == e.2;
            if joins {
                runs.last_mut().expect("a run").push(e.3);
            } else {
                runs.push(vec![e.3]);
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

    /// Batch chunks of t rows drawn from runs long enough to hold them with their drafted positions and last
    /// target, each row's branch group following a rejected row below it, none while no run is long enough
    pub fn chunks(&mut self, batch: usize, t: usize, depth: usize) -> Option<Vec<ChunkIdx>> {
        let need = t + depth + 1;
        let runs = self.runs();
        let spans: Vec<(usize, usize)> = runs.iter().enumerate().filter(|(_, r)| r.len() >= need).map(|(i, r)| (i, r.len() - need + 1)).collect();
        let total: usize = spans.iter().map(|(_, n)| *n).sum();
        if total == 0 {
            return None;
        }
        let mut out = Vec::with_capacity(batch);
        for _ in 0..batch {
            let mut pick = (self.next() % total as u64) as usize;
            let mut chosen = (0, 0);
            for (i, n) in &spans {
                if pick < *n {
                    chosen = (*i, pick);
                    break;
                }
                pick -= n;
            }
            let run: Vec<u32> = runs[chosen.0][chosen.1..chosen.1 + need].to_vec();
            let mut c = ChunkIdx { rows: run[..t].to_vec(), main_tokens: Vec::with_capacity(t), main_targets: vec![Vec::with_capacity(t); depth], main_truth: vec![Vec::with_capacity(t); depth], branch_tokens: Vec::with_capacity(t), branch_targets: vec![Vec::with_capacity(t); depth] };
            for i in 0..t {
                c.main_tokens.push(self.entry(run[i + 1]).token);
                for j in 0..depth {
                    c.main_targets[j].push(run[i + 1 + j] as i32);
                    c.main_truth[j].push(self.entry(run[i + 2 + j]).token as i32);
                }
                let branches: Vec<u32> = self.entry(run[i]).kids.iter().copied().filter(|k| !self.entry(*k).main).collect();
                if branches.is_empty() {
                    c.branch_tokens.push(-1);
                    for j in 0..depth {
                        c.branch_targets[j].push(-1);
                    }
                    continue;
                }
                let mut at = branches[(self.next() % branches.len() as u64) as usize];
                c.branch_tokens.push(self.entry(at).token as i32);
                let mut ended = false;
                for j in 0..depth {
                    if ended {
                        c.branch_targets[j].push(-1);
                        continue;
                    }
                    c.branch_targets[j].push(at as i32);
                    let kids = self.entry(at).kids.clone();
                    if kids.is_empty() {
                        ended = true;
                    } else {
                        at = kids[(self.next() % kids.len() as u64) as usize];
                    }
                }
            }
            out.push(c);
        }
        Some(out)
    }

    /// The v most frequent tokens of the rows kept, in rising id order
    pub fn frequent(&self, n_vocab: usize, v: usize) -> Vec<u32> {
        let mut counts = vec![0u64; n_vocab];
        for e in self.entries.iter().filter(|e| e.live) {
            if (e.token as usize) < n_vocab {
                counts[e.token as usize] += 1;
            }
        }
        let mut ids: Vec<u32> = (0..n_vocab as u32).collect();
        ids.sort_by(|a, b| counts[*b as usize].cmp(&counts[*a as usize]).then(a.cmp(b)));
        ids.truncate(v.min(n_vocab));
        ids.sort_unstable();
        ids
    }
}

/// A tree of the block's drafts below a root token from the block's distribution at every drafted position, the
/// paths of highest product probability first, every row's children among the position's top candidates
pub fn tree_of_block(root: u32, tops: &[Vec<(u32, f32)>], rows: usize) -> super::Tree {
    let mut tree = super::Tree::root(root);
    if rows <= 1 || tops.is_empty() {
        return tree;
    }
    // Open edges by product probability, ties by the shallower row then the earlier candidate
    let mut open: Vec<(f32, usize, usize, usize)> = tops[0].iter().enumerate().map(|(c, (_, p))| (*p, 0usize, 0usize, c)).collect();
    while tree.len() < rows {
        let Some(best) = open.iter().enumerate().max_by(|a, b| a.1.0.total_cmp(&b.1.0).then(b.1.1.cmp(&a.1.1)).then(b.1.3.cmp(&a.1.3))).map(|(i, _)| i) else { break };
        let (p, depth, parent, c) = open.swap_remove(best);
        if p <= 0.0 {
            break;
        }
        let token = tops[depth][c].0;
        if tree.has_child(parent, token) {
            continue;
        }
        let Some(row) = tree.push(parent, token) else { break };
        if depth + 1 < tops.len() {
            open.extend(tops[depth + 1].iter().enumerate().map(|(cc, (_, q))| (p * q, depth + 1, row, cc)));
        }
    }
    tree
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tree rows join the ring as branches below the slot's last main row, compaction promotes the kept path, and
    /// chunks follow branches for the second query group
    #[test]
    fn ring_keeps_branches_and_samples_them() {
        let mut ring = RingMeta::new(64, 1, 3);
        let mut last = -1i32;
        for p in 0..8u32 {
            last = ring.append(0, p, p, true, last) as i32;
        }
        let r0 = ring.append(0, 8, 100, false, last);
        let r1 = ring.append(0, 9, 101, false, r0 as i32);
        let r2 = ring.append(0, 9, 102, false, r0 as i32);
        let r3 = ring.append(0, 10, 103, false, r1 as i32);
        for (i, r) in [r0, r1, r3].iter().enumerate() {
            ring.promote(*r, 8 + i as u32);
        }
        let runs = ring.runs();
        assert_eq!((runs.len(), runs[0].len()), (1, 11), "the kept path extends the run");
        assert_eq!(ring.entry(r0).kids, vec![r1, r2]);
        let chunks = ring.chunks(1, 4, 2).expect("chunks");
        let c = &chunks[0];
        assert_eq!((c.rows.len(), c.main_tokens.len(), c.main_targets.len()), (4, 4, 2));
        if let Some(i) = c.main_tokens.iter().position(|t| *t == 101) {
            assert_eq!(c.branch_tokens[i], 102);
            assert!(c.branch_targets[0][i] == r2 as i32 && c.branch_targets[1][i] == -1, "the branch ends after its one row");
        }
        assert!(c.main_truth[0].iter().all(|t| *t >= 0));
        let mut small = RingMeta::new(2, 1, 3);
        let a = small.append(0, 0, 1, true, -1);
        let b = small.append(0, 1, 2, true, a as i32);
        assert_eq!(small.entry(a).kids, vec![b]);
        small.append(0, 2, 3, true, b as i32);
        assert_eq!(small.entry(b).parent, -1, "the overwritten parent is forgotten");
        assert_eq!(small.frequent(10, 1), vec![2]);
    }

    #[test]
    fn block_trees_follow_product_probability() {
        let tops = vec![vec![(10u32, 0.6f32), (11, 0.3)], vec![(20, 0.5), (21, 0.4)], vec![(30, 0.9)]];
        let t = tree_of_block(5, &tops, 5);
        assert_eq!(t.tokens, vec![5, 10, 11, 20, 30], "equal products take the shallower row first");
        assert_eq!(t.parents, vec![-1, 0, 0, 1, 3]);
        let t = tree_of_block(5, &tops, 7);
        assert_eq!((t.tokens, t.parents), (vec![5, 10, 11, 20, 30, 21, 30], vec![-1, 0, 0, 1, 3, 1, 5]), "the second branch's child at 0.216 outranks the 0.15 edge");
        assert_eq!(tree_of_block(5, &tops, 1), crate::Tree::root(5));
        let cfg = DraftConfig::for_width(2048);
        assert!(cfg.check().is_ok());
        assert_eq!(cfg.param_shapes().len(), 13);
        assert!((cfg.learner_lr(1) - cfg.lr).abs() < 1e-9 && (cfg.learner_lr(2) - 2.0 * cfg.lr).abs() < 1e-9);
        let saved = draft_save(&cfg, &[3, 8, 15], &[1, 2], &cfg.param_shapes().iter().map(|(_, r, c)| vec![0.5; r * c]).collect::<Vec<_>>());
        let (taps, vocab, params) = draft_load(&cfg, &saved).expect("load");
        assert_eq!((taps, vocab, params.len()), (vec![3, 8, 15], vec![1, 2], 13));
    }
}

use crate::graph::{Graph, Op, Wid};
use crate::plan::Plan;
use anyhow::Result;
use std::ops::Range;

/// Raw weight bytes and their ggml type, backends dequantize or upload as they see fit
pub trait WeightSource: Send + Sync {
    fn bytes(&self, w: Wid) -> Result<&[u8]>;
    fn ggml_type(&self, w: Wid) -> u32;
    fn dims(&self, w: Wid) -> &[usize];
    fn name(&self, w: Wid) -> &str;
    fn len(&self) -> usize;
}

/// Most rows a tree pass may hold, the keys one ancestor mask covers
pub const TREE_ROWS: usize = 64;

/// Consecutive rows one chunk of the draft head's training holds, the span its attention sees
pub const HEAD_CHUNK: usize = 64;

/// Scored rows the feature ring keeps for the draft head to learn from
pub const FEATURE_RING_ROWS: usize = 16384;

/// What one training step of the draft head reports: the loss before the update and the share of the batch's
/// positions whose actual next token stood among the children the head's first depth would have drafted
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrainStep {
    pub loss: f32,
    pub hit: f32,
}

/// The keys of its own pass one row of a tree sees, bit k of mask standing for the key k below the row's last key, every
/// bit at or past the pass's row count set, and the row's position for the window rule, which its cache index is not
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TreeRow {
    pub mask: u64,
    pub position: u32,
}

impl TreeRow {
    /// Whether key j is visible to a row whose last key is hi, keys further than the mask reaches always being so
    pub fn sees(&self, hi: u32, j: u32) -> bool {
        j <= hi && (hi - j >= 64 || (self.mask >> (hi - j)) & 1 == 1)
    }
}

/// A tree of drafted tokens, the first the token already chosen for the next position and every other row's parent
/// an earlier row, see Session::feed_tree
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Tree {
    pub tokens: Vec<u32>,
    pub parents: Vec<i32>,
}

impl Tree {
    /// A tree of one root
    pub fn root(own: u32) -> Self {
        Self { tokens: vec![own], parents: vec![-1] }
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Adds a child of row parent and returns its row, none when the row would not fit a pass
    pub fn push(&mut self, parent: usize, token: u32) -> Option<usize> {
        if self.tokens.len() >= TREE_ROWS {
            return None;
        }
        self.tokens.push(token);
        self.parents.push(parent as i32);
        Some(self.tokens.len() - 1)
    }

    /// The child of row parent holding token, when one does
    pub fn child(&self, parent: usize, token: u32) -> Option<usize> {
        (parent + 1..self.tokens.len()).find(|i| self.parents[*i] == parent as i32 && self.tokens[*i] == token)
    }

    /// Whether row parent already has a child holding token
    pub fn has_child(&self, parent: usize, token: u32) -> bool {
        self.child(parent, token).is_some()
    }

    /// Adds a chain of tokens below the root, following the children already there and growing where it leaves
    /// them, no row added past limit rows in all
    pub fn graft(&mut self, chain: &[u32], limit: usize) {
        let mut at = 0;
        for &t in chain {
            at = match self.child(at, t) {
                Some(c) => c,
                None if self.len() >= limit => break,
                None => match self.push(at, t) {
                    Some(row) => row,
                    None => break,
                },
            };
        }
    }

    /// The tokens from the root to row i, the root first
    pub fn path_tokens(&self, i: usize) -> Vec<u32> {
        let mut out = Vec::new();
        let mut at = i as i32;
        while at >= 0 {
            out.push(self.tokens[at as usize]);
            at = self.parents[at as usize];
        }
        out.reverse();
        out
    }
}

/// The shape and schedule of a draft head, one attention layer over the last window features with a gated MLP, which
/// predicts the model's next residual row from the current one and the next token's embedding
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeadConfig {
    /// Width of the residual rows, the model's embedding width
    pub d: usize,
    pub heads: usize,
    /// Width of the gated MLP
    pub ff: usize,
    /// Positions a row attends back over
    pub window: usize,
    pub eps: f32,
    pub lr: f32,
    pub weight_decay: f32,
    pub seed: u64,
}

impl HeadConfig {
    /// A head for a model of width d, heads of 128 when the width allows and sixteen heads otherwise, a gated MLP
    /// twice as wide, a window of 64
    pub fn for_width(d: usize) -> Self {
        let heads = if d.is_multiple_of(128) { d / 128 } else { 16 };
        Self { d, heads, ff: 2 * d, window: 64, eps: 1e-5, lr: 3e-4, weight_decay: 0.01, seed: 0x5eed }
    }

    pub fn head_dim(&self) -> usize {
        self.d / self.heads
    }

    /// Device bytes the head takes to train and draft: parameters with their gradients and moments, the work of batch
    /// chunks with the scoring of their predictions, the drafting rows and the windows of n_seq slots, the feature
    /// ring, and the draft vocabulary's head rows
    pub fn device_bytes(&self, batch: usize, n_seq: usize, ring_rows: usize, draft_vocab: usize) -> u64 {
        let (d, ff, h, v) = (self.d as u64, self.ff as u64, self.heads as u64, draft_vocab as u64);
        let (n, rows, chunk) = ((batch * HEAD_CHUNK) as u64, TREE_ROWS as u64, HEAD_CHUNK as u64);
        let params = 4 * 4 * self.n_params() as u64;
        let work = 4 * (23 * n * d + 6 * n * ff + batch as u64 * h * chunk * chunk);
        let scoring = 4 * ((n + rows) * (v + d + 2 * TREE_ROWS as u64));
        let draft = 4 * (14 * rows * d + 3 * rows * ff + 8 * rows * d.max(ff));
        let windows = 4 * (n_seq as u64 * (2 * (self.window as u64 - 1) + 1) * d);
        let ring = 4 * ring_rows as u64 * d;
        let vocab = v * d * 2 + v * d / 32 * 4;
        params + work + scoring + draft + windows + ring + vocab
    }

    /// Parameters of the head in order: fc of f and of e, q k v o, norm one, gate up down, norm two
    pub fn param_shapes(&self) -> Vec<(&'static str, usize, usize)> {
        let (d, ff) = (self.d, self.ff);
        vec![("fc_f", d, d), ("fc_e", d, d), ("wq", d, d), ("wk", d, d), ("wv", d, d), ("wo", d, d), ("n1", 1, d), ("wg", d, ff), ("wu", d, ff), ("wd", ff, d), ("n2", 1, d)]
    }

    pub fn n_params(&self) -> usize {
        self.param_shapes().iter().map(|(_, r, c)| r * c).sum()
    }
}

/// The saved form of a head: a magic word, the configuration, the vocabulary, and every parameter in order, all little endian
pub fn head_save(cfg: &HeadConfig, vocab: &[u32], params: &[Vec<f32>]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"KFHEAD01");
    for v in [cfg.d, cfg.heads, cfg.ff, cfg.window] {
        out.extend_from_slice(&(v as u32).to_le_bytes());
    }
    out.extend_from_slice(&cfg.eps.to_le_bytes());
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

/// The configuration, vocabulary, and parameters of a saved head, checked against the configuration they load into
pub fn head_load(cfg: &HeadConfig, bytes: &[u8]) -> Result<(Vec<u32>, Vec<Vec<f32>>)> {
    use anyhow::{bail, ensure};
    let mut at = 0usize;
    let mut take = |n: usize| -> Result<&[u8]> {
        ensure!(at + n <= bytes.len(), "a saved head of {} bytes ends early", bytes.len());
        at += n;
        Ok(&bytes[at - n..at])
    };
    let u32_of = |b: &[u8]| u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    if take(8)? != b"KFHEAD01" {
        bail!("not a saved head");
    }
    let dims: Vec<usize> = (0..4).map(|_| take(4).map(|b| u32_of(b) as usize)).collect::<Result<_>>()?;
    ensure!(dims == [cfg.d, cfg.heads, cfg.ff, cfg.window], "a saved head of shape {dims:?} for a head of {:?}", [cfg.d, cfg.heads, cfg.ff, cfg.window]);
    let eps = f32::from_le_bytes(take(4)?.try_into().expect("four bytes"));
    ensure!(eps == cfg.eps, "a saved head with eps {eps} for a head with {}", cfg.eps);
    let nv = u32_of(take(4)?) as usize;
    let vocab: Vec<u32> = (0..nv).map(|_| take(4).map(u32_of)).collect::<Result<_>>()?;
    let np = u32_of(take(4)?) as usize;
    let shapes = cfg.param_shapes();
    ensure!(np == shapes.len(), "a saved head of {np} parameter tensors for a head of {}", shapes.len());
    let mut params = Vec::with_capacity(np);
    for (_, r, c) in &shapes {
        let n = u32_of(take(4)?) as usize;
        ensure!(n == r * c, "a saved parameter of {n} values for a tensor of {} by {}", r, c);
        let raw = take(4 * n)?;
        params.push(raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect());
    }
    Ok((vocab, params))
}

/// What a pass feeds, token ids with their positions and sequence slots and, for a graph that starts mid model, the residual rows
#[derive(Clone, Debug, Default)]
pub struct Inputs {
    pub tokens: Vec<u32>,
    pub positions: Vec<u32>,
    /// Sequence slot of each row, every row in slot zero when empty, rows of one slot contiguous and in position order
    pub seqs: Vec<u32>,
    /// Row major residual activations for the hidden input, one row per token
    pub hidden: Vec<f32>,
    pub hidden_cols: usize,
    /// Four rotary position streams per row, the positions themselves in every stream when empty
    pub rope: Vec<[u32; 4]>,
    /// Last key index each row attends to, its own position when empty, later rows for image tokens that see their whole image
    pub attend: Vec<u32>,
    /// The rows of a tree pass, each seeing its ancestors among the pass's rows and every key before them, empty when
    /// every row follows the one before it
    pub tree: Vec<TreeRow>,
}

impl Inputs {
    /// Contiguous runs of rows that belong to one slot, the whole pass as slot zero when no slots were given
    pub fn runs(&self) -> Vec<(usize, Range<usize>)> {
        seq_runs(&self.seqs, self.tokens.len())
    }

    pub fn seq_of(&self, row: usize) -> usize {
        self.seqs.get(row).map_or(0, |s| *s as usize)
    }

    /// The rows of one run as inputs of their own
    pub fn slice(&self, range: Range<usize>) -> Inputs {
        let cols = self.hidden_cols;
        let hidden = if cols > 0 && self.hidden.len() >= range.end * cols { self.hidden[range.start * cols..range.end * cols].to_vec() } else { Vec::new() };
        Inputs {
            tokens: self.tokens[range.clone()].to_vec(),
            positions: self.positions[range.clone()].to_vec(),
            seqs: if self.seqs.is_empty() { Vec::new() } else { self.seqs[range.clone()].to_vec() },
            hidden,
            hidden_cols: cols,
            rope: if self.rope.is_empty() { Vec::new() } else { self.rope[range.clone()].to_vec() },
            attend: if self.attend.is_empty() { Vec::new() } else { self.attend[range.clone()].to_vec() },
            tree: if self.tree.is_empty() { Vec::new() } else { self.tree[range].to_vec() },
        }
    }

    /// The tree row of a row, none in a pass whose rows follow one another
    pub fn tree_of(&self, row: usize) -> Option<TreeRow> {
        self.tree.get(row).copied()
    }

    /// The rotary streams of a row, the position in every stream when none were given
    pub fn rope_of(&self, row: usize) -> [u32; 4] {
        match self.rope.get(row) {
            Some(r) => *r,
            None => {
                let p = self.positions.get(row).copied().unwrap_or(0);
                [p, p, p, p]
            }
        }
    }

    /// The last key index a row attends to
    pub fn attend_of(&self, row: usize) -> u32 {
        self.attend.get(row).copied().unwrap_or_else(|| self.positions.get(row).copied().unwrap_or(0))
    }
}

/// Runs of equal slots over rows in order, one run of slot zero when the slots are empty
pub fn seq_runs(seqs: &[u32], rows: usize) -> Vec<(usize, Range<usize>)> {
    if seqs.is_empty() {
        return vec![(0, 0..rows)];
    }
    let mut out = Vec::new();
    let mut start = 0;
    for r in 1..=rows {
        if r == rows || seqs[r] != seqs[start] {
            out.push((seqs[start] as usize, start..r));
            start = r;
        }
    }
    out
}

/// Executes graphs and owns per sequence state across passes, one slot per sequence
pub trait Backend: Send {
    fn name(&self) -> &'static str;
    fn supports(&self, op: &Op) -> bool;
    /// Places weights, state, and expert cache slots per layer as the plan says, with the plan's sequence slots
    fn apply_plan(&mut self, plan: &Plan) -> Result<()>;
    /// Runs the graph and returns the output tensor row major as f32
    fn run(&mut self, g: &Graph, inputs: &Inputs, weights: &dyn WeightSource) -> Result<Vec<f32>>;
    /// One full row of the tensor the last pass's final top k read, valid until the next pass
    fn dense_row(&mut self, row: usize) -> Result<Vec<f32>>;
    /// The residual rows before the final norm the last pass kept, one per row of logits, valid until the next pass
    fn hidden_rows(&mut self) -> Result<Vec<f32>>;
    /// Starts the draft head over the model's embedding and head weights, learning from the rows every pass scores,
    /// drafting over a vocabulary of the draft_vocab most frequent tokens, batch chunks a training step
    fn drafter_start(&mut self, cfg: HeadConfig, embed: Wid, head: Wid, n_vocab: usize, draft_vocab: usize, batch: usize, weights: &dyn WeightSource) -> Result<()>;
    /// One training step of the head on the rows scored so far, what it reports, none while too few rows stand
    fn drafter_step(&mut self) -> Result<Option<TrainStep>>;
    /// Rows the passes have handed the head so far, counting every row that entered its ring
    fn drafter_rows(&self) -> u64;
    /// Keeps the hidden row of the last pass at row for a slot's coming draft, the row's position given
    fn drafter_hold(&mut self, seq: usize, row: usize, pos: u32) -> Result<()>;
    /// Brings a slot's head state up to the rows the passes left for it
    fn drafter_sync(&mut self, seq: usize) -> Result<()>;
    /// A tree rooted at own, the token chosen after the held row of a slot, of at most budget rows, the root alone
    /// while the head cannot draft for the slot
    fn drafter_draft(&mut self, seq: usize, own: u32, budget: usize) -> Result<Tree>;
    /// Recounts the draft vocabulary from the tokens the passes scored
    fn drafter_refresh_vocab(&mut self) -> Result<()>;
    /// The head's parameters and vocabulary as bytes another process of the same model loads
    fn drafter_save(&mut self) -> Result<Vec<u8>>;
    /// Replaces the head's parameters and vocabulary with saved ones
    fn drafter_load(&mut self, bytes: &[u8]) -> Result<()>;
    /// Drops the state of one slot past its first n positions, an error when a recurrent state cannot roll back
    fn truncate(&mut self, seq: usize, n: usize) -> Result<()>;
    /// Moves the cache rows base plus keep[i] of one slot to base plus i for every layer and drops the rows past
    /// them, so the accepted path of a tree pass stands where its rows would have stood fed one after another
    fn compact(&mut self, seq: usize, base: usize, keep: &[usize]) -> Result<()>;
    /// Drops every position of every slot
    fn reset(&mut self);
    /// Drops every position of one slot
    fn reset_seq(&mut self, seq: usize);
    /// Every byte of one slot's state, what another process of the same plan loads to continue the sequence
    fn save_state(&mut self, seq: usize) -> Result<Vec<u8>>;
    /// Replaces one slot's state with one saved under the same plan
    fn load_state(&mut self, seq: usize, bytes: &[u8]) -> Result<()>;
    /// Layer, slots, hits, and misses of every expert cache the backend holds
    fn expert_cache_stats(&self) -> Vec<(usize, usize, u64, u64)> {
        Vec::new()
    }
    /// Expert selections counted per layer since the plan was applied, a count per expert
    fn routing_profile(&self) -> Vec<(usize, Vec<u64>)> {
        Vec::new()
    }
    /// Counts per expert per layer that fill each expert cache with its most used experts when the cache is made
    fn set_routing_profile(&mut self, _profile: Vec<(usize, Vec<u64>)>) {}
}

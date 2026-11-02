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

/// What one training step of the draft block reports: the mean loss before the update and the share of first step
/// rows whose actual next token stood among the children the block would have drafted
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

    /// The parent of every tree row, the nearest earlier row its mask names, minus one for the root, a chain of
    /// parents when the pass holds no tree
    pub fn tree_parents(&self) -> Vec<i32> {
        let n = self.tokens.len();
        if self.tree.is_empty() {
            return (0..n).map(|i| i as i32 - 1).collect();
        }
        (0..n).map(|i| (0..i).rev().find(|j| self.tree[i].sees((n - 1) as u32, *j as u32)).map_or(-1, |j| j as i32)).collect()
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
    /// The residual rows after every tap layer the last pass kept, one row per kept hidden row with the taps side
    /// by side in the graph's order, valid until the next pass
    fn tap_rows(&mut self) -> Result<Vec<f32>>;
    /// Starts collecting the layer statistics the tap selection reads from every pass over all n_layers tap rows
    fn draft_calibrate(&mut self, n_layers: usize) -> Result<()>;
    /// Rows the tap selection has kept so far
    fn draft_calibrated(&self) -> u64;
    /// The m layers the collected statistics select, by concatenated matrix entropy, the expansion pool by CKA
    /// transition when cka is set, and ends the collection
    fn draft_select_taps(&mut self, m: usize, cka: bool) -> Result<Vec<usize>>;
    /// Starts the draft block over the model's embedding, head, and final norm weights with the tap layers the
    /// passes keep, learning from the rows every pass scores, drafting over a vocabulary of the draft_vocab most
    /// frequent tokens, batch chunks a training step, the end token left out of the rejection costs
    #[allow(clippy::too_many_arguments)]
    fn draft_start(&mut self, cfg: super::DraftConfig, tap_layers: &[usize], embed: Wid, head: Wid, norm: Wid, n_vocab: usize, eos: Option<u32>, draft_vocab: usize, batch: usize, weights: &dyn WeightSource) -> Result<()>;
    /// One training step of the block on the rows scored so far, what it reports, none while too few rows stand
    fn draft_step(&mut self) -> Result<Option<TrainStep>>;
    /// Rows the passes have handed the block so far, counting every row that entered its ring
    fn draft_rows(&self) -> u64;
    /// Writes the step zero rows of the last pass's accepted path into the slot's history, the rows kept in order
    /// and the root token picked at the last of them, and drafts a tree of up to rows below the root from the
    /// block's own pass, the root alone when rows is one
    fn draft_tree(&mut self, seq: usize, path: &[usize], root: u32, rows: usize) -> Result<Tree>;
    /// Recounts the draft vocabulary from the tokens the passes scored
    fn draft_refresh_vocab(&mut self) -> Result<()>;
    /// The block's parameters and vocabulary as bytes another process of the same model loads
    fn draft_save(&mut self) -> Result<Vec<u8>>;
    /// Replaces the block's parameters and vocabulary with saved ones
    fn draft_load(&mut self, bytes: &[u8]) -> Result<()>;
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

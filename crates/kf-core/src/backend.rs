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
            attend: if self.attend.is_empty() { Vec::new() } else { self.attend[range].to_vec() },
        }
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
    /// Drops the state of one slot past its first n positions, an error when a recurrent state cannot roll back
    fn truncate(&mut self, seq: usize, n: usize) -> Result<()>;
    /// Drops every position of every slot
    fn reset(&mut self);
    /// Drops every position of one slot
    fn reset_seq(&mut self, seq: usize);
    /// Every byte of one slot's state, what another process of the same plan loads to continue the sequence
    fn save_state(&mut self, seq: usize) -> Result<Vec<u8>>;
    /// Replaces one slot's state with one saved under the same plan
    fn load_state(&mut self, seq: usize, bytes: &[u8]) -> Result<()>;
    /// Layer, slots, hits, misses, experts prefetched, and prefetched experts then chosen of every expert cache the backend holds
    fn expert_cache_stats(&self) -> Vec<(usize, usize, u64, u64, u64, u64)> {
        Vec::new()
    }
    /// Expert selections counted per layer since the plan was applied, a count per expert
    fn routing_profile(&self) -> Vec<(usize, Vec<u64>)> {
        Vec::new()
    }
    /// Counts per expert per layer that fill each expert cache with its most used experts when the cache is made
    fn set_routing_profile(&mut self, _profile: Vec<(usize, Vec<u64>)>) {}
}

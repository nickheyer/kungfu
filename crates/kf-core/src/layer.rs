use crate::graph::{GraphBuilder, Tid, Wid};
use anyhow::Result;
use std::cell::RefCell;
use std::rc::Rc;

/// Graph build context for one forward pass
#[derive(Clone, Debug)]
pub struct Ctx {
    pub layer: usize,
    pub tokens: Tid,
    pub positions: Tid,
    pub n_tokens: usize,
    /// Scaled token embedding before any block
    pub embed: Tid,
    /// Residual stream entering the current block
    pub block_input: Tid,
    /// Residual stream entering the current sublayer, before its norm
    pub residual_input: Tid,
    /// Tensors one block hands to later blocks, lagged mixes and banked checkpoints
    pub handoff: Rc<RefCell<Vec<Tid>>>,
}

impl Ctx {
    pub fn new(tokens: Tid, positions: Tid, n_tokens: usize) -> Self {
        Self { layer: 0, tokens, positions, n_tokens, embed: tokens, block_input: tokens, residual_input: tokens, handoff: Rc::new(RefCell::new(Vec::new())) }
    }
}

/// Per sequence state a layer keeps between passes, the planner reads these and never the layer type
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StateNeed {
    KvCache { n_head_kv: usize, head_dim_k: usize, head_dim_v: usize, window: Option<usize> },
    Recurrent { conv_state: usize, ssm_state: usize },
    Latent { rank: usize, rope_dim: usize, window: Option<usize> },
    /// Normed activations of the last token per shift slot, each of width channels
    TokenShift { n: usize, width: usize },
    /// Indexer key rows per token
    IndexerCache { dim: usize },
    /// Compressed key rows, one per ratio tokens
    Compressed { dim: usize, ratio: usize },
    /// Last n token ids per sequence
    TokenHistory { n: usize },
}

/// Anything that maps an activation to an activation by emitting graph nodes
pub trait Layer: Send + Sync {
    fn forward(&self, g: &mut GraphBuilder, x: Tid, ctx: &Ctx) -> Result<Tid>;
    fn weights(&self) -> Vec<Wid>;
    fn states(&self) -> Vec<StateNeed> {
        Vec::new()
    }
}

impl<T: Layer + ?Sized> Layer for Box<T> {
    fn forward(&self, g: &mut GraphBuilder, x: Tid, ctx: &Ctx) -> Result<Tid> {
        (**self).forward(g, x, ctx)
    }
    fn weights(&self) -> Vec<Wid> {
        (**self).weights()
    }
    fn states(&self) -> Vec<StateNeed> {
        (**self).states()
    }
}

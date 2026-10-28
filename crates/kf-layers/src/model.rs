use crate::basic::{Embedding, Linear};
use anyhow::{Result, bail};
use kf_core::{Ctx, GraphBuilder, Layer, Op, StateNeed, Tid, Wid};

/// The parts applied at one layer index, in order
pub struct Block {
    pub layer: usize,
    pub parts: Vec<Box<dyn Layer>>,
}

impl Block {
    pub fn new(layer: usize) -> Self {
        Self { layer, parts: Vec::new() }
    }
    pub fn then(mut self, part: impl Layer + 'static) -> Self {
        self.parts.push(Box::new(part));
        self
    }
    pub fn states(&self) -> Vec<StateNeed> {
        self.parts.iter().flat_map(|p| p.states()).collect()
    }
}

/// The part of the forward pass one process runs, blocks from a layer to one past the last, the embedding before and the head after when asked
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Segment {
    pub from: usize,
    pub to: usize,
    /// Start from the token embedding, else from the hidden input
    pub embed: bool,
    /// The hidden input stands in for the token embedding, scaled and prepared like one, image rows at the first layer
    pub inject: bool,
    /// Apply the final norm and head to the last n rows
    pub n_out: Option<usize>,
}

impl Segment {
    pub fn whole(n_layers: usize, n_out: usize) -> Self {
        Self { from: 0, to: n_layers, embed: true, inject: false, n_out: Some(n_out) }
    }
}

pub struct Model {
    pub embed: Embedding,
    /// Applied to the embedding before the blocks, such as an embedding norm
    pub pre: Vec<Box<dyn Layer>>,
    /// Residual streams the blocks run on, the embedding is repeated this many times
    pub streams: usize,
    pub blocks: Vec<Block>,
    /// Collapses a multi stream residual before the final norm, hyper connected models only
    pub pre_head: Option<Box<dyn Layer>>,
    pub final_norm: Box<dyn Layer>,
    pub head: Linear,
    pub logit_scale: f32,
    pub final_softcap: f32,
    /// Token ids whose logits are forced to minus infinity
    pub logit_mask: Option<(usize, usize)>,
    /// Heads beyond the main one, such as next token prediction layers
    pub extras: Vec<Box<dyn Layer>>,
    /// Whether any block hands tensors to later blocks, probed once
    pub hands_off: std::sync::OnceLock<bool>,
}

impl Model {
    pub fn plain(embed: Embedding, blocks: Vec<Block>, final_norm: impl Layer + 'static, head: Linear) -> Self {
        Self { embed, pre: Vec::new(), streams: 1, blocks, pre_head: None, final_norm: Box::new(final_norm), head, logit_scale: 1.0, final_softcap: 0.0, logit_mask: None, extras: Vec::new(), hands_off: std::sync::OnceLock::new() }
    }

    /// Emits the whole forward pass and returns logits for the last n_out tokens
    pub fn logits(&self, g: &mut GraphBuilder, n_tokens: usize, n_out: usize) -> Result<Tid> {
        self.segment(g, n_tokens, Segment::whole(self.n_layers(), n_out))
    }

    /// Whether any block hands tensors to later blocks, which no cut between blocks can carry
    pub fn hands_off(&self) -> bool {
        *self.hands_off.get_or_init(|| {
            let mut g = GraphBuilder::new();
            let tokens = g.input("tokens");
            let positions = g.input("positions");
            let mut ctx = Ctx::new(tokens, positions, 1);
            let Ok(mut x) = self.embed.forward(&mut g, tokens, &ctx) else { return true };
            ctx.embed = x;
            if self.streams > 1 {
                x = g.push(Op::Tile { n: self.streams }, &[x]);
            }
            for block in &self.blocks {
                ctx.layer = block.layer;
                ctx.block_input = x;
                ctx.residual_input = x;
                for part in &block.parts {
                    match part.forward(&mut g, x, &ctx) {
                        Ok(y) => x = y,
                        Err(_) => return true,
                    }
                }
                if !ctx.handoff.borrow().is_empty() {
                    return true;
                }
            }
            false
        })
    }

    /// Emits one segment, scaled token embeddings or unscaled injected rows through the range's blocks, then the head when asked
    pub fn segment(&self, g: &mut GraphBuilder, n_tokens: usize, seg: Segment) -> Result<Tid> {
        let n_layers = self.n_layers();
        if seg.from > seg.to || seg.to > n_layers {
            bail!("segment {}..{} of a {n_layers} layer model", seg.from, seg.to);
        }
        if (seg.from > 0 || seg.to < n_layers) && self.hands_off() {
            bail!("the model hands tensors between blocks, so no cut between layers can carry its state");
        }
        let tokens = g.input("tokens");
        let positions = g.input("positions");
        let mut ctx = Ctx::new(tokens, positions, n_tokens);
        let mut x = if seg.embed || seg.inject {
            let mut x = if seg.inject {
                if seg.from != 0 {
                    bail!("injected rows enter at layer 0, not {}", seg.from);
                }
                let h = g.input("hidden");
                ctx.embed = h;
                h
            } else {
                let e = self.embed.forward(g, tokens, &ctx)?;
                ctx.embed = e;
                e
            };
            for p in &self.pre {
                x = p.forward(g, x, &ctx)?;
            }
            if self.streams > 1 {
                x = g.push(Op::Tile { n: self.streams }, &[x]);
            }
            x
        } else {
            let e = self.embed.forward(g, tokens, &ctx)?;
            ctx.embed = e;
            g.input("hidden")
        };
        for block in self.blocks.iter().filter(|b| b.layer >= seg.from && b.layer < seg.to) {
            ctx.layer = block.layer;
            ctx.block_input = x;
            ctx.residual_input = x;
            g.layer = Some(block.layer);
            for part in &block.parts {
                x = part.forward(g, x, &ctx)?;
            }
        }
        g.layer = None;
        let Some(n_out) = seg.n_out else { return Ok(x) };
        if n_out < n_tokens {
            x = g.push(Op::LastRows { n: n_out }, &[x]);
        }
        if let Some(p) = &self.pre_head {
            x = p.forward(g, x, &ctx)?;
        }
        let x = self.final_norm.forward(g, x, &ctx)?;
        let mut l = self.head.apply(g, x);
        l = g.scale(l, self.logit_scale);
        if self.final_softcap != 0.0 {
            l = g.push(Op::Softcap(self.final_softcap), &[l]);
        }
        if let Some((start, len)) = self.logit_mask {
            l = g.push(Op::MaskCols { start, len, value: f32::NEG_INFINITY }, &[l]);
        }
        Ok(l)
    }

    pub fn n_layers(&self) -> usize {
        self.blocks.iter().map(|b| b.layer + 1).max().unwrap_or(0)
    }

    pub fn weights(&self) -> Vec<Wid> {
        let mut w = self.embed.weights();
        for p in &self.pre {
            w.extend(p.weights());
        }
        for b in &self.blocks {
            for p in &b.parts {
                w.extend(p.weights());
            }
        }
        w.extend(self.pre_head.iter().flat_map(|p| p.weights()));
        w.extend(self.final_norm.weights());
        w.extend(self.head.weights());
        for e in &self.extras {
            w.extend(e.weights());
        }
        w.sort();
        w.dedup();
        w
    }

    /// State every block needs, keyed by layer index
    pub fn states(&self) -> Vec<(usize, StateNeed)> {
        self.blocks.iter().flat_map(|b| b.states().into_iter().map(move |s| (b.layer, s))).collect()
    }
}

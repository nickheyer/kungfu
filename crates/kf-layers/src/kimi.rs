use anyhow::Result;
use kf_core::{Ctx, GraphBuilder, Layer, StateNeed, Tid, Wid};

/// Kimi K3 block with cross layer residual attention over banked checkpoints carried in the stream
pub struct ResMixBlock {
    pub n_embd: usize,
    pub block_size: usize,
    pub layer: usize,
    /// Checkpoints banked before this layer, the stream carries that many extra copies
    pub n_ckpt: usize,
    pub eps: f32,
    pub attn_score: Wid,
    pub ffn_score: Wid,
    pub attn_norm: Box<dyn Layer>,
    pub attn: Box<dyn Layer>,
    pub ffn_norm: Box<dyn Layer>,
    pub ffn: Box<dyn Layer>,
}

impl Layer for ResMixBlock {
    fn forward(&self, g: &mut GraphBuilder, x: Tid, ctx: &Ctx) -> Result<Tid> {
        res_mix_block_graph(g, self, x, ctx)
    }
    fn weights(&self) -> Vec<Wid> {
        let mut w = vec![self.attn_score, self.ffn_score];
        w.extend(self.attn_norm.weights());
        w.extend(self.attn.weights());
        w.extend(self.ffn_norm.weights());
        w.extend(self.ffn.weights());
        w
    }
    fn states(&self) -> Vec<StateNeed> {
        let mut s = self.attn.states();
        s.extend(self.ffn.states());
        s
    }
}

/// Final cross layer residual mix before the output norm
pub struct ResMixHead {
    pub n_embd: usize,
    pub n_ckpt: usize,
    pub eps: f32,
    pub score: Wid,
}

impl Layer for ResMixHead {
    fn forward(&self, g: &mut GraphBuilder, x: Tid, ctx: &Ctx) -> Result<Tid> {
        res_mix_head_graph(g, self, x, ctx)
    }
    fn weights(&self) -> Vec<Wid> {
        vec![self.score]
    }
}

use crate::util::{join, mul_weight, rms, unary};
use kf_core::Op;

/// Softmax over the normed dot of each checkpoint and the current stream with the score vector, then the raw weighted sum
fn res_mix(g: &mut GraphBuilder, stack: &[Tid], cur: Tid, score: Wid, eps: f32) -> Tid {
    let mut scores = Vec::with_capacity(stack.len() + 1);
    for c in stack.iter().chain(std::iter::once(&cur)) {
        let n = rms(g, *c, None, eps);
        let n = mul_weight(g, n, score);
        scores.push(g.push(Op::SumRows, &[n]));
    }
    let s = join(g, &scores);
    let p = unary(g, Op::Softmax, s);
    let mut all: Vec<Tid> = stack.to_vec();
    all.push(cur);
    let streams = join(g, &all);
    g.push(Op::HcMix { n_streams: all.len() }, &[streams, p])
}

/// Kimi K3 block, both sublayers read a cross layer mix of the banked checkpoints and the running stream
pub fn res_mix_block_graph(g: &mut GraphBuilder, b: &ResMixBlock, x: Tid, ctx: &Ctx) -> Result<Tid> {
    let mut c = ctx.clone();
    let banked = b.block_size > 0 && b.layer.is_multiple_of(b.block_size);
    let stack: Vec<Tid> = ctx.handoff.borrow().clone();
    let h = if b.block_size > 0 { res_mix(g, &stack, x, b.attn_score, b.eps) } else { x };
    if banked {
        ctx.handoff.borrow_mut().push(x);
    }
    c.residual_input = h;
    let h = b.attn_norm.forward(g, h, &c)?;
    let a = b.attn.forward(g, h, &c)?;
    let p = if banked { a } else { g.add(x, a) };
    let stack: Vec<Tid> = ctx.handoff.borrow().clone();
    let h = if b.block_size > 0 { res_mix(g, &stack, p, b.ffn_score, b.eps) } else { p };
    c.residual_input = h;
    let h = b.ffn_norm.forward(g, h, &c)?;
    let f = b.ffn.forward(g, h, &c)?;
    let _ = b.n_ckpt + b.n_embd;
    Ok(g.add(p, f))
}

pub fn res_mix_head_graph(g: &mut GraphBuilder, h: &ResMixHead, x: Tid, ctx: &Ctx) -> Result<Tid> {
    let stack: Vec<Tid> = ctx.handoff.borrow().clone();
    let _ = h.n_ckpt + h.n_embd;
    Ok(res_mix(g, &stack, x, h.score, h.eps))
}

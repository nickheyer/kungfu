use crate::basic::{Linear, RmsNorm};
use anyhow::Result;
use kf_core::{Ctx, GraphBuilder, Layer, StateNeed, Tid, Wid};

/// MiniMax lightning linear attention with a per head decay and a sigmoid output gate
pub struct LightningAttn {
    pub n_head: usize,
    pub head_dim: usize,
    pub qkv: Linear,
    pub norm: RmsNorm,
    pub gate: Linear,
    pub out: Linear,
    pub decay: Vec<f32>,
}

impl Layer for LightningAttn {
    fn forward(&self, g: &mut GraphBuilder, x: Tid, ctx: &Ctx) -> Result<Tid> {
        lightning_attention_graph(g, self, x, ctx)
    }
    fn weights(&self) -> Vec<Wid> {
        let mut w = self.qkv.weights();
        w.extend(self.norm.weights());
        w.extend(self.gate.weights());
        w.extend(self.out.weights());
        w
    }
    fn states(&self) -> Vec<StateNeed> {
        vec![StateNeed::Recurrent { conv_state: 0, ssm_state: self.n_head * self.head_dim * self.head_dim }]
    }
}

use crate::util::unary;
use kf_core::Op;
use std::sync::Arc;

/// Lightning attention, silu q k v per head, the decayed recurrence, a norm over the heads and a sigmoid gate
pub fn lightning_attention_graph(g: &mut GraphBuilder, l: &LightningAttn, x: Tid, ctx: &Ctx) -> Result<Tid> {
    let t = ctx.n_tokens;
    let (h, d) = (l.n_head, l.head_dim);
    let qkv = l.qkv.apply(g, x);
    let qkv = unary(g, Op::Silu, qkv);
    let per = g.reshape(qkv, t * h, 3 * d);
    let q = g.cols(per, 0, d);
    let k = g.cols(per, d, d);
    let v = g.cols(per, 2 * d, d);
    let q = g.reshape(q, t, h * d);
    let k = g.reshape(k, t, h * d);
    let v = g.reshape(v, t, h * d);
    let o = g.push(Op::LightningAttn { layer: ctx.layer, n_head: h, head_dim: d, decay: Arc::new(l.decay.clone()) }, &[q, k, v]);
    let o = l.norm.apply(g, o);
    let gate = l.gate.apply(g, x);
    let gate = unary(g, Op::Sigmoid, gate);
    let y = g.mul(o, gate);
    Ok(l.out.apply(g, y))
}

use crate::attention::Attention;
use crate::basic::{Linear, RmsNorm};
use anyhow::Result;
use kf_core::{Ctx, GraphBuilder, Layer, StateNeed, Tid, Wid};

/// A projection with a per token switched low rank delta picked by an adapter slot
pub struct SwitchedLinear {
    pub w: Linear,
    pub a: Wid,
    pub b: Wid,
}

impl SwitchedLinear {
    pub fn weights(&self) -> Vec<Wid> {
        let mut w = self.w.weights();
        w.extend([self.a, self.b]);
        w
    }
}

/// Granite switch block, an in graph adapter router then switched LoRA on every projection
pub struct SwitchedBlock {
    pub n_adapters: usize,
    pub router_gain: f32,
    pub router_layer: usize,
    pub activate: Vec<u32>,
    pub head_dim: usize,
    pub attn_norm: RmsNorm,
    pub attn: Attention,
    pub q: SwitchedLinear,
    pub k: SwitchedLinear,
    pub v: SwitchedLinear,
    pub o: SwitchedLinear,
    pub ffn_norm: RmsNorm,
    pub gate: SwitchedLinear,
    pub up: SwitchedLinear,
    pub down: SwitchedLinear,
    pub residual_scale: f32,
}

impl Layer for SwitchedBlock {
    fn forward(&self, g: &mut GraphBuilder, x: Tid, ctx: &Ctx) -> Result<Tid> {
        switched_block_graph(g, self, x, ctx)
    }
    fn weights(&self) -> Vec<Wid> {
        let mut w = self.attn_norm.weights();
        w.extend(self.attn.weights());
        w.extend(self.q.weights());
        w.extend(self.k.weights());
        w.extend(self.v.weights());
        w.extend(self.o.weights());
        w.extend(self.ffn_norm.weights());
        w.extend(self.gate.weights());
        w.extend(self.up.weights());
        w.extend(self.down.weights());
        w
    }
    fn states(&self) -> Vec<StateNeed> {
        let mut s = self.attn.states();
        s.push(StateNeed::KvCache { n_head_kv: 1, head_dim_k: self.head_dim, head_dim_v: self.head_dim, window: None });
        s
    }
}

use crate::util::{fill, unary};
use kf_core::Op;
use std::sync::Arc;

/// Adapter slot per token from an in graph router, then the block with switched deltas on every projection
pub fn switched_block_graph(g: &mut GraphBuilder, b: &SwitchedBlock, x: Tid, ctx: &Ctx) -> Result<Tid> {
    let t = ctx.n_tokens;
    let gain = b.router_gain;
    let ksig = Arc::new(b.activate.iter().map(|tok| (*tok, gain)).collect::<Vec<_>>());
    let vval = Arc::new(b.activate.iter().enumerate().map(|(i, tok)| (*tok, (i + 1) as f32)).collect::<Vec<_>>());
    let kq = g.push(Op::MapTokens { table: ksig, default: Some(-gain) }, &[ctx.tokens]);
    let vq = g.push(Op::MapTokens { table: vval, default: Some(0.0) }, &[ctx.tokens]);
    let qq = fill(g, t, 1, 1.0);
    let pad = |g: &mut GraphBuilder, v: Tid| g.push(Op::PadCols { to: b.head_dim }, &[v]);
    let (qq, kq, vq) = (pad(g, qq), pad(g, kq), pad(g, vq));
    let route = Op::Attention {
        layer: b.router_layer,
        n_head: 1,
        n_head_kv: 1,
        head_dim: b.head_dim,
        head_dim_v: b.head_dim,
        scale: 1.0,
        window: None,
        softcap: 0.0,
        sinks: None,
        alibi_bias: 0.0,
        kv_source: None,
        top_k: None,
    };
    let out = g.push(route, &[qq, kq, vq]);
    let slot = g.cols(out, 0, 1);
    let slot = g.push(Op::Clamp { min: 0.0, max: b.n_adapters as f32 }, &[slot]);
    let slot = unary(g, Op::Round, slot);
    let delta = |g: &mut GraphBuilder, sw: &SwitchedLinear, inp: Tid| g.push(Op::LoraSwitched { a: sw.a, b: sw.b }, &[inp, slot]);
    let mut c = ctx.clone();
    c.residual_input = x;
    let h = b.attn_norm.apply(g, x);
    let (q, k, v) = b.attn.project(g, h);
    let dq = delta(g, &b.q, h);
    let dk = delta(g, &b.k, h);
    let dv = delta(g, &b.v, h);
    let q = g.add(q, dq);
    let k = g.add(k, dk);
    let v = g.add(v, dv);
    let q = b.attn.rope_q(g, q, &c);
    let k = b.attn.rope_k(g, k, &c);
    let o = g.push(b.attn.attention_op(ctx.layer, None), &[q, k, v]);
    let a = b.attn.out.apply(g, o);
    let da = delta(g, &b.o, o);
    let a = g.add(a, da);
    let a = g.scale(a, if b.residual_scale != 0.0 { b.residual_scale } else { 1.0 });
    let x = g.add(x, a);
    let h = b.ffn_norm.apply(g, x);
    let gt = b.gate.w.apply(g, h);
    let dg = delta(g, &b.gate, h);
    let gt = g.add(gt, dg);
    let up = b.up.w.apply(g, h);
    let du = delta(g, &b.up, h);
    let up = g.add(up, du);
    let m = g.push(Op::GatedAct { act: kf_core::Act::Silu, limit: None }, &[gt, up]);
    let f = b.down.w.apply(g, m);
    let df = delta(g, &b.down, m);
    let f = g.add(f, df);
    let f = g.scale(f, if b.residual_scale != 0.0 { b.residual_scale } else { 1.0 });
    Ok(g.add(x, f))
}

use crate::attention::QkvProj;
use crate::basic::Linear;
use anyhow::Result;
use kf_core::{Ctx, GraphBuilder, Layer, StateNeed, Tid, Wid};

pub enum BetaAlpha {
    Fused(Wid),
    Separate { beta: Wid, alpha: Wid },
}

/// How the delta net projections are laid out in the file
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeltaLayout {
    /// ssm_in groups q k v z per key head and ssm_ba groups beta then alpha per key head
    LegacyGrouped,
    /// attn_qkv holds q k v in order and attn_gate holds z, beta and alpha separate or fused side by side
    Split,
}

/// Which key head a value head reads when there are more value heads
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadMap {
    Grouped,
    Tiled,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutGate {
    Silu,
    Sigmoid,
}

/// Gated delta net linear attention, the recurrent mixer of the Qwen3 Next family
pub struct DeltaNet {
    pub n_head_k: usize,
    pub n_head_v: usize,
    pub head_dim: usize,
    pub conv_kernel: usize,
    pub layout: DeltaLayout,
    pub head_map: HeadMap,
    pub out_gate: OutGate,
    /// q k v projection, with z appended when gate is absent
    pub in_proj: Linear,
    pub gate: Option<Linear>,
    pub conv1d: Wid,
    pub dt_bias: Wid,
    /// Holds minus exp of A_log as the converter writes it
    pub a: Wid,
    pub beta_alpha: BetaAlpha,
    pub norm: Wid,
    pub norm_eps: f32,
    pub l2_eps: f32,
    pub out: Linear,
}

impl Layer for DeltaNet {
    fn forward(&self, g: &mut GraphBuilder, x: Tid, ctx: &Ctx) -> Result<Tid> {
        gated_delta_net_graph(g, self, x, ctx)
    }
    fn weights(&self) -> Vec<Wid> {
        let mut w = self.in_proj.weights();
        w.extend(self.gate.iter().flat_map(Layer::weights));
        w.extend([self.conv1d, self.dt_bias, self.a, self.norm]);
        match self.beta_alpha {
            BetaAlpha::Fused(b) => w.push(b),
            BetaAlpha::Separate { beta, alpha } => w.extend([beta, alpha]),
        }
        w.extend(self.out.weights());
        w
    }
    fn states(&self) -> Vec<StateNeed> {
        let conv_width = self.n_head_k * self.head_dim * 2 + self.n_head_v * self.head_dim;
        vec![StateNeed::Recurrent {
            conv_state: (self.conv_kernel - 1) * conv_width,
            ssm_state: self.n_head_v * self.head_dim * self.head_dim,
        }]
    }
}

/// Kimi delta attention, separate q k v with short convs and a low rank or full rank decay gate
pub struct Kda {
    pub n_head: usize,
    pub head_dim: usize,
    pub conv_kernel: usize,
    /// Minus infinity selects the softplus gate, a finite bound selects the sigmoid gate
    pub gate_lower_bound: f32,
    /// a stores exp of A_log instead of minus exp
    pub a_is_exp: bool,
    pub qkv: QkvProj,
    pub conv_q: Wid,
    pub conv_k: Wid,
    pub conv_v: Wid,
    pub f_a: Wid,
    pub f_b: Option<Wid>,
    pub g_a: Wid,
    pub g_b: Option<Wid>,
    pub beta: Wid,
    pub dt_bias: Wid,
    pub a: Wid,
    pub norm: Wid,
    pub norm_eps: f32,
    pub l2_eps: f32,
    pub out: Linear,
}

impl Layer for Kda {
    fn forward(&self, g: &mut GraphBuilder, x: Tid, ctx: &Ctx) -> Result<Tid> {
        kimi_delta_attention_graph(g, self, x, ctx)
    }
    fn weights(&self) -> Vec<Wid> {
        let mut w = self.qkv.weights();
        w.extend([self.conv_q, self.conv_k, self.conv_v, self.f_a, self.g_a, self.beta, self.dt_bias, self.a, self.norm]);
        w.extend(self.f_b);
        w.extend(self.g_b);
        w.extend(self.out.weights());
        w
    }
    fn states(&self) -> Vec<StateNeed> {
        vec![StateNeed::Recurrent {
            conv_state: (self.conv_kernel - 1) * self.n_head * self.head_dim * 3,
            ssm_state: self.n_head * self.head_dim * self.head_dim,
        }]
    }
}

use crate::util::{add_weight, fill, mul_head_cols, mul_per_head, mul_weight, per_head_l2, per_head_rms, repeat_heads, row_weight, unary};
use anyhow::bail;
use kf_core::Op;

/// Gated delta net, projections laid out per the file, short conv, l2 normed q k, the recurrence and the gated norm
pub fn gated_delta_net_graph(g: &mut GraphBuilder, d: &DeltaNet, x: Tid, ctx: &Ctx) -> Result<Tid> {
    let t = ctx.n_tokens;
    let (hk, hv, s) = (d.n_head_k, d.n_head_v, d.head_dim);
    if hk == 0 || !hv.is_multiple_of(hk) {
        bail!("delta net with {hv} value heads over {hk} key heads");
    }
    let r = hv / hk;
    let c = 2 * hk * s + hv * s;
    let (qkv, z) = match d.layout {
        DeltaLayout::LegacyGrouped => {
            let y = d.in_proj.apply(g, x);
            let per = g.reshape(y, t * hk, 2 * s + 2 * r * s);
            let q = g.cols(per, 0, s);
            let k = g.cols(per, s, s);
            let v = g.cols(per, 2 * s, r * s);
            let zz = g.cols(per, 2 * s + r * s, r * s);
            let q = g.reshape(q, t, hk * s);
            let k = g.reshape(k, t, hk * s);
            let v = g.reshape(v, t, hv * s);
            let z = g.reshape(zz, t, hv * s);
            let qk = g.concat(q, k);
            (g.concat(qk, v), z)
        }
        DeltaLayout::Split => {
            let y = d.in_proj.apply(g, x);
            match &d.gate {
                Some(gate) => (y, gate.apply(g, x)),
                None => (g.cols(y, 0, c), g.cols(y, c, hv * s)),
            }
        }
    };
    let (b, a) = match d.beta_alpha {
        BetaAlpha::Fused(w) => {
            let ba = g.push(Op::MatMul { weight: w, bias: None, scale: None }, &[x]);
            match d.layout {
                DeltaLayout::LegacyGrouped => {
                    let per = g.reshape(ba, t * hk, 2 * r);
                    let b = g.cols(per, 0, r);
                    let a = g.cols(per, r, r);
                    (g.reshape(b, t, hv), g.reshape(a, t, hv))
                }
                DeltaLayout::Split => (g.cols(ba, 0, hv), g.cols(ba, hv, hv)),
            }
        }
        BetaAlpha::Separate { beta, alpha } => (
            g.push(Op::MatMul { weight: beta, bias: None, scale: None }, &[x]),
            g.push(Op::MatMul { weight: alpha, bias: None, scale: None }, &[x]),
        ),
    };
    let conv = g.push(Op::CausalConv { layer: ctx.layer, slot: 0, kernel: d.conv_kernel, dilation: 1, weight: d.conv1d, bias: None }, &[qkv]);
    let conv = unary(g, Op::Silu, conv);
    let q = g.cols(conv, 0, hk * s);
    let k = g.cols(conv, hk * s, hk * s);
    let v = g.cols(conv, 2 * hk * s, hv * s);
    let q = per_head_l2(g, q, t, hk, s, d.l2_eps);
    let k = per_head_l2(g, k, t, hk, s, d.l2_eps);
    let beta = unary(g, Op::Sigmoid, b);
    let gate = add_weight(g, a, d.dt_bias);
    let gate = unary(g, Op::Softplus { beta: 1.0 }, gate);
    let gate = mul_weight(g, gate, d.a);
    let (q, k, n_head_k) = match d.head_map {
        HeadMap::Grouped => (repeat_heads(g, q, t, hk, s, r), repeat_heads(g, k, t, hk, s, r), hv),
        HeadMap::Tiled => (q, k, hk),
    };
    let o = g.push(Op::GatedDeltaNet { layer: ctx.layer, n_head_k, n_head_v: hv, head_dim: s, gate_per_channel: false }, &[q, k, v, gate, beta]);
    let y = per_head_rms(g, o, t, hv, s, Some(d.norm), d.norm_eps);
    let act = match d.out_gate {
        OutGate::Silu => unary(g, Op::Silu, z),
        OutGate::Sigmoid => unary(g, Op::Sigmoid, z),
    };
    let y = g.mul(y, act);
    Ok(d.out.apply(g, y))
}

/// Kimi delta attention, three convolved streams, a per key dim decay gate, the recurrence and the gated norm
pub fn kimi_delta_attention_graph(g: &mut GraphBuilder, k: &Kda, x: Tid, ctx: &Ctx) -> Result<Tid> {
    let t = ctx.n_tokens;
    let (h, d) = (k.n_head, k.head_dim);
    let inner = h * d;
    let (q, kk, v) = match &k.qkv {
        QkvProj::Separate { q, k: kw, v } => (q.apply(g, x), kw.apply(g, x), v.apply(g, x)),
        QkvProj::Fused { w, .. } => {
            let all = w.apply(g, x);
            (g.cols(all, 0, inner), g.cols(all, inner, inner), g.cols(all, 2 * inner, inner))
        }
    };
    let conv = |g: &mut GraphBuilder, slot: usize, w, x| {
        let c = g.push(Op::CausalConv { layer: ctx.layer, slot, kernel: k.conv_kernel, dilation: 1, weight: w, bias: None }, &[x]);
        unary(g, Op::Silu, c)
    };
    let q = conv(g, 0, k.conv_q, q);
    let kk = conv(g, 1, k.conv_k, kk);
    let v = conv(g, 2, k.conv_v, v);
    let f = g.push(Op::MatMul { weight: k.f_a, bias: None, scale: None }, &[x]);
    let f = match k.f_b {
        Some(fb) => g.push(Op::MatMul { weight: fb, bias: None, scale: None }, &[f]),
        None => f,
    };
    let f = add_weight(g, f, k.dt_bias);
    let gate = if k.gate_lower_bound.is_finite() {
        let af = mul_per_head(g, f, k.a, t, h, d);
        let af = if k.a_is_exp { af } else { g.scale(af, -1.0) };
        let sg = unary(g, Op::Sigmoid, af);
        g.scale(sg, k.gate_lower_bound)
    } else {
        let sp = unary(g, Op::Softplus { beta: 1.0 }, f);
        let neg = if k.a_is_exp { -1.0 } else { 1.0 };
        let gt = mul_per_head(g, sp, k.a, t, h, d);
        g.scale(gt, neg)
    };
    let beta = g.push(Op::MatMul { weight: k.beta, bias: None, scale: None }, &[x]);
    let beta = unary(g, Op::Sigmoid, beta);
    let q = per_head_l2(g, q, t, h, d, k.l2_eps);
    let kk = per_head_l2(g, kk, t, h, d, k.l2_eps);
    let o = g.push(Op::GatedDeltaNet { layer: ctx.layer, n_head_k: h, n_head_v: h, head_dim: d, gate_per_channel: true }, &[q, kk, v, gate, beta]);
    let o = per_head_rms(g, o, t, h, d, Some(k.norm), k.norm_eps);
    let g2 = g.push(Op::MatMul { weight: k.g_a, bias: None, scale: None }, &[x]);
    let g2 = match k.g_b {
        Some(gb) => g.push(Op::MatMul { weight: gb, bias: None, scale: None }, &[g2]),
        None => g2,
    };
    let g2 = unary(g, Op::Sigmoid, g2);
    let y = g.mul(o, g2);
    let _ = (fill, mul_head_cols, row_weight);
    Ok(k.out.apply(g, y))
}

use crate::basic::Linear;
use anyhow::Result;
use kf_core::{Ctx, GraphBuilder, Layer, StateNeed, Tid, Wid};

/// RWKV6 time mix with the data dependent lerp lora and the wkv6 recurrence, QRWKV when first is absent
pub struct Rwkv6TimeMix {
    pub n_embd: usize,
    pub head_size: usize,
    pub n_head_kv: usize,
    pub extra_dim: usize,
    pub decay_extra_dim: usize,
    pub w1: Wid,
    pub w2: Wid,
    pub lerp_x: Wid,
    pub lerp_fused: Option<Wid>,
    pub lerp_w: Option<Wid>,
    pub lerp_k: Option<Wid>,
    pub lerp_v: Option<Wid>,
    pub lerp_r: Option<Wid>,
    pub lerp_g: Option<Wid>,
    pub first: Option<Wid>,
    pub decay: Wid,
    pub decay_w1: Wid,
    pub decay_w2: Wid,
    pub key: Linear,
    pub value: Linear,
    pub receptance: Linear,
    pub gate: Linear,
    pub ln: Option<Wid>,
    pub ln_b: Option<Wid>,
    pub output: Linear,
    pub shift_slot: usize,
}

impl Layer for Rwkv6TimeMix {
    fn forward(&self, g: &mut GraphBuilder, x: Tid, ctx: &Ctx) -> Result<Tid> {
        rwkv6_time_mix_graph(g, self, x, ctx)
    }
    fn weights(&self) -> Vec<Wid> {
        let mut w = vec![self.w1, self.w2, self.lerp_x, self.decay, self.decay_w1, self.decay_w2];
        w.extend([self.lerp_fused, self.lerp_w, self.lerp_k, self.lerp_v, self.lerp_r, self.lerp_g, self.first, self.ln, self.ln_b].into_iter().flatten());
        w.extend(self.key.weights());
        w.extend(self.value.weights());
        w.extend(self.receptance.weights());
        w.extend(self.gate.weights());
        w.extend(self.output.weights());
        w
    }
    fn states(&self) -> Vec<StateNeed> {
        vec![StateNeed::TokenShift { n: 1, width: self.n_embd }, StateNeed::Recurrent { conv_state: 0, ssm_state: self.n_embd * self.head_size }]
    }
}

/// RWKV6 channel mix, relu squared key and a sigmoid receptance gate
pub struct Rwkv6ChannelMix {
    pub n_embd: usize,
    pub lerp_k: Wid,
    pub lerp_r: Wid,
    pub key: Linear,
    pub value: Linear,
    pub receptance: Linear,
    pub shift_slot: usize,
}

impl Layer for Rwkv6ChannelMix {
    fn forward(&self, g: &mut GraphBuilder, x: Tid, ctx: &Ctx) -> Result<Tid> {
        rwkv6_channel_mix_graph(g, self, x, ctx)
    }
    fn weights(&self) -> Vec<Wid> {
        let mut w = vec![self.lerp_k, self.lerp_r];
        w.extend(self.key.weights());
        w.extend(self.value.weights());
        w.extend(self.receptance.weights());
        w
    }
    fn states(&self) -> Vec<StateNeed> {
        vec![StateNeed::TokenShift { n: 1, width: self.n_embd }]
    }
}

/// RWKV7 time mix with the decay, iclr, value residual and gate loras and the wkv7 recurrence
pub struct Rwkv7TimeMix {
    pub n_embd: usize,
    pub head_size: usize,
    pub layer: usize,
    pub w0: Wid,
    pub w1: Wid,
    pub w2: Wid,
    pub a0: Wid,
    pub a1: Wid,
    pub a2: Wid,
    pub v0: Option<Wid>,
    pub v1: Option<Wid>,
    pub v2: Option<Wid>,
    pub g1: Option<Wid>,
    pub g2: Option<Wid>,
    pub lerp_fused: Wid,
    pub n_lerp: usize,
    pub k_k: Wid,
    pub k_a: Wid,
    pub r_k: Wid,
    pub key: Linear,
    pub value: Linear,
    pub receptance: Linear,
    pub ln: Option<Wid>,
    pub ln_b: Option<Wid>,
    pub output: Linear,
    pub shift_slot: usize,
}

impl Layer for Rwkv7TimeMix {
    fn forward(&self, g: &mut GraphBuilder, x: Tid, ctx: &Ctx) -> Result<Tid> {
        rwkv7_time_mix_graph(g, self, x, ctx)
    }
    fn weights(&self) -> Vec<Wid> {
        let mut w = vec![self.w0, self.w1, self.w2, self.a0, self.a1, self.a2, self.lerp_fused, self.k_k, self.k_a, self.r_k];
        w.extend([self.v0, self.v1, self.v2, self.g1, self.g2, self.ln, self.ln_b].into_iter().flatten());
        w.extend(self.key.weights());
        w.extend(self.value.weights());
        w.extend(self.receptance.weights());
        w.extend(self.output.weights());
        w
    }
    fn states(&self) -> Vec<StateNeed> {
        vec![StateNeed::TokenShift { n: 1, width: self.n_embd }, StateNeed::Recurrent { conv_state: 0, ssm_state: self.n_embd * self.head_size }]
    }
}

/// RWKV7 channel mix, value of the relu squared key
pub struct Rwkv7ChannelMix {
    pub n_embd: usize,
    pub lerp_k: Wid,
    pub key: Linear,
    pub value: Linear,
    pub shift_slot: usize,
}

impl Layer for Rwkv7ChannelMix {
    fn forward(&self, g: &mut GraphBuilder, x: Tid, ctx: &Ctx) -> Result<Tid> {
        rwkv7_channel_mix_graph(g, self, x, ctx)
    }
    fn weights(&self) -> Vec<Wid> {
        let mut w = vec![self.lerp_k];
        w.extend(self.key.weights());
        w.extend(self.value.weights());
        w
    }
    fn states(&self) -> Vec<StateNeed> {
        vec![StateNeed::TokenShift { n: 1, width: self.n_embd }]
    }
}

use crate::util::{add_weight, mul_weight, row_weight, unary};
use anyhow::bail;
use kf_core::Op;

fn shift_diff(g: &mut GraphBuilder, x: Tid, layer: usize, slot: usize) -> Tid {
    let prev = g.push(Op::TokenShift { layer, slot }, &[x]);
    g.sub(prev, x)
}

/// x plus sx times a per channel constant
fn lerp(g: &mut GraphBuilder, x: Tid, sx: Tid, w: Wid) -> Tid {
    let m = mul_weight(g, sx, w);
    g.add(x, m)
}

/// Layer norm over each head without affine terms, the group norm of the rwkv time mixes
fn head_layer_norm(g: &mut GraphBuilder, y: Tid, t: usize, n_head: usize, hs: usize) -> Tid {
    let r = g.reshape(y, t * n_head, hs);
    let n = g.push(Op::LayerNorm { weight: None, bias: None, eps: 64e-5 }, &[r]);
    g.reshape(n, t, n_head * hs)
}

/// RWKV6 time mix, the data dependent lerps, the decay lora and the wkv6 or gated linear recurrence
pub fn rwkv6_time_mix_graph(g: &mut GraphBuilder, m: &Rwkv6TimeMix, x: Tid, ctx: &Ctx) -> Result<Tid> {
    let t = ctx.n_tokens;
    let (e, hs, dd) = (m.n_embd, m.head_size, m.extra_dim);
    let n_head = e / hs;
    let sx = shift_diff(g, x, ctx.layer, m.shift_slot);
    let xxx = lerp(g, x, sx, m.lerp_x);
    let mm = g.push(Op::MatMul { weight: m.w1, bias: None, scale: None }, &[xxx]);
    let mm = unary(g, Op::Tanh, mm);
    let fused = m.lerp_fused.map(|w| row_weight(g, w, t));
    let separate = [m.lerp_w, m.lerp_k, m.lerp_v, m.lerp_r, m.lerp_g];
    let mut xs = Vec::with_capacity(5);
    for gi in 0..5 {
        let mg = g.cols(mm, gi * dd, dd);
        let dg = g.push(Op::MatMulSlice { weight: m.w2, index: gi }, &[mg]);
        let dg = match (fused, separate[gi]) {
            (Some(f), _) => {
                let c = g.cols(f, gi * e, e);
                g.add(dg, c)
            }
            (None, Some(w)) => add_weight(g, dg, w),
            (None, None) => bail!("rwkv6 time mix without lerp weights for slot {gi}"),
        };
        let s = g.mul(sx, dg);
        xs.push(g.add(x, s));
    }
    let (xw, xk, xv, xr, xg) = (xs[0], xs[1], xs[2], xs[3], xs[4]);
    let r = m.receptance.apply(g, xr);
    let mut k = m.key.apply(g, xk);
    let mut v = m.value.apply(g, xv);
    let gt = m.gate.apply(g, xg);
    let gt = match m.first {
        Some(_) => unary(g, Op::Silu, gt),
        None => unary(g, Op::Sigmoid, gt),
    };
    if m.n_head_kv != 0 && m.n_head_kv != n_head {
        let rep = n_head / m.n_head_kv;
        let kr = g.reshape(k, t * m.n_head_kv, hs);
        let kr = g.push(Op::Tile { n: rep }, &[kr]);
        k = g.reshape(kr, t, e);
        let vr = g.reshape(v, t * m.n_head_kv, hs);
        let vr = g.push(Op::Tile { n: rep }, &[vr]);
        v = g.reshape(vr, t, e);
    }
    let w = g.push(Op::MatMul { weight: m.decay_w1, bias: None, scale: None }, &[xw]);
    let w = unary(g, Op::Tanh, w);
    let w = g.push(Op::MatMul { weight: m.decay_w2, bias: None, scale: None }, &[w]);
    let w = add_weight(g, w, m.decay);
    let w = unary(g, Op::Exp, w);
    let w = g.scale(w, -1.0);
    let w = unary(g, Op::Exp, w);
    let _ = m.decay_extra_dim;
    let mut y = match m.first {
        Some(first) => g.push(Op::Wkv6 { layer: ctx.layer, head_size: hs, first }, &[k, v, r, w]),
        None => {
            let kw = g.mul(k, w);
            let k = g.sub(k, kw);
            g.push(Op::Gla { layer: ctx.layer, head_size: hs, scale: (hs as f32).powf(-0.5) }, &[k, v, r, w])
        }
    };
    if m.first.is_some() {
        y = head_layer_norm(g, y, t, n_head, hs);
        if let Some(ln) = m.ln {
            y = mul_weight(g, y, ln);
        }
        if let Some(b) = m.ln_b {
            y = add_weight(g, y, b);
        }
    }
    let y = g.mul(y, gt);
    Ok(m.output.apply(g, y))
}

/// RWKV6 channel mix, squared relu key under a sigmoid receptance
pub fn rwkv6_channel_mix_graph(g: &mut GraphBuilder, m: &Rwkv6ChannelMix, x: Tid, ctx: &Ctx) -> Result<Tid> {
    let sx = shift_diff(g, x, ctx.layer, m.shift_slot);
    let xk = lerp(g, x, sx, m.lerp_k);
    let xr = lerp(g, x, sx, m.lerp_r);
    let r = m.receptance.apply(g, xr);
    let r = unary(g, Op::Sigmoid, r);
    let k = m.key.apply(g, xk);
    let k = unary(g, Op::ReluSqr, k);
    let v = m.value.apply(g, k);
    Ok(g.mul(r, v))
}

/// RWKV7 time mix with the decay, value residual, in context rate and gate loras around the wkv7 recurrence
pub fn rwkv7_time_mix_graph(g: &mut GraphBuilder, m: &Rwkv7TimeMix, x: Tid, ctx: &Ctx) -> Result<Tid> {
    let t = ctx.n_tokens;
    let (e, hs) = (m.n_embd, m.head_size);
    let n_head = e / hs;
    let sx = shift_diff(g, x, ctx.layer, m.shift_slot);
    let fused = row_weight(g, m.lerp_fused, t);
    let slot = |g: &mut GraphBuilder, s: usize| {
        let c = g.cols(fused, s * e, e);
        let d = g.mul(sx, c);
        g.add(x, d)
    };
    let xr = slot(g, 0);
    let xw = slot(g, 1);
    let xk = slot(g, 2);
    let xv = slot(g, 3);
    let xa = slot(g, 4);
    let r = m.receptance.apply(g, xr);
    let w = g.push(Op::MatMul { weight: m.w1, bias: None, scale: None }, &[xw]);
    let w = unary(g, Op::Tanh, w);
    let w = g.push(Op::MatMul { weight: m.w2, bias: None, scale: None }, &[w]);
    let w = add_weight(g, w, m.w0);
    let w = unary(g, Op::Sigmoid, w);
    let w = g.scale(w, -0.606_531);
    let w = unary(g, Op::Exp, w);
    let k = m.key.apply(g, xk);
    let mut v = m.value.apply(g, xv);
    if m.layer == 0 {
        let mut ho = ctx.handoff.borrow_mut();
        ho.clear();
        ho.push(v);
    } else {
        let first = ctx.handoff.borrow().first().copied().ok_or_else(|| anyhow::anyhow!("rwkv7 layer {} without the first layer value", m.layer))?;
        let (v0, v1, v2) = match (m.v0, m.v1, m.v2) {
            (Some(a), Some(b), Some(c)) => (a, b, c),
            _ => bail!("rwkv7 layer {} without its value residual lora", m.layer),
        };
        let mix = g.push(Op::MatMul { weight: v1, bias: None, scale: None }, &[xv]);
        let mix = g.push(Op::MatMul { weight: v2, bias: None, scale: None }, &[mix]);
        let mix = add_weight(g, mix, v0);
        let mix = unary(g, Op::Sigmoid, mix);
        let diff = g.sub(first, v);
        let diff = g.mul(diff, mix);
        v = g.add(v, diff);
    }
    let gate = match (m.g1, m.g2, m.n_lerp > 5) {
        (Some(g1), Some(g2), true) => {
            let xg = slot(g, 5);
            let gg = g.push(Op::MatMul { weight: g1, bias: None, scale: None }, &[xg]);
            let gg = unary(g, Op::Sigmoid, gg);
            Some(g.push(Op::MatMul { weight: g2, bias: None, scale: None }, &[gg]))
        }
        _ => None,
    };
    let a = g.push(Op::MatMul { weight: m.a1, bias: None, scale: None }, &[xa]);
    let a = g.push(Op::MatMul { weight: m.a2, bias: None, scale: None }, &[a]);
    let a = add_weight(g, a, m.a0);
    let a = unary(g, Op::Sigmoid, a);
    let kk = mul_weight(g, k, m.k_k);
    let kk = g.reshape(kk, t * n_head, hs);
    let kk = g.push(Op::L2Norm { eps: 1e-24 }, &[kk]);
    let kk = g.reshape(kk, t, e);
    let ka = mul_weight(g, k, m.k_a);
    let aka = g.mul(a, ka);
    let k = g.add(k, aka);
    let k = g.sub(k, ka);
    let neg_kk = g.scale(kk, -1.0);
    let kka = g.mul(kk, a);
    let mut y = g.push(Op::Wkv7 { layer: ctx.layer, head_size: hs }, &[r, w, k, v, neg_kk, kka]);
    if let (Some(ln), Some(lnb)) = (m.ln, m.ln_b) {
        y = head_layer_norm(g, y, t, n_head, hs);
        y = mul_weight(g, y, ln);
        y = add_weight(g, y, lnb);
    }
    let kr = g.mul(k, r);
    let kr = mul_weight(g, kr, m.r_k);
    let kr = g.reshape(kr, t * n_head, hs);
    let rk = g.push(Op::SumRows, &[kr]);
    let vr = g.reshape(v, t * n_head, hs);
    let bonus = g.mul(vr, rk);
    let bonus = g.reshape(bonus, t, e);
    y = g.add(y, bonus);
    if let Some(gt) = gate {
        y = g.mul(y, gt);
    }
    Ok(m.output.apply(g, y))
}

/// RWKV7 channel mix, value of the squared relu key with no receptance
pub fn rwkv7_channel_mix_graph(g: &mut GraphBuilder, m: &Rwkv7ChannelMix, x: Tid, ctx: &Ctx) -> Result<Tid> {
    let sx = shift_diff(g, x, ctx.layer, m.shift_slot);
    let xk = lerp(g, x, sx, m.lerp_k);
    let k = m.key.apply(g, xk);
    let k = unary(g, Op::ReluSqr, k);
    Ok(m.value.apply(g, k))
}

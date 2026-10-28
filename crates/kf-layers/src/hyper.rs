use anyhow::Result;
use kf_core::{Ctx, GraphBuilder, Layer, StateNeed, Tid, Wid};

/// Mixing weights of one hyper connection site
pub struct HcMix {
    pub fn_: Option<Wid>,
    pub base: Option<Wid>,
    pub scale: Option<Wid>,
    pub norm: Option<Wid>,
    pub down: Option<Wid>,
    pub up: Option<Wid>,
    pub inject: Option<Wid>,
    pub down_inject: Option<Wid>,
}

impl HcMix {
    pub fn weights(&self) -> Vec<Wid> {
        [self.fn_, self.base, self.scale, self.norm, self.down, self.up, self.inject, self.down_inject]
            .into_iter()
            .flatten()
            .collect()
    }
}

/// The three hyper connection families, DeepSeek sinkhorn, Qwen low rank gated, and independent iHC
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HcKind {
    Sinkhorn { iters: usize, eps: f32 },
    LowRank { rank: usize },
    Independent { magnitude: f32, eps: f32 },
}

/// A block whose residual is n streams mixed before and after each sublayer
pub struct HyperConnected {
    pub n_streams: usize,
    pub n_embd: usize,
    pub kind: HcKind,
    pub attn_mix: HcMix,
    pub ffn_mix: HcMix,
    pub attn_norm: Box<dyn Layer>,
    pub attn: Box<dyn Layer>,
    pub ffn_norm: Box<dyn Layer>,
    pub ffn: Box<dyn Layer>,
    /// V4.1 single pass, each sublayer collapses with the previous sublayer's pre weights
    pub single_pass: bool,
    pub eps: f32,
}

impl Layer for HyperConnected {
    fn forward(&self, g: &mut GraphBuilder, x: Tid, ctx: &Ctx) -> Result<Tid> {
        hyper_connection_block_graph(g, self, x, ctx)
    }
    fn weights(&self) -> Vec<Wid> {
        let mut w = self.attn_mix.weights();
        w.extend(self.ffn_mix.weights());
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

/// Learned collapse of the residual streams into one before the output head
pub struct HcHead {
    pub n_streams: usize,
    pub n_embd: usize,
    pub kind: HcKind,
    pub mix: HcMix,
    pub eps: f32,
}

impl Layer for HcHead {
    fn forward(&self, g: &mut GraphBuilder, x: Tid, ctx: &Ctx) -> Result<Tid> {
        hyper_connection_head_graph(g, self, x, ctx)
    }
    fn weights(&self) -> Vec<Wid> {
        self.mix.weights()
    }
}

/// Plain mean of the residual streams before the output head
pub struct HcMean {
    pub n_streams: usize,
}

impl Layer for HcMean {
    fn forward(&self, g: &mut GraphBuilder, x: Tid, ctx: &Ctx) -> Result<Tid> {
        hyper_connection_mean_graph(g, self, x, ctx)
    }
    fn weights(&self) -> Vec<Wid> {
        Vec::new()
    }
}

use crate::util::{fill, grouped_rms, row_weight, unary};
use kf_core::Op;

fn need(w: Option<Wid>, what: &str) -> Result<Wid> {
    w.ok_or_else(|| anyhow::anyhow!("hyper connection mix without its {what} weight"))
}

/// Pre weights, post weights and the sinkhorn combination of the DeepSeek V4 form
fn mixes_sinkhorn(g: &mut GraphBuilder, mix: &HcMix, x: Tid, n: usize, iters: usize, hc_eps: f32, eps: f32, t: usize) -> Result<(Tid, Tid, Tid)> {
    let (f, scale, base) = (need(mix.fn_, "fn")?, need(mix.scale, "scale")?, need(mix.base, "base")?);
    let r = g.push(Op::RmsNorm { weight: None, bias: None, eps }, &[x]);
    let m = g.push(Op::MatMul { weight: f, bias: None, scale: None }, &[r]);
    let sc = row_weight(g, scale, t);
    let bs = row_weight(g, base, t);
    let affine = |g: &mut GraphBuilder, start: usize, len: usize, si: usize| {
        let v = g.cols(m, start, len);
        let s = g.cols(sc, si, 1);
        let b = g.cols(bs, start, len);
        let v = g.mul(v, s);
        g.add(v, b)
    };
    let pre = affine(g, 0, n, 0);
    let pre = unary(g, Op::Sigmoid, pre);
    let pre = g.push(Op::ScaleBias { scale: 1.0, bias: hc_eps }, &[pre]);
    let post = affine(g, n, n, 1);
    let post = unary(g, Op::Sigmoid, post);
    let post = g.scale(post, 2.0);
    let c0 = affine(g, 2 * n, n * n, 2);
    let comb = g.push(Op::HcSinkhorn { n_streams: n, iters, eps: hc_eps }, &[c0]);
    Ok((pre, post, comb))
}

/// Pre and post weights of the independent form, the post scaled by the magnitude
fn mixes_independent(g: &mut GraphBuilder, mix: &HcMix, x: Tid, n: usize, magnitude: f32, hc_eps: f32, eps: f32, t: usize) -> Result<(Tid, Tid)> {
    let (f, scale, base) = (need(mix.fn_, "fn")?, need(mix.scale, "scale")?, need(mix.base, "base")?);
    let r = g.push(Op::RmsNorm { weight: None, bias: None, eps }, &[x]);
    let m = g.push(Op::MatMul { weight: f, bias: None, scale: None }, &[r]);
    let sc = row_weight(g, scale, t);
    let bs = row_weight(g, base, t);
    let affine = |g: &mut GraphBuilder, start: usize, si: usize| {
        let v = g.cols(m, start, n);
        let s = g.cols(sc, si, 1);
        let b = g.cols(bs, start, n);
        let v = g.mul(v, s);
        let v = g.add(v, b);
        unary(g, Op::Sigmoid, v)
    };
    let pre = affine(g, 0, 0);
    let pre = g.push(Op::ScaleBias { scale: 1.0, bias: hc_eps }, &[pre]);
    let post = affine(g, n, 1);
    let post = g.push(Op::ScaleBias { scale: magnitude, bias: hc_eps }, &[post]);
    Ok((pre, post))
}

/// Low rank gated mix of the Qwen form, the mixed input and the inject logits per stream
fn mix_low_rank(g: &mut GraphBuilder, mix: &HcMix, x: Tid, n: usize, e: usize, rank: usize, eps: f32, t: usize) -> Result<(Tid, Option<Tid>)> {
    let xn = grouped_rms(g, x, t, n, e, mix.norm, eps);
    let (lo, inj) = match mix.down_inject {
        Some(di) => {
            let z = g.push(Op::MatMul { weight: di, bias: None, scale: None }, &[xn]);
            let z = g.scale(z, 1.0 / n as f32);
            (g.cols(z, 0, rank), Some(g.cols(z, rank, n)))
        }
        None => {
            let lo = g.push(Op::MatMul { weight: need(mix.down, "down")?, bias: None, scale: None }, &[xn]);
            let lo = g.scale(lo, 1.0 / n as f32);
            let inj = match mix.inject {
                Some(w) => {
                    let v = g.push(Op::MatMul { weight: w, bias: None, scale: None }, &[xn]);
                    Some(g.scale(v, 1.0 / n as f32))
                }
                None => None,
            };
            (lo, inj)
        }
    };
    let lo = unary(g, Op::Silu, lo);
    let gate = g.push(Op::MatMul { weight: need(mix.up, "up")?, bias: None, scale: None }, &[lo]);
    let m = g.push(Op::HcMixGated { n_streams: n, scale: 1.0 / n as f32 }, &[xn, gate]);
    Ok((m, inj))
}

/// Pre weights of a head mix in the sinkhorn or independent form
fn head_pre(g: &mut GraphBuilder, mix: &HcMix, x: Tid, _n: usize, hc_eps: f32, eps: f32, t: usize) -> Result<Tid> {
    let (f, scale, base) = (need(mix.fn_, "fn")?, need(mix.scale, "scale")?, need(mix.base, "base")?);
    let r = g.push(Op::RmsNorm { weight: None, bias: None, eps }, &[x]);
    let m = g.push(Op::MatMul { weight: f, bias: None, scale: None }, &[r]);
    let s = row_weight(g, scale, t);
    let b = row_weight(g, base, t);
    let m = g.mul(m, s);
    let m = g.add(m, b);
    let m = unary(g, Op::Sigmoid, m);
    Ok(g.push(Op::ScaleBias { scale: 1.0, bias: hc_eps }, &[m]))
}

/// One block over n residual streams mixed before and after each sublayer per the family
pub fn hyper_connection_block_graph(g: &mut GraphBuilder, hc: &HyperConnected, x: Tid, ctx: &Ctx) -> Result<Tid> {
    let t = ctx.n_tokens;
    let n = hc.n_streams;
    let mut c = ctx.clone();
    match hc.kind {
        HcKind::Sinkhorn { iters, eps: hc_eps } if hc.single_pass => {
            let prev = ctx.handoff.borrow().first().copied();
            let (a_pre, a_post, a_comb) = mixes_sinkhorn(g, &hc.attn_mix, x, n, iters, hc_eps, hc.eps, t)?;
            let h = match prev {
                Some(p) => g.push(Op::HcMix { n_streams: n }, &[x, p]),
                None => g.cols(x, 0, hc.n_embd),
            };
            c.residual_input = h;
            let h = hc.attn_norm.forward(g, h, &c)?;
            let a = hc.attn.forward(g, h, &c)?;
            let x = g.push(Op::HcCombine { n_streams: n, comb: true }, &[x, a, a_post, a_comb]);
            let (f_pre, f_post, f_comb) = mixes_sinkhorn(g, &hc.ffn_mix, x, n, iters, hc_eps, hc.eps, t)?;
            let h = g.push(Op::HcMix { n_streams: n }, &[x, a_pre]);
            c.residual_input = h;
            let h = hc.ffn_norm.forward(g, h, &c)?;
            let f = hc.ffn.forward(g, h, &c)?;
            let x = g.push(Op::HcCombine { n_streams: n, comb: true }, &[x, f, f_post, f_comb]);
            let mut ho = ctx.handoff.borrow_mut();
            ho.clear();
            ho.push(f_pre);
            Ok(x)
        }
        HcKind::Sinkhorn { iters, eps: hc_eps } => {
            let (pre, post, comb) = mixes_sinkhorn(g, &hc.attn_mix, x, n, iters, hc_eps, hc.eps, t)?;
            let h = g.push(Op::HcMix { n_streams: n }, &[x, pre]);
            c.residual_input = h;
            let h = hc.attn_norm.forward(g, h, &c)?;
            let a = hc.attn.forward(g, h, &c)?;
            let x = g.push(Op::HcCombine { n_streams: n, comb: true }, &[x, a, post, comb]);
            let (pre, post, comb) = mixes_sinkhorn(g, &hc.ffn_mix, x, n, iters, hc_eps, hc.eps, t)?;
            let h = g.push(Op::HcMix { n_streams: n }, &[x, pre]);
            c.residual_input = h;
            let h = hc.ffn_norm.forward(g, h, &c)?;
            let f = hc.ffn.forward(g, h, &c)?;
            Ok(g.push(Op::HcCombine { n_streams: n, comb: true }, &[x, f, post, comb]))
        }
        HcKind::Independent { magnitude, eps: hc_eps } => {
            let (pre, post) = mixes_independent(g, &hc.attn_mix, x, n, magnitude, hc_eps, hc.eps, t)?;
            let h = g.push(Op::HcMix { n_streams: n }, &[x, pre]);
            c.residual_input = h;
            let h = hc.attn_norm.forward(g, h, &c)?;
            let a = hc.attn.forward(g, h, &c)?;
            let x = g.push(Op::HcCombine { n_streams: n, comb: false }, &[x, a, post]);
            let (pre, post) = mixes_independent(g, &hc.ffn_mix, x, n, magnitude, hc_eps, hc.eps, t)?;
            let h = g.push(Op::HcMix { n_streams: n }, &[x, pre]);
            c.residual_input = h;
            let h = hc.ffn_norm.forward(g, h, &c)?;
            let f = hc.ffn.forward(g, h, &c)?;
            Ok(g.push(Op::HcCombine { n_streams: n, comb: false }, &[x, f, post]))
        }
        HcKind::LowRank { rank } => {
            let e = hc.n_embd;
            let (m, inj) = mix_low_rank(g, &hc.attn_mix, x, n, e, rank, hc.eps, t)?;
            let inj = inj.ok_or_else(|| anyhow::anyhow!("low rank hyper connection block without an inject weight"))?;
            c.residual_input = m;
            let h = hc.attn_norm.forward(g, m, &c)?;
            let a = hc.attn.forward(g, h, &c)?;
            let w = unary(g, Op::Sigmoid, inj);
            let w = g.scale(w, 2.0);
            let x = g.push(Op::HcCombine { n_streams: n, comb: false }, &[x, a, w]);
            let (m, inj) = mix_low_rank(g, &hc.ffn_mix, x, n, e, rank, hc.eps, t)?;
            let inj = inj.ok_or_else(|| anyhow::anyhow!("low rank hyper connection block without an inject weight"))?;
            c.residual_input = m;
            let h = hc.ffn_norm.forward(g, m, &c)?;
            let f = hc.ffn.forward(g, h, &c)?;
            let w = unary(g, Op::Sigmoid, inj);
            let w = g.scale(w, 2.0);
            Ok(g.push(Op::HcCombine { n_streams: n, comb: false }, &[x, f, w]))
        }
    }
}

/// Collapse of the streams before the head, learned per family or the lagged pre mix
pub fn hyper_connection_head_graph(g: &mut GraphBuilder, hd: &HcHead, x: Tid, ctx: &Ctx) -> Result<Tid> {
    let t = ctx.n_tokens;
    let n = hd.n_streams;
    if hd.mix.weights().is_empty() {
        let prev = ctx.handoff.borrow().first().copied().ok_or_else(|| anyhow::anyhow!("head collapse without a lagged pre mix"))?;
        return Ok(g.push(Op::HcMix { n_streams: n }, &[x, prev]));
    }
    match hd.kind {
        HcKind::Sinkhorn { eps: hc_eps, .. } | HcKind::Independent { eps: hc_eps, .. } => {
            let pre = head_pre(g, &hd.mix, x, n, hc_eps, hd.eps, t)?;
            Ok(g.push(Op::HcMix { n_streams: n }, &[x, pre]))
        }
        HcKind::LowRank { rank } => Ok(mix_low_rank(g, &hd.mix, x, n, hd.n_embd, rank, hd.eps, t)?.0),
    }
}

pub fn hyper_connection_mean_graph(g: &mut GraphBuilder, m: &HcMean, x: Tid, ctx: &Ctx) -> Result<Tid> {
    let w = fill(g, ctx.n_tokens, m.n_streams, 1.0 / m.n_streams as f32);
    Ok(g.push(Op::HcMix { n_streams: m.n_streams }, &[x, w]))
}

use crate::attention::Attention;
use crate::basic::{Linear, RmsNorm};
use crate::ffn::Ffn;
use anyhow::Result;
use kf_core::{Ctx, GraphBuilder, Layer, StateNeed, Tid, Wid};

/// Gemma per layer embeddings, the hashed table row and model projection mixed into a per layer input
pub struct PerLayerEmbedding {
    pub width: usize,
    pub n_layer: usize,
    pub layer: usize,
    pub n_embd: usize,
    pub table: Wid,
    pub model_proj: Wid,
    pub proj_norm: Wid,
    pub inp_gate: Linear,
    pub proj: Linear,
    pub post_norm: RmsNorm,
    pub eps: f32,
}

impl Layer for PerLayerEmbedding {
    fn forward(&self, g: &mut GraphBuilder, x: Tid, ctx: &Ctx) -> Result<Tid> {
        per_layer_embedding_graph(g, self, x, ctx)
    }
    fn weights(&self) -> Vec<Wid> {
        let mut w = vec![self.table, self.model_proj, self.proj_norm];
        w.extend(self.inp_gate.weights());
        w.extend(self.proj.weights());
        w.extend(self.post_norm.weights());
        w
    }
}

/// Expands the embedding into the alternating update streams with magnitude matched projections
pub struct AltUpEntry {
    pub proj: Wid,
    pub n_altup: usize,
    pub n_embd: usize,
}

impl Layer for AltUpEntry {
    fn forward(&self, g: &mut GraphBuilder, x: Tid, ctx: &Ctx) -> Result<Tid> {
        altup_entry_graph(g, self, x, ctx)
    }
    fn weights(&self) -> Vec<Wid> {
        vec![self.proj]
    }
}

/// Collapses the alternating update streams to one before the final norm
pub struct AltUpExit {
    pub unembd: Wid,
    pub n_altup: usize,
    pub n_embd: usize,
    pub active: usize,
}

impl Layer for AltUpExit {
    fn forward(&self, g: &mut GraphBuilder, x: Tid, ctx: &Ctx) -> Result<Tid> {
        altup_exit_graph(g, self, x, ctx)
    }
    fn weights(&self) -> Vec<Wid> {
        vec![self.unembd]
    }
}

/// Gemma 3n block, altup predict and correct around attention with laurel and a sparsified feed forward
pub struct Gemma3nBlock {
    pub n_altup: usize,
    pub active: usize,
    pub n_embd: usize,
    pub eps: f32,
    pub router: Wid,
    pub router_norm: Wid,
    pub predict_coef: Wid,
    pub correct_coef: Wid,
    pub correct_scale: Wid,
    pub laurel_l: Wid,
    pub laurel_r: Wid,
    pub laurel_post_norm: Wid,
    pub attn_norm: RmsNorm,
    pub attn: Attention,
    pub post_attn_norm: RmsNorm,
    pub ffn_norm: RmsNorm,
    pub ffn: Ffn,
    pub post_ffn_norm: RmsNorm,
    /// Standard deviations above the mean kept by the gate sparsity, none when inactive
    pub sparsity: Option<f32>,
    pub n_ff: usize,
    pub ple: Option<PerLayerEmbedding>,
}

impl Layer for Gemma3nBlock {
    fn forward(&self, g: &mut GraphBuilder, x: Tid, ctx: &Ctx) -> Result<Tid> {
        gemma3n_block_graph(g, self, x, ctx)
    }
    fn weights(&self) -> Vec<Wid> {
        let mut w = vec![self.router, self.router_norm, self.predict_coef, self.correct_coef, self.correct_scale, self.laurel_l, self.laurel_r, self.laurel_post_norm];
        w.extend(self.attn_norm.weights());
        w.extend(self.attn.weights());
        w.extend(self.post_attn_norm.weights());
        w.extend(self.ffn_norm.weights());
        w.extend(self.ffn.weights());
        w.extend(self.post_ffn_norm.weights());
        w.extend(self.ple.iter().flat_map(Layer::weights));
        w
    }
    fn states(&self) -> Vec<StateNeed> {
        self.attn.states()
    }
}

use crate::util::{join, magnitude, mul_weight, rms, split_streams, unary};
use anyhow::bail;
use kf_core::Op;

/// The layer's slice of the per layer inputs, gated by the stream and projected back with a norm
pub fn per_layer_embedding_graph(g: &mut GraphBuilder, p: &PerLayerEmbedding, x: Tid, ctx: &Ctx) -> Result<Tid> {
    let (w, e) = (p.width, p.n_embd);
    let rows = g.push(Op::Gather { weight: p.table }, &[ctx.tokens]);
    let r = g.cols(rows, p.layer * w, w);
    let r = g.scale(r, (w as f32).sqrt());
    let q = g.push(Op::MatMul { weight: p.model_proj, bias: None, scale: None }, &[ctx.embed]);
    let q = g.cols(q, p.layer * w, w);
    let q = g.scale(q, 1.0 / (e as f32).sqrt());
    let q = rms(g, q, Some(p.proj_norm), p.eps);
    let pl = g.add(q, r);
    let pl = g.scale(pl, std::f32::consts::FRAC_1_SQRT_2);
    let gate = p.inp_gate.apply(g, x);
    let gate = unary(g, Op::Gelu, gate);
    let v = g.mul(gate, pl);
    let v = p.proj.apply(g, v);
    let _ = p.n_layer;
    Ok(p.post_norm.apply(g, v))
}

/// The embedding widened into magnitude matched streams
pub fn altup_entry_graph(g: &mut GraphBuilder, a: &AltUpEntry, x: Tid, _ctx: &Ctx) -> Result<Tid> {
    let mag = magnitude(g, x);
    let mut parts = vec![x];
    for k in 1..a.n_altup {
        let y = g.push(Op::MatMulSlice { weight: a.proj, index: k - 1 }, &[x]);
        let my = magnitude(g, y);
        let y = g.mul(y, mag);
        parts.push(g.div(y, my));
    }
    let _ = a.n_embd;
    Ok(join(g, &parts))
}

/// Streams collapsed to one, the side streams unembedded and matched to the active magnitude
pub fn altup_exit_graph(g: &mut GraphBuilder, a: &AltUpExit, x: Tid, _ctx: &Ctx) -> Result<Tid> {
    let streams = split_streams(g, x, a.n_altup, a.n_embd);
    let t = magnitude(g, streams[a.active]);
    let mut acc = streams[0];
    for k in 1..a.n_altup {
        let u = g.push(Op::MatMulSlice { weight: a.unembd, index: k - 1 }, &[streams[k]]);
        let mu = magnitude(g, u);
        let u = g.mul(u, t);
        let u = g.div(u, mu);
        acc = g.add(acc, u);
    }
    Ok(g.scale(acc, 1.0 / a.n_altup as f32))
}

/// Router modalities of one stream, tanh of the normed scaled projection
fn modalities(g: &mut GraphBuilder, b: &Gemma3nBlock, x: Tid) -> Tid {
    let r = rms(g, x, Some(b.router_norm), b.eps);
    let r = g.scale(r, 1.0 / b.n_embd as f32);
    let m = g.push(Op::MatMul { weight: b.router, bias: None, scale: None }, &[r]);
    unary(g, Op::Tanh, m)
}

/// Gemma 3n block over the altup streams with laurel, sparsity, post norms and per layer embeddings
pub fn gemma3n_block_graph(g: &mut GraphBuilder, b: &Gemma3nBlock, x: Tid, ctx: &Ctx) -> Result<Tid> {
    let (n, e, act) = (b.n_altup, b.n_embd, b.active);
    if act >= n {
        bail!("active altup stream {act} of {n}");
    }
    let streams = split_streams(g, x, n, e);
    let m = modalities(g, b, streams[act]);
    let coef = g.push(Op::MatMul { weight: b.predict_coef, bias: None, scale: None }, &[m]);
    let mut pred = Vec::with_capacity(n);
    for c in 0..n {
        let mut acc = streams[c];
        for (j, sj) in streams.iter().enumerate() {
            let w = g.cols(coef, c * n + j, 1);
            let term = g.mul(*sj, w);
            acc = g.add(acc, term);
        }
        pred.push(acc);
    }
    let act_in = pred[act];
    let mut c = ctx.clone();
    c.residual_input = act_in;
    let h = b.attn_norm.apply(g, act_in);
    let lau = g.push(Op::MatMul { weight: b.laurel_l, bias: None, scale: None }, &[h]);
    let lau = g.push(Op::MatMul { weight: b.laurel_r, bias: None, scale: None }, &[lau]);
    let lau = rms(g, lau, Some(b.laurel_post_norm), b.eps);
    let lau = g.add(h, lau);
    let a = b.attn.forward(g, h, &c)?;
    let a = b.post_attn_norm.apply(g, a);
    let a = g.add(a, act_in);
    let al = g.add(a, lau);
    let al = g.scale(al, std::f32::consts::FRAC_1_SQRT_2);
    let tt = b.ffn_norm.apply(g, al);
    let up = b.ffn.up.apply(g, tt);
    let Some(gate) = &b.ffn.gate else { bail!("gemma 3n feed forward without a gate projection") };
    let mut gt = gate.apply(g, tt);
    if let Some(s) = b.sparsity {
        let nf = b.n_ff as f32;
        let mean = g.push(Op::SumRows, &[gt]);
        let mean = g.scale(mean, 1.0 / nf);
        let diff = g.sub(gt, mean);
        let var = unary(g, Op::Sqr, diff);
        let var = g.push(Op::SumRows, &[var]);
        let var = g.scale(var, 1.0 / (nf - 1.0));
        let std = unary(g, Op::Sqrt, var);
        let std = g.scale(std, s);
        let cut = g.add(mean, std);
        gt = g.sub(gt, cut);
        gt = unary(g, Op::Relu, gt);
    }
    let gt = unary(g, Op::Gelu, gt);
    let hid = g.mul(gt, up);
    let f = b.ffn.down.apply(g, hid);
    let f = b.post_ffn_norm.apply(g, f);
    let y = g.add(f, al);
    let m2 = modalities(g, b, y);
    let innov = g.sub(y, act_in);
    let coef2 = g.push(Op::MatMul { weight: b.correct_coef, bias: None, scale: None }, &[m2]);
    let coef2 = g.push(Op::ScaleBias { scale: 1.0, bias: 1.0 }, &[coef2]);
    let mut out = Vec::with_capacity(n);
    for (k, pk) in pred.iter().enumerate() {
        let w = g.cols(coef2, k, 1);
        let term = g.mul(innov, w);
        out.push(g.add(*pk, term));
    }
    if let Some(ple) = &b.ple {
        let p = mul_weight(g, out[act], b.correct_scale);
        let p = ple.forward(g, p, &c)?;
        for o in out.iter_mut().skip(1) {
            *o = g.add(*o, p);
        }
    }
    Ok(join(g, &out))
}

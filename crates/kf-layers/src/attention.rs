use crate::basic::{HeadNorm, Linear, NormKind, RmsNorm};
use crate::mla::DsaIndexer;
use anyhow::Result;
use kf_core::{Ctx, GraphBuilder, Layer, Op, RopeParams, StateNeed, Tid, Wid, Window};

pub enum QkvProj {
    Separate { q: Linear, k: Linear, v: Linear },
    /// One projection, with separate q k v biases added after the split when the file carries them
    Fused { w: Linear, split_bias: Option<[Wid; 3]> },
}

impl QkvProj {
    pub fn weights(&self) -> Vec<Wid> {
        match self {
            QkvProj::Separate { q, k, v } => [q.weights(), k.weights(), v.weights()].concat(),
            QkvProj::Fused { w, split_bias } => {
                let mut ws = w.weights();
                ws.extend(split_bias.iter().flatten());
                ws
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GateKind {
    Sigmoid,
    Softplus,
    Softplus2,
}

/// Output gate of an attention mixer, projected from the mixer input or carried inside q
pub struct AttnGate {
    pub proj: Option<Linear>,
    pub in_q: bool,
    pub per_head: bool,
    pub bias: Option<Wid>,
    pub kind: GateKind,
}

impl AttnGate {
    pub fn weights(&self) -> Vec<Wid> {
        let mut w: Vec<Wid> = self.proj.iter().flat_map(Layer::weights).collect();
        w.extend(self.bias);
        w
    }
    /// Applies the gate to [tokens, heads * dim], gate values are per element or one per head
    pub fn apply(&self, g: &mut GraphBuilder, a: Tid, gate: Tid, n_tokens: usize, n_head: usize, dim: usize) -> Tid {
        let mut gv = gate;
        if let Some(b) = self.bias {
            gv = g.push(Op::AddWeight { weight: b }, &[gv]);
        }
        gv = match self.kind {
            GateKind::Sigmoid => g.push(Op::Sigmoid, &[gv]),
            GateKind::Softplus => g.push(Op::Softplus { beta: 1.0 }, &[gv]),
            GateKind::Softplus2 => {
                let s = g.scale(gv, std::f32::consts::LN_2);
                let s = g.push(Op::Softplus { beta: 1.0 }, &[s]);
                g.scale(s, 1.0 / std::f32::consts::LN_2)
            }
        };
        if self.per_head {
            let ar = g.reshape(a, n_tokens * n_head, dim);
            let gr = g.reshape(gv, n_tokens * n_head, 1);
            let m = g.mul(ar, gr);
            g.reshape(m, n_tokens, n_head * dim)
        } else {
            g.mul(a, gv)
        }
    }
}

/// Query temperature by position, applied after rope
#[derive(Clone, Copy, Debug)]
pub struct AttnTemp {
    pub floor_scale: f32,
    pub scale: f32,
    pub offset: f32,
    pub pow: bool,
}

/// Multi head attention with grouped keys, the common dense and MoE mixer
pub struct Attention {
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub head_dim_v: usize,
    pub qkv: QkvProj,
    /// v is the raw k projection
    pub k_as_v: bool,
    pub q_norm: Option<HeadNorm>,
    pub k_norm: Option<HeadNorm>,
    pub v_norm: Option<HeadNorm>,
    pub norm_after_rope: bool,
    pub rope: Option<RopeParams>,
    pub rope_offset: usize,
    pub out: Linear,
    pub gate: Option<AttnGate>,
    pub clamp: f32,
    pub temp: Option<AttnTemp>,
    pub window: Option<Window>,
    pub scale: f32,
    pub softcap: f32,
    pub sinks: Option<Wid>,
    pub alibi: f32,
    /// Reads another layer's cache without writing, q only
    pub kv_source: Option<usize>,
    /// Weightless per head norm on the attention result before the gate
    pub out_head_norm: bool,
    /// Norm on the attention output before the output projection
    pub sub_norm: Option<RmsNorm>,
    /// Sparse selection of keys through an indexer, attention reads only the chosen cells
    pub sparse: Option<DsaIndexer>,
}

impl Attention {
    /// q k v projections, q still holding the interleaved gate when the gate rides in q
    pub fn project(&self, g: &mut GraphBuilder, x: Tid) -> (Tid, Tid, Tid) {
        let gate_in_q = self.gate.as_ref().is_some_and(|gt| gt.in_q);
        let q_width = self.n_head * self.head_dim * if gate_in_q { 2 } else { 1 };
        let k_width = self.n_head_kv * self.head_dim;
        let v_width = self.n_head_kv * self.head_dim_v;
        let (q, k, v) = match &self.qkv {
            QkvProj::Separate { q, k, v } => {
                let kk = k.apply(g, x);
                let vv = if self.k_as_v { kk } else { v.apply(g, x) };
                (q.apply(g, x), kk, vv)
            }
            QkvProj::Fused { w, split_bias } => {
                let all = w.apply(g, x);
                let mut q = g.cols(all, 0, q_width);
                let mut k = g.cols(all, q_width, k_width);
                let mut v = if self.k_as_v { k } else { g.cols(all, q_width + k_width, v_width) };
                if let Some([qb, kb, vb]) = split_bias {
                    q = g.push(Op::AddWeight { weight: *qb }, &[q]);
                    k = g.push(Op::AddWeight { weight: *kb }, &[k]);
                    if !self.k_as_v {
                        v = g.push(Op::AddWeight { weight: *vb }, &[v]);
                    }
                }
                (q, k, v)
            }
        };
        if self.clamp > 0.0 {
            let c = Op::Clamp { min: -self.clamp, max: self.clamp };
            (g.push(c.clone(), &[q]), g.push(c.clone(), &[k]), g.push(c, &[v]))
        } else {
            (q, k, v)
        }
    }

    /// Splits the per head interleaved [q_h | g_h] projection into q and the gate
    pub fn split_q_gate(&self, g: &mut GraphBuilder, q: Tid, n_tokens: usize) -> (Tid, Tid) {
        let r = g.reshape(q, n_tokens * self.n_head, 2 * self.head_dim);
        let qq = g.cols(r, 0, self.head_dim);
        let gg = g.cols(r, self.head_dim, self.head_dim);
        (g.reshape(qq, n_tokens, self.n_head * self.head_dim), g.reshape(gg, n_tokens, self.n_head * self.head_dim))
    }

    pub fn rope_q(&self, g: &mut GraphBuilder, q: Tid, ctx: &Ctx) -> Tid {
        match &self.rope {
            Some(r) => g.push(Op::Rope { params: r.clone(), n_head: self.n_head, head_dim: self.head_dim, offset: self.rope_offset, inverse: false }, &[q, ctx.positions]),
            None => q,
        }
    }

    pub fn rope_k(&self, g: &mut GraphBuilder, k: Tid, ctx: &Ctx) -> Tid {
        match &self.rope {
            Some(r) => g.push(Op::Rope { params: r.clone(), n_head: self.n_head_kv, head_dim: self.head_dim, offset: self.rope_offset, inverse: false }, &[k, ctx.positions]),
            None => k,
        }
    }

    /// q k v after norms, rope and temperature, plus the gate values when the gate rides in q
    pub fn prepare(&self, g: &mut GraphBuilder, x: Tid, ctx: &Ctx) -> (Tid, Tid, Tid, Option<Tid>) {
        let (mut q, mut k, mut v) = self.project(g, x);
        let mut fused_gate = None;
        if self.gate.as_ref().is_some_and(|gt| gt.in_q) {
            let (qq, gg) = self.split_q_gate(g, q, ctx.n_tokens);
            q = qq;
            fused_gate = Some(gg);
        }
        if !self.norm_after_rope {
            if let Some(n) = &self.q_norm {
                q = n.apply(g, q, ctx.n_tokens, self.n_head, self.head_dim);
            }
            if let Some(n) = &self.k_norm {
                k = n.apply(g, k, ctx.n_tokens, self.n_head_kv, self.head_dim);
            }
        }
        if let Some(n) = &self.v_norm {
            v = n.apply(g, v, ctx.n_tokens, self.n_head_kv, self.head_dim_v);
        }
        q = self.rope_q(g, q, ctx);
        k = self.rope_k(g, k, ctx);
        if self.norm_after_rope {
            if let Some(n) = &self.q_norm {
                q = n.apply(g, q, ctx.n_tokens, self.n_head, self.head_dim);
            }
            if let Some(n) = &self.k_norm {
                k = n.apply(g, k, ctx.n_tokens, self.n_head_kv, self.head_dim);
            }
        }
        if let Some(t) = &self.temp {
            q = g.push(Op::AttnTemp { floor_scale: t.floor_scale, scale: t.scale, offset: t.offset, pow: t.pow }, &[q, ctx.positions]);
        }
        (q, k, v, fused_gate)
    }

    /// Gate, sub norm and output projection applied to the attention result
    pub fn finish(&self, g: &mut GraphBuilder, a: Tid, x: Tid, fused_gate: Option<Tid>, ctx: &Ctx) -> Tid {
        let mut a = a;
        if self.out_head_norm {
            a = HeadNorm::rms_weightless(self.sub_norm.map_or(1e-6, |n| n.eps)).apply(g, a, ctx.n_tokens, self.n_head, self.head_dim_v);
        }
        if let Some(gate) = &self.gate {
            let gv = match (&gate.proj, fused_gate) {
                (Some(p), _) => p.apply(g, x),
                (None, Some(f)) => f,
                (None, None) => a,
            };
            a = gate.apply(g, a, gv, ctx.n_tokens, self.n_head, self.head_dim_v);
        }
        if let Some(n) = &self.sub_norm {
            a = n.apply(g, a);
        }
        self.out.apply(g, a)
    }

    pub fn attention_op(&self, layer: usize, top_k: Option<usize>) -> Op {
        Op::Attention {
            layer,
            n_head: self.n_head,
            n_head_kv: self.n_head_kv,
            head_dim: self.head_dim,
            head_dim_v: self.head_dim_v,
            scale: self.scale,
            window: self.window,
            softcap: self.softcap,
            sinks: self.sinks,
            alibi_bias: self.alibi,
            kv_source: self.kv_source,
            top_k,
        }
    }
}

impl Layer for Attention {
    fn forward(&self, g: &mut GraphBuilder, x: Tid, ctx: &Ctx) -> Result<Tid> {
        if let Some(indexer) = &self.sparse {
            return sparse_attention_graph(g, self, indexer, x, ctx);
        }
        let (q, k, v, fused_gate) = self.prepare(g, x, ctx);
        let a = match self.kv_source {
            Some(_) => g.push(self.attention_op(ctx.layer, None), &[q]),
            None => g.push(self.attention_op(ctx.layer, None), &[q, k, v]),
        };
        Ok(self.finish(g, a, x, fused_gate, ctx))
    }

    fn weights(&self) -> Vec<Wid> {
        let mut w = self.qkv.weights();
        w.extend(self.q_norm.iter().flat_map(HeadNorm::weights));
        w.extend(self.k_norm.iter().flat_map(HeadNorm::weights));
        w.extend(self.v_norm.iter().flat_map(HeadNorm::weights));
        w.extend(self.out.weights());
        w.extend(self.gate.iter().flat_map(AttnGate::weights));
        w.extend(self.sinks);
        w.extend(self.sub_norm.iter().flat_map(Layer::weights));
        w.extend(self.sparse.iter().flat_map(DsaIndexer::weights));
        w.extend(self.rope.iter().filter_map(|r| r.factors));
        w
    }

    fn states(&self) -> Vec<StateNeed> {
        if self.kv_source.is_some() {
            return self.sparse.iter().flat_map(DsaIndexer::states).collect();
        }
        let mut s = vec![StateNeed::KvCache { n_head_kv: self.n_head_kv, head_dim_k: self.head_dim, head_dim_v: self.head_dim_v, window: self.window.map(|w| w.size) }];
        s.extend(self.sparse.iter().flat_map(DsaIndexer::states));
        s
    }
}

use crate::util::{fill, per_head_rms, rope};
use kf_core::{Candidate, IndexerKind, PoolKeys};

/// Index tensor and its width from an indexer, keys from x and queries from q_in, or a source layer's selection
pub fn indexer_select(g: &mut GraphBuilder, idx: &DsaIndexer, x: Tid, q_in: Tid, k_rows: Option<Tid>, keys_from: Option<usize>, candidate: Option<Candidate>, ctx: &Ctx) -> (Tid, usize) {
    let (d, h, t) = (idx.head_dim, idx.n_head, ctx.n_tokens);
    let width = match idx.kind {
        IndexerKind::Dense | IndexerKind::Compressed { .. } => idx.top_k,
        IndexerKind::Pool { kpool, .. } => idx.top_k + kpool - 1,
        IndexerKind::Block { block, .. } => h * idx.top_k * block,
    };
    let op = |source: Option<usize>, pool: Option<PoolKeys>| Op::IndexerTopK { layer: ctx.layer, n_head: h, head_dim: d, top_k: idx.top_k, kind: idx.kind, source, keys_from, pool, candidate };
    if let Some(src) = idx.source {
        return (g.push(op(Some(src), None), &[x]), width);
    }
    let pooled = matches!(idx.kind, IndexerKind::Pool { .. });
    let pool_ape = pooled && idx.pool_ape.is_some();
    let offset = |p: &kf_core::RopeParams| if idx.rope_tail { d - p.dims.min(d) } else { 0 };
    let eps = idx.k_norm.as_ref().map_or(1e-6, |n| n.eps);
    let mut q = idx.q.apply(g, q_in);
    if !pool_ape {
        if let Some(qn) = idx.q_norm {
            q = per_head_rms(g, q, t, h, d, Some(qn), eps);
        }
        if let Some(rp) = &idx.rope {
            q = rope(g, q, rp, h, d, offset(rp), ctx.positions);
        }
    }
    let w = match &idx.proj {
        Some(p) => {
            let w = p.apply(g, x);
            g.scale(w, 1.0 / ((d * h) as f32).sqrt())
        }
        None => fill(g, t, h, 1.0 / (d as f32).sqrt()),
    };
    let mut inputs = vec![q];
    let mut pool = None;
    match (k_rows, idx.kind) {
        (Some(rows), _) => inputs.push(rows),
        (None, IndexerKind::Compressed { .. }) => inputs.push(fill(g, 0, d, 0.0)),
        (None, _) => {
            let (Some(kp), Some(kn)) = (&idx.k, &idx.k_norm) else { return (fill(g, t, width, -1.0), width) };
            let k = kp.apply(g, x);
            if pooled && !pool_ape {
                inputs.push(k);
                pool = Some(PoolKeys { ape: None, norm: Some(kn.w), eps: kn.eps, rope: idx.rope.clone() });
            } else {
                let mut k = match kn.kind {
                    NormKind::Rms => g.push(Op::RmsNorm { weight: Some(kn.w), bias: kn.b, eps: kn.eps }, &[k]),
                    NormKind::Layer => g.push(Op::LayerNorm { weight: Some(kn.w), bias: kn.b, eps: kn.eps }, &[k]),
                };
                if !pool_ape && let Some(rp) = &idx.rope {
                    k = rope(g, k, rp, 1, d, offset(rp), ctx.positions);
                }
                inputs.push(k);
                if pool_ape {
                    pool = Some(PoolKeys { ape: idx.pool_ape, norm: None, eps: kn.eps, rope: None });
                }
            }
        }
    }
    inputs.push(w);
    if pool_ape && let Some(pg) = &idx.pool_gate {
        inputs.push(pg.apply(g, x));
    }
    (g.push(op(None, pool), &inputs), width)
}

/// Dense attention restricted to the cells an indexer selects
pub fn sparse_attention_graph(g: &mut GraphBuilder, attn: &Attention, idx: &DsaIndexer, x: Tid, ctx: &Ctx) -> Result<Tid> {
    let (q, k, v, fused_gate) = attn.prepare(g, x, ctx);
    let (sel, width) = indexer_select(g, idx, x, x, None, None, None, ctx);
    let a = match attn.kv_source {
        Some(_) => g.push(attn.attention_op(ctx.layer, Some(width)), &[q, sel]),
        None => g.push(attn.attention_op(ctx.layer, Some(width)), &[q, k, v, sel]),
    };
    Ok(attn.finish(g, a, x, fused_gate, ctx))
}

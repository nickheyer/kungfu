use crate::attention::{AttnGate, AttnTemp};
use crate::basic::{Linear, NormKind, RmsNorm};
use anyhow::Result;
use kf_core::{Ctx, GraphBuilder, IndexerKind, Layer, RopeParams, StateNeed, Tid, Wid, Window};

pub enum MlaQ {
    Full(Linear),
    LowRank { a: Linear, a_norm: RmsNorm, b: Linear },
}

impl MlaQ {
    pub fn weights(&self) -> Vec<Wid> {
        match self {
            MlaQ::Full(l) => l.weights(),
            MlaQ::LowRank { a, a_norm, b } => [a.weights(), a_norm.weights(), b.weights()].concat(),
        }
    }
}

/// Norm on indexer keys, a layer norm with bias for DeepSeek and an rms norm for Qwen
#[derive(Clone, Copy, Debug)]
pub struct IndexerNorm {
    pub kind: NormKind,
    pub w: Wid,
    pub b: Option<Wid>,
    pub eps: f32,
}

/// Lightning indexer that scores keys so attention reads only the top cells
pub struct DsaIndexer {
    pub n_head: usize,
    pub head_dim: usize,
    pub top_k: usize,
    pub kind: IndexerKind,
    /// Per head weighting of the scores, absent means the constant 1 over sqrt of the head dim
    pub proj: Option<Linear>,
    /// Key projection of the hidden state, absent where the keys come from compressed rows
    pub k: Option<Linear>,
    pub k_norm: Option<IndexerNorm>,
    pub q: Linear,
    pub q_norm: Option<Wid>,
    /// Rope over the first dims of every indexer head, or the trailing dims when rope_tail is set
    pub rope: Option<RopeParams>,
    pub rope_tail: bool,
    pub hadamard: bool,
    /// Pool gate and ape of a pooled indexer
    pub pool_gate: Option<Linear>,
    pub pool_ape: Option<Wid>,
    /// Layer whose selection this layer reuses
    pub source: Option<usize>,
}

impl DsaIndexer {
    pub fn weights(&self) -> Vec<Wid> {
        let mut w: Vec<Wid> = self.proj.iter().flat_map(Layer::weights).collect();
        w.extend(self.k.iter().flat_map(Layer::weights));
        if let Some(n) = &self.k_norm {
            w.push(n.w);
            w.extend(n.b);
        }
        w.extend(self.q.weights());
        w.extend(self.q_norm);
        w.extend(self.pool_gate.iter().flat_map(Layer::weights));
        w.extend(self.pool_ape);
        w
    }
    pub fn states(&self) -> Vec<StateNeed> {
        if self.source.is_some() {
            return Vec::new();
        }
        let dim = match self.kind {
            IndexerKind::Pool { .. } => 3 * self.head_dim,
            _ => self.head_dim,
        };
        vec![StateNeed::IndexerCache { dim }]
    }
}

/// Multi head latent attention with a compressed kv cache, DeepSeek and GLM style
pub struct Mla {
    pub n_head: usize,
    pub qk_nope_dim: usize,
    pub qk_rope_dim: usize,
    pub v_head_dim: usize,
    pub kv_lora_rank: usize,
    pub q: MlaQ,
    pub kv_a_mqa: Linear,
    pub kv_a_norm: RmsNorm,
    /// Legacy decompression to a full per head cache
    pub kv_b: Option<Linear>,
    pub k_b: Option<Wid>,
    pub v_b: Option<Wid>,
    /// Norm on the shared rope key before rope
    pub k_pe_norm: Option<RmsNorm>,
    /// Legacy keys hold the rope part first
    pub legacy_pe_first: bool,
    pub out: Linear,
    pub rope: Option<RopeParams>,
    pub scale: f32,
    pub temp: Option<AttnTemp>,
    pub gate: Option<AttnGate>,
    pub sinks: Option<Wid>,
    pub indexer: Option<DsaIndexer>,
    pub window: Option<Window>,
}

impl Layer for Mla {
    fn forward(&self, g: &mut GraphBuilder, x: Tid, ctx: &Ctx) -> Result<Tid> {
        mla_attention_graph(g, self, x, ctx)
    }
    fn weights(&self) -> Vec<Wid> {
        let mut w = self.q.weights();
        w.extend(self.kv_a_mqa.weights());
        w.extend(self.kv_a_norm.weights());
        w.extend(self.kv_b.iter().flat_map(Layer::weights));
        w.extend(self.k_b);
        w.extend(self.v_b);
        w.extend(self.k_pe_norm.iter().flat_map(Layer::weights));
        w.extend(self.out.weights());
        w.extend(self.gate.iter().flat_map(AttnGate::weights));
        w.extend(self.sinks);
        w.extend(self.indexer.iter().flat_map(DsaIndexer::weights));
        w.extend(self.rope.iter().filter_map(|r| r.factors));
        w
    }
    fn states(&self) -> Vec<StateNeed> {
        let mut s = vec![match self.kv_b {
            Some(_) => StateNeed::KvCache { n_head_kv: self.n_head, head_dim_k: self.qk_nope_dim + self.qk_rope_dim, head_dim_v: self.v_head_dim, window: self.window.map(|w| w.size) },
            None => StateNeed::Latent { rank: self.kv_lora_rank, rope_dim: self.qk_rope_dim, window: self.window.map(|w| w.size) },
        }];
        s.extend(self.indexer.iter().flat_map(DsaIndexer::states));
        s
    }
}

use crate::attention::indexer_select;
use crate::util::{concat_heads, head_cols, rope};
use anyhow::bail;
use kf_core::Op;

/// Latent attention, absorbed against the latent cache or decompressed through the legacy kv_b
pub fn mla_attention_graph(g: &mut GraphBuilder, m: &Mla, x: Tid, ctx: &Ctx) -> Result<Tid> {
    let t = ctx.n_tokens;
    let h = m.n_head;
    let (nope, r, dv, l) = (m.qk_nope_dim, m.qk_rope_dim, m.v_head_dim, m.kv_lora_rank);
    let hd = nope + r;
    let (q, qr) = match &m.q {
        MlaQ::Full(lin) => (lin.apply(g, x), x),
        MlaQ::LowRank { a, a_norm, b } => {
            let qr = a.apply(g, x);
            let qr = a_norm.apply(g, qr);
            (b.apply(g, qr), qr)
        }
    };
    let c = m.kv_a_mqa.apply(g, x);
    let kv_c = g.cols(c, 0, l);
    let kv_c = m.kv_a_norm.apply(g, kv_c);
    let mut q = q;
    let mut k_pe = (r > 0).then(|| g.cols(c, l, r));
    if let (Some(n), Some(kp)) = (&m.k_pe_norm, k_pe) {
        k_pe = Some(n.apply(g, kp));
    }
    if let Some(rp) = &m.rope {
        q = rope(g, q, rp, h, hd, nope, ctx.positions);
        if let Some(kp) = k_pe {
            k_pe = Some(rope(g, kp, rp, 1, r, 0, ctx.positions));
        }
    }
    if let Some(tp) = &m.temp {
        q = g.push(Op::AttnTemp { floor_scale: tp.floor_scale, scale: tp.scale, offset: tp.offset, pow: tp.pow }, &[q, ctx.positions]);
    }
    let sel = m.indexer.as_ref().map(|idx| indexer_select(g, idx, x, qr, None, None, None, ctx));
    let q_nope = head_cols(g, q, t, h, hd, 0, nope);
    let q_pe = (r > 0).then(|| head_cols(g, q, t, h, hd, nope, r));
    let attn = |layer: usize, n_head_kv: usize, head_dim: usize, head_dim_v: usize| Op::Attention {
        layer,
        n_head: h,
        n_head_kv,
        head_dim,
        head_dim_v,
        scale: m.scale,
        window: m.window,
        softcap: 0.0,
        sinks: m.sinks,
        alibi_bias: 0.0,
        kv_source: None,
        top_k: sel.map(|s| s.1),
    };
    let mut a = match (m.k_b, m.v_b, &m.kv_b) {
        (Some(kb), Some(vb), _) => {
            let q_lat = g.push(Op::MatMulHeads { weight: kb }, &[q_nope]);
            let qq = match q_pe {
                Some(qp) => concat_heads(g, q_lat, qp, t, h, l, r),
                None => q_lat,
            };
            let kk = match k_pe {
                Some(kp) => g.concat(kv_c, kp),
                None => kv_c,
            };
            let mut inputs = vec![qq, kk, kv_c];
            inputs.extend(sel.map(|s| s.0));
            let o = g.push(attn(ctx.layer, 1, l + r, l), &inputs);
            g.push(Op::MatMulHeads { weight: vb }, &[o])
        }
        (_, _, Some(kvb)) => {
            let kv = kvb.apply(g, kv_c);
            let kvh = g.reshape(kv, t * h, nope + dv);
            let k_nope = g.cols(kvh, 0, nope);
            let v = g.cols(kvh, nope, dv);
            let v = g.reshape(v, t, h * dv);
            let qh = g.reshape(q_nope, t * h, nope);
            let (kh, qh) = match (k_pe, q_pe) {
                (Some(kp), Some(qp)) => {
                    let kp = g.push(Op::Tile { n: h }, &[kp]);
                    let kp = g.reshape(kp, t * h, r);
                    let qp = g.reshape(qp, t * h, r);
                    if m.legacy_pe_first { (g.concat(kp, k_nope), g.concat(qp, qh)) } else { (g.concat(k_nope, kp), g.concat(qh, qp)) }
                }
                _ => (k_nope, qh),
            };
            let k = g.reshape(kh, t, h * hd);
            let qq = g.reshape(qh, t, h * hd);
            let mut inputs = vec![qq, k, v];
            inputs.extend(sel.map(|s| s.0));
            g.push(attn(ctx.layer, h, hd, dv), &inputs)
        }
        _ => bail!("latent attention needs attn_k_b and attn_v_b or the legacy attn_kv_b"),
    };
    if let Some(gate) = &m.gate {
        let gv = match &gate.proj {
            Some(p) => p.apply(g, x),
            None => bail!("latent attention gate without a projection"),
        };
        a = gate.apply(g, a, gv, t, h, dv);
    }
    Ok(m.out.apply(g, a))
}

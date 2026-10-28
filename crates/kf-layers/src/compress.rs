use crate::basic::{Linear, RmsNorm};
use crate::mla::DsaIndexer;
use anyhow::Result;
use kf_core::{Ctx, GraphBuilder, Layer, RopeParams, StateNeed, Tid, Wid};

/// Two level block selection of DeepSeek V4.1 index sources
#[derive(Clone, Copy, Debug)]
pub struct CandidateBlocks {
    pub source: usize,
    pub top_blocks: usize,
    pub block: usize,
}

/// Overlap compressor of the indexer keys with its own kv, gate, ape and norm
pub struct IndexCompressor {
    pub kv: Linear,
    pub gate: Linear,
    pub ape: Wid,
    pub norm: Wid,
}

/// DeepSeek V4 attention over a raw window plus softmax compressed key rows, K equals V
pub struct CompressedAttention {
    pub n_head: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub compress_ratio: usize,
    pub overlap: bool,
    pub q_a: Linear,
    pub q_a_norm: RmsNorm,
    pub q_b: Linear,
    /// Weightless rms norm of every q head, V4 only
    pub q_head_norm: bool,
    pub kv: Linear,
    pub kv_norm: RmsNorm,
    pub compressor_kv: Option<Linear>,
    pub compressor_gate: Option<Linear>,
    pub compressor_ape: Option<Wid>,
    pub compressor_norm: Option<Wid>,
    /// Grouped low rank output, out_a stacked per group then out_b
    pub out_a: Wid,
    pub out_b: Linear,
    pub o_groups: usize,
    pub o_lora_rank: usize,
    pub sinks: Wid,
    /// Index key projection and norm of a kv source layer, applied to each compressed row
    pub index_key: Option<(Linear, Wid)>,
    /// Compressor of the index keys over the hidden state, V4 ratio four layers
    pub index_compressor: Option<IndexCompressor>,
    pub index_dim: usize,
    pub indexer: Option<DsaIndexer>,
    /// Layer whose compressed rows this layer reads
    pub kv_source: Option<usize>,
    /// Layer whose top k this layer reuses
    pub topk_source: Option<usize>,
    pub candidate: Option<CandidateBlocks>,
    pub scale: f32,
    pub rope: RopeParams,
    pub window: usize,
    pub eps: f32,
}

impl Layer for CompressedAttention {
    fn forward(&self, g: &mut GraphBuilder, x: Tid, ctx: &Ctx) -> Result<Tid> {
        compressed_attention_graph(g, self, x, ctx)
    }
    fn weights(&self) -> Vec<Wid> {
        let mut w = self.q_a.weights();
        w.extend(self.q_a_norm.weights());
        w.extend(self.q_b.weights());
        w.extend(self.kv.weights());
        w.extend(self.kv_norm.weights());
        w.extend(self.compressor_kv.iter().flat_map(Layer::weights));
        w.extend(self.compressor_gate.iter().flat_map(Layer::weights));
        w.extend([self.compressor_ape, self.compressor_norm].into_iter().flatten());
        w.push(self.out_a);
        w.extend(self.out_b.weights());
        w.push(self.sinks);
        if let Some((k, n)) = &self.index_key {
            w.extend(k.weights());
            w.push(*n);
        }
        if let Some(ic) = &self.index_compressor {
            w.extend(ic.kv.weights());
            w.extend(ic.gate.weights());
            w.extend([ic.ape, ic.norm]);
        }
        w.extend(self.indexer.iter().flat_map(DsaIndexer::weights));
        w
    }
    fn states(&self) -> Vec<StateNeed> {
        let mut s = vec![StateNeed::KvCache { n_head_kv: 1, head_dim_k: self.head_dim, head_dim_v: 0, window: Some(self.window) }];
        if self.compress_ratio > 0 && self.kv_source.is_none() {
            let dim = self.head_dim + if self.index_key.is_some() { self.index_dim } else { 0 };
            s.push(StateNeed::Compressed { dim, ratio: self.compress_ratio });
        }
        if self.index_compressor.is_some() {
            s.push(StateNeed::Compressed { dim: self.index_dim, ratio: self.compress_ratio });
        }
        s
    }
}

use crate::attention::indexer_select;
use crate::util::{fill, join, per_head_rms, rope};
use anyhow::bail;
use kf_core::{Candidate, Compressor, IndexKey, IndexerKind, Op};

/// Low rank q and a single roped kv head over the raw window and the compressed rows, grouped low rank output
pub fn compressed_attention_graph(g: &mut GraphBuilder, c: &CompressedAttention, x: Tid, ctx: &Ctx) -> Result<Tid> {
    let t = ctx.n_tokens;
    let (h, d, r) = (c.n_head, c.head_dim, c.rope_dim);
    let ratio = c.compress_ratio;
    let qr = c.q_a.apply(g, x);
    let qr = c.q_a_norm.apply(g, qr);
    let mut q = c.q_b.apply(g, qr);
    if c.q_head_norm {
        q = per_head_rms(g, q, t, h, d, None, c.eps);
    }
    q = rope(g, q, &c.rope, h, d, d - r, ctx.positions);
    let kv = c.kv.apply(g, x);
    let kv = c.kv_norm.apply(g, kv);
    let kv = rope(g, kv, &c.rope, 1, d, d - r, ctx.positions);
    let mut index_rows = None;
    if ratio > 0 && c.kv_source.is_none() {
        let width = if c.overlap { 2 * d } else { d };
        let ckv = c.compressor_kv.as_ref().ok_or_else(|| anyhow::anyhow!("compressed attention owner without a compressor kv projection"))?.apply(g, x);
        let score = match &c.compressor_gate {
            Some(gt) => gt.apply(g, x),
            None => fill(g, t, width, 0.0),
        };
        let norm = c.compressor_norm.ok_or_else(|| anyhow::anyhow!("compressed attention owner without a compressor norm"))?;
        let index = c.index_key.as_ref().map(|(k, n)| IndexKey { k: k.w, norm: *n, eps: c.eps, rope: c.rope.clone(), head_dim: c.index_dim });
        let comp = Compressor { ratio, overlap: c.overlap, ape: c.compressor_ape, norm, eps: c.eps };
        g.push(Op::CompressKv { layer: ctx.layer, slot: 0, comp, rope: c.rope.clone(), head_dim: d, index }, &[ckv, score]);
        if let Some(ic) = &c.index_compressor {
            let ikv = ic.kv.apply(g, x);
            let isc = ic.gate.apply(g, x);
            let comp = Compressor { ratio, overlap: true, ape: Some(ic.ape), norm: ic.norm, eps: c.eps };
            index_rows = Some(g.push(Op::CompressKv { layer: ctx.layer, slot: 1, comp, rope: c.rope.clone(), head_dim: c.index_dim, index: None }, &[ikv, isc]));
        }
    }
    let candidate = c.candidate.map(|cb| Candidate { source: (cb.source != ctx.layer).then_some(cb.source), top_blocks: cb.top_blocks, block: cb.block });
    let sel = if ratio == 0 {
        None
    } else {
        match (&c.indexer, c.topk_source) {
            (Some(idx), _) => Some(indexer_select(g, idx, x, qr, index_rows, c.kv_source, candidate, ctx)),
            (None, Some(src)) => {
                let width = c.index_top_k();
                let op = Op::IndexerTopK { layer: ctx.layer, n_head: 1, head_dim: c.index_dim, top_k: width, kind: IndexerKind::Compressed { ratio, overlap: c.overlap }, source: Some(src), keys_from: None, pool: None, candidate: None };
                Some((g.push(op, &[x]), width))
            }
            (None, None) => None,
        }
    };
    let op = Op::CompressedAttention {
        layer: ctx.layer,
        n_head: h,
        head_dim: d,
        scale: c.scale,
        window: c.window,
        sinks: c.sinks,
        rope: c.rope.clone(),
        ratio,
        kv_source: c.kv_source,
        top_k: sel.map(|s| s.1),
    };
    let mut inputs = vec![q, kv];
    inputs.extend(sel.map(|s| s.0));
    let o = g.push(op, &inputs);
    if c.o_groups == 0 || !h.is_multiple_of(c.o_groups) {
        bail!("{} heads over {} output groups", h, c.o_groups);
    }
    let gd = h / c.o_groups * d;
    let parts: Vec<Tid> = (0..c.o_groups)
        .map(|gi| {
            let og = g.cols(o, gi * gd, gd);
            g.push(Op::MatMulSlice { weight: c.out_a, index: gi }, &[og])
        })
        .collect();
    let u = join(g, &parts);
    let _ = c.o_lora_rank;
    Ok(c.out_b.apply(g, u))
}

impl CompressedAttention {
    /// Selection width a shared layer reads from its top k source
    fn index_top_k(&self) -> usize {
        self.indexer.as_ref().map_or(self.index_dim, |i| i.top_k)
    }
}

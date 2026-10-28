use crate::compose::*;
use crate::{Architecture, Hparams, Keys, Weights};
use anyhow::{Result, bail};
use kf_core::{Gating, IndexerKind, RopeKind, RopeParams, WeightNorm, Yarn};
use kf_gguf::Metadata;
use kf_layers::{Block, CandidateBlocks, CompressedAttention, DsaIndexer, Engram, HcHead, HcKind, HcMix, HyperConnected, Linear, Model};

/// DeepSeek V4.1 Flash, single pass hyper connections, compressed attention shared across layers, n gram memory
pub struct DeepSeek41;

impl Architecture for DeepSeek41 {
    fn id(&self) -> &'static str {
        "deepseek41"
    }

    fn hparams(&self, m: &Metadata) -> Result<Hparams> {
        let mut hp = Hparams::common(m, self.id())?;
        let k = Keys::new(m, self.id());
        hp.rope.kind = RopeKind::Normal;
        hp.need_rms_eps()?;
        hp.need_attention()?;
        hp.need_moe()?;
        hp.need_hc()?;
        hp.need_indexer()?;
        if hp.hc.count != 4 {
            bail!("{}: hyper_connection.count must be 4", hp.arch);
        }
        if hp.gating != Some(Gating::SqrtSoftplus) {
            bail!("{}: expert_gating_func must be 4", hp.arch);
        }
        if hp.mla.q_lora == 0 || hp.swa_window.is_none() || hp.dsv4.o_groups == 0 || hp.dsv4.o_lora_rank == 0 || hp.dsv4.compress_rope_base == 0.0 {
            bail!("{}: q_lora_rank, sliding_window, output_group_count, output_lora_rank and compress_rope_freq_base are required", hp.arch);
        }
        if hp.dsv4.compress_ratios.len() < hp.n_layer_all || hp.dsv4.compress_ratios.iter().any(|r| *r > 2) {
            bail!("{}: attention.compress_ratios must list every layer with values 0, 1 or 2", hp.arch);
        }
        if !k.has("swiglu_clamp_shexp") {
            hp.swiglu_clamp_shexp = hp.swiglu_clamp_exp.clone();
        }
        if hp.swiglu_clamp_exp.iter().all(|v| *v == 0.0) && !k.has("swiglu_clamp_exp") {
            bail!("{}: swiglu_clamp_exp is required", hp.arch);
        }
        if !hp.ple.layers.is_empty() {
            hp.need_ple()?;
            if hp.dsv4.engram_token_map.is_empty() || hp.ple.embd_per_layer == 0 {
                bail!("{}: engram.token_map and embedding_length_per_layer_input are required", hp.arch);
            }
        }
        hp.swa = vec![true; hp.n_layer_all];
        Ok(hp)
    }

    fn build(&self, hp: &Hparams, w: &mut Weights) -> Result<Model> {
        let (embed, out) = embed_and_head(w, false)?;
        let final_norm = rms_named(w, "output_norm", hp.rms_eps)?;
        let sources = resolve_sources(hp)?;
        let engram_table = if hp.ple.layers.is_empty() { None } else { Some(w.take("per_layer_token_embd.weight")?) };
        let mut blocks = Vec::with_capacity(hp.n_layer);
        for l in 0..hp.n_layer {
            let mut block = Block::new(l);
            if let (Some(table), Some(e)) = (engram_table, hp.ple.layers.iter().position(|x| *x == l)) {
                block = block.then(engram(hp, w, l, e, table)?);
            }
            let attn = compressed_attention(hp, w, l, &sources)?;
            block = block.then(HyperConnected {
                n_streams: hp.hc.count,
                n_embd: hp.n_embd,
                kind: HcKind::Sinkhorn { iters: hp.hc.sinkhorn, eps: hp.hc.epsilon },
                attn_mix: hc_mix(w, &format!("blk.{l}.hc_attn")),
                ffn_mix: hc_mix(w, &format!("blk.{l}.hc_ffn")),
                attn_norm: Box::new(rms(w, l, "attn_norm", hp.rms_eps)?),
                attn: Box::new(attn),
                ffn_norm: Box::new(rms(w, l, "ffn_norm", hp.rms_eps)?),
                ffn: Box::new(experts(hp, w, l)?),
                single_pass: true,
                eps: hp.rms_eps,
            });
            blocks.push(block);
        }
        for l in hp.n_layer..hp.n_layer_all {
            w.claim_prefix(&format!("blk.{l}."));
        }
        let mut model = model(embed, blocks, final_norm, out);
        model.streams = hp.hc.count;
        model.pre_head = Some(Box::new(HcHead {
            n_streams: hp.hc.count,
            n_embd: hp.n_embd,
            kind: HcKind::Sinkhorn { iters: hp.hc.sinkhorn, eps: hp.hc.epsilon },
            mix: HcMix { fn_: None, base: None, scale: None, norm: None, down: None, up: None, inject: None, down_inject: None },
            eps: hp.rms_eps,
        }));
        finish(hp, w, model)
    }
}

/// Per layer kv and selection sources, the latest listed source at or before each compressed layer
struct Sources {
    kv: Vec<Option<usize>>,
    topk: Vec<Option<usize>>,
}

fn resolve_sources(hp: &Hparams) -> Result<Sources> {
    let latest = |set: &[usize], l: usize| set.iter().copied().filter(|s| *s <= l).max();
    let mut kv = Vec::with_capacity(hp.n_layer_all);
    let mut topk = Vec::with_capacity(hp.n_layer_all);
    for l in 0..hp.n_layer_all {
        let ratio = hp.dsv4.compress_ratios[l];
        if ratio == 0 {
            kv.push(None);
            topk.push(None);
            continue;
        }
        let ks = latest(&hp.dsv4.kv_source, l).ok_or_else(|| anyhow::anyhow!("{}: layer {l} has no kv source", hp.arch))?;
        let ts = latest(&hp.dsv4.topk_source, l).ok_or_else(|| anyhow::anyhow!("{}: layer {l} has no index source", hp.arch))?;
        if ts < ks || hp.dsv4.compress_ratios[ks] != ratio || hp.dsv4.compress_ratios[ts] != ratio {
            bail!("{}: layer {l} sources {ks} and {ts} disagree on the compress ratio or order", hp.arch);
        }
        kv.push(Some(ks));
        topk.push(Some(ts));
    }
    Ok(Sources { kv, topk })
}

fn layer_rope(hp: &Hparams, ratio: usize) -> RopeParams {
    let yarn = hp.rope_scaling == crate::hparams::RopeScaling::Yarn;
    if ratio == 0 {
        return RopeParams { kind: RopeKind::Normal, dims: hp.rope.dims, base: hp.rope.base, scale: 1.0, orig_ctx: 0, attn_factor: 1.0, yarn: None, sections: [0; 4], factors: None };
    }
    RopeParams {
        kind: RopeKind::Normal,
        dims: hp.rope.dims,
        base: hp.dsv4.compress_rope_base,
        scale: hp.rope.scale,
        orig_ctx: hp.rope.orig_ctx,
        attn_factor: if yarn { 1.0 / (1.0 + 0.1 * (1.0 / hp.rope.scale).ln()) } else { 1.0 },
        yarn: yarn.then(|| Yarn { ext_factor: 1.0, beta_fast: hp.yarn_beta_fast, beta_slow: hp.yarn_beta_slow }),
        sections: [0; 4],
        factors: None,
    }
}

fn compressed_attention(hp: &Hparams, w: &mut Weights, l: usize, src: &Sources) -> Result<CompressedAttention> {
    let ratio = hp.dsv4.compress_ratios[l];
    let owns_kv = ratio != 0 && src.kv[l] == Some(l);
    let owns_topk = ratio != 0 && src.topk[l] == Some(l);
    let rope = layer_rope(hp, ratio);
    let d = hp.head_dim_k;
    let indexer = if owns_topk {
        Some(DsaIndexer {
            n_head: hp.indexer.n_head,
            head_dim: hp.indexer.head_dim,
            top_k: hp.indexer.top_k,
            kind: IndexerKind::Compressed { ratio, overlap: false },
            proj: Some(linear(w, l, "indexer.proj")?),
            k: None,
            k_norm: None,
            q: linear(w, l, "indexer.attn_q_b")?,
            q_norm: None,
            rope: Some(rope.clone()),
            rope_tail: true,
            hadamard: false,
            pool_gate: None,
            pool_ape: None,
            source: None,
        })
    } else {
        None
    };
    let candidate = hp.dsv4.candidate_source.filter(|c| owns_topk && l >= *c).map(|c| CandidateBlocks { source: c, top_blocks: hp.dsv4.candidate_top_blocks, block: hp.dsv4.candidate_block });
    Ok(CompressedAttention {
        n_head: hp.n_head,
        head_dim: d,
        rope_dim: hp.rope.dims,
        compress_ratio: ratio,
        overlap: false,
        q_a: linear(w, l, "attn_q_a")?,
        q_a_norm: rms(w, l, "attn_q_a_norm", hp.rms_eps)?,
        q_b: linear(w, l, "attn_q_b")?,
        q_head_norm: false,
        kv: linear(w, l, "attn_kv")?,
        kv_norm: rms(w, l, "attn_kv_a_norm", hp.rms_eps)?,
        compressor_kv: if owns_kv { Some(linear(w, l, "attn_compressor_kv")?) } else { None },
        compressor_gate: if owns_kv && ratio > 1 { Some(linear(w, l, "attn_compressor_gate")?) } else { None },
        compressor_ape: None,
        compressor_norm: if owns_kv { Some(w.take(&blk(l, "attn_compressor_norm"))?) } else { None },
        out_a: w.take(&blk(l, "attn_output_a"))?,
        out_b: linear(w, l, "attn_output_b")?,
        o_groups: hp.dsv4.o_groups,
        o_lora_rank: hp.dsv4.o_lora_rank,
        sinks: w.take(&blk(l, "attn_sinks"))?,
        index_key: if owns_kv { Some((Linear::new(w.take(&blk(l, "indexer.attn_k"))?), w.take(&blk(l, "indexer.k_norm"))?)) } else { None },
        index_compressor: None,
        index_dim: hp.indexer.head_dim,
        indexer,
        kv_source: src.kv[l].filter(|s| *s != l),
        topk_source: src.topk[l].filter(|s| *s != l),
        candidate,
        scale: 1.0 / (d as f32).sqrt(),
        rope,
        window: hp.swa_window.unwrap_or(0),
        eps: hp.rms_eps,
    })
}

fn experts(hp: &Hparams, w: &mut Weights, l: usize) -> Result<kf_layers::Moe> {
    moe(hp, w, l, MoeOpts {
        gating: Gating::SqrtSoftplus,
        norm_weights: if hp.expert_norm { WeightNorm::AddEps } else { WeightNorm::None },
        scale: hp.expert_scale,
        shared: SharedOpt::Plain,
        probs_bias: true,
        probs_bias_required: true,
        limit: clamp_limit(&hp.swiglu_clamp_exp, l, true),
        shared_limit: clamp_limit(&hp.swiglu_clamp_shexp, l, true),
        ..Default::default()
    })
}

fn engram(hp: &Hparams, w: &mut Weights, l: usize, e: usize, table: kf_core::Wid) -> Result<Engram> {
    let n = hp.ple.ngram;
    let n_cols = (n - 1) * hp.ple.heads_per_ngram;
    let slice = |v: &[u64], start: usize, len: usize| -> Result<Vec<u64>> {
        v.get(start..start + len).map(<[u64]>::to_vec).ok_or_else(|| anyhow::anyhow!("{}: engram table shorter than module {e} needs", hp.arch))
    };
    let offsets: Vec<u64> = hp.ple.head_offsets.iter().map(|v| *v as u64).collect();
    let primes: Vec<u64> = hp.ple.head_vocab.iter().map(|v| *v as u64).collect();
    Ok(Engram {
        ngram_size: n,
        heads_per_ngram: hp.ple.heads_per_ngram,
        head_dim: hp.ple.embd_per_layer,
        n_streams: hp.hc.count,
        n_embd: hp.n_embd,
        multipliers: slice(&hp.ple.multipliers, e * n, n)?,
        offsets: slice(&offsets, e * n_cols, n_cols)?.into_iter().map(|v| v as u32).collect(),
        primes: slice(&primes, e * n_cols, n_cols)?.into_iter().map(|v| v as u32).collect(),
        token_map: hp.dsv4.engram_token_map.clone(),
        pad_token: hp.dsv4.engram_pad,
        eps: hp.rms_eps,
        table,
        kv: w.take(&blk(l, "engram_kv"))?,
        gate: w.take(&blk(l, "engram_gate"))?,
    })
}

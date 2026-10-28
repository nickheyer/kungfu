use crate::archs::qwen3next::shared_moe;
use crate::compose::*;
use crate::{Architecture, Hparams, Weights};
use anyhow::{Result, bail};
use kf_core::{IndexerKind, Wid};
use kf_gguf::Metadata;
use kf_layers::{Block, HcHead, HcKind, HeadMap, HyperConnected, Identity, Layer, Linear, Model, NextN, OutGate, Ple, QkvProj, RopeKind};

/// Qwen 3.8 Flash Next, low rank hyper connections, pooled sparse attention, n gram memory and experts
pub struct Qwen4Exp;

impl Architecture for Qwen4Exp {
    fn id(&self) -> &'static str {
        "qwen4exp"
    }

    fn hparams(&self, m: &Metadata) -> Result<Hparams> {
        let mut hp = Hparams::common(m, self.id())?;
        hp.rope.kind = RopeKind::IMRope;
        if hp.rope.sections.iter().all(|s| *s == 0) {
            bail!("{}: rope.dimension_sections is required", hp.arch);
        }
        hp.need_rms_eps()?;
        hp.need_attention()?;
        hp.need_ssm()?;
        hp.need_moe()?;
        hp.need_hc()?;
        if hp.hc.low_rank == 0 {
            bail!("{}: hyper_connection.low_rank is required", hp.arch);
        }
        hp.need_indexer()?;
        hp.recurrent_pattern(m, 4);
        let ratios: Vec<usize> = hp.dsv4.compress_ratios.iter().copied().filter(|r| *r > 0).collect();
        if let Some(first) = ratios.first() {
            if ratios.iter().any(|r| r != first) || *first == 1 || !hp.indexer.top_k.is_multiple_of(*first) {
                bail!("{}: attention.compress_ratios must hold one pool size dividing indexer.top_k", hp.arch);
            }
            hp.indexer.kpool = *first;
        }
        if !hp.ple.layers.is_empty() {
            if hp.ple.layers.len() != 1 {
                bail!("{}: exactly one ple layer is allowed", hp.arch);
            }
            hp.need_ple()?;
            if hp.ple.conv_kernel == 0 || hp.ple.embd_per_layer == 0 {
                bail!("{}: ple.conv_kernel and embedding_length_per_layer_input are required", hp.arch);
            }
            if !hp.is_recurrent(hp.ple.layers[0]) {
                bail!("{}: the ple layer must be recurrent", hp.arch);
            }
        }
        Ok(hp)
    }

    fn build(&self, hp: &Hparams, w: &mut Weights) -> Result<Model> {
        let (embed, out) = embed_and_head(w, true)?;
        let main = Main { embed: embed.w, head: out, norm: kf_layers::RmsNorm::weightless(hp.rms_eps) };
        let pre_head = HcHead { n_streams: hp.hc.count, n_embd: hp.n_embd, kind: HcKind::LowRank { rank: hp.hc.low_rank }, mix: hc_mix(w, "output_hc"), eps: hp.rms_eps };
        let ple_table = if hp.ple.layers.is_empty() { None } else { Some(w.take("per_layer_token_embd.weight")?) };
        let mut blocks = Vec::with_capacity(hp.n_layer);
        for l in 0..hp.n_layer {
            let mut block = Block::new(l);
            if let (Some(table), true) = (ple_table, hp.ple.layers.contains(&l)) {
                block = block.then(ple(hp, w, l, table)?);
            }
            blocks.push(block.then(hc_block(hp, w, l)?));
        }
        let mut model = Model {
            embed,
            pre: Vec::new(),
            streams: hp.hc.count,
            blocks,
            pre_head: Some(Box::new(pre_head)),
            final_norm: Box::new(Identity),
            head: out,
            logit_scale: 1.0,
            final_softcap: 0.0,
            logit_mask: None,
            extras: Vec::new(),
            hands_off: Default::default(),
        };
        for l in hp.n_layer..hp.n_layer_all {
            if !has_nextn(w, l) {
                w.claim_prefix(&format!("blk.{l}."));
                continue;
            }
            let block = Block::new(l).then(hc_block(hp, w, l)?);
            let head_mix = HcHead { n_streams: hp.hc.count, n_embd: hp.n_embd, kind: HcKind::LowRank { rank: hp.hc.low_rank }, mix: hc_mix(w, &format!("blk.{l}.nextn.hc_head")), eps: hp.rms_eps };
            model.extras.push(Box::new(NextN {
                eh_proj: linear(w, l, "nextn.eh_proj")?,
                enorm: rms(w, l, "nextn.enorm", hp.rms_eps)?,
                hnorm: rms(w, l, "nextn.hnorm", hp.rms_eps)?,
                hnorm_per_stream: true,
                embed_tokens: main.embed,
                head: main.head,
                head_norm: Box::new(Identity),
                block,
                n_streams: hp.hc.count,
                n_embd: hp.n_embd,
                pre_head: Some(Box::new(head_mix)),
            }));
        }
        finish(hp, w, model)
    }
}

/// One hyper connected block, a delta net or pooled sparse attention mixer then the experts
fn hc_block(hp: &Hparams, w: &mut Weights, l: usize) -> Result<HyperConnected> {
    let mixer: Box<dyn Layer> = if hp.is_recurrent(l) { Box::new(delta_net(hp, w, l, HeadMap::Tiled, OutGate::Sigmoid)?) } else { Box::new(qsa_attention(hp, w, l)?) };
    let experts = shared_moe(hp, w, l)?;
    Ok(HyperConnected {
        n_streams: hp.hc.count,
        n_embd: hp.n_embd,
        kind: HcKind::LowRank { rank: hp.hc.low_rank },
        attn_mix: hc_mix(w, &format!("blk.{l}.hc_attn")),
        ffn_mix: hc_mix(w, &format!("blk.{l}.hc_ffn")),
        attn_norm: Box::new(Identity),
        attn: mixer,
        ffn_norm: Box::new(Identity),
        ffn: Box::new(experts),
        single_pass: false,
        eps: hp.rms_eps,
    })
}

/// Gated attention on separate unbiased projections, keys pooled and selected where the layer compresses
fn qsa_attention(hp: &Hparams, w: &mut Weights, l: usize) -> Result<kf_layers::Attention> {
    let q = Linear::new(w.take(&blk(l, "attn_q"))?);
    let k = Linear::new(w.take(&blk(l, "attn_k"))?);
    let v = Linear::new(w.take(&blk(l, "attn_v"))?);
    for n in ["attn_q", "attn_k", "attn_v"] {
        w.take_opt(&blk_bias(l, n));
        w.take_opt(&format!("blk.{l}.{n}.scale"));
        w.take_opt(&format!("blk.{l}.{n}.input_scale"));
    }
    let mut attn = attention_with_qkv(hp, w, l, QkvProj::Separate { q, k, v }, AttnOpts { qk_norm: QkNorm::Shared, gate: GateOpt::InQ, rope: Some(hp.rope_full()), ..Default::default() })?;
    let ratio = hp.dsv4.compress_ratios.get(l).copied().unwrap_or(0);
    let mut idx = indexer(hp, w, l, IndexerKind::Pool { kpool: hp.indexer.kpool.max(1), tail: 1 }, Some(hp.rope_full()), false, None)?;
    if idx.q_norm.is_none() {
        bail!("{}: blk.{l}.indexer.q_norm.weight is required", hp.arch);
    }
    idx.proj = None;
    attn.sparse = (ratio > 0 && hp.indexer.kpool > 0).then_some(idx);
    Ok(attn)
}

fn ple(hp: &Hparams, w: &mut Weights, l: usize, table: Wid) -> Result<Ple> {
    let n_heads = (hp.ple.ngram - 1) * hp.ple.heads_per_ngram;
    Ok(Ple {
        ngram_size: hp.ple.ngram,
        heads_per_ngram: hp.ple.heads_per_ngram,
        head_dim: hp.ple.embd_per_layer,
        n_heads,
        n_streams: hp.hc.count,
        n_embd: hp.n_embd,
        conv_kernel: hp.ple.conv_kernel,
        multipliers: hp.ple.multipliers.clone(),
        head_offsets: hp.ple.head_offsets.clone(),
        head_vocab_sizes: hp.ple.head_vocab.clone(),
        eos_token: hp.ple.eos,
        image_token: hp.ple.image,
        eps: hp.rms_eps,
        table,
        key: w.take(&blk(l, "ple_key"))?,
        value: w.take(&blk(l, "ple_value"))?,
        norm_key: w.take(&blk(l, "ple_norm_key"))?,
        norm_query: w.take(&blk(l, "ple_norm_query"))?,
        norm_conv: w.take(&blk(l, "ple_norm_conv"))?,
        conv1d: w.take(&blk(l, "ple_conv1d"))?,
    })
}

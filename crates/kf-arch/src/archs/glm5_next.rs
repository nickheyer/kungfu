use crate::compose::*;
use crate::{Architecture, Hparams, Weights};
use anyhow::{Result, bail};
use kf_core::{Gating, IndexerKind, WeightNorm};
use kf_gguf::Metadata;
use kf_layers::{Block, HcKind, HcMean, HyperConnected, Layer, Model, RopeKind};

/// GLM 5.3 Flash, sinkhorn hyper connections around Kimi delta attention and pooled sparse latent attention
pub struct Glm5Next;

impl Architecture for Glm5Next {
    fn id(&self) -> &'static str {
        "glm5-next"
    }

    fn hparams(&self, m: &Metadata) -> Result<Hparams> {
        let mut hp = Hparams::common(m, self.id())?;
        hp.rope.kind = RopeKind::Neox;
        hp.need_rms_eps()?;
        hp.need_attention()?;
        hp.need_mla()?;
        hp.need_indexer()?;
        hp.need_kda()?;
        hp.need_moe()?;
        hp.need_hc()?;
        if hp.indexer.kpool <= 1 || !hp.indexer.top_k.is_multiple_of(hp.indexer.kpool) {
            bail!("{}: attention.indexer.kpool above one dividing indexer.top_k is required", hp.arch);
        }
        if hp.gating.is_none() {
            hp.gating = Some(Gating::Sigmoid);
        }
        if hp.indexer.full.is_empty() {
            hp.indexer.full = vec![true; hp.n_layer_all];
        }
        let k = crate::hparams::Keys::new(m, self.id());
        if !k.has("swiglu_clamp_shexp") {
            hp.swiglu_clamp_shexp = hp.swiglu_clamp_exp.clone();
        }
        hp.recurrent_from_kv_heads();
        Ok(hp)
    }

    fn build(&self, hp: &Hparams, w: &mut Weights) -> Result<Model> {
        let (embed, out) = embed_and_head(w, false)?;
        let final_norm = rms_named(w, "output_norm", hp.rms_eps)?;
        let main = Main { embed: embed.w, head: out, norm: final_norm };
        let kind = IndexerKind::Pool { kpool: hp.indexer.kpool, tail: hp.indexer.kpool_tail };
        let mut blocks = Vec::with_capacity(hp.n_layer);
        let mut last_full = None;
        for l in 0..hp.n_layer {
            let mixer: Box<dyn Layer> = if hp.is_recurrent(l) {
                Box::new(kda(hp, w, l, 1e-6)?)
            } else {
                let full = hp.indexer.full.get(l).copied().unwrap_or(true);
                let sparse = if full {
                    last_full = Some(l);
                    indexer(hp, w, l, kind, None, false, None)?
                } else {
                    let src = last_full.ok_or_else(|| anyhow::anyhow!("{}: layer {l} shares an indexer before any full layer", hp.arch))?;
                    indexer_shared(hp, w, l, kind, src, embed.w)
                };
                Box::new(latent(hp, w, l, Some(sparse))?)
            };
            let ffn = feed_forward(hp, w, l)?;
            blocks.push(Block::new(l).then(HyperConnected {
                n_streams: hp.hc.count,
                n_embd: hp.n_embd,
                kind: HcKind::Sinkhorn { iters: hp.hc.sinkhorn, eps: hp.hc.epsilon },
                attn_mix: hc_mix(w, &format!("blk.{l}.hc_attn")),
                ffn_mix: hc_mix(w, &format!("blk.{l}.hc_ffn")),
                attn_norm: Box::new(rms(w, l, "attn_norm", hp.rms_eps)?),
                attn: mixer,
                ffn_norm: Box::new(rms(w, l, "ffn_norm", hp.rms_eps)?),
                ffn,
                single_pass: false,
                eps: hp.rms_eps,
            }));
        }
        let mut model = model(embed, blocks, final_norm, out);
        model.streams = hp.hc.count;
        model.pre_head = Some(Box::new(HcMean { n_streams: hp.hc.count }));
        for l in hp.n_layer..hp.n_layer_all {
            if !has_nextn(w, l) {
                w.claim_prefix(&format!("blk.{l}."));
                continue;
            }
            let sparse = indexer(hp, w, l, kind, None, false, None)?;
            let attn = latent(hp, w, l, Some(sparse))?;
            let ffn = feed_forward(hp, w, l)?;
            let block = Block::new(l).then(pre_norm(w, l, "attn_norm", hp.rms_eps, attn)?).then(pre_norm(w, l, "ffn_norm", hp.rms_eps, ffn)?);
            model.extras.push(Box::new(nextn(hp, w, l, block, 1, None, &main)?));
        }
        finish(hp, w, model)
    }
}

/// Dense feed forward on the leading layers, experts with shared experts elsewhere, both swiglu clamped
fn feed_forward(hp: &Hparams, w: &mut Weights, l: usize) -> Result<Box<dyn Layer>> {
    Ok(if l < hp.n_dense_lead {
        Box::new(ffn_with(hp, w, l, FfnOpts { limit: clamp_limit(&hp.swiglu_clamp_shexp, l, true), require_gate: true, ..Default::default() })?)
    } else {
        Box::new(moe(hp, w, l, MoeOpts {
            gating: hp.gating.unwrap_or(Gating::Sigmoid),
            norm_weights: if hp.expert_norm { WeightNorm::Clamp } else { WeightNorm::None },
            scale: hp.expert_scale,
            shared: SharedOpt::Plain,
            probs_bias: true,
            probs_bias_required: true,
            limit: clamp_limit(&hp.swiglu_clamp_exp, l, true),
            shared_limit: clamp_limit(&hp.swiglu_clamp_shexp, l, true),
            ..Default::default()
        })?)
    })
}

fn latent(hp: &Hparams, w: &mut Weights, l: usize, sparse: Option<kf_layers::DsaIndexer>) -> Result<kf_layers::Mla> {
    mla(hp, w, l, MlaOpts {
        rope: None,
        sparse,
        scale: 1.0 / (hp.mla.head_k as f32).sqrt(),
        temp: None,
        gate: GateOpt::None,
        sinks: false,
        window: None,
        k_pe_norm: false,
        legacy_pe_first: false,
        head_k: hp.mla.head_k,
        head_v: hp.mla.head_v,
        kv_lora: hp.mla.kv_lora,
    })
}

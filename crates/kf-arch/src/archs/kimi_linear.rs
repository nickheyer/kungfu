use crate::compose::*;
use crate::{Architecture, Hparams, Weights};
use anyhow::{Result, bail};
use kf_core::WeightNorm;
use kf_gguf::Metadata;
use kf_layers::{Block, Layer, Model, RopeKind};

/// Kimi Linear, delta attention layers with unroped latent attention every few layers and experts
pub struct KimiLinear;

impl Architecture for KimiLinear {
    fn id(&self) -> &'static str {
        "kimi-linear"
    }

    fn hparams(&self, m: &Metadata) -> Result<Hparams> {
        let mut hp = Hparams::common(m, self.id())?;
        hp.rope.kind = RopeKind::Neox;
        hp.need_rms_eps()?;
        hp.need_attention()?;
        hp.need_mla()?;
        hp.need_kda()?;
        hp.need_moe()?;
        if hp.gating.is_none() {
            bail!("{}: expert_gating_func is required", hp.arch);
        }
        hp.recurrent_from_kv_heads();
        Ok(hp)
    }

    fn build(&self, hp: &Hparams, w: &mut Weights) -> Result<Model> {
        let (embed, out) = embed_and_head(w, false)?;
        let final_norm = rms_named(w, "output_norm", hp.rms_eps)?;
        let mut blocks = Vec::with_capacity(hp.n_layer);
        for l in 0..hp.n_layer {
            let mixer: Box<dyn Layer> = if hp.is_recurrent(l) {
                Box::new(kda(hp, w, l, hp.rms_eps)?)
            } else {
                let legacy = w.has(&blk(l, "attn_kv_b"));
                Box::new(mla(hp, w, l, MlaOpts {
                    rope: None,
                    sparse: None,
                    scale: 1.0 / (hp.mla.head_k as f32).sqrt(),
                    temp: None,
                    gate: GateOpt::None,
                    sinks: false,
                    window: None,
                    k_pe_norm: false,
                    legacy_pe_first: legacy,
                    head_k: hp.mla.head_k,
                    head_v: hp.mla.head_v,
                    kv_lora: hp.mla.kv_lora,
                })?)
            };
            let ffn: Box<dyn Layer> = if l < hp.n_dense_lead {
                Box::new(ffn_with(hp, w, l, FfnOpts { require_gate: true, ..Default::default() })?)
            } else {
                Box::new(moe(hp, w, l, MoeOpts {
                    gating: hp.gating.unwrap_or_default(),
                    norm_weights: WeightNorm::Clamp,
                    scale: hp.expert_scale,
                    shared: SharedOpt::PlainOpt,
                    probs_bias: true,
                    probs_bias_required: true,
                    ..Default::default()
                })?)
            };
            blocks.push(Block::new(l).then(pre_norm(w, l, "attn_norm", hp.rms_eps, mixer)?).then(pre_norm(w, l, "ffn_norm", hp.rms_eps, ffn)?));
        }
        finish(hp, w, model(embed, blocks, final_norm, out))
    }
}

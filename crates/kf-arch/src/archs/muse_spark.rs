use crate::compose::*;
use crate::{Architecture, Hparams, Keys, Weights};
use anyhow::{Result, bail};
use kf_core::WeightNorm;
use kf_gguf::Metadata;
use kf_layers::{Block, Latent, Model, MulRms, RmsNorm, RopeKind, ScaledResidual};

/// Muse Spark, norm preserving residuals and latent space experts on every layer
pub struct MuseSpark;

impl Architecture for MuseSpark {
    fn id(&self) -> &'static str {
        "muse-spark"
    }

    fn hparams(&self, m: &Metadata) -> Result<Hparams> {
        let mut hp = Hparams::common(m, self.id())?;
        let k = Keys::new(m, self.id());
        hp.rope.kind = RopeKind::Normal;
        hp.need_rms_eps()?;
        hp.need_attention()?;
        hp.swa_pattern(m, 0, false);
        hp.need_swa()?;
        hp.need_moe()?;
        hp.rope_base_swa = k.req_f32("rope.freq_base_swa")?;
        hp.final_softcap = k.req_f32("final_logit_softcapping")?;
        hp.post_norm_eps = k.req_f32("post_norm_epsilon")?;
        if hp.logit_scale == 0.0 || hp.moe_latent == 0 || hp.gating.is_none() || !k.has("expert_weights_norm") || !k.has("expert_weights_scale") {
            bail!("{}: logit_scale, moe_latent_size, expert_weights_norm, expert_weights_scale and expert_gating_func are required", hp.arch);
        }
        if !k.has("attention.sliding_window_pattern") {
            bail!("{}: attention.sliding_window_pattern is required", hp.arch);
        }
        Ok(hp)
    }

    fn build(&self, hp: &Hparams, w: &mut Weights) -> Result<Model> {
        let (embed, out) = embed_and_head(w, false)?;
        let final_norm = rms_named(w, "output_norm", hp.rms_eps)?;
        let mut blocks = Vec::with_capacity(hp.n_layer);
        for l in 0..hp.n_layer {
            let swa = hp.is_swa(l);
            let attn = attention(hp, w, l, AttnOpts {
                qk_norm: QkNorm::Shared,
                gate: GateOpt::Full,
                out_head_norm: true,
                rope: swa.then(|| hp.rope_for(l)),
                window: window_for(hp, l),
                ..Default::default()
            })?;
            let latent = Latent { down: linear(w, l, "ffn_latent_down")?, up: linear(w, l, "ffn_latent_up")?, pre_norm: Some(rms(w, l, "ffn_pre_exp_norm", hp.rms_eps)?), post_norm: None };
            let post_exp = w.take(&blk(l, "ffn_post_exp_norm"))?;
            let experts = moe(hp, w, l, MoeOpts {
                gating: hp.gating.unwrap_or_default(),
                norm_weights: if hp.expert_norm { WeightNorm::Clamp } else { WeightNorm::None },
                scale: hp.expert_scale,
                probs_bias: true,
                probs_bias_required: true,
                latent: Some(latent),
                post_norm: Some((post_exp, hp.post_norm_eps)),
                ..Default::default()
            })?;
            let attn_res = ScaledResidual {
                norm: Box::new(rms(w, l, "attn_norm", hp.rms_eps)?),
                inner: Box::new(attn),
                post: Some(Box::new(RmsNorm::weightless(hp.post_norm_eps))),
                alpha: w.take(&blk(l, "attn_resid_alpha"))?,
                beta: w.take(&blk(l, "attn_resid_beta"))?,
            };
            let ffn_res = ScaledResidual {
                norm: Box::new(rms(w, l, "ffn_norm", hp.rms_eps)?),
                inner: Box::new(experts),
                post: Some(Box::new(MulRms { w: w.take(&blk(l, "post_ffw_norm"))?, eps: hp.post_norm_eps })),
                alpha: w.take(&blk(l, "ffn_resid_alpha"))?,
                beta: w.take(&blk(l, "ffn_resid_beta"))?,
            };
            blocks.push(Block::new(l).then(attn_res).then(ffn_res));
        }
        let mut model = model(embed, blocks, final_norm, out);
        model.pre.push(Box::new(RmsNorm::weightless(hp.rms_eps)));
        model.logit_scale = hp.logit_scale;
        model.final_softcap = hp.final_softcap;
        finish(hp, w, model)
    }
}

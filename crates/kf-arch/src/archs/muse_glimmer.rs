use crate::compose::*;
use crate::{Architecture, Hparams, Keys, Weights};
use anyhow::{Result, bail};
use kf_gguf::Metadata;
use kf_layers::{Block, Model, Residual, RmsNorm, RopeKind};

/// Muse Glimmer, gated attention with sandwich norms, rope on window layers only, dense feed forward
pub struct MuseGlimmer;

impl Architecture for MuseGlimmer {
    fn id(&self) -> &'static str {
        "muse-glimmer"
    }

    fn hparams(&self, m: &Metadata) -> Result<Hparams> {
        let mut hp = Hparams::common(m, self.id())?;
        let k = Keys::new(m, self.id());
        hp.rope.kind = RopeKind::Normal;
        hp.need_rms_eps()?;
        hp.need_attention()?;
        hp.swa_pattern(m, 4, false);
        hp.need_swa()?;
        hp.rope_base_swa = k.f32("rope.freq_base_swa").unwrap_or(hp.rope.base);
        hp.final_softcap = k.f32("final_logit_softcapping").unwrap_or(30.0);
        if hp.logit_scale == 0.0 {
            bail!("{}: logit_scale is required", hp.arch);
        }
        Ok(hp)
    }

    fn build(&self, hp: &Hparams, w: &mut Weights) -> Result<Model> {
        let (embed, out) = embed_and_head(w, false)?;
        let final_norm = rms_named(w, "output_norm", hp.rms_eps)?;
        let mut blocks = Vec::with_capacity(hp.n_layer);
        for l in 0..hp.n_layer {
            let swa = hp.is_swa(l);
            let attn = attention(hp, w, l, AttnOpts { qk_norm: QkNorm::Shared, gate: GateOpt::Full, rope: swa.then(|| hp.rope_for(l)), window: window_for(hp, l), ..Default::default() })?;
            let ffn = ffn_with(hp, w, l, FfnOpts { require_gate: true, ..Default::default() })?;
            let attn_res = Residual::new(rms(w, l, "attn_norm", hp.rms_eps)?, attn).with_post(rms(w, l, "post_attention_norm", 1e-8)?);
            let ffn_res = Residual::new(rms(w, l, "ffn_norm", hp.rms_eps)?, ffn).with_post(rms(w, l, "post_ffw_norm", 1e-8)?);
            blocks.push(Block::new(l).then(attn_res).then(ffn_res));
        }
        let mut model = model(embed, blocks, final_norm, out);
        model.pre.push(Box::new(RmsNorm::weightless(hp.rms_eps)));
        model.logit_scale = hp.logit_scale;
        model.final_softcap = hp.final_softcap;
        finish(hp, w, model)
    }
}

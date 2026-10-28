use crate::compose::*;
use crate::{Architecture, Hparams, Keys, Weights};
use anyhow::Result;
use kf_core::Act;
use kf_gguf::Metadata;
use kf_layers::{Block, Model, Residual, RopeKind};

/// Gemma 2, sliding window every other layer, softcapped attention and logits, sandwich norms
pub struct Gemma2;

impl Architecture for Gemma2 {
    fn id(&self) -> &'static str {
        "gemma2"
    }

    fn hparams(&self, m: &Metadata) -> Result<Hparams> {
        let mut hp = Hparams::common(m, self.id())?;
        let k = Keys::new(m, self.id());
        hp.rope.kind = RopeKind::Neox;
        hp.need_rms_eps()?;
        hp.need_attention()?;
        hp.swa_window = Some(k.usize("attention.sliding_window").unwrap_or(4096));
        hp.swa_pattern(m, 2, false);
        hp.rope_base_swa = k.f32("rope.freq_base_swa").unwrap_or(hp.rope.base);
        hp.rope_scale_swa = hp.rope.scale;
        hp.attn_softcap = k.f32("attn_logit_softcapping").unwrap_or(50.0);
        hp.final_softcap = k.f32("final_logit_softcapping").unwrap_or(30.0);
        hp.attention_scale = if hp.n_layer == 46 { 1.0 / (hp.n_embd as f32 / hp.n_head as f32).sqrt() } else { 1.0 / (hp.head_dim_k as f32).sqrt() };
        Ok(hp)
    }

    fn build(&self, hp: &Hparams, w: &mut Weights) -> Result<Model> {
        let (mut embed, final_norm, out) = head(hp, w)?;
        embed.scale = (hp.n_embd as f32).sqrt();
        let mut blocks = Vec::with_capacity(hp.n_layer);
        for l in 0..hp.n_layer {
            let attn = attention(hp, w, l, AttnOpts { rope: Some(hp.rope_for(l)), window: window_for(hp, l), softcap: hp.attn_softcap, scale: Some(hp.attention_scale), ..Default::default() })?;
            let ffn = ffn_with(hp, w, l, FfnOpts { act: Some(Act::Gelu), require_gate: true, ..Default::default() })?;
            let attn_res = Residual::new(rms(w, l, "attn_norm", hp.rms_eps)?, attn).with_post(rms(w, l, "post_attention_norm", hp.rms_eps)?);
            let ffn_res = Residual::new(rms(w, l, "ffn_norm", hp.rms_eps)?, ffn).with_post(rms(w, l, "post_ffw_norm", hp.rms_eps)?);
            blocks.push(Block::new(l).then(attn_res).then(ffn_res));
        }
        let mut model = model(embed, blocks, final_norm, out);
        model.final_softcap = hp.final_softcap;
        finish(hp, w, model)
    }
}

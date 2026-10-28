use crate::compose::*;
use crate::{Architecture, Hparams, Keys, Weights};
use anyhow::Result;
use kf_core::Act;
use kf_gguf::Metadata;
use kf_layers::{Block, Model, Residual, RopeKind};

/// Gemma 3, q k norms, five window layers per full layer, local rope at the default base, sandwich norms
pub struct Gemma3;

impl Architecture for Gemma3 {
    fn id(&self) -> &'static str {
        "gemma3"
    }

    fn hparams(&self, m: &Metadata) -> Result<Hparams> {
        let mut hp = Hparams::common(m, self.id())?;
        let k = Keys::new(m, self.id());
        hp.rope.kind = RopeKind::Neox;
        hp.need_rms_eps()?;
        hp.need_attention()?;
        if hp.swa_window.is_some() {
            hp.swa_pattern(m, 6, false);
        }
        hp.final_softcap = k.f32("final_logit_softcapping").unwrap_or(0.0);
        hp.attention_scale = if hp.n_layer == 62 { 1.0 / (hp.n_embd as f32 / hp.n_head as f32).sqrt() } else { 1.0 / (hp.head_dim_k as f32).sqrt() };
        Ok(hp)
    }

    fn build(&self, hp: &Hparams, w: &mut Weights) -> Result<Model> {
        let (mut embed, final_norm, out) = head(hp, w)?;
        embed.scale = (hp.n_embd as f32).sqrt();
        claim(w, &["dense_2.weight".into(), "dense_3.weight".into()]);
        let mut blocks = Vec::with_capacity(hp.n_layer);
        for l in 0..hp.n_layer {
            let attn = attention(hp, w, l, AttnOpts { qk_norm: QkNorm::Shared, rope: Some(hp.rope_for(l)), window: window_for(hp, l), scale: Some(hp.attention_scale), ..Default::default() })?;
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

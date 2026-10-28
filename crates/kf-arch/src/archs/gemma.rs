use crate::compose::*;
use crate::{Architecture, Hparams, Weights};
use anyhow::Result;
use kf_core::Act;
use kf_gguf::Metadata;
use kf_layers::{Block, Model, RopeKind};

/// Gemma 1, scaled embeddings, queries scaled by the value head size, gelu feed forward, tied head
pub struct Gemma;

impl Architecture for Gemma {
    fn id(&self) -> &'static str {
        "gemma"
    }

    fn hparams(&self, m: &Metadata) -> Result<Hparams> {
        let mut hp = Hparams::common(m, self.id())?;
        hp.rope.kind = RopeKind::Neox;
        hp.need_rms_eps()?;
        hp.need_attention()?;
        Ok(hp)
    }

    fn build(&self, hp: &Hparams, w: &mut Weights) -> Result<Model> {
        let (mut embed, final_norm, out) = head(hp, w)?;
        embed.scale = (hp.n_embd as f32).sqrt();
        let mut blocks = Vec::with_capacity(hp.n_layer);
        for l in 0..hp.n_layer {
            let attn = attention(hp, w, l, AttnOpts { rope: Some(hp.rope_full()), scale: Some(1.0 / (hp.head_dim_v as f32).sqrt()), ..Default::default() })?;
            let ffn = ffn_with(hp, w, l, FfnOpts { act: Some(Act::Gelu), require_gate: true, ..Default::default() })?;
            blocks.push(Block::new(l).then(pre_norm(w, l, "attn_norm", hp.rms_eps, attn)?).then(pre_norm(w, l, "ffn_norm", hp.rms_eps, ffn)?));
        }
        finish(hp, w, model(embed, blocks, final_norm, out))
    }
}

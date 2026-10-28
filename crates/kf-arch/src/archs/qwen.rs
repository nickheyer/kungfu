use crate::compose::*;
use crate::{Architecture, Hparams, Weights};
use anyhow::Result;
use kf_gguf::Metadata;
use kf_layers::{Block, Model, RopeKind};

/// Qwen 1, fused qkv with bias and a gated silu feed forward of half the listed width
pub struct Qwen;

impl Architecture for Qwen {
    fn id(&self) -> &'static str {
        "qwen"
    }

    fn hparams(&self, m: &Metadata) -> Result<Hparams> {
        let mut hp = Hparams::common(m, self.id())?;
        hp.rope.kind = RopeKind::Neox;
        hp.need_rms_eps()?;
        hp.need_attention()?;
        Ok(hp)
    }

    fn build(&self, hp: &Hparams, w: &mut Weights) -> Result<Model> {
        let (embed, out) = embed_and_head(w, false)?;
        let final_norm = rms_named(w, "output_norm", hp.rms_eps)?;
        let mut blocks = Vec::with_capacity(hp.n_layer);
        for l in 0..hp.n_layer {
            let attn = attention(hp, w, l, AttnOpts { rope: Some(hp.rope_full()), ..Default::default() })?;
            let ffn = ffn_with(hp, w, l, FfnOpts { require_gate: true, ..Default::default() })?;
            blocks.push(Block::new(l).then(pre_norm(w, l, "attn_norm", hp.rms_eps, attn)?).then(pre_norm(w, l, "ffn_norm", hp.rms_eps, ffn)?));
        }
        finish(hp, w, model(embed, blocks, final_norm, out))
    }
}

use crate::compose::*;
use crate::{Architecture, Hparams, Weights};
use anyhow::Result;
use kf_core::{Gating, WeightNorm};
use kf_gguf::Metadata;
use kf_layers::{Block, Layer, Model, RopeKind};

/// Llama and Mixtral style files, softmax experts on every layer when the file has them
pub struct Llama;

impl Architecture for Llama {
    fn id(&self) -> &'static str {
        "llama"
    }

    fn hparams(&self, m: &Metadata) -> Result<Hparams> {
        let mut hp = Hparams::common(m, self.id())?;
        hp.rope.kind = RopeKind::Normal;
        hp.need_rms_eps()?;
        hp.need_attention()?;
        Ok(hp)
    }

    fn build(&self, hp: &Hparams, w: &mut Weights) -> Result<Model> {
        let (embed, final_norm, out) = head(hp, w)?;
        let mut rope = hp.rope_full();
        rope.factors = rope_factors(hp, w)?;
        let mut blocks = Vec::with_capacity(hp.n_layer);
        for l in 0..hp.n_layer {
            let attn = attention(hp, w, l, AttnOpts { rope: Some(rope.clone()), ..Default::default() })?;
            let ffn: Box<dyn Layer> = if hp.n_expert > 0 {
                Box::new(moe(hp, w, l, MoeOpts { gating: Gating::Softmax, norm_weights: WeightNorm::Clamp, ..Default::default() })?)
            } else {
                Box::new(ffn_with(hp, w, l, FfnOpts { require_gate: true, ..Default::default() })?)
            };
            blocks.push(Block::new(l).then(pre_norm(w, l, "attn_norm", hp.rms_eps, attn)?).then(pre_norm(w, l, "ffn_norm", hp.rms_eps, ffn)?));
        }
        finish(hp, w, model(embed, blocks, final_norm, out))
    }
}

use crate::compose::*;
use crate::{Architecture, Hparams, Weights};
use anyhow::Result;
use kf_core::{Gating, WeightNorm};
use kf_gguf::Metadata;
use kf_layers::{Block, Model, RopeKind};

/// Qwen 3 MoE, softmax experts with normalized weights and no shared expert
pub struct Qwen3Moe;

impl Architecture for Qwen3Moe {
    fn id(&self) -> &'static str {
        "qwen3moe"
    }

    fn hparams(&self, m: &Metadata) -> Result<Hparams> {
        let mut hp = Hparams::common(m, self.id())?;
        hp.rope.kind = RopeKind::Neox;
        hp.need_rms_eps()?;
        hp.need_attention()?;
        hp.need_moe()?;
        Ok(hp)
    }

    fn build(&self, hp: &Hparams, w: &mut Weights) -> Result<Model> {
        let (embed, final_norm, out) = head(hp, w)?;
        let mut blocks = Vec::with_capacity(hp.n_layer);
        for l in 0..hp.n_layer {
            let attn = attention(hp, w, l, AttnOpts { qk_norm: QkNorm::Shared, rope: Some(hp.rope_full()), ..Default::default() })?;
            let experts = moe(hp, w, l, MoeOpts { gating: Gating::Softmax, norm_weights: WeightNorm::Clamp, ..Default::default() })?;
            blocks.push(Block::new(l).then(pre_norm(w, l, "attn_norm", hp.rms_eps, attn)?).then(pre_norm(w, l, "ffn_norm", hp.rms_eps, experts)?));
        }
        finish(hp, w, model(embed, blocks, final_norm, out))
    }
}

use crate::compose::*;
use crate::{Architecture, Hparams, Weights};
use anyhow::Result;
use kf_core::Gating;
use kf_gguf::Metadata;
use kf_layers::{Block, Model, RopeKind};

/// Qwen 2 MoE, softmax experts without weight normalization plus a sigmoid gated shared expert
pub struct Qwen2Moe;

impl Architecture for Qwen2Moe {
    fn id(&self) -> &'static str {
        "qwen2moe"
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
        let (embed, out) = embed_and_head(w, false)?;
        let final_norm = rms_named(w, "output_norm", hp.rms_eps)?;
        let mut blocks = Vec::with_capacity(hp.n_layer);
        for l in 0..hp.n_layer {
            let attn = attention(hp, w, l, AttnOpts { rope: Some(hp.rope_full()), ..Default::default() })?;
            let experts = moe(hp, w, l, MoeOpts { gating: Gating::Softmax, shared: SharedOpt::Gated, n_ff_shexp: hp.n_ff_shexp, ..Default::default() })?;
            blocks.push(Block::new(l).then(pre_norm(w, l, "attn_norm", hp.rms_eps, attn)?).then(pre_norm(w, l, "ffn_norm", hp.rms_eps, experts)?));
        }
        finish(hp, w, model(embed, blocks, final_norm, out))
    }
}

use crate::compose::*;
use crate::{Architecture, Hparams, Weights};
use anyhow::Result;
use kf_gguf::Metadata;
use kf_layers::{Block, Model, RopeKind};

/// Qwen 3 dense, q k norms before rope and a gated silu feed forward
pub struct Qwen3;

impl Architecture for Qwen3 {
    fn id(&self) -> &'static str {
        "qwen3"
    }

    fn hparams(&self, m: &Metadata) -> Result<Hparams> {
        let mut hp = Hparams::common(m, self.id())?;
        hp.rope.kind = RopeKind::Neox;
        hp.need_rms_eps()?;
        hp.need_attention()?;
        Ok(hp)
    }

    fn build(&self, hp: &Hparams, w: &mut Weights) -> Result<Model> {
        let (embed, final_norm, out) = head(hp, w)?;
        claim(w, &["cls.output.weight".into(), "cls.output.bias".into(), "dense_2.weight".into(), "dense_2.bias".into()]);
        let mut blocks = Vec::with_capacity(hp.n_layer);
        for l in 0..hp.n_layer {
            let attn = attention(hp, w, l, AttnOpts { qk_norm: QkNorm::Shared, rope: Some(hp.rope_full()), ..Default::default() })?;
            let ffn = ffn_with(hp, w, l, FfnOpts { require_gate: true, ..Default::default() })?;
            blocks.push(Block::new(l).then(pre_norm(w, l, "attn_norm", hp.rms_eps, attn)?).then(pre_norm(w, l, "ffn_norm", hp.rms_eps, ffn)?));
        }
        finish(hp, w, model(embed, blocks, final_norm, out))
    }
}

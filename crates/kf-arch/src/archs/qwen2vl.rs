use crate::compose::*;
use crate::{Architecture, Hparams, Weights};
use anyhow::{Result, bail};
use kf_gguf::Metadata;
use kf_layers::{Block, Model, RopeKind};

/// Qwen 2 VL text stack, multi section rope and norms with the epsilon the loader never reads
pub struct Qwen2Vl;

impl Architecture for Qwen2Vl {
    fn id(&self) -> &'static str {
        "qwen2vl"
    }

    fn hparams(&self, m: &Metadata) -> Result<Hparams> {
        let mut hp = Hparams::common(m, self.id())?;
        hp.rope.kind = RopeKind::MRope;
        hp.rms_eps = 0.0;
        if hp.rope.sections.iter().all(|s| *s == 0) {
            bail!("{}: rope.dimension_sections is required", hp.arch);
        }
        hp.need_attention()?;
        Ok(hp)
    }

    fn build(&self, hp: &Hparams, w: &mut Weights) -> Result<Model> {
        let (embed, final_norm, mut out) = head(hp, w)?;
        out.b = None;
        w.take_opt("output.bias");
        let mut blocks = Vec::with_capacity(hp.n_layer);
        for l in 0..hp.n_layer {
            let attn = attention(hp, w, l, AttnOpts { rope: Some(hp.rope_full()), ..Default::default() })?;
            let ffn = ffn_with(hp, w, l, FfnOpts { require_gate: true, ..Default::default() })?;
            blocks.push(Block::new(l).then(pre_norm(w, l, "attn_norm", hp.rms_eps, attn)?).then(pre_norm(w, l, "ffn_norm", hp.rms_eps, ffn)?));
        }
        finish(hp, w, model(embed, blocks, final_norm, out))
    }
}

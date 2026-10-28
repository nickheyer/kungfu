use crate::archs::qwen3next::hybrid_block;
use crate::compose::*;
use crate::{Architecture, Hparams, Weights};
use anyhow::{Result, bail};
use kf_gguf::Metadata;
use kf_layers::{Block, HeadMap, Model, OutGate, RopeKind};

/// Qwen 3.5 dense, delta net layers with a dense feed forward and interleaved rope attention every fourth
pub struct Qwen35;

impl Architecture for Qwen35 {
    fn id(&self) -> &'static str {
        "qwen35"
    }

    fn hparams(&self, m: &Metadata) -> Result<Hparams> {
        let mut hp = Hparams::common(m, self.id())?;
        hp.rope.kind = RopeKind::IMRope;
        if hp.rope.sections.iter().all(|s| *s == 0) {
            bail!("{}: rope.dimension_sections is required", hp.arch);
        }
        hp.need_rms_eps()?;
        hp.need_attention()?;
        hp.need_ssm()?;
        hp.recurrent_pattern(m, 4);
        Ok(hp)
    }

    fn build(&self, hp: &Hparams, w: &mut Weights) -> Result<Model> {
        let (embed, final_norm, out) = head(hp, w)?;
        claim(w, &["cls.output.weight".into(), "cls.output.bias".into()]);
        let main = Main { embed: embed.w, head: out, norm: final_norm };
        let mut blocks = Vec::with_capacity(hp.n_layer);
        for l in 0..hp.n_layer {
            let ffn = ffn_with(hp, w, l, FfnOpts { require_gate: true, ..Default::default() })?;
            blocks.push(hybrid_block(hp, w, l, HeadMap::Tiled, OutGate::Silu, Box::new(ffn))?);
        }
        let mut model = model(embed, blocks, final_norm, out);
        for l in hp.n_layer..hp.n_layer_all {
            if has_nextn(w, l) {
                let ffn = ffn_with(hp, w, l, FfnOpts { require_gate: true, ..Default::default() })?;
                let block: Block = hybrid_block(hp, w, l, HeadMap::Tiled, OutGate::Silu, Box::new(ffn))?;
                model.extras.push(Box::new(nextn(hp, w, l, block, 1, None, &main)?));
            } else {
                w.claim_prefix(&format!("blk.{l}."));
            }
        }
        finish(hp, w, model)
    }
}

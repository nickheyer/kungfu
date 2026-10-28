use crate::archs::qwen3next::{hybrid_block, shared_moe};
use crate::compose::*;
use crate::{Architecture, Hparams, Weights};
use anyhow::{Result, bail};
use kf_gguf::Metadata;
use kf_layers::{HeadMap, Model, OutGate, RopeKind};

/// Qwen 3.5 MoE, the dense hybrid with routed and shared experts in place of the feed forward
pub struct Qwen35Moe;

impl Architecture for Qwen35Moe {
    fn id(&self) -> &'static str {
        "qwen35moe"
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
        hp.need_moe()?;
        hp.recurrent_pattern(m, 4);
        Ok(hp)
    }

    fn build(&self, hp: &Hparams, w: &mut Weights) -> Result<Model> {
        let (embed, final_norm, out) = head(hp, w)?;
        let main = Main { embed: embed.w, head: out, norm: final_norm };
        let mut blocks = Vec::with_capacity(hp.n_layer);
        for l in 0..hp.n_layer {
            let experts = shared_moe(hp, w, l)?;
            blocks.push(hybrid_block(hp, w, l, HeadMap::Tiled, OutGate::Silu, Box::new(experts))?);
        }
        let mut model = model(embed, blocks, final_norm, out);
        for l in hp.n_layer..hp.n_layer_all {
            if has_nextn(w, l) {
                let experts = shared_moe(hp, w, l)?;
                let block = hybrid_block(hp, w, l, HeadMap::Tiled, OutGate::Silu, Box::new(experts))?;
                model.extras.push(Box::new(nextn(hp, w, l, block, 1, None, &main)?));
            } else {
                w.claim_prefix(&format!("blk.{l}."));
            }
        }
        finish(hp, w, model)
    }
}

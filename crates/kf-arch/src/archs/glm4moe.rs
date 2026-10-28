use crate::compose::*;
use crate::{Architecture, Hparams, Weights};
use anyhow::Result;
use kf_core::{Gating, WeightNorm};
use kf_gguf::Metadata;
use kf_layers::{Block, Layer, Model, RopeKind};

/// GLM4 MoE, optional q k norms, leading dense layers, then sigmoid routed experts under the post attention norm
pub struct Glm4Moe;

impl Architecture for Glm4Moe {
    fn id(&self) -> &'static str {
        "glm4moe"
    }

    fn hparams(&self, m: &Metadata) -> Result<Hparams> {
        let mut hp = Hparams::common(m, self.id())?;
        hp.rope.kind = if hp.rope.sections[0] > 0 && hp.rope.sections[1] > 0 { RopeKind::MRope } else { RopeKind::Neox };
        hp.need_rms_eps()?;
        hp.need_attention()?;
        hp.need_moe()?;
        if hp.gating.is_none() {
            hp.gating = Some(Gating::Sigmoid);
        }
        Ok(hp)
    }

    fn build(&self, hp: &Hparams, w: &mut Weights) -> Result<Model> {
        let (embed, final_norm, out) = head(hp, w)?;
        let main = Main { embed: embed.w, head: out, norm: final_norm };
        let mut blocks = Vec::with_capacity(hp.n_layer);
        for l in 0..hp.n_layer {
            let ffn: Box<dyn Layer> = if l < hp.n_dense_lead { Box::new(ffn_with(hp, w, l, FfnOpts { require_gate: true, ..Default::default() })?) } else { Box::new(experts(hp, w, l)?) };
            blocks.push(block(hp, w, l, ffn)?);
        }
        let mut model = model(embed, blocks, final_norm, out);
        for l in hp.n_layer..hp.n_layer_all {
            if !has_nextn(w, l) {
                w.claim_prefix(&format!("blk.{l}."));
                continue;
            }
            let e = experts(hp, w, l)?;
            let b = block(hp, w, l, Box::new(e))?;
            model.extras.push(Box::new(nextn(hp, w, l, b, 1, None, &main)?));
        }
        finish(hp, w, model)
    }
}

fn experts(hp: &Hparams, w: &mut Weights, l: usize) -> Result<kf_layers::Moe> {
    moe(hp, w, l, MoeOpts {
        gating: hp.gating.unwrap_or(Gating::Sigmoid),
        norm_weights: if hp.expert_norm { WeightNorm::Clamp } else { WeightNorm::None },
        scale: hp.expert_scale,
        shared: if hp.n_expert_shared > 0 { SharedOpt::Plain } else { SharedOpt::None },
        probs_bias: true,
        probs_bias_required: true,
        ..Default::default()
    })
}

/// Attention under attn_norm then the feed forward under post_attention_norm, no norm between
fn block(hp: &Hparams, w: &mut Weights, l: usize, ffn: Box<dyn Layer>) -> Result<Block> {
    let attn = attention(hp, w, l, AttnOpts { qk_norm: QkNorm::SharedOpt, rope: Some(hp.rope_full()), ..Default::default() })?;
    Ok(Block::new(l).then(pre_norm(w, l, "attn_norm", hp.rms_eps, attn)?).then(pre_norm(w, l, "post_attention_norm", hp.rms_eps, ffn)?))
}

use crate::compose::*;
use crate::{Architecture, Hparams, Weights};
use anyhow::Result;
use kf_gguf::Metadata;
use kf_layers::{Block, Model, Residual, RopeKind};

/// GLM4 dense, sandwich norms around both sublayers and a fused gate up feed forward
pub struct Glm4;

impl Architecture for Glm4 {
    fn id(&self) -> &'static str {
        "glm4"
    }

    fn hparams(&self, m: &Metadata) -> Result<Hparams> {
        let mut hp = Hparams::common(m, self.id())?;
        hp.rope.kind = if hp.rope.sections[0] > 0 && hp.rope.sections[1] > 0 { RopeKind::MRope } else { RopeKind::Normal };
        hp.need_rms_eps()?;
        hp.need_attention()?;
        Ok(hp)
    }

    fn build(&self, hp: &Hparams, w: &mut Weights) -> Result<Model> {
        let (embed, final_norm, out) = head(hp, w)?;
        let mut blocks = Vec::with_capacity(hp.n_layer);
        for l in 0..hp.n_layer {
            let attn = attention(hp, w, l, AttnOpts { rope: Some(hp.rope_full()), ..Default::default() })?;
            let ffn = ffn(hp, w, l)?;
            let attn_res = Residual::new(rms(w, l, "attn_norm", hp.rms_eps)?, attn).with_post(rms(w, l, "post_attention_norm", hp.rms_eps)?);
            let ffn_res = Residual::new(rms(w, l, "ffn_norm", hp.rms_eps)?, ffn).with_post(rms(w, l, "post_ffw_norm", hp.rms_eps)?);
            blocks.push(Block::new(l).then(attn_res).then(ffn_res));
        }
        for l in hp.n_layer..hp.n_layer_all {
            w.claim_prefix(&format!("blk.{l}."));
        }
        finish(hp, w, model(embed, blocks, final_norm, out))
    }
}

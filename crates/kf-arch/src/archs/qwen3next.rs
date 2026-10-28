use crate::compose::*;
use crate::{Architecture, Hparams, Weights};
use anyhow::Result;
use kf_core::{Gating, WeightNorm};
use kf_gguf::Metadata;
use kf_layers::{Block, HeadMap, Layer, Model, OutGate, RopeKind};

/// Qwen3 Next, three delta net layers then one gated attention layer, routed plus shared experts
pub struct Qwen3Next;

impl Architecture for Qwen3Next {
    fn id(&self) -> &'static str {
        "qwen3next"
    }

    fn hparams(&self, m: &Metadata) -> Result<Hparams> {
        let mut hp = Hparams::common(m, self.id())?;
        hp.rope.kind = RopeKind::Neox;
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
            blocks.push(hybrid_block(hp, w, l, HeadMap::Grouped, OutGate::Silu, Box::new(experts))?);
        }
        let mut model = model(embed, blocks, final_norm, out);
        for l in hp.n_layer..hp.n_layer_all {
            if has_nextn(w, l) {
                let experts = shared_moe(hp, w, l)?;
                let block = hybrid_block(hp, w, l, HeadMap::Grouped, OutGate::Silu, Box::new(experts))?;
                model.extras.push(Box::new(nextn(hp, w, l, block, 1, None, &main)?));
            } else {
                w.claim_prefix(&format!("blk.{l}."));
            }
        }
        finish(hp, w, model)
    }
}

/// Softmax experts with normalized weights beside a sigmoid gated shared expert
pub fn shared_moe(hp: &Hparams, w: &mut Weights, l: usize) -> Result<kf_layers::Moe> {
    moe(hp, w, l, MoeOpts { gating: Gating::Softmax, norm_weights: WeightNorm::Clamp, shared: SharedOpt::Gated, n_ff_shexp: hp.n_ff_shexp, ..Default::default() })
}

/// A delta net or gated attention mixer under attn_norm, then the feed forward under post_attention_norm
pub fn hybrid_block(hp: &Hparams, w: &mut Weights, l: usize, head_map: HeadMap, out_gate: OutGate, ffn: Box<dyn Layer>) -> Result<Block> {
    let mixer: Box<dyn Layer> = if hp.is_recurrent(l) {
        Box::new(delta_net(hp, w, l, head_map, out_gate)?)
    } else {
        Box::new(attention(hp, w, l, AttnOpts { qk_norm: QkNorm::Shared, gate: GateOpt::InQ, rope: Some(hp.rope_full()), ..Default::default() })?)
    };
    Ok(Block::new(l).then(pre_norm(w, l, "attn_norm", hp.rms_eps, mixer)?).then(pre_norm(w, l, "post_attention_norm", hp.rms_eps, ffn)?))
}

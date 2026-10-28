use crate::compose::*;
use crate::{Architecture, Hparams, Weights};
use anyhow::Result;
use kf_core::{Gating, WeightNorm};
use kf_gguf::Metadata;
use kf_layers::{AttnTemp, Block, Layer, Model, RopeKind};

/// DeepSeek V2 and V3, latent attention with leading dense layers then routed and shared experts
pub struct DeepSeek2;

impl Architecture for DeepSeek2 {
    fn id(&self) -> &'static str {
        "deepseek2"
    }

    fn hparams(&self, m: &Metadata) -> Result<Hparams> {
        let mut hp = Hparams::common(m, self.id())?;
        hp.rope.kind = RopeKind::Normal;
        hp.need_rms_eps()?;
        hp.need_attention()?;
        hp.need_moe()?;
        if hp.mla.kv_lora == 0 {
            anyhow::bail!("{}: attention.kv_lora_rank is required", hp.arch);
        }
        if hp.gating.is_none() {
            hp.gating = Some(if (hp.n_layer_all == 47 || hp.n_layer_all == 48) && hp.n_vocab == 154880 { Gating::Sigmoid } else { Gating::Softmax });
        }
        Ok(hp)
    }

    fn build(&self, hp: &Hparams, w: &mut Weights) -> Result<Model> {
        let (embed, final_norm, out) = head(hp, w)?;
        let main = Main { embed: embed.w, head: out, norm: final_norm };
        let mut blocks = Vec::with_capacity(hp.n_layer);
        for l in 0..hp.n_layer {
            let attn = latent(hp, w, l)?;
            let ffn: Box<dyn Layer> = if l < hp.n_dense_lead { Box::new(ffn_with(hp, w, l, FfnOpts { require_gate: true, ..Default::default() })?) } else { Box::new(experts(hp, w, l)?) };
            blocks.push(Block::new(l).then(pre_norm(w, l, "attn_norm", hp.rms_eps, attn)?).then(pre_norm(w, l, "ffn_norm", hp.rms_eps, ffn)?));
        }
        let mut model = model(embed, blocks, final_norm, out);
        for l in hp.n_layer..hp.n_layer_all {
            if !has_nextn(w, l) {
                w.claim_prefix(&format!("blk.{l}."));
                continue;
            }
            let attn = latent(hp, w, l)?;
            let ffn = experts(hp, w, l)?;
            let block = Block::new(l).then(pre_norm(w, l, "attn_norm", hp.rms_eps, attn)?).then(pre_norm(w, l, "ffn_norm", hp.rms_eps, ffn)?);
            model.extras.push(Box::new(nextn(hp, w, l, block, 1, None, &main)?));
        }
        finish(hp, w, model)
    }
}

/// Routed experts with the file's gating, bias and normalization plus the required shared experts
pub fn experts(hp: &Hparams, w: &mut Weights, l: usize) -> Result<kf_layers::Moe> {
    moe(hp, w, l, MoeOpts {
        gating: hp.gating.unwrap_or(Gating::Softmax),
        norm_weights: if hp.expert_norm { WeightNorm::Clamp } else { WeightNorm::None },
        scale: hp.expert_scale,
        shared: SharedOpt::Plain,
        probs_bias: true,
        ..Default::default()
    })
}

/// Absorbed or legacy latent attention with the yarn scale and optional temperature of DeepSeek
pub fn latent(hp: &Hparams, w: &mut Weights, l: usize) -> Result<kf_layers::Mla> {
    let (head_k, head_v) = if hp.mla.head_k > 0 { (hp.mla.head_k, hp.mla.head_v) } else { (hp.head_dim_k, hp.head_dim_v) };
    let temp = (hp.attn_temp_scale != 0.0 && hp.attn_temp_length > 0).then(|| AttnTemp { floor_scale: hp.attn_temp_length as f32, scale: hp.attn_temp_scale, offset: 0.0, pow: false });
    mla(hp, w, l, MlaOpts {
        rope: Some(hp.rope_full()),
        sparse: None,
        scale: mla_scale(hp, head_k),
        temp,
        gate: GateOpt::None,
        sinks: false,
        window: None,
        k_pe_norm: false,
        legacy_pe_first: false,
        head_k,
        head_v,
        kv_lora: hp.mla.kv_lora,
    })
}

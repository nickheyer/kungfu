use crate::compose::*;
use crate::{Architecture, Hparams, Weights};
use anyhow::Result;
use kf_core::{Gating, IndexerKind, WeightNorm};
use kf_gguf::Metadata;
use kf_layers::{Block, Layer, Model, RopeKind};

/// GLM 5 with sparse latent attention, a full indexer picks keys that shared layers reuse
pub struct GlmDsa;

impl Architecture for GlmDsa {
    fn id(&self) -> &'static str {
        "glm-dsa"
    }

    fn hparams(&self, m: &Metadata) -> Result<Hparams> {
        let mut hp = Hparams::common(m, self.id())?;
        hp.rope.kind = RopeKind::Normal;
        hp.need_rms_eps()?;
        hp.need_attention()?;
        hp.need_mla()?;
        hp.need_indexer()?;
        hp.need_moe()?;
        if hp.gating.is_none() {
            hp.gating = Some(Gating::Sigmoid);
        }
        if hp.indexer.full.is_empty() {
            let all = hp.n_ctx_train < 1_048_576;
            hp.indexer.full = (0..hp.n_layer_all).map(|l| all || l < 3 || l % 4 == 2).collect();
        }
        Ok(hp)
    }

    fn build(&self, hp: &Hparams, w: &mut Weights) -> Result<Model> {
        let (embed, final_norm, out) = head(hp, w)?;
        let main = Main { embed: embed.w, head: out, norm: final_norm };
        let mut blocks = Vec::with_capacity(hp.n_layer);
        let mut last_full = None;
        for l in 0..hp.n_layer {
            let full = hp.indexer.full.get(l).copied().unwrap_or(true);
            let sparse = if full {
                last_full = Some(l);
                indexer(hp, w, l, IndexerKind::Dense, Some(hp.rope_full()), false, None)?
            } else {
                let src = last_full.ok_or_else(|| anyhow::anyhow!("{}: layer {l} shares an indexer before any full layer", hp.arch))?;
                indexer_shared(hp, w, l, IndexerKind::Dense, src, embed.w)
            };
            let attn = latent(hp, w, l, Some(sparse))?;
            let ffn: Box<dyn Layer> = if l < hp.n_dense_lead { Box::new(ffn_with(hp, w, l, FfnOpts { require_gate: true, ..Default::default() })?) } else { Box::new(experts(hp, w, l)?) };
            blocks.push(Block::new(l).then(pre_norm(w, l, "attn_norm", hp.rms_eps, attn)?).then(pre_norm(w, l, "ffn_norm", hp.rms_eps, ffn)?));
        }
        let mut model = model(embed, blocks, final_norm, out);
        for l in hp.n_layer..hp.n_layer_all {
            if !has_nextn(w, l) {
                w.claim_prefix(&format!("blk.{l}."));
                continue;
            }
            w.claim_prefix(&format!("blk.{l}.indexer"));
            let attn = latent(hp, w, l, None)?;
            let ffn = experts(hp, w, l)?;
            let block = Block::new(l).then(pre_norm(w, l, "attn_norm", hp.rms_eps, attn)?).then(pre_norm(w, l, "ffn_norm", hp.rms_eps, ffn)?);
            model.extras.push(Box::new(nextn(hp, w, l, block, 1, None, &main)?));
        }
        finish(hp, w, model)
    }
}

fn experts(hp: &Hparams, w: &mut Weights, l: usize) -> Result<kf_layers::Moe> {
    moe(hp, w, l, MoeOpts {
        gating: hp.gating.unwrap_or(Gating::Sigmoid),
        norm_weights: if hp.expert_norm { WeightNorm::Clamp } else { WeightNorm::None },
        scale: hp.expert_scale,
        shared: SharedOpt::Plain,
        probs_bias: true,
        ..Default::default()
    })
}

fn latent(hp: &Hparams, w: &mut Weights, l: usize, sparse: Option<kf_layers::DsaIndexer>) -> Result<kf_layers::Mla> {
    mla(hp, w, l, MlaOpts {
        rope: Some(hp.rope_full()),
        sparse,
        scale: mla_scale(hp, hp.mla.head_k),
        temp: None,
        gate: GateOpt::None,
        sinks: false,
        window: None,
        k_pe_norm: false,
        legacy_pe_first: false,
        head_k: hp.mla.head_k,
        head_v: hp.mla.head_v,
        kv_lora: hp.mla.kv_lora,
    })
}

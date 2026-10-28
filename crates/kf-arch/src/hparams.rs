use anyhow::{Result, anyhow, bail};
use kf_core::{Gating, Groups, RopeKind, RopeParams, SwaType, Yarn};
use kf_gguf::Metadata;
use serde::Serialize;

#[derive(Clone, Debug, Default, Serialize)]
pub struct Ssm {
    pub conv_kernel: usize,
    pub inner: usize,
    pub state: usize,
    pub dt_rank: usize,
    pub n_group: usize,
    pub dt_b_c_rms: bool,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct MlaDims {
    pub q_lora: usize,
    pub kv_lora: usize,
    pub head_k: usize,
    pub head_v: usize,
    pub kv_lora_swa: usize,
    pub head_k_swa: usize,
    pub head_v_swa: usize,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Indexer {
    pub n_head: usize,
    pub head_dim: usize,
    pub top_k: usize,
    pub kpool: usize,
    pub kpool_tail: usize,
    pub block_size: usize,
    pub local_blocks: usize,
    /// Per layer, true where the indexer scores every key
    pub full: Vec<bool>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Hc {
    pub count: usize,
    pub low_rank: usize,
    pub sinkhorn: usize,
    pub epsilon: f32,
    pub magnitude: f32,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct PleParams {
    pub layers: Vec<usize>,
    pub ngram: usize,
    pub heads_per_ngram: usize,
    pub conv_kernel: usize,
    pub multipliers: Vec<u64>,
    pub head_offsets: Vec<u32>,
    pub head_vocab: Vec<u32>,
    pub eos: u32,
    pub image: Option<u32>,
    pub embd_per_layer: usize,
}

/// Layer wiring of DeepSeek V4, which layers feed kv and indexer scores to which
#[derive(Clone, Debug, Default, Serialize)]
pub struct Dsv4 {
    pub kv_source: Vec<usize>,
    pub topk_source: Vec<usize>,
    pub compress_ratios: Vec<usize>,
    pub compress_rope_base: f32,
    pub o_groups: usize,
    pub o_lora_rank: usize,
    pub hash_layers: usize,
    pub engram_pad: u32,
    pub engram_token_map: Vec<u32>,
    pub candidate_source: Option<usize>,
    pub candidate_top_blocks: usize,
    pub candidate_block: usize,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct KdaParams {
    pub head_dim: usize,
    pub gate_lower_bound: f32,
    pub safe_gate: bool,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Rwkv {
    pub head_size: usize,
    pub time_mix_extra_dim: usize,
    pub time_decay_extra_dim: usize,
    pub rescale_every: usize,
    pub token_shift_count: usize,
    pub lora_decay: usize,
    pub lora_iclr: usize,
    pub lora_value_res_mix: usize,
    pub lora_gate: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub enum RopeScaling {
    None,
    #[default]
    Linear,
    Yarn,
    LongRope,
}

/// Every hyperparameter any supported family reads, absent ones stay zero or empty
#[derive(Clone, Debug, Default, Serialize)]
pub struct Hparams {
    pub arch: String,
    pub n_layer: usize,
    pub n_layer_all: usize,
    pub n_nextn: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim_k: usize,
    pub head_dim_v: usize,
    pub head_dim_k_swa: usize,
    pub head_dim_v_swa: usize,
    pub n_ff: usize,
    pub n_vocab: usize,
    pub n_ctx_train: usize,
    /// Runtime context length, set by the loader before build
    pub n_ctx: usize,
    pub n_head_arr: Vec<usize>,
    pub n_head_kv_arr: Vec<usize>,
    pub n_ff_arr: Vec<usize>,
    pub n_ff_exp_arr: Vec<usize>,
    pub n_expert_used_arr: Vec<usize>,
    pub rms_eps: f32,
    pub ln_eps: f32,
    pub post_norm_eps: f32,
    pub rope: RopeParams,
    pub rope_dims_swa: usize,
    pub rope_scaling: RopeScaling,
    pub rope_attn_factor: f32,
    pub yarn_log_mul: f32,
    pub yarn_beta_fast: f32,
    pub yarn_beta_slow: f32,
    pub rope_base_swa: f32,
    pub rope_scale_swa: f32,
    pub rope_alpha: f32,
    pub rope_pattern: Vec<bool>,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub n_ff_exp: usize,
    pub n_ff_shexp: usize,
    pub n_ff_chexp: usize,
    pub n_expert_shared: usize,
    pub n_dense_lead: usize,
    pub expert_scale: f32,
    pub expert_norm: bool,
    pub gating: Option<Gating>,
    pub expert_groups: Option<Groups>,
    pub expert_group_scale: f32,
    pub experts_per_group: usize,
    pub interleave_moe_step: usize,
    pub moe_latent: usize,
    pub expert_latent: usize,
    pub logit_scale: f32,
    pub final_softcap: f32,
    pub attn_softcap: f32,
    pub router_softcap: f32,
    pub residual_scale: f32,
    pub embedding_scale: f32,
    pub attention_scale: f32,
    pub attn_out_scale: f32,
    pub value_scale: f32,
    pub clamp_kqv: f32,
    pub max_alibi_bias: f32,
    pub attn_temp_scale: f32,
    pub attn_temp_length: usize,
    pub use_parallel_residual: bool,
    pub swa_window: Option<usize>,
    pub swa_type: SwaType,
    /// Per layer, true on sliding window layers
    pub swa: Vec<bool>,
    /// Per layer, true on recurrent layers
    pub recurrent: Vec<bool>,
    pub shared_kv_layers: usize,
    pub n_layer_kv_from_start: Option<usize>,
    pub ssm: Ssm,
    pub mla: MlaDims,
    pub indexer: Indexer,
    pub hc: Hc,
    pub ple: PleParams,
    pub kda: KdaParams,
    pub dsv4: Dsv4,
    pub rwkv: Rwkv,
    pub shortconv_l_cache: usize,
    pub swiglu_clamp_exp: Vec<f32>,
    pub swiglu_clamp_shexp: Vec<f32>,
    pub xielu: Vec<[f32; 4]>,
    pub num_loops: usize,
    pub skip_loop_final_norm: bool,
}

/// Typed access to the arch prefixed keys of a file
pub struct Keys<'a> {
    pub m: &'a Metadata,
    pub arch: &'a str,
}

impl<'a> Keys<'a> {
    pub fn new(m: &'a Metadata, arch: &'a str) -> Self {
        Self { m, arch }
    }
    pub fn k(&self, s: &str) -> String {
        format!("{}.{s}", self.arch)
    }
    pub fn has(&self, s: &str) -> bool {
        self.m.has(&self.k(s))
    }
    pub fn usize(&self, s: &str) -> Option<usize> {
        self.m.usize(&self.k(s))
    }
    pub fn f32(&self, s: &str) -> Option<f32> {
        self.m.f32(&self.k(s))
    }
    pub fn bool(&self, s: &str) -> Option<bool> {
        self.m.bool(&self.k(s))
    }
    pub fn str(&self, s: &str) -> Option<&'a str> {
        self.m.str(&self.k(s))
    }
    pub fn arr_usize(&self, s: &str) -> Option<Vec<usize>> {
        self.m.arr_usize(&self.k(s))
    }
    pub fn arr_u64(&self, s: &str) -> Option<Vec<u64>> {
        self.m.arr(&self.k(s))?.iter().map(|v| v.as_u64()).collect()
    }
    pub fn arr_f32(&self, s: &str) -> Option<Vec<f32>> {
        self.m.arr_f32(&self.k(s))
    }
    pub fn req(&self, s: &str) -> Result<usize> {
        self.usize(s).ok_or_else(|| anyhow!("{} missing", self.k(s)))
    }
    pub fn req_f32(&self, s: &str) -> Result<f32> {
        self.f32(s).ok_or_else(|| anyhow!("{} missing", self.k(s)))
    }
    /// A key that is one value or one value per layer, the first value when it is an array
    pub fn usize_or_arr(&self, s: &str) -> Option<usize> {
        self.usize(s).or_else(|| self.arr_usize(s).and_then(|v| v.first().copied()))
    }
    /// A scalar broadcast to n layers, or the per layer array itself
    pub fn key_or_arr(&self, s: &str, n: usize) -> Option<Vec<usize>> {
        match self.usize(s) {
            Some(v) => Some(vec![v; n]),
            None => self.arr_usize(s),
        }
    }
    pub fn key_or_arr_f32(&self, s: &str, n: usize) -> Option<Vec<f32>> {
        match self.f32(s) {
            Some(v) => Some(vec![v; n]),
            None => self.arr_f32(s),
        }
    }
}

fn per_layer(v: &[usize], l: usize, fallback: usize) -> usize {
    v.get(l).copied().unwrap_or(fallback)
}

impl Hparams {
    /// Reads the keys every family shares, leaving the rest at llama.cpp's defaults
    pub fn common(m: &Metadata, arch: &str) -> Result<Self> {
        let k = Keys { m, arch };
        let n_layer_all = k.req("block_count")?;
        let n_nextn = k.usize("nextn_predict_layers").unwrap_or(0);
        if n_nextn >= n_layer_all {
            bail!("nextn_predict_layers {n_nextn} leaves no trunk layers");
        }
        let n_layer = n_layer_all - n_nextn;
        let n_embd = k.req("embedding_length")?;
        let n_head_arr = k.key_or_arr("attention.head_count", n_layer_all).unwrap_or_default();
        let n_head = n_head_arr.first().copied().unwrap_or(0);
        let n_head_kv_arr = k.key_or_arr("attention.head_count_kv", n_layer_all).unwrap_or_else(|| n_head_arr.clone());
        let n_head_kv = n_head_kv_arr.first().copied().unwrap_or(n_head);
        let n_ff_arr = k.key_or_arr("feed_forward_length", n_layer_all).unwrap_or_default();
        let n_ff_exp_arr = k.key_or_arr("expert_feed_forward_length", n_layer_all).unwrap_or_default();
        let n_expert_used_arr = k.key_or_arr("expert_used_count", n_layer_all).unwrap_or_default();
        let head_dim_k = k.usize("attention.key_length").unwrap_or(if n_head > 0 { n_embd / n_head } else { 0 });
        let head_dim_v = k.usize("attention.value_length").unwrap_or(if n_head > 0 { n_embd / n_head } else { 0 });
        let n_vocab = k.usize("vocab_size").or_else(|| m.arr_len("tokenizer.ggml.tokens")).unwrap_or(0);
        let rope_dims = k.usize("rope.dimension_count").unwrap_or(head_dim_k);
        let rope_scaling = match k.str("rope.scaling.type").unwrap_or("linear") {
            "none" => RopeScaling::None,
            "yarn" => RopeScaling::Yarn,
            "longrope" => RopeScaling::LongRope,
            _ => RopeScaling::Linear,
        };
        let factor = k.f32("rope.scaling.factor").or_else(|| k.f32("rope.scale_linear")).unwrap_or(0.0);
        let scale = if rope_scaling == RopeScaling::None || factor == 0.0 { 1.0 } else { 1.0 / factor };
        let mut sections = [0usize; 4];
        if let Some(s) = k.arr_usize("rope.dimension_sections") {
            for (dst, src) in sections.iter_mut().zip(s) {
                *dst = src;
            }
        }
        let n_ctx_train = k.usize("context_length").unwrap_or(0);
        let gating = match k.usize("expert_gating_func") {
            Some(1) => Some(Gating::Softmax),
            Some(2) => Some(Gating::Sigmoid),
            Some(3) => Some(Gating::SoftmaxWeight),
            Some(4) => Some(Gating::SqrtSoftplus),
            _ => None,
        };
        let expert_groups = match (k.usize("expert_group_count"), k.usize("expert_group_used_count")) {
            (Some(n_group), Some(n_used)) if n_group > 1 => Some(Groups { n_group, n_used }),
            _ => None,
        };
        let indexer_types = k.key_or_arr("attention.indexer.types", n_layer_all).map(|v| v.into_iter().map(|x| x != 0).collect()).unwrap_or_default();
        let rope_pattern = k.key_or_arr("attention.rope_pattern", n_layer_all).map(|v| v.into_iter().map(|x| x != 0).collect()).unwrap_or_else(|| vec![true; n_layer_all]);
        let xielu = match (k.m.arr_f32("xielu.alpha_n"), k.m.arr_f32("xielu.alpha_p"), k.m.arr_f32("xielu.beta"), k.m.arr_f32("xielu.eps")) {
            (Some(an), Some(ap), Some(b), Some(e)) => an.iter().zip(&ap).zip(&b).zip(&e).map(|(((an, ap), b), e)| [*an, *ap, *b, *e]).collect(),
            _ => match (k.m.f32("xielu.alpha_n"), k.m.f32("xielu.alpha_p"), k.m.f32("xielu.beta"), k.m.f32("xielu.eps")) {
                (Some(an), Some(ap), Some(b), Some(e)) => vec![[an, ap, b, e]; n_layer_all],
                _ => Vec::new(),
            },
        };
        Ok(Self {
            arch: arch.to_string(),
            n_layer,
            n_layer_all,
            n_nextn,
            n_embd,
            n_head,
            n_head_kv,
            head_dim_k,
            head_dim_v,
            head_dim_k_swa: k.usize("attention.key_length_swa").unwrap_or(head_dim_k),
            head_dim_v_swa: k.usize("attention.value_length_swa").unwrap_or(head_dim_v),
            n_ff: n_ff_arr.first().copied().unwrap_or(0),
            n_vocab,
            n_ctx_train,
            n_ctx: n_ctx_train,
            n_head_arr,
            n_head_kv_arr,
            n_ff_arr,
            n_ff_exp: n_ff_exp_arr.first().copied().unwrap_or(0),
            n_ff_exp_arr,
            n_expert_used: n_expert_used_arr.first().copied().unwrap_or(0),
            n_expert_used_arr,
            rms_eps: k.f32("attention.layer_norm_rms_epsilon").unwrap_or(0.0),
            ln_eps: k.f32("attention.layer_norm_epsilon").unwrap_or(0.0),
            post_norm_eps: k.f32("post_norm_epsilon").unwrap_or(1e-8),
            rope: RopeParams {
                kind: RopeKind::Neox,
                dims: rope_dims,
                base: k.f32("rope.freq_base").unwrap_or(10000.0),
                scale,
                orig_ctx: k.usize("rope.scaling.original_context_length").unwrap_or(n_ctx_train),
                attn_factor: 1.0,
                yarn: None,
                sections,
                factors: None,
            },
            rope_dims_swa: k.usize("rope.dimension_count_swa").unwrap_or(rope_dims),
            rope_scaling,
            rope_attn_factor: k.f32("rope.scaling.attn_factor").unwrap_or(1.0),
            yarn_log_mul: k.f32("rope.scaling.yarn_log_multiplier").unwrap_or(0.0),
            yarn_beta_fast: k.f32("rope.scaling.yarn_beta_fast").unwrap_or(32.0),
            yarn_beta_slow: k.f32("rope.scaling.yarn_beta_slow").unwrap_or(1.0),
            rope_base_swa: k.f32("rope.freq_base_swa").unwrap_or(10000.0),
            rope_scale_swa: 1.0,
            rope_alpha: k.f32("rope.scaling.alpha").unwrap_or(0.0),
            rope_pattern,
            n_expert: k.usize("expert_count").unwrap_or(0),
            n_ff_shexp: k.usize("expert_shared_feed_forward_length").unwrap_or(0),
            n_ff_chexp: k.usize("expert_chunk_feed_forward_length").unwrap_or(0),
            n_expert_shared: k.usize("expert_shared_count").unwrap_or(0),
            n_dense_lead: k.usize("leading_dense_block_count").unwrap_or(0),
            expert_scale: k.f32("expert_weights_scale").unwrap_or(0.0),
            expert_norm: k.bool("expert_weights_norm").unwrap_or(false),
            gating,
            expert_groups,
            expert_group_scale: k.f32("expert_group_scale").unwrap_or(0.05),
            experts_per_group: k.usize("experts_per_group").unwrap_or(0),
            interleave_moe_step: k.usize("interleave_moe_layer_step").unwrap_or(0),
            moe_latent: k.usize("moe_latent_size").unwrap_or(0),
            expert_latent: k.usize("expert_latent_length").unwrap_or(0),
            logit_scale: k.f32("logit_scale").unwrap_or(0.0),
            final_softcap: k.f32("final_logit_softcapping").unwrap_or(0.0),
            attn_softcap: k.f32("attn_logit_softcapping").unwrap_or(0.0),
            router_softcap: k.f32("router_logit_softcapping").unwrap_or(0.0),
            residual_scale: k.f32("residual_scale").unwrap_or(0.0),
            embedding_scale: k.f32("embedding_scale").unwrap_or(0.0),
            attention_scale: k.f32("attention.scale").unwrap_or(0.0),
            attn_out_scale: k.f32("attention.output_scale").unwrap_or(0.0),
            value_scale: k.f32("attention.value_scale").unwrap_or(0.0),
            clamp_kqv: k.f32("attention.clamp_kqv").unwrap_or(0.0),
            max_alibi_bias: k.f32("attention.max_alibi_bias").unwrap_or(0.0),
            attn_temp_scale: k.f32("attention.temperature_scale").unwrap_or(0.0),
            attn_temp_length: k.usize("attention.temperature_length").unwrap_or(0),
            use_parallel_residual: k.bool("use_parallel_residual").unwrap_or(false),
            swa_window: k.usize("attention.sliding_window").filter(|w| *w > 0),
            swa_type: SwaType::Standard,
            swa: Vec::new(),
            recurrent: Vec::new(),
            shared_kv_layers: k.usize("attention.shared_kv_layers").unwrap_or(0),
            n_layer_kv_from_start: None,
            ssm: Ssm {
                conv_kernel: k.usize("ssm.conv_kernel").unwrap_or(0),
                inner: k.usize("ssm.inner_size").unwrap_or(0),
                state: k.usize("ssm.state_size").unwrap_or(0),
                dt_rank: k.usize("ssm.time_step_rank").unwrap_or(0),
                n_group: k.usize("ssm.group_count").unwrap_or(0),
                dt_b_c_rms: k.bool("ssm.dt_b_c_rms").unwrap_or(false),
            },
            mla: MlaDims {
                q_lora: k.usize("attention.q_lora_rank").unwrap_or(0),
                kv_lora: k.usize("attention.kv_lora_rank").unwrap_or(0),
                head_k: k.usize("attention.key_length_mla").unwrap_or(0),
                head_v: k.usize("attention.value_length_mla").unwrap_or(0),
                kv_lora_swa: k.usize("attention.kv_lora_rank_swa").unwrap_or(0),
                head_k_swa: k.usize("attention.key_length_mla_swa").unwrap_or(0),
                head_v_swa: k.usize("attention.value_length_mla_swa").unwrap_or(0),
            },
            indexer: Indexer {
                n_head: k.usize("attention.indexer.head_count").unwrap_or(0),
                head_dim: k.usize("attention.indexer.key_length").unwrap_or(0),
                top_k: k.usize("attention.indexer.top_k").unwrap_or(0),
                kpool: k.usize("attention.indexer.kpool").unwrap_or(0),
                kpool_tail: k.usize("attention.indexer.kpool_select_tail").unwrap_or(0),
                block_size: k.usize("attention.indexer.block_size").unwrap_or(0),
                local_blocks: k.usize("attention.indexer.local_blocks").unwrap_or(0),
                full: indexer_types,
            },
            hc: Hc {
                count: k.usize("hyper_connection.count").unwrap_or(0),
                low_rank: k.usize("hyper_connection.low_rank").unwrap_or(0),
                sinkhorn: k.usize("hyper_connection.sinkhorn_iterations").unwrap_or(0),
                epsilon: k.f32("hyper_connection.epsilon").unwrap_or(0.0),
                magnitude: k.f32("hyper_connection.magnitude").unwrap_or(0.0),
            },
            ple: PleParams {
                layers: k.arr_usize("ple.layers").unwrap_or_default(),
                ngram: k.usize("ple.ngram_size").unwrap_or(0),
                heads_per_ngram: k.usize("ple.heads_per_ngram").unwrap_or(0),
                conv_kernel: k.usize("ple.conv_kernel").unwrap_or(0),
                multipliers: k.arr_u64("ple.layer_multipliers").unwrap_or_default(),
                head_offsets: k.arr_u64("ple.head_offsets").unwrap_or_default().into_iter().map(|v| v as u32).collect(),
                head_vocab: k.arr_u64("ple.head_vocab_sizes").unwrap_or_default().into_iter().map(|v| v as u32).collect(),
                eos: k.usize("ple.eos_token_id").unwrap_or(0) as u32,
                image: k.usize("ple.image_token_id").filter(|v| *v > 0).map(|v| v as u32),
                embd_per_layer: k.usize("embedding_length_per_layer_input").unwrap_or(0),
            },
            kda: KdaParams {
                head_dim: k.usize("kda.head_dim").unwrap_or(0),
                gate_lower_bound: k.f32("kda.gate_lower_bound").unwrap_or(f32::NEG_INFINITY),
                safe_gate: k.bool("kda.safe_gate").unwrap_or(false),
            },
            dsv4: Dsv4 {
                kv_source: k.arr_usize("attention.kv_source_layers").unwrap_or_default(),
                topk_source: k.arr_usize("attention.indexer.source_layers").unwrap_or_default(),
                compress_ratios: k.key_or_arr("attention.compress_ratios", n_layer_all).unwrap_or_default(),
                compress_rope_base: k.f32("attention.compress_rope_freq_base").unwrap_or(0.0),
                o_groups: k.usize("attention.output_group_count").unwrap_or(0),
                o_lora_rank: k.usize("attention.output_lora_rank").unwrap_or(0),
                hash_layers: k.usize("hash_layer_count").unwrap_or(0),
                engram_pad: k.usize("engram.pad_token_id").unwrap_or(0) as u32,
                engram_token_map: k.arr_usize("engram.token_map").unwrap_or_default().into_iter().map(|v| v as u32).collect(),
                candidate_source: m.i64(&k.k("attention.indexer.candidate_source_layer")).filter(|v| *v >= 0).map(|v| v as usize),
                candidate_top_blocks: k.usize("attention.indexer.candidate_top_k_blocks").unwrap_or(0),
                candidate_block: k.usize("attention.indexer.candidate_block_size").unwrap_or(0),
            },
            rwkv: Rwkv {
                head_size: k.usize("wkv.head_size").unwrap_or(0),
                time_mix_extra_dim: k.usize("time_mix_extra_dim").unwrap_or(0),
                time_decay_extra_dim: k.usize("time_decay_extra_dim").unwrap_or(0),
                rescale_every: k.usize("rescale_every_n_layers").unwrap_or(0),
                token_shift_count: k.usize("token_shift_count").unwrap_or(2),
                lora_decay: k.usize("attention.decay_lora_rank").unwrap_or(0),
                lora_iclr: k.usize("attention.iclr_lora_rank").unwrap_or(0),
                lora_value_res_mix: k.usize("attention.value_residual_mix_lora_rank").unwrap_or(0),
                lora_gate: k.usize("attention.gate_lora_rank").unwrap_or(0),
            },
            shortconv_l_cache: k.usize("shortconv.l_cache").unwrap_or(0),
            swiglu_clamp_exp: k.key_or_arr_f32("swiglu_clamp_exp", n_layer_all).unwrap_or_else(|| vec![0.0; n_layer_all]),
            swiglu_clamp_shexp: k.key_or_arr_f32("swiglu_clamp_shexp", n_layer_all).unwrap_or_else(|| vec![0.0; n_layer_all]),
            xielu,
            num_loops: k.usize("num_loops").unwrap_or(1),
            skip_loop_final_norm: k.bool("skip_loop_final_norm").unwrap_or(false),
        })
    }

    pub fn n_head(&self, l: usize) -> usize {
        per_layer(&self.n_head_arr, l, self.n_head)
    }

    pub fn n_head_kv(&self, l: usize) -> usize {
        per_layer(&self.n_head_kv_arr, l, self.n_head_kv)
    }

    pub fn n_ff(&self, l: usize) -> usize {
        per_layer(&self.n_ff_arr, l, self.n_ff)
    }

    /// Expert width of a layer, the dense width over the used count when the key is absent
    pub fn n_ff_exp(&self, l: usize) -> usize {
        let v = per_layer(&self.n_ff_exp_arr, l, self.n_ff_exp);
        if v == 0 && self.n_expert_used(l) > 0 { self.n_ff(l) / self.n_expert_used(l) } else { v }
    }

    pub fn n_expert_used(&self, l: usize) -> usize {
        per_layer(&self.n_expert_used_arr, l, self.n_expert_used)
    }

    pub fn head_dim_k(&self, l: usize) -> usize {
        if self.is_swa(l) { self.head_dim_k_swa } else { self.head_dim_k }
    }

    pub fn head_dim_v(&self, l: usize) -> usize {
        if self.is_swa(l) { self.head_dim_v_swa } else { self.head_dim_v }
    }

    pub fn rope_dims(&self, l: usize) -> usize {
        if self.is_swa(l) { self.rope_dims_swa } else { self.rope.dims }
    }

    pub fn is_recurrent(&self, l: usize) -> bool {
        self.recurrent.get(l).copied().unwrap_or(false)
    }

    pub fn is_swa(&self, l: usize) -> bool {
        self.swa.get(l).copied().unwrap_or(false)
    }

    pub fn has_rope(&self, l: usize) -> bool {
        self.rope_pattern.get(l).copied().unwrap_or(true)
    }

    pub fn is_moe_layer(&self, l: usize) -> bool {
        self.n_expert > 0 && l >= self.n_dense_lead
    }

    /// Fills the sliding window flags the way llama.cpp does, an array of flags or a scalar period
    pub fn swa_pattern(&mut self, m: &Metadata, default_n: usize, dense_first: bool) {
        let k = Keys { m, arch: &self.arch };
        let n_layer = self.n_layer;
        let flags: Vec<bool> = match k.arr_usize("attention.sliding_window_pattern") {
            Some(v) => v.into_iter().map(|x| x != 0).collect(),
            None => {
                let n = k.usize("attention.sliding_window_pattern").unwrap_or(default_n);
                (0..n_layer).map(|l| n == 0 || if dense_first { l % n != 0 } else { l % n < n - 1 }).collect()
            }
        };
        let mut swa: Vec<bool> = flags.into_iter().take(n_layer).collect();
        swa.resize(self.n_layer_all, false);
        self.swa = swa;
    }

    /// Fills the recurrent flags from attention.recurrent_layers or a full attention interval
    pub fn recurrent_pattern(&mut self, m: &Metadata, default_interval: usize) {
        let k = Keys { m, arch: &self.arch };
        let n_layer = self.n_layer;
        let n_all = self.n_layer_all;
        self.recurrent = k
            .key_or_arr("attention.recurrent_layers", n_all)
            .map(|v| v.into_iter().enumerate().map(|(l, x)| l < n_layer && x != 0).collect())
            .unwrap_or_else(|| {
                let interval = k.usize("full_attention_interval").unwrap_or(default_interval);
                (0..n_all).map(|l| l < n_layer && interval > 0 && (l + 1) % interval != 0).collect()
            });
    }

    /// Marks layers without kv heads as recurrent
    pub fn recurrent_from_kv_heads(&mut self) {
        let n_layer = self.n_layer;
        self.recurrent = (0..self.n_layer_all).map(|l| l < n_layer && self.n_head_kv(l) == 0).collect();
    }

    pub fn need_rms_eps(&self) -> Result<()> {
        if self.rms_eps == 0.0 {
            bail!("{}: attention.layer_norm_rms_epsilon is required", self.arch);
        }
        Ok(())
    }

    pub fn need_attention(&self) -> Result<()> {
        if self.n_head == 0 || self.head_dim_k == 0 {
            bail!("{}: attention head count and dimension are required", self.arch);
        }
        Ok(())
    }

    pub fn need_moe(&self) -> Result<()> {
        if self.n_expert == 0 || self.n_expert_used == 0 {
            bail!("{}: expert_count and expert_used_count are required", self.arch);
        }
        Ok(())
    }

    pub fn need_ssm(&self) -> Result<()> {
        let s = &self.ssm;
        if s.conv_kernel == 0 || s.state == 0 || s.dt_rank == 0 {
            bail!("{}: ssm.conv_kernel, state_size and time_step_rank are required", self.arch);
        }
        Ok(())
    }

    pub fn need_mla(&self) -> Result<()> {
        if self.mla.kv_lora == 0 || self.mla.head_k == 0 || self.mla.head_v == 0 {
            bail!("{}: attention.kv_lora_rank, key_length_mla and value_length_mla are required", self.arch);
        }
        Ok(())
    }

    pub fn need_indexer(&self) -> Result<()> {
        let i = &self.indexer;
        if i.n_head == 0 || i.head_dim == 0 || i.top_k == 0 {
            bail!("{}: attention.indexer.head_count, key_length and top_k are required", self.arch);
        }
        Ok(())
    }

    pub fn need_hc(&self) -> Result<()> {
        if self.hc.count <= 1 {
            bail!("{}: hyper_connection.count above one is required", self.arch);
        }
        Ok(())
    }

    pub fn need_ple(&self) -> Result<()> {
        let p = &self.ple;
        if p.ngram < 2 || p.heads_per_ngram == 0 || p.head_vocab.is_empty() {
            bail!("{}: ple.ngram_size, heads_per_ngram and head_vocab_sizes are required", self.arch);
        }
        Ok(())
    }

    pub fn need_kda(&self) -> Result<()> {
        if self.kda.head_dim == 0 || self.ssm.conv_kernel == 0 {
            bail!("{}: kda.head_dim and ssm.conv_kernel are required", self.arch);
        }
        Ok(())
    }

    pub fn need_swa(&self) -> Result<()> {
        if self.swa_window.is_none() || self.swa.len() < self.n_layer {
            bail!("{}: attention.sliding_window and a sliding window pattern are required", self.arch);
        }
        Ok(())
    }

    /// The yarn magnitude llama.cpp folds into the rope, before rope.scaling.attn_factor
    pub fn yarn_mscale(&self) -> f32 {
        if self.rope_scaling != RopeScaling::Yarn || self.rope.scale >= 1.0 {
            return 1.0;
        }
        let f = 1.0 / self.rope.scale;
        let get = |m: f32| 1.0 + 0.1 * m * f.ln();
        let log_mul = self.yarn_log_mul / 0.1;
        let base = if log_mul != 0.0 { get(1.0) / get(log_mul) } else { get(1.0) };
        base / (1.0 + 0.1 * f.ln())
    }

    /// Rope for a layer, the sliding window base and scale on window layers, yarn on full layers
    pub fn rope_for(&self, l: usize) -> RopeParams {
        let mut r = self.rope.clone();
        r.dims = self.rope_dims(l);
        if self.is_swa(l) {
            r.base = self.rope_base_swa;
            r.scale = self.rope_scale_swa;
            r.attn_factor = 1.0;
            r.yarn = None;
            return r;
        }
        r.attn_factor = self.yarn_mscale() * self.rope_attn_factor;
        if self.rope_scaling == RopeScaling::Yarn {
            r.yarn = Some(Yarn { ext_factor: 1.0, beta_fast: self.yarn_beta_fast, beta_slow: self.yarn_beta_slow });
        }
        r
    }

    /// Rope with yarn on every layer regardless of the window, for archs that never switch
    pub fn rope_full(&self) -> RopeParams {
        let mut r = self.rope.clone();
        r.attn_factor = self.yarn_mscale() * self.rope_attn_factor;
        if self.rope_scaling == RopeScaling::Yarn {
            r.yarn = Some(Yarn { ext_factor: 1.0, beta_fast: self.yarn_beta_fast, beta_slow: self.yarn_beta_slow });
        }
        r
    }
}

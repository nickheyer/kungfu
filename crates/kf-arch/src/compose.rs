//! Composition helpers shared by the architecture modules, each one names tensors the ggml way
use crate::{Hparams, Weights};
use anyhow::{Result, anyhow, bail};
use kf_core::{Act, Gating, GluLimit, RopeParams, Selection, WeightNorm, Wid, Window};
use kf_layers::{
    Attention, AttnGate, AttnTemp, BetaAlpha, Block, DeltaLayout, DeltaNet, DsaIndexer, Embedding, Ffn, GateKind, HcMix, HeadLayout, HeadNorm, IndexerNorm, Kda, Latent, Layer, LayerNorm, Linear,
    Mla, MlaQ, Moe, NextN, NormKind, OutGate, QkvProj, Residual, RmsNorm, RouterInput, SharedExperts,
};

pub fn blk(l: usize, name: &str) -> String {
    format!("blk.{l}.{name}.weight")
}

pub fn blk_bias(l: usize, name: &str) -> String {
    format!("blk.{l}.{name}.bias")
}

/// Marks present tensors used without building anything from them
pub fn claim(w: &mut Weights, names: &[String]) {
    for n in names {
        w.take_opt(n);
    }
}

/// Claims the optional scale and input scale siblings of a tensor
fn claim_scales(w: &mut Weights, prefix: &str) -> Option<Wid> {
    w.take_opt(&format!("{prefix}.input_scale"));
    w.take_opt(&format!("{prefix}.scale"))
}

pub fn linear(w: &mut Weights, l: usize, name: &str) -> Result<Linear> {
    let weight = w.take(&blk(l, name))?;
    let prefix = format!("blk.{l}.{name}");
    Ok(Linear { w: weight, b: w.take_opt(&blk_bias(l, name)), s: claim_scales(w, &prefix) })
}

pub fn linear_opt(w: &mut Weights, l: usize, name: &str) -> Option<Linear> {
    let weight = w.take_opt(&blk(l, name))?;
    let prefix = format!("blk.{l}.{name}");
    Some(Linear { w: weight, b: w.take_opt(&blk_bias(l, name)), s: claim_scales(w, &prefix) })
}

pub fn linear_named(w: &mut Weights, name: &str) -> Result<Linear> {
    Ok(Linear { w: w.take(&format!("{name}.weight"))?, b: w.take_opt(&format!("{name}.bias")), s: claim_scales(w, name) })
}

pub fn linear_named_opt(w: &mut Weights, name: &str) -> Option<Linear> {
    let weight = w.take_opt(&format!("{name}.weight"))?;
    Some(Linear { w: weight, b: w.take_opt(&format!("{name}.bias")), s: claim_scales(w, name) })
}

pub fn rms(w: &mut Weights, l: usize, name: &str, eps: f32) -> Result<RmsNorm> {
    Ok(RmsNorm { w: Some(w.take(&blk(l, name))?), b: w.take_opt(&blk_bias(l, name)), eps })
}

pub fn rms_opt(w: &mut Weights, l: usize, name: &str, eps: f32) -> Option<RmsNorm> {
    let weight = w.take_opt(&blk(l, name))?;
    Some(RmsNorm { w: Some(weight), b: w.take_opt(&blk_bias(l, name)), eps })
}

/// Rms norm whose tensor name carries no weight suffix, PLaMo style
pub fn rms_bare(w: &mut Weights, l: usize, name: &str, eps: f32) -> Result<RmsNorm> {
    Ok(RmsNorm { w: Some(w.take(&format!("blk.{l}.{name}"))?), b: None, eps })
}

pub fn rms_named(w: &mut Weights, name: &str, eps: f32) -> Result<RmsNorm> {
    Ok(RmsNorm { w: Some(w.take(&format!("{name}.weight"))?), b: w.take_opt(&format!("{name}.bias")), eps })
}

pub fn layer_norm(w: &mut Weights, l: usize, name: &str, eps: f32) -> Result<LayerNorm> {
    Ok(LayerNorm { w: Some(w.take(&blk(l, name))?), b: w.take_opt(&blk_bias(l, name)), eps })
}

pub fn layer_norm_opt(w: &mut Weights, l: usize, name: &str, eps: f32) -> Option<LayerNorm> {
    let weight = w.take_opt(&blk(l, name))?;
    Some(LayerNorm { w: Some(weight), b: w.take_opt(&blk_bias(l, name)), eps })
}

pub fn layer_norm_named(w: &mut Weights, name: &str, eps: f32) -> Result<LayerNorm> {
    Ok(LayerNorm { w: Some(w.take(&format!("{name}.weight"))?), b: w.take_opt(&format!("{name}.bias")), eps })
}

/// Token embedding and output head, the head falling back to the embedding unless the arch forbids it
pub fn embed_and_head(w: &mut Weights, tie: bool) -> Result<(Embedding, Linear)> {
    let embed = Embedding::plain(w.take("token_embd.weight")?);
    let out = match linear_named_opt(w, "output") {
        Some(t) => t,
        None if tie => Linear::new(embed.w),
        None => bail!("missing tensor output.weight"),
    };
    Ok((embed, out))
}

/// Token embedding, final rms norm, and output head, the head falling back to the embedding
pub fn head(hp: &Hparams, w: &mut Weights) -> Result<(Embedding, RmsNorm, Linear)> {
    let (embed, out) = embed_and_head(w, true)?;
    let norm = rms_named(w, "output_norm", hp.rms_eps)?;
    Ok((embed, norm, out))
}

pub fn pre_norm(w: &mut Weights, l: usize, name: &str, eps: f32, inner: impl Layer + 'static) -> Result<Residual> {
    Ok(Residual::new(rms(w, l, name, eps)?, inner))
}

/// Which q k norm tensors an attention layer carries and how their weights are laid out
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum QkNorm {
    #[default]
    None,
    /// attn_q_norm and attn_k_norm rms, one weight row shared by every head
    Shared,
    /// Taken only when the tensors exist
    SharedOpt,
    /// Rms norm with a weight row per head
    PerHeadRms,
    /// Layer norm with a weight and optional bias per head
    PerHeadLayer,
    /// One rms over the whole projected width
    FullWidth,
    /// Weightless rms per head
    Weightless,
    /// Rms over each half of the head
    HalfHead,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum GateOpt {
    #[default]
    None,
    /// attn_gate projection over the full attention width, sigmoid
    Full,
    /// The gate rides in q, one half per head
    InQ,
    /// attn_gate of one scalar per head, sigmoid
    PerHead,
    /// attn_gate with a per head bias
    FullWithHeadBias,
    /// attn_gate read from its width, per head when n_head wide, softplus
    SoftplusAuto,
    /// attn_gate over the full width, base two softplus
    Softplus2,
}

#[derive(Default)]
pub struct AttnOpts {
    pub qk_norm: QkNorm,
    pub v_norm: bool,
    pub norm_after_rope: bool,
    pub gate: GateOpt,
    pub rope: Option<RopeParams>,
    pub rope_offset: usize,
    pub window: Option<Window>,
    pub sinks: bool,
    pub sinks_opt: bool,
    pub softcap: f32,
    pub alibi: f32,
    pub clamp: f32,
    pub temp: Option<AttnTemp>,
    pub scale: Option<f32>,
    pub kv_source: Option<usize>,
    pub k_as_v: bool,
    pub sub_norm: bool,
    pub out_head_norm: bool,
    pub sparse: Option<DsaIndexer>,
    /// Norm kind of per head and full width norms
    pub norm_kind: Option<NormKind>,
    pub norm_eps: Option<f32>,
}

/// Fused attn_qkv with its bias or the concatenated q k v biases, else separate q k v
pub fn qkv_proj(w: &mut Weights, l: usize, k_optional: bool) -> Result<QkvProj> {
    if let Some(fused) = linear_opt(w, l, "attn_qkv") {
        let mut split_bias = None;
        if fused.b.is_none() {
            let qb = w.take_opt(&blk_bias(l, "attn_q"));
            let kb = w.take_opt(&blk_bias(l, "attn_k"));
            let vb = w.take_opt(&blk_bias(l, "attn_v"));
            if let (Some(q), Some(k), Some(v)) = (qb, kb, vb) {
                split_bias = Some([q, k, v]);
            }
        }
        return Ok(QkvProj::Fused { w: fused, split_bias });
    }
    let q = linear(w, l, "attn_q")?;
    let k = match linear_opt(w, l, "attn_k") {
        Some(k) => k,
        None if k_optional => q,
        None => bail!("missing tensor {}", blk(l, "attn_k")),
    };
    let v = match linear_opt(w, l, "attn_v") {
        Some(v) => v,
        None => k,
    };
    Ok(QkvProj::Separate { q, k, v })
}

fn head_norm(w: &mut Weights, l: usize, name: &str, qk: QkNorm, kind: NormKind, eps: f32) -> Result<Option<HeadNorm>> {
    let weight = w.take_opt(&blk(l, name));
    let bias = w.take_opt(&blk_bias(l, name));
    let layout = match qk {
        QkNorm::None => return Ok(None),
        QkNorm::Weightless => return Ok(Some(HeadNorm { kind, w: None, b: None, layout: HeadLayout::Shared, eps })),
        QkNorm::SharedOpt if weight.is_none() => return Ok(None),
        QkNorm::Shared | QkNorm::SharedOpt => HeadLayout::Shared,
        QkNorm::PerHeadRms | QkNorm::PerHeadLayer => HeadLayout::PerHead,
        QkNorm::FullWidth => HeadLayout::FullWidth,
        QkNorm::HalfHead => HeadLayout::HalfHead,
    };
    let weight = weight.ok_or_else(|| anyhow!("missing tensor {}", blk(l, name)))?;
    Ok(Some(HeadNorm { kind, w: Some(weight), b: bias, layout, eps }))
}

/// Grouped query attention from attn_q k v or a fused attn_qkv, with every variant the families need
pub fn attention(hp: &Hparams, w: &mut Weights, l: usize, o: AttnOpts) -> Result<Attention> {
    let qkv = qkv_proj(w, l, o.kv_source.is_some())?;
    attention_with_qkv(hp, w, l, qkv, o)
}

/// Attention around projections the caller already built
pub fn attention_with_qkv(hp: &Hparams, w: &mut Weights, l: usize, qkv: QkvProj, o: AttnOpts) -> Result<Attention> {
    let n_head = hp.n_head(l);
    let n_head_kv = hp.n_head_kv(l);
    let head_dim = hp.head_dim_k(l);
    let head_dim_v = hp.head_dim_v(l);
    let kind = o.norm_kind.unwrap_or(match o.qk_norm {
        QkNorm::PerHeadLayer => NormKind::Layer,
        _ => NormKind::Rms,
    });
    let eps = o.norm_eps.unwrap_or(if kind == NormKind::Rms { hp.rms_eps } else { hp.ln_eps });
    let q_norm = head_norm(w, l, "attn_q_norm", o.qk_norm, kind, eps)?;
    let k_norm = head_norm(w, l, "attn_k_norm", o.qk_norm, kind, eps)?;
    let gate = match o.gate {
        GateOpt::None => None,
        GateOpt::Full => Some(AttnGate { proj: Some(linear(w, l, "attn_gate")?), in_q: false, per_head: false, bias: None, kind: GateKind::Sigmoid }),
        GateOpt::InQ => Some(AttnGate { proj: None, in_q: true, per_head: false, bias: None, kind: GateKind::Sigmoid }),
        GateOpt::PerHead => Some(AttnGate { proj: Some(linear(w, l, "attn_gate")?), in_q: false, per_head: true, bias: None, kind: GateKind::Sigmoid }),
        GateOpt::FullWithHeadBias => {
            let weight = w.take(&blk(l, "attn_gate"))?;
            let bias = w.take(&blk_bias(l, "attn_gate"))?;
            Some(AttnGate { proj: Some(Linear { w: weight, b: None, s: claim_scales(w, &format!("blk.{l}.attn_gate")) }), in_q: false, per_head: true, bias: Some(bias), kind: GateKind::Sigmoid })
        }
        GateOpt::SoftplusAuto => {
            let p = linear(w, l, "attn_gate")?;
            let per_head = w.dims(p.w).get(1).copied() == Some(n_head);
            Some(AttnGate { proj: Some(p), in_q: false, per_head, bias: None, kind: GateKind::Softplus })
        }
        GateOpt::Softplus2 => Some(AttnGate { proj: Some(linear(w, l, "attn_gate")?), in_q: false, per_head: false, bias: None, kind: GateKind::Softplus2 }),
    };
    Ok(Attention {
        n_head,
        n_head_kv,
        head_dim,
        head_dim_v,
        qkv,
        k_as_v: o.k_as_v,
        q_norm,
        k_norm,
        v_norm: o.v_norm.then(|| HeadNorm::rms_weightless(hp.rms_eps)),
        norm_after_rope: o.norm_after_rope,
        rope: o.rope,
        rope_offset: o.rope_offset,
        out: linear(w, l, "attn_output")?,
        gate,
        clamp: o.clamp,
        temp: o.temp,
        window: o.window,
        scale: o.scale.unwrap_or(1.0 / (head_dim as f32).sqrt()),
        softcap: o.softcap,
        sinks: if o.sinks { Some(w.take(&blk(l, "attn_sinks"))?) } else if o.sinks_opt { w.take_opt(&blk(l, "attn_sinks")) } else { None },
        alibi: o.alibi,
        kv_source: o.kv_source,
        out_head_norm: o.out_head_norm,
        sub_norm: if o.sub_norm { Some(rms(w, l, "attn_sub_norm", hp.rms_eps)?) } else { None },
        sparse: o.sparse,
    })
}

/// The standard window of a layer when it is a sliding window layer
pub fn window_for(hp: &Hparams, l: usize) -> Option<Window> {
    match (hp.is_swa(l), hp.swa_window) {
        (true, Some(size)) => Some(Window { size, kind: hp.swa_type }),
        _ => None,
    }
}

#[derive(Default)]
pub struct FfnOpts {
    pub act: Option<Act>,
    pub limit: Option<GluLimit>,
    pub sub_norm: bool,
    pub act_scales: bool,
    /// Gate projection must exist
    pub require_gate: bool,
}

/// Dense feed forward, a missing gate with a double width up means the two are fused
pub fn ffn(hp: &Hparams, w: &mut Weights, l: usize) -> Result<Ffn> {
    ffn_with(hp, w, l, FfnOpts::default())
}

pub fn ffn_with(hp: &Hparams, w: &mut Weights, l: usize, o: FfnOpts) -> Result<Ffn> {
    let up = linear(w, l, "ffn_up")?;
    let gate = linear_opt(w, l, "ffn_gate");
    if o.require_gate && gate.is_none() {
        bail!("missing tensor {}", blk(l, "ffn_gate"));
    }
    let fused = gate.is_none() && w.dims(up.w).get(1).copied() == Some(2 * hp.n_ff(l));
    Ok(Ffn {
        up,
        gate,
        down: linear(w, l, "ffn_down")?,
        act: o.act.unwrap_or(Act::Silu),
        fused_gate_up: fused,
        limit: o.limit,
        act_scales: if o.act_scales { w.take_opt(&format!("blk.{l}.ffn.act.scales")) } else { None },
        sub_norm: if o.sub_norm { Some(rms(w, l, "ffn_sub_norm", hp.rms_eps)?) } else { None },
    })
}

/// Shared expert feed forward from the shexp tensors, gate optional, fused when up is double width
pub fn ffn_shexp(w: &mut Weights, l: usize, act: Act, limit: Option<GluLimit>, n_ff_shexp: usize) -> Result<Ffn> {
    let up = linear(w, l, "ffn_up_shexp")?;
    let gate = linear_opt(w, l, "ffn_gate_shexp");
    let fused = gate.is_none() && n_ff_shexp > 0 && w.dims(up.w).get(1).copied() == Some(2 * n_ff_shexp);
    Ok(Ffn { up, gate, down: linear(w, l, "ffn_down_shexp")?, act, fused_gate_up: fused, limit, act_scales: None, sub_norm: None })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SharedOpt {
    #[default]
    None,
    /// Shared experts summed in
    Plain,
    /// Shared experts gated by sigmoid of ffn_gate_inp_shexp
    Gated,
    /// Taken when the tensors exist
    PlainOpt,
}

pub struct MoeOpts {
    pub gating: Gating,
    pub selection: Selection,
    pub norm_weights: WeightNorm,
    pub scale: f32,
    pub act: Act,
    pub limit: Option<GluLimit>,
    pub shared: SharedOpt,
    pub shared_limit: Option<GluLimit>,
    pub shared_scale: f32,
    pub probs_bias: bool,
    pub probs_bias_required: bool,
    pub per_expert_bias: bool,
    pub weight_before_ffn: bool,
    pub router_input: RouterInput,
    pub latent: Option<Latent>,
    pub post_norm: Option<(Wid, f32)>,
    pub hash_router: bool,
    pub sum_scale: f32,
    pub n_ff_shexp: usize,
}

impl Default for MoeOpts {
    fn default() -> Self {
        Self {
            gating: Gating::Softmax,
            selection: Selection::Probs,
            norm_weights: WeightNorm::None,
            scale: 0.0,
            act: Act::Silu,
            limit: None,
            shared: SharedOpt::None,
            shared_limit: None,
            shared_scale: 1.0,
            probs_bias: false,
            probs_bias_required: false,
            per_expert_bias: false,
            weight_before_ffn: false,
            router_input: RouterInput::Same,
            latent: None,
            post_norm: None,
            hash_router: false,
            sum_scale: 1.0,
        n_ff_shexp: 0,
        }
    }
}

/// Routed experts from ffn_gate_inp and the stacked expert tensors
pub fn moe(hp: &Hparams, w: &mut Weights, l: usize, o: MoeOpts) -> Result<Moe> {
    let router = linear(w, l, "ffn_gate_inp")?;
    let n_expert = w.dims(router.w).get(1).copied().unwrap_or(hp.n_expert);
    let n_used = hp.n_expert_used(l);
    if n_expert == 0 || n_used == 0 {
        bail!("{}: expert_count and expert_used_count are required", hp.arch);
    }
    let (gate_exps, up_exps, fused) = match w.take_opt(&blk(l, "ffn_gate_up_exps")) {
        Some(fused) => (None, fused, true),
        None => (w.take_opt(&blk(l, "ffn_gate_exps")), w.take(&blk(l, "ffn_up_exps"))?, false),
    };
    let down_exps = w.take(&blk(l, "ffn_down_exps"))?;
    for name in ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", "ffn_gate_up_exps"] {
        w.take_opt(&format!("blk.{l}.{name}.input_scale"));
    }
    w.take_opt(&format!("blk.{l}.ffn_gate_exps.scale"));
    w.take_opt(&format!("blk.{l}.ffn_up_exps.scale"));
    w.take_opt(&format!("blk.{l}.ffn_gate_up_exps.scale"));
    let exps_scale = w.take_opt(&format!("blk.{l}.ffn_down_exps.scale"));
    let probs_bias = if o.probs_bias {
        let b = w.take_opt(&blk_bias(l, "exp_probs_b")).or_else(|| w.take_opt(&format!("blk.{l}.exp_probs_b")));
        if o.probs_bias_required && b.is_none() {
            bail!("missing tensor {}", blk_bias(l, "exp_probs_b"));
        }
        b
    } else {
        None
    };
    w.take_opt(&blk_bias(l, "exp_probs_b_vl"));
    let shared = match o.shared {
        SharedOpt::None => None,
        SharedOpt::PlainOpt if !w.has(&blk(l, "ffn_up_shexp")) => None,
        SharedOpt::Plain | SharedOpt::PlainOpt => Some(SharedExperts { ffn: ffn_shexp(w, l, o.act, o.shared_limit, o.n_ff_shexp)?, gate: None, scale: o.shared_scale }),
        SharedOpt::Gated => Some(SharedExperts { ffn: ffn_shexp(w, l, o.act, o.shared_limit, o.n_ff_shexp)?, gate: Some(linear(w, l, "ffn_gate_inp_shexp")?), scale: o.shared_scale }),
    };
    let (gate_b, up_b, down_b) = if o.per_expert_bias {
        (w.take_opt(&blk_bias(l, "ffn_gate_exps")), w.take_opt(&blk_bias(l, "ffn_up_exps")), w.take_opt(&blk_bias(l, "ffn_down_exps")))
    } else {
        (None, None, None)
    };
    Ok(Moe {
        router,
        probs_bias,
        gate_exps,
        up_exps,
        down_exps: Some(down_exps),
        gate_b,
        up_b,
        down_b,
        exps_scale,
        n_expert,
        n_used,
        gating: o.gating,
        selection: o.selection,
        norm_weights: o.norm_weights,
        scale: o.scale,
        act: o.act,
        limit: o.limit,
        groups: hp.expert_groups,
        fused_gate_up: fused,
        weight_before_ffn: o.weight_before_ffn,
        post_norm: o.post_norm,
        hash_router: if o.hash_router { Some(w.take(&blk(l, "ffn_gate_tid2eid"))?) } else { None },
        shared,
        latent: o.latent,
        router_input: o.router_input,
        chunk: None,
        sum_scale: o.sum_scale,
    })
}

/// The swiglu clamp of a layer from a per layer limit list, fused or unfused
pub fn clamp_limit(limits: &[f32], l: usize, fused: bool) -> Option<GluLimit> {
    let v = limits.get(l).copied().unwrap_or(0.0);
    (v > 1e-6).then(|| if fused { GluLimit::Fused(v) } else { GluLimit::Unfused(v) })
}

/// Gated delta net from the legacy grouped ssm_in layout, the split attn_qkv layout, or the merged tensors
pub fn delta_net(hp: &Hparams, w: &mut Weights, l: usize, head_map: kf_layers::HeadMap, out_gate: OutGate) -> Result<DeltaNet> {
    hp.need_ssm()?;
    let (in_proj, gate, layout) = if let Some(merged) = linear_opt(w, l, "attn_qkv_gate") {
        (merged, None, DeltaLayout::Split)
    } else if let Some(qkv) = linear_opt(w, l, "attn_qkv") {
        (qkv, Some(linear(w, l, "attn_gate")?), DeltaLayout::Split)
    } else {
        (linear(w, l, "ssm_in")?, None, DeltaLayout::LegacyGrouped)
    };
    let beta_alpha = if let Some(fused) = w.take_opt(&blk(l, "ssm_ba")) {
        BetaAlpha::Fused(fused)
    } else if let Some(merged) = w.take_opt(&blk(l, "ssm_beta_alpha")) {
        BetaAlpha::Fused(merged)
    } else {
        let beta = w.take(&blk(l, "ssm_beta"))?;
        let alpha = w.take(&blk(l, "ssm_alpha"))?;
        claim_scales(w, &format!("blk.{l}.ssm_beta"));
        claim_scales(w, &format!("blk.{l}.ssm_alpha"));
        BetaAlpha::Separate { beta, alpha }
    };
    Ok(DeltaNet {
        n_head_k: hp.ssm.n_group,
        n_head_v: hp.ssm.dt_rank,
        head_dim: hp.ssm.state,
        conv_kernel: hp.ssm.conv_kernel,
        layout,
        head_map,
        out_gate,
        in_proj,
        gate,
        conv1d: w.take(&blk(l, "ssm_conv1d"))?,
        dt_bias: w.take(&blk_bias(l, "ssm_dt"))?,
        a: w.take(&format!("blk.{l}.ssm_a"))?,
        beta_alpha,
        norm: w.take(&blk(l, "ssm_norm"))?,
        norm_eps: hp.rms_eps,
        l2_eps: hp.rms_eps,
        out: linear(w, l, "ssm_out")?,
    })
}

/// A short conv kernel stored as {K, 1, C, 1} or {K, 1, C} or {K, C}
fn conv_kernel(w: &mut Weights, l: usize, name: &str) -> Result<Wid> {
    w.take(&blk(l, name))
}

/// Kimi delta attention from separate q k v projections and three short convolutions
pub fn kda(hp: &Hparams, w: &mut Weights, l: usize, l2_eps: f32) -> Result<Kda> {
    hp.need_kda()?;
    let qkv = qkv_proj(w, l, false)?;
    let g_a = w.take(&blk(l, "ssm_g_a")).or_else(|_| w.take(&blk(l, "ssm_g")))?;
    Ok(Kda {
        n_head: hp.n_head(l),
        head_dim: hp.kda.head_dim,
        conv_kernel: hp.ssm.conv_kernel,
        gate_lower_bound: hp.kda.gate_lower_bound,
        a_is_exp: false,
        qkv,
        conv_q: conv_kernel(w, l, "ssm_conv1d_q")?,
        conv_k: conv_kernel(w, l, "ssm_conv1d_k")?,
        conv_v: conv_kernel(w, l, "ssm_conv1d_v")?,
        f_a: w.take(&blk(l, "ssm_f_a"))?,
        f_b: w.take_opt(&blk(l, "ssm_f_b")),
        g_a,
        g_b: w.take_opt(&blk(l, "ssm_g_b")),
        beta: w.take(&blk(l, "ssm_beta"))?,
        dt_bias: w.take(&blk_bias(l, "ssm_dt"))?,
        a: w.take(&format!("blk.{l}.ssm_a"))?,
        norm: w.take(&blk(l, "ssm_norm"))?,
        norm_eps: hp.rms_eps,
        l2_eps,
        out: linear(w, l, "attn_output")?,
    })
}

/// DeepSeek style indexer with attn_k and attn_q_b, or the Qwen k_proj and q_proj naming
pub fn indexer(hp: &Hparams, w: &mut Weights, l: usize, kind: kf_core::IndexerKind, rope: Option<RopeParams>, rope_tail: bool, source: Option<usize>) -> Result<DsaIndexer> {
    hp.need_indexer()?;
    let (k, q) = match linear_opt(w, l, "indexer.attn_k") {
        Some(k) => (k, linear(w, l, "indexer.attn_q_b")?),
        None => (linear(w, l, "indexer.k_proj")?, linear(w, l, "indexer.q_proj")?),
    };
    let k_norm_w = w.take(&blk(l, "indexer.k_norm"))?;
    let k_norm_b = w.take_opt(&blk_bias(l, "indexer.k_norm"));
    let k_norm = IndexerNorm { kind: if k_norm_b.is_some() { NormKind::Layer } else { NormKind::Rms }, w: k_norm_w, b: k_norm_b, eps: if k_norm_b.is_some() { hp.ln_eps } else { hp.rms_eps } };
    Ok(DsaIndexer {
        n_head: hp.indexer.n_head,
        head_dim: hp.indexer.head_dim,
        top_k: hp.indexer.top_k,
        kind,
        proj: linear_opt(w, l, "indexer.proj"),
        k: Some(k),
        k_norm: Some(k_norm),
        q,
        q_norm: w.take_opt(&blk(l, "indexer.q_norm")),
        rope,
        rope_tail,
        hadamard: false,
        pool_gate: linear_opt(w, l, "indexer_compressor_gate"),
        pool_ape: w.take_opt(&blk(l, "indexer_compressor_ape")),
        source,
    })
}

/// A shared indexer layer, its optional tensors claimed and its selection read from the source layer
pub fn indexer_shared(hp: &Hparams, w: &mut Weights, l: usize, kind: kf_core::IndexerKind, source: usize, q_placeholder: Wid) -> DsaIndexer {
    for n in ["indexer.attn_k", "indexer.attn_q_b", "indexer.k_proj", "indexer.q_proj", "indexer.proj", "indexer.q_norm", "indexer_compressor_gate", "indexer_compressor_ape"] {
        w.take_opt(&blk(l, n));
        w.take_opt(&blk_bias(l, n));
    }
    w.take_opt(&blk(l, "indexer.k_norm"));
    w.take_opt(&blk_bias(l, "indexer.k_norm"));
    DsaIndexer {
        n_head: hp.indexer.n_head,
        head_dim: hp.indexer.head_dim,
        top_k: hp.indexer.top_k,
        kind,
        proj: None,
        k: None,
        k_norm: None,
        q: Linear::new(q_placeholder),
        q_norm: None,
        rope: None,
        rope_tail: false,
        hadamard: false,
        pool_gate: None,
        pool_ape: None,
        source: Some(source),
    }
}

pub struct MlaOpts {
    pub rope: Option<RopeParams>,
    pub sparse: Option<DsaIndexer>,
    pub scale: f32,
    pub temp: Option<AttnTemp>,
    pub gate: GateOpt,
    pub sinks: bool,
    pub window: Option<Window>,
    pub k_pe_norm: bool,
    pub legacy_pe_first: bool,
    /// Head dims of this layer, the swa variants on window layers
    pub head_k: usize,
    pub head_v: usize,
    pub kv_lora: usize,
}

/// Latent attention, q through a low rank pair when attn_q_a exists, absorbed k_b v_b or legacy kv_b
pub fn mla(hp: &Hparams, w: &mut Weights, l: usize, o: MlaOpts) -> Result<Mla> {
    let q = match linear_opt(w, l, "attn_q_a") {
        Some(a) => MlaQ::LowRank { a, a_norm: rms(w, l, "attn_q_a_norm", hp.rms_eps)?, b: linear(w, l, "attn_q_b")? },
        None => MlaQ::Full(linear(w, l, "attn_q")?),
    };
    let rope_dim = o.rope.as_ref().map_or(0, |r| r.dims);
    let kv_b = linear_opt(w, l, "attn_kv_b");
    let gate = match o.gate {
        GateOpt::None => None,
        GateOpt::PerHead => Some(AttnGate { proj: Some(linear(w, l, "attn_gate")?), in_q: false, per_head: true, bias: None, kind: GateKind::Sigmoid }),
        GateOpt::Full => Some(AttnGate { proj: Some(linear(w, l, "attn_gate")?), in_q: false, per_head: false, bias: None, kind: GateKind::Sigmoid }),
        other => bail!("{}: attention gate {other:?} is not an MLA gate", hp.arch),
    };
    Ok(Mla {
        n_head: hp.n_head(l),
        qk_nope_dim: o.head_k - rope_dim,
        qk_rope_dim: rope_dim,
        v_head_dim: o.head_v,
        kv_lora_rank: o.kv_lora,
        q,
        kv_a_mqa: linear(w, l, "attn_kv_a_mqa")?,
        kv_a_norm: rms(w, l, "attn_kv_a_norm", hp.rms_eps)?,
        kv_b,
        k_b: w.take_opt(&blk(l, "attn_k_b")),
        v_b: w.take_opt(&blk(l, "attn_v_b")),
        k_pe_norm: if o.k_pe_norm { Some(rms(w, l, "attn_k_norm", hp.rms_eps)?) } else { None },
        legacy_pe_first: o.legacy_pe_first,
        out: linear(w, l, "attn_output")?,
        rope: o.rope,
        scale: o.scale,
        temp: o.temp,
        gate,
        sinks: if o.sinks { Some(w.take(&blk(l, "attn_sinks"))?) } else { None },
        indexer: o.sparse,
        window: o.window,
    })
}

/// The MLA kq scale of DeepSeek, the yarn mscale squared over sqrt of the mla head dim
pub fn mla_scale(hp: &Hparams, head_k: usize) -> f32 {
    let mscale = if hp.rope.scale < 1.0 && hp.rope_scaling == crate::hparams::RopeScaling::Yarn {
        let f = 1.0 / hp.rope.scale;
        let log_mul = hp.yarn_log_mul / 0.1;
        hp.yarn_mscale() * hp.rope_attn_factor * (1.0 + 0.1 * log_mul * f.ln())
    } else {
        hp.rope_attn_factor
    };
    mscale * mscale / (head_k as f32).sqrt()
}

/// Every hyper connection weight a site may carry, each one optional
pub fn hc_mix(w: &mut Weights, prefix: &str) -> HcMix {
    HcMix {
        fn_: w.take_opt(&format!("{prefix}_fn.weight")),
        base: w.take_opt(&format!("{prefix}_base.weight")),
        scale: w.take_opt(&format!("{prefix}_scale.weight")),
        norm: w.take_opt(&format!("{prefix}_norm.weight")),
        down: w.take_opt(&format!("{prefix}_down.weight")),
        up: w.take_opt(&format!("{prefix}_up.weight")),
        inject: w.take_opt(&format!("{prefix}_inject.weight")),
        down_inject: w.take_opt(&format!("{prefix}_down_inject.weight")),
    }
}

pub fn hc_mix_present(m: &HcMix) -> bool {
    !m.weights().is_empty()
}

/// The main table, head and final norm a draft head falls back to
pub struct Main {
    pub embed: Wid,
    pub head: Linear,
    pub norm: RmsNorm,
}

/// Next token prediction head at layer l wrapping a block, heads falling back to the main ones
pub fn nextn(hp: &Hparams, w: &mut Weights, l: usize, block: Block, n_streams: usize, pre_head: Option<Box<dyn Layer>>, main: &Main) -> Result<NextN> {
    let head_norm: Box<dyn Layer> = match rms_opt(w, l, "nextn.shared_head_norm", hp.rms_eps) {
        Some(n) => Box::new(n),
        None => Box::new(main.norm),
    };
    Ok(NextN {
        eh_proj: linear(w, l, "nextn.eh_proj")?,
        enorm: rms(w, l, "nextn.enorm", hp.rms_eps)?,
        hnorm: rms(w, l, "nextn.hnorm", hp.rms_eps)?,
        hnorm_per_stream: false,
        embed_tokens: w.take_opt(&blk(l, "nextn.embed_tokens")).unwrap_or(main.embed),
        head: linear_opt(w, l, "nextn.shared_head_head").unwrap_or(main.head),
        head_norm,
        block,
        n_streams,
        n_embd: hp.n_embd,
        pre_head,
    })
}

/// A single stream model with the defaults every family starts from
pub fn model(embed: Embedding, blocks: Vec<Block>, final_norm: impl Layer + 'static, head: Linear) -> kf_layers::Model {
    kf_layers::Model::plain(embed, blocks, final_norm, head)
}

/// Logit scale as the model applies it, one when the key is absent
pub fn logit_scale(hp: &Hparams) -> f32 {
    if hp.logit_scale == 0.0 { 1.0 } else { hp.logit_scale }
}

/// Whether the file carries a next token prediction layer at l
pub fn has_nextn(w: &Weights, l: usize) -> bool {
    w.has(&blk(l, "nextn.eh_proj"))
}

pub fn finish(hp: &Hparams, w: &Weights, model: kf_layers::Model) -> Result<kf_layers::Model> {
    let unused = w.unused();
    if !unused.is_empty() {
        return Err(anyhow!("{}: {} tensors unclaimed by the composition, first {:?}", hp.arch, unused.len(), &unused[..unused.len().min(8)]));
    }
    Ok(model)
}

pub fn wid_list(ws: &[Option<Wid>]) -> Vec<Wid> {
    ws.iter().copied().flatten().collect()
}

/// rope_factors_long when the context exceeds the trained length, else rope_factors_short, both claimed
pub fn longrope_factors_select(hp: &Hparams, w: &mut Weights) -> Result<Wid> {
    let long = w.take_opt("rope_factors_long.weight");
    let short = w.take_opt("rope_factors_short.weight");
    let chosen = if hp.n_ctx > hp.rope.orig_ctx { long } else { short };
    chosen.ok_or_else(|| anyhow!("{}: rope_factors_long.weight and rope_factors_short.weight are required for longrope", hp.arch))
}

/// The rope factors a llama style file carries, rope_freqs or the longrope pair
pub fn rope_factors(hp: &Hparams, w: &mut Weights) -> Result<Option<Wid>> {
    if hp.rope_scaling == crate::hparams::RopeScaling::LongRope {
        return Ok(Some(longrope_factors_select(hp, w)?));
    }
    Ok(w.take_opt("rope_freqs.weight"))
}

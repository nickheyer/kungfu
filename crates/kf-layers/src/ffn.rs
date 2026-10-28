use crate::basic::{Linear, RmsNorm};
use anyhow::{Result, bail};
use kf_core::{Act, Ctx, Gating, GluLimit, GraphBuilder, Groups, Layer, Op, Selection, StateNeed, Tid, WeightNorm, Wid};

/// Dense feed forward, gated when gate is present or fused into up
pub struct Ffn {
    pub up: Linear,
    pub gate: Option<Linear>,
    pub down: Linear,
    pub act: Act,
    /// up holds gate and up side by side, halves split inside the activation
    pub fused_gate_up: bool,
    pub limit: Option<GluLimit>,
    /// Divides the activation by a per channel scale, the AWQ act scales
    pub act_scales: Option<Wid>,
    /// Norm on the hidden activation before the down projection
    pub sub_norm: Option<RmsNorm>,
}

impl Ffn {
    pub fn plain(up: Linear, gate: Option<Linear>, down: Linear, act: Act) -> Self {
        Self { up, gate, down, act, fused_gate_up: false, limit: None, act_scales: None, sub_norm: None }
    }
    pub fn apply(&self, g: &mut GraphBuilder, x: Tid) -> Result<Tid> {
        let u = self.up.apply(g, x);
        let mut h = match (&self.gate, self.fused_gate_up) {
            (Some(gate), _) => {
                let gt = gate.apply(g, x);
                g.push(Op::GatedAct { act: self.act, limit: self.limit }, &[gt, u])
            }
            (None, true) => g.push(Op::GatedActFused { act: self.act, limit: self.limit }, &[u]),
            (None, false) => match self.act {
                Act::Silu => g.push(Op::Silu, &[u]),
                Act::Gelu => g.push(Op::Gelu, &[u]),
                Act::GeluErf => g.push(Op::GeluErf, &[u]),
                Act::Relu => g.push(Op::Relu, &[u]),
                Act::ReluSqr => g.push(Op::ReluSqr, &[u]),
                Act::SwigluOai => bail!("swiglu oai needs a gate projection"),
            },
        };
        if let Some(s) = self.act_scales {
            h = g.push(Op::DivWeight { weight: s }, &[h]);
        }
        if let Some(n) = &self.sub_norm {
            h = n.apply(g, h);
        }
        Ok(self.down.apply(g, h))
    }
}

impl Layer for Ffn {
    fn forward(&self, g: &mut GraphBuilder, x: Tid, _ctx: &Ctx) -> Result<Tid> {
        self.apply(g, x)
    }
    fn weights(&self) -> Vec<Wid> {
        let mut w = self.up.weights();
        w.extend(self.gate.iter().flat_map(Layer::weights));
        w.extend(self.down.weights());
        w.extend(self.act_scales);
        w.extend(self.sub_norm.iter().flat_map(Layer::weights));
        w
    }
}

/// Ungated activation applied in place, used for xielu and other single input activations
pub struct Activation {
    pub op: Op,
}

impl Layer for Activation {
    fn forward(&self, g: &mut GraphBuilder, x: Tid, _ctx: &Ctx) -> Result<Tid> {
        Ok(g.push(self.op.clone(), &[x]))
    }
    fn weights(&self) -> Vec<Wid> {
        Vec::new()
    }
}

/// up then activation then down, the activation given as a layer
pub struct ActFfn {
    pub up: Linear,
    pub act: Box<dyn Layer>,
    pub down: Linear,
}

impl Layer for ActFfn {
    fn forward(&self, g: &mut GraphBuilder, x: Tid, ctx: &Ctx) -> Result<Tid> {
        let u = self.up.apply(g, x);
        let h = self.act.forward(g, u, ctx)?;
        Ok(self.down.apply(g, h))
    }
    fn weights(&self) -> Vec<Wid> {
        let mut w = self.up.weights();
        w.extend(self.act.weights());
        w.extend(self.down.weights());
        w
    }
}

pub struct SharedExperts {
    pub ffn: Ffn,
    pub gate: Option<Linear>,
    pub scale: f32,
}

/// Low rank path around the routed experts
pub struct Latent {
    pub down: Linear,
    pub up: Linear,
    pub pre_norm: Option<RmsNorm>,
    pub post_norm: Option<RmsNorm>,
}

/// Where the router reads its logits from, the expert input, the block input, or a transform of the sublayer input
pub enum RouterInput {
    Same,
    BlockInput,
    Transformed(Box<dyn Layer>),
}

/// Second routed pass over chunk experts fed by the routed output
pub struct ChunkExperts {
    pub gate: Wid,
    pub up: Wid,
    pub down: Wid,
    pub n_chunk: usize,
    pub n_used: usize,
    pub div: usize,
    pub scale: f32,
}

/// Routed experts, the hot path of every MoE model and one fused op
pub struct Moe {
    pub router: Linear,
    pub probs_bias: Option<Wid>,
    pub gate_exps: Option<Wid>,
    pub up_exps: Wid,
    pub down_exps: Option<Wid>,
    pub gate_b: Option<Wid>,
    pub up_b: Option<Wid>,
    pub down_b: Option<Wid>,
    pub exps_scale: Option<Wid>,
    pub n_expert: usize,
    pub n_used: usize,
    pub gating: Gating,
    pub selection: Selection,
    pub norm_weights: WeightNorm,
    pub scale: f32,
    pub act: Act,
    pub limit: Option<GluLimit>,
    pub groups: Option<Groups>,
    pub fused_gate_up: bool,
    pub weight_before_ffn: bool,
    pub post_norm: Option<(Wid, f32)>,
    /// Token id to expert id table that replaces the top k
    pub hash_router: Option<Wid>,
    pub shared: Option<SharedExperts>,
    pub latent: Option<Latent>,
    pub router_input: RouterInput,
    pub chunk: Option<ChunkExperts>,
    pub sum_scale: f32,
}

impl Moe {
    fn op(&self, router_in: bool, selected_in: bool) -> Op {
        Op::MoeFfn {
            router: self.router.w,
            router_bias: self.router.b,
            probs_bias: self.probs_bias,
            gate_exps: self.gate_exps,
            up_exps: self.up_exps,
            down_exps: self.down_exps,
            gate_b: self.gate_b,
            up_b: self.up_b,
            down_b: self.down_b,
            exps_scale: self.exps_scale,
            n_expert: self.n_expert,
            n_used: self.n_used,
            gating: self.gating,
            selection: self.selection,
            norm_weights: self.norm_weights,
            scale: if self.scale == 0.0 { 1.0 } else { self.scale },
            act: self.act,
            limit: self.limit,
            groups: self.groups,
            fused_gate_up: self.fused_gate_up,
            weight_before_ffn: self.weight_before_ffn,
            index_div: 1,
            post_norm: self.post_norm.map(|p| p.0),
            post_norm_eps: self.post_norm.map_or(0.0, |p| p.1),
            router_in,
            selected_in,
        }
    }
}

impl Layer for Moe {
    fn forward(&self, g: &mut GraphBuilder, x: Tid, ctx: &Ctx) -> Result<Tid> {
        let router_x = match &self.router_input {
            RouterInput::Same => None,
            RouterInput::BlockInput => Some(ctx.block_input),
            RouterInput::Transformed(t) => Some(t.forward(g, ctx.residual_input, ctx)?),
        };
        let mut m = x;
        if let Some(l) = &self.latent {
            m = l.down.apply(g, m);
            if let Some(n) = &l.pre_norm {
                m = n.apply(g, m);
            }
        }
        let needs_logits = router_x.is_some() || self.latent.is_some() || self.chunk.is_some() || self.hash_router.is_some();
        let logits = needs_logits.then(|| self.router.apply(g, router_x.unwrap_or(x)));
        let mut inputs = vec![m];
        if let Some(l) = logits {
            inputs.push(l);
        }
        if let Some(h) = self.hash_router {
            inputs.push(g.push(Op::Gather { weight: h }, &[ctx.tokens]));
        }
        let mut routed = g.push(self.op(logits.is_some(), self.hash_router.is_some()), &inputs);
        if let Some(l) = &self.latent {
            if let Some(n) = &l.post_norm {
                routed = n.apply(g, routed);
            }
            routed = l.up.apply(g, routed);
        }
        if let Some(c) = &self.chunk {
            let mut op = self.op(true, false);
            if let Op::MoeFfn { gate_exps, up_exps, down_exps, n_expert, n_used, index_div, probs_bias, selection, .. } = &mut op {
                *gate_exps = Some(c.gate);
                *up_exps = c.up;
                *down_exps = Some(c.down);
                *n_expert = c.n_chunk;
                *n_used = c.n_used;
                *index_div = c.div;
                *probs_bias = None;
                *selection = Selection::SigmoidLogits;
            }
            let logits = logits.ok_or_else(|| anyhow::anyhow!("chunk experts need router logits"))?;
            let cx = g.push(op, &[routed, logits]);
            let cx = g.scale(cx, c.scale);
            routed = g.add(routed, cx);
        }
        let Some(shared) = &self.shared else { return Ok(g.scale(routed, self.sum_scale)) };
        let mut s = shared.ffn.apply(g, x)?;
        if let Some(gate) = &shared.gate {
            let sg = gate.apply(g, x);
            let sg = g.push(Op::Sigmoid, &[sg]);
            s = g.mul(s, sg);
        }
        let s = g.scale(s, shared.scale);
        let out = g.add(routed, s);
        Ok(g.scale(out, self.sum_scale))
    }

    fn weights(&self) -> Vec<Wid> {
        let mut w = self.router.weights();
        w.extend(self.probs_bias);
        w.extend(self.gate_exps);
        w.push(self.up_exps);
        w.extend(self.down_exps);
        w.extend([self.gate_b, self.up_b, self.down_b, self.exps_scale, self.hash_router].into_iter().flatten());
        w.extend(self.post_norm.map(|p| p.0));
        if let Some(s) = &self.shared {
            w.extend(s.ffn.weights());
            w.extend(s.gate.iter().flat_map(Layer::weights));
        }
        if let Some(l) = &self.latent {
            w.extend(l.down.weights());
            w.extend(l.up.weights());
            w.extend(l.pre_norm.iter().flat_map(Layer::weights));
            w.extend(l.post_norm.iter().flat_map(Layer::weights));
        }
        if let RouterInput::Transformed(t) = &self.router_input {
            w.extend(t.weights());
        }
        if let Some(c) = &self.chunk {
            w.extend([c.gate, c.up, c.down]);
        }
        w
    }

    fn states(&self) -> Vec<StateNeed> {
        Vec::new()
    }
}

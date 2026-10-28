use crate::basic::{Linear, RmsNorm};
use anyhow::Result;
use kf_core::{Ctx, GraphBuilder, Layer, StateNeed, Tid, Wid};

/// Mamba selective scan with per element decay, dt B C through x_proj and dt_proj
pub struct Mamba1 {
    pub d_inner: usize,
    pub d_state: usize,
    pub dt_rank: usize,
    pub conv_kernel: usize,
    pub in_proj: Linear,
    pub conv1d: Wid,
    pub conv_bias: Option<Wid>,
    pub x_proj: Linear,
    pub dt_proj: Linear,
    /// Minus exp of A_log, d_state by d_inner
    pub a: Wid,
    pub d: Wid,
    pub out: Linear,
    pub dt_norm: Option<RmsNorm>,
    pub b_norm: Option<RmsNorm>,
    pub c_norm: Option<RmsNorm>,
    /// Weightless rms norm on dt B C when the norms are absent
    pub dt_b_c_rms: bool,
    pub eps: f32,
}

impl Layer for Mamba1 {
    fn forward(&self, g: &mut GraphBuilder, x: Tid, ctx: &Ctx) -> Result<Tid> {
        mamba1_graph(g, self, x, ctx)
    }
    fn weights(&self) -> Vec<Wid> {
        let mut w = self.in_proj.weights();
        w.push(self.conv1d);
        w.extend(self.conv_bias);
        w.extend(self.x_proj.weights());
        w.extend(self.dt_proj.weights());
        w.extend([self.a, self.d]);
        w.extend(self.out.weights());
        w.extend(self.dt_norm.iter().flat_map(Layer::weights));
        w.extend(self.b_norm.iter().flat_map(Layer::weights));
        w.extend(self.c_norm.iter().flat_map(Layer::weights));
        w
    }
    fn states(&self) -> Vec<StateNeed> {
        vec![StateNeed::Recurrent { conv_state: (self.conv_kernel - 1) * self.d_inner, ssm_state: self.d_state * self.d_inner }]
    }
}

/// Mamba2 state space duality scan with scalar decay per head and grouped B C
pub struct Mamba2 {
    pub d_inner: usize,
    pub d_state: usize,
    pub n_head: usize,
    pub n_group: usize,
    pub conv_kernel: usize,
    pub in_proj: Linear,
    pub conv1d: Wid,
    pub conv_bias: Option<Wid>,
    pub dt_bias: Wid,
    pub a: Wid,
    pub d: Wid,
    /// Grouped rms norm weight of d_inner over n_group by n_group
    pub norm: Option<Wid>,
    pub out: Linear,
    pub eps: f32,
}

impl Layer for Mamba2 {
    fn forward(&self, g: &mut GraphBuilder, x: Tid, ctx: &Ctx) -> Result<Tid> {
        mamba2_graph(g, self, x, ctx)
    }
    fn weights(&self) -> Vec<Wid> {
        let mut w = self.in_proj.weights();
        w.push(self.conv1d);
        w.extend(self.conv_bias);
        w.extend([self.dt_bias, self.a, self.d]);
        w.extend(self.norm);
        w.extend(self.out.weights());
        w
    }
    fn states(&self) -> Vec<StateNeed> {
        vec![StateNeed::Recurrent { conv_state: (self.conv_kernel - 1) * (self.d_inner + 2 * self.n_group * self.d_state), ssm_state: self.d_state * self.d_inner }]
    }
}

/// PLaMo2 variant, z and x interleaved per head, dt B C through x_proj with norms, scalar decay per head
pub struct Plamo2Mamba {
    pub d_inner: usize,
    pub d_state: usize,
    pub n_head: usize,
    pub dt_dim: usize,
    pub conv_kernel: usize,
    pub in_proj: Linear,
    pub conv1d: Wid,
    pub x_proj: Linear,
    pub dt_proj: Linear,
    pub dt_norm: RmsNorm,
    pub b_norm: RmsNorm,
    pub c_norm: RmsNorm,
    pub a: Wid,
    pub d: Wid,
    pub out: Linear,
    pub eps: f32,
}

impl Layer for Plamo2Mamba {
    fn forward(&self, g: &mut GraphBuilder, x: Tid, ctx: &Ctx) -> Result<Tid> {
        plamo2_mamba_graph(g, self, x, ctx)
    }
    fn weights(&self) -> Vec<Wid> {
        let mut w = self.in_proj.weights();
        w.push(self.conv1d);
        w.extend(self.x_proj.weights());
        w.extend(self.dt_proj.weights());
        w.extend(self.dt_norm.weights());
        w.extend(self.b_norm.weights());
        w.extend(self.c_norm.weights());
        w.extend([self.a, self.d]);
        w.extend(self.out.weights());
        w
    }
    fn states(&self) -> Vec<StateNeed> {
        vec![StateNeed::Recurrent { conv_state: (self.conv_kernel - 1) * self.d_inner, ssm_state: self.d_state * self.d_inner }]
    }
}

use crate::util::{add_weight, mul_per_head, mul_weight, rms, unary};
use kf_core::Op;

fn silu_gate(g: &mut GraphBuilder, z: Tid, y: Tid) -> Tid {
    let s = unary(g, Op::Silu, z);
    g.mul(s, y)
}

/// Mamba selective scan with the dt B C projection, optional norms, per element decay and the gated skip
pub fn mamba1_graph(g: &mut GraphBuilder, m: &Mamba1, x: Tid, ctx: &Ctx) -> Result<Tid> {
    let (di, n, r) = (m.d_inner, m.d_state, m.dt_rank);
    let xz = m.in_proj.apply(g, x);
    let xs = g.cols(xz, 0, di);
    let z = g.cols(xz, di, di);
    let xc = g.push(Op::CausalConv { layer: ctx.layer, slot: 0, kernel: m.conv_kernel, dilation: 1, weight: m.conv1d, bias: m.conv_bias }, &[xs]);
    let xc = unary(g, Op::Silu, xc);
    let xdb = m.x_proj.apply(g, xc);
    let mut dt = g.cols(xdb, 0, r);
    let mut b = g.cols(xdb, r, n);
    let mut c = g.cols(xdb, r + n, n);
    let all_norms = m.dt_norm.is_some() && m.b_norm.is_some() && m.c_norm.is_some();
    if m.dt_b_c_rms || all_norms {
        dt = rms(g, dt, m.dt_norm.and_then(|w| w.w), m.eps);
        b = rms(g, b, m.b_norm.and_then(|w| w.w), m.eps);
        c = rms(g, c, m.c_norm.and_then(|w| w.w), m.eps);
    }
    let dt = m.dt_proj.apply(g, dt);
    let y = g.push(Op::SsmScan { layer: ctx.layer, n_head: di, head_dim: 1, d_state: n, n_group: 1, a: m.a, mamba2: false }, &[xc, dt, b, c]);
    let skip = mul_weight(g, xc, m.d);
    let y = g.add(y, skip);
    let y = silu_gate(g, z, y);
    Ok(m.out.apply(g, y))
}

/// Mamba2 scan with scalar decay per head, grouped B C, the D skip, the z gate and the grouped norm
pub fn mamba2_graph(g: &mut GraphBuilder, m: &Mamba2, x: Tid, ctx: &Ctx) -> Result<Tid> {
    let t = ctx.n_tokens;
    let (di, n, nh, ng) = (m.d_inner, m.d_state, m.n_head, m.n_group.max(1));
    let p = di / nh;
    let conv_dim = di + 2 * ng * n;
    let zxbcdt = m.in_proj.apply(g, x);
    let z = g.cols(zxbcdt, 0, di);
    let xbc = g.cols(zxbcdt, di, conv_dim);
    let dt = g.cols(zxbcdt, di + conv_dim, nh);
    let xbc = g.push(Op::CausalConv { layer: ctx.layer, slot: 0, kernel: m.conv_kernel, dilation: 1, weight: m.conv1d, bias: m.conv_bias }, &[xbc]);
    let xbc = unary(g, Op::Silu, xbc);
    let xs = g.cols(xbc, 0, di);
    let b = g.cols(xbc, di, ng * n);
    let c = g.cols(xbc, di + ng * n, ng * n);
    let dt = add_weight(g, dt, m.dt_bias);
    let y = g.push(Op::SsmScan { layer: ctx.layer, n_head: nh, head_dim: p, d_state: n, n_group: ng, a: m.a, mamba2: true }, &[xs, dt, b, c]);
    let skip = mul_per_head(g, xs, m.d, t, nh, p);
    let y = g.add(y, skip);
    let mut y = silu_gate(g, z, y);
    if let Some(w) = m.norm {
        let r = g.reshape(y, t * ng, di / ng);
        let r = rms(g, r, None, m.eps);
        let r = g.reshape(r, t, di);
        y = mul_weight(g, r, w);
    }
    Ok(m.out.apply(g, y))
}

/// PLaMo2 scan, z and x interleaved per head, normed dt B C, scalar decay per head, no conv bias
pub fn plamo2_mamba_graph(g: &mut GraphBuilder, m: &Plamo2Mamba, x: Tid, ctx: &Ctx) -> Result<Tid> {
    let t = ctx.n_tokens;
    let (di, n, nh) = (m.d_inner, m.d_state, m.n_head);
    let p = di / nh;
    let zx = m.in_proj.apply(g, x);
    let per = g.reshape(zx, t * nh, 2 * p);
    let z = g.cols(per, 0, p);
    let xs = g.cols(per, p, p);
    let z = g.reshape(z, t, di);
    let xs = g.reshape(xs, t, di);
    let xc = g.push(Op::CausalConv { layer: ctx.layer, slot: 0, kernel: m.conv_kernel, dilation: 1, weight: m.conv1d, bias: None }, &[xs]);
    let xc = unary(g, Op::Silu, xc);
    let bcdt = m.x_proj.apply(g, xc);
    let b = g.cols(bcdt, 0, n);
    let c = g.cols(bcdt, n, n);
    let dt = g.cols(bcdt, 2 * n, m.dt_dim);
    let b = m.b_norm.apply(g, b);
    let c = m.c_norm.apply(g, c);
    let dt = m.dt_norm.apply(g, dt);
    let dt = m.dt_proj.apply(g, dt);
    let y = g.push(Op::SsmScan { layer: ctx.layer, n_head: nh, head_dim: p, d_state: n, n_group: 1, a: m.a, mamba2: true }, &[xc, dt, b, c]);
    let skip = mul_per_head(g, xc, m.d, t, nh, p);
    let y = g.add(y, skip);
    let y = silu_gate(g, z, y);
    let _ = m.eps;
    Ok(m.out.apply(g, y))
}

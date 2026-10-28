use crate::{Buf, CpuBackend};
use anyhow::{Result, anyhow, bail};
use kf_core::{Act, GluLimit, Inputs, Op, WeightSource, Wid};
use rayon::prelude::*;

pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

pub fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.797_884_56 * (x + 0.044_715 * x * x * x)).tanh())
}

pub fn gelu_erf(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x as f64 / std::f64::consts::SQRT_2) as f32)
}

/// Error function by the Abramowitz and Stegun rational approximation in double precision
pub fn erf(x: f64) -> f64 {
    let s = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let p = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    s * (1.0 - p * (-x * x).exp())
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn softplus(x: f32, beta: f32) -> f32 {
    if beta * x > 20.0 { x } else { (beta * x).exp().ln_1p() / beta }
}

pub fn activate(act: Act, x: f32) -> f32 {
    match act {
        Act::Silu | Act::SwigluOai => silu(x),
        Act::Gelu => gelu_f16(x),
        Act::GeluErf => gelu_erf(x),
        Act::Relu => x.max(0.0),
        Act::ReluSqr => x.max(0.0) * x.max(0.0),
    }
}

/// Gated activation of one gate and up pair with the swiglu clamp forms and the oai variant
pub fn gated(act: Act, limit: Option<GluLimit>, g: f32, u: f32) -> f32 {
    match (act, limit) {
        (Act::SwigluOai, _) => {
            let g = g.min(7.0);
            let u = u.clamp(-7.0, 7.0);
            g * sigmoid(1.702 * g) * (u + 1.0)
        }
        (_, Some(GluLimit::Unfused(l))) => activate(act, g).min(l) * u.clamp(-l, l),
        (_, Some(GluLimit::Fused(l))) => activate(act, g.min(l)) * u.clamp(-l, l),
        (_, None) => activate(act, g) * u,
    }
}

pub fn map(x: &Buf, f: impl Fn(f32) -> f32 + Sync) -> Buf {
    Buf { data: x.data.par_iter().map(|a| f(*a)).collect(), rows: x.rows, cols: x.cols }
}

/// Elementwise op with a column vector broadcast over columns or a row vector broadcast over rows
pub fn binary(a: &Buf, b: &Buf, f: impl Fn(f32, f32) -> f32 + Sync) -> Result<Buf> {
    if a.rows == b.rows && a.cols == b.cols {
        return Ok(Buf { data: a.data.par_iter().zip(&b.data).map(|(x, y)| f(*x, *y)).collect(), rows: a.rows, cols: a.cols });
    }
    if a.data.len() == b.data.len() {
        return Ok(Buf { data: a.data.par_iter().zip(&b.data).map(|(x, y)| f(*x, *y)).collect(), rows: a.rows, cols: a.cols });
    }
    if b.rows == a.rows && b.cols == 1 {
        let mut out = a.clone();
        for r in 0..a.rows {
            let s = b.data[r];
            for v in out.row_mut(r) {
                *v = f(*v, s);
            }
        }
        return Ok(out);
    }
    if b.rows == 1 && b.cols == a.cols {
        let mut out = a.clone();
        for r in 0..a.rows {
            for (v, y) in out.row_mut(r).iter_mut().zip(&b.data) {
                *v = f(*v, *y);
            }
        }
        return Ok(out);
    }
    if b.data.len() == 1 {
        let s = b.data[0];
        return Ok(map(a, |x| f(x, s)));
    }
    bail!("binary op on {}x{} and {}x{}", a.rows, a.cols, b.rows, b.cols)
}

/// Combines each row with a weight vector of the row width or a single scalar
pub fn row_vec(x: &Buf, w: &[f32], f: impl Fn(f32, f32) -> f32) -> Result<Buf> {
    if w.len() == 1 {
        let s = w[0];
        let mut out = x.clone();
        out.data.iter_mut().for_each(|v| *v = f(*v, s));
        return Ok(out);
    }
    if w.len() != x.cols {
        bail!("row vector of {} against {} columns", w.len(), x.cols);
    }
    let mut out = x.clone();
    for r in 0..out.rows {
        for (v, s) in out.row_mut(r).iter_mut().zip(w) {
            *v = f(*v, *s);
        }
    }
    Ok(out)
}

pub fn mul_row_vec(x: &Buf, w: &[f32]) -> Result<Buf> {
    row_vec(x, w, |a, b| a * b)
}

pub fn cols(x: &Buf, start: usize, len: usize) -> Result<Buf> {
    if start + len > x.cols {
        bail!("columns {}..{} of a {} column tensor", start, start + len, x.cols);
    }
    let mut out = Buf::new(x.rows, len);
    for r in 0..x.rows {
        out.row_mut(r).copy_from_slice(&x.row(r)[start..start + len]);
    }
    Ok(out)
}

pub fn rows(x: &Buf, start: usize, len: usize) -> Result<Buf> {
    if start + len > x.rows {
        bail!("rows {}..{} of a {} row tensor", start, start + len, x.rows);
    }
    Ok(Buf { data: x.data[start * x.cols..(start + len) * x.cols].to_vec(), rows: len, cols: x.cols })
}

pub fn concat(a: &Buf, b: &Buf) -> Result<Buf> {
    if a.rows != b.rows {
        bail!("concat of {} rows with {} rows", a.rows, b.rows);
    }
    let mut out = Buf::new(a.rows, a.cols + b.cols);
    for r in 0..a.rows {
        let o = out.row_mut(r);
        o[..a.cols].copy_from_slice(a.row(r));
        o[a.cols..].copy_from_slice(b.row(r));
    }
    Ok(out)
}

pub fn gated_act(gate: &Buf, up: &Buf, act: Act, limit: Option<GluLimit>) -> Result<Buf> {
    if gate.data.len() != up.data.len() {
        bail!("gated activation on {}x{} and {}x{}", gate.rows, gate.cols, up.rows, up.cols);
    }
    Ok(Buf { data: gate.data.par_iter().zip(&up.data).map(|(g, u)| gated(act, limit, *g, *u)).collect(), rows: gate.rows, cols: gate.cols })
}

pub fn softmax_rows(x: &Buf) -> Buf {
    let mut out = x.clone();
    for r in 0..out.rows {
        softmax_inplace(out.row_mut(r));
    }
    out
}

pub fn softmax_inplace(v: &mut [f32]) {
    let m = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut s = 0.0f64;
    for x in v.iter_mut() {
        *x = (*x - m).exp();
        s += *x as f64;
    }
    let inv = (1.0 / s) as f32;
    for x in v.iter_mut() {
        *x *= inv;
    }
}

/// Gelu through the half precision table the reference uses inside the ten unit window
pub fn gelu_f16(x: f32) -> f32 {
    if x <= -10.0 {
        0.0
    } else if x >= 10.0 {
        x
    } else {
        let xh = half::f16::from_f32(x).to_f32();
        half::f16::from_f32(gelu(xh)).to_f32()
    }
}

pub fn rms_norm(x: &Buf, w: Option<&[f32]>, b: Option<&[f32]>, eps: f32) -> Result<Buf> {
    if let Some(w) = w
        && w.len() < x.cols
    {
        bail!("norm weight of {} against {} columns", w.len(), x.cols);
    }
    let mut out = x.clone();
    for r in 0..out.rows {
        let row = out.row_mut(r);
        let ss = (row.iter().map(|a| (a * a) as f64).sum::<f64>() / row.len() as f64) as f32;
        let inv = 1.0 / (ss + eps).sqrt();
        for (i, v) in row.iter_mut().enumerate() {
            *v = *v * inv * w.map_or(1.0, |w| w[i]) + b.map_or(0.0, |b| b[i]);
        }
    }
    Ok(out)
}

pub fn layer_norm(x: &Buf, w: Option<&[f32]>, b: Option<&[f32]>, eps: f32) -> Result<Buf> {
    if let Some(w) = w
        && w.len() < x.cols
    {
        bail!("layer norm weights shorter than {} columns", x.cols);
    }
    let mut out = x.clone();
    for r in 0..out.rows {
        let row = out.row_mut(r);
        let n = row.len() as f64;
        let mean = (row.iter().map(|a| *a as f64).sum::<f64>() / n) as f32;
        let var = (row.iter().map(|a| ((a - mean) * (a - mean)) as f64).sum::<f64>() / n) as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for (i, v) in row.iter_mut().enumerate() {
            *v = (*v - mean) * inv * w.map_or(1.0, |w| w[i]) + b.map_or(0.0, |b| b[i]);
        }
    }
    Ok(out)
}

pub fn l2_norm(x: &Buf, eps: f32) -> Buf {
    let mut out = x.clone();
    for r in 0..out.rows {
        let row = out.row_mut(r);
        let ss: f32 = row.iter().map(|a| a * a).sum();
        let inv = 1.0 / (ss + eps).sqrt();
        row.iter_mut().for_each(|v| *v *= inv);
    }
    out
}

/// Rows of a table for each integer id in the index buffer, ids of one row side by side
pub fn gather(w: Wid, weights: &dyn WeightSource, ids: &Buf) -> Result<Buf> {
    let q = kf_quant::by_id(weights.ggml_type(w)).ok_or_else(|| anyhow!("{}: unknown ggml type {}", weights.name(w), weights.ggml_type(w)))?;
    let dims = weights.dims(w);
    let width = dims[0];
    let n_rows: usize = dims.iter().skip(1).product::<usize>().max(1);
    let rb = q.info().row_bytes(width);
    let bytes = weights.bytes(w)?;
    let mut out = Buf::new(ids.rows, ids.cols * width);
    for r in 0..ids.rows {
        for (k, id) in ids.row(r).iter().enumerate() {
            let id = *id as i64;
            if id < 0 || id as usize >= n_rows {
                bail!("{}: row {id} past {n_rows} rows", weights.name(w));
            }
            let id = id as usize;
            q.dequant_row(&bytes[id * rb..(id + 1) * rb], &mut out.row_mut(r)[k * width..(k + 1) * width]);
        }
    }
    Ok(out)
}

pub fn embed(w: Wid, weights: &dyn WeightSource, tokens: &[u32]) -> Result<Buf> {
    let ids = Buf { data: tokens.iter().map(|t| *t as f32).collect(), rows: tokens.len(), cols: 1 };
    gather(w, weights, &ids)
}

/// x times the transpose of a quantized weight stored as n_out rows of n_in elements
pub fn matmul(x: &Buf, w: Wid, weights: &dyn WeightSource, bias: Option<&[f32]>) -> Result<Buf> {
    matmul_slice(x, w, weights, bias, 0)
}

/// Matmul against slice index of a weight whose third dimension stacks matrices
pub fn matmul_slice(x: &Buf, w: Wid, weights: &dyn WeightSource, bias: Option<&[f32]>, index: usize) -> Result<Buf> {
    let q = kf_quant::by_id(weights.ggml_type(w)).ok_or_else(|| anyhow!("{}: unknown ggml type {}", weights.name(w), weights.ggml_type(w)))?;
    let dims = weights.dims(w);
    let n_in = dims[0];
    let n_out = dims.get(1).copied().unwrap_or(1);
    let n_slices = dims.iter().skip(2).product::<usize>().max(1);
    if x.cols != n_in {
        bail!("{}: {} columns against {} inputs", weights.name(w), x.cols, n_in);
    }
    if index >= n_slices {
        bail!("{}: slice {index} of {n_slices}", weights.name(w));
    }
    if let Some(b) = bias
        && b.len() < n_out
    {
        bail!("{}: bias of {} for {} outputs", weights.name(w), b.len(), n_out);
    }
    let rb = q.info().row_bytes(n_in);
    let bytes = &weights.bytes(w)?[index * n_out * rb..(index + 1) * n_out * rb];
    let kind = kf_quant::act_kind(q);
    let xr: Vec<Vec<f32>> = (0..x.rows).map(|r| kf_quant::round_activations(kind, x.row(r))).collect();
    let mut out = vec![0.0f32; x.rows * n_out];
    let cols: Vec<Vec<f32>> = (0..n_out)
        .into_par_iter()
        .map(|o| {
            let row = &bytes[o * rb..(o + 1) * rb];
            let b = bias.map_or(0.0, |b| b[o]);
            if x.rows <= 2 {
                xr.iter().map(|xq| q.dot_row(row, xq) + b).collect()
            } else {
                let mut tmp = vec![0.0f32; n_in];
                q.dequant_row(row, &mut tmp);
                xr.iter().map(|xq| xq.iter().zip(&tmp).map(|(a, c)| a * c).sum::<f32>() + b).collect()
            }
        })
        .collect();
    for (o, col) in cols.iter().enumerate() {
        for (r, v) in col.iter().enumerate() {
            out[r * n_out + o] = *v;
        }
    }
    Ok(Buf { data: out, rows: x.rows, cols: n_out })
}

fn unary(op: &Op) -> Option<fn(f32) -> f32> {
    Some(match op {
        Op::Silu => silu,
        Op::GeluQuick => |x| x * sigmoid(1.702 * x),
        Op::Gelu => gelu_f16,
        Op::GeluErf => gelu_erf,
        Op::Relu => |x| x.max(0.0),
        Op::ReluSqr => |x| x.max(0.0) * x.max(0.0),
        Op::Sigmoid => sigmoid,
        Op::Tanh => f32::tanh,
        Op::Exp => f32::exp,
        Op::Sqrt => f32::sqrt,
        Op::Sqr => |x| x * x,
        Op::Abs => f32::abs,
        Op::Sign => |x| if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 },
        Op::Log => f32::ln,
        Op::Round => f32::round,
        _ => return None,
    })
}

/// Every op without per layer state, the elementwise, shape, norm and projection family
pub fn exec(b: &mut CpuBackend, op: &Op, ins: &[Buf], _inputs: &Inputs, weights: &dyn WeightSource) -> Result<Buf> {
    let need = |n: usize| -> Result<()> {
        if ins.len() < n {
            bail!("{} inputs for an op needing {n}", ins.len());
        }
        Ok(())
    };
    if let Some(f) = unary(op) {
        need(1)?;
        return Ok(map(&ins[0], f));
    }
    match op {
        Op::Embed { weight } => {
            need(1)?;
            gather(*weight, weights, &ins[0])
        }
        Op::Gather { weight } => {
            need(1)?;
            gather(*weight, weights, &ins[0])
        }
        Op::RowBroadcast { weight, rows } => {
            let v = b.small(*weight, weights)?.to_vec();
            let cols = v.len();
            Ok(Buf { data: v.iter().copied().cycle().take(rows * cols).collect(), rows: *rows, cols })
        }
        Op::MatMul { weight, bias, scale } => {
            need(1)?;
            let bias = b.small_opt(*bias, weights)?;
            let mut y = matmul(&ins[0], *weight, weights, bias.as_deref())?;
            if let Some(s) = scale {
                let s = b.small(*s, weights)?[0];
                y.data.iter_mut().for_each(|v| *v *= s);
            }
            Ok(y)
        }
        Op::MatMulSlice { weight, index } => {
            need(1)?;
            matmul_slice(&ins[0], *weight, weights, None, *index)
        }
        Op::MatMulHeads { weight } => {
            need(1)?;
            let x = &ins[0];
            let d = weights.dims(*weight);
            let n_in = d[0];
            let heads = d.iter().skip(2).product::<usize>().max(1);
            if x.cols != heads * n_in {
                bail!("{}: {} columns against {heads} heads of {n_in}", weights.name(*weight), x.cols);
            }
            let mut out: Option<Buf> = None;
            for h in 0..heads {
                let xh = cols(x, h * n_in, n_in)?;
                let y = matmul_slice(&xh, *weight, weights, None, h)?;
                let o = out.get_or_insert_with(|| Buf::new(x.rows, heads * y.cols));
                for r in 0..x.rows {
                    o.row_mut(r)[h * y.cols..(h + 1) * y.cols].copy_from_slice(y.row(r));
                }
            }
            Ok(out.unwrap_or_else(|| Buf::new(x.rows, 0)))
        }
        Op::Fill { rows, cols, value } => Ok(Buf { data: vec![*value; rows * cols], rows: *rows, cols: *cols }),
        Op::PoolGrid { side, kernel } => {
            need(1)?;
            let x = &ins[0];
            if x.rows != side * side || *kernel == 0 || !side.is_multiple_of(*kernel) {
                bail!("grid pool of {} rows as a {side} by {side} grid in blocks of {kernel}", x.rows);
            }
            let o = side / kernel;
            let mut out = Buf::new(o * o, x.cols);
            let inv = 1.0 / (kernel * kernel) as f32;
            for by in 0..o {
                for bx in 0..o {
                    let orow = out.row_mut(by * o + bx);
                    for dy in 0..*kernel {
                        for dx in 0..*kernel {
                            let r = (by * kernel + dy) * side + bx * kernel + dx;
                            for (acc, v) in orow.iter_mut().zip(x.row(r)) {
                                *acc += v * inv;
                            }
                        }
                    }
                }
            }
            Ok(out)
        }
        Op::RmsNorm { weight, bias, eps } => {
            need(1)?;
            let w = b.small_opt(*weight, weights)?;
            let bb = b.small_opt(*bias, weights)?;
            rms_norm(&ins[0], w.as_deref(), bb.as_deref(), *eps)
        }
        Op::LayerNorm { weight, bias, eps } => {
            need(1)?;
            let w = b.small_opt(*weight, weights)?;
            let bb = b.small_opt(*bias, weights)?;
            layer_norm(&ins[0], w.as_deref(), bb.as_deref(), *eps)
        }
        Op::L2Norm { eps } => {
            need(1)?;
            Ok(l2_norm(&ins[0], *eps))
        }
        Op::GatedAct { act, limit } => {
            need(2)?;
            gated_act(&ins[0], &ins[1], *act, *limit)
        }
        Op::GatedActFused { act, limit } => {
            need(1)?;
            let half = ins[0].cols / 2;
            let g = cols(&ins[0], 0, half)?;
            let u = cols(&ins[0], half, half)?;
            gated_act(&g, &u, *act, *limit)
        }
        Op::Situ { beta, linear_beta } => {
            need(2)?;
            let (beta, lb) = (*beta, *linear_beta);
            binary(&ins[0], &ins[1], move |g, u| beta * (g / beta).tanh() * sigmoid(g) * if lb > 0.0 { lb * (u / lb).tanh() } else { u })
        }
        Op::Add => {
            need(2)?;
            binary(&ins[0], &ins[1], |a, c| a + c)
        }
        Op::Sub => {
            need(2)?;
            binary(&ins[0], &ins[1], |a, c| a - c)
        }
        Op::Mul => {
            need(2)?;
            binary(&ins[0], &ins[1], |a, c| a * c)
        }
        Op::Div => {
            need(2)?;
            binary(&ins[0], &ins[1], |a, c| a / c)
        }
        Op::MulWeight { weight } => {
            need(1)?;
            let w = b.small(*weight, weights)?.to_vec();
            row_vec(&ins[0], &w, |a, c| a * c)
        }
        Op::DivWeight { weight } => {
            need(1)?;
            let w = b.small(*weight, weights)?.to_vec();
            row_vec(&ins[0], &w, |a, c| a / c)
        }
        Op::AddWeight { weight } => {
            need(1)?;
            let w = b.small(*weight, weights)?.to_vec();
            row_vec(&ins[0], &w, |a, c| a + c)
        }
        Op::Scale(s) => {
            need(1)?;
            let s = *s;
            Ok(map(&ins[0], move |a| a * s))
        }
        Op::ScaleBias { scale, bias } => {
            need(1)?;
            let (s, c) = (*scale, *bias);
            Ok(map(&ins[0], move |a| a * s + c))
        }
        Op::Clamp { min, max } => {
            need(1)?;
            let (lo, hi) = (*min, *max);
            Ok(map(&ins[0], move |a| a.clamp(lo, hi)))
        }
        Op::Softplus { beta } => {
            need(1)?;
            let beta = *beta;
            Ok(map(&ins[0], move |a| softplus(a, beta)))
        }
        Op::Xielu { alpha_n, alpha_p, beta, eps } => {
            need(1)?;
            let an = beta + softplus(*alpha_n, 1.0);
            let ap = softplus(*alpha_p, 1.0);
            let (beta, eps) = (*beta, *eps);
            Ok(map(&ins[0], move |u| if u > 0.0 { ap * u * u + beta * u } else { (u.min(eps).exp_m1() - u) * an + beta * u }))
        }
        Op::Softmax => {
            need(1)?;
            Ok(softmax_rows(&ins[0]))
        }
        Op::Softcap(c) => {
            need(1)?;
            let c = *c;
            Ok(map(&ins[0], move |a| c * (a / c).tanh()))
        }
        Op::Reshape { rows, cols } => {
            need(1)?;
            let x = &ins[0];
            if x.data.len() != rows * cols {
                bail!("reshape {}x{} onto {} elements", rows, cols, x.data.len());
            }
            Ok(Buf { data: x.data.clone(), rows: *rows, cols: *cols })
        }
        Op::Cols { start, len } => {
            need(1)?;
            cols(&ins[0], *start, *len)
        }
        Op::Rows { start, len } => {
            need(1)?;
            rows(&ins[0], *start, *len)
        }
        Op::LastRows { n } => {
            need(1)?;
            let x = &ins[0];
            let n = (*n).min(x.rows);
            rows(x, x.rows - n, n)
        }
        Op::Concat => {
            need(2)?;
            concat(&ins[0], &ins[1])
        }
        Op::Tile { n } => {
            need(1)?;
            let x = &ins[0];
            let mut out = Buf::new(x.rows, x.cols * n);
            for r in 0..x.rows {
                for k in 0..*n {
                    out.row_mut(r)[k * x.cols..(k + 1) * x.cols].copy_from_slice(x.row(r));
                }
            }
            Ok(out)
        }
        Op::PadCols { to } => {
            need(1)?;
            let x = &ins[0];
            if *to < x.cols {
                bail!("pad {} columns down to {to}", x.cols);
            }
            let mut out = Buf::new(x.rows, *to);
            for r in 0..x.rows {
                out.row_mut(r)[..x.cols].copy_from_slice(x.row(r));
            }
            Ok(out)
        }
        Op::SumRows => {
            need(1)?;
            let x = &ins[0];
            Ok(Buf { data: (0..x.rows).map(|r| x.row(r).iter().sum()).collect(), rows: x.rows, cols: 1 })
        }
        Op::MaskCols { start, len, value } => {
            need(1)?;
            let mut out = ins[0].clone();
            if start + len > out.cols {
                bail!("mask columns {}..{} of a {} column tensor", start, start + len, out.cols);
            }
            for r in 0..out.rows {
                out.row_mut(r)[*start..start + len].iter_mut().for_each(|v| *v = *value);
            }
            Ok(out)
        }
        Op::Input { .. } => bail!("input op routed to the elementwise module"),
        other => bail!("{other:?} has no host implementation"),
    }
}


use crate::{Buf, CpuBackend, ops};
use anyhow::{Result, anyhow, bail};
use kf_core::{Act, Gating, GluLimit, Groups, Inputs, Op, Selection, WeightNorm, WeightSource, Wid};
use rayon::prelude::*;

/// Output rows of one expert of a stacked tensor multiplied against one input row
pub fn expert_matvec(w: Wid, weights: &dyn WeightSource, expert: usize, x: &[f32]) -> Result<Vec<f32>> {
    let q = kf_quant::by_id(weights.ggml_type(w)).ok_or_else(|| anyhow!("{}: unknown ggml type {}", weights.name(w), weights.ggml_type(w)))?;
    let d = weights.dims(w);
    if d.len() < 3 {
        bail!("{}: expert tensor needs three dims", weights.name(w));
    }
    let (n_in, n_out, n_exp) = (d[0], d[1], d[2]);
    if x.len() != n_in {
        bail!("{}: {} inputs against {} columns", weights.name(w), n_in, x.len());
    }
    if expert >= n_exp {
        bail!("{}: expert {expert} of {n_exp}", weights.name(w));
    }
    let rb = q.info().row_bytes(n_in);
    let bytes = &weights.bytes(w)?[expert * n_out * rb..(expert + 1) * n_out * rb];
    let xq = kf_quant::round_activations(kf_quant::act_kind(q), x);
    Ok((0..n_out).into_par_iter().map(|o| q.dot_row(&bytes[o * rb..(o + 1) * rb], &xq)).collect())
}

/// Top experts by score, restricted to the best groups when grouped, highest first
pub fn pick(scores: &[f32], n_used: usize, groups: Option<Groups>) -> Vec<usize> {
    let mut allowed = vec![true; scores.len()];
    if let Some(g) = groups
        && g.n_group > 1
        && scores.len().is_multiple_of(g.n_group)
    {
        let per = scores.len() / g.n_group;
        let mut gs: Vec<(usize, f32)> = (0..g.n_group)
            .map(|gi| {
                let mut v: Vec<f32> = scores[gi * per..(gi + 1) * per].to_vec();
                v.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                (gi, v.iter().take(2).sum())
            })
            .collect();
        gs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        allowed.iter_mut().for_each(|a| *a = false);
        for (gi, _) in gs.iter().take(g.n_used) {
            allowed[gi * per..(gi + 1) * per].iter_mut().for_each(|a| *a = true);
        }
    }
    let mut idx: Vec<usize> = (0..scores.len()).filter(|i| allowed[*i]).collect();
    idx.sort_by(|a, b| scores[*b].partial_cmp(&scores[*a]).unwrap_or(std::cmp::Ordering::Equal));
    idx.truncate(n_used);
    idx
}

struct ExpertWeights<'a> {
    gate: Option<Wid>,
    up: Wid,
    down: Option<Wid>,
    gate_b: Option<&'a [f32]>,
    up_b: Option<&'a [f32]>,
    down_b: Option<&'a [f32]>,
    fused: bool,
    act: Act,
    limit: Option<GluLimit>,
}

/// One expert applied to one row, gated hidden then the down projection when one exists
fn expert_forward(w: &ExpertWeights, weights: &dyn WeightSource, e: usize, x: &[f32]) -> Result<Vec<f32>> {
    let hidden = match w.gate {
        Some(g) => {
            let mut gt = expert_matvec(g, weights, e, x)?;
            let mut up = expert_matvec(w.up, weights, e, x)?;
            let n_ff = gt.len();
            if let Some(b) = w.gate_b {
                gt.iter_mut().zip(&b[e * n_ff..(e + 1) * n_ff]).for_each(|(v, b)| *v += b);
            }
            if let Some(b) = w.up_b {
                up.iter_mut().zip(&b[e * n_ff..(e + 1) * n_ff]).for_each(|(v, b)| *v += b);
            }
            gt.iter().zip(&up).map(|(g, u)| ops::gated(w.act, w.limit, *g, *u)).collect::<Vec<f32>>()
        }
        None if w.fused => {
            let both = expert_matvec(w.up, weights, e, x)?;
            let n_ff = both.len() / 2;
            (0..n_ff)
                .map(|o| {
                    let g = both[o] + w.gate_b.map_or(0.0, |b| b[e * n_ff + o]);
                    let u = both[n_ff + o] + w.up_b.map_or(0.0, |b| b[e * n_ff + o]);
                    ops::gated(w.act, w.limit, g, u)
                })
                .collect()
        }
        None => {
            let mut up = expert_matvec(w.up, weights, e, x)?;
            let n_ff = up.len();
            if let Some(b) = w.up_b {
                up.iter_mut().zip(&b[e * n_ff..(e + 1) * n_ff]).for_each(|(v, b)| *v += b);
            }
            up.iter().map(|u| ops::activate(w.act, *u)).collect()
        }
    };
    let Some(down) = w.down else { return Ok(hidden) };
    let mut y = expert_matvec(down, weights, e, &hidden)?;
    if let Some(b) = w.down_b {
        let n = y.len();
        y.iter_mut().zip(&b[e * n..(e + 1) * n]).for_each(|(v, b)| *v += b);
    }
    Ok(y)
}

/// Routed experts with every gating, selection, normalization and expert form the op describes
pub fn exec(b: &mut CpuBackend, op: &Op, ins: &[Buf], _inputs: &Inputs, weights: &dyn WeightSource) -> Result<Buf> {
    let Op::MoeFfn {
        router,
        router_bias,
        probs_bias,
        gate_exps,
        up_exps,
        down_exps,
        gate_b,
        up_b,
        down_b,
        exps_scale,
        n_expert,
        n_used,
        gating,
        selection,
        norm_weights,
        scale,
        act,
        limit,
        groups,
        fused_gate_up,
        weight_before_ffn,
        index_div,
        post_norm,
        post_norm_eps,
        router_in,
        selected_in,
    } = op
    else {
        bail!("{op:?} routed to the expert module")
    };
    let n_in = 1 + usize::from(*router_in) + usize::from(*selected_in);
    if ins.len() < n_in {
        bail!("{} inputs for experts needing {n_in}", ins.len());
    }
    let x = &ins[0];
    let logits = if *router_in {
        ins[1].clone()
    } else {
        let rb = b.small_opt(*router_bias, weights)?;
        ops::matmul(x, *router, weights, rb.as_deref())?
    };
    if *index_div <= 1 && logits.cols != *n_expert {
        bail!("router gives {} experts, model says {}", logits.cols, n_expert);
    }
    let selected = selected_in.then(|| &ins[n_in - 1]);
    let probs_bias = b.small_opt(*probs_bias, weights)?;
    let gate_bv = b.small_opt(*gate_b, weights)?;
    let up_bv = b.small_opt(*up_b, weights)?;
    let down_bv = b.small_opt(*down_b, weights)?;
    let exps_scale = b.small_opt(*exps_scale, weights)?;
    let post_w = b.small_opt(*post_norm, weights)?;
    let ew = ExpertWeights {
        gate: *gate_exps,
        up: *up_exps,
        down: *down_exps,
        gate_b: gate_bv.as_deref(),
        up_b: up_bv.as_deref(),
        down_b: down_bv.as_deref(),
        fused: *fused_gate_up,
        act: *act,
        limit: *limit,
    };
    let mut out: Option<Buf> = None;
    for r in 0..x.rows {
        let l = logits.row(r);
        let probs: Vec<f32> = match gating {
            Gating::Softmax => {
                let mut p = l.to_vec();
                ops::softmax_inplace(&mut p);
                p
            }
            Gating::Sigmoid => l.iter().map(|v| ops::sigmoid(*v)).collect(),
            Gating::SoftmaxWeight => l.to_vec(),
            Gating::SqrtSoftplus => l.iter().map(|v| ops::softplus(*v, 1.0).sqrt()).collect(),
        };
        let mut sel: Vec<f32> = match selection {
            Selection::Probs => probs.clone(),
            Selection::Logits => l.to_vec(),
            Selection::SigmoidLogits => l.iter().map(|v| ops::sigmoid(*v)).collect(),
        };
        if let Some(pb) = &probs_bias {
            sel.iter_mut().zip(pb.iter()).for_each(|(s, b)| *s += b);
        }
        let chosen: Vec<usize> = match selected {
            Some(s) => s.row(r).iter().filter(|v| **v >= 0.0).map(|v| *v as usize).collect(),
            None => pick(&sel, *n_used, *groups),
        };
        let chosen: Vec<usize> = if *index_div > 1 { chosen.iter().map(|e| e / index_div).collect() } else { chosen };
        let mut ws: Vec<f32> = match gating {
            Gating::SoftmaxWeight => {
                let mut v: Vec<f32> = chosen.iter().map(|e| l[*e]).collect();
                ops::softmax_inplace(&mut v);
                v
            }
            _ => chosen.iter().map(|e| probs[*e]).collect(),
        };
        match norm_weights {
            WeightNorm::None => {}
            WeightNorm::Clamp => {
                let sum: f32 = ws.iter().sum();
                ws.iter_mut().for_each(|w| *w /= sum.max(6.103_515_6e-5));
            }
            WeightNorm::AddEps => {
                if chosen.len() > 1 {
                    let sum: f32 = ws.iter().sum();
                    ws.iter_mut().for_each(|w| *w /= sum + 1e-20);
                }
            }
        }
        if *scale != 1.0 {
            ws.iter_mut().for_each(|w| *w *= scale);
        }
        let xr = x.row(r);
        let mut acc: Option<Vec<f32>> = None;
        for (e, w) in chosen.iter().zip(&ws) {
            if *e >= *n_expert {
                bail!("expert {e} chosen of {n_expert}");
            }
            let input: Vec<f32> = if *weight_before_ffn { xr.iter().map(|v| v * w).collect() } else { xr.to_vec() };
            let mut y = expert_forward(&ew, weights, *e, &input)?;
            if let Some(s) = &exps_scale {
                let s = s[*e];
                y.iter_mut().for_each(|v| *v *= s);
            }
            if let Some(pw) = &post_w {
                let yb = Buf { cols: y.len(), data: y, rows: 1 };
                let yb = ops::mul_row_vec(&yb, pw)?;
                y = ops::rms_norm(&yb, None, None, *post_norm_eps)?.data;
            }
            let wv = if *weight_before_ffn { 1.0 } else { *w };
            match &mut acc {
                Some(a) => a.iter_mut().zip(&y).for_each(|(a, v)| *a += wv * v),
                None => acc = Some(y.iter().map(|v| wv * v).collect()),
            }
        }
        let y = acc.unwrap_or_else(|| vec![0.0; xr.len()]);
        let o = out.get_or_insert_with(|| Buf::new(x.rows, y.len()));
        if o.cols != y.len() {
            bail!("expert outputs of {} against {} columns", y.len(), o.cols);
        }
        o.row_mut(r).copy_from_slice(&y);
    }
    Ok(out.unwrap_or_else(|| Buf::new(0, x.cols)))
}

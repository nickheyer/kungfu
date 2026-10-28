use crate::{Buf, CpuBackend, LayerState, ops};
use anyhow::{Result, anyhow, bail};
use kf_core::{Inputs, Op, WeightSource};

/// Tokens of state kept for roll back after a truncation
pub const SNAPSHOTS: usize = 8;

/// Records the state after one token under a key, keeping the newest snapshots only
pub fn snap(st: &mut LayerState, key: &str, pos: usize, state: Buf) {
    let v = st.snaps.entry(key.to_string()).or_default();
    v.retain(|(p, _)| *p < pos);
    v.push((pos, state));
    if v.len() > SNAPSHOTS {
        v.remove(0);
    }
}

/// Restores every snapshotted field to its state after position n tokens, or fails naming the field
pub fn roll_back(st: &mut LayerState, past: usize, n: usize) -> Result<()> {
    if n >= past {
        return Ok(());
    }
    let keys: Vec<String> = st.snaps.keys().cloned().collect();
    for key in keys {
        let found = st.snaps[&key].iter().find(|(p, _)| *p == n).map(|(_, s)| s.clone());
        let Some(state) = found else { bail!("layer state {key} cannot roll back from {past} to {n} tokens") };
        match key.split_once(':') {
            Some(("conv", slot)) => {
                st.conv.insert(slot.parse()?, state);
            }
            Some(("shift", slot)) => {
                st.shift.insert(slot.parse()?, state.data);
            }
            Some(("compressed", slot)) => {
                st.compressed.insert(slot.parse()?, state);
            }
            Some(("ring_kv", slot)) => {
                st.ring_kv.insert(slot.parse()?, state);
            }
            Some(("ring_score", slot)) => {
                st.ring_score.insert(slot.parse()?, state);
            }
            _ if key == "ssm" => st.ssm = state.data,
            _ if key == "topk" => st.topk = state,
            _ if key == "cand" => st.cand = state,
            _ => bail!("layer state {key} has no roll back rule"),
        }
        st.snaps.get_mut(&key).map(|v| v.retain(|(p, _)| *p <= n));
    }
    Ok(())
}

fn need(ins: &[Buf], n: usize) -> Result<()> {
    if ins.len() < n {
        bail!("{} inputs for an op needing {n}", ins.len());
    }
    Ok(())
}

/// Depthwise causal convolution with the carried history rows of one slot
#[allow(clippy::too_many_arguments)]
fn causal_conv(st: &mut LayerState, slot: usize, kernel: usize, dilation: usize, w: &[f32], bias: Option<&[f32]>, x: &Buf, base: usize) -> Result<Buf> {
    let c = x.cols;
    if w.len() != kernel * c {
        bail!("conv kernel of {} for {} taps over {} channels", w.len(), kernel, c);
    }
    let h = (kernel - 1) * dilation;
    let hist = st.conv.entry(slot).or_insert_with(|| Buf::new(h, c)).clone();
    if hist.cols != c || hist.rows != h {
        bail!("conv history {}x{} for {} rows of {} channels", hist.rows, hist.cols, h, c);
    }
    let mut padded = Buf::new(h + x.rows, c);
    padded.data[..h * c].copy_from_slice(&hist.data);
    padded.data[h * c..].copy_from_slice(&x.data);
    let mut out = Buf::new(x.rows, c);
    for t in 0..x.rows {
        let o = out.row_mut(t);
        for k in 0..kernel {
            let src = padded.row(h + t - (kernel - 1 - k) * dilation);
            for ch in 0..c {
                o[ch] += w[ch * kernel + k] * src[ch];
            }
        }
        if let Some(b) = bias {
            o.iter_mut().zip(b).for_each(|(v, b)| *v += b);
        }
        if t + SNAPSHOTS >= x.rows {
            let rows = Buf { data: padded.data[(t + 1) * c..(t + 1 + h) * c].to_vec(), rows: h, cols: c };
            snap(st, &format!("conv:{slot}"), base + t + 1, rows);
        }
    }
    st.conv.insert(slot, Buf { data: padded.data[x.rows * c..].to_vec(), rows: h, cols: c });
    Ok(out)
}

/// Selective scan over the carried state, scalar decay per head or a decay per state dim
#[allow(clippy::too_many_arguments)]
fn ssm_scan(st: &mut LayerState, x: &Buf, dt: &Buf, bm: &Buf, cm: &Buf, n_head: usize, head_dim: usize, d_state: usize, n_group: usize, a: &[f32], mamba2: bool, base: usize) -> Result<Buf> {
    if x.cols != n_head * head_dim || dt.cols != n_head || bm.cols != n_group * d_state || cm.cols != n_group * d_state {
        bail!("ssm scan shapes x {} dt {} B {} C {} for {} heads of {} and {} groups of {}", x.cols, dt.cols, bm.cols, cm.cols, n_head, head_dim, n_group, d_state);
    }
    let need_a = if mamba2 { n_head } else { n_head * d_state };
    if a.len() < need_a {
        bail!("ssm decay of {} values for {} heads", a.len(), n_head);
    }
    let size = n_head * head_dim * d_state;
    if st.ssm.is_empty() {
        st.ssm = vec![0.0; size];
    }
    if st.ssm.len() != size {
        bail!("ssm state of {} for {} heads of {} by {}", st.ssm.len(), n_head, head_dim, d_state);
    }
    let per_group = n_head / n_group.max(1);
    let mut out = Buf::new(x.rows, x.cols);
    for t in 0..x.rows {
        let (xr, dtr, br, cr) = (x.row(t), dt.row(t), bm.row(t), cm.row(t));
        let o = out.row_mut(t);
        for h in 0..n_head {
            let g = h / per_group.max(1);
            let sp = ops::softplus(dtr[h], 1.0);
            let da = if mamba2 { (sp * a[h]).exp() } else { 0.0 };
            for p in 0..head_dim {
                let xd = xr[h * head_dim + p] * sp;
                let s = &mut st.ssm[(h * head_dim + p) * d_state..(h * head_dim + p + 1) * d_state];
                let mut y = 0.0f32;
                for i in 0..d_state {
                    let decay = if mamba2 { da } else { (sp * a[h * d_state + i]).exp() };
                    s[i] = s[i] * decay + br[g * d_state + i] * xd;
                    y += s[i] * cr[g * d_state + i];
                }
                o[h * head_dim + p] = y;
            }
        }
        if t + SNAPSHOTS >= x.rows {
            snap(st, "ssm", base + t + 1, Buf { data: st.ssm.clone(), rows: 1, cols: size });
        }
    }
    Ok(out)
}

/// Gated delta rule recurrence, the state of every value head updated token by token
#[allow(clippy::too_many_arguments)]
fn gated_delta_net(st: &mut LayerState, q: &Buf, k: &Buf, v: &Buf, g: &Buf, beta: &Buf, n_head_k: usize, n_head_v: usize, d: usize, per_channel: bool, base: usize) -> Result<Buf> {
    if q.cols != n_head_k * d || k.cols != n_head_k * d || v.cols != n_head_v * d || beta.cols != n_head_v {
        bail!("delta net shapes q {} k {} v {} beta {} for {} key heads and {} value heads of {}", q.cols, k.cols, v.cols, beta.cols, n_head_k, n_head_v, d);
    }
    let g_width = if per_channel { n_head_v * d } else { n_head_v };
    if g.cols != g_width {
        bail!("delta net gate of {} columns, expected {g_width}", g.cols);
    }
    if n_head_k == 0 || !n_head_v.is_multiple_of(n_head_k) {
        bail!("delta net with {n_head_v} value heads over {n_head_k} key heads");
    }
    let size = n_head_v * d * d;
    if st.ssm.is_empty() {
        st.ssm = vec![0.0; size];
    }
    if st.ssm.len() != size {
        bail!("delta net state of {} for {} heads of {}", st.ssm.len(), n_head_v, d);
    }
    let inv = 1.0 / (d as f32).sqrt();
    let mut out = Buf::new(q.rows, n_head_v * d);
    let mut delta = vec![0.0f32; d];
    for t in 0..q.rows {
        for j in 0..n_head_v {
            let hk = j % n_head_k;
            let qh = &q.row(t)[hk * d..(hk + 1) * d];
            let kh = &k.row(t)[hk * d..(hk + 1) * d];
            let vh = &v.row(t)[j * d..(j + 1) * d];
            let bt = beta.row(t)[j];
            let s = &mut st.ssm[j * d * d..(j + 1) * d * d];
            for i in 0..d {
                let gi = if per_channel { g.row(t)[j * d + i] } else { g.row(t)[j] };
                let decay = gi.exp();
                s[i * d..(i + 1) * d].iter_mut().for_each(|x| *x *= decay);
            }
            for c in 0..d {
                let mut sk = 0.0f32;
                for i in 0..d {
                    sk += s[i * d + c] * kh[i];
                }
                delta[c] = bt * (vh[c] - sk);
            }
            for i in 0..d {
                let ki = kh[i];
                for c in 0..d {
                    s[i * d + c] += ki * delta[c];
                }
            }
            let o = &mut out.row_mut(t)[j * d..(j + 1) * d];
            for c in 0..d {
                let mut acc = 0.0f32;
                for i in 0..d {
                    acc += s[i * d + c] * qh[i];
                }
                o[c] = acc * inv;
            }
        }
        if t + SNAPSHOTS >= q.rows {
            snap(st, "ssm", base + t + 1, Buf { data: st.ssm.clone(), rows: 1, cols: size });
        }
    }
    Ok(out)
}

fn heads(width: usize, hs: usize) -> Result<usize> {
    if hs == 0 || !width.is_multiple_of(hs) {
        bail!("{width} channels over heads of {hs}");
    }
    Ok(width / hs)
}

fn ensure_state(st: &mut LayerState, size: usize) -> Result<()> {
    if st.ssm.is_empty() {
        st.ssm = vec![0.0; size];
    }
    if st.ssm.len() != size {
        bail!("recurrent state of {} where {size} is needed", st.ssm.len());
    }
    Ok(())
}

/// RWKV6 recurrence with the bonus on the current token
fn wkv6(st: &mut LayerState, k: &Buf, v: &Buf, r: &Buf, w: &Buf, u: &[f32], hs: usize, base: usize) -> Result<Buf> {
    let n = k.cols;
    let nh = heads(n, hs)?;
    if v.cols != n || r.cols != n || w.cols != n || u.len() < n {
        bail!("wkv6 shapes k {} v {} r {} w {} bonus {}", k.cols, v.cols, r.cols, w.cols, u.len());
    }
    ensure_state(st, n * hs)?;
    let mut out = Buf::new(k.rows, n);
    for t in 0..k.rows {
        for h in 0..nh {
            let s = &mut st.ssm[h * hs * hs..(h + 1) * hs * hs];
            let (kr, vr, rr, wr) = (&k.row(t)[h * hs..], &v.row(t)[h * hs..], &r.row(t)[h * hs..], &w.row(t)[h * hs..]);
            let o = &mut out.row_mut(t)[h * hs..(h + 1) * hs];
            for i in 0..hs {
                let (ki, ri, wi, ui) = (kr[i], rr[i], wr[i], u[h * hs + i]);
                for j in 0..hs {
                    let kv = ki * vr[j];
                    o[j] += ri * (ui * kv + s[i * hs + j]);
                    s[i * hs + j] = s[i * hs + j] * wi + kv;
                }
            }
        }
        if t + SNAPSHOTS >= k.rows {
            snap(st, "ssm", base + t + 1, Buf { data: st.ssm.clone(), rows: 1, cols: n * hs });
        }
    }
    Ok(out)
}

/// Gated linear attention, the state decayed per key channel then read by the query
fn gla(st: &mut LayerState, k: &Buf, v: &Buf, q: &Buf, g: &Buf, hs: usize, scale: f32, base: usize) -> Result<Buf> {
    let n = k.cols;
    let nh = heads(n, hs)?;
    if v.cols != n || q.cols != n || g.cols != n {
        bail!("gla shapes k {} v {} q {} g {}", k.cols, v.cols, q.cols, g.cols);
    }
    ensure_state(st, n * hs)?;
    let mut out = Buf::new(k.rows, n);
    for t in 0..k.rows {
        for h in 0..nh {
            let s = &mut st.ssm[h * hs * hs..(h + 1) * hs * hs];
            let (kr, vr, qr, gr) = (&k.row(t)[h * hs..], &v.row(t)[h * hs..], &q.row(t)[h * hs..], &g.row(t)[h * hs..]);
            let o = &mut out.row_mut(t)[h * hs..(h + 1) * hs];
            for i in 0..hs {
                let (ki, qi, gi) = (kr[i], qr[i] * scale, gr[i]);
                for j in 0..hs {
                    s[i * hs + j] = s[i * hs + j] * gi + ki * vr[j];
                    o[j] += qi * s[i * hs + j];
                }
            }
        }
        if t + SNAPSHOTS >= k.rows {
            snap(st, "ssm", base + t + 1, Buf { data: st.ssm.clone(), rows: 1, cols: n * hs });
        }
    }
    Ok(out)
}

/// RWKV7 generalized delta rule with the state rows over value channels
#[allow(clippy::too_many_arguments)]
fn wkv7(st: &mut LayerState, r: &Buf, w: &Buf, k: &Buf, v: &Buf, a: &Buf, bb: &Buf, hs: usize, base: usize) -> Result<Buf> {
    let n = k.cols;
    let nh = heads(n, hs)?;
    if [r, w, v, a, bb].iter().any(|x| x.cols != n) {
        bail!("wkv7 inputs disagree on {n} channels");
    }
    ensure_state(st, n * hs)?;
    let mut out = Buf::new(k.rows, n);
    let mut sa = vec![0.0f32; hs];
    for t in 0..k.rows {
        for h in 0..nh {
            let s = &mut st.ssm[h * hs * hs..(h + 1) * hs * hs];
            let (rr, wr, kr, vr, ar, br) = (&r.row(t)[h * hs..], &w.row(t)[h * hs..], &k.row(t)[h * hs..], &v.row(t)[h * hs..], &a.row(t)[h * hs..], &bb.row(t)[h * hs..]);
            for i in 0..hs {
                let mut acc = 0.0f32;
                for j in 0..hs {
                    acc += ar[j] * s[i * hs + j];
                }
                sa[i] = acc;
            }
            let o = &mut out.row_mut(t)[h * hs..(h + 1) * hs];
            for i in 0..hs {
                let vi = vr[i];
                let mut y = 0.0f32;
                for j in 0..hs {
                    let ns = s[i * hs + j] * wr[j] + vi * kr[j] + sa[i] * br[j];
                    s[i * hs + j] = ns;
                    y += ns * rr[j];
                }
                o[i] = y;
            }
        }
        if t + SNAPSHOTS >= k.rows {
            snap(st, "ssm", base + t + 1, Buf { data: st.ssm.clone(), rows: 1, cols: n * hs });
        }
    }
    Ok(out)
}

/// Previous rows from the carried slot with the last row stored for the next pass
fn token_shift(st: &mut LayerState, slot: usize, x: &Buf, base: usize) -> Result<Buf> {
    let prev = st.shift.entry(slot).or_insert_with(|| vec![0.0; x.cols]);
    if prev.len() != x.cols {
        bail!("token shift state of {} for {} channels", prev.len(), x.cols);
    }
    let mut out = Buf::new(x.rows, x.cols);
    out.row_mut(0).copy_from_slice(prev);
    for t in 1..x.rows {
        out.row_mut(t).copy_from_slice(x.row(t - 1));
    }
    *prev = x.row(x.rows - 1).to_vec();
    for t in x.rows.saturating_sub(SNAPSHOTS)..x.rows {
        snap(st, &format!("shift:{slot}"), base + t + 1, Buf { data: x.row(t).to_vec(), rows: 1, cols: x.cols });
    }
    Ok(out)
}

/// Lightning linear attention with one decay per head applied before each update
#[allow(clippy::too_many_arguments)]
fn lightning(st: &mut LayerState, q: &Buf, k: &Buf, v: &Buf, n_head: usize, d: usize, decay: &[f32], base: usize) -> Result<Buf> {
    if q.cols != n_head * d || k.cols != n_head * d || v.cols != n_head * d || decay.len() < n_head {
        bail!("lightning attention shapes q {} k {} v {} decays {} for {} heads of {}", q.cols, k.cols, v.cols, decay.len(), n_head, d);
    }
    ensure_state(st, n_head * d * d)?;
    let mut out = Buf::new(q.rows, n_head * d);
    for t in 0..q.rows {
        for h in 0..n_head {
            let s = &mut st.ssm[h * d * d..(h + 1) * d * d];
            let dec = (-decay[h]).exp();
            let (qr, kr, vr) = (&q.row(t)[h * d..], &k.row(t)[h * d..], &v.row(t)[h * d..]);
            let o = &mut out.row_mut(t)[h * d..(h + 1) * d];
            for i in 0..d {
                for j in 0..d {
                    s[i * d + j] = dec * s[i * d + j] + kr[i] * vr[j];
                    o[j] += qr[i] * s[i * d + j];
                }
            }
        }
        if t + SNAPSHOTS >= q.rows {
            snap(st, "ssm", base + t + 1, Buf { data: st.ssm.clone(), rows: 1, cols: n_head * d * d });
        }
    }
    Ok(out)
}

/// Every op that carries recurrent, convolutional or shift state across passes
pub fn exec(b: &mut CpuBackend, op: &Op, ins: &[Buf], _inputs: &Inputs, weights: &dyn WeightSource) -> Result<Buf> {
    let base = b.tokens().len();
    match op {
        Op::CausalConv { layer, slot, kernel, dilation, weight, bias } => {
            need(ins, 1)?;
            let w = b.small(*weight, weights)?.to_vec();
            let bias = b.small_opt(*bias, weights)?;
            causal_conv(b.layer(*layer), *slot, *kernel, (*dilation).max(1), &w, bias.as_deref(), &ins[0], base)
        }
        Op::SsmScan { layer, n_head, head_dim, d_state, n_group, a, mamba2 } => {
            need(ins, 4)?;
            let a = b.small(*a, weights)?.to_vec();
            ssm_scan(b.layer(*layer), &ins[0], &ins[1], &ins[2], &ins[3], *n_head, *head_dim, *d_state, (*n_group).max(1), &a, *mamba2, base)
        }
        Op::GatedDeltaNet { layer, n_head_k, n_head_v, head_dim, gate_per_channel } => {
            need(ins, 5)?;
            gated_delta_net(b.layer(*layer), &ins[0], &ins[1], &ins[2], &ins[3], &ins[4], *n_head_k, *n_head_v, *head_dim, *gate_per_channel, base)
        }
        Op::Wkv6 { layer, head_size, first } => {
            need(ins, 4)?;
            let u = b.small(*first, weights)?.to_vec();
            wkv6(b.layer(*layer), &ins[0], &ins[1], &ins[2], &ins[3], &u, *head_size, base)
        }
        Op::Gla { layer, head_size, scale } => {
            need(ins, 4)?;
            gla(b.layer(*layer), &ins[0], &ins[1], &ins[2], &ins[3], *head_size, *scale, base)
        }
        Op::Wkv7 { layer, head_size } => {
            need(ins, 6)?;
            wkv7(b.layer(*layer), &ins[0], &ins[1], &ins[2], &ins[3], &ins[4], &ins[5], *head_size, base)
        }
        Op::TokenShift { layer, slot } => {
            need(ins, 1)?;
            token_shift(b.layer(*layer), *slot, &ins[0], base)
        }
        Op::LightningAttn { layer, n_head, head_dim, decay } => {
            need(ins, 3)?;
            lightning(b.layer(*layer), &ins[0], &ins[1], &ins[2], *n_head, *head_dim, decay, base)
        }
        other => Err(anyhow!("{other:?} routed to the recurrent module")),
    }
}

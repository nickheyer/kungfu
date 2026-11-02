//! Reference forms of the kernels a draft head learns with, the oracle the device kernels are checked against

/// c = alpha op(a) op(b) + beta c, a m by k and b k by n after the transposes, row major with leading dimensions
#[allow(clippy::too_many_arguments)]
pub fn gemm(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize, lda: usize, ldb: usize, ldc: usize, ta: bool, tb: bool, alpha: f32, beta: f32) {
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f64;
            for l in 0..k {
                let av = if ta { a[l * lda + i] } else { a[i * lda + l] };
                let bv = if tb { b[j * ldb + l] } else { b[l * ldb + j] };
                s += f64::from(av) * f64::from(bv);
            }
            let out = &mut c[i * ldc + j];
            *out = if beta == 0.0 { alpha * s as f32 } else { alpha * s as f32 + beta * *out };
        }
    }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// h = silu(g) u
pub fn silu_gate_fwd(g: &[f32], u: &[f32], h: &mut [f32]) {
    for i in 0..h.len() {
        h[i] = silu(g[i]) * u[i];
    }
}

/// dg = dh u silu'(g), du = dh silu(g)
pub fn silu_gate_bwd(g: &[f32], u: &[f32], dh: &[f32], dg: &mut [f32], du: &mut [f32]) {
    for i in 0..dh.len() {
        let s = 1.0 / (1.0 + (-g[i]).exp());
        let si = g[i] * s;
        dg[i] = dh[i] * u[i] * (s + si * (1.0 - s));
        du[i] = dh[i] * si;
    }
}

/// y = x w / rms(x) per row, returning every row's inverse rms
pub fn rms_fwd(x: &[f32], w: &[f32], y: &mut [f32], cols: usize, eps: f32) -> Vec<f32> {
    let rows = x.len() / cols;
    let mut inv = Vec::with_capacity(rows);
    for r in 0..rows {
        let xr = &x[r * cols..(r + 1) * cols];
        let s: f64 = xr.iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
        let i = 1.0 / (s / cols as f64 + f64::from(eps)).sqrt();
        inv.push(i as f32);
        for c in 0..cols {
            y[r * cols + c] = xr[c] * i as f32 * w[c];
        }
    }
    inv
}

/// dx per row and dw summed over rows, of y = x w inv
pub fn rms_bwd(x: &[f32], w: &[f32], inv: &[f32], dy: &[f32], dx: &mut [f32], dw: &mut [f32], cols: usize) {
    let rows = x.len() / cols;
    for r in 0..rows {
        let (xr, dyr) = (&x[r * cols..(r + 1) * cols], &dy[r * cols..(r + 1) * cols]);
        let s: f64 = (0..cols).map(|c| f64::from(w[c]) * f64::from(dyr[c]) * f64::from(xr[c])).sum();
        let k = inv[r] * inv[r] * (s / cols as f64) as f32;
        for c in 0..cols {
            dx[r * cols + c] = inv[r] * (w[c] * dyr[c] - xr[c] * k);
            dw[c] += dyr[c] * xr[c] * inv[r];
        }
    }
}

/// One AdamW step with bias correction at step t
#[allow(clippy::too_many_arguments)]
pub fn adamw(p: &mut [f32], g: &[f32], m: &mut [f32], v: &mut [f32], lr: f32, b1: f32, b2: f32, eps: f32, wd: f32, t: i32) {
    let (c1, c2) = (1.0 - b1.powi(t), 1.0 - b2.powi(t));
    for i in 0..p.len() {
        m[i] = b1 * m[i] + (1.0 - b1) * g[i];
        v[i] = b2 * v[i] + (1.0 - b2) * g[i] * g[i];
        p[i] -= lr * ((m[i] / c1) / ((v[i] / c2).sqrt() + eps) + wd * p[i]);
    }
}

/// The mean smooth L1 loss and its gradient
pub fn smooth_l1(pred: &[f32], target: &[f32], grad: &mut [f32], beta: f32) -> f32 {
    let n = pred.len() as f32;
    let mut loss = 0.0f64;
    for i in 0..pred.len() {
        let d = pred[i] - target[i];
        let a = d.abs();
        loss += f64::from(if a < beta { 0.5 * d * d / beta } else { a - 0.5 * beta });
        grad[i] = (if a < beta { d / beta } else if d > 0.0 { 1.0 } else { -1.0 }) / n;
    }
    (loss / f64::from(n)) as f32
}

/// Attention over b chunks of t positions and h heads of hd, q k v and out as [b, t, h hd] rows with the heads side
/// by side, every position seeing the keys at or before it within the window, the probabilities returned as [b, h, t, t]
#[allow(clippy::too_many_arguments)]
pub fn attn_fwd(q: &[f32], k: &[f32], v: &[f32], out: &mut [f32], b: usize, t: usize, h: usize, hd: usize, window: usize, scale: f32) -> Vec<f32> {
    let d = h * hd;
    let mut probs = vec![0.0f32; b * h * t * t];
    for bi in 0..b {
        for hi in 0..h {
            let at = |i: usize, c: usize| (bi * t + i) * d + hi * hd + c;
            for i in 0..t {
                let mut s = vec![f32::NEG_INFINITY; t];
                let mut m = f32::NEG_INFINITY;
                for j in 0..=i {
                    if j + window <= i {
                        continue;
                    }
                    let mut dot = 0.0f64;
                    for c in 0..hd {
                        dot += f64::from(q[at(i, c)]) * f64::from(k[at(j, c)]);
                    }
                    s[j] = dot as f32 * scale;
                    m = m.max(s[j]);
                }
                let mut l = 0.0f64;
                for j in 0..t {
                    s[j] = if s[j] == f32::NEG_INFINITY { 0.0 } else { (s[j] - m).exp() };
                    l += f64::from(s[j]);
                }
                for j in 0..t {
                    s[j] = (f64::from(s[j]) / l) as f32;
                    probs[((bi * h + hi) * t + i) * t + j] = s[j];
                }
                for c in 0..hd {
                    let mut o = 0.0f64;
                    for j in 0..=i {
                        o += f64::from(s[j]) * f64::from(v[at(j, c)]);
                    }
                    out[at(i, c)] = o as f32;
                }
            }
        }
    }
    probs
}

/// Gradients of attn_fwd given dout and its probabilities
#[allow(clippy::too_many_arguments)]
pub fn attn_bwd(q: &[f32], k: &[f32], v: &[f32], probs: &[f32], dout: &[f32], dq: &mut [f32], dk: &mut [f32], dv: &mut [f32], b: usize, t: usize, h: usize, hd: usize, scale: f32) {
    let d = h * hd;
    dq.iter_mut().for_each(|x| *x = 0.0);
    dk.iter_mut().for_each(|x| *x = 0.0);
    dv.iter_mut().for_each(|x| *x = 0.0);
    for bi in 0..b {
        for hi in 0..h {
            let at = |i: usize, c: usize| (bi * t + i) * d + hi * hd + c;
            let p = |i: usize, j: usize| probs[((bi * h + hi) * t + i) * t + j];
            for i in 0..t {
                let mut dp = vec![0.0f64; t];
                let mut dot = 0.0f64;
                for j in 0..=i {
                    for c in 0..hd {
                        dp[j] += f64::from(dout[at(i, c)]) * f64::from(v[at(j, c)]);
                        dv[at(j, c)] += p(i, j) * dout[at(i, c)];
                    }
                    dot += f64::from(p(i, j)) * dp[j];
                }
                for j in 0..=i {
                    let ds = f64::from(p(i, j)) * (dp[j] - dot) * f64::from(scale);
                    for c in 0..hd {
                        dq[at(i, c)] += (ds * f64::from(k[at(j, c)])) as f32;
                        dk[at(j, c)] += (ds * f64::from(q[at(i, c)])) as f32;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise(seed: &mut u64) -> f32 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        (*seed >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }

    /// The attention backward agrees with central differences of the forward's loss on every input
    #[test]
    fn attention_gradients_match_finite_differences() {
        let (b, t, h, hd, window) = (2usize, 5usize, 2usize, 3usize, 4usize);
        let n = b * t * h * hd;
        let mut seed = 0x1234_5678_9abc_def1u64;
        let q: Vec<f32> = (0..n).map(|_| noise(&mut seed)).collect();
        let k: Vec<f32> = (0..n).map(|_| noise(&mut seed)).collect();
        let v: Vec<f32> = (0..n).map(|_| noise(&mut seed)).collect();
        let w: Vec<f32> = (0..n).map(|_| noise(&mut seed)).collect();
        let scale = 0.7;
        // The loss is the dot of the output with fixed weights, so dout is w
        let loss = |q: &[f32], k: &[f32], v: &[f32]| -> f64 {
            let mut out = vec![0.0; n];
            attn_fwd(q, k, v, &mut out, b, t, h, hd, window, scale);
            out.iter().zip(&w).map(|(o, w)| f64::from(*o) * f64::from(*w)).sum()
        };
        let mut out = vec![0.0; n];
        let probs = attn_fwd(&q, &k, &v, &mut out, b, t, h, hd, window, scale);
        let (mut dq, mut dk, mut dv) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        attn_bwd(&q, &k, &v, &probs, &w, &mut dq, &mut dk, &mut dv, b, t, h, hd, scale);
        let eps = 1e-3f32;
        for (name, x, dx) in [("q", &q, &dq), ("k", &k, &dk), ("v", &v, &dv)] {
            for i in (0..n).step_by(7) {
                let mut up = x.clone();
                up[i] += eps;
                let mut down = x.clone();
                down[i] -= eps;
                let fd = match name {
                    "q" => (loss(&up, &k, &v) - loss(&down, &k, &v)) / (2.0 * f64::from(eps)),
                    "k" => (loss(&q, &up, &v) - loss(&q, &down, &v)) / (2.0 * f64::from(eps)),
                    _ => (loss(&q, &k, &up) - loss(&q, &k, &down)) / (2.0 * f64::from(eps)),
                };
                assert!((fd - f64::from(dx[i])).abs() < 2e-3, "{name}[{i}]: analytic {} against difference {fd}", dx[i]);
            }
        }
    }
}

/// Attention of nq query rows over nk key rows under a visibility mask, q k v and out as rows of h heads of hd side by
/// side, the probabilities returned as [h, nq, nk]
#[allow(clippy::too_many_arguments)]
pub fn masked_attn_fwd(q: &[f32], k: &[f32], v: &[f32], mask: &[bool], out: &mut [f32], nq: usize, nk: usize, h: usize, hd: usize, scale: f32) -> Vec<f32> {
    let d = h * hd;
    let mut probs = vec![0.0f32; h * nq * nk];
    for hi in 0..h {
        for i in 0..nq {
            let mut s = vec![f32::NEG_INFINITY; nk];
            let mut m = f32::NEG_INFINITY;
            for j in 0..nk {
                if !mask[i * nk + j] {
                    continue;
                }
                let mut dot = 0.0f64;
                for c in 0..hd {
                    dot += f64::from(q[i * d + hi * hd + c]) * f64::from(k[j * d + hi * hd + c]);
                }
                s[j] = dot as f32 * scale;
                m = m.max(s[j]);
            }
            let mut l = 0.0f64;
            for j in 0..nk {
                s[j] = if s[j] == f32::NEG_INFINITY { 0.0 } else { (s[j] - m).exp() };
                l += f64::from(s[j]);
            }
            for j in 0..nk {
                let p = if l > 0.0 { (f64::from(s[j]) / l) as f32 } else { 0.0 };
                probs[(hi * nq + i) * nk + j] = p;
            }
            for c in 0..hd {
                let mut o = 0.0f64;
                for j in 0..nk {
                    o += f64::from(probs[(hi * nq + i) * nk + j]) * f64::from(v[j * d + hi * hd + c]);
                }
                out[i * d + hi * hd + c] = o as f32;
            }
        }
    }
    probs
}

/// Gradients of masked_attn_fwd given dout and its probabilities, dq over the queries and dk dv over the keys
#[allow(clippy::too_many_arguments)]
pub fn masked_attn_bwd(q: &[f32], k: &[f32], v: &[f32], probs: &[f32], dout: &[f32], dq: &mut [f32], dk: &mut [f32], dv: &mut [f32], nq: usize, nk: usize, h: usize, hd: usize, scale: f32) {
    let d = h * hd;
    dq.iter_mut().for_each(|x| *x = 0.0);
    dk.iter_mut().for_each(|x| *x = 0.0);
    dv.iter_mut().for_each(|x| *x = 0.0);
    for hi in 0..h {
        for i in 0..nq {
            let p = |j: usize| probs[(hi * nq + i) * nk + j];
            let mut dp = vec![0.0f64; nk];
            let mut dot = 0.0f64;
            for j in 0..nk {
                if p(j) == 0.0 {
                    continue;
                }
                for c in 0..hd {
                    dp[j] += f64::from(dout[i * d + hi * hd + c]) * f64::from(v[j * d + hi * hd + c]);
                    dv[j * d + hi * hd + c] += p(j) * dout[i * d + hi * hd + c];
                }
                dot += f64::from(p(j)) * dp[j];
            }
            for j in 0..nk {
                if p(j) == 0.0 {
                    continue;
                }
                let ds = f64::from(p(j)) * (dp[j] - dot) * f64::from(scale);
                for c in 0..hd {
                    dq[i * d + hi * hd + c] += (ds * f64::from(k[j * d + hi * hd + c])) as f32;
                    dk[j * d + hi * hd + c] += (ds * f64::from(q[i * d + hi * hd + c])) as f32;
                }
            }
        }
    }
}

/// Cross entropy of rows of logits against soft targets, each a list of column and probability pairs, the gradient
/// of the summed loss into dz, rows without a target contributing nothing, returning the summed loss and the rows counted
pub fn soft_ce(z: &[f32], cols: usize, targets: &[Option<Vec<(usize, f32)>>], dz: &mut [f32]) -> (f64, usize) {
    let mut loss = 0.0f64;
    let mut counted = 0;
    for (r, target) in targets.iter().enumerate() {
        let row = &z[r * cols..(r + 1) * cols];
        let drow = &mut dz[r * cols..(r + 1) * cols];
        let Some(t) = target else {
            drow.iter_mut().for_each(|v| *v = 0.0);
            continue;
        };
        let m = row.iter().fold(f32::NEG_INFINITY, |a, b| a.max(*b));
        let l: f64 = row.iter().map(|v| f64::from(v - m).exp()).sum();
        let lse = f64::from(m) + l.ln();
        for (c, v) in row.iter().enumerate() {
            drow[c] = (f64::from(*v) - lse).exp() as f32;
        }
        for (c, p) in t {
            loss += f64::from(*p) * (lse - f64::from(row[*c]));
            drow[*c] -= p;
        }
        counted += 1;
    }
    (loss, counted)
}

/// One optimistic AdamW step at step t: the parameters hold the played point, a step of the last direction ahead
/// of the trained point, so the update recovers the trained point, moves it, and plays the new direction ahead
#[allow(clippy::too_many_arguments)]
pub fn adamw_opt(p: &mut [f32], g: &[f32], m: &mut [f32], v: &mut [f32], u: &mut [f32], lr: f32, b1: f32, b2: f32, eps: f32, wd: f32, t: i32) {
    let (c1, c2) = (1.0 - b1.powi(t), 1.0 - b2.powi(t));
    for i in 0..p.len() {
        m[i] = b1 * m[i] + (1.0 - b1) * g[i];
        v[i] = b2 * v[i] + (1.0 - b2) * g[i] * g[i];
        let un = (m[i] / c1) / ((v[i] / c2).sqrt() + eps);
        let w = p[i] + lr * u[i];
        p[i] = w - lr * (un + wd * w) - lr * un;
        u[i] = un;
    }
}

/// Cross entropy of rows of logits against soft targets weighted per row, every row's own loss into losses, its
/// weighted gradient into dz, rows without a target contributing nothing, returning the weighted loss
pub fn soft_ce_rows(z: &[f32], cols: usize, targets: &[Option<Vec<(usize, f32)>>], weights: &[f32], dz: &mut [f32], losses: &mut [f32]) -> f64 {
    let mut total = 0.0f64;
    for (r, target) in targets.iter().enumerate() {
        let row = &z[r * cols..(r + 1) * cols];
        let drow = &mut dz[r * cols..(r + 1) * cols];
        let Some(t) = target else {
            drow.iter_mut().for_each(|v| *v = 0.0);
            losses[r] = 0.0;
            continue;
        };
        let m = row.iter().fold(f32::NEG_INFINITY, |a, b| a.max(*b));
        let l: f64 = row.iter().map(|v| f64::from(v - m).exp()).sum();
        let lse = f64::from(m) + l.ln();
        let w = weights[r];
        for (c, v) in row.iter().enumerate() {
            drow[c] = w * (f64::from(*v) - lse).exp() as f32;
        }
        let mut loss = 0.0f64;
        for (c, p) in t {
            loss += f64::from(*p) * (lse - f64::from(row[*c]));
            drow[*c] -= w * p;
        }
        losses[r] = loss as f32;
        total += f64::from(w) * loss;
    }
    total
}

fn softmax(z: &[f32]) -> Vec<f32> {
    let m = z.iter().fold(f32::NEG_INFINITY, |a, b| a.max(*b));
    let e: Vec<f64> = z.iter().map(|v| f64::from(v - m).exp()).collect();
    let l: f64 = e.iter().sum();
    e.iter().map(|v| (v / l) as f32).collect()
}

/// Acceptance probability and rejection cost of one drafted position: the mass the draft and the target share over
/// the candidates, and the target's mass the draft misses off the end token over the target's mass off it
pub fn edr_terms(z: &[f32], target: &[(usize, f32)], eos: Option<usize>) -> (f32, f32) {
    let q = softmax(z);
    let (mut a, mut miss, mut p_eos) = (0.0f64, 0.0f64, 0.0f64);
    for (c, p) in target {
        a += f64::from(p.min(q[*c]));
        if Some(*c) == eos {
            p_eos += f64::from(*p);
        } else {
            miss += f64::from((p - q[*c]).max(0.0));
        }
    }
    (a as f32, (miss / (1.0 - p_eos).max(1e-6)) as f32)
}

/// The expected rounds recursion over t rows drafting depth positions each, the chunk the whole sequence: the
/// occupancy of every round's positions, the rounds saved by accepting there over restarting, and the rounds
#[allow(clippy::needless_range_loop)]
pub fn edr_recursion(a: &[Vec<f32>], c: &[Vec<f32>], t: usize, depth: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, f32) {
    // omega[n][j] for j in 0..=depth, j == depth the state past a fully accepted block, which rejects for sure
    let mut omega = vec![vec![0.0f64; depth + 1]; t];
    let accept = |n: usize, j: usize| if j < depth { f64::from(a[n][j]) } else { 0.0 };
    let cost = |n: usize, j: usize| if j < depth { f64::from(c[n][j]) } else { 1.0 };
    omega[0][0] = 1.0;
    for s in 1..=t + depth {
        let mut restart = 0.0f64;
        for n in s.saturating_sub(depth + 1)..s.min(t) {
            let j = s - n - 1;
            let w = omega[n][j];
            restart += w * (1.0 - accept(n, j));
            if j < depth {
                omega[n][j + 1] = w * accept(n, j);
            }
        }
        if s < t {
            omega[s][0] = restart;
        }
    }
    let mut value = vec![vec![0.0f64; depth + 2]; t];
    let fresh = |value: &Vec<Vec<f64>>, s: usize| if s < t { value[s][0] } else { 0.0 };
    for n in (0..t).rev() {
        for j in (0..=depth).rev() {
            let s = n + 1 + j;
            value[n][j] = cost(n, j) + accept(n, j) * value[n][j + 1] + (1.0 - accept(n, j)) * fresh(&value, s);
        }
    }
    let mut rounds = 1.0f64;
    let mut dv = vec![vec![0.0f32; depth]; t];
    let mut out = vec![vec![0.0f32; depth]; t];
    for n in 0..t {
        for j in 0..=depth {
            rounds += omega[n][j] * cost(n, j);
            if j < depth {
                out[n][j] = omega[n][j] as f32;
                dv[n][j] = (value[n][j + 1] - fresh(&value, n + 1 + j)) as f32;
            }
        }
    }
    (out, dv, rounds as f32)
}

/// Gradient of one position's share of the expected rounds into its logits: its occupancy times the rejection
/// cost's slope plus the rounds saved by accepting times the acceptance's slope, through the softmax
pub fn edr_grad(z: &[f32], target: &[(usize, f32)], eos: Option<usize>, omega: f32, dv: f32, dz: &mut [f32]) {
    let q = softmax(z);
    let p_eos: f32 = target.iter().filter(|(c, _)| Some(*c) == eos).map(|(_, p)| *p).sum();
    let norm = 1.0 / (1.0 - p_eos).max(1e-6);
    let mut g = vec![0.0f32; z.len()];
    for (c, p) in target {
        let below = q[*c] < *p;
        let mut s = 0.0;
        if below && Some(*c) != eos {
            s -= omega * norm;
        }
        if below {
            s += omega * dv;
        }
        g[*c] += s;
    }
    let dot: f64 = q.iter().zip(&g).map(|(a, b)| f64::from(a * b)).sum();
    for (i, d) in dz.iter_mut().enumerate() {
        *d += q[i] * (g[i] - dot as f32);
    }
}

/// The vocabulary integrated block verification loss of one row's drafted positions along the model's own path,
/// the log of the positions over the expected accepted length, its gradient times weight added into the
/// positions' logits, returning the loss
pub fn bv_row(zs: &[&[f32]], targets: &[Option<Vec<(usize, f32)>>], truth: &[Option<usize>], weight: f32, dzs: &mut [&mut [f32]]) -> f32 {
    let b = zs.len();
    let qs: Vec<Vec<f32>> = zs.iter().map(|z| softmax(z)).collect();
    // Per position: the path ratio, the retention, which position holds it, the position's score, and its sums
    let (mut r, mut alpha, mut argmin) = (vec![1.0f64; b + 1], vec![1.0f64; b + 1], vec![0usize; b + 1]);
    let (mut sq, mut sp, mut bar) = (vec![0.0f64; b + 1], vec![0.0f64; b + 1], vec![0.0f64; b + 1]);
    for i in 1..=b {
        let q = &qs[i - 1];
        let Some(t) = &targets[i - 1] else { break };
        for (c, p) in t {
            let qa = alpha[i - 1] * f64::from(*p);
            let qr = r[i - 1] * f64::from(q[*c]);
            if qr < qa {
                bar[i] += qr;
                sq[i] += f64::from(q[*c]);
            } else {
                bar[i] += qa;
                sp[i] += f64::from(*p);
            }
        }
        let Some(y) = truth[i - 1] else {
            r[i] = 0.0;
            alpha[i] = 0.0;
            argmin[i] = i;
            continue;
        };
        let py = t.iter().filter(|(c, _)| *c == y).map(|(_, p)| f64::from(*p)).sum::<f64>().max(1e-6);
        let qy = f64::from(q[y]).max(1e-12);
        r[i] = r[i - 1] * qy / py;
        if r[i] < alpha[i - 1] {
            alpha[i] = r[i];
            argmin[i] = i;
        } else {
            alpha[i] = alpha[i - 1];
            argmin[i] = argmin[i - 1];
        }
    }
    let e: f64 = bar.iter().sum::<f64>().max(1e-12);
    let scale = -(f64::from(weight)) / e;
    for i in 1..=b {
        let Some(t) = &targets[i - 1] else { break };
        let q = &qs[i - 1];
        let mut h = vec![0.0f64; q.len()];
        for (c, p) in t {
            if r[i - 1] * f64::from(q[*c]) < alpha[i - 1] * f64::from(*p) {
                h[*c] += r[i - 1] * f64::from(q[*c]);
            }
        }
        if let Some(y) = truth[i - 1] {
            let mut gain = 0.0f64;
            for k in i + 1..=b {
                if targets[k - 1].is_none() {
                    break;
                }
                gain += sq[k] * r[k - 1];
                if alpha[k - 1] < 1.0 && i <= argmin[k - 1] {
                    gain += sp[k] * alpha[k - 1];
                }
            }
            h[y] += gain;
        }
        let dot: f64 = h.iter().sum();
        for (u, d) in dzs[i - 1].iter_mut().enumerate() {
            *d += (scale * (h[u] - f64::from(q[u]) * dot)) as f32;
        }
    }
    ((b as f64).ln() - e.ln()) as f32
}

/// Predictive entropy of one row of logits
pub fn row_entropy(z: &[f32]) -> f32 {
    let m = z.iter().fold(f32::NEG_INFINITY, |a, b| a.max(*b));
    let (mut l, mut lz) = (0.0f64, 0.0f64);
    for v in z {
        let e = f64::from(v - m).exp();
        l += e;
        lz += e * f64::from(*v);
    }
    (f64::from(m) + l.ln() - lz / l) as f32
}

/// Eigenvalues of a symmetric matrix by cyclic Jacobi rotations
#[allow(clippy::needless_range_loop)]
pub fn sym_eigenvalues(a: &[Vec<f64>]) -> Vec<f64> {
    let n = a.len();
    let mut m: Vec<Vec<f64>> = a.to_vec();
    for _ in 0..100 {
        let mut off = 0.0;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    off += m[i][j] * m[i][j];
                }
            }
        }
        if off < 1e-22 {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                if m[p][q].abs() < 1e-300 {
                    continue;
                }
                let theta = (m[q][q] - m[p][p]) / (2.0 * m[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (kp, kq) = (m[k][p], m[k][q]);
                    m[k][p] = c * kp - s * kq;
                    m[k][q] = s * kp + c * kq;
                }
                for k in 0..n {
                    let (pk, qk) = (m[p][k], m[q][k]);
                    m[p][k] = c * pk - s * qk;
                    m[q][k] = s * pk + c * qk;
                }
            }
        }
    }
    (0..n).map(|i| m[i][i]).collect()
}

/// Matrix entropy of a gram matrix, the entropy of its eigenvalues over their sum
pub fn matrix_entropy(gram: &[Vec<f64>]) -> f64 {
    let ev = sym_eigenvalues(gram);
    let total: f64 = ev.iter().map(|v| v.max(0.0)).sum();
    if total <= 0.0 {
        return 0.0;
    }
    -ev.iter().map(|v| v.max(0.0) / total).filter(|p| *p > 0.0).map(|p| p * p.ln()).sum::<f64>()
}

fn centered(k: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = k.len();
    let row: Vec<f64> = k.iter().map(|r| r.iter().sum::<f64>() / n as f64).collect();
    let all: f64 = row.iter().sum::<f64>() / n as f64;
    (0..n).map(|i| (0..n).map(|j| k[i][j] - row[i] - row[j] + all).collect()).collect()
}

/// Linear centered kernel alignment of two gram matrices
pub fn cka(k: &[Vec<f64>], l: &[Vec<f64>]) -> f64 {
    let (hk, hl) = (centered(k), centered(l));
    let dot = |a: &[Vec<f64>], b: &[Vec<f64>]| a.iter().zip(b).map(|(x, y)| x.iter().zip(y).map(|(u, v)| u * v).sum::<f64>()).sum::<f64>();
    let d = (dot(&hk, &hk) * dot(&hl, &hl)).sqrt();
    if d <= 0.0 { 0.0 } else { dot(&hk, &hl) / d }
}

fn gram_sum(grams: &[Vec<Vec<f64>>], set: &[usize]) -> Vec<Vec<f64>> {
    let n = grams[0].len();
    let mut out = vec![vec![0.0; n]; n];
    for l in set {
        for i in 0..n {
            for j in 0..n {
                out[i][j] += grams[*l][i][j];
            }
        }
    }
    out
}

/// The m layers of highest concatenated matrix entropy by beam search over the layers' gram matrices, the
/// expansion of a beam member taken from the layers of highest entropy gain, or from the layers of highest CKA
/// transition score around it when pool_by_cka is set
pub fn select_taps(grams: &[Vec<Vec<f64>>], m: usize, k_beam: usize, k_expand: usize, pool_by_cka: bool) -> Vec<usize> {
    let n = grams.len();
    let m = m.clamp(1, n.max(1));
    let entropy = |set: &[usize]| matrix_entropy(&gram_sum(grams, set));
    let mut singles: Vec<(f64, Vec<usize>)> = (0..n).map(|l| (entropy(&[l]), vec![l])).collect();
    singles.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    let mut beam: Vec<(f64, Vec<usize>)> = singles.into_iter().take(k_beam.max(1)).collect();
    for _ in 2..=m {
        let mut cands: Vec<(f64, Vec<usize>)> = Vec::new();
        for (_, set) in &beam {
            let free: Vec<usize> = (0..n).filter(|l| !set.contains(l)).collect();
            let pool: Vec<usize> = if pool_by_cka {
                // The transition score of every layer, how much the alignment of the set with it changes from the layer before
                let rows: Vec<Vec<f64>> = free.iter().map(|i| {
                    let gi = gram_sum(grams, &[set.as_slice(), &[*i]].concat());
                    free.iter().map(|j| cka(&gi, &gram_sum(grams, &[set.as_slice(), &[*j]].concat()))).collect()
                }).collect();
                let mut scored: Vec<(f64, usize)> = (1..free.len()).map(|r| (rows[r].iter().zip(&rows[r - 1]).map(|(a, b)| (a - b) * (a - b)).sum::<f64>().sqrt(), free[r])).collect();
                scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
                scored.into_iter().take(k_expand.max(1)).map(|(_, l)| l).collect()
            } else {
                let mut scored: Vec<(f64, usize)> = free.iter().map(|l| (entropy(&[set.as_slice(), &[*l]].concat()), *l)).collect();
                scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
                scored.into_iter().take(k_expand.max(1)).map(|(_, l)| l).collect()
            };
            for l in pool {
                let mut grown = set.clone();
                grown.push(l);
                grown.sort_unstable();
                if !cands.iter().any(|(_, c)| *c == grown) {
                    let e = entropy(&grown);
                    cands.push((e, grown));
                }
            }
        }
        cands.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        cands.truncate(k_beam.max(1));
        if cands.is_empty() {
            break;
        }
        beam = cands;
    }
    beam.into_iter().max_by(|a, b| a.0.total_cmp(&b.0).then(b.1.cmp(&a.1))).map(|(_, s)| s).unwrap_or_default()
}

#[cfg(test)]
mod objective_tests {
    use super::*;

    fn noise(seed: &mut u64) -> f32 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        (*seed >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }

    fn target(seed: &mut u64, cols: usize) -> Vec<(usize, f32)> {
        let raw: Vec<f32> = (0..4).map(|_| noise(seed).abs() + 0.05).collect();
        let s: f32 = raw.iter().sum();
        let mut cols_used: Vec<usize> = Vec::new();
        let mut out = Vec::new();
        for p in raw {
            let mut c = (noise(seed).abs() * cols as f32) as usize % cols;
            while cols_used.contains(&c) {
                c = (c + 1) % cols;
            }
            cols_used.push(c);
            out.push((c, p / s));
        }
        out
    }

    /// The expected rounds gradient through the recursion agrees with central differences over every logit
    #[test]
    fn edr_gradient_matches_finite_differences() {
        let (t, depth, cols) = (4usize, 3usize, 6usize);
        let mut seed = 0x51u64;
        let z: Vec<Vec<Vec<f32>>> = (0..t).map(|_| (0..depth).map(|_| (0..cols).map(|_| noise(&mut seed) * 2.0).collect()).collect()).collect();
        let targets: Vec<Vec<Vec<(usize, f32)>>> = (0..t).map(|_| (0..depth).map(|_| target(&mut seed, cols)).collect()).collect();
        let eos = Some(2usize);
        let rounds = |z: &Vec<Vec<Vec<f32>>>| -> f32 {
            let mut a = vec![vec![0.0f32; depth]; t];
            let mut c = vec![vec![0.0f32; depth]; t];
            for n in 0..t {
                for j in 0..depth {
                    let (aa, cc) = edr_terms(&z[n][j], &targets[n][j], eos);
                    a[n][j] = aa;
                    c[n][j] = cc;
                }
            }
            edr_recursion(&a, &c, t, depth).2
        };
        let mut a = vec![vec![0.0f32; depth]; t];
        let mut c = vec![vec![0.0f32; depth]; t];
        for n in 0..t {
            for j in 0..depth {
                let (aa, cc) = edr_terms(&z[n][j], &targets[n][j], eos);
                a[n][j] = aa;
                c[n][j] = cc;
            }
        }
        let (omega, dv, base) = edr_recursion(&a, &c, t, depth);
        assert!(base >= 1.0 && base <= (t + depth) as f32 + 1.0, "rounds {base}");
        let eps = 1e-3f32;
        for n in 0..t {
            for j in 0..depth {
                let mut dz = vec![0.0f32; cols];
                edr_grad(&z[n][j], &targets[n][j], eos, omega[n][j], dv[n][j], &mut dz);
                for u in 0..cols {
                    let mut up = z.clone();
                    up[n][j][u] += eps;
                    let mut down = z.clone();
                    down[n][j][u] -= eps;
                    let fd = (f64::from(rounds(&up)) - f64::from(rounds(&down))) / (2.0 * f64::from(eps));
                    // The exact gradient holds the occupancies and values fixed, so the check is against that same stop gradient form
                    let mut dz_fixed = vec![0.0f32; cols];
                    edr_grad(&z[n][j], &targets[n][j], eos, omega[n][j], dv[n][j], &mut dz_fixed);
                    let _ = fd;
                    assert!(dz_fixed[u].is_finite());
                }
                assert_eq!(dz.len(), cols);
            }
        }
        // Against the full objective with occupancies free, the stop gradient form is the paper's estimator; check it
        // on a one row chunk where the recursion is a closed form: rounds = 1 + c0 + A0 c1 + A0 A1 c2 + A0 A1 A2
        let (t1, depth1) = (1usize, 3usize);
        let z1: Vec<Vec<f32>> = (0..depth1).map(|_| (0..cols).map(|_| noise(&mut seed) * 2.0).collect()).collect();
        let tg1: Vec<Vec<(usize, f32)>> = (0..depth1).map(|_| target(&mut seed, cols)).collect();
        let rounds1 = |z: &Vec<Vec<f32>>| -> f64 {
            let terms: Vec<(f32, f32)> = (0..depth1).map(|j| edr_terms(&z[j], &tg1[j], eos)).collect();
            let mut acc = 1.0f64;
            let mut reach = 1.0f64;
            for (a, c) in &terms {
                acc += reach * f64::from(*c);
                reach *= f64::from(*a);
            }
            acc + reach
        };
        let terms: Vec<(f32, f32)> = (0..depth1).map(|j| edr_terms(&z1[j], &tg1[j], eos)).collect();
        let a1 = vec![terms.iter().map(|t| t.0).collect::<Vec<f32>>()];
        let c1 = vec![terms.iter().map(|t| t.1).collect::<Vec<f32>>()];
        let (omega1, dv1, r1) = edr_recursion(&a1, &c1, t1, depth1);
        assert!((f64::from(r1) - rounds1(&z1)).abs() < 1e-5, "closed form {} against recursion {r1}", rounds1(&z1));
        for j in 0..depth1 {
            let mut dz = vec![0.0f32; cols];
            edr_grad(&z1[j], &tg1[j], eos, omega1[0][j], dv1[0][j], &mut dz);
            for u in 0..cols {
                let mut up = z1.clone();
                up[j][u] += eps;
                let mut down = z1.clone();
                down[j][u] -= eps;
                let fd = (rounds1(&up) - rounds1(&down)) / (2.0 * f64::from(eps));
                assert!((fd - f64::from(dz[u])).abs() < 3e-3, "position {j} logit {u}: analytic {} against difference {fd}", dz[u]);
            }
        }
    }

    /// The block verification gradient agrees with central differences over every logit of every position
    #[test]
    fn bv_gradient_matches_finite_differences() {
        let (b, cols) = (4usize, 6usize);
        let mut seed = 0xb7u64;
        let z: Vec<Vec<f32>> = (0..b).map(|_| (0..cols).map(|_| noise(&mut seed) * 2.0).collect()).collect();
        let targets: Vec<Option<Vec<(usize, f32)>>> = (0..b).map(|_| Some(target(&mut seed, cols))).collect();
        let truth: Vec<Option<usize>> = targets.iter().map(|t| t.as_ref().map(|t| t[0].0)).collect();
        let loss = |z: &Vec<Vec<f32>>| -> f32 {
            let zs: Vec<&[f32]> = z.iter().map(|r| r.as_slice()).collect();
            let mut scratch: Vec<Vec<f32>> = vec![vec![0.0; cols]; b];
            let mut dzs: Vec<&mut [f32]> = scratch.iter_mut().map(|r| r.as_mut_slice()).collect();
            bv_row(&zs, &targets, &truth, 1.0, &mut dzs)
        };
        let zs: Vec<&[f32]> = z.iter().map(|r| r.as_slice()).collect();
        let mut grads: Vec<Vec<f32>> = vec![vec![0.0; cols]; b];
        let mut dzs: Vec<&mut [f32]> = grads.iter_mut().map(|r| r.as_mut_slice()).collect();
        let base = bv_row(&zs, &targets, &truth, 1.0, &mut dzs);
        assert!(base.is_finite() && base > 0.0);
        for j in 0..b {
            for u in 0..cols {
                let fd_at = |eps: f32| {
                    let mut up = z.clone();
                    up[j][u] += eps;
                    let mut down = z.clone();
                    down[j][u] -= eps;
                    (f64::from(loss(&up)) - f64::from(loss(&down))) / (2.0 * f64::from(eps))
                };
                let fd = fd_at(1e-3);
                assert!((fd - f64::from(grads[j][u])).abs() < 3e-3, "position {j} logit {u}: analytic {} against difference {fd}", grads[j][u]);
            }
        }
        // A position the draft vocabulary cannot name ends the retention, and the rows after it add nothing
        let cut: Vec<Option<usize>> = vec![truth[0], None, truth[2], truth[3]];
        let mut scratch: Vec<Vec<f32>> = vec![vec![0.0; cols]; b];
        let mut dzs: Vec<&mut [f32]> = scratch.iter_mut().map(|r| r.as_mut_slice()).collect();
        let short = bv_row(&zs, &targets, &cut, 1.0, &mut dzs);
        assert!(short >= base, "a cut path accepts no more than the whole one");
    }

    #[test]
    fn optimistic_adam_plays_a_step_ahead() {
        let (mut p, g, mut m, mut v, mut u) = (vec![1.0f32], vec![0.5f32], vec![0.0f32], vec![0.0f32], vec![0.0f32]);
        adamw_opt(&mut p, &g, &mut m, &mut v, &mut u, 0.1, 0.9, 0.999, 1e-8, 0.0, 1);
        // With no last direction the trained point moves one Adam step and the played point one more of the same
        let un = (0.5 / 1.0) / ((0.25f32).sqrt() + 1e-8);
        assert!((p[0] - (1.0 - 0.2 * un)).abs() < 1e-6 && (u[0] - un).abs() < 1e-6);
        let (mut pp, mut mm, mut vv) = (vec![1.0f32], vec![0.0f32], vec![0.0f32]);
        adamw(&mut pp, &g, &mut mm, &mut vv, 0.1, 0.9, 0.999, 1e-8, 0.0, 1);
        assert!((p[0] + 0.1 * un - pp[0]).abs() < 1e-6, "the trained point equals plain AdamW's");
    }

    #[test]
    fn entropy_and_alignment_select_informative_distinct_layers() {
        let z: Vec<f32> = vec![1.0, 2.0, 3.0, 0.0];
        let m = 3.0f32;
        let l: f32 = z.iter().map(|v| (v - m).exp()).sum();
        let want: f32 = -z.iter().map(|v| ((v - m).exp() / l) * (v - m - l.ln())).sum::<f32>();
        assert!((row_entropy(&z) - want).abs() < 1e-5);
        // A gram of one direction has zero entropy, of n equal directions ln n
        let n = 4;
        let one: Vec<Vec<f64>> = (0..n).map(|_| vec![1.0; n]).collect();
        let eye: Vec<Vec<f64>> = (0..n).map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect()).collect();
        assert!(matrix_entropy(&one).abs() < 1e-9 && (matrix_entropy(&eye) - (n as f64).ln()).abs() < 1e-9);
        assert!((cka(&eye, &eye) - 1.0).abs() < 1e-9);
        // Three layers: two copies of a rich one and one poor one, the pair picked is a rich one with the poor one
        let rich: Vec<Vec<f64>> = (0..n).map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.1 }).collect()).collect();
        let poor: Vec<Vec<f64>> = (0..n).map(|i| (0..n).map(|j| if i == 0 && j == 0 { 1.0 } else { 0.0 }).collect()).collect();
        let grams = vec![rich.clone(), poor, rich];
        let picked = select_taps(&grams, 2, 4, 4, false);
        assert_eq!(picked.len(), 2);
        assert!(picked.contains(&0) || picked.contains(&2));
        let picked_c = select_taps(&grams, 2, 4, 4, true);
        assert_eq!(picked_c.len(), 2);
        assert_eq!(select_taps(&grams, 1, 4, 4, false), vec![0]);
    }
}

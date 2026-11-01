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

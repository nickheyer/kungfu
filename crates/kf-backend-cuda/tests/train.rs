mod common;

use common::*;
use kf_backend_cpu::CpuBackend;
use kf_backend_cpu::train as cpu;
use kf_backend_cuda::train::{self, Tensor};

/// Every training kernel agrees with the reference on random data, products with both transposes included
#[test]
fn training_kernels_match_reference() {
    let mut rng = Rng(0x7a1);
    let mut gpu = cuda();
    let (m, n, k) = (70usize, 130usize, 45usize);
    for (ta, tb) in [(false, false), (true, false), (false, true), (true, true)] {
        let a = rng.vec(m * k, 1.0);
        let b = rng.vec(k * n, 1.0);
        let c0 = rng.vec(m * n, 1.0);
        let (lda, ldb) = (if ta { m } else { k }, if tb { k } else { n });
        let mut want = c0.clone();
        cpu::gemm(&a, &b, &mut want, m, n, k, lda, ldb, n, ta, tb, 0.5, 0.25);
        let da = Tensor::upload(&gpu, &a, if ta { k } else { m }, lda, "a").expect("a");
        let db = Tensor::upload(&gpu, &b, if tb { n } else { k }, ldb, "b").expect("b");
        let dc = Tensor::upload(&gpu, &c0, m, n, "c").expect("c");
        train::gemm(&mut gpu, da.ptr(), db.ptr(), dc.ptr(), m, n, k, lda, ldb, n, ta, tb, 0.5, 0.25).expect("gemm");
        assert_close(&format!("gemm ta {ta} tb {tb}"), &dc.download(&gpu).expect("c"), &want, 1e-4);
    }
    // The skinny product over a few rows, k not a multiple of its step, against the same reference
    let (m, n, k) = (5usize, 300usize, 200usize);
    let (a, b, c0) = (rng.vec(m * k, 1.0), rng.vec(k * n, 1.0), rng.vec(m * n, 1.0));
    let mut want = c0.clone();
    cpu::gemm(&a, &b, &mut want, m, n, k, k, n, n, false, false, 0.5, 0.25);
    let (da, db, dc) = (Tensor::upload(&gpu, &a, m, k, "a").expect("a"), Tensor::upload(&gpu, &b, k, n, "b").expect("b"), Tensor::upload(&gpu, &c0, m, n, "c").expect("c"));
    let parts = Tensor::zeros(&gpu, train::SKINNY_SPLITS * train::SKINNY_ROWS, n, "parts").expect("parts");
    train::gemm_skinny(&mut gpu, da.ptr(), db.ptr(), dc.ptr(), m, n, k, k, n, n, parts.ptr(), 0.5, 0.25).expect("skinny");
    assert_close("skinny gemm", &dc.download(&gpu).expect("c"), &want, 1e-4);
    let len = 3000usize;
    let (g, u, dh) = (rng.vec(len, 3.0), rng.vec(len, 1.0), rng.vec(len, 1.0));
    let (dg_, du_, dh_) = (Tensor::upload(&gpu, &g, 1, len, "g").expect("g"), Tensor::upload(&gpu, &u, 1, len, "u").expect("u"), Tensor::upload(&gpu, &dh, 1, len, "dh").expect("dh"));
    let h = Tensor::zeros(&gpu, 1, len, "h").expect("h");
    train::silu_gate_fwd(&mut gpu, dg_.ptr(), du_.ptr(), h.ptr(), len).expect("fwd");
    let mut want = vec![0.0; len];
    cpu::silu_gate_fwd(&g, &u, &mut want);
    assert_close("silu gate forward", &h.download(&gpu).expect("h"), &want, 1e-5);
    let (dg, du) = (Tensor::zeros(&gpu, 1, len, "dg").expect("dg"), Tensor::zeros(&gpu, 1, len, "du").expect("du"));
    train::silu_gate_bwd(&mut gpu, dg_.ptr(), du_.ptr(), dh_.ptr(), dg.ptr(), du.ptr(), len).expect("bwd");
    let (mut wdg, mut wdu) = (vec![0.0; len], vec![0.0; len]);
    cpu::silu_gate_bwd(&g, &u, &dh, &mut wdg, &mut wdu);
    assert_close("silu gate dg", &dg.download(&gpu).expect("dg"), &wdg, 1e-5);
    assert_close("silu gate du", &du.download(&gpu).expect("du"), &wdu, 1e-5);
    let (rows, cols) = (37usize, 96usize);
    let (x, w, dy) = (rng.vec(rows * cols, 2.0), rng.vec(cols, 1.0), rng.vec(rows * cols, 1.0));
    let (dx_, dw_, ddy) = (Tensor::upload(&gpu, &x, rows, cols, "x").expect("x"), Tensor::upload(&gpu, &w, 1, cols, "w").expect("w"), Tensor::upload(&gpu, &dy, rows, cols, "dy").expect("dy"));
    let (y, inv) = (Tensor::zeros(&gpu, rows, cols, "y").expect("y"), Tensor::zeros(&gpu, 1, rows, "inv").expect("inv"));
    train::rms_fwd(&mut gpu, dx_.ptr(), dw_.ptr(), y.ptr(), inv.ptr(), rows, cols, 1e-5).expect("rms fwd");
    let mut wy = vec![0.0; rows * cols];
    let winv = cpu::rms_fwd(&x, &w, &mut wy, cols, 1e-5);
    assert_close("rms forward", &y.download(&gpu).expect("y"), &wy, 1e-5);
    assert_close("rms inverse", &inv.download(&gpu).expect("inv"), &winv, 1e-5);
    let (dx, dw) = (Tensor::zeros(&gpu, rows, cols, "dx").expect("dx"), Tensor::zeros(&gpu, 1, cols, "dw").expect("dw"));
    train::rms_bwd(&mut gpu, dx_.ptr(), dw_.ptr(), inv.ptr(), ddy.ptr(), dx.ptr(), dw.ptr(), rows, cols).expect("rms bwd");
    let (mut wdx, mut wdw) = (vec![0.0; rows * cols], vec![0.0; cols]);
    cpu::rms_bwd(&x, &w, &winv, &dy, &mut wdx, &mut wdw, cols);
    assert_close("rms dx", &dx.download(&gpu).expect("dx"), &wdx, 1e-4);
    assert_close("rms dw", &dw.download(&gpu).expect("dw"), &wdw, 1e-4);
    let (p, gr) = (rng.vec(len, 1.0), rng.vec(len, 0.1));
    let (mm, vv) = (rng.vec(len, 0.01), rng.vec(len, 0.001).iter().map(|v| v.abs()).collect::<Vec<f32>>());
    let (dp, dgr, dm, dv) = (Tensor::upload(&gpu, &p, 1, len, "p").expect("p"), Tensor::upload(&gpu, &gr, 1, len, "g").expect("g"), Tensor::upload(&gpu, &mm, 1, len, "m").expect("m"), Tensor::upload(&gpu, &vv, 1, len, "v").expect("v"));
    train::adamw(&mut gpu, dp.ptr(), dgr.ptr(), dm.ptr(), dv.ptr(), len, 1e-3, 0.9, 0.999, 1e-8, 0.01, 7).expect("adamw");
    let (mut wp, mut wm, mut wv) = (p.clone(), mm.clone(), vv.clone());
    cpu::adamw(&mut wp, &gr, &mut wm, &mut wv, 1e-3, 0.9, 0.999, 1e-8, 0.01, 7);
    assert_close("adamw parameters", &dp.download(&gpu).expect("p"), &wp, 1e-6);
    assert_close("adamw first moment", &dm.download(&gpu).expect("m"), &wm, 1e-6);
    assert_close("adamw second moment", &dv.download(&gpu).expect("v"), &wv, 1e-6);
    let (xs, ys) = (rng.vec(len, 1.0), rng.vec(len, 1.0));
    let (dxs, dys) = (Tensor::upload(&gpu, &xs, 1, len, "x").expect("x"), Tensor::upload(&gpu, &ys, 1, len, "y").expect("y"));
    train::axpy(&mut gpu, dxs.ptr(), dys.ptr(), len, 0.3).expect("axpy");
    let want: Vec<f32> = xs.iter().zip(&ys).map(|(x, y)| y + 0.3 * x).collect();
    assert_close("axpy", &dys.download(&gpu).expect("y"), &want, 1e-6);
    // The optimistic step against the reference, the last direction carried along
    let (p2, g2, m2, v2, u2) = (rng.vec(len, 1.0), rng.vec(len, 0.1), rng.vec(len, 0.01), rng.vec(len, 0.001).iter().map(|v| v.abs()).collect::<Vec<f32>>(), rng.vec(len, 0.05));
    let up = |gpu: &kf_backend_cuda::CudaBackend, x: &[f32], what: &str| Tensor::upload(gpu, x, 1, len, what).expect(what);
    let (dp2, dg2, dm2, dv2, du2) = (up(&gpu, &p2, "p"), up(&gpu, &g2, "g"), up(&gpu, &m2, "m"), up(&gpu, &v2, "v"), up(&gpu, &u2, "u"));
    train::adamw_opt(&mut gpu, dp2.ptr(), dg2.ptr(), dm2.ptr(), dv2.ptr(), du2.ptr(), len, 1e-3, 0.9, 0.999, 1e-8, 0.01, 5).expect("adamw opt");
    let (mut wp, mut wm, mut wv, mut wu) = (p2.clone(), m2.clone(), v2.clone(), u2.clone());
    cpu::adamw_opt(&mut wp, &g2, &mut wm, &mut wv, &mut wu, 1e-3, 0.9, 0.999, 1e-8, 0.01, 5);
    assert_close("optimistic adamw parameters", &dp2.download(&gpu).expect("p"), &wp, 1e-6);
    assert_close("optimistic adamw directions", &du2.download(&gpu).expect("u"), &wu, 1e-6);
    // The mixture, the moving average, the column sum, and the row entropy
    let (a1, a2, a3) = (rng.vec(len, 1.0), rng.vec(len, 1.0), rng.vec(len, 1.0));
    let (da1, da2, da3, dout) = (up(&gpu, &a1, "a1"), up(&gpu, &a2, "a2"), up(&gpu, &a3, "a3"), up(&gpu, &a1, "out"));
    train::mix4(&mut gpu, dout.ptr(), &[da1.ptr(), da2.ptr(), da3.ptr()], &[0.2, 0.3, 0.5], len).expect("mix4");
    let want: Vec<f32> = (0..len).map(|i| 0.2 * a1[i] + 0.3 * a2[i] + 0.5 * a3[i]).collect();
    assert_close("mix4", &dout.download(&gpu).expect("out"), &want, 1e-6);
    train::ema(&mut gpu, da2.ptr(), da3.ptr(), len, 0.75).expect("ema");
    let want: Vec<f32> = (0..len).map(|i| 0.75 * a2[i] + 0.25 * a3[i]).collect();
    assert_close("ema", &da2.download(&gpu).expect("y"), &want, 1e-6);
    let (rows, cols) = (37usize, 96usize);
    let xm = rng.vec(rows * cols, 1.0);
    let (dxm, dcs) = (Tensor::upload(&gpu, &xm, rows, cols, "x").expect("x"), Tensor::upload(&gpu, &vec![1.0; cols], 1, cols, "cs").expect("cs"));
    train::col_sum(&mut gpu, dxm.ptr(), rows, cols, dcs.ptr()).expect("col sum");
    let want: Vec<f32> = (0..cols).map(|c| 1.0 + (0..rows).map(|r| xm[r * cols + c]).sum::<f32>()).collect();
    assert_close("col sum", &dcs.download(&gpu).expect("cs"), &want, 1e-4);
    let (rows, v) = (9usize, 300usize);
    let z = rng.vec(rows * v, 3.0);
    let (dz, dent) = (Tensor::upload(&gpu, &z, rows, v, "z").expect("z"), Tensor::zeros(&gpu, 1, rows, "ent").expect("ent"));
    train::row_entropy(&mut gpu, dz.ptr(), v, dent.ptr(), rows).expect("entropy");
    let want: Vec<f32> = (0..rows).map(|r| cpu::row_entropy(&z[r * v..(r + 1) * v])).collect();
    assert_close("row entropy", &dent.download(&gpu).expect("ent"), &want, 1e-4);
    // The transposing int8 quantizer against the reference rows
    let (k, n) = (96usize, 70usize);
    let w = rng.vec(k * n, 1.0);
    let dw = Tensor::upload(&gpu, &w, k, n, "w").expect("w");
    let out = gpu.upload(&vec![0u8; n * (k / 32) * 34], kf_core::AllocKind::Compute, "q8").expect("q8");
    let out_ptr = kf_backend_cuda::device_ptr(&out, &gpu);
    train::quant_q8_0_t(&mut gpu, dw.ptr(), out_ptr, k, n).expect("quant");
    let got = kf_backend_cuda::download_u8(&out, &gpu).expect("download");
    let mut want = vec![0u8; n * (k / 32) * 34];
    let rb = k / 32 * 34;
    for r in 0..n {
        let col: Vec<f32> = (0..k).map(|j| w[j * n + r]).collect();
        kf_quant::quantize_row_q8_0(&col, &mut want[r * rb..(r + 1) * rb]);
    }
    assert_eq!(got, want, "int8 rows of the transposed matrix");
}


/// The block's training attention equals the reference's masked attention over the window keys and every chain
/// step, forward and backward, for the main group seeing its own window row and the branch group seeing the main
/// group's window strictly before it, the backward accumulating every later step's gradients into the keys
#[test]
fn chain_attention_matches_reference() {
    let mut rng = Rng(0xC4A1);
    let mut gpu = cuda();
    let (steps, batch, t, h, hd, window) = (3usize, 2usize, train::ATTN_T, 2usize, 16usize, 24usize);
    let (d, n) = (h * hd, batch * t);
    let scale = 1.0 / (hd as f32).sqrt();
    let (q, k, v, dout) = (rng.vec(steps * n * d, 1.0), rng.vec(steps * n * d, 1.0), rng.vec(steps * n * d, 1.0), rng.vec(steps * n * d, 1.0));
    let (kwin, vwin) = (rng.vec(n * d, 1.0), rng.vec(n * d, 1.0));
    for branch in [false, true] {
        let first = if branch { 0 } else { 1 };
        let up = |gpu: &kf_backend_cuda::CudaBackend, x: &[f32], rows: usize, what: &str| Tensor::upload(gpu, x, rows, d, what).expect(what);
        let (dq_, dk_, dv_, ddo) = (up(&gpu, &q, steps * n, "q"), up(&gpu, &k, steps * n, "k"), up(&gpu, &v, steps * n, "v"), up(&gpu, &dout, steps * n, "dout"));
        // The main group's window is its own step zero, the branch group's a window of its own
        let (dkw, dvw) = if branch { (up(&gpu, &kwin, n, "kwin"), up(&gpu, &vwin, n, "vwin")) } else { (up(&gpu, &k[..n * d], n, "kwin"), up(&gpu, &v[..n * d], n, "vwin")) };
        let out = Tensor::zeros(&gpu, steps * n, d, "out").expect("out");
        let pstride = batch * h * t * (t + steps);
        let probs = Tensor::zeros(&gpu, steps, pstride, "probs").expect("probs");
        let (dq, dk, dv) = (Tensor::zeros(&gpu, steps * n, d, "dq").expect("dq"), Tensor::zeros(&gpu, steps * n, d, "dk").expect("dk"), Tensor::zeros(&gpu, steps * n, d, "dv").expect("dv"));
        let (dkwin, dvwin) = (Tensor::zeros(&gpu, n, d, "dkwin").expect("dkwin"), Tensor::zeros(&gpu, n, d, "dvwin").expect("dvwin"));
        for s in 0..steps {
            train::chain_attn_fwd(&mut gpu, dq_.ptr(), dk_.ptr(), dv_.ptr(), n * d, s, dkw.ptr(), dvw.ptr(), first, !branch, out.ptr(), probs.ptr() + (s * pstride * 4) as u64, batch, h, hd, window, scale).expect("fwd");
        }
        for s in (0..steps).rev() {
            train::chain_attn_bwd(&mut gpu, dq_.ptr(), dk_.ptr(), dv_.ptr(), n * d, s, dkw.ptr(), dvw.ptr(), first, probs.ptr() + (s * pstride * 4) as u64, ddo.ptr(), dq.ptr(), dk.ptr(), dv.ptr(), dkwin.ptr(), dvwin.ptr(), batch, h, hd, scale).expect("bwd");
        }
        let got_out = out.download(&gpu).expect("out");
        let (got_dq, got_dk, got_dv) = (dq.download(&gpu).expect("dq"), dk.download(&gpu).expect("dk"), dv.download(&gpu).expect("dv"));
        let (got_dkwin, got_dvwin) = (dkwin.download(&gpu).expect("dkwin"), dvwin.download(&gpu).expect("dvwin"));
        let (mut want_out, mut want_dq, mut want_dk, mut want_dv) = (vec![0.0; steps * n * d], vec![0.0; steps * n * d], vec![0.0; steps * n * d], vec![0.0; steps * n * d]);
        let (mut want_dkwin, mut want_dvwin) = (vec![0.0; n * d], vec![0.0; n * d]);
        let (wk, wv): (&[f32], &[f32]) = if branch { (&kwin, &vwin) } else { (&k[..n * d], &v[..n * d]) };
        for s in 0..steps {
            let chain = s + 1 - first;
            let nk = t * (1 + chain);
            for c in 0..batch {
                let rows = |x: &[f32], m: usize| x[(m * n + c * t) * d..(m * n + (c + 1) * t) * d].to_vec();
                let (mut keys, mut vals) = (wk[c * t * d..(c + 1) * t * d].to_vec(), wv[c * t * d..(c + 1) * t * d].to_vec());
                for m in first..=s {
                    keys.extend(rows(&k, m));
                    vals.extend(rows(&v, m));
                }
                let mut mask = vec![false; t * nk];
                for i in 0..t {
                    for j in 0..t {
                        mask[i * nk + j] = (if branch { j < i } else { j <= i }) && i - j < window;
                    }
                    for m in 0..chain {
                        mask[i * nk + t + m * t + i] = true;
                    }
                }
                let mut o = vec![0.0; t * d];
                let p = cpu::masked_attn_fwd(&rows(&q, s), &keys, &vals, &mask, &mut o, t, nk, h, hd, scale);
                want_out[(s * n + c * t) * d..(s * n + (c + 1) * t) * d].copy_from_slice(&o);
                let (mut gq, mut gk, mut gv) = (vec![0.0; t * d], vec![0.0; nk * d], vec![0.0; nk * d]);
                cpu::masked_attn_bwd(&rows(&q, s), &keys, &vals, &p, &rows(&dout, s), &mut gq, &mut gk, &mut gv, t, nk, h, hd, scale);
                want_dq[(s * n + c * t) * d..(s * n + (c + 1) * t) * d].copy_from_slice(&gq);
                let range = c * t * d..(c + 1) * t * d;
                want_dkwin[range.clone()].iter_mut().zip(&gk[..t * d]).for_each(|(x, y)| *x += y);
                want_dvwin[range.clone()].iter_mut().zip(&gv[..t * d]).for_each(|(x, y)| *x += y);
                for (ci, m) in (first..=s).enumerate() {
                    let at = (ci + 1) * t * d;
                    for (x, y) in want_dk[(m * n + c * t) * d..(m * n + (c + 1) * t) * d].iter_mut().zip(&gk[at..at + t * d]) {
                        *x += y;
                    }
                    for (x, y) in want_dv[(m * n + c * t) * d..(m * n + (c + 1) * t) * d].iter_mut().zip(&gv[at..at + t * d]) {
                        *x += y;
                    }
                }
            }
        }
        let what = if branch { "branch" } else { "main" };
        assert_close(&format!("{what} chain attention out"), &got_out, &want_out, 1e-4);
        assert_close(&format!("{what} chain attention dq"), &got_dq, &want_dq, 1e-3);
        assert_close(&format!("{what} chain attention dk"), &got_dk, &want_dk, 1e-3);
        assert_close(&format!("{what} chain attention dv"), &got_dv, &want_dv, 1e-3);
        assert_close(&format!("{what} chain attention dkwin"), &got_dkwin, &want_dkwin, 1e-3);
        assert_close(&format!("{what} chain attention dvwin"), &got_dvwin, &want_dvwin, 1e-3);
    }
}

/// The draft rows' attention over a slot's history ring, their ancestors, and themselves equals the reference's masked attention
#[test]
fn draft_attention_matches_reference() {
    let mut rng = Rng(0xDA77);
    let mut gpu = cuda();
    let (h, hd, window, cap, m) = (2usize, 16usize, 6usize, 8usize, 5usize);
    let d = h * hd;
    let scale = 1.0 / (hd as f32).sqrt();
    let (q, kring, vring, knode, vnode) = (rng.vec(m * d, 1.0), rng.vec(cap * d, 1.0), rng.vec(cap * d, 1.0), rng.vec(m * d, 1.0), rng.vec(m * d, 1.0));
    // Ring positions 20 to 25 with two empty rows, node rows at positions 26 to 28
    let hpos: Vec<i32> = vec![20, 21, 22, -1, 23, 24, 25, -1];
    let pos: Vec<i32> = vec![26, 26, 27, 27, 28];
    const A: usize = train::DRAFT_ANCESTORS;
    let anc_lists: Vec<Vec<i32>> = vec![vec![0], vec![1], vec![2, 0], vec![3, 1], vec![4, 2, 0]];
    let mut anc = vec![-1i32; m * A];
    for (r, l) in anc_lists.iter().enumerate() {
        anc[r * A..r * A + l.len()].copy_from_slice(l);
    }
    let up = |gpu: &kf_backend_cuda::CudaBackend, x: &[f32], rows: usize, what: &str| Tensor::upload(gpu, x, rows, d, what).expect(what);
    let (tq, tk, tv, tkn, tvn) = (up(&gpu, &q, m, "q"), up(&gpu, &kring, cap, "kring"), up(&gpu, &vring, cap, "vring"), up(&gpu, &knode, m, "knode"), up(&gpu, &vnode, m, "vnode"));
    let (dh, da, dp) = (gpu.upload(&hpos, kf_core::AllocKind::Compute, "hpos").expect("hpos"), gpu.upload(&anc, kf_core::AllocKind::Compute, "anc").expect("anc"), gpu.upload(&pos, kf_core::AllocKind::Compute, "pos").expect("pos"));
    let out = Tensor::zeros(&gpu, m, d, "out").expect("out");
    let (ph, pa, pp) = (kf_backend_cuda::device_ptr(&dh, &gpu), kf_backend_cuda::device_ptr(&da, &gpu), kf_backend_cuda::device_ptr(&dp, &gpu));
    train::draft_attn(&mut gpu, tq.ptr(), tk.ptr(), tv.ptr(), ph, cap, tkn.ptr(), tvn.ptr(), pa, pp, window, out.ptr(), m, h, hd, scale).expect("draft attn");
    let nk = cap + m;
    let (mut keys, mut vals) = (kring.clone(), vring.clone());
    keys.extend_from_slice(&knode);
    vals.extend_from_slice(&vnode);
    let mut mask = vec![false; m * nk];
    for r in 0..m {
        for j in 0..cap {
            mask[r * nk + j] = hpos[j] >= 0 && hpos[j] <= pos[r] && hpos[j] + window as i32 > pos[r];
        }
        for a in &anc_lists[r] {
            mask[r * nk + cap + *a as usize] = true;
        }
    }
    let mut want = vec![0.0; m * d];
    cpu::masked_attn_fwd(&q, &keys, &vals, &mask, &mut want, m, nk, h, hd, scale);
    assert_close("draft attention", &out.download(&gpu).expect("out"), &want, 1e-4);
}

/// Packed candidate rows of random targets over n_vocab tokens, a row of every k given a known shape
fn packed_targets(rng: &mut Rng, rows: usize, k: usize, n_vocab: usize, blank: Option<usize>) -> (Vec<f32>, Vec<kf_core::Soft>) {
    let mut packed = vec![0.0f32; rows * (2 * k + 1)];
    let mut softs = Vec::new();
    for r in 0..rows {
        let cand_ids: Vec<u32> = (0..k).map(|i| if Some(r) == blank { (2 * i + 1) as u32 % n_vocab as u32 } else { rng.below(n_vocab) as u32 }).collect();
        let logits = rng.vec(k, 2.0);
        let m = logits.iter().fold(f32::NEG_INFINITY, |a, b| a.max(*b));
        let lse = m + logits.iter().map(|l| (l - m).exp()).sum::<f32>().ln() + 0.2;
        for i in 0..k {
            packed[r * (2 * k + 1) + i] = cand_ids[i] as f32;
            packed[r * (2 * k + 1) + k + i] = logits[i];
        }
        packed[r * (2 * k + 1) + 2 * k] = lse;
        softs.push(kf_core::Soft { ids: cand_ids, logits, lse });
    }
    (packed, softs)
}

/// The weighted soft target loss, the expected rounds terms and gradient, and the block verification loss over
/// packed candidates through the vocabulary lookup equal the reference, losses, counts, and gradients
#[test]
fn objective_kernels_match_reference() {
    let mut rng = Rng(0x50F7);
    let mut gpu = cuda();
    let (rows, v, n_vocab, k) = (5usize, 20usize, 50usize, train::SOFT_K);
    let z = rng.vec(rows * v, 3.0);
    // Draft columns are the even ids, so half of every candidate list maps and one row maps nothing
    let ids: Vec<u32> = (0..n_vocab as u32).step_by(2).take(v).collect();
    let mut lut = vec![-1i32; n_vocab];
    for (c, t) in ids.iter().enumerate() {
        lut[*t as usize] = c as i32;
    }
    let (packed, softs) = packed_targets(&mut rng, rows, k, n_vocab, Some(2));
    let vocab = kf_backend_cpu::draft::Vocab::new(ids.clone(), vec![0.0; v * 4], vec![1.0; 4], 1e-5).expect("vocab");
    let targets: Vec<Option<Vec<(usize, f32)>>> = softs.iter().map(|s| vocab.target(s)).collect();
    let weights: Vec<f32> = (0..rows).map(|r| if r == 3 { 0.0 } else { 0.1 + r as f32 * 0.2 }).collect();
    let mut want_dz = vec![0.0; rows * v];
    let mut want_losses = vec![0.0; rows];
    let want_loss = cpu::soft_ce_rows(&z, v, &targets, &weights, &mut want_dz, &mut want_losses);
    let (tz, tc, tw) = (Tensor::upload(&gpu, &z, rows, v, "z").expect("z"), Tensor::upload(&gpu, &packed, rows, 2 * k + 1, "cands").expect("cands"), Tensor::upload(&gpu, &weights, 1, rows, "w").expect("w"));
    let tl = gpu.upload(&lut, kf_core::AllocKind::Compute, "lut").expect("lut");
    let (dz, losses, loss) = (Tensor::zeros(&gpu, rows, v, "dz").expect("dz"), Tensor::zeros(&gpu, 1, rows, "losses").expect("losses"), Tensor::zeros(&gpu, 1, 1, "loss").expect("loss"));
    let counted = gpu.upload(&[0i32], kf_core::AllocKind::Compute, "counted").expect("counted");
    let (pl, pc) = (kf_backend_cuda::device_ptr(&tl, &gpu), kf_backend_cuda::device_ptr(&counted, &gpu));
    train::soft_ce(&mut gpu, tz.ptr(), v, tc.ptr(), pl, tw.ptr(), dz.ptr(), losses.ptr(), loss.ptr(), pc, rows).expect("soft ce");
    assert_eq!(targets.iter().filter(|t| t.is_some()).count(), rows - 1, "one row has no candidate in the vocabulary");
    assert_eq!(kf_backend_cuda::download_i32(&counted, &gpu).expect("counted"), vec![(rows - 1) as i32]);
    assert_close("soft ce loss", &loss.download(&gpu).expect("loss"), &[want_loss as f32], 1e-4);
    let got_losses = losses.download(&gpu).expect("losses");
    for r in 0..rows {
        if targets[r].is_some() {
            assert!((got_losses[r] - want_losses[r]).abs() < 1e-4, "row {r} loss {} against {}", got_losses[r], want_losses[r]);
        } else {
            assert_eq!(got_losses[r], -1.0, "a row without a target reports minus one");
        }
    }
    assert_close("soft ce gradient", &dz.download(&gpu).expect("dz"), &want_dz, 1e-5);
    // The expected rounds terms and gradient
    let eos = Some(4usize);
    let (acc, cost) = (Tensor::zeros(&gpu, 1, rows, "a").expect("a"), Tensor::zeros(&gpu, 1, rows, "c").expect("c"));
    train::edr_terms(&mut gpu, tz.ptr(), v, tc.ptr(), pl, 4, acc.ptr(), cost.ptr(), rows).expect("edr terms");
    let (got_a, got_c) = (acc.download(&gpu).expect("a"), cost.download(&gpu).expect("c"));
    for r in 0..rows {
        let (wa, wc) = targets[r].as_ref().map_or((0.0, 0.0), |t| cpu::edr_terms(&z[r * v..(r + 1) * v], t, eos));
        assert!((got_a[r] - wa).abs() < 1e-5 && (got_c[r] - wc).abs() < 1e-5, "row {r}: acceptance {} cost {} against {wa} {wc}", got_a[r], got_c[r]);
    }
    let (omega, dvals) = (rng.vec(rows, 1.0).iter().map(|x| x.abs()).collect::<Vec<f32>>(), rng.vec(rows, 1.0));
    let (tom, tdv) = (Tensor::upload(&gpu, &omega, 1, rows, "omega").expect("omega"), Tensor::upload(&gpu, &dvals, 1, rows, "dv").expect("dv"));
    dz.zero(&gpu).expect("zero");
    train::edr_grad(&mut gpu, tz.ptr(), v, tc.ptr(), pl, 4, tom.ptr(), tdv.ptr(), tw.ptr(), dz.ptr(), rows).expect("edr grad");
    let mut want = vec![0.0; rows * v];
    for r in 0..rows {
        if let Some(t) = &targets[r] {
            let mut g = vec![0.0; v];
            cpu::edr_grad(&z[r * v..(r + 1) * v], t, eos, omega[r], dvals[r], &mut g);
            for u in 0..v {
                want[r * v + u] = weights[r] * g[u];
            }
        }
    }
    assert_close("edr gradient", &dz.download(&gpu).expect("dz"), &want, 1e-5);
    // The block verification loss over depth positions of every row, the positions a stride apart
    let (depth, n) = (3usize, rows);
    let zb = rng.vec(depth * n * v, 2.0);
    let (packed_b, softs_b) = packed_targets(&mut rng, depth * n, k, n_vocab, None);
    let targets_b: Vec<Option<Vec<(usize, f32)>>> = softs_b.iter().map(|s| vocab.target(s)).collect();
    let truth: Vec<i32> = (0..depth * n).map(|i| if i % 4 == 3 { -1 } else { targets_b[i].as_ref().map_or(-1, |t| t[0].0 as i32) }).collect();
    let (tzb, tcb, ttr) = (Tensor::upload(&gpu, &zb, depth * n, v, "z").expect("z"), Tensor::upload(&gpu, &packed_b, depth * n, 2 * k + 1, "cands").expect("cands"), gpu.upload(&truth, kf_core::AllocKind::Compute, "truth").expect("truth"));
    let (dzb, lb) = (Tensor::zeros(&gpu, depth * n, v, "dz").expect("dz"), Tensor::zeros(&gpu, 1, n, "losses").expect("losses"));
    let ptr_truth = kf_backend_cuda::device_ptr(&ttr, &gpu);
    train::bv_row(&mut gpu, tzb.ptr(), v, n * v, tcb.ptr(), n * (2 * k + 1), ptr_truth, n, pl, depth, tw.ptr(), dzb.ptr(), lb.ptr(), n).expect("bv");
    let (got_dz, got_l) = (dzb.download(&gpu).expect("dz"), lb.download(&gpu).expect("losses"));
    for r in 0..n {
        let zs: Vec<&[f32]> = (0..depth).map(|s| &zb[(s * n + r) * v..(s * n + r + 1) * v]).collect();
        let tg: Vec<Option<Vec<(usize, f32)>>> = (0..depth).map(|s| targets_b[s * n + r].clone()).collect();
        let tr: Vec<Option<usize>> = (0..depth).map(|s| (truth[s * n + r] >= 0).then_some(truth[s * n + r] as usize)).collect();
        let mut scratch: Vec<Vec<f32>> = vec![vec![0.0; v]; depth];
        let mut dzs: Vec<&mut [f32]> = scratch.iter_mut().map(|x| x.as_mut_slice()).collect();
        let want_l = cpu::bv_row(&zs, &tg, &tr, weights[r], &mut dzs);
        assert!((got_l[r] - want_l).abs() < 1e-4, "row {r} block loss {} against {want_l}", got_l[r]);
        for s in 0..depth {
            assert_close(&format!("block verification gradient row {r} position {s}"), &got_dz[(s * n + r) * v..(s * n + r + 1) * v], &scratch[s], 1e-4);
        }
    }
}

/// A tiny model whose two tap rows are the normed embeddings and the residual with half of them added back, the
/// head tied to the table, every row scored by a top k of DRAFT_CANDS
struct Tiny {
    w: Mem,
    table: kf_core::Wid,
    norm: kf_core::Wid,
    d: usize,
    vocab: usize,
}

impl Tiny {
    fn new(seed: u64, d: usize, vocab: usize) -> Self {
        let mut rng = Rng(seed);
        let mut w = Mem::default();
        let table = w.f32("table", &[d, vocab], &rng.vec(vocab * d, 1.0));
        let norm = w.f32("norm", &[d], &rng.vec(d, 0.3).iter().map(|v| 1.0 + v).collect::<Vec<f32>>());
        Self { w, table, norm, d, vocab }
    }

    fn graph(&self) -> kf_core::Graph {
        let mut g = kf_core::GraphBuilder::new();
        let tokens = g.input("tokens");
        let x = g.push(kf_core::Op::Embed { weight: self.table }, &[tokens]);
        let m = g.push(kf_core::Op::RmsNorm { weight: None, bias: None, eps: 1e-5 }, &[x]);
        g.keep_tap(m);
        let half = g.push(kf_core::Op::Scale(0.5), &[m]);
        let y = g.push(kf_core::Op::Add, &[x, half]);
        g.keep_hidden(y);
        g.keep_tap(y);
        let yn = g.push(kf_core::Op::RmsNorm { weight: Some(self.norm), bias: None, eps: 1e-5 }, &[y]);
        let l = g.push(kf_core::Op::MatMul { weight: self.table, bias: None, scale: None }, &[yn]);
        let top = g.push(kf_core::Op::TopK { k: kf_core::DRAFT_CANDS }, &[l]);
        g.finish(top)
    }

    fn plan(pool: kf_core::Pool) -> kf_core::Plan {
        kf_core::Plan { layers: vec![kf_core::LayerPlan { layer: 0, weights: pool, state: pool, expert_cache_slots: 0, kv_resident: 0 }], pinned: Vec::new(), weights_bytes: 0, state_bytes: 0, n_ctx: 1024, n_batch: 256, n_seq: 2 }
    }

    fn inputs(&self, seq: usize, start: usize, n: usize, salt: usize) -> kf_core::Inputs {
        let mut i = kf_core::Inputs::default();
        for t in 0..n {
            i.tokens.push(((seq * 7 + start + t) * 13 + salt) as u32 % self.vocab as u32);
            i.positions.push((start + t) as u32);
            i.seqs.push(seq as u32);
        }
        i
    }
}

/// The device snapshot's int8 products equal the host reference's over every matrix of a tiny block
#[test]
fn snapshot_products_match_reference() {
    let mut rng = Rng(0x5A9);
    let mut gpu = cuda();
    let cfg = kf_core::DraftConfig { d: 32, heads: 2, ff: 64, window: 16, taps: 2, depth: 3, fanout: [3, 2, 2, 1, 1, 1, 1, 1], learners: 1, objective: kf_core::Objective::Ce, keep_frac: 1.0, ema: 0.5, hedge: 1.0, eps: 1e-5, lr: 1e-3, weight_decay: 0.0, seed: 4 };
    let params = kf_backend_cpu::draft::init_params(&cfg);
    let snap = kf_backend_cpu::draft::Snapshot::of(&cfg, &params);
    for which in kf_backend_cpu::draft::MATRICES {
        let (_, k, _) = cfg.param_shapes()[which];
        for m in [1usize, 5, 9] {
            let x = rng.vec(m * k, 1.0);
            let want = snap.product(which, &x, m);
            let got = kf_backend_cuda::draft::probe_product(&mut gpu, &cfg, &params, which, &x, m).expect("product");
            assert_close(&format!("matrix {which} over {m} rows"), &got, &want, 1e-4);
        }
    }
}

/// Asserts two trees drafted below the same root agree, a row's token allowed to differ only where the host
/// block's own distribution at that row's position puts the two tokens within a thousandth of each other
fn assert_trees_agree(what: &str, host: &kf_core::Tree, device: &kf_core::Tree, tops: &[Vec<(u32, f32)>]) {
    assert_eq!(host.parents, device.parents, "{what}: tree shapes");
    assert_eq!(host.len(), device.len(), "{what}: tree sizes");
    let depth = |t: &kf_core::Tree, mut i: usize| {
        let mut n = 0;
        while t.parents[i] >= 0 {
            i = t.parents[i] as usize;
            n += 1;
        }
        n
    };
    for i in 0..host.len() {
        let (a, b) = (host.tokens[i], device.tokens[i]);
        if a == b {
            continue;
        }
        let d = depth(host, i);
        assert!(d >= 1, "{what}: roots differ, {a} against {b}");
        let dist = &tops[d - 1];
        let p = |t: u32| dist.iter().find(|(x, _)| *x == t).map_or(0.0, |(_, p)| *p);
        let (pa, pb) = (p(a), p(b));
        assert!((pa - pb).abs() <= 1e-3 * pa.max(pb).max(1e-6), "{what}: row {i} holds {a} on the host and {b} on the device at probabilities {pa} and {pb}");
    }
}

/// The host and device draft blocks fed the same passes keep the same ring, draft the same trees after every
/// pass, learn the same step, carry the same accepted path into their histories, and agree again after a tree
/// pass is compacted and the vocabulary recounted
#[test]
fn device_draft_block_matches_host_through_passes() {
    use kf_core::Backend;
    let model = Tiny::new(0x7EA, 32, 80);
    let cfg = kf_core::DraftConfig { d: model.d, heads: 2, ff: 64, window: 16, taps: 2, depth: 3, fanout: [3, 2, 2, 1, 1, 1, 1, 1], learners: 2, objective: kf_core::Objective::Ce, keep_frac: 0.6, ema: 0.5, hedge: 1.0, eps: 1e-5, lr: 1e-3, weight_decay: 0.01, seed: 11 };
    let mut cpu = CpuBackend::new();
    cpu.apply_plan(&Tiny::plan(kf_core::Pool::Host)).expect("cpu plan");
    let mut gpu = cuda();
    gpu.apply_plan(&Tiny::plan(kf_core::Pool::Device(0))).expect("cuda plan");
    for b in [&mut cpu as &mut dyn Backend, &mut gpu as &mut dyn Backend] {
        b.draft_start(cfg, &[0, 1], model.table, model.table, model.norm, model.vocab, Some(3), 40, 2, &model.w).expect("draft start");
    }
    let n = kf_core::DRAFT_CHUNK;
    let feed = |cpu: &mut CpuBackend, gpu: &mut kf_backend_cuda::CudaBackend, inp: &kf_core::Inputs, what: &str| {
        let g = model.graph();
        both(what, &g, inp, &model.w, cpu, gpu, 1e-3);
    };
    // Three chunks over slot zero fill the ring with one run longer than a training chunk, a tree drafted after each
    let mut root = 5u32;
    for c in 0..3 {
        feed(&mut cpu, &mut gpu, &model.inputs(0, c * n, n, 1), &format!("chunk {c}"));
        let tops = cpu.draft_block().expect("a block").tops(0, &[0], root).expect("tops");
        let (tc, tg) = (cpu.draft_tree(0, &[0], root, 6).expect("cpu tree"), gpu.draft_tree(0, &[0], root, 6).expect("cuda tree"));
        assert_trees_agree(&format!("trees after chunk {c}"), &tc, &tg, &tops);
        assert!(tc.len() > 1 && tc.len() <= 6, "the block drafts something below the root");
        root = tc.tokens[1];
    }
    assert_eq!(cpu.draft_rows(), gpu.draft_rows());
    assert_eq!(cpu.draft_rows(), 3 * n as u64);
    // The gradients of one batch agree before any update
    let gc = cpu.draft_block_mut().expect("a block").probe_grads().expect("cpu grads").expect("grads");
    let gg = gpu.draft_probe_grads().expect("cuda grads").expect("grads");
    let errs: Vec<(&str, f32)> = gc
        .iter()
        .zip(&gg)
        .enumerate()
        .map(|(i, (a, b))| {
            let num: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt();
            let den: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            (cfg.param_shapes()[i].0, if den > 0.0 { num / den } else { num })
        })
        .collect();
    assert!(errs.iter().all(|(_, e)| *e < 2e-3), "gradients of one batch differ: {errs:?}");
    let (sc, sg) = (cpu.draft_step().expect("cpu step").expect("a step"), gpu.draft_step().expect("cuda step").expect("a step"));
    assert!((sc.loss - sg.loss).abs() < 2e-3 * (1.0 + sc.loss.abs()), "loss {} against {}", sg.loss, sc.loss);
    assert_eq!(sc.hit, sg.hit, "hit rates");
    let (pc, pg) = (cpu.draft_save().expect("cpu save"), gpu.draft_save().expect("cuda save"));
    let (lc, vc, wc) = kf_core::draft_load(&cfg, &pc).expect("cpu params");
    let (lg, vg, wg) = kf_core::draft_load(&cfg, &pg).expect("cuda params");
    assert_eq!((lc, vc), (lg, vg), "tap layers and vocabularies");
    // The first step moves every parameter by the sign of its gradient, so the two sides may part only where a
    // gradient near zero flips sign under rounding: by at most two optimistic steps, on a sliver of the elements
    let most = 2.0 * 2.0 * cfg.learner_lr(1) * 1.05;
    for (i, (a, b)) in wc.iter().zip(&wg).enumerate() {
        let flips = a.iter().zip(b).filter(|(x, y)| (*x - *y).abs() > 1e-5).count();
        let worst = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        assert!(worst <= most, "parameter {i} after one step differs by {worst}, more than a flipped step of {most}");
        assert!(flips * 50 < a.len().max(50), "parameter {i} after one step differs on {flips} of {} elements", a.len());
    }
    // A tree pass of eight rows below the last root, then the path through rows 0 3 6 is kept and drafted from
    let tokens: Vec<u32> = vec![root, 5, 9, 14, 23, 31, 40, 47];
    let parents: Vec<i32> = vec![-1, 0, 0, 0, 1, 1, 3, 6];
    let base = 3 * n;
    let tree = tree_inputs(0, base, &tokens, &parents);
    feed(&mut cpu, &mut gpu, &tree, "tree pass");
    assert_eq!(cpu.draft_rows(), 3 * n as u64 + 8, "every tree row joined the ring");
    let keep = [0usize, 3, 6];
    cpu.compact(0, base, &keep).expect("cpu compact");
    gpu.compact(0, base, &keep).expect("cuda compact");
    let next = 17u32;
    let tops = cpu.draft_block().expect("a block").tops(0, &keep, next).expect("tops");
    let (tc, tg) = (cpu.draft_tree(0, &keep, next, 6).expect("cpu tree"), gpu.draft_tree(0, &keep, next, 6).expect("cuda tree"));
    assert_trees_agree("trees after the kept path", &tc, &tg, &tops);
    // One more row after the kept path, rooted at the drafted token, drafts the same on both
    let mut one = model.inputs(0, base + keep.len(), 1, 0);
    one.tokens[0] = next;
    feed(&mut cpu, &mut gpu, &one, "row after the path");
    let tops = cpu.draft_block().expect("a block").tops(0, &[0], 9).expect("tops");
    let (tc, tg) = (cpu.draft_tree(0, &[0], 9, 6).expect("cpu tree"), gpu.draft_tree(0, &[0], 9, 6).expect("cuda tree"));
    assert_trees_agree("trees after the path", &tc, &tg, &tops);
    cpu.draft_refresh_vocab().expect("cpu vocab");
    gpu.draft_refresh_vocab().expect("cuda vocab");
    let (sc, sg) = (cpu.draft_step().expect("cpu step").expect("a step"), gpu.draft_step().expect("cuda step").expect("a step"));
    assert!((sc.loss - sg.loss).abs() < 2e-3 * (1.0 + sc.loss.abs()), "loss {} against {} after the vocabulary recount", sg.loss, sc.loss);
}

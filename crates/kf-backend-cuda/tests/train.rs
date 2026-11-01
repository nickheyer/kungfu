mod common;

use common::*;
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
    let (pred, target) = (rng.vec(len, 1.0), rng.vec(len, 1.0));
    let (dpred, dtarget, grad, loss) = (Tensor::upload(&gpu, &pred, 1, len, "pred").expect("pred"), Tensor::upload(&gpu, &target, 1, len, "target").expect("target"), Tensor::zeros(&gpu, 1, len, "grad").expect("grad"), Tensor::zeros(&gpu, 1, 1, "loss").expect("loss"));
    train::smooth_l1(&mut gpu, dpred.ptr(), dtarget.ptr(), grad.ptr(), loss.ptr(), len, 1.0).expect("loss");
    let mut wgrad = vec![0.0; len];
    let wloss = cpu::smooth_l1(&pred, &target, &mut wgrad, 1.0);
    assert_close("smooth l1 gradient", &grad.download(&gpu).expect("grad"), &wgrad, 1e-5);
    assert_close("smooth l1 loss", &loss.download(&gpu).expect("loss"), &[wloss], 1e-4);
    let (xs, ys) = (rng.vec(len, 1.0), rng.vec(len, 1.0));
    let (dxs, dys) = (Tensor::upload(&gpu, &xs, 1, len, "x").expect("x"), Tensor::upload(&gpu, &ys, 1, len, "y").expect("y"));
    train::axpy(&mut gpu, dxs.ptr(), dys.ptr(), len, 0.3).expect("axpy");
    let want: Vec<f32> = xs.iter().zip(&ys).map(|(x, y)| y + 0.3 * x).collect();
    assert_close("axpy", &dys.download(&gpu).expect("y"), &want, 1e-6);
}

/// The head's attention forward and backward agree with the reference over a batch of chunks
#[test]
fn head_attention_matches_reference() {
    let mut rng = Rng(0x4a77);
    let mut gpu = cuda();
    let (batch, t, h, hd, window) = (3usize, train::ATTN_T, 4usize, 32usize, 24usize);
    let n = batch * t * h * hd;
    let (q, k, v, dout) = (rng.vec(n, 1.0), rng.vec(n, 1.0), rng.vec(n, 1.0), rng.vec(n, 1.0));
    let scale = 1.0 / (hd as f32).sqrt();
    let (dq_, dk_, dv_, ddo) = (Tensor::upload(&gpu, &q, batch * t, h * hd, "q").expect("q"), Tensor::upload(&gpu, &k, batch * t, h * hd, "k").expect("k"), Tensor::upload(&gpu, &v, batch * t, h * hd, "v").expect("v"), Tensor::upload(&gpu, &dout, batch * t, h * hd, "do").expect("do"));
    let (out, probs) = (Tensor::zeros(&gpu, batch * t, h * hd, "out").expect("out"), Tensor::zeros(&gpu, batch * h * t, t, "probs").expect("probs"));
    train::head_attn_fwd(&mut gpu, dq_.ptr(), dk_.ptr(), dv_.ptr(), out.ptr(), probs.ptr(), batch, h, hd, window, scale).expect("fwd");
    let mut want = vec![0.0; n];
    let wprobs = cpu::attn_fwd(&q, &k, &v, &mut want, batch, t, h, hd, window, scale);
    assert_close("head attention out", &out.download(&gpu).expect("out"), &want, 1e-4);
    assert_close("head attention probs", &probs.download(&gpu).expect("probs"), &wprobs, 1e-4);
    let (dq, dk, dv) = (Tensor::zeros(&gpu, batch * t, h * hd, "dq").expect("dq"), Tensor::zeros(&gpu, batch * t, h * hd, "dk").expect("dk"), Tensor::zeros(&gpu, batch * t, h * hd, "dv").expect("dv"));
    train::head_attn_bwd(&mut gpu, dq_.ptr(), dk_.ptr(), dv_.ptr(), probs.ptr(), ddo.ptr(), dq.ptr(), dk.ptr(), dv.ptr(), batch, h, hd, scale).expect("bwd");
    let (mut wdq, mut wdk, mut wdv) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    cpu::attn_bwd(&q, &k, &v, &wprobs, &dout, &mut wdq, &mut wdk, &mut wdv, batch, t, h, hd, scale);
    assert_close("head attention dq", &dq.download(&gpu).expect("dq"), &wdq, 1e-3);
    assert_close("head attention dk", &dk.download(&gpu).expect("dk"), &wdk, 1e-3);
    assert_close("head attention dv", &dv.download(&gpu).expect("dv"), &wdv, 1e-3);
}

/// One training step of the device head equals the host head's: prediction, loss, gradients, and the updated parameters
#[test]
fn device_head_step_matches_host_head() {
    let mut gpu = cuda();
    let cfg = kf_core::HeadConfig { d: 16, heads: 2, ff: 32, window: 24, eps: 1e-5, lr: 1e-3, weight_decay: 0.01, seed: 31 };
    let (batch, t) = (2usize, train::ATTN_T);
    let n = batch * t;
    let mut seed = 5u64;
    let f = kf_backend_cpu::drafter::init_values(n * cfg.d, 1.0, &mut seed);
    let e = kf_backend_cpu::drafter::init_values(n * cfg.d, 1.0, &mut seed);
    let y = kf_backend_cpu::drafter::init_values(n * cfg.d, 1.0, &mut seed);
    let mut host = kf_backend_cpu::drafter::CpuHead::new(cfg).expect("host head");
    let mut dev = kf_backend_cuda::drafter::Head::new(&gpu, cfg).expect("device head");
    let work = dev.work(&gpu, batch).expect("work");
    let (df, de, dy) = (Tensor::upload(&gpu, &f, n, cfg.d, "f").expect("f"), Tensor::upload(&gpu, &e, n, cfg.d, "e").expect("e"), Tensor::upload(&gpu, &y, n, cfg.d, "y").expect("y"));
    let trace = host.forward(&f, &e, batch, t);
    dev.forward(&mut gpu, &work, df.ptr(), de.ptr(), batch).expect("forward");
    assert_close("head prediction", &work.pred.download(&gpu).expect("pred"), &trace.pred, 1e-4);
    let want_loss = host.backward(&f, &e, &y, &trace, batch, t);
    let loss = dev.backward(&mut gpu, &work, df.ptr(), de.ptr(), dy.ptr(), batch).expect("backward");
    assert!((loss - want_loss).abs() < 1e-4, "loss {loss} against {want_loss}");
    for (i, (g, w)) in dev.grads_host(&gpu).expect("grads").iter().zip(&host.grads).enumerate() {
        assert_close(&format!("gradient of parameter {i}"), g, w, 2e-3);
    }
    host.step();
    dev.step(&mut gpu).expect("step");
    for (i, (p, w)) in dev.params_host(&gpu).expect("params").iter().zip(&host.params).enumerate() {
        assert_close(&format!("parameter {i} after the step"), p, w, 1e-4);
    }
}

/// The drafter learns from rows passes left in the ring: fed an embedding whose next row is a function of the next
/// token and a head table that names the token after, the loss falls and the hit rate rises to one across steps
#[test]
fn trainer_learns_from_the_ring() {
    use kf_core::{Backend, GraphBuilder, Op};
    let mut rng = Rng(0x5a5a);
    let mut w = Mem::default();
    let d = 96usize;
    let vals = rng.vec(32 * d, 1.0);
    let table = w.f32("table", &[d, 32], &vals);
    // The token sequence steps by seven, so a head table whose row b is the embedding of b minus seven scores the
    // predicted next embedding as the token after it
    let shifted: Vec<f32> = (0..32).flat_map(|b| vals[((b + 32 - 7) % 32) * d..((b + 32 - 7) % 32 + 1) * d].to_vec()).collect();
    let head = w.f32("head", &[d, 32], &shifted);
    let mut gpu = cuda();
    gpu.start_ring(d).expect("ring");
    let mut g = GraphBuilder::new();
    let tokens = g.input("tokens");
    let x = g.push(Op::Embed { weight: table }, &[tokens]);
    let y = g.push(Op::RmsNorm { weight: None, bias: None, eps: 1e-5 }, &[x]);
    g.keep_hidden(x);
    let graph = g.finish(y);
    // Three hundred rows of one slot in passes of twenty
    for c in 0..15 {
        let toks: Vec<u32> = (0..20).map(|i| ((c * 20 + i) * 7 % 32) as u32).collect();
        gpu.run(&graph, &inputs(&toks, (c * 20) as u32), &w).expect("pass");
    }
    let cfg = kf_core::HeadConfig { d, heads: 4, ff: 192, window: 64, eps: 1e-5, lr: 2e-3, weight_decay: 0.0, seed: 3 };
    let mut drafter = kf_backend_cuda::drafter::Drafter::new(&mut gpu, cfg, 4, 1, table, head, &w).expect("drafter");
    drafter.set_vocab(&mut gpu, &(0..32).collect::<Vec<u32>>()).expect("vocab");
    let first = drafter.step(&mut gpu).expect("step").expect("a run long enough");
    let mut last = first;
    for _ in 0..40 {
        last = drafter.step(&mut gpu).expect("step").expect("a run long enough");
    }
    eprintln!("trainer loss {} to {} and hit {} to {} over {} steps", first.loss, last.loss, first.hit, last.hit, drafter.trainer.steps);
    assert!(first.loss.is_finite() && last.loss.is_finite() && last.loss < first.loss * 0.5, "loss {} to {}", first.loss, last.loss);
    assert!((0.0..=1.0).contains(&first.hit) && last.hit >= 0.95, "hit {} to {}", first.hit, last.hit);
}

/// The drafter grows a well formed tree from a slot's held row, and its root prediction equals the host head attending
/// over the same cached pairs
#[test]
fn drafter_grows_a_tree_matching_the_host_head() {
    use kf_core::{Backend, GraphBuilder, Op};
    let mut rng = Rng(0xd4a4);
    let mut w = Mem::default();
    let (d, n_vocab) = (64usize, 64usize);
    let q8 = kf_quant::by_name("q8_0").expect("q8_0").info().id;
    let table = w.add("table", q8, &[d, n_vocab], synth(&mut rng, q8, n_vocab, d));
    let mut gpu = cuda();
    gpu.start_ring(d).expect("ring");
    let mut g = GraphBuilder::new();
    let tokens = g.input("tokens");
    let x = g.push(Op::Embed { weight: table }, &[tokens]);
    let y = g.push(Op::RmsNorm { weight: None, bias: None, eps: 1e-5 }, &[x]);
    g.keep_hidden(x);
    let graph = g.finish(y);
    let seq_tokens: Vec<u32> = (0..100).map(|i| (i * 11 % n_vocab) as u32).collect();
    for c in 0..5 {
        gpu.run(&graph, &inputs(&seq_tokens[c * 20..(c + 1) * 20], (c * 20) as u32), &w).expect("pass");
    }
    let cfg = kf_core::HeadConfig { d, heads: 4, ff: 128, window: 64, eps: 1e-5, lr: 1e-3, weight_decay: 0.0, seed: 11 };
    let mut drafter = kf_backend_cuda::drafter::Drafter::new(&mut gpu, cfg, 2, 2, table, table, &w).expect("drafter");
    for _ in 0..5 {
        drafter.trainer.step(&mut gpu).expect("step").expect("a run");
    }
    let vocab: Vec<u32> = (0..n_vocab as u32).collect();
    drafter.set_vocab(&mut gpu, &vocab).expect("vocab");
    drafter.sync(&mut gpu, 0).expect("sync");
    // Rows 0 to 98 pair with their successors, the window keeps the last 63 pairs, row 99 is held for the draft
    assert_eq!(drafter.cached(0), 63);
    let hidden = gpu.hidden_rows().expect("hidden");
    let last_row = &hidden[19 * d..20 * d];
    let hidden_ptr = {
        // The held row comes from the device, so upload the last row again as its source
        Tensor::upload(&gpu, last_row, 1, d, "held source").expect("held")
    };
    drafter.hold(&gpu, 0, hidden_ptr.ptr(), 99).expect("hold");
    let own = 7u32;
    let tree = drafter.draft(&mut gpu, 0, own, 2).expect("draft of the root and one child");
    let gathered = drafter.vocab_bytes(&gpu).expect("vocab bytes");
    let native = kf_core::WeightSource::bytes(&w, table).expect("table bytes");
    assert_eq!(gathered.len(), native.len());
    assert_eq!(&gathered[..200], &native[..200], "gathered head rows against the table");
    let logits = drafter.last_logits(&gpu, 1).expect("logits");
    let pred = drafter.root_prediction(&gpu).expect("prediction");
    let embd_rows = kf_backend_cpu::dequant_all(table, &w).expect("table");
    let want: Vec<f32> = (0..n_vocab).map(|t| (0..d).map(|c| pred[c] * embd_rows[t * d + c]).sum()).collect();
    eprintln!("device logits {:?}", &logits[..6]);
    eprintln!("host logits   {:?}", &want[..6]);
    assert_close("root draft logits against the head rows", &logits, &want, 5e-2);
    assert_eq!(tree.len(), 2);
    let tree = drafter.draft(&mut gpu, 0, own, 12).expect("draft");
    assert_eq!(tree.tokens[0], own);
    assert!(tree.len() > 1 && tree.len() <= 12, "tree of {} rows", tree.len());
    for i in 1..tree.len() {
        assert!(tree.parents[i] >= 0 && (tree.parents[i] as usize) < i, "row {i} parent {}", tree.parents[i]);
        assert!((tree.tokens[i] as usize) < n_vocab);
        assert_eq!(tree.child(tree.parents[i] as usize, tree.tokens[i]), Some(i), "row {i} is its parent's only child with its token");
    }
    let root_children = (1..tree.len()).filter(|i| tree.parents[*i] == 0).count();
    assert!(root_children >= 2 && root_children <= 4, "{root_children} children of the root");
    // The host head over the last 63 cached pairs then the held row with own, 64 rows in its window, predicts the same root row
    let params = drafter.trainer.head.params_host(&gpu).expect("params");
    let host = kf_backend_cpu::drafter::CpuHead::with_params(cfg, params).expect("host head");
    let embd = kf_backend_cpu::dequant_all(table, &w).expect("embedding");
    let ring_rows: Vec<f32> = {
        let idx: Vec<u32> = (0..100).map(|p| {
            let r = gpu.ring.as_ref().expect("ring");
            (0..r.cap).find(|i| r.entry(*i).live && r.entry(*i).pos == p && r.entry(*i).seq == 0).expect("ring row") as u32
        }).collect();
        kf_backend_cuda::features::ring_rows(&mut gpu, &idx).expect("rows")
    };
    let t = 64;
    let mut f = Vec::with_capacity(t * d);
    let mut e = Vec::with_capacity(t * d);
    for p in 36..100 {
        f.extend_from_slice(&ring_rows[p * d..(p + 1) * d]);
        let next = if p + 1 < 100 { seq_tokens[p + 1] } else { own } as usize;
        e.extend_from_slice(&embd[next * d..(next + 1) * d]);
    }
    let trace = host.forward(&f, &e, 1, t);
    let want = &trace.pred[(t - 1) * d..t * d];
    let got = drafter.root_prediction(&gpu).expect("prediction");
    assert_close("root prediction against the host head", &got, want, 2e-3);
}

/// The host drafter and the device drafter grow the same tree from the same rows and the same head
#[test]
fn host_and_device_drafters_agree() {
    use kf_core::{Backend, GraphBuilder, Op};
    let mut rng = Rng(0xbeef);
    let mut w = Mem::default();
    let (d, n_vocab) = (64usize, 48usize);
    let q8 = kf_quant::by_name("q8_0").expect("q8_0").info().id;
    let table = w.add("table", q8, &[d, n_vocab], synth(&mut rng, q8, n_vocab, d));
    let mut g = GraphBuilder::new();
    let tokens = g.input("tokens");
    let x = g.push(Op::Embed { weight: table }, &[tokens]);
    let y = g.push(Op::RmsNorm { weight: None, bias: None, eps: 1e-5 }, &[x]);
    g.keep_hidden(x);
    let graph = g.finish(y);
    let cfg = kf_core::HeadConfig { d, heads: 4, ff: 96, window: 16, eps: 1e-5, lr: 1e-3, weight_decay: 0.0, seed: 5 };
    let seq_tokens: Vec<u32> = (0..90).map(|i| (i * 7 % n_vocab) as u32).collect();
    let mut cpu = kf_backend_cpu::CpuBackend::new();
    let mut gpu = cuda();
    cpu.drafter_start(cfg, table, table, n_vocab, n_vocab, 2, &w).expect("cpu drafter");
    gpu.drafter_start(cfg, table, table, n_vocab, n_vocab, 2, &w).expect("cuda drafter");
    for c in 0..6 {
        let inp = inputs(&seq_tokens[c * 15..(c + 1) * 15], (c * 15) as u32);
        cpu.run(&graph, &inp, &w).expect("cpu pass");
        gpu.run(&graph, &inp, &w).expect("cuda pass");
    }
    assert_eq!((cpu.drafter_rows(), gpu.drafter_rows()), (90, 90));
    // The same training steps on both, from the same initial head and the same rows
    for _ in 0..3 {
        let a = cpu.drafter_step().expect("cpu step").expect("rows");
        let b = gpu.drafter_step().expect("cuda step").expect("rows");
        assert!((a.loss - b.loss).abs() < 1e-3 * (1.0 + a.loss.abs()), "losses {} and {}", a.loss, b.loss);
        assert!((a.hit - b.hit).abs() <= 0.05 && (0.0..=1.0).contains(&a.hit), "hit rates {} and {}", a.hit, b.hit);
    }
    cpu.drafter_refresh_vocab().expect("vocab");
    gpu.drafter_refresh_vocab().expect("vocab");
    cpu.drafter_sync(0).expect("sync");
    gpu.drafter_sync(0).expect("sync");
    cpu.drafter_hold(0, 14, 89).expect("hold");
    gpu.drafter_hold(0, 14, 89).expect("hold");
    let own = 3u32;
    let a = cpu.drafter_draft(0, own, 10).expect("cpu draft");
    let b = gpu.drafter_draft(0, own, 10).expect("cuda draft");
    assert_eq!(a, b, "trees differ");
    assert!(a.len() >= 4, "a tree of {} rows", a.len());
    let saved = gpu.drafter_save().expect("save");
    cpu.drafter_load(&saved).expect("load");
    assert_eq!(cpu.drafter_draft(0, own, 10).expect("cpu draft after load"), b);
    // A rollback to row 40 starts a new epoch, the slot continues from there with an unrelated run of tokens, and the
    // windows of both follow the new run from position 40 on
    cpu.truncate(0, 40).expect("truncate");
    gpu.truncate(0, 40).expect("truncate");
    let more: Vec<u32> = (0..45).map(|i| (i * 5 + 2) as u32 % n_vocab as u32).collect();
    for c in 0..3 {
        let inp = inputs(&more[c * 15..(c + 1) * 15], (40 + c * 15) as u32);
        cpu.run(&graph, &inp, &w).expect("cpu pass");
        gpu.run(&graph, &inp, &w).expect("cuda pass");
    }
    cpu.drafter_sync(0).expect("sync");
    gpu.drafter_sync(0).expect("sync");
    cpu.drafter_hold(0, 14, 84).expect("hold");
    gpu.drafter_hold(0, 14, 84).expect("hold");
    let a = cpu.drafter_draft(0, own, 10).expect("cpu draft");
    let b = gpu.drafter_draft(0, own, 10).expect("cuda draft");
    assert_eq!(a, b, "trees differ after the rollback");
    assert!(a.len() >= 4, "a tree of {} rows after the rollback", a.len());
    // A held row the window does not run up to is drafted from without the window on both
    cpu.drafter_hold(0, 10, 80).expect("hold");
    gpu.drafter_hold(0, 10, 80).expect("hold");
    assert_eq!(cpu.drafter_draft(0, own, 10).expect("cpu draft"), gpu.drafter_draft(0, own, 10).expect("cuda draft"), "trees differ off the window");
}

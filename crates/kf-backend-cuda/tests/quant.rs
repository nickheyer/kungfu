mod common;

use common::*;
use kf_backend_cpu::CpuBackend;
use kf_core::{Backend, GraphBuilder, Op, Wid};

#[test]
fn every_kf_quant_type_has_kernels() {
    let gpu = cuda();
    for q in kf_quant::types::all() {
        assert!(gpu.kernels.has_quant(q.info().id), "{} has no kernel module", q.info().name);
    }
}

#[test]
fn dequant_matches_reference_for_every_type() {
    let mut gpu = cuda();
    let mut rng = Rng(0x9e3779b97f4a7c15);
    let mut w = Mem::default();
    let (rows, n) = (6, 512);
    let ids: Vec<(String, Wid)> = kf_quant::types::all().iter().map(|q| (q.info().name.to_string(), w.add(q.info().name, q.info().id, &[n, rows], synth(&mut rng, q.info().id, rows, n)))).collect();
    for (name, id) in ids {
        let got = gpu.dequant(id, &w).unwrap_or_else(|e| panic!("{name}: {e:#}"));
        let want = kf_backend_cpu::dequant_all(id, &w).expect("reference dequant");
        assert_close(&format!("dequant {name}"), &got, &want, 1e-5);
    }
}

/// An f32 embedding table of sixteen random rows that turns token ids into activations
fn embed_table(w: &mut Mem, rng: &mut Rng, n: usize) -> Wid {
    let v = rng.vec(16 * n, 1.0);
    w.f32("embed", &[n, 16], &v)
}

/// Product of embedded rows with one weight, matrix vector for one or two rows and tiled above
fn check_matmul(what: &str, w: &Mem, table: Wid, id: Wid, bias: Option<Wid>, rows: usize, cpu: &mut CpuBackend, gpu: &mut kf_backend_cuda::CudaBackend, tol: f32) {
    let mut g = GraphBuilder::new();
    let tokens = g.input("tokens");
    let x = g.push(Op::Embed { weight: table }, &[tokens]);
    let y = g.push(Op::MatMul { weight: id, bias, scale: None }, &[x]);
    let graph = g.finish(y);
    let toks: Vec<u32> = (0..rows as u32).map(|r| (r * 7 + 3) % 16).collect();
    both(&format!("{what} rows {rows} bias {}", bias.is_some()), &graph, &inputs(&toks, 0), w, cpu, gpu, tol);
}

#[test]
fn matvec_and_matmul_match_reference_for_every_type() {
    let mut gpu = cuda();
    let mut cpu = CpuBackend::new();
    let mut rng = Rng(0x1234_5678_9abc_def1);
    let (n_in, n_out) = (768, 200);
    let mut w = Mem::default();
    let table = embed_table(&mut w, &mut rng, n_in);
    let bias = w.f32("bias", &[n_out], &rng.vec(n_out, 0.5));
    let ids: Vec<(String, Wid)> = kf_quant::types::all().iter().map(|q| (q.info().name.to_string(), w.add(q.info().name, q.info().id, &[n_in, n_out], synth(&mut rng, q.info().id, n_out, n_in)))).collect();
    for (name, id) in ids {
        for (rows, with_bias) in [(1, false), (1, true), (2, false), (5, true), (8, false), (9, true), (13, true), (16, false), (23, true), (32, false), (33, true), (40, false)] {
            check_matmul(&format!("matmul {name}"), &w, table, id, with_bias.then_some(bias), rows, &mut cpu, &mut gpu, 1e-5);
        }
    }
}

/// The int8 tensor core tile over every block type, row counts and widths that leave partial 64 by 64 tiles and partial 64 wide k steps
#[test]
fn int8_tensor_core_matmul_matches_reference_for_every_block_type() {
    let mut gpu = cuda();
    if gpu.kernels.arch() < 80 {
        eprintln!("skipping, the int8 tensor core path needs sm_80");
        return;
    }
    let mut cpu = CpuBackend::new();
    let mut rng = Rng(0xfeed_face_cafe_beef);
    let mut w = Mem::default();
    for (n_in, n_out) in [(1024, 200), (768, 64), (256, 130)] {
        let table = embed_table(&mut w, &mut rng, n_in);
        let bias = w.f32("bias", &[n_out], &rng.vec(n_out, 0.5));
        let ids: Vec<(String, Wid)> = kf_backend_cuda::sources::INT8_TYPES
            .iter()
            .map(|(name, _, _)| {
                let q = kf_quant::by_name(name).expect("type");
                assert!(n_in % q.info().block == 0, "{name} block {} does not divide {n_in}", q.info().block);
                (name.to_string(), w.add(name, q.info().id, &[n_in, n_out], synth(&mut rng, q.info().id, n_out, n_in)))
            })
            .collect();
        for (name, id) in ids {
            for (rows, with_bias) in [(9, false), (64, true), (65, false), (130, true)] {
                check_matmul(&format!("int8 {name} n_in {n_in}"), &w, table, id, with_bias.then_some(bias), rows, &mut cpu, &mut gpu, 1e-5);
            }
        }
    }
}

#[test]
fn float_weights_with_ragged_rows_match_reference() {
    let mut gpu = cuda();
    let mut cpu = CpuBackend::new();
    let mut rng = Rng(77);
    let n_in = 37;
    let mut w = Mem::default();
    let table = embed_table(&mut w, &mut rng, n_in);
    let ids: Vec<(String, Wid)> = ["f32", "f16", "bf16"]
        .iter()
        .map(|name| {
            let q = kf_quant::by_name(name).expect("type");
            (name.to_string(), w.add(name, q.info().id, &[n_in, 29], synth(&mut rng, q.info().id, 29, n_in)))
        })
        .collect();
    for (name, id) in ids {
        for rows in [1, 5] {
            check_matmul(&format!("ragged {name}"), &w, table, id, None, rows, &mut cpu, &mut gpu, 1e-5);
        }
    }
}

#[test]
fn real_tensors_match_reference() {
    let mut gpu = cuda();
    let mut cpu = CpuBackend::new();
    let mut rng = Rng(5);
    let qwen = gguf(QWEN);
    let llama = gguf(LLAMA);
    let mut w = Mem::default();
    let t1024 = embed_table(&mut w, &mut rng, 1024);
    let t2048 = embed_table(&mut w, &mut rng, 2048);
    let t3072 = embed_table(&mut w, &mut rng, 3072);
    let t8192 = embed_table(&mut w, &mut rng, 8192);
    let cases = [
        (w.gguf(&qwen, "blk.0.attn_q.weight", None), t1024, "q8_0 attn_q"),
        (w.gguf(&qwen, "blk.3.ffn_down.weight", None), t3072, "q8_0 ffn_down"),
        (w.gguf(&llama, "blk.0.attn_q.weight", None), t2048, "q4_K attn_q"),
        (w.gguf(&llama, "blk.0.ffn_up.weight", None), t2048, "q4_K ffn_up"),
        (w.gguf(&llama, "blk.0.attn_v.weight", None), t2048, "q6_K attn_v"),
        (w.gguf(&llama, "blk.0.ffn_down.weight", None), t8192, "q6_K ffn_down"),
    ];
    let norms = [w.gguf(&qwen, "blk.0.attn_norm.weight", None), w.gguf(&llama, "output_norm.weight", None)];
    for (id, table, what) in cases {
        let got = gpu.dequant(id, &w).expect("dequant");
        let want = kf_backend_cpu::dequant_all(id, &w).expect("reference dequant");
        assert_close(&format!("dequant {what}"), &got, &want, 1e-5);
        for rows in [1, 9] {
            check_matmul(what, &w, table, id, None, rows, &mut cpu, &mut gpu, 1e-5);
        }
    }
    for id in norms {
        let got = gpu.dequant(id, &w).expect("dequant");
        assert_close("f32 norm", &got, &kf_backend_cpu::dequant_all(id, &w).expect("reference"), 1e-5);
    }
}

/// Several products of one input launch as one group, over both the tensor core tile and the matvec, for every
/// block type, and each member's output matches the reference
#[test]
fn grouped_products_match_reference() {
    let mut gpu = cuda();
    if gpu.kernels.arch() < 80 {
        eprintln!("skipping, the int8 path needs sm_80");
        return;
    }
    let mut cpu = CpuBackend::new();
    let mut rng = Rng(0x6c0_0b5e);
    let n_in = 1024;
    let mut w = Mem::default();
    let table = embed_table(&mut w, &mut rng, n_in);
    let bias = w.f32("bias", &[256], &rng.vec(256, 0.5));
    let outs = [512usize, 256, 1024];
    for (name, _, _) in kf_backend_cuda::sources::INT8_TYPES {
        let q = kf_quant::by_name(name).expect("type");
        let ids: Vec<Wid> = outs.iter().enumerate().map(|(i, &n_out)| w.add(&format!("{name}{i}"), q.info().id, &[n_in, n_out], synth(&mut rng, q.info().id, n_out, n_in))).collect();
        for rows in [70usize, 3] {
            let mut g = GraphBuilder::new();
            let tokens = g.input("tokens");
            let x = g.push(Op::Embed { weight: table }, &[tokens]);
            let y0 = g.push(Op::MatMul { weight: ids[0], bias: None, scale: None }, &[x]);
            let y1 = g.push(Op::MatMul { weight: ids[1], bias: Some(bias), scale: None }, &[x]);
            let y2 = g.push(Op::MatMul { weight: ids[2], bias: None, scale: None }, &[x]);
            let c = g.concat(y0, y1);
            let all = g.concat(c, y2);
            let graph = g.finish(all);
            let toks: Vec<u32> = (0..rows as u32).map(|r| (r * 7 + 3) % 16).collect();
            both(&format!("group {name} rows {rows}"), &graph, &inputs(&toks, 0), &w, &mut cpu, &mut gpu, 1e-5);
        }
    }
}

/// Error of each member of a grouped product for one offset type, for finding which segment goes wrong
#[test]
#[ignore]
fn grouped_member_errors() {
    let mut gpu = cuda();
    let mut cpu = CpuBackend::new();
    let mut rng = Rng(0x6c0_0b5e);
    let n_in = 2048;
    let q = kf_quant::by_name("q4_K").expect("type");
    let mut w = Mem::default();
    let table = embed_table(&mut w, &mut rng, n_in);
    let outs = [512usize, 512, 2048];
    let ids: Vec<Wid> = outs.iter().enumerate().map(|(i, &n_out)| w.add(&format!("w{i}"), q.info().id, &[n_in, n_out], synth(&mut rng, q.info().id, n_out, n_in))).collect();
    for rows in [70usize, 3] {
        let mut g = GraphBuilder::new();
        let tokens = g.input("tokens");
        let x = g.push(Op::Embed { weight: table }, &[tokens]);
        let y0 = g.push(Op::MatMul { weight: ids[0], bias: None, scale: None }, &[x]);
        let y1 = g.push(Op::MatMul { weight: ids[1], bias: None, scale: None }, &[x]);
        let y2 = g.push(Op::MatMul { weight: ids[2], bias: None, scale: None }, &[x]);
        let c = g.concat(y0, y1);
        let all = g.concat(c, y2);
        let graph = g.finish(all);
        let toks: Vec<u32> = (0..rows as u32).map(|r| (r * 7 + 3) % 16).collect();
        let inp = inputs(&toks, 0);
        {
            let mut g2 = GraphBuilder::new();
            let tokens = g2.input("tokens");
            let x = g2.push(Op::Embed { weight: table }, &[tokens]);
            let _y0 = g2.push(Op::MatMul { weight: ids[0], bias: None, scale: None }, &[x]);
            let _y1 = g2.push(Op::MatMul { weight: ids[1], bias: None, scale: None }, &[x]);
            let y2 = g2.push(Op::MatMul { weight: ids[2], bias: None, scale: None }, &[x]);
            let only = g2.finish(y2);
            let want = cpu.run(&only, &inp, &w).expect("cpu");
            let got = gpu.run(&only, &inp, &w).expect("cuda");
            eprintln!("rows {rows} third member as the graph output: relative error {:.2e}", rel_err(&got, &want));
        }
        let want = cpu.run(&graph, &inp, &w).expect("cpu");
        let got = gpu.run(&graph, &inp, &w).expect("cuda");
        let total: usize = outs.iter().sum();
        let mut start = 0;
        for (i, &n) in outs.iter().enumerate() {
            let mut a = Vec::new();
            let mut b = Vec::new();
            for r in 0..rows {
                a.extend_from_slice(&got[r * total + start..r * total + start + n]);
                b.extend_from_slice(&want[r * total + start..r * total + start + n]);
            }
            eprintln!("rows {rows} member {i} ({n} outputs): relative error {:.2e}", rel_err(&a, &b));
            start += n;
        }
    }
}

mod common;

use common::*;
use kf_backend_cpu::CpuBackend;
use kf_core::{GraphBuilder, Op, Wid};

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
        for (rows, with_bias) in [(1, false), (1, true), (2, false), (5, true), (8, false), (9, true), (13, true), (40, false)] {
            check_matmul(&format!("matmul {name}"), &w, table, id, with_bias.then_some(bias), rows, &mut cpu, &mut gpu, 1e-5);
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

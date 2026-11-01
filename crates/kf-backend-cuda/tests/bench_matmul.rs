mod common;

use common::*;
use kf_core::{Backend, GraphBuilder, Op};
use std::time::Instant;

/// Throughput of one matmul at prefill shape, run with --ignored --nocapture, KF_BENCH_TYPES to pick types.
/// The output is cut to one row before download so the number is the kernels, not PCIe.
#[test]
#[ignore]
fn matmul_throughput() {
    let mut rng = Rng(0x1234_5678_9abc_def1);
    let shapes = std::env::var("KF_BENCH_SHAPES").unwrap_or_else(|_| "1024x2048x6144".into());
    let types = std::env::var("KF_BENCH_TYPES").unwrap_or_else(|_| "q8_0,q4_K,iq2_xxs".into());
    for shape in shapes.split(',') {
    let dims: Vec<usize> = shape.split('x').map(|v| v.parse().expect("shape")).collect();
    let (rows, n_in, n_out) = (dims[0], dims[1], dims[2]);
    for name in types.split(',') {
        // A fresh backend per case, so one case's prepared launches never stand in for the next
        let mut gpu = cuda();
        let q = kf_quant::by_name(name).expect("type");
        let mut w = Mem::default();
        let table = w.f32("embed", &[n_in, 16], &rng.vec(16 * n_in, 1.0));
        let id = w.add(name, q.info().id, &[n_in, n_out], synth(&mut rng, q.info().id, n_out, n_in));
        let mut g = GraphBuilder::new();
        let tokens = g.input("tokens");
        let x = g.push(Op::Embed { weight: table }, &[tokens]);
        let y = g.push(Op::MatMul { weight: id, bias: None, scale: None }, &[x]);
        let last = g.push(Op::LastRows { n: 1 }, &[y]);
        let graph = g.finish(last);
        let toks: Vec<u32> = (0..rows as u32).map(|r| (r * 7 + 3) % 16).collect();
        let inp = inputs(&toks, 0);
        gpu.run(&graph, &inp, &w).expect("warm");
        let iters = 20;
        let t = Instant::now();
        for _ in 0..iters {
            gpu.run(&graph, &inp, &w).expect("run");
        }
        let s = t.elapsed().as_secs_f64() / iters as f64;
        let flops = 2.0 * rows as f64 * n_in as f64 * n_out as f64;
        eprintln!("{name:8} {rows}x{n_in}x{n_out}: {:.1} us per step  {:.1} TOPS", s * 1e6, flops / s / 1e12);
    }
    }
}

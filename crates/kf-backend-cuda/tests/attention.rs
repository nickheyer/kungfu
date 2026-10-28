mod common;

use common::*;
use kf_backend_cpu::CpuBackend;
use kf_core::{Backend, Graph, GraphBuilder, Inputs, LayerPlan, Op, Plan, Pool, SwaType, Wid, Window};

const N_EMBD: usize = 64;
const N_HEAD: usize = 4;
const N_HEAD_KV: usize = 2;
const HD: usize = 16;
const N_CTX: usize = 512;
const N_BATCH: usize = 32;

struct Model {
    w: Mem,
    table: Wid,
    wq: Wid,
    wk: Wid,
    wv: Wid,
}

fn model(seed: u64) -> Model {
    let mut rng = Rng(seed);
    let mut w = Mem::default();
    let table = w.f32("table", &[N_EMBD, 64], &rng.vec(64 * N_EMBD, 1.0));
    let wq = w.f32("wq", &[N_EMBD, N_HEAD * HD], &rng.vec(N_EMBD * N_HEAD * HD, 0.25));
    let wk = w.f32("wk", &[N_EMBD, N_HEAD_KV * HD], &rng.vec(N_EMBD * N_HEAD_KV * HD, 0.25));
    let wv = w.f32("wv", &[N_EMBD, N_HEAD_KV * HD], &rng.vec(N_EMBD * N_HEAD_KV * HD, 0.25));
    Model { w, table, wq, wk, wv }
}

fn graph(m: &Model) -> Graph {
    let mut g = GraphBuilder::new();
    let tokens = g.input("tokens");
    let x = g.push(Op::Embed { weight: m.table }, &[tokens]);
    g.layer = Some(0);
    let q = g.push(Op::MatMul { weight: m.wq, bias: None, scale: None }, &[x]);
    let k = g.push(Op::MatMul { weight: m.wk, bias: None, scale: None }, &[x]);
    let v = g.push(Op::MatMul { weight: m.wv, bias: None, scale: None }, &[x]);
    let a = g.push(Op::Attention { layer: 0, n_head: N_HEAD, n_head_kv: N_HEAD_KV, head_dim: HD, head_dim_v: HD, scale: 0.25, window: None, softcap: 0.0, sinks: None, alibi_bias: 0.0, kv_source: None, top_k: None }, &[q, k, v]);
    g.finish(a)
}

fn plan(pool: Pool, resident: usize) -> Plan {
    Plan { layers: vec![LayerPlan { layer: 0, weights: pool, state: pool, expert_cache_slots: 0, kv_resident: resident }], pinned: Vec::new(), weights_bytes: 0, state_bytes: 0, n_ctx: N_CTX, n_batch: N_BATCH, n_seq: 2 }
}

fn run_inputs(seqs: &[(usize, usize, usize)]) -> Inputs {
    let mut i = Inputs::default();
    for (seq, start, n) in seqs {
        for t in 0..*n {
            i.tokens.push(((seq * 7 + start + t) * 13 % 61) as u32);
            i.positions.push((start + t) as u32);
            i.seqs.push(*seq as u32);
        }
    }
    i
}

/// Prefills slot zero past the window, slot one inside it, decodes both together, truncates, and refills
fn exercise(what: &str, resident: usize, tol: f32) -> (CpuBackend, kf_backend_cuda::CudaBackend, Model) {
    let m = model(0xA11CE);
    let g = graph(&m);
    let mut cpu = CpuBackend::new();
    cpu.apply_plan(&plan(Pool::Host, 0)).expect("cpu plan");
    let mut gpu = cuda();
    gpu.apply_plan(&plan(Pool::Device(0), resident)).expect("cuda plan");
    for c in 0..5 {
        both(&format!("{what} prefill chunk {c}"), &g, &run_inputs(&[(0, c * N_BATCH, N_BATCH)]), &m.w, &mut cpu, &mut gpu, tol);
    }
    for c in 0..2 {
        both(&format!("{what} slot one chunk {c}"), &g, &run_inputs(&[(1, c * N_BATCH, N_BATCH)]), &m.w, &mut cpu, &mut gpu, tol);
    }
    for s in 0..3 {
        both(&format!("{what} decode step {s}"), &g, &run_inputs(&[(0, 5 * N_BATCH + s, 1), (1, 2 * N_BATCH + s, 1)]), &m.w, &mut cpu, &mut gpu, tol);
    }
    cpu.truncate(0, 100).expect("cpu truncate");
    gpu.truncate(0, 100).expect("cuda truncate");
    both(&format!("{what} after truncate"), &g, &run_inputs(&[(0, 100, N_BATCH)]), &m.w, &mut cpu, &mut gpu, tol);
    both(&format!("{what} decode after truncate"), &g, &run_inputs(&[(1, 2 * N_BATCH + 3, 1), (0, 132, 1)]), &m.w, &mut cpu, &mut gpu, tol);
    (cpu, gpu, m)
}

#[test]
fn resident_cache_matches_reference_across_slots() {
    exercise("resident", 0, 2e-2);
}

#[test]
fn streamed_cache_matches_reference_across_slots() {
    let (mut cpu, mut gpu, m) = exercise("streamed", 64, 2e-2);
    let g = graph(&m);
    let saved = gpu.save_state(0).expect("save");
    let mut fresh = cuda();
    fresh.apply_plan(&plan(Pool::Device(0), 64)).expect("cuda plan");
    fresh.load_state(0, &saved).expect("load");
    let want = cpu.run(&g, &run_inputs(&[(0, 133, 1)]), &m.w).expect("cpu");
    let got = fresh.run(&g, &run_inputs(&[(0, 133, 1)]), &m.w).expect("cuda");
    assert_close("streamed after save and load", &got, &want, 2e-2);
    let got2 = gpu.run(&g, &run_inputs(&[(0, 133, 1)]), &m.w).expect("cuda original");
    assert_close("streamed original after save", &got2, &want, 2e-2);
}

/// Full attention, quick gelu, and grid pooling on the device match the reference, and rows that attend ahead inside a pass match too
#[test]
fn vision_ops_and_attend_ahead_match_reference() {
    let m = model(0xBEEF);
    let mut cpu = CpuBackend::new();
    cpu.apply_plan(&plan(Pool::Host, 0)).expect("cpu plan");
    let mut gpu = cuda();
    gpu.apply_plan(&plan(Pool::Device(0), 0)).expect("cuda plan");
    let mut g = GraphBuilder::new();
    let tokens = g.input("tokens");
    let x = g.push(Op::Embed { weight: m.table }, &[tokens]);
    let q = g.push(Op::MatMul { weight: m.wq, bias: None, scale: None }, &[x]);
    let k = g.push(Op::MatMul { weight: m.wq, bias: None, scale: None }, &[x]);
    let v = g.push(Op::MatMul { weight: m.wq, bias: None, scale: None }, &[x]);
    let a = g.push(Op::AttentionFull { n_head: N_HEAD, head_dim: HD, scale: 0.25 }, &[q, k, v]);
    let gq = g.push(Op::GeluQuick, &[a]);
    let pooled = g.push(Op::PoolGrid { side: 8, kernel: 2 }, &[gq]);
    let graph = g.finish(pooled);
    both("full attention, quick gelu, grid pool", &graph, &run_inputs(&[(0, 0, 64)]), &m.w, &mut cpu, &mut gpu, 2e-3);
    let causal = graph_attend(&m);
    let mut inp = run_inputs(&[(0, 0, 32)]);
    inp.attend = (0..32u32).map(|r| if (8..24).contains(&r) { 23 } else { r }).collect();
    both("attend ahead inside an image span", &causal, &inp, &m.w, &mut cpu, &mut gpu, 2e-2);
    let mut next = run_inputs(&[(0, 32, 1)]);
    next.attend = vec![32];
    both("decode after the span", &causal, &next, &m.w, &mut cpu, &mut gpu, 2e-2);
}

fn graph_attend(m: &Model) -> Graph {
    graph(m)
}

const N_HEAD_W: usize = 8;
const N_HEAD_KV_W: usize = 4;
const HD_W: usize = 8;

struct Wide {
    w: Mem,
    table: Wid,
    wq: Wid,
    wk: Wid,
    wv: Wid,
    sinks: Wid,
}

fn wide(seed: u64) -> Wide {
    let mut rng = Rng(seed);
    let mut w = Mem::default();
    let table = w.f32("table", &[N_EMBD, 64], &rng.vec(64 * N_EMBD, 1.0));
    let wq = w.f32("wq", &[N_EMBD, N_HEAD_W * HD_W], &rng.vec(N_EMBD * N_HEAD_W * HD_W, 0.25));
    let wk = w.f32("wk", &[N_EMBD, N_HEAD_KV_W * HD_W], &rng.vec(N_EMBD * N_HEAD_KV_W * HD_W, 0.25));
    let wv = w.f32("wv", &[N_EMBD, N_HEAD_KV_W * HD_W], &rng.vec(N_EMBD * N_HEAD_KV_W * HD_W, 0.25));
    let sinks = w.f32("sinks", &[N_HEAD_W], &rng.vec(N_HEAD_W, 1.0));
    Wide { w, table, wq, wk, wv, sinks }
}

fn wide_graph(m: &Wide, window: Option<Window>, sinks: bool) -> Graph {
    let mut g = GraphBuilder::new();
    let tokens = g.input("tokens");
    let x = g.push(Op::Embed { weight: m.table }, &[tokens]);
    g.layer = Some(0);
    let q = g.push(Op::MatMul { weight: m.wq, bias: None, scale: None }, &[x]);
    let k = g.push(Op::MatMul { weight: m.wk, bias: None, scale: None }, &[x]);
    let v = g.push(Op::MatMul { weight: m.wv, bias: None, scale: None }, &[x]);
    let a = g.push(Op::Attention { layer: 0, n_head: N_HEAD_W, n_head_kv: N_HEAD_KV_W, head_dim: HD_W, head_dim_v: HD_W, scale: 0.35, window, softcap: 0.0, sinks: sinks.then_some(m.sinks), alibi_bias: 0.0, kv_source: None, top_k: None }, &[q, k, v]);
    g.finish(a)
}

/// Prefill passes with enough rows and heads run attention undivided, plain and with a window and sinks, and the divided decode steps after them agree too
#[test]
fn undivided_and_windowed_attention_match_reference() {
    let m = wide(0xC0FFEE);
    for (what, window, sinks) in [("plain", None, false), ("windowed with sinks", Some(Window { size: 64, kind: SwaType::Standard }), true)] {
        let g = wide_graph(&m, window, sinks);
        let mut cpu = CpuBackend::new();
        cpu.apply_plan(&plan(Pool::Host, 0)).expect("cpu plan");
        let mut gpu = cuda();
        gpu.apply_plan(&plan(Pool::Device(0), 0)).expect("cuda plan");
        for c in 0..3 {
            both(&format!("{what} chunk {c}"), &g, &run_inputs(&[(0, c * N_BATCH, N_BATCH)]), &m.w, &mut cpu, &mut gpu, 2e-2);
        }
        both(&format!("{what} slot one"), &g, &run_inputs(&[(1, 0, N_BATCH)]), &m.w, &mut cpu, &mut gpu, 2e-2);
        for s in 0..2 {
            both(&format!("{what} decode {s}"), &g, &run_inputs(&[(0, 3 * N_BATCH + s, 1), (1, N_BATCH + s, 1)]), &m.w, &mut cpu, &mut gpu, 2e-2);
        }
    }
}

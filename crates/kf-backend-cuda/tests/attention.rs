mod common;

use common::*;
use kf_backend_cpu::CpuBackend;
use kf_core::{Backend, Graph, GraphBuilder, Inputs, LayerPlan, Op, Plan, Pool, RopeKind, RopeParams, SwaType, Wid, Window};

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

/// Attention at the head sizes the tensor core kernel compiles for, over partial query tiles, two slots, a window with
/// sinks, a softcap, and rows that attend ahead, then decode steps over the same cache
#[test]
fn tensor_core_attention_matches_reference() {
    let mut gpu = cuda();
    if gpu.kernels.arch() < 80 {
        eprintln!("skipping, the tensor core attention needs sm_80");
        return;
    }
    for (hd, n_head, n_head_kv) in [(64usize, 4usize, 2usize), (128, 4, 2), (80, 2, 1), (256, 2, 2)] {
        let mut rng = Rng(0xF1A5 + hd as u64);
        let mut w = Mem::default();
        let n_embd = 64;
        let table = w.f32("table", &[n_embd, 64], &rng.vec(64 * n_embd, 1.0));
        let wq = w.f32("wq", &[n_embd, n_head * hd], &rng.vec(n_embd * n_head * hd, 0.25));
        let wk = w.f32("wk", &[n_embd, n_head_kv * hd], &rng.vec(n_embd * n_head_kv * hd, 0.25));
        let wv = w.f32("wv", &[n_embd, n_head_kv * hd], &rng.vec(n_embd * n_head_kv * hd, 0.25));
        let sinks = w.f32("sinks", &[n_head], &rng.vec(n_head, 1.0));
        for (what, window, with_sinks, softcap) in [("plain", None, false, 0.0f32), ("windowed with sinks", Some(Window { size: 200, kind: SwaType::Standard }), true, 0.0), ("softcap", None, false, 20.0)] {
            let mut g = GraphBuilder::new();
            let tokens = g.input("tokens");
            let x = g.push(Op::Embed { weight: table }, &[tokens]);
            g.layer = Some(0);
            let q = g.push(Op::MatMul { weight: wq, bias: None, scale: None }, &[x]);
            let k = g.push(Op::MatMul { weight: wk, bias: None, scale: None }, &[x]);
            let v = g.push(Op::MatMul { weight: wv, bias: None, scale: None }, &[x]);
            let a = g.push(Op::Attention { layer: 0, n_head, n_head_kv, head_dim: hd, head_dim_v: hd, scale: 1.0 / (hd as f32).sqrt(), window, softcap, sinks: with_sinks.then_some(sinks), alibi_bias: 0.0, kv_source: None, top_k: None }, &[q, k, v]);
            let graph = g.finish(a);
            let n_batch = 160;
            let plan = |pool: Pool| Plan { layers: vec![LayerPlan { layer: 0, weights: pool, state: pool, expert_cache_slots: 0, kv_resident: 0 }], pinned: Vec::new(), weights_bytes: 0, state_bytes: 0, n_ctx: 1024, n_batch, n_seq: 2 };
            let mut cpu = CpuBackend::new();
            cpu.apply_plan(&plan(Pool::Host)).expect("cpu plan");
            gpu.apply_plan(&plan(Pool::Device(0))).expect("cuda plan");
            let label = format!("hd {hd} {what}");
            both(&format!("{label} slot zero first chunk"), &graph, &run_inputs(&[(0, 0, 160)]), &w, &mut cpu, &mut gpu, 2e-2);
            both(&format!("{label} slot zero second chunk"), &graph, &run_inputs(&[(0, 160, 100)]), &w, &mut cpu, &mut gpu, 2e-2);
            both(&format!("{label} slot one"), &graph, &run_inputs(&[(1, 0, 70)]), &w, &mut cpu, &mut gpu, 2e-2);
            both(&format!("{label} both slots in one pass"), &graph, &run_inputs(&[(1, 70, 40), (0, 260, 90)]), &w, &mut cpu, &mut gpu, 2e-2);
            let mut ahead = run_inputs(&[(0, 350, 48)]);
            ahead.attend = (350..398).map(|p| if (360..380).contains(&p) { 379 } else { p as u32 }).collect();
            both(&format!("{label} rows attending ahead"), &graph, &ahead, &w, &mut cpu, &mut gpu, 2e-2);
            both(&format!("{label} decode"), &graph, &run_inputs(&[(0, 398, 1), (1, 110, 1)]), &w, &mut cpu, &mut gpu, 2e-2);
        }
    }
}

/// A tree pass gives every row what feeding its path alone would, on both backends, and keeping one path leaves the
/// slot as if that path alone had been fed
#[test]
fn tree_rows_match_their_paths_and_compaction_keeps_one() {
    let m = model(0x7EE);
    let g = graph(&m);
    let mut cpu = CpuBackend::new();
    cpu.apply_plan(&plan(Pool::Host, 0)).expect("cpu plan");
    let mut gpu = cuda();
    gpu.apply_plan(&plan(Pool::Device(0), 0)).expect("cuda plan");
    // Both slots hold the same 48 tokens of history, fed within the batch
    let history = run_inputs(&[(0, 0, 24), (0, 24, 24)]);
    let mut history1 = history.clone();
    history1.seqs = vec![1; 48];
    for (c, chunk) in [(0usize, 0usize), (1, 24)] {
        both(&format!("tree history slot zero chunk {c}"), &g, &history.slice(chunk..chunk + 24), &m.w, &mut cpu, &mut gpu, 2e-2);
        both(&format!("tree history slot one chunk {c}"), &g, &history1.slice(chunk..chunk + 24), &m.w, &mut cpu, &mut gpu, 2e-2);
    }
    //      0
    //    / | \
    //   1  2  3
    //  / \    |
    // 4   5   6
    //         |
    //         7
    let tokens: Vec<u32> = vec![5, 9, 14, 23, 31, 40, 47, 52];
    let parents: Vec<i32> = vec![-1, 0, 0, 0, 1, 1, 3, 6];
    let tree = both("tree pass", &g, &tree_inputs(0, 48, &tokens, &parents), &m.w, &mut cpu, &mut gpu, 2e-2);
    let cols = N_HEAD * HD;
    for leaf in [4usize, 5, 7] {
        let mut path = vec![leaf];
        while parents[*path.last().expect("path")] >= 0 {
            path.push(parents[*path.last().expect("path")] as usize);
        }
        path.reverse();
        let mut cpu1 = CpuBackend::new();
        cpu1.apply_plan(&plan(Pool::Host, 0)).expect("cpu plan");
        let mut gpu1 = cuda();
        gpu1.apply_plan(&plan(Pool::Device(0), 0)).expect("cuda plan");
        for chunk in [0usize, 24] {
            both("path history", &g, &history.slice(chunk..chunk + 24), &m.w, &mut cpu1, &mut gpu1, 2e-2);
        }
        let linear = Inputs { tokens: path.iter().map(|i| tokens[*i]).collect(), positions: (48..48 + path.len() as u32).collect(), seqs: vec![0; path.len()], ..Default::default() };
        let want = both(&format!("path to {leaf} fed alone"), &g, &linear, &m.w, &mut cpu1, &mut gpu1, 2e-2);
        for (k, i) in path.iter().enumerate() {
            assert_close(&format!("tree row {i} against its path"), &tree[i * cols..(i + 1) * cols], &want[k * cols..(k + 1) * cols], 2e-2);
        }
    }
    // Keep the path to row 7, then both slots continue from the same 52 rows and must agree
    let keep = [0usize, 3, 6, 7];
    cpu.compact(0, 48, &keep).expect("cpu compact");
    gpu.compact(0, 48, &keep).expect("cuda compact");
    let path1 = Inputs { tokens: keep.iter().map(|i| tokens[*i]).collect(), positions: (48..52).collect(), seqs: vec![1; 4], ..Default::default() };
    both("slot one fed the kept path", &g, &path1, &m.w, &mut cpu, &mut gpu, 2e-2);
    let step = Inputs { tokens: vec![33, 33], positions: vec![52, 52], seqs: vec![0, 1], ..Default::default() };
    let out = both("step after compaction", &g, &step, &m.w, &mut cpu, &mut gpu, 2e-2);
    assert_close("compacted slot against the path fed alone", &out[..cols], &out[cols..], 2e-2);
}

/// A chain shaped tree pass equals the plain pass of the same rows on each backend by itself
#[test]
fn chain_tree_equals_plain_rows_per_backend() {
    let m = model(0x7EE);
    let g = graph(&m);
    let history = run_inputs(&[(0, 0, 24)]);
    let tokens: Vec<u32> = vec![5, 9, 14, 23];
    let parents: Vec<i32> = vec![-1, 0, 1, 2];
    let plain = Inputs { tokens: tokens.clone(), positions: (24..28).collect(), seqs: vec![0; 4], ..Default::default() };
    let tree = tree_inputs(0, 24, &tokens, &parents);
    let cols = N_HEAD * HD;
    for backend in ["cpu", "cuda"] {
        let mut outs = Vec::new();
        for inputs in [&plain, &tree] {
            let out = if backend == "cpu" {
                let mut b = CpuBackend::new();
                b.apply_plan(&plan(Pool::Host, 0)).expect("plan");
                b.run(&g, &history, &m.w).expect("history");
                b.run(&g, inputs, &m.w).expect("pass")
            } else {
                let mut b = cuda();
                b.apply_plan(&plan(Pool::Device(0), 0)).expect("plan");
                b.run(&g, &history, &m.w).expect("history");
                b.run(&g, inputs, &m.w).expect("pass")
            };
            outs.push(out);
        }
        for r in 0..4 {
            let e = rel_err(&outs[1][r * cols..(r + 1) * cols], &outs[0][r * cols..(r + 1) * cols]);
            eprintln!("{backend} row {r}: chain tree against plain rows, relative error {e}");
        }
        assert_close(&format!("{backend} chain tree against plain rows"), &outs[1], &outs[0], 1e-3);
    }
}

/// The tree tests again at a head size the decode kernel serves, with rope ahead of the attention as a model has
#[test]
fn tree_rows_match_their_paths_through_the_decode_kernel() {
    const HD64: usize = 64;
    let mut rng = Rng(0xD3C0DE);
    let mut w = Mem::default();
    let table = w.f32("table", &[N_EMBD, 64], &rng.vec(64 * N_EMBD, 1.0));
    let wq = w.f32("wq", &[N_EMBD, N_HEAD * HD64], &rng.vec(N_EMBD * N_HEAD * HD64, 0.125));
    let wk = w.f32("wk", &[N_EMBD, N_HEAD_KV * HD64], &rng.vec(N_EMBD * N_HEAD_KV * HD64, 0.125));
    let wv = w.f32("wv", &[N_EMBD, N_HEAD_KV * HD64], &rng.vec(N_EMBD * N_HEAD_KV * HD64, 0.125));
    let rope = RopeParams { kind: RopeKind::Neox, dims: HD64, base: 10000.0, scale: 1.0, orig_ctx: 4096, attn_factor: 1.0, yarn: None, sections: [0; 4], factors: None };
    let mut g = GraphBuilder::new();
    let tokens = g.input("tokens");
    let positions = g.input("positions");
    let x = g.push(Op::Embed { weight: table }, &[tokens]);
    g.layer = Some(0);
    let q = g.push(Op::MatMul { weight: wq, bias: None, scale: None }, &[x]);
    let k = g.push(Op::MatMul { weight: wk, bias: None, scale: None }, &[x]);
    let v = g.push(Op::MatMul { weight: wv, bias: None, scale: None }, &[x]);
    let q = g.push(Op::Rope { params: rope.clone(), n_head: N_HEAD, head_dim: HD64, offset: 0, inverse: false }, &[q, positions]);
    let k = g.push(Op::Rope { params: rope, n_head: N_HEAD_KV, head_dim: HD64, offset: 0, inverse: false }, &[k, positions]);
    let a = g.push(Op::Attention { layer: 0, n_head: N_HEAD, n_head_kv: N_HEAD_KV, head_dim: HD64, head_dim_v: HD64, scale: 0.125, window: None, softcap: 0.0, sinks: None, alibi_bias: 0.0, kv_source: None, top_k: None }, &[q, k, v]);
    let g = g.finish(a);
    let cols = N_HEAD * HD64;
    let history = run_inputs(&[(0, 0, 24), (0, 24, 24)]);
    let tokens_t: Vec<u32> = vec![5, 9, 14, 23, 31, 40, 47, 52];
    let parents: Vec<i32> = vec![-1, 0, 0, 0, 1, 1, 3, 6];
    let fresh = |cpu: &mut CpuBackend, gpu: &mut kf_backend_cuda::CudaBackend| {
        for chunk in [0usize, 24] {
            both("decode tree history", &g, &history.slice(chunk..chunk + 24), &w, cpu, gpu, 2e-2);
        }
    };
    let mut cpu = CpuBackend::new();
    cpu.apply_plan(&plan(Pool::Host, 0)).expect("cpu plan");
    let mut gpu = cuda();
    gpu.apply_plan(&plan(Pool::Device(0), 0)).expect("cuda plan");
    fresh(&mut cpu, &mut gpu);
    let tree = both("decode tree pass", &g, &tree_inputs(0, 48, &tokens_t, &parents), &w, &mut cpu, &mut gpu, 2e-2);
    for leaf in [4usize, 5, 7] {
        let mut path = vec![leaf];
        while parents[*path.last().expect("path")] >= 0 {
            path.push(parents[*path.last().expect("path")] as usize);
        }
        path.reverse();
        let mut cpu1 = CpuBackend::new();
        cpu1.apply_plan(&plan(Pool::Host, 0)).expect("cpu plan");
        let mut gpu1 = cuda();
        gpu1.apply_plan(&plan(Pool::Device(0), 0)).expect("cuda plan");
        fresh(&mut cpu1, &mut gpu1);
        let linear = Inputs { tokens: path.iter().map(|i| tokens_t[*i]).collect(), positions: (48..48 + path.len() as u32).collect(), seqs: vec![0; path.len()], ..Default::default() };
        let want = both(&format!("decode path to {leaf} fed alone"), &g, &linear, &w, &mut cpu1, &mut gpu1, 2e-2);
        for (k, i) in path.iter().enumerate() {
            assert_close(&format!("decode tree row {i} against its path"), &tree[i * cols..(i + 1) * cols], &want[k * cols..(k + 1) * cols], 2e-2);
        }
    }
    let keep = [0usize, 3, 6, 7];
    cpu.compact(0, 48, &keep).expect("cpu compact");
    gpu.compact(0, 48, &keep).expect("cuda compact");
    let mut cpu1 = CpuBackend::new();
    cpu1.apply_plan(&plan(Pool::Host, 0)).expect("cpu plan");
    let mut gpu1 = cuda();
    gpu1.apply_plan(&plan(Pool::Device(0), 0)).expect("cuda plan");
    fresh(&mut cpu1, &mut gpu1);
    let path1 = Inputs { tokens: keep.iter().map(|i| tokens_t[*i]).collect(), positions: (48..52).collect(), seqs: vec![0; 4], ..Default::default() };
    both("decode kept path fed alone", &g, &path1, &w, &mut cpu1, &mut gpu1, 2e-2);
    let step = Inputs { tokens: vec![33], positions: vec![52], seqs: vec![0], ..Default::default() };
    let after = both("decode step after compaction", &g, &step, &w, &mut cpu, &mut gpu, 2e-2);
    let alone = both("decode step after the path alone", &g, &step, &w, &mut cpu1, &mut gpu1, 2e-2);
    assert_close("decode compacted slot against the path fed alone", &after, &alone, 2e-2);
}

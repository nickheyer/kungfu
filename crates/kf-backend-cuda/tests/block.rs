mod common;

use common::*;
use kf_backend_cpu::CpuBackend;
use kf_core::{Backend, Graph, GraphBuilder, Inputs, LayerPlan, Op, Plan, Pool, RopeKind, RopeParams, Tid, Wid};

const N_CTX: usize = 1024;
const N_BATCH: usize = 256;

/// The first block of a real model, its token table cut to the first rows so small token ids index it
struct Block {
    w: Mem,
    table: Wid,
    attn_norm: Wid,
    wq: Wid,
    wk: Wid,
    wv: Wid,
    wo: Wid,
    q_norm: Option<Wid>,
    k_norm: Option<Wid>,
    ffn_norm: Wid,
    gate: Wid,
    up: Wid,
    down: Wid,
    n_head: usize,
    n_head_kv: usize,
    hd: usize,
    rope: RopeKind,
}

fn block(file: &str, n_head: usize, n_head_kv: usize, hd: usize, qk_norm: bool, rope: RopeKind) -> Block {
    let g = gguf(file);
    let mut w = Mem::default();
    let table = w.gguf(&g, "token_embd.weight", Some(64));
    let t = |w: &mut Mem, name: &str| w.gguf(&g, &format!("blk.0.{name}.weight"), None);
    let attn_norm = t(&mut w, "attn_norm");
    let wq = t(&mut w, "attn_q");
    let wk = t(&mut w, "attn_k");
    let wv = t(&mut w, "attn_v");
    let wo = t(&mut w, "attn_output");
    let q_norm = qk_norm.then(|| t(&mut w, "attn_q_norm"));
    let k_norm = qk_norm.then(|| t(&mut w, "attn_k_norm"));
    let ffn_norm = t(&mut w, "ffn_norm");
    let gate = t(&mut w, "ffn_gate");
    let up = t(&mut w, "ffn_up");
    let down = t(&mut w, "ffn_down");
    Block { w, table, attn_norm, wq, wk, wv, wo, q_norm, k_norm, ffn_norm, gate, up, down, n_head, n_head_kv, hd, rope }
}

fn head_norm(g: &mut GraphBuilder, x: Tid, w: Wid, n_tokens: usize, n_head: usize, hd: usize) -> Tid {
    let r = g.reshape(x, n_tokens * n_head, hd);
    let n = g.push(Op::RmsNorm { weight: Some(w), bias: None, eps: 1e-6 }, &[r]);
    g.reshape(n, n_tokens, n_head * hd)
}

/// Every tensor of the block in the order the layer crate emits them, the one named by stop as the output
fn graph(m: &Block, n_tokens: usize, stop: &str) -> Graph {
    let mut g = GraphBuilder::new();
    let tokens = g.input("tokens");
    let positions = g.input("positions");
    let x = g.push(Op::Embed { weight: m.table }, &[tokens]);
    g.layer = Some(0);
    let mm = |g: &mut GraphBuilder, w: Wid, x: Tid| g.push(Op::MatMul { weight: w, bias: None, scale: None }, &[x]);
    let h = g.push(Op::RmsNorm { weight: Some(m.attn_norm), bias: None, eps: 1e-6 }, &[x]);
    let k = mm(&mut g, m.wk, h);
    let v = mm(&mut g, m.wv, h);
    let q = mm(&mut g, m.wq, h);
    let qn = m.q_norm.map_or(q, |w| head_norm(&mut g, q, w, n_tokens, m.n_head, m.hd));
    let kn = m.k_norm.map_or(k, |w| head_norm(&mut g, k, w, n_tokens, m.n_head_kv, m.hd));
    let rope = RopeParams { kind: m.rope, dims: m.hd, base: 10000.0, scale: 1.0, orig_ctx: N_CTX, attn_factor: 1.0, yarn: None, sections: [0; 4], factors: None };
    let rq = g.push(Op::Rope { params: rope.clone(), n_head: m.n_head, head_dim: m.hd, offset: 0, inverse: false }, &[qn, positions]);
    let rk = g.push(Op::Rope { params: rope, n_head: m.n_head_kv, head_dim: m.hd, offset: 0, inverse: false }, &[kn, positions]);
    let a = g.push(Op::Attention { layer: 0, n_head: m.n_head, n_head_kv: m.n_head_kv, head_dim: m.hd, head_dim_v: m.hd, scale: 1.0 / (m.hd as f32).sqrt(), window: None, softcap: 0.0, sinks: None, alibi_bias: 0.0, kv_source: None, top_k: None }, &[rq, rk, v]);
    let o = mm(&mut g, m.wo, a);
    let x1 = g.add(x, o);
    let h2 = g.push(Op::RmsNorm { weight: Some(m.ffn_norm), bias: None, eps: 1e-6 }, &[x1]);
    let gt = mm(&mut g, m.gate, h2);
    let up = mm(&mut g, m.up, h2);
    let act = g.push(Op::GatedAct { act: kf_core::Act::Silu, limit: None }, &[gt, up]);
    let d = mm(&mut g, m.down, act);
    let y = g.add(x1, d);
    let out = match stop {
        "norm" => h,
        "k" => k,
        "v" => v,
        "q" => q,
        "qn" => qn,
        "kn" => kn,
        "ropeq" => rq,
        "ropek" => rk,
        "attn" => a,
        "o" => o,
        "add1" => x1,
        "norm2" => h2,
        "gate" => gt,
        "up" => up,
        "act" => act,
        "down" => d,
        _ => y,
    };
    g.finish(out)
}

fn plan(pool: Pool) -> Plan {
    Plan { layers: vec![LayerPlan { layer: 0, weights: pool, state: pool, expert_cache_slots: 0, kv_resident: 0 }], pinned: Vec::new(), weights_bytes: 0, state_bytes: 0, n_ctx: N_CTX, n_batch: N_BATCH, n_seq: 2 }
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

fn blocks() -> Vec<(&'static str, Block)> {
    vec![("qwen3 q8_0", block(QWEN, 16, 8, 128, true, RopeKind::Neox)), ("llama q4_K_M", block(LLAMA, 32, 8, 64, false, RopeKind::Normal))]
}

/// The first block of real models on the device matches the reference over prefill chunks that take the tensor core
/// products and the fused norms, activations, ropes, cache appends and residual adds, then over decode steps, and
/// every intermediate tensor can still be read back as the graph output, with single row decodes run through the
/// launch sequence and then through the persistent pass
#[test]
fn fused_block_matches_reference() {
    for persistent in [false, true] {
        // The switch is read when a decode graph is first captured, before any other thread of this binary runs
        unsafe {
            if persistent {
                std::env::set_var("KF_PERSIST", "1");
            } else {
                std::env::remove_var("KF_PERSIST");
            }
        }
        block_matches_reference(persistent);
    }
}

fn block_matches_reference(persistent: bool) {
    for (what, m) in blocks() {
        let what = format!("{what} persistent {persistent}");
        let mut cpu = CpuBackend::new();
        cpu.apply_plan(&plan(Pool::Host)).expect("cpu plan");
        let mut gpu = cuda();
        gpu.apply_plan(&plan(Pool::Device(0))).expect("cuda plan");
        for (n, start, seq) in [(70usize, 0usize, 0usize), (130, 70, 0), (33, 0, 1)] {
            both(&format!("{what} prefill {n} rows of slot {seq} at {start}"), &graph(&m, n, "out"), &run_inputs(&[(seq, start, n)]), &m.w, &mut cpu, &mut gpu, 2e-2);
        }
        for s in 0..3 {
            both(&format!("{what} decode step {s}"), &graph(&m, 2, "out"), &run_inputs(&[(0, 200 + s, 1), (1, 33 + s, 1)]), &m.w, &mut cpu, &mut gpu, 2e-2);
        }
        both(&format!("{what} single decode"), &graph(&m, 1, "out"), &run_inputs(&[(1, 36, 1)]), &m.w, &mut cpu, &mut gpu, 2e-2);
        for stop in ["norm", "kn", "ropek", "act", "add1"] {
            cpu.reset();
            gpu.reset();
            both(&format!("{what} output at {stop}"), &graph(&m, 40, stop), &run_inputs(&[(0, 0, 40)]), &m.w, &mut cpu, &mut gpu, 2e-2);
        }
        // Short prefills below the tensor core attention's row count, then decode steps whose appended key and value
        // carry most of the attention
        for stop in ["attn", "out"] {
            for n in [2usize, 6, 12] {
                cpu.reset();
                gpu.reset();
                both(&format!("{what} prefill of {n} at {stop}"), &graph(&m, n, stop), &run_inputs(&[(0, 0, n)]), &m.w, &mut cpu, &mut gpu, 2e-2);
                for s in 0..3 {
                    both(&format!("{what} decode {s} after a prefill of {n} at {stop}"), &graph(&m, 1, stop), &run_inputs(&[(0, n + s, 1)]), &m.w, &mut cpu, &mut gpu, 2e-2);
                }
            }
        }
    }
}

/// A tree pass of more rows than the tensor core attention takes runs the decode kernel with its quantizing fusion
/// on a real block and matches the reference, as the learned head's trees of sixteen rows and more do, and a chain
/// shaped tree of such rows equals the same rows fed plain
#[test]
fn wide_tree_pass_matches_reference() {
    for (what, m) in blocks() {
        let mut cpu = CpuBackend::new();
        cpu.apply_plan(&plan(Pool::Host)).expect("cpu plan");
        let mut gpu = cuda();
        gpu.apply_plan(&plan(Pool::Device(0))).expect("cuda plan");
        both(&format!("{what} history"), &graph(&m, 40, "out"), &run_inputs(&[(0, 0, 40)]), &m.w, &mut cpu, &mut gpu, 2e-2);
        // Twenty rows, four children of the root and a chain of four below each
        let tokens: Vec<u32> = (0..20).map(|i| (i * 11 + 3) as u32 % 61).collect();
        let parents: Vec<i32> = (0..20).map(|i| if i == 0 { -1 } else if i <= 4 { 0 } else { (i - 4) as i32 }).collect();
        both(&format!("{what} tree of 20 rows"), &graph(&m, 20, "out"), &tree_inputs(0, 40, &tokens, &parents), &m.w, &mut cpu, &mut gpu, 2e-2);
        cpu.truncate(0, 40).expect("truncate");
        gpu.truncate(0, 40).expect("truncate");
        let chain: Vec<u32> = (0..24).map(|i| (i * 5 + 1) as u32 % 61).collect();
        let chain_parents: Vec<i32> = (0..24).map(|i| i as i32 - 1).collect();
        let plain = Inputs { tokens: chain.clone(), positions: (40..64).collect(), seqs: vec![0; 24], ..Default::default() };
        let want = cpu.run(&graph(&m, 24, "out"), &plain, &m.w).expect("plain rows");
        cpu.truncate(0, 40).expect("truncate");
        let tree = both(&format!("{what} chain tree of 24 rows"), &graph(&m, 24, "out"), &tree_inputs(0, 40, &chain, &chain_parents), &m.w, &mut cpu, &mut gpu, 2e-2);
        assert_close(&format!("{what} chain tree against plain rows"), &tree, &want, 2e-2);
    }
}

/// Relative error of every stage of the block against the reference, for finding where the device drifts
#[test]
#[ignore]
fn block_stage_errors() {
    for (what, m) in blocks() {
        let n = 70;
        for stop in ["norm", "k", "v", "q", "qn", "kn", "ropeq", "ropek", "attn", "o", "add1", "norm2", "gate", "up", "act", "down", "out"] {
            let graph = graph(&m, n, stop);
            let mut cpu = CpuBackend::new();
            cpu.apply_plan(&plan(Pool::Host)).expect("cpu plan");
            let mut gpu = cuda();
            gpu.apply_plan(&plan(Pool::Device(0))).expect("cuda plan");
            let inp = run_inputs(&[(0, 0, n)]);
            let want = cpu.run(&graph, &inp, &m.w).expect("cpu");
            let got = gpu.run(&graph, &inp, &m.w).expect("cuda");
            let scale = want.iter().fold(0.0f32, |a, v| a.max(v.abs()));
            eprintln!("{what} stage {stop:6}: relative error {:.2e}  max |ref| {scale:.3}", rel_err(&got, &want));
        }
    }
}

/// Error of every pass of the fused block test in order, for finding the pass after which a later one drifts
#[test]
#[ignore]
fn block_reuse_probe() {
    let (what, m) = blocks().remove(0);
    let mut cpu = CpuBackend::new();
    cpu.apply_plan(&plan(Pool::Host)).expect("cpu plan");
    let mut gpu = cuda();
    gpu.apply_plan(&plan(Pool::Device(0))).expect("cuda plan");
    let probe = |cpu: &mut CpuBackend, gpu: &mut kf_backend_cuda::CudaBackend, label: &str, g: &Graph, inp: &Inputs| {
        let want = cpu.run(g, inp, &m.w).expect("cpu");
        let got = gpu.run(g, inp, &m.w).expect("cuda");
        eprintln!("{what} {label}: relative error {:.2e}", rel_err(&got, &want));
    };
    for (n, start, seq) in [(70usize, 0usize, 0usize), (130, 70, 0), (33, 0, 1)] {
        probe(&mut cpu, &mut gpu, &format!("prefill {n} rows of slot {seq} at {start}"), &graph(&m, n, "out"), &run_inputs(&[(seq, start, n)]));
    }
    for s in 0..3 {
        probe(&mut cpu, &mut gpu, &format!("decode step {s}"), &graph(&m, 2, "out"), &run_inputs(&[(0, 200 + s, 1), (1, 33 + s, 1)]));
    }
    probe(&mut cpu, &mut gpu, "single decode", &graph(&m, 1, "out"), &run_inputs(&[(1, 36, 1)]));
    for stop in ["norm", "kn", "ropek", "act", "add1", "out"] {
        cpu.reset();
        gpu.reset();
        probe(&mut cpu, &mut gpu, &format!("output at {stop}"), &graph(&m, 40, stop), &run_inputs(&[(0, 0, 40)]));
    }
}

mod common;

use common::*;
use kf_backend_cpu::CpuBackend;
use kf_core::{Act, Backend, Gating, GraphBuilder, LayerPlan, Op, Plan, Pool, Selection, WeightNorm, Wid};

const N_EMBD: usize = 256;
const N_FF: usize = 256;

struct Moe {
    w: Mem,
    table: Wid,
    router: Wid,
    gate: Wid,
    up: Wid,
    down: Wid,
    n_expert: usize,
    n_used: usize,
}

/// Synthetic routed experts of one quant type with a random router and embedding table
fn moe(ty: u32, seed: u64, n_expert: usize, n_used: usize) -> Moe {
    let mut rng = Rng(seed);
    let mut w = Mem::default();
    let tv = rng.vec(32 * N_EMBD, 1.0);
    let table = w.f32("table", &[N_EMBD, 32], &tv);
    let rv = rng.vec(N_EMBD * n_expert, 0.5);
    let router = w.f32("router", &[N_EMBD, n_expert], &rv);
    let gate = w.add("gate_exps", ty, &[N_EMBD, N_FF, n_expert], synth(&mut rng, ty, N_FF * n_expert, N_EMBD));
    let up = w.add("up_exps", ty, &[N_EMBD, N_FF, n_expert], synth(&mut rng, ty, N_FF * n_expert, N_EMBD));
    let down = w.add("down_exps", ty, &[N_FF, N_EMBD, n_expert], synth(&mut rng, ty, N_EMBD * n_expert, N_FF));
    Moe { w, table, router, gate, up, down, n_expert, n_used }
}

fn op(m: &Moe) -> Op {
    Op::MoeFfn {
        router: m.router,
        router_bias: None,
        probs_bias: None,
        gate_exps: Some(m.gate),
        up_exps: m.up,
        down_exps: Some(m.down),
        gate_b: None,
        up_b: None,
        down_b: None,
        exps_scale: None,
        n_expert: m.n_expert,
        n_used: m.n_used,
        gating: Gating::Softmax,
        selection: Selection::Probs,
        norm_weights: WeightNorm::Clamp,
        scale: 1.0,
        act: Act::Silu,
        limit: None,
        groups: None,
        fused_gate_up: false,
        weight_before_ffn: false,
        index_div: 1,
        post_norm: None,
        post_norm_eps: 0.0,
        router_in: false,
        selected_in: false,
    }
}

fn graph(m: &Moe) -> kf_core::Graph {
    let mut g = GraphBuilder::new();
    let tokens = g.input("tokens");
    let x = g.push(Op::Embed { weight: m.table }, &[tokens]);
    g.layer = Some(0);
    let y = g.push(op(m), &[x]);
    g.finish(y)
}

fn plan(slots: usize, pinned: Vec<Wid>) -> Plan {
    Plan { layers: vec![LayerPlan { layer: 0, weights: Pool::Device(0), state: Pool::Device(0), expert_cache_slots: slots, kv_resident: 0 }], pinned, weights_bytes: 0, state_bytes: 0, n_ctx: 64, n_batch: 16, n_seq: 1 }
}

fn check(name: &str, ty: u32, tol: f32) {
    check_with(name, ty, tol, 8, 2);
    check_with(name, ty, tol, 128, 8);
}

fn check_with(name: &str, ty: u32, tol: f32, n_expert: usize, n_used: usize) {
    let m = moe(ty, 7, n_expert, n_used);
    let name = &format!("{name} {n_used} of {n_expert}");
    let g = graph(&m);
    let mut cpu = CpuBackend::new();
    let mut gpu = cuda();
    gpu.apply_plan(&plan(0, Vec::new())).expect("resident plan");
    for toks in [&[3u32][..], &[3, 17, 0, 31, 9, 12, 1]] {
        both(&format!("{name} resident experts with {} rows", toks.len()), &g, &inputs(toks, 0), &m.w, &mut cpu, &mut gpu, tol);
    }
    let mut gpu = cuda();
    gpu.apply_plan(&plan(3, vec![m.gate, m.up, m.down])).expect("cached plan");
    for toks in [&[3u32][..], &[3, 17, 0, 31, 9, 12, 1], &[5, 5, 2]] {
        both(&format!("{name} cached experts with {} rows", toks.len()), &g, &inputs(toks, 0), &m.w, &mut cpu, &mut gpu, tol);
    }
}

#[test]
fn f32_experts_match_reference() {
    check("f32", 0, 1e-3);
}

#[test]
fn q8_0_experts_match_reference() {
    check("q8_0", kf_quant::by_name("q8_0").expect("q8_0").info().id, 2e-2);
}

#[test]
fn i_quant_experts_match_reference() {
    for name in ["iq2_xxs", "iq2_xs", "iq2_s", "iq3_xxs", "iq3_s", "iq4_xs", "q4_K", "q5_K", "q6_K"] {
        check(name, kf_quant::by_name(name).expect(name).info().id, 2e-2);
    }
}

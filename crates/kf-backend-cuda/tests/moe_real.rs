mod common;

use common::*;
use kf_backend_cpu::CpuBackend;
use kf_core::{Act, Backend, Gating, GraphBuilder, LayerPlan, Op, Plan, Pool, Selection, WeightNorm, Wid};

const MOE: &str = "Qwen3-30B-A3B-UD-IQ2_XXS.gguf";

fn plan(slots: usize, pinned: Vec<Wid>) -> Plan {
    Plan { layers: vec![LayerPlan { layer: 0, weights: Pool::Device(0), state: Pool::Device(0), expert_cache_slots: slots, kv_resident: 0 }], pinned, weights_bytes: 0, state_bytes: 0, n_ctx: 64, n_batch: 16, n_seq: 1 }
}

fn moe_op(router: Wid, gate: Wid, up: Wid, down: Wid) -> Op {
    Op::MoeFfn {
        router,
        router_bias: None,
        probs_bias: None,
        gate_exps: Some(gate),
        up_exps: up,
        down_exps: Some(down),
        gate_b: None,
        up_b: None,
        down_b: None,
        exps_scale: None,
        n_expert: 128,
        n_used: 8,
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

const TOKS: [u32; 5] = [785, 6722, 315, 9625, 374];

/// The real embedding rows normed, the input every block sees first
fn normed_embed(g: &mut GraphBuilder, table: Wid, norm: Wid) -> kf_core::Tid {
    let tokens = g.input("tokens");
    let x = g.push(Op::Embed { weight: table }, &[tokens]);
    g.push(Op::RmsNorm { weight: Some(norm), bias: None, eps: 1e-6 }, &[x])
}

#[test]
fn real_block_tensors_match_reference() {
    let file = gguf(MOE);
    let mut w = Mem::default();
    let table = w.gguf(&file, "token_embd.weight", None);
    let norm = w.gguf(&file, "blk.0.attn_norm.weight", None);
    let q = w.gguf(&file, "blk.0.attn_q.weight", None);
    let ffn_norm = w.gguf(&file, "blk.0.ffn_norm.weight", None);
    let router = w.gguf(&file, "blk.0.ffn_gate_inp.weight", None);
    let gate = w.gguf(&file, "blk.0.ffn_gate_exps.weight", None);
    let up = w.gguf(&file, "blk.0.ffn_up_exps.weight", None);
    let down = w.gguf(&file, "blk.0.ffn_down_exps.weight", None);
    let mut cpu = CpuBackend::new();
    let mut gpu = cuda();
    gpu.apply_plan(&plan(0, Vec::new())).expect("plan");
    for toks in [&TOKS[..1], &TOKS[..]] {
        let mut g = GraphBuilder::new();
        let tokens = g.input("tokens");
        let x = g.push(Op::Embed { weight: table }, &[tokens]);
        let graph = g.finish(x);
        both(&format!("q4_K embedding with {} rows", toks.len()), &graph, &inputs(toks, 0), &w, &mut cpu, &mut gpu, 1e-3);
        let mut g = GraphBuilder::new();
        let x = normed_embed(&mut g, table, norm);
        let y = g.push(Op::MatMul { weight: q, bias: None, scale: None }, &[x]);
        let graph = g.finish(y);
        both(&format!("iq4_xs attn_q with {} rows", toks.len()), &graph, &inputs(toks, 0), &w, &mut cpu, &mut gpu, 2e-2);
        let mut g = GraphBuilder::new();
        let x = normed_embed(&mut g, table, ffn_norm);
        g.layer = Some(0);
        let y = g.push(moe_op(router, gate, up, down), &[x]);
        let graph = g.finish(y);
        both(&format!("iq2_xxs experts resident with {} rows", toks.len()), &graph, &inputs(toks, 0), &w, &mut cpu, &mut gpu, 2e-2);
    }
    let mut gpu = cuda();
    gpu.apply_plan(&plan(6, vec![gate, up, down])).expect("plan");
    for toks in [&TOKS[..1], &TOKS[..]] {
        let mut g = GraphBuilder::new();
        let x = normed_embed(&mut g, table, ffn_norm);
        g.layer = Some(0);
        let y = g.push(moe_op(router, gate, up, down), &[x]);
        let graph = g.finish(y);
        both(&format!("iq2_xxs experts cached with {} rows", toks.len()), &graph, &inputs(toks, 0), &w, &mut cpu, &mut gpu, 2e-2);
    }
}

#[test]
fn real_last_block_tensors_match_reference() {
    let file = gguf(MOE);
    let mut w = Mem::default();
    let table = w.gguf(&file, "token_embd.weight", None);
    let norm = w.gguf(&file, "blk.47.attn_norm.weight", None);
    let q = w.gguf(&file, "blk.47.attn_q.weight", None);
    let o = w.gguf(&file, "blk.47.attn_output.weight", None);
    let ffn_norm = w.gguf(&file, "blk.47.ffn_norm.weight", None);
    let router = w.gguf(&file, "blk.47.ffn_gate_inp.weight", None);
    let gate = w.gguf(&file, "blk.47.ffn_gate_exps.weight", None);
    let up = w.gguf(&file, "blk.47.ffn_up_exps.weight", None);
    let down = w.gguf(&file, "blk.47.ffn_down_exps.weight", None);
    let mut cpu = CpuBackend::new();
    let mut gpu = cuda();
    gpu.apply_plan(&plan(0, Vec::new())).expect("plan");
    for toks in [&TOKS[..1], &TOKS[..]] {
        let mut g = GraphBuilder::new();
        let x = normed_embed(&mut g, table, norm);
        let y = g.push(Op::MatMul { weight: q, bias: None, scale: None }, &[x]);
        let z = g.push(Op::MatMul { weight: o, bias: None, scale: None }, &[y]);
        let graph = g.finish(z);
        both(&format!("q5_K and q6_K attention projections with {} rows", toks.len()), &graph, &inputs(toks, 0), &w, &mut cpu, &mut gpu, 2e-2);
        let mut g = GraphBuilder::new();
        let x = normed_embed(&mut g, table, ffn_norm);
        g.layer = Some(0);
        let y = g.push(moe_op(router, gate, up, down), &[x]);
        let graph = g.finish(y);
        both(&format!("iq2_s and iq4_xs experts resident with {} rows", toks.len()), &graph, &inputs(toks, 0), &w, &mut cpu, &mut gpu, 2e-2);
    }
}

/// The same warm up and prefill on two fresh backends, cached experts mixing device and host work, must give the same bits
#[test]
fn cached_experts_are_deterministic() {
    let file = gguf(MOE);
    let mut w = Mem::default();
    let table = w.gguf(&file, "token_embd.weight", None);
    let ffn_norm = w.gguf(&file, "blk.0.ffn_norm.weight", None);
    let router = w.gguf(&file, "blk.0.ffn_gate_inp.weight", None);
    let gate = w.gguf(&file, "blk.0.ffn_gate_exps.weight", None);
    let up = w.gguf(&file, "blk.0.ffn_up_exps.weight", None);
    let down = w.gguf(&file, "blk.0.ffn_down_exps.weight", None);
    let mut g = GraphBuilder::new();
    let x = normed_embed(&mut g, table, ffn_norm);
    g.layer = Some(0);
    let y = g.push(moe_op(router, gate, up, down), &[x]);
    let graph = g.finish(y);
    let prompt: Vec<u32> = (0..33u32).map(|i| (i * 977 + 13) % 150_000).collect();
    let run = || -> Vec<Vec<f32>> {
        let mut gpu = cuda();
        gpu.apply_plan(&plan(2, vec![gate, up, down])).expect("plan");
        let mut outs = Vec::new();
        for toks in [&[151643u32][..], &[151645u32][..], &prompt[..], &prompt[..1], &prompt[1..2]] {
            outs.push(gpu.run(&graph, &inputs(toks, 0), &w).expect("cuda"));
        }
        outs
    };
    let a = run();
    let b = run();
    for (k, (x, y)) in a.iter().zip(&b).enumerate() {
        let same = x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits());
        let diff = x.iter().zip(y).map(|(p, q)| (p - q).abs()).fold(0f32, f32::max);
        assert!(same, "pass {k} differs between runs, largest difference {diff}");
    }
}

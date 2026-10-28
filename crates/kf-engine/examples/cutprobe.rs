//! Runs a prompt through both backends up to the end of one layer and prints how far their outputs differ
use anyhow::Result;
use kf_core::{GraphBuilder, Inputs};
use kf_engine::{BackendChoice, CtxChoice};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let loaded = kf_engine::load(&args[1])?;
    let ids: Vec<u32> = serde_json::from_str(&args[2])?;
    let mut g = GraphBuilder::new();
    let out = loaded.model.logits(&mut g, ids.len(), 1)?;
    let full = g.finish(out);
    let short = |op: &kf_core::Op| format!("{op:?}").split([' ', '(', '{']).next().unwrap_or("op").to_string();
    if let Some(layer) = args[3].strip_prefix("list:") {
        let layer: Option<usize> = layer.parse().ok();
        for (i, n) in full.nodes.iter().enumerate().filter(|(i, n)| n.layer == layer && (layer.is_some() || *i > 2)) {
            println!("node {i} t{} {} inputs {:?}", n.out.0, short(&n.op), n.inputs);
            if matches!(n.op, kf_core::Op::MoeFfn { .. }) {
                println!("  {:?}", n.op);
            }
        }
        return Ok(());
    }
    if let Some(spec) = args[3].strip_prefix("moe_on:") {
        let (src, moe): (usize, usize) = { let (a, b) = spec.split_once(':').expect("moe_on:src:moe"); (a.parse()?, b.parse()?) };
        let mut g = GraphBuilder::new();
        for _ in 0..=full.n_tensors() {
            g.input("tokens");
        }
        g.graph.inputs = full.inputs.clone();
        g.graph.nodes = full.nodes[..=src].to_vec();
        let mut m = full.nodes[moe].clone();
        m.inputs = vec![full.nodes[src].out];
        let out = m.out;
        g.graph.nodes.push(m);
        if args.get(4).is_some_and(|a| a == "add") {
            g.graph.nodes.push(kf_core::Node { op: kf_core::Op::Add, inputs: vec![full.nodes[src].out, out], out: kf_core::Tid(full.n_tensors() as u32), layer: Some(0) });
        }
        let last = g.graph.nodes.last().expect("node").out;
        let graph = g.finish(last);
        let inputs = Inputs { tokens: ids.clone(), positions: (0..ids.len() as u32).collect(), ..Default::default() };
        for choice in [BackendChoice::Cpu, BackendChoice::Cuda { device: 0 }] {
            let mut p = kf_engine::prepare(&loaded, choice, CtxChoice::Tokens(4096), 512, 1, None, None, None, &kf_engine::Scope::all(loaded.model.n_layers()))?;
            let out = p.backend.run(&graph, &inputs, loaded.store.as_ref())?;
            println!("{choice:?}: {:?}", &out[..4]);
        }
        return Ok(());
    }
    let cuts: Vec<usize> = args[3].split(',').map(|s| match s.strip_prefix("n:") {
        Some(k) => k.parse().expect("node"),
        None => full.nodes.iter().rposition(|n| n.layer == Some(s.parse::<usize>().expect("layer"))).expect("layer in graph"),
    }).collect();
    for k in cuts {
        let layer = format!("node {k} {} layer {:?}", short(&full.nodes[k].op), full.nodes[k].layer);
        let mut g = GraphBuilder::new();
        for _ in 0..full.n_tensors() {
            g.input("tokens");
        }
        g.graph.inputs = full.inputs.clone();
        g.graph.nodes = full.nodes[..=k].to_vec();
        let graph = g.finish(full.nodes[k].out);
        let inputs = Inputs { tokens: ids.clone(), positions: (0..ids.len() as u32).collect(), ..Default::default() };
        let mut rows = Vec::new();
        for choice in [BackendChoice::Cpu, BackendChoice::Cuda { device: 0 }] {
            let mut p = kf_engine::prepare(&loaded, choice, CtxChoice::Tokens(4096), 512, 1, None, None, None, &kf_engine::Scope::all(loaded.model.n_layers()))?;
            rows.push(p.backend.run(&graph, &inputs, loaded.store.as_ref())?);
        }
        let (a, b) = (&rows[0], &rows[1]);
        let cols = a.len() / ids.len();
        for t in 0..ids.len() {
            let (ra, rb) = (&a[t * cols..(t + 1) * cols], &b[t * cols..(t + 1) * cols]);
            let diff = ra.iter().zip(rb).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
            let amax = ra.iter().map(|x| x.abs()).fold(0f32, f32::max);
            let dot: f32 = ra.iter().zip(rb).map(|(x, y)| x * y).sum();
            let na: f32 = ra.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb: f32 = rb.iter().map(|x| x * x).sum::<f32>().sqrt();
            println!("{layer} pos {t}: max diff {diff:.4} cpu amax {amax:.3} cos {:.6} cpu[:3] {:?} cuda[:3] {:?}", dot / (na * nb), &ra[..3], &rb[..3]);
        }
    }
    Ok(())
}

//! After one prompt, runs forced tokens as a batch and the first of them alone up to a cut and prints how the first row differs
use anyhow::Result;
use kf_core::{GraphBuilder, Inputs, Op};
use kf_engine::{BackendChoice, CtxChoice, Session};
use kf_layers::Segment;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let loaded = kf_engine::load(&args[1])?;
    let prompt: Vec<u32> = serde_json::from_str(&args[2])?;
    let forced: Vec<u32> = serde_json::from_str(&args[3])?;
    let choice = if args.get(5).is_some_and(|a| a == "cuda") { BackendChoice::Cuda { device: 0 } } else { BackendChoice::Cpu };
    let scope = kf_engine::Scope::all(loaded.model.n_layers());
    let n_layers = loaded.model.n_layers();
    let short = |op: &Op| format!("{op:?}").split([' ', '(', '{']).next().unwrap_or("op").to_string();
    for cut in args[4].split(',') {
        let mut outs = Vec::new();
        for toks in [forced.clone(), forced[..1].to_vec()] {
            let p = kf_engine::prepare(&loaded, choice, CtxChoice::Tokens(4096), 512, 1, None, None, None, &scope)?;
            let mut s = Session::new(&loaded, p.backend, &p.plan);
            s.feed(0, &prompt)?;
            let mut g = GraphBuilder::new();
            let full = loaded.model.segment(&mut g, toks.len(), Segment { from: 0, to: n_layers, embed: true, inject: false, n_out: Some(toks.len()) })?;
            let (graph, label) = match cut.strip_prefix("n:") {
                Some(k) => {
                    let k: usize = k.parse()?;
                    let mut gg = g;
                    gg.graph.nodes.truncate(k + 1);
                    let out = gg.graph.nodes[k].out;
                    let label = format!("node {k} {} layer {:?}", short(&gg.graph.nodes[k].op), gg.graph.nodes[k].layer);
                    (gg.finish(out), label)
                }
                None => {
                    let layer: usize = cut.parse()?;
                    let k = g.graph.nodes.iter().rposition(|n| n.layer == Some(layer)).expect("layer");
                    let mut gg = g;
                    gg.graph.nodes.truncate(k + 1);
                    let out = gg.graph.nodes[k].out;
                    let _ = full;
                    (gg.finish(out), format!("layer {layer}"))
                }
            };
            let n_past = s.n_past(0);
            let inputs = Inputs { tokens: toks.clone(), positions: (n_past..n_past + toks.len()).map(|p| p as u32).collect(), ..Default::default() };
            let out = s.backend_mut().run(&graph, &inputs, loaded.store.as_ref())?;
            let cols = out.len() / toks.len();
            outs.push((label, out[..cols].to_vec()));
        }
        let (a, b) = (&outs[0].1, &outs[1].1);
        let diff = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        let amax = b.iter().map(|x| x.abs()).fold(0f32, f32::max);
        println!("{}: row 0 max diff {diff:.6} single amax {amax:.4}", outs[0].0);
    }
    Ok(())
}

//! Prints every node of a one token forward pass with its shape and leading values
use anyhow::Result;
use kf_backend_cpu::{Buf, CpuBackend};
use kf_core::{GraphBuilder, Inputs};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let loaded = kf_engine::load(&args[1])?;
    let ids: Vec<u32> = serde_json::from_str(&args[2])?;
    let limit: usize = args.get(3).map_or(400, |s| s.parse().unwrap_or(400));
    let dump = args.get(4).cloned();
    let mut g = GraphBuilder::new();
    let out = loaded.model.logits(&mut g, ids.len(), ids.len())?;
    let graph = g.finish(out);
    let inputs = Inputs { tokens: ids.clone(), positions: (0..ids.len() as u32).collect(), ..Default::default() };
    let mut b = CpuBackend::new();
    let mut t: Vec<Option<Buf>> = vec![None; graph.n_tensors()];
    for (i, node) in graph.nodes.iter().enumerate().take(limit) {
        let ins: Vec<Buf> = node.inputs.iter().map(|id| t[id.0 as usize].clone().expect("input")).collect();
        let o = b.exec_node(&node.op, &ins, &inputs, loaded.store.as_ref())?;
        let name = format!("{:?}", node.op);
        let name = name.split([' ', '(', '{']).next().unwrap_or("op");
        let amax = o.data.iter().fold(0f32, |m, v| m.max(v.abs()));
        let head: Vec<String> = o.data.iter().take(4).map(|v| format!("{v:.5}")).collect();
        let last: Vec<String> = if o.rows > 1 { o.row(o.rows - 1).iter().take(4).map(|v| format!("{v:.5}")).collect() } else { Vec::new() };
        println!("{i:4} t{:<4} L{:<3} {name:<14} {}x{} amax {amax:.5} [{}] last [{}]", node.out.0, node.layer.map_or(-1, |l| l as i64), o.rows, o.cols, head.join(", "), last.join(", "));
        if i + 2 == graph.nodes.len()
            && let Some(path) = &dump
        {
            let rows: Vec<&[f32]> = (0..o.rows).map(|r| o.row(r)).collect();
            std::fs::write(path, serde_json::to_string(&rows)?)?;
        }
        if i + 1 == graph.nodes.len() {
            let row = o.row(o.rows - 1);
            let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let lse = row.iter().map(|v| (v - m).exp()).sum::<f32>().ln() + m;
            let mut idx: Vec<usize> = (0..row.len()).collect();
            idx.sort_by(|a, b| row[*b].partial_cmp(&row[*a]).unwrap_or(std::cmp::Ordering::Equal));
            let top: Vec<String> = idx.iter().take(6).map(|i| format!("({i}, {:.3})", row[*i] - lse)).collect();
            println!("top: {}", top.join(" "));
        }
        t[node.out.0 as usize] = Some(o);
    }
    Ok(())
}

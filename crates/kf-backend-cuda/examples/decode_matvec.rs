//! Times one token through every block's projections of a GGUF file on the CUDA backend
use anyhow::{Context, Result, anyhow};
use kf_backend_cuda::CudaBackend;
use kf_core::{Act, Backend, GraphBuilder, Inputs, Op, WeightSource, Wid};
use kf_gguf::{Gguf, TensorInfo};
use std::path::PathBuf;
use std::time::Instant;

struct File {
    g: Gguf,
    t: Vec<TensorInfo>,
}

impl WeightSource for File {
    fn bytes(&self, w: Wid) -> Result<&[u8]> {
        self.g.bytes(self.t.get(w.0 as usize).ok_or_else(|| anyhow!("no weight {}", w.0))?)
    }
    fn ggml_type(&self, w: Wid) -> u32 {
        self.t[w.0 as usize].ggml_type
    }
    fn dims(&self, w: Wid) -> &[usize] {
        &self.t[w.0 as usize].dims
    }
    fn name(&self, w: Wid) -> &str {
        &self.t[w.0 as usize].name
    }
    fn len(&self) -> usize {
        self.t.len()
    }
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let path = PathBuf::from(args.next().context("usage decode_matvec FILE [CACHE_DIR]")?);
    let cache = args.next().map(PathBuf::from);
    let g = Gguf::open(&path)?;
    let t = g.tensors.clone();
    let file = File { g, t };
    let wid = |name: &str| file.t.iter().position(|t| t.name == name).map(|i| Wid(i as u32)).with_context(|| format!("{name} missing"));
    let n_layer = (0..).take_while(|l| wid(&format!("blk.{l}.attn_q.weight")).is_ok()).count();
    let embd = wid("token_embd.weight")?;
    let mut gb = GraphBuilder::new();
    let mut x = gb.push(Op::Embed { weight: embd }, &[]);
    let mut bytes = 0usize;
    for l in 0..n_layer {
        let w = |s: &str| wid(&format!("blk.{l}.{s}.weight"));
        for s in ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"] {
            bytes += file.bytes(w(s)?)?.len();
        }
        let mm = |gb: &mut GraphBuilder, weight: Wid, x| gb.push(Op::MatMul { weight, bias: None, scale: None }, &[x]);
        let q = mm(&mut gb, w("attn_q")?, x);
        let k = mm(&mut gb, w("attn_k")?, x);
        let v = mm(&mut gb, w("attn_v")?, x);
        let o = mm(&mut gb, w("attn_output")?, q);
        let gate = mm(&mut gb, w("ffn_gate")?, x);
        let up = mm(&mut gb, w("ffn_up")?, x);
        let h = gb.push(Op::GatedAct { act: Act::Silu, limit: None }, &[gate, up]);
        let d = mm(&mut gb, w("ffn_down")?, h);
        gb.add(k, v);
        let y = gb.add(o, d);
        let y = gb.add(x, y);
        x = gb.push(Op::RmsNorm { weight: Some(w("attn_norm")?), bias: None, eps: 1e-6 }, &[y]);
    }
    let graph = gb.finish(x);
    let mut gpu = CudaBackend::new(0, cache.as_deref())?;
    let inputs = Inputs { tokens: vec![1], positions: vec![0], ..Default::default() };
    for _ in 0..5 {
        gpu.run(&graph, &inputs, &file)?;
    }
    let iters = 200;
    let start = Instant::now();
    for _ in 0..iters {
        gpu.run(&graph, &inputs, &file)?;
    }
    let secs = start.elapsed().as_secs_f64();
    let (captures, replays) = gpu.graph_stats();
    println!(
        "{} blocks {:.1} MB of projections, {:.1} tokens per second, {:.0} GB per second, graphs captured {captures} replayed {replays}",
        n_layer,
        bytes as f64 / 1e6,
        iters as f64 / secs,
        bytes as f64 * iters as f64 / secs / 1e9
    );
    Ok(())
}

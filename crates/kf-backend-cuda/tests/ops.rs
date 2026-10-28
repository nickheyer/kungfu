mod common;

use common::*;
use kf_backend_cpu::CpuBackend;
use kf_core::{Act, GraphBuilder, Op, RopeKind, RopeParams, Tid, Wid, Yarn};

struct Setup {
    w: Mem,
    rng: Rng,
    cpu: CpuBackend,
    gpu: kf_backend_cuda::CudaBackend,
    table: Wid,
}

const N: usize = 96;

fn setup(seed: u64) -> Setup {
    let mut rng = Rng(seed);
    let mut w = Mem::default();
    let v = rng.vec(32 * N, 2.0);
    let table = w.f32("table", &[N, 32], &v);
    Setup { w, rng, cpu: CpuBackend::new(), gpu: cuda(), table }
}

const TOKS: [u32; 5] = [3, 17, 0, 31, 9];

impl Setup {
    /// Builds a graph whose first tensor is rows of the random table, runs both backends, compares
    fn check(&mut self, what: &str, build: impl Fn(&mut GraphBuilder, Tid, usize) -> Tid) {
        for toks in [&TOKS[..1], &TOKS[..]] {
            let mut g = GraphBuilder::new();
            let tokens = g.input("tokens");
            let x = g.push(Op::Embed { weight: self.table }, &[tokens]);
            let y = build(&mut g, x, toks.len());
            let graph = g.finish(y);
            both(&format!("{what} with {} rows", toks.len()), &graph, &inputs(toks, 11), &self.w, &mut self.cpu, &mut self.gpu, 1e-3);
        }
    }
}

#[test]
fn inputs_and_embedding() {
    let mut s = setup(1);
    s.check("tokens input", |g, _, _| g.input("tokens"));
    s.check("positions input", |g, _, _| g.input("positions"));
    s.check("embed f32", |_, x, _| x);
    let qwen = gguf(QWEN);
    let llama = gguf(LLAMA);
    let q8 = s.w.gguf(&qwen, "token_embd.weight", Some(64));
    let q6 = s.w.gguf(&llama, "token_embd.weight", Some(64));
    s.check("embed q8_0", |g, _, _| {
        let tokens = g.input("tokens");
        g.push(Op::Embed { weight: q8 }, &[tokens])
    });
    s.check("embed q6_K", |g, _, _| {
        let tokens = g.input("tokens");
        g.push(Op::Embed { weight: q6 }, &[tokens])
    });
}

#[test]
fn norms() {
    let mut s = setup(2);
    let wv = s.rng.vec(N, 1.5);
    let bv = s.rng.vec(N, 0.5);
    let nw = s.w.f32("norm", &[N], &wv);
    let nb = s.w.f32("bias", &[N], &bv);
    s.check("rms norm", |g, x, _| g.push(Op::RmsNorm { weight: Some(nw), bias: None, eps: 1e-6 }, &[x]));
    s.check("rms norm unweighted", |g, x, _| g.push(Op::RmsNorm { weight: None, bias: None, eps: 1e-5 }, &[x]));
    s.check("layer norm", |g, x, _| g.push(Op::LayerNorm { weight: Some(nw), bias: Some(nb), eps: 1e-5 }, &[x]));
    s.check("layer norm unbiased", |g, x, _| g.push(Op::LayerNorm { weight: Some(nw), bias: None, eps: 1e-6 }, &[x]));
    s.check("rms norm per head", |g, x, n| {
        let heads = g.push(Op::Reshape { rows: n * 4, cols: 24 }, &[x]);
        let normed = g.push(Op::RmsNorm { weight: None, bias: None, eps: 1e-6 }, &[heads]);
        g.push(Op::Reshape { rows: n, cols: 96 }, &[normed])
    });
    s.check("rms norm biased", |g, x, _| g.push(Op::RmsNorm { weight: Some(nw), bias: Some(nb), eps: 1e-6 }, &[x]));
    s.check("layer norm weightless", |g, x, _| g.push(Op::LayerNorm { weight: None, bias: None, eps: 1e-6 }, &[x]));
    s.check("l2 norm", |g, x, _| g.push(Op::L2Norm { eps: 1e-6 }, &[x]));
}

#[test]
fn rope_variants() {
    let mut s = setup(3);
    let fv: Vec<f32> = (0..12).map(|i| 1.0 + i as f32 * 0.5).collect();
    let factors = s.w.f32("rope_freqs", &[12], &fv);
    let base = RopeParams { kind: RopeKind::Neox, dims: 24, base: 10000.0, scale: 1.0, orig_ctx: 4096, attn_factor: 1.0, yarn: None, sections: [0; 4], factors: None };
    let yarn = Yarn { ext_factor: 1.0, beta_fast: 32.0, beta_slow: 1.0 };
    let cases = [
        ("neox", base.clone()),
        ("normal", RopeParams { kind: RopeKind::Normal, ..base.clone() }),
        ("mrope", RopeParams { kind: RopeKind::MRope, sections: [4, 4, 4, 0], ..base.clone() }),
        ("partial", RopeParams { dims: 16, ..base.clone() }),
        ("scaled yarn", RopeParams { scale: 0.25, yarn: Some(yarn), ..base.clone() }),
        ("yarn without extrapolation", RopeParams { scale: 0.5, attn_factor: 0.8, yarn: Some(Yarn { ext_factor: 0.0, ..yarn }), ..base.clone() }),
        ("factors", RopeParams { factors: Some(factors), base: 500000.0, ..base.clone() }),
    ];
    for (what, params) in cases {
        s.check(&format!("rope {what}"), |g, x, _| {
            let pos = g.input("positions");
            g.push(Op::Rope { params: params.clone(), n_head: 4, head_dim: 24, offset: 0, inverse: false }, &[x, pos])
        });
    }
}

#[test]
fn real_rope_factors() {
    let mut s = setup(4);
    let llama = gguf(LLAMA);
    let factors = s.w.gguf(&llama, "rope_freqs.weight", None);
    let params = RopeParams { kind: RopeKind::Normal, dims: 64, base: 500000.0, scale: 1.0, orig_ctx: 8192, attn_factor: 1.0, yarn: None, sections: [0; 4], factors: Some(factors) };
    let v = s.rng.vec(32 * 128, 1.0);
    let wide = s.w.f32("wide", &[128, 32], &v);
    s.check("rope llama factors", |g, _, _| {
        let tokens = g.input("tokens");
        let x = g.push(Op::Embed { weight: wide }, &[tokens]);
        let pos = g.input("positions");
        g.push(Op::Rope { params: params.clone(), n_head: 2, head_dim: 64, offset: 0, inverse: false }, &[x, pos])
    });
}

#[test]
fn activations_and_elementwise() {
    let mut s = setup(5);
    let wv = s.rng.vec(N, 1.0);
    let mw = s.w.f32("mulw", &[N], &wv);
    let v = s.rng.vec(32 * N, 1.0);
    let other = s.w.f32("other", &[N, 32], &v);
    for act in [Act::Silu, Act::Gelu, Act::Relu] {
        s.check(&format!("gated {act:?}"), |g, x, _| {
            let tokens = g.input("tokens");
            let y = g.push(Op::Embed { weight: other }, &[tokens]);
            g.push(Op::GatedAct { act, limit: None }, &[x, y])
        });
        s.check(&format!("gated fused {act:?}"), |g, x, _| g.push(Op::GatedActFused { act, limit: None }, &[x]));
    }
    s.check("add", |g, x, _| {
        let tokens = g.input("tokens");
            let y = g.push(Op::Embed { weight: other }, &[tokens]);
        g.add(x, y)
    });
    s.check("mul", |g, x, _| {
        let tokens = g.input("tokens");
            let y = g.push(Op::Embed { weight: other }, &[tokens]);
        g.mul(x, y)
    });
    s.check("mul weight", |g, x, _| g.push(Op::MulWeight { weight: mw }, &[x]));
    s.check("scale", |g, x, _| g.push(Op::Scale(-0.37), &[x]));
    s.check("silu", |g, x, _| g.push(Op::Silu, &[x]));
    s.check("gelu", |g, x, _| g.push(Op::Gelu, &[x]));
    s.check("relu", |g, x, _| g.push(Op::Relu, &[x]));
    s.check("sigmoid", |g, x, _| g.push(Op::Sigmoid, &[x]));
    s.check("softmax", |g, x, _| g.push(Op::Softmax, &[x]));
    s.check("softcap", |g, x, _| g.push(Op::Softcap(1.5), &[x]));
}

#[test]
fn views_and_copies() {
    let mut s = setup(6);
    let v = s.rng.vec(32 * N, 1.0);
    let other = s.w.f32("other", &[N, 32], &v);
    s.check("reshape", |g, x, n| {
        let r = g.push(Op::Reshape { rows: n * 4, cols: 24 }, &[x]);
        let y = g.push(Op::Silu, &[r]);
        g.push(Op::Reshape { rows: n, cols: 96 }, &[y])
    });
    s.check("cols", |g, x, _| g.push(Op::Cols { start: 8, len: 40 }, &[x]));
    s.check("rows", |g, x, n| {
        let r = g.push(Op::Reshape { rows: n * 4, cols: 24 }, &[x]);
        g.push(Op::Rows { start: 1, len: n * 4 - 1 }, &[r])
    });
    s.check("last rows", |g, x, n| {
        let r = g.push(Op::Reshape { rows: n * 4, cols: 24 }, &[x]);
        g.push(Op::LastRows { n: 2 }, &[r])
    });
    s.check("concat", |g, x, _| {
        let tokens = g.input("tokens");
        let y = g.push(Op::Embed { weight: other }, &[tokens]);
        let y = g.push(Op::Cols { start: 0, len: 10 }, &[y]);
        g.concat(x, y)
    });
    s.check("tile", |g, x, _| g.push(Op::Tile { n: 3 }, &[x]));
    s.check("pad cols", |g, x, _| g.push(Op::PadCols { to: 100 }, &[x]));
    s.check("sum rows", |g, x, _| g.push(Op::SumRows, &[x]));
    s.check("mask cols", |g, x, _| g.push(Op::MaskCols { start: 3, len: 5, value: -7.5 }, &[x]));
    s.check("fill", |g, _, n| g.push(Op::Fill { rows: n, cols: 12, value: 0.25 }, &[]));
}

#[test]
fn broadcasts_and_unary() {
    let mut s = setup(7);
    let v = s.rng.vec(32 * N, 1.0);
    let other = s.w.f32("other", &[N, 32], &v);
    let wv = s.rng.vec(N, 1.0);
    let row = s.w.f32("roww", &[N], &wv);
    let one = s.w.f32("one", &[1], &[0.75]);
    for (name, op) in [("sub", Op::Sub), ("div", Op::Div), ("mul", Op::Mul), ("add", Op::Add)] {
        s.check(&format!("{name} same shape"), |g, x, _| {
            let tokens = g.input("tokens");
            let y = g.push(Op::Embed { weight: other }, &[tokens]);
            let y = g.push(Op::ScaleBias { scale: 1.0, bias: 2.0 }, &[y]);
            g.push(op.clone(), &[x, y])
        });
        s.check(&format!("{name} column broadcast"), |g, x, _| {
            let col = g.push(Op::SumRows, &[x]);
            let col = g.push(Op::ScaleBias { scale: 0.1, bias: 3.0 }, &[col]);
            g.push(op.clone(), &[x, col])
        });
        s.check(&format!("{name} row broadcast"), |g, x, _| {
            let r = g.push(Op::RowBroadcast { weight: row, rows: 1 }, &[]);
            let r = g.push(Op::ScaleBias { scale: 1.0, bias: 2.0 }, &[r]);
            g.push(op.clone(), &[x, r])
        });
    }
    s.check("add weight", |g, x, _| g.push(Op::AddWeight { weight: row }, &[x]));
    s.check("div weight", |g, x, _| {
        let y = g.push(Op::ScaleBias { scale: 1.0, bias: 3.0 }, &[x]);
        g.push(Op::DivWeight { weight: row }, &[y])
    });
    s.check("mul scalar weight", |g, x, _| g.push(Op::MulWeight { weight: one }, &[x]));
    let unaries = [
        ("tanh", Op::Tanh),
        ("exp", Op::Exp),
        ("sqr", Op::Sqr),
        ("abs", Op::Abs),
        ("sign", Op::Sign),
        ("round", Op::Round),
        ("softplus", Op::Softplus { beta: 1.5 }),
        ("clamp", Op::Clamp { min: -0.4, max: 0.6 }),
        ("scale bias", Op::ScaleBias { scale: 2.0, bias: -0.5 }),
        ("gelu erf", Op::GeluErf),
        ("relu sqr", Op::ReluSqr),
    ];
    for (name, op) in unaries {
        s.check(name, |g, x, _| g.push(op.clone(), &[x]));
    }
    s.check("sqrt", |g, x, _| {
        let y = g.push(Op::Abs, &[x]);
        g.push(Op::Sqrt, &[y])
    });
    s.check("log", |g, x, _| {
        let y = g.push(Op::Abs, &[x]);
        let y = g.push(Op::ScaleBias { scale: 1.0, bias: 0.5 }, &[y]);
        g.push(Op::Log, &[y])
    });
    s.check("gated gelu erf", |g, x, _| {
        let tokens = g.input("tokens");
        let y = g.push(Op::Embed { weight: other }, &[tokens]);
        g.push(Op::GatedAct { act: Act::GeluErf, limit: None }, &[x, y])
    });
}

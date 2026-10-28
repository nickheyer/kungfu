//! Small graph helpers shared by the builders
use kf_core::{GraphBuilder, Op, RopeParams, Tid, Wid};

/// Rows copies of a weight vector, for weights used as per token operands
pub fn row_weight(g: &mut GraphBuilder, w: Wid, rows: usize) -> Tid {
    g.push(Op::RowBroadcast { weight: w, rows }, &[])
}

/// One element of a weight vector as a column of rows copies
pub fn weight_scalar(g: &mut GraphBuilder, w: Wid, index: usize, rows: usize) -> Tid {
    let r = row_weight(g, w, rows);
    g.cols(r, index, 1)
}

pub fn fill(g: &mut GraphBuilder, rows: usize, cols: usize, value: f32) -> Tid {
    g.push(Op::Fill { rows, cols, value }, &[])
}

/// Rms norm over each head's slice of a [tokens, heads * dim] tensor with one shared weight or none
pub fn per_head_rms(g: &mut GraphBuilder, x: Tid, n_tokens: usize, n_head: usize, dim: usize, w: Option<Wid>, eps: f32) -> Tid {
    let r = g.reshape(x, n_tokens * n_head, dim);
    let n = g.push(Op::RmsNorm { weight: w, bias: None, eps }, &[r]);
    g.reshape(n, n_tokens, n_head * dim)
}

pub fn per_head_l2(g: &mut GraphBuilder, x: Tid, n_tokens: usize, n_head: usize, dim: usize, eps: f32) -> Tid {
    let r = g.reshape(x, n_tokens * n_head, dim);
    let n = g.push(Op::L2Norm { eps }, &[r]);
    g.reshape(n, n_tokens, n_head * dim)
}

/// Rms over each stream of a [tokens, n * width] tensor, then a weight holding one row per stream
pub fn grouped_rms(g: &mut GraphBuilder, x: Tid, n_tokens: usize, n: usize, width: usize, w: Option<Wid>, eps: f32) -> Tid {
    let r = g.reshape(x, n_tokens * n, width);
    let nrm = g.push(Op::RmsNorm { weight: None, bias: None, eps }, &[r]);
    let back = g.reshape(nrm, n_tokens, n * width);
    match w {
        Some(w) => g.push(Op::MulWeight { weight: w }, &[back]),
        None => back,
    }
}

/// Per head concat of two head major tensors of widths da and db per head
pub fn concat_heads(g: &mut GraphBuilder, a: Tid, b: Tid, n_tokens: usize, n_head: usize, da: usize, db: usize) -> Tid {
    let ra = g.reshape(a, n_tokens * n_head, da);
    let rb = g.reshape(b, n_tokens * n_head, db);
    let c = g.concat(ra, rb);
    g.reshape(c, n_tokens, n_head * (da + db))
}

/// Columns of a head major tensor restricted to one slice of every head
pub fn head_cols(g: &mut GraphBuilder, x: Tid, n_tokens: usize, n_head: usize, dim: usize, start: usize, len: usize) -> Tid {
    let r = g.reshape(x, n_tokens * n_head, dim);
    let c = g.cols(r, start, len);
    g.reshape(c, n_tokens, n_head * len)
}

/// Each head of x multiplied by one scalar of a weight holding n_head values
pub fn mul_per_head(g: &mut GraphBuilder, x: Tid, w: Wid, n_tokens: usize, n_head: usize, dim: usize) -> Tid {
    let r = g.reshape(x, n_tokens * n_head, dim);
    let s = row_weight(g, w, n_tokens);
    let s = g.reshape(s, n_tokens * n_head, 1);
    let m = g.mul(r, s);
    g.reshape(m, n_tokens, n_head * dim)
}

/// Each head of x multiplied by a per token per head column tensor of n_head columns
pub fn mul_head_cols(g: &mut GraphBuilder, x: Tid, s: Tid, n_tokens: usize, n_head: usize, dim: usize) -> Tid {
    let r = g.reshape(x, n_tokens * n_head, dim);
    let sc = g.reshape(s, n_tokens * n_head, 1);
    let m = g.mul(r, sc);
    g.reshape(m, n_tokens, n_head * dim)
}

/// Heads of a key head major tensor repeated so value head j reads key head floor(j / r)
pub fn repeat_heads(g: &mut GraphBuilder, x: Tid, n_tokens: usize, n_head: usize, dim: usize, r: usize) -> Tid {
    if r <= 1 {
        return x;
    }
    let rs = g.reshape(x, n_tokens * n_head, dim);
    let t = g.push(Op::Tile { n: r }, &[rs]);
    g.reshape(t, n_tokens, n_head * r * dim)
}

pub fn split_streams(g: &mut GraphBuilder, x: Tid, n: usize, width: usize) -> Vec<Tid> {
    (0..n).map(|s| g.cols(x, s * width, width)).collect()
}

pub fn join(g: &mut GraphBuilder, parts: &[Tid]) -> Tid {
    let mut it = parts.iter();
    let mut acc = *it.next().expect("join of no parts");
    for p in it {
        acc = g.concat(acc, *p);
    }
    acc
}

/// Per token magnitude sqrt of the squared sum as a one column tensor
pub fn magnitude(g: &mut GraphBuilder, x: Tid) -> Tid {
    let s = g.push(Op::Sqr, &[x]);
    let s = g.push(Op::SumRows, &[s]);
    g.push(Op::Sqrt, &[s])
}

pub fn rope(g: &mut GraphBuilder, x: Tid, p: &RopeParams, n_head: usize, head_dim: usize, offset: usize, positions: Tid) -> Tid {
    g.push(Op::Rope { params: p.clone(), n_head, head_dim, offset, inverse: false }, &[x, positions])
}

pub fn unary(g: &mut GraphBuilder, op: Op, x: Tid) -> Tid {
    g.push(op, &[x])
}

pub fn add_weight(g: &mut GraphBuilder, x: Tid, w: Wid) -> Tid {
    g.push(Op::AddWeight { weight: w }, &[x])
}

pub fn mul_weight(g: &mut GraphBuilder, x: Tid, w: Wid) -> Tid {
    g.push(Op::MulWeight { weight: w }, &[x])
}

pub fn matmul(g: &mut GraphBuilder, x: Tid, w: Wid) -> Tid {
    g.push(Op::MatMul { weight: w, bias: None, scale: None }, &[x])
}

pub fn rms(g: &mut GraphBuilder, x: Tid, w: Option<Wid>, eps: f32) -> Tid {
    g.push(Op::RmsNorm { weight: w, bias: None, eps }, &[x])
}

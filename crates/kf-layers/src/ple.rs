use anyhow::Result;
use kf_core::{Ctx, GraphBuilder, Layer, StateNeed, Tid, Wid};

/// Per layer n gram embedding read through a hashed key table, Qwen style over the streams
pub struct Ple {
    pub ngram_size: usize,
    pub heads_per_ngram: usize,
    pub head_dim: usize,
    pub n_heads: usize,
    pub n_streams: usize,
    pub n_embd: usize,
    pub conv_kernel: usize,
    pub multipliers: Vec<u64>,
    pub head_offsets: Vec<u32>,
    pub head_vocab_sizes: Vec<u32>,
    pub eos_token: u32,
    pub image_token: Option<u32>,
    pub eps: f32,
    pub table: Wid,
    pub key: Wid,
    pub value: Wid,
    pub norm_key: Wid,
    pub norm_query: Wid,
    pub norm_conv: Wid,
    pub conv1d: Wid,
}

impl Layer for Ple {
    fn forward(&self, g: &mut GraphBuilder, x: Tid, ctx: &Ctx) -> Result<Tid> {
        ple_ngram_graph(g, self, x, ctx)
    }
    fn weights(&self) -> Vec<Wid> {
        vec![self.table, self.key, self.value, self.norm_key, self.norm_query, self.norm_conv, self.conv1d]
    }
    fn states(&self) -> Vec<StateNeed> {
        vec![StateNeed::TokenHistory { n: self.ngram_size - 1 }, StateNeed::Recurrent { conv_state: (self.conv_kernel - 1) * self.ngram_size * self.n_streams * self.n_embd, ssm_state: 0 }]
    }
}

/// Hashed n gram memory of DeepSeek V4.1 with a token map, one module per listed layer
pub struct Engram {
    pub ngram_size: usize,
    pub heads_per_ngram: usize,
    pub head_dim: usize,
    pub n_streams: usize,
    pub n_embd: usize,
    pub multipliers: Vec<u64>,
    pub offsets: Vec<u32>,
    pub primes: Vec<u32>,
    pub token_map: Vec<u32>,
    pub pad_token: u32,
    pub eps: f32,
    pub table: Wid,
    pub kv: Wid,
    pub gate: Wid,
}

impl Layer for Engram {
    fn forward(&self, g: &mut GraphBuilder, x: Tid, ctx: &Ctx) -> Result<Tid> {
        engram_graph(g, self, x, ctx)
    }
    fn weights(&self) -> Vec<Wid> {
        vec![self.table, self.kv, self.gate]
    }
    fn states(&self) -> Vec<StateNeed> {
        vec![StateNeed::TokenHistory { n: self.ngram_size - 1 }]
    }
}

use crate::util::{grouped_rms, mul_weight, rms, unary};
use kf_core::Op;
use std::sync::Arc;

/// Signed square root sigmoid gate of a dot score column
fn sign_sqrt_gate(g: &mut GraphBuilder, score: Tid) -> Tid {
    let a = unary(g, Op::Abs, score);
    let a = g.push(Op::Clamp { min: 1e-6, max: f32::INFINITY }, &[a]);
    let a = unary(g, Op::Sqrt, a);
    let s = unary(g, Op::Sign, score);
    let v = g.mul(s, a);
    unary(g, Op::Sigmoid, v)
}

/// Hashed n gram rows gathered, keyed against the streams and injected through a dilated conv
pub fn ple_ngram_graph(g: &mut GraphBuilder, p: &Ple, x: Tid, ctx: &Ctx) -> Result<Tid> {
    let t = ctx.n_tokens;
    let (n, e) = (p.n_streams, p.n_embd);
    let rows = g.push(
        Op::PleHash {
            layer: ctx.layer,
            ngram: p.ngram_size,
            heads_per_ngram: p.heads_per_ngram,
            multipliers: Arc::new(p.multipliers.clone()),
            offsets: Arc::new(p.head_offsets.clone()),
            vocab: Arc::new(p.head_vocab_sizes.clone()),
            eos: p.eos_token,
            image: p.image_token,
        },
        &[ctx.tokens],
    );
    let emb = g.push(Op::Gather { weight: p.table }, &[rows]);
    let key = g.push(Op::MatMul { weight: p.key, bias: None, scale: None }, &[emb]);
    let value = g.push(Op::MatMul { weight: p.value, bias: None, scale: None }, &[emb]);
    let ks = grouped_rms(g, key, t, n, e, Some(p.norm_key), p.eps);
    let qs = grouped_rms(g, x, t, n, e, Some(p.norm_query), p.eps);
    let dot = g.mul(ks, qs);
    let dot = g.reshape(dot, t * n, e);
    let score = g.push(Op::SumRows, &[dot]);
    let score = g.scale(score, 1.0 / (e as f32).sqrt());
    let gate = sign_sqrt_gate(g, score);
    let vt = g.push(Op::Tile { n }, &[value]);
    let vt = g.reshape(vt, t * n, e);
    let gated = g.mul(vt, gate);
    let nrm = rms(g, gated, None, p.eps);
    let nrm = g.reshape(nrm, t, n * e);
    let nrm = mul_weight(g, nrm, p.norm_conv);
    let conv = g.push(Op::CausalConv { layer: ctx.layer, slot: 1, kernel: p.conv_kernel, dilation: p.ngram_size, weight: p.conv1d, bias: None }, &[nrm]);
    let conv = unary(g, Op::Silu, conv);
    let gated = g.reshape(gated, t, n * e);
    let y = g.add(x, gated);
    let _ = (p.head_dim, p.n_heads);
    Ok(g.add(y, conv))
}

/// Engram memory, hashed rows gathered, split into per stream keys and a value, gated into each stream
pub fn engram_graph(g: &mut GraphBuilder, en: &Engram, x: Tid, ctx: &Ctx) -> Result<Tid> {
    let t = ctx.n_tokens;
    let (n, e) = (en.n_streams, en.n_embd);
    let rows = g.push(
        Op::EngramHash {
            layer: ctx.layer,
            ngram: en.ngram_size,
            heads_per_ngram: en.heads_per_ngram,
            multipliers: Arc::new(en.multipliers.clone()),
            offsets: Arc::new(en.offsets.clone()),
            primes: Arc::new(en.primes.clone()),
            token_map: Arc::new(en.token_map.clone()),
            pad: en.pad_token,
        },
        &[ctx.tokens],
    );
    let emb = g.push(Op::Gather { weight: en.table }, &[rows]);
    let kv = g.push(Op::MatMul { weight: en.kv, bias: None, scale: None }, &[emb]);
    let keys = g.cols(kv, 0, n * e);
    let value = g.cols(kv, n * e, e);
    let xs = grouped_rms(g, x, t, n, e, None, en.eps);
    let ks = grouped_rms(g, keys, t, n, e, None, en.eps);
    let prod = g.mul(xs, ks);
    let prod = mul_weight(g, prod, en.gate);
    let prod = g.reshape(prod, t * n, e);
    let dot = g.push(Op::SumRows, &[prod]);
    let dot = g.scale(dot, 1.0 / (e as f32).sqrt());
    let gate = sign_sqrt_gate(g, dot);
    let vt = g.push(Op::Tile { n }, &[value]);
    let vt = g.reshape(vt, t * n, e);
    let add = g.mul(vt, gate);
    let add = g.reshape(add, t, n * e);
    let _ = en.head_dim;
    Ok(g.add(x, add))
}

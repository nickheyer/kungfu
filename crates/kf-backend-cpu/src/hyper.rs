use crate::{Buf, CpuBackend, ops};
use anyhow::{Result, bail};
use kf_core::{Inputs, Op, WeightSource};

fn need(ins: &[Buf], n: usize) -> Result<()> {
    if ins.len() < n {
        bail!("{} inputs for an op needing {n}", ins.len());
    }
    Ok(())
}

fn streams(x: &Buf, n: usize) -> Result<usize> {
    if n == 0 || !x.cols.is_multiple_of(n) {
        bail!("{} columns do not split into {n} streams", x.cols);
    }
    Ok(x.cols / n)
}

/// Weighted sum of the streams by one weight per stream and token
fn hc_mix(x: &Buf, w: &Buf, n: usize) -> Result<Buf> {
    let e = streams(x, n)?;
    if w.rows != x.rows || w.cols != n {
        bail!("stream weights {}x{} for {} tokens of {n} streams", w.rows, w.cols, x.rows);
    }
    let mut out = Buf::new(x.rows, e);
    for t in 0..x.rows {
        let o = out.row_mut(t);
        for s in 0..n {
            let ws = w.row(t)[s];
            for (o, v) in o.iter_mut().zip(&x.row(t)[s * e..(s + 1) * e]) {
                *o += ws * v;
            }
        }
    }
    Ok(out)
}

/// Sum of the streams gated per element by the sigmoid of the gates, then scaled
fn hc_mix_gated(x: &Buf, g: &Buf, n: usize, scale: f32) -> Result<Buf> {
    let e = streams(x, n)?;
    if g.rows != x.rows || g.cols != x.cols {
        bail!("stream gates {}x{} for streams {}x{}", g.rows, g.cols, x.rows, x.cols);
    }
    let mut out = Buf::new(x.rows, e);
    for t in 0..x.rows {
        let o = out.row_mut(t);
        for s in 0..n {
            for i in 0..e {
                o[i] += x.row(t)[s * e + i] * ops::sigmoid(g.row(t)[s * e + i]);
            }
        }
        o.iter_mut().for_each(|v| *v *= scale);
    }
    Ok(out)
}

/// Sinkhorn normalized combination matrices, destination d of source s at column d plus n times s
fn hc_sinkhorn(c0: &Buf, n: usize, iters: usize, eps: f32) -> Result<Buf> {
    if c0.cols != n * n {
        bail!("combination of {} columns for {n} streams", c0.cols);
    }
    let mut out = c0.clone();
    let norm_cols = |c: &mut [f32]| {
        for d in 0..n {
            let sum = eps + (0..n).map(|s| c[d + n * s]).sum::<f32>();
            let inv = 1.0 / sum;
            for s in 0..n {
                c[d + n * s] *= inv;
            }
        }
    };
    let norm_rows = |c: &mut [f32]| {
        for s in 0..n {
            let sum = eps + (0..n).map(|d| c[d + n * s]).sum::<f32>();
            let inv = 1.0 / sum;
            for d in 0..n {
                c[d + n * s] *= inv;
            }
        }
    };
    for t in 0..out.rows {
        let c = out.row_mut(t);
        for s in 0..n {
            let m = (0..n).map(|d| c[d + n * s]).fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0;
            for d in 0..n {
                c[d + n * s] = (c[d + n * s] - m).exp();
                sum += c[d + n * s];
            }
            for d in 0..n {
                c[d + n * s] = c[d + n * s] / sum + eps;
            }
        }
        norm_cols(c);
        for _ in 1..iters.max(1) {
            norm_rows(c);
            norm_cols(c);
        }
    }
    Ok(out)
}

/// Streams updated by the sublayer output times the post weights plus the combined residual
fn hc_combine(x: &Buf, out: &Buf, post: &Buf, comb: Option<&Buf>, n: usize) -> Result<Buf> {
    let e = streams(x, n)?;
    if out.rows != x.rows || out.cols != e || post.rows != x.rows || post.cols != n {
        bail!("combine of out {}x{} and post {}x{} onto streams {}x{}", out.rows, out.cols, post.rows, post.cols, x.rows, x.cols);
    }
    if let Some(c) = comb
        && (c.rows != x.rows || c.cols != n * n)
    {
        bail!("combination {}x{} for {} tokens of {n} streams", c.rows, c.cols, x.rows);
    }
    let mut y = Buf::new(x.rows, x.cols);
    for t in 0..x.rows {
        for d in 0..n {
            let pd = post.row(t)[d];
            let yd = &mut y.row_mut(t)[d * e..(d + 1) * e];
            for i in 0..e {
                let mut v = out.row(t)[i] * pd;
                match comb {
                    Some(c) => {
                        for s in 0..n {
                            v += x.row(t)[s * e + i] * c.row(t)[d + n * s];
                        }
                    }
                    None => v += x.row(t)[d * e + i],
                }
                yd[i] = v;
            }
        }
    }
    Ok(y)
}

/// Token ids of the sequence before this pass followed by the pass's own tokens
fn history(b: &CpuBackend, inputs: &Inputs) -> Vec<i64> {
    b.tokens().iter().chain(&inputs.tokens).map(|t| *t as i64).collect()
}

/// Hashed n gram row ids per head from the token and its predecessors, an eos cutting the context
#[allow(clippy::too_many_arguments)]
fn ple_hash(hist: &[i64], first: usize, n: usize, ngram: usize, heads_per_ngram: usize, mult: &[u64], offsets: &[u32], vocab: &[u32], eos: u32) -> Result<Buf> {
    let n_heads = (ngram - 1) * heads_per_ngram;
    if mult.len() < ngram || offsets.len() < n_heads || vocab.len() < n_heads {
        bail!("ple tables of {} multipliers, {} offsets, {} vocab sizes for {ngram} grams of {heads_per_ngram} heads", mult.len(), offsets.len(), vocab.len());
    }
    let mut out = Buf::new(n, n_heads);
    let mut ctx = vec![0u64; ngram];
    for i in 0..n {
        let a = first + i;
        ctx[0] = hist[a] as u64;
        let mut cut = false;
        for s in 1..ngram {
            let t = if a >= s { hist[a - s] } else { -1 };
            cut = cut || t < 0 || t == eos as i64;
            ctx[s] = if cut { eos as u64 } else { t as u64 };
        }
        for len in 2..=ngram {
            let mut mixed = 0u64;
            for (j, c) in ctx.iter().take(len).enumerate() {
                mixed ^= c.wrapping_mul(mult[j]);
            }
            for g in 0..heads_per_ngram {
                let h = (len - 2) * heads_per_ngram + g;
                out.row_mut(i)[h] = (mixed % vocab[h] as u64 + offsets[h] as u64) as f32;
            }
        }
    }
    Ok(out)
}

/// Rolling xor hash over mapped token ids, dead shifts padded, one row id per column
#[allow(clippy::too_many_arguments)]
fn engram_hash(hist: &[i64], first: usize, n: usize, ngram: usize, heads_per_ngram: usize, mult: &[u64], offsets: &[u32], primes: &[u32], map: &[u32], pad: u32) -> Result<Buf> {
    let n_cols = (ngram - 1) * heads_per_ngram;
    if mult.len() < ngram || offsets.len() < n_cols || primes.len() < n_cols {
        bail!("engram tables of {} multipliers, {} offsets, {} primes for {ngram} grams of {heads_per_ngram} heads", mult.len(), offsets.len(), primes.len());
    }
    let pad = *map.get(pad as usize).ok_or_else(|| anyhow::anyhow!("engram pad token {pad} outside the token map"))? as u64;
    let mut out = Buf::new(n, n_cols);
    let mut toks = vec![0u64; ngram];
    for i in 0..n {
        let a = first + i;
        let mut dead = false;
        for s in 0..ngram {
            let id = if a >= s { map.get(hist[a - s] as usize).copied() } else { None };
            dead = dead || id.is_none() || hist[a - s.min(a)] < 0;
            toks[s] = if dead { pad } else { id.unwrap_or(pad as u32) as u64 };
        }
        let mut rolling = toks[0].wrapping_mul(mult[0]);
        for o in 1..ngram {
            rolling ^= toks[o].wrapping_mul(mult[o]);
            for h in 0..heads_per_ngram {
                let col = (o - 1) * heads_per_ngram + h;
                out.row_mut(i)[col] = (rolling % primes[col] as u64 + offsets[col] as u64) as f32;
            }
        }
    }
    Ok(out)
}

/// Stream mixing, sinkhorn combination, hashed memories, switched adapters and token maps
pub fn exec(b: &mut CpuBackend, op: &Op, ins: &[Buf], inputs: &Inputs, weights: &dyn WeightSource) -> Result<Buf> {
    match op {
        Op::HcMix { n_streams } => {
            need(ins, 2)?;
            hc_mix(&ins[0], &ins[1], *n_streams)
        }
        Op::HcMixGated { n_streams, scale } => {
            need(ins, 2)?;
            hc_mix_gated(&ins[0], &ins[1], *n_streams, *scale)
        }
        Op::HcSinkhorn { n_streams, iters, eps } => {
            need(ins, 1)?;
            hc_sinkhorn(&ins[0], *n_streams, *iters, *eps)
        }
        Op::HcCombine { n_streams, comb } => {
            need(ins, 3 + usize::from(*comb))?;
            hc_combine(&ins[0], &ins[1], &ins[2], comb.then(|| &ins[3]), *n_streams)
        }
        Op::PleHash { ngram, heads_per_ngram, multipliers, offsets, vocab, eos, .. } => {
            need(ins, 1)?;
            let hist = history(b, inputs);
            let n = ins[0].rows;
            ple_hash(&hist, hist.len() - n, n, *ngram, *heads_per_ngram, multipliers, offsets, vocab, *eos)
        }
        Op::EngramHash { ngram, heads_per_ngram, multipliers, offsets, primes, token_map, pad, .. } => {
            need(ins, 1)?;
            let hist = history(b, inputs);
            let n = ins[0].rows;
            engram_hash(&hist, hist.len() - n, n, *ngram, *heads_per_ngram, multipliers, offsets, primes, token_map, *pad)
        }
        Op::LoraSwitched { a, b: bw } => {
            need(ins, 2)?;
            let x = &ins[0];
            let slot = &ins[1];
            if slot.rows != x.rows {
                bail!("{} adapter slots for {} rows", slot.rows, x.rows);
            }
            let mut out: Option<Buf> = None;
            for r in 0..x.rows {
                let s = slot.row(r)[0] as usize;
                let xr = Buf { data: x.row(r).to_vec(), rows: 1, cols: x.cols };
                let h = ops::matmul_slice(&xr, *a, weights, None, s)?;
                let y = ops::matmul_slice(&h, *bw, weights, None, s)?;
                let o = out.get_or_insert_with(|| Buf::new(x.rows, y.cols));
                o.row_mut(r).copy_from_slice(&y.data);
            }
            Ok(out.unwrap_or_else(|| Buf::new(0, 0)))
        }
        Op::MapTokens { table, default } => {
            need(ins, 1)?;
            let t = &ins[0];
            let data = t.data.iter().map(|id| table.iter().find(|(k, _)| *k as f32 == *id).map_or(default.unwrap_or(*id), |(_, v)| *v)).collect();
            Ok(Buf { data, rows: t.rows, cols: 1 })
        }
        other => bail!("{other:?} routed to the stream module"),
    }
}

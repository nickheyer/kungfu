use crate::recurrent::{SNAPSHOTS, snap};
use crate::{Buf, CpuBackend, KvLayer, LayerState, ops, rope};
use anyhow::{Result, anyhow, bail};
use kf_core::{Candidate, Compressor, IndexKey, IndexerKind, Inputs, Op, PoolKeys, RopeParams, WeightSource};

fn need(ins: &[Buf], n: usize) -> Result<()> {
    if ins.len() < n {
        bail!("{} inputs for an op needing {n}", ins.len());
    }
    Ok(())
}

/// Lightning indexer score of one query over one key, relu of each head's dot weighted per head
fn score(q: &[f32], w: &[f32], k: &[f32], n_head: usize, d: usize) -> f32 {
    let mut s = 0.0f32;
    for h in 0..n_head {
        let mut dot = 0.0f32;
        for i in 0..d {
            dot += q[h * d + i] * k[i];
        }
        s += dot.max(0.0) * w[h];
    }
    s
}

/// Indices of the n largest finite scores, ascending
fn top(scores: &[f32], n: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..scores.len()).filter(|i| scores[*i].is_finite()).collect();
    idx.sort_by(|a, b| scores[*b].partial_cmp(&scores[*a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(b)));
    idx.truncate(n);
    idx.sort_unstable();
    idx
}

fn fill(out: &mut Buf, r: usize, cells: &[usize]) -> Result<()> {
    let row = out.row_mut(r);
    if cells.len() > row.len() {
        bail!("{} selected cells for a row of {}", cells.len(), row.len());
    }
    row.iter_mut().for_each(|v| *v = -1.0);
    for (o, c) in row.iter_mut().zip(cells) {
        *o = *c as f32;
    }
    Ok(())
}

/// Softmax pooled, mean pooled or raw rows written into the pooled slot of a completed pool
fn pool_row(st: &mut LayerState, b: usize, kpool: usize, d: usize, pool: &PoolKeys, ape: Option<&[f32]>, norm: Option<&[f32]>) -> Result<()> {
    let first = b * kpool;
    let mut pooled = vec![0.0f32; d];
    match ape {
        Some(ape) => {
            for c in 0..d {
                let mut m = f32::NEG_INFINITY;
                let mut logits = vec![0.0f32; kpool];
                for p in 0..kpool {
                    let row = st.indexer.row(first + p);
                    logits[p] = row[d + c] + ape[p * d + c];
                    m = m.max(logits[p]);
                }
                let mut sum = 0.0;
                for l in logits.iter_mut() {
                    *l = (*l - m).exp();
                    sum += *l;
                }
                for p in 0..kpool {
                    pooled[c] += logits[p] / sum * st.indexer.row(first + p)[c];
                }
            }
        }
        None => {
            for p in 0..kpool {
                for (o, v) in pooled.iter_mut().zip(&st.indexer.row(first + p)[..d]) {
                    *o += v / kpool as f32;
                }
            }
            let mut buf = Buf { data: pooled, rows: 1, cols: d };
            buf = ops::rms_norm(&buf, norm, None, pool.eps)?;
            if let Some(rp) = &pool.rope {
                buf = rope::apply(&buf, &[rope::text_pos(first as u32, true)], rp, 1, d, d - rp.dims.min(d), false, None)?;
            }
            pooled = buf.data;
        }
    }
    st.indexer.row_mut(first + kpool - 1)[2 * d..3 * d].copy_from_slice(&pooled);
    Ok(())
}

/// Key selection for every query row of the pass, per the indexer kind
#[allow(clippy::too_many_arguments)]
fn indexer_top_k(b: &mut CpuBackend, layer: usize, ins: &[Buf], n_head: usize, d: usize, top_k: usize, kind: IndexerKind, pool: Option<&PoolKeys>, candidate: Option<&Candidate>, weights: &dyn WeightSource) -> Result<Buf> {
    let (q, k, w) = (&ins[0], &ins[1], &ins[2]);
    let t_rows = q.rows;
    if q.cols != n_head * d || (k.cols != d && k.rows > 0) || w.rows != t_rows || w.cols != n_head {
        bail!("indexer shapes q {}x{} k {}x{} w {}x{} for {n_head} heads of {d}", q.rows, q.cols, k.rows, k.cols, w.rows, w.cols);
    }
    let base = b.tokens().len();
    let ape = match pool.and_then(|p| p.ape) {
        Some(a) => Some(b.small(a, weights)?.to_vec()),
        None => None,
    };
    let norm = match pool.and_then(|p| p.norm) {
        Some(n) => Some(b.small(n, weights)?.to_vec()),
        None => None,
    };
    let cand_rows = match candidate.and_then(|c| c.source) {
        Some(src) => Some(b.layers().get(&src).map(|s| s.cand.clone()).ok_or_else(|| anyhow!("layer {src} has no candidate blocks for layer {layer}"))?),
        None => None,
    };
    let st = b.layer(layer);
    let mut out;
    match kind {
        IndexerKind::Dense => {
            if k.rows != t_rows {
                bail!("{} indexer keys for {t_rows} queries", k.rows);
            }
            st.indexer_ratio = 1;
            if st.indexer.rows == 0 {
                st.indexer = Buf::new(0, d);
            }
            st.indexer.data.extend_from_slice(&k.data);
            st.indexer.rows += t_rows;
            let n_before = st.indexer.rows - t_rows;
            out = Buf::new(t_rows, top_k);
            for r in 0..t_rows {
                let pos = n_before + r;
                let scores: Vec<f32> = (0..=pos).map(|j| score(q.row(r), w.row(r), st.indexer.row(j), n_head, d)).collect();
                fill(&mut out, r, &top(&scores, top_k))?;
            }
        }
        IndexerKind::Pool { kpool, tail } => {
            let pool = pool.ok_or_else(|| anyhow!("pooled indexer without pool settings"))?;
            if k.rows != t_rows {
                bail!("{} indexer keys for {t_rows} queries", k.rows);
            }
            if kpool == 0 {
                bail!("indexer pool of size zero");
            }
            let gate = if ape.is_some() {
                need(ins, 4)?;
                Some(&ins[3])
            } else {
                None
            };
            st.indexer_ratio = 1;
            if st.indexer.rows == 0 {
                st.indexer = Buf::new(0, 3 * d);
            }
            let n_before = st.indexer.rows;
            for r in 0..t_rows {
                let mut row = vec![0.0f32; 3 * d];
                row[..d].copy_from_slice(k.row(r));
                if let Some(g) = gate {
                    row[d..2 * d].copy_from_slice(g.row(r));
                }
                st.indexer.data.extend_from_slice(&row);
                st.indexer.rows += 1;
                let pos = n_before + r;
                if (pos + 1).is_multiple_of(kpool) {
                    pool_row(st, (pos + 1) / kpool - 1, kpool, d, pool, ape.as_deref(), norm.as_deref())?;
                }
            }
            let n_top = (top_k / kpool).max(1);
            out = Buf::new(t_rows, top_k + kpool - 1);
            for r in 0..t_rows {
                let pos = n_before + r;
                let n_pool = (pos + 1) / kpool;
                let scores: Vec<f32> = (0..n_pool).map(|p| score(q.row(r), w.row(r), &st.indexer.row(p * kpool + kpool - 1)[2 * d..3 * d], n_head, d)).collect();
                let chosen = top(&scores, n_top);
                let mut cells: Vec<usize> = chosen.iter().flat_map(|p| p * kpool..(p + 1) * kpool).collect();
                if tail != 0 {
                    cells.extend(n_pool * kpool..=pos);
                }
                cells.sort_unstable();
                fill(&mut out, r, &cells)?;
            }
        }
        IndexerKind::Compressed { ratio, .. } => {
            if ratio == 0 {
                bail!("compressed indexer with ratio zero");
            }
            st.indexer_ratio = ratio;
            if st.indexer.rows == 0 {
                st.indexer = Buf::new(0, d);
            }
            if k.rows > 0 {
                st.indexer.data.extend_from_slice(&k.data);
                st.indexer.rows += k.rows;
            }
            out = Buf::new(t_rows, top_k);
            let mut cand_out = candidate.filter(|c| c.source.is_none()).map(|c| Buf::new(t_rows, c.top_blocks));
            for r in 0..t_rows {
                let pos = base + r;
                let n_vis = ((pos + 1) / ratio).min(st.indexer.rows);
                let mut scores: Vec<f32> = (0..n_vis).map(|j| score(q.row(r), w.row(r), st.indexer.row(j), n_head, d)).collect();
                if let Some(c) = candidate {
                    let block = c.block.max(1);
                    match (&cand_rows, &mut cand_out) {
                        (Some(rows), _) => {
                            let allowed: Vec<usize> = rows.row(r).iter().filter(|v| **v >= 0.0).map(|v| *v as usize).collect();
                            for (j, s) in scores.iter_mut().enumerate() {
                                if !allowed.contains(&(j / block)) {
                                    *s = f32::NEG_INFINITY;
                                }
                            }
                        }
                        (None, Some(co)) => {
                            let n_blk = n_vis.div_ceil(block);
                            let mut bs: Vec<f32> = (0..n_blk).map(|g| scores[g * block..((g + 1) * block).min(n_vis)].iter().copied().fold(f32::NEG_INFINITY, f32::max)).collect();
                            if n_vis > 0 {
                                bs[(n_vis - 1) / block] += 1e30;
                            }
                            fill(co, r, &top(&bs, c.top_blocks))?;
                        }
                        (None, None) => {}
                    }
                }
                fill(&mut out, r, &top(&scores, top_k))?;
            }
            if let Some(co) = cand_out {
                st.cand = co;
            }
        }
        IndexerKind::Block { block, local } => {
            if k.rows != t_rows {
                bail!("{} indexer keys for {t_rows} queries", k.rows);
            }
            let block = block.max(1);
            st.indexer_ratio = 1;
            if st.indexer.rows == 0 {
                st.indexer = Buf::new(0, d);
            }
            st.indexer.data.extend_from_slice(&k.data);
            st.indexer.rows += t_rows;
            let n_before = st.indexer.rows - t_rows;
            out = Buf::new(t_rows, n_head * top_k * block);
            for r in 0..t_rows {
                let pos = n_before + r;
                let n_blk = (pos + 1).div_ceil(block);
                let mut cells = Vec::new();
                for g in 0..n_head {
                    let qg = &q.row(r)[g * d..(g + 1) * d];
                    let mut bs = vec![f32::NEG_INFINITY; n_blk];
                    for j in 0..=pos {
                        let kj = st.indexer.row(j);
                        let dot: f32 = qg.iter().zip(kj).map(|(a, c)| a * c).sum();
                        bs[j / block] = bs[j / block].max(dot);
                    }
                    let cur = pos / block;
                    for g2 in cur.saturating_sub(local.saturating_sub(1))..=cur {
                        bs[g2] += 1e30;
                    }
                    let mut sel: Vec<usize> = top(&bs, top_k).iter().flat_map(|g2| g2 * block..((g2 + 1) * block).min(pos + 1)).collect();
                    sel.sort_unstable();
                    sel.resize(top_k * block, usize::MAX);
                    cells.extend(sel);
                }
                let row = out.row_mut(r);
                for (o, c) in row.iter_mut().zip(&cells) {
                    *o = if *c == usize::MAX { -1.0 } else { *c as f32 };
                }
            }
        }
    }
    st.topk = out.clone();
    Ok(out)
}

/// One completed block pooled from the ring rows into a normed, roped compressed row
fn compress_block(ring_kv: &Buf, ring_score: &Buf, ring_start: usize, block: usize, comp: &Compressor, d: usize, norm: &[f32], _rp: &RopeParams) -> Result<Buf> {
    let ratio = comp.ratio;
    let rows: Vec<(i64, usize)> = if comp.overlap {
        (0..2 * ratio).map(|j| (block as i64 * ratio as i64 - ratio as i64 + j as i64, if j < ratio { 0 } else { d })).collect()
    } else {
        (0..ratio).map(|j| ((block * ratio + j) as i64, 0)).collect()
    };
    let mut out = vec![0.0f32; d];
    let mut logits = vec![0.0f32; rows.len()];
    for c in 0..d {
        let mut m = f32::NEG_INFINITY;
        for (n, (p, half)) in rows.iter().enumerate() {
            logits[n] = if *p < 0 {
                f32::NEG_INFINITY
            } else {
                let i = *p as usize - ring_start;
                ring_score.row(i)[half + c]
            };
            m = m.max(logits[n]);
        }
        let mut sum = 0.0;
        for l in logits.iter_mut() {
            *l = (*l - m).exp();
            sum += *l;
        }
        for (n, (p, half)) in rows.iter().enumerate() {
            if *p >= 0 {
                let i = *p as usize - ring_start;
                out[c] += logits[n] / sum * ring_kv.row(i)[half + c];
            }
        }
    }
    let buf = Buf { data: out, rows: 1, cols: d };
    ops::rms_norm(&buf, Some(norm), None, comp.eps)
}

/// Rope over the trailing dims of one row at a block start position
fn rope_tail(row: &Buf, rp: &RopeParams, pos: usize) -> Result<Buf> {
    let d = row.cols;
    let dims = rp.dims.min(d);
    rope::apply(row, &[rope::text_pos(pos as u32, false)], rp, 1, d, d - dims, false, None)
}

/// Pools each block the pass completes and appends the rows to the slot's compressed cache
#[allow(clippy::too_many_arguments)]
fn compress_kv(st: &mut LayerState, slot: usize, comp: &Compressor, rp: &RopeParams, d: usize, kv: &Buf, sc: &Buf, ape: Option<&[f32]>, norm: &[f32], index: Option<(&IndexKey, &[f32])>, weights: &dyn WeightSource, base: usize) -> Result<Buf> {
    let width = if comp.overlap { 2 * d } else { d };
    if comp.ratio == 0 || kv.cols != width || sc.cols != width || sc.rows != kv.rows {
        bail!("compressor rows kv {}x{} score {}x{} for ratio {} over {d}", kv.rows, kv.cols, sc.rows, sc.cols, comp.ratio);
    }
    let keep = if comp.overlap { 2 * comp.ratio } else { comp.ratio };
    st.ring_kv.entry(slot).or_insert_with(|| Buf::new(0, width));
    st.ring_score.entry(slot).or_insert_with(|| Buf::new(0, width));
    st.compressed.entry(slot).or_insert_with(|| Buf::new(0, d));
    st.ring_start.entry(slot).or_insert(0);
    let mut out = Buf::new(0, d);
    for t in 0..kv.rows {
        let rk = st.ring_kv.get_mut(&slot).ok_or_else(|| anyhow!("ring slot {slot}"))?;
        let rs = st.ring_score.get_mut(&slot).ok_or_else(|| anyhow!("ring slot {slot}"))?;
        let compressed = st.compressed.get_mut(&slot).ok_or_else(|| anyhow!("ring slot {slot}"))?;
        let start = st.ring_start.get_mut(&slot).ok_or_else(|| anyhow!("ring slot {slot}"))?;
        let pos = base + t;
        let mut srow = sc.row(t).to_vec();
        if let Some(a) = ape {
            let col = pos % comp.ratio;
            srow.iter_mut().zip(&a[col * width..(col + 1) * width]).for_each(|(s, a)| *s += a);
        }
        rk.data.extend_from_slice(kv.row(t));
        rk.rows += 1;
        rs.data.extend_from_slice(&srow);
        rs.rows += 1;
        if (pos + 1).is_multiple_of(comp.ratio) {
            let block = (pos + 1) / comp.ratio - 1;
            let normed = compress_block(rk, rs, *start, block, comp, d, norm, rp)?;
            let row = rope_tail(&normed, rp, block * comp.ratio)?.data;
            compressed.data.extend_from_slice(&row);
            compressed.rows += 1;
            out.data.extend_from_slice(&row);
            out.rows += 1;
            if let Some((ik, ik_norm)) = index {
                let key = ops::matmul(&normed, ik.k, weights, None)?;
                let key = ops::rms_norm(&key, Some(ik_norm), None, ik.eps)?;
                let key = rope_tail(&key, &ik.rope, block * comp.ratio)?;
                if st.indexer.rows == 0 {
                    st.indexer = Buf::new(0, ik.head_dim);
                }
                st.indexer_ratio = comp.ratio;
                st.indexer.data.extend_from_slice(&key.data);
                st.indexer.rows += 1;
            }
        }
        let rk = st.ring_kv.get_mut(&slot).ok_or_else(|| anyhow!("ring slot {slot}"))?;
        let rs = st.ring_score.get_mut(&slot).ok_or_else(|| anyhow!("ring slot {slot}"))?;
        let start = st.ring_start.get_mut(&slot).ok_or_else(|| anyhow!("ring slot {slot}"))?;
        if rk.rows > keep {
            let drop = rk.rows - keep;
            rk.data.drain(..drop * width);
            rs.data.drain(..drop * width);
            rk.rows -= drop;
            rs.rows -= drop;
            *start += drop;
        }
        if t + SNAPSHOTS >= kv.rows {
            let (rk2, rs2, s2) = (rk.clone(), rs.clone(), *start);
            snap(st, &format!("ring_kv:{slot}"), pos + 1, rk2);
            snap(st, &format!("ring_score:{slot}"), pos + 1, rs2);
            snap(st, &format!("ring_start:{slot}"), pos + 1, Buf { data: vec![s2 as f32], rows: 1, cols: 1 });
            return compress_kv_tail(st, slot, comp, rp, d, kv, sc, ape, norm, index, weights, base, t + 1, out);
        }
    }
    Ok(out)
}

/// Continues the compressor loop after a snapshot released the borrows
#[allow(clippy::too_many_arguments)]
fn compress_kv_tail(st: &mut LayerState, slot: usize, comp: &Compressor, rp: &RopeParams, d: usize, kv: &Buf, sc: &Buf, ape: Option<&[f32]>, norm: &[f32], index: Option<(&IndexKey, &[f32])>, weights: &dyn WeightSource, base: usize, from: usize, mut out: Buf) -> Result<Buf> {
    if from >= kv.rows {
        return Ok(out);
    }
    let rest_kv = ops::rows(kv, from, kv.rows - from)?;
    let rest_sc = ops::rows(sc, from, sc.rows - from)?;
    let more = compress_kv(st, slot, comp, rp, d, &rest_kv, &rest_sc, ape, norm, index, weights, base + from)?;
    out.data.extend_from_slice(&more.data);
    out.rows += more.rows;
    Ok(out)
}

/// Attention over the raw window plus the visible compressed rows, K equal to V, with sinks and rope back
#[allow(clippy::too_many_arguments)]
fn compressed_attention(cache: &KvLayer, comp: Option<&Buf>, q: &Buf, n_head: usize, d: usize, scale: f32, window: usize, sinks: &[f32], rp: &RopeParams, ratio: usize, topk: Option<&Buf>) -> Result<Buf> {
    if q.cols != n_head * d || cache.k_width != d || sinks.len() < n_head {
        bail!("compressed attention shapes q {} cache {} sinks {} for {n_head} heads of {d}", q.cols, cache.k_width, sinks.len());
    }
    let n_before = cache.n - q.rows;
    let mut out = Buf::new(q.rows, n_head * d);
    let mut scores: Vec<f32> = Vec::new();
    for r in 0..q.rows {
        let pos = n_before + r;
        let first = pos.saturating_sub(window.max(1) - 1);
        let raw: Vec<f32> = (first..=pos).flat_map(|j| cache.k_row(j).iter().map(|v| v.to_f32())).collect();
        let mut keys: Vec<&[f32]> = raw.chunks_exact(d).collect();
        if let Some(c) = comp
            && ratio > 0
        {
            let n_vis = ((pos + 1) / ratio).min(c.rows);
            match topk {
                Some(t) => keys.extend(t.row(r).iter().filter(|v| **v >= 0.0).map(|v| *v as usize).filter(|b| *b < n_vis).map(|b| c.row(b))),
                None => keys.extend((0..n_vis).map(|b| c.row(b))),
            }
        }
        for h in 0..n_head {
            let qh = &q.row(r)[h * d..(h + 1) * d];
            scores.clear();
            for k in &keys {
                scores.push(qh.iter().zip(*k).map(|(a, c)| a * c).sum::<f32>() * scale);
            }
            let m = scores.iter().copied().fold(sinks[h], f32::max);
            let mut sum = (sinks[h] - m).exp();
            for s in scores.iter_mut() {
                *s = (*s - m).exp();
                sum += *s;
            }
            let o = &mut out.row_mut(r)[h * d..(h + 1) * d];
            for (p, k) in scores.iter().zip(&keys) {
                let p = p / sum;
                for i in 0..d {
                    o[i] += p * k[i];
                }
            }
        }
    }
    let positions: Vec<[f32; 4]> = (0..q.rows).map(|r| rope::text_pos((n_before + r) as u32, false)).collect();
    let dims = rp.dims.min(d);
    rope::apply(&out, &positions, rp, n_head, d, d - dims, true, None)
}

/// Indexer selection, kv compression and the compressed attention of DeepSeek V4
pub fn exec(b: &mut CpuBackend, op: &Op, ins: &[Buf], _inputs: &Inputs, weights: &dyn WeightSource) -> Result<Buf> {
    match op {
        Op::IndexerTopK { layer, n_head, head_dim, top_k, kind, source, keys_from, pool, candidate } => {
            if let Some(src) = source {
                return b.layers().get(src).map(|s| s.topk.clone()).filter(|t| t.rows > 0).ok_or_else(|| anyhow!("layer {src} has no selection for layer {layer} to reuse"));
            }
            need(ins, 3)?;
            if let Some(from) = keys_from {
                let keys = b.layers().get(from).map(|s| (s.indexer.clone(), s.indexer_ratio)).ok_or_else(|| anyhow!("layer {from} has no indexer keys for layer {layer}"))?;
                let st = b.layer(*layer);
                st.indexer = keys.0;
                st.indexer_ratio = keys.1;
            }
            indexer_top_k(b, *layer, ins, *n_head, *head_dim, *top_k, *kind, pool.as_ref(), candidate.as_ref(), weights)
        }
        Op::CompressKv { layer, slot, comp, rope, head_dim, index } => {
            need(ins, 2)?;
            let ape = b.small_opt(comp.ape, weights)?;
            let norm = b.small(comp.norm, weights)?.to_vec();
            let ik_norm = match index {
                Some(ik) => Some(b.small(ik.norm, weights)?.to_vec()),
                None => None,
            };
            let base = b.tokens().len();
            let index = index.as_ref().zip(ik_norm.as_deref());
            compress_kv(b.layer(*layer), *slot, comp, rope, *head_dim, &ins[0], &ins[1], ape.as_deref(), &norm, index, weights, base)
        }
        Op::CompressedAttention { layer, n_head, head_dim, scale, window, sinks, rope, ratio, kv_source, top_k } => {
            need(ins, 2 + usize::from(top_k.is_some()))?;
            let sinks = b.small(*sinks, weights)?.to_vec();
            let topk = top_k.map(|_| &ins[2]);
            if ins[1].rows > b.n_batch {
                bail!("compressed attention fed {} rows past the batch of {}", ins[1].rows, b.n_batch);
            }
            let empty = Buf::new(ins[1].rows, 0);
            b.cache_for(*layer, *head_dim, 0, Some(*window))?.append(&ins[1], &empty)?;
            let src = kv_source.unwrap_or(*layer);
            let comp = b.layers().get(&src).and_then(|s| s.compressed.get(&0)).cloned();
            let cache = b.layers().get(layer).and_then(|s| s.kv.as_ref()).ok_or_else(|| anyhow!("layer {layer} lost its cache"))?;
            compressed_attention(cache, comp.as_ref(), &ins[0], *n_head, *head_dim, *scale, *window, &sinks, rope, *ratio, topk)
        }
        other => bail!("{other:?} routed to the sparse module"),
    }
}

use crate::{Buf, CpuBackend, KvLayer};
use anyhow::{Result, anyhow, bail};
use kf_core::{Inputs, Op, SwaType, WeightSource, Window};

/// Scoring and masking rules of one attention call
pub struct Spec<'a> {
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub head_dim_v: usize,
    pub scale: f32,
    pub window: Option<Window>,
    pub softcap: f32,
    pub sinks: Option<&'a [f32]>,
    pub alibi_bias: f32,
    /// Allowed key positions per query row, replacing the causal mask
    pub top_k: Option<&'a Buf>,
    /// Last key position each query row sees, its own position when empty
    pub attend: &'a [u32],
}

/// ALiBi slope of one head by the llama rule over the query head count
pub fn alibi_slope(h: usize, n_head: usize, max_bias: f32) -> f32 {
    let n2 = 1usize << (usize::BITS - 1 - n_head.leading_zeros());
    let m0 = 2f32.powf(-max_bias / n2 as f32);
    let m1 = 2f32.powf(-(max_bias / 2.0) / n2 as f32);
    if h < n2 { m0.powi(h as i32 + 1) } else { m1.powi((2 * (h - n2) + 1) as i32) }
}

/// Whether key position j is visible to the query at position pos seeing keys through hi, under the window rule
pub fn visible(pos: usize, hi: usize, j: usize, window: Option<Window>) -> bool {
    if j > hi {
        return false;
    }
    if j > pos {
        return true;
    }
    match window {
        None => true,
        Some(Window { size, kind: SwaType::Standard }) => pos - j < size,
        Some(Window { size, kind: SwaType::Chunked }) => j >= (pos / size) * size,
    }
}

fn f16(v: f32) -> f32 {
    half::f16::from_f32(v).to_f32()
}

/// Causal attention of the query rows over the cache with the reference's half precision rounding of q and the value accumulator
pub fn run(cache: &KvLayer, q: &Buf, s: &Spec) -> Result<Buf> {
    if q.cols != s.n_head * s.head_dim {
        bail!("attention shapes q {} for {} heads of {}", q.cols, s.n_head, s.head_dim);
    }
    if cache.k_width != s.n_head_kv * s.head_dim || cache.v_width != s.n_head_kv * s.head_dim_v {
        bail!("attention cache widths {} and {} for {} kv heads of {} and {}", cache.k_width, cache.v_width, s.n_head_kv, s.head_dim, s.head_dim_v);
    }
    if s.n_head_kv == 0 || !s.n_head.is_multiple_of(s.n_head_kv) {
        bail!("attention with {} heads over {} kv heads", s.n_head, s.n_head_kv);
    }
    if cache.n < q.rows {
        bail!("attention over a cache of {} rows for {} queries", cache.n, q.rows);
    }
    if cache.cap == 0 {
        bail!("attention over a cache with no rows");
    }
    if let Some(sk) = s.sinks
        && sk.len() < s.n_head
    {
        bail!("attention sinks of {} for {} heads", sk.len(), s.n_head);
    }
    if let Some(t) = s.top_k
        && t.rows != q.rows
    {
        bail!("top k index of {} rows for {} queries", t.rows, q.rows);
    }
    let n_before = cache.n - q.rows;
    let group = s.n_head / s.n_head_kv;
    let per_group = s.top_k.is_some_and(|t| t.cols.is_multiple_of(s.n_head_kv) && s.n_head_kv > 1);
    let mut out = Buf::new(q.rows, s.n_head * s.head_dim_v);
    let mut keys: Vec<usize> = Vec::with_capacity(cache.n);
    let mut qh: Vec<f32> = vec![0.0; s.head_dim];
    let mut acc: Vec<f32> = vec![0.0; s.head_dim_v];
    for r in 0..q.rows {
        let pos = n_before + r;
        let hi = s.attend.get(r).map_or(pos, |a| (*a as usize).min(cache.n - 1)).max(pos);
        for h in 0..s.n_head {
            let hk = h / group;
            if h == 0 || (per_group && h % group == 0) {
                keys.clear();
                match s.top_k {
                    Some(t) => {
                        let row = t.row(r);
                        let seg = if per_group { &row[hk * (t.cols / s.n_head_kv)..(hk + 1) * (t.cols / s.n_head_kv)] } else { row };
                        keys.extend(seg.iter().filter(|v| **v >= 0.0).map(|v| *v as usize).filter(|j| *j < cache.n));
                    }
                    None => keys.extend((0..=hi).filter(|j| visible(pos, hi, *j, s.window))),
                }
            }
            for (d, v) in qh.iter_mut().zip(&q.row(r)[h * s.head_dim..(h + 1) * s.head_dim]) {
                *d = f16(*v);
            }
            let slope = if s.alibi_bias > 0.0 { alibi_slope(h, s.n_head, s.alibi_bias) } else { 0.0 };
            let mut m = f32::NEG_INFINITY;
            let mut sum = 0.0f32;
            acc.iter_mut().for_each(|v| *v = 0.0);
            for &j in &keys {
                let kj = &cache.k_row(j)[hk * s.head_dim..][..s.head_dim];
                let mut dot = 0.0f64;
                for i in 0..s.head_dim {
                    dot += (qh[i] * kj[i].to_f32()) as f64;
                }
                let mut sc = dot as f32 * s.scale;
                if s.softcap != 0.0 {
                    sc = s.softcap * (sc / s.softcap).tanh();
                }
                if slope != 0.0 {
                    sc -= slope * (pos as f32 - j as f32);
                }
                let (ms, vs) = if sc > m {
                    let ms = (m - sc).exp();
                    m = sc;
                    (ms, 1.0)
                } else {
                    (1.0, (sc - m).exp())
                };
                let vj = &cache.v_row(j)[hk * s.head_dim_v..][..s.head_dim_v];
                for (a, v) in acc.iter_mut().zip(vj) {
                    *a = f16(f16(*a * ms) + v.to_f32() * vs);
                }
                sum = sum * ms + vs;
            }
            if let Some(sk) = s.sinks {
                let sv = sk[h];
                let (ms, vs) = if sv > m { ((m - sv).exp(), 1.0) } else { (1.0, (sv - m).exp()) };
                acc.iter_mut().for_each(|a| *a *= ms);
                sum = sum * ms + vs;
            }
            let inv = if sum == 0.0 { 0.0 } else { 1.0 / sum };
            let oh = &mut out.row_mut(r)[h * s.head_dim_v..(h + 1) * s.head_dim_v];
            for (o, a) in oh.iter_mut().zip(&acc) {
                *o = a * inv;
            }
        }
    }
    Ok(out)
}

/// Attention over the rows of the pass alone, every row seeing every other, plain single precision
pub fn full(q: &Buf, k: &Buf, v: &Buf, n_head: usize, head_dim: usize, scale: f32) -> Result<Buf> {
    if q.cols != n_head * head_dim || k.cols != q.cols || v.cols != q.cols || k.rows != q.rows || v.rows != q.rows {
        bail!("full attention shapes q {}x{} k {}x{} v {}x{} for {n_head} heads of {head_dim}", q.rows, q.cols, k.rows, k.cols, v.rows, v.cols);
    }
    let n = q.rows;
    let mut out = Buf::new(n, q.cols);
    let mut scores = vec![0.0f32; n];
    for r in 0..n {
        for h in 0..n_head {
            let qh = &q.row(r)[h * head_dim..(h + 1) * head_dim];
            let mut m = f32::NEG_INFINITY;
            for j in 0..n {
                let kj = &k.row(j)[h * head_dim..(h + 1) * head_dim];
                let dot: f32 = qh.iter().zip(kj).map(|(a, b)| a * b).sum();
                scores[j] = dot * scale;
                m = m.max(scores[j]);
            }
            let mut sum = 0.0f32;
            for sc in scores.iter_mut() {
                *sc = (*sc - m).exp();
                sum += *sc;
            }
            let oh = &mut out.row_mut(r)[h * head_dim..(h + 1) * head_dim];
            for j in 0..n {
                let p = scores[j] / sum;
                for (o, vv) in oh.iter_mut().zip(&v.row(j)[h * head_dim..(h + 1) * head_dim]) {
                    *o += p * vv;
                }
            }
        }
    }
    Ok(out)
}

/// Dense attention with its cache, reading another layer's cache when the op names a source
pub fn exec(b: &mut CpuBackend, op: &Op, ins: &[Buf], inputs: &Inputs, weights: &dyn WeightSource) -> Result<Buf> {
    if let Op::AttentionFull { n_head, head_dim, scale } = op {
        if ins.len() < 3 {
            bail!("{} inputs for full attention needing q k v", ins.len());
        }
        return full(&ins[0], &ins[1], &ins[2], *n_head, *head_dim, *scale);
    }
    let Op::Attention { layer, n_head, n_head_kv, head_dim, head_dim_v, scale, window, softcap, sinks, alibi_bias, kv_source, top_k } = op else {
        bail!("{op:?} routed to the attention module")
    };
    let sinks = b.small_opt(*sinks, weights)?;
    let n_in = if kv_source.is_some() { 1 } else { 3 } + usize::from(top_k.is_some());
    if ins.len() < n_in {
        bail!("{} inputs for attention needing {n_in}", ins.len());
    }
    let q = &ins[0];
    let topk = top_k.map(|_| &ins[n_in - 1]);
    let spec = Spec {
        n_head: *n_head,
        n_head_kv: *n_head_kv,
        head_dim: *head_dim,
        head_dim_v: *head_dim_v,
        scale: *scale,
        window: *window,
        softcap: *softcap,
        sinks: sinks.as_deref(),
        alibi_bias: *alibi_bias,
        top_k: topk,
        attend: &inputs.attend,
    };
    match kv_source {
        Some(src) => {
            let cache = b.layers().get(src).and_then(|s| s.kv.as_ref()).ok_or_else(|| anyhow!("layer {src} has no cache for layer {layer} to read"))?;
            run(cache, q, &spec)
        }
        None => {
            if ins[1].rows > b.n_batch {
                bail!("attention fed {} rows past the batch of {}", ins[1].rows, b.n_batch);
            }
            let cache = b.cache_for(*layer, n_head_kv * head_dim, n_head_kv * head_dim_v, window.map(|w| w.size))?;
            cache.append(&ins[1], &ins[2])?;
            run(cache, q, &spec)
        }
    }
}

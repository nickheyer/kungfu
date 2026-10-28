use crate::CudaBackend;
use crate::exec::{self, ExpertT, MoeDims, Place, Prepared, S_HOST, S_SEL, S_SLOTS, S_WTS, dptr, dtoh, htod, row_bytes};
use anyhow::{Context, Result, anyhow, bail};
use cudarc::driver::{CudaEvent, CudaSlice, PinnedHostSlice};
use kf_backend_cpu::{Buf, CpuBackend, ops};
use kf_core::{AllocKind, Gating, Inputs, Op, Pool, Selection, WeightSource, Wid};
use std::collections::{BTreeMap, HashMap};

struct CachedTensor {
    wid: Wid,
    eb: usize,
    _dev: CudaSlice<u8>,
    ptr: u64,
    pinned: Option<PinnedHostSlice<u8>>,
}

/// Device slots holding whole experts of one layer, evicted by least recent use, selections counted per expert
pub(crate) struct ExpertCache {
    pub hits: u64,
    pub misses: u64,
    pub counts: Vec<u64>,
    owner: Vec<Option<usize>>,
    slot_of: HashMap<usize, usize>,
    stamp: Vec<u64>,
    clock: u64,
    tensors: [Option<CachedTensor>; 3],
    /// Copies in flight on the copy stream, waited for before the slots are read again
    pending: Option<CudaEvent>,
    /// Experts copied ahead for this layer by the previous layer's hidden state, checked against the real routing
    predicted: Vec<usize>,
    pub prefetches: u64,
    pub prefetch_hits: u64,
}

impl ExpertCache {
    pub fn slots(&self) -> usize {
        self.owner.len()
    }

    fn lookup(&mut self, e: usize) -> Option<usize> {
        match self.slot_of.get(&e) {
            Some(&s) => {
                self.hits += 1;
                self.stamp[s] = self.clock;
                Some(s)
            }
            None => {
                self.misses += 1;
                None
            }
        }
    }

    /// Least recently used slot untouched this step, empty slots first
    fn victim(&self) -> Option<usize> {
        (0..self.owner.len()).filter(|&s| self.stamp[s] != self.clock).min_by_key(|&s| (self.owner[s].is_some(), self.stamp[s]))
    }

    /// Least recently used slot of a layer that has not run this step, empty slots first
    fn victim_any(&self) -> Option<usize> {
        (0..self.owner.len()).min_by_key(|&s| (self.owner[s].is_some(), self.stamp[s]))
    }

    fn assign(&mut self, s: usize, e: usize) {
        if let Some(old) = self.owner[s].replace(e) {
            self.slot_of.remove(&old);
        }
        self.slot_of.insert(e, s);
        self.stamp[s] = self.clock;
    }
}

/// Allocates the slots of a layer once, staging pinned host copies of the expert tensors the plan pins
pub(crate) fn ensure_cache(b: &mut CudaBackend, layer: Option<usize>, op: &Op, weights: &dyn WeightSource) -> Result<()> {
    let layer = layer.context("expert cache on a node outside any block")?;
    if b.res.experts.contains_key(&layer) {
        return Ok(());
    }
    let Op::MoeFfn { gate_exps, up_exps, down_exps, n_expert, .. } = op else { bail!("expert cache for {op:?}") };
    let slots = b.placement(Some(layer)).2.min(*n_expert).max(1);
    let mut tensors = [None, None, None];
    for (k, w) in [*gate_exps, Some(*up_exps), *down_exps].into_iter().enumerate() {
        let Some(w) = w else { continue };
        let d = weights.dims(w);
        let eb = row_bytes(weights.ggml_type(w), d[0])? * d[1];
        let dev = b.alloc::<u8>(slots * eb, AllocKind::Experts, weights.name(w))?;
        let ptr = dptr(&dev, &b.stream);
        let pinned = if b.pinned.contains(&w) {
            let bytes = weights.bytes(w)?;
            let mut h = unsafe { b.ctx.alloc_pinned::<u8>(bytes.len()) }.with_context(|| format!("pinned copy of {}", weights.name(w)))?;
            h.as_mut_slice()?.copy_from_slice(bytes);
            kf_core::event::alloc(Pool::Host, AllocKind::Experts, bytes.len() as u64);
            Some(h)
        } else {
            None
        };
        tensors[k] = Some(CachedTensor { wid: w, eb, _dev: dev, ptr, pinned });
    }
    b.res.experts.insert(layer, ExpertCache { hits: 0, misses: 0, counts: vec![0; *n_expert], owner: vec![None; slots], slot_of: HashMap::new(), stamp: vec![0; slots], clock: 0, tensors, pending: None, predicted: Vec::new(), prefetches: 0, prefetch_hits: 0 });
    b.res.epoch += 1;
    if let Some(profile) = b.profile.get(&layer).cloned() {
        let mut order: Vec<(usize, u64)> = profile.iter().copied().enumerate().filter(|(e, c)| *c > 0 && *e < *n_expert).collect();
        order.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        let warm: Vec<usize> = order.into_iter().take(slots).map(|(e, _)| e).collect();
        fill(b, layer, &warm, weights, true)?;
    }
    Ok(())
}

const V_GATE: Wid = Wid(u32::MAX - 2);
const V_UP: Wid = Wid(u32::MAX - 1);
const V_DOWN: Wid = Wid(u32::MAX);

/// One expert of the stacked tensors presented as plain matrices under reserved ids
struct ExpertView<'a> {
    inner: &'a dyn WeightSource,
    expert: usize,
    map: [(Wid, Option<Wid>, [usize; 2]); 3],
}

impl<'a> ExpertView<'a> {
    fn new(inner: &'a dyn WeightSource, expert: usize, gate: Option<Wid>, up: Wid, down: Wid) -> Self {
        let dims = |w: Option<Wid>| {
            w.map_or([0, 0], |w| {
                let d = inner.dims(w);
                [d[0], d[1]]
            })
        };
        Self { inner, expert, map: [(V_GATE, gate, dims(gate)), (V_UP, Some(up), dims(Some(up))), (V_DOWN, Some(down), dims(Some(down)))] }
    }

    fn find(&self, w: Wid) -> Option<(Wid, &[usize; 2])> {
        self.map.iter().find_map(|(v, real, d)| if *v == w { real.map(|r| (r, d)) } else { None })
    }
}

impl WeightSource for ExpertView<'_> {
    fn bytes(&self, w: Wid) -> Result<&[u8]> {
        match self.find(w) {
            Some((real, d)) => {
                let eb = row_bytes(self.inner.ggml_type(real), d[0])? * d[1];
                let all = self.inner.bytes(real)?;
                all.get(self.expert * eb..(self.expert + 1) * eb).ok_or_else(|| anyhow!("expert {} past the end of {}", self.expert, self.inner.name(real)))
            }
            None => self.inner.bytes(w),
        }
    }
    fn ggml_type(&self, w: Wid) -> u32 {
        self.inner.ggml_type(self.find(w).map_or(w, |f| f.0))
    }
    fn dims(&self, w: Wid) -> &[usize] {
        match self.find(w) {
            Some((_, d)) => d,
            None => self.inner.dims(w),
        }
    }
    fn name(&self, w: Wid) -> &str {
        self.inner.name(self.find(w).map_or(w, |f| f.0))
    }
    fn len(&self) -> usize {
        self.inner.len()
    }
}

/// Routed sum of the experts missing from the slots, computed through the reference backend row group by expert
fn host_experts(host: &mut CpuBackend, op: &Op, missed: &BTreeMap<usize, Vec<(usize, f32)>>, xh: &[f32], d: &MoeDims, inputs: &Inputs, weights: &dyn WeightSource) -> Result<Vec<f32>> {
    let Op::MoeFfn { gate_exps, up_exps, down_exps: Some(down_exps), scale, act, .. } = op else { bail!("host experts for {op:?}") };
    let n = d.n_embd;
    let mut part = vec![0.0f32; d.rows * n];
    for (&e, list) in missed {
        let mut x = Buf::new(list.len(), n);
        for (k, (r, _)) in list.iter().enumerate() {
            x.row_mut(k).copy_from_slice(&xh[r * n..(r + 1) * n]);
        }
        let view = ExpertView::new(weights, e, *gate_exps, *up_exps, *down_exps);
        let hidden = match gate_exps {
            Some(_) => {
                let g = host.exec_node(&Op::MatMul { weight: V_GATE, bias: None, scale: None }, std::slice::from_ref(&x), inputs, &view)?;
                let u = host.exec_node(&Op::MatMul { weight: V_UP, bias: None, scale: None }, &[x], inputs, &view)?;
                host.exec_node(&Op::GatedAct { act: *act, limit: None }, &[g, u], inputs, &view)?
            }
            None => {
                let both = host.exec_node(&Op::MatMul { weight: V_UP, bias: None, scale: None }, &[x], inputs, &view)?;
                host.exec_node(&Op::GatedActFused { act: *act, limit: None }, &[both], inputs, &view)?
            }
        };
        let y = host.exec_node(&Op::MatMul { weight: V_DOWN, bias: None, scale: None }, &[hidden], inputs, &view)?;
        for (k, (r, w)) in list.iter().enumerate() {
            let ws = w * scale;
            for (acc, v) in part[r * n..(r + 1) * n].iter_mut().zip(y.row(k)) {
                *acc += ws * v;
            }
        }
    }
    Ok(part)
}

/// Copies experts into victim slots in the order given on the copy stream, after the compute stream's reads of the slots, so later tokens hit, returning the experts copied
fn fill(b: &mut CudaBackend, layer: usize, order: &[usize], weights: &dyn WeightSource, ahead: bool) -> Result<Vec<usize>> {
    let copy = b.copy_stream.clone();
    let after_reads = b.stream.record_event(None).context("record slot reads")?;
    copy.wait(&after_reads).context("wait slot reads")?;
    let cache = b.res.experts.get_mut(&layer).context("expert cache")?;
    let mut copied = Vec::new();
    for &e in order {
        if cache.slot_of.contains_key(&e) {
            continue;
        }
        let Some(s) = (if ahead { cache.victim_any() } else { cache.victim() }) else { break };
        cache.assign(s, e);
        for ct in cache.tensors.iter().flatten() {
            let range = e * ct.eb..(e + 1) * ct.eb;
            let src = match &ct.pinned {
                Some(h) => h.as_slice()?.get(range),
                None => weights.bytes(ct.wid)?.get(range),
            };
            let src = src.ok_or_else(|| anyhow!("expert {e} past the end of {}", weights.name(ct.wid)))?;
            htod(&copy, ct.ptr + (s * ct.eb) as u64, src)?;
        }
        copied.push(e);
    }
    if !copied.is_empty() {
        cache.pending = Some(copy.record_event(None).context("record expert copies")?);
    }
    Ok(copied)
}

/// The experts the next cached layer would route the current layer's expert input to, by its own router, most chosen first
fn predict_next(b: &mut CudaBackend, p: &Prepared, i: usize, xh: &[f32], d: &MoeDims, weights: &dyn WeightSource) -> Result<Option<(usize, Vec<usize>)>> {
    let Some(j) = (i + 1..p.nodes.len()).find(|j| p.place[*j] == Place::Experts) else { return Ok(None) };
    let node = &p.nodes[j];
    let Op::MoeFfn { router, router_bias, probs_bias, n_expert, n_used, gating, selection, groups, index_div, .. } = &node.op else { return Ok(None) };
    let layer = node.layer.context("expert node outside any block")?;
    if weights.dims(*router).first() != Some(&d.n_embd) {
        return Ok(None);
    }
    let x = Buf { data: xh.to_vec(), rows: d.rows, cols: d.n_embd };
    let rb = b.host.small_opt(*router_bias, weights)?;
    let pb = b.host.small_opt(*probs_bias, weights)?;
    let logits = ops::matmul(&x, *router, weights, rb.as_deref())?;
    let mut tally: BTreeMap<usize, usize> = BTreeMap::new();
    for r in 0..logits.rows {
        let l = logits.row(r);
        let probs: Vec<f32> = match gating {
            Gating::Softmax | Gating::SoftmaxWeight => {
                let mut p = l.to_vec();
                ops::softmax_inplace(&mut p);
                p
            }
            Gating::Sigmoid => l.iter().map(|v| ops::sigmoid(*v)).collect(),
            Gating::SqrtSoftplus => l.iter().map(|v| ops::softplus(*v, 1.0).sqrt()).collect(),
        };
        let mut sel: Vec<f32> = match selection {
            Selection::Probs => probs,
            Selection::Logits => l.to_vec(),
            Selection::SigmoidLogits => l.iter().map(|v| ops::sigmoid(*v)).collect(),
        };
        if let Some(pb) = &pb {
            sel.iter_mut().zip(pb.iter()).for_each(|(s, b)| *s += b);
        }
        for e in kf_backend_cpu::moe::pick(&sel, *n_used, *groups) {
            let e = if *index_div > 1 { e / index_div } else { e };
            if e < *n_expert {
                *tally.entry(e).or_default() += 1;
            }
        }
    }
    let mut order: Vec<(usize, usize)> = tally.into_iter().collect();
    order.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    Ok(Some((layer, order.into_iter().map(|(e, _)| e).collect())))
}

/// Copies the predicted experts of a later layer into its free or least recent slots while the layers between compute
fn prefetch(b: &mut CudaBackend, p: &Prepared, layer: usize, predicted: Vec<usize>, weights: &dyn WeightSource) -> Result<()> {
    let Some(j) = p.nodes.iter().position(|n| n.layer == Some(layer) && matches!(n.op, Op::MoeFfn { .. })) else { return Ok(()) };
    let op = p.nodes[j].op.clone();
    ensure_cache(b, Some(layer), &op, weights)?;
    let copied = fill(b, layer, &predicted, weights, true)?;
    let cache = b.res.experts.get_mut(&layer).context("expert cache")?;
    cache.prefetches += copied.len() as u64;
    cache.predicted = copied;
    Ok(())
}

/// Routes on the device, runs resident experts there and missing ones on the host, then adds both sums
pub(crate) fn run(b: &mut CudaBackend, p: &mut Prepared, i: usize, inputs: &Inputs, weights: &dyn WeightSource) -> Result<()> {
    let node = p.nodes[i].clone();
    let Op::MoeFfn { router, router_bias, probs_bias, .. } = &node.op else { bail!("expert step on {:?}", node.op) };
    let layer = node.layer.context("expert cache on a node outside any block")?;
    exec::ensure_weight(b, *router, weights)?;
    for v in [router_bias, probs_bias].into_iter().flatten() {
        exec::ensure_vec(b, *v, weights)?;
    }
    ensure_cache(b, node.layer, &node.op, weights)?;
    if let Some(ev) = b.res.experts.get_mut(&layer).and_then(|c| c.pending.take()) {
        b.stream.wait(&ev).context("wait expert copies")?;
    }
    if p.shapes[node.out.0 as usize].is_none() {
        exec::plan_device_node(b, p, i, weights)?;
    }
    let cache_slots = b.res.experts.get(&layer).context("expert cache")?.slots();
    let d = exec::moe_dims(&node.op, p.shape(node.inputs[0])?, cache_slots, weights)?;
    for l in exec::route_launches(b, p, i, weights)? {
        l.run(&b.stream)?;
    }
    let total = d.rows * d.n_used;
    let mut sel = vec![0i32; total];
    let mut wts = vec![0.0f32; total];
    let mut xh = vec![0.0f32; d.rows * d.n_embd];
    dtoh(&b.stream, &mut sel, p.scratch_ptr(i, S_SEL))?;
    dtoh(&b.stream, &mut wts, p.scratch_ptr(i, S_WTS))?;
    dtoh(&b.stream, &mut xh, p.ptr(node.inputs[0])?)?;
    let cache = b.res.experts.get_mut(&layer).context("expert cache")?;
    cache.clock += 1;
    let predicted = std::mem::take(&mut cache.predicted);
    cache.prefetch_hits += predicted.iter().filter(|e| sel.contains(&(**e as i32))).count() as u64;
    let mut slot_of = vec![-1i32; total];
    let mut missed: BTreeMap<usize, Vec<(usize, f32)>> = BTreeMap::new();
    for (rj, &e) in sel.iter().enumerate() {
        let Ok(e) = usize::try_from(e) else { continue };
        if let Some(c) = cache.counts.get_mut(e) {
            *c += 1;
        }
        match cache.lookup(e) {
            Some(s) => slot_of[rj] = s as i32,
            None => missed.entry(e).or_default().push((rj / d.n_used, wts[rj])),
        }
    }
    let mut t = [None; 3];
    for (k, ct) in cache.tensors.iter().enumerate() {
        if let Some(ct) = ct {
            t[k] = Some(ExpertT::new(ct.ptr, ct.wid, weights)?);
        }
    }
    let slots = p.scratch_ptr(i, S_SLOTS);
    htod(&b.stream, slots, &slot_of)?;
    let host_ptr = if missed.is_empty() { 0 } else { p.scratch_ptr(i, S_HOST) };
    let (body, combine) = exec::expert_launches(b, p, i, &d, t, slots, host_ptr)?;
    for l in &body {
        l.run(&b.stream)?;
    }
    let next = predict_next(b, p, i, &xh, &d, weights)?;
    if !missed.is_empty() {
        let part = host_experts(&mut b.host, &node.op, &missed, &xh, &d, inputs, weights)?;
        let mut order: Vec<(usize, usize)> = missed.iter().map(|(e, v)| (*e, v.len())).collect();
        order.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        let order: Vec<usize> = order.into_iter().map(|(e, _)| e).collect();
        fill(b, layer, &order, weights, false)?;
        htod(&b.stream, host_ptr, &part)?;
    }
    combine.run(&b.stream)?;
    if let Some((layer_next, predicted)) = next {
        prefetch(b, p, layer_next, predicted, weights)?;
    }
    Ok(())
}

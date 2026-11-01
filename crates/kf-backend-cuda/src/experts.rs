//! Device slots holding whole experts of a layer behind a device side table from expert to slot. During decode the
//! coprocessor computes the experts the slots miss, see coproc.rs, and the slots are refilled at pass boundaries by
//! selection counts. During prefill every expert streams through the device.
use crate::CudaBackend;
use crate::coproc::{self, COPROC_ROWS, LayerDesc, TensorRef};
use crate::exec::{self, ExpertT, MoeDims, Prepared, S_SEL, S_SLOTS, dptr, dtoh, htod, row_bytes};
use anyhow::{Context, Result, anyhow, bail, ensure};
use cudarc::driver::{CudaContext, CudaEvent, CudaSlice, CudaStream, PinnedHostSlice};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use kf_core::{AllocKind, Inputs, Op, WeightSource, Wid};
use std::collections::{HashMap, HashSet};

/// Device bytes of one of the two buffers experts stream through during passes of many rows
const EXPERT_STAGE_BYTES: usize = 64 << 20;

/// Held slots one pass boundary may give to other experts, each copy landing while the next pass runs, free slots
/// taking experts without limit
const MAX_SWAPS: usize = 16;

/// Passes within which a second miss of an expert admits it to a held slot
const ADMIT_WINDOW: u32 = 16;

/// Table entries the pinned staging ring holds for the asynchronous slot table writes of pass boundaries
const TABLE_RING: usize = 4096;

/// Writes one slot table entry on a stream from a pinned ring, so the write never stalls the host
fn write_table_from(ring: &mut PinnedHostSlice<i32>, at: &mut usize, stream: &CudaStream, ptr: u64, value: i32) -> Result<()> {
    let i = *at % TABLE_RING;
    *at += 1;
    let slot = unsafe { ring.as_mut_ptr().context("table ring")?.add(i) };
    unsafe { *slot = value };
    unsafe { cudarc::driver::result::memcpy_htod_async(ptr, std::slice::from_raw_parts(slot, 1), stream.cu_stream()) }.context("slot table write")?;
    Ok(())
}

fn write_table(b: &mut CudaBackend, stream: &CudaStream, ptr: u64, value: i32) -> Result<()> {
    if b.res.table_stage.is_none() {
        b.res.table_stage = Some((unsafe { b.ctx.alloc_pinned_with_flags::<i32>(TABLE_RING, 0) }.context("pinned table ring")?, 0));
    }
    let (ring, at) = b.res.table_stage.as_mut().context("table ring")?;
    write_table_from(ring, at, stream, ptr, value)
}

/// Bytes of the copier's pinned staging ring
const STAGING_BYTES: usize = 64 << 20;

/// One expert to copy into a slot, the mapped bytes of each tensor, the slot addresses, and the compute stream event
/// the copy must follow
struct Swap {
    srcs: Vec<(*const u8, usize)>,
    dsts: Vec<u64>,
    after: CudaEvent,
}

unsafe impl Send for Swap {}

/// A thread that moves experts into slots behind the passes, each swap's bytes go from the mapping into a pinned
/// staging ring by memcpy, then to the device by DMA on the copy stream, so the host thread of the passes only unmaps
/// and hands over, and maps the slot at the boundary after the one the copies landed in
pub(crate) struct Copier {
    tx: Option<std::sync::mpsc::Sender<Swap>>,
    pending: Arc<AtomicUsize>,
    handle: Option<std::thread::JoinHandle<()>>,
    failed: Arc<AtomicBool>,
    copy: Arc<CudaStream>,
}

impl Copier {
    fn new(ctx: Arc<CudaContext>, copy: Arc<CudaStream>) -> Result<Self> {
        let (tx, rx) = std::sync::mpsc::channel::<Swap>();
        let pending = Arc::new(AtomicUsize::new(0));
        let failed = Arc::new(AtomicBool::new(false));
        let (p2, f2, c2) = (pending.clone(), failed.clone(), copy.clone());
        let handle = std::thread::Builder::new()
            .name("kf-expert-copier".into())
            .spawn(move || {
                if let Err(e) = copier_loop(ctx, c2, rx, &p2) {
                    eprintln!("kungfu event=error code=expert_copier detail={e:#}");
                    f2.store(true, Ordering::Release);
                }
            })
            .context("spawn expert copier")?;
        Ok(Self { tx: Some(tx), pending, handle: Some(handle), failed, copy })
    }

    fn send(&self, swap: Swap) -> Result<()> {
        ensure!(!self.failed.load(Ordering::Acquire), "the expert copier stopped");
        self.pending.fetch_add(1, Ordering::AcqRel);
        self.tx.as_ref().context("expert copier gone")?.send(swap).map_err(|_| anyhow!("expert copier gone"))
    }

    /// Waits until every handed over swap has been issued to the copy stream
    fn drain(&self) -> Result<()> {
        while self.pending.load(Ordering::Acquire) > 0 {
            ensure!(!self.failed.load(Ordering::Acquire), "the expert copier stopped");
            std::thread::yield_now();
        }
        Ok(())
    }
}

impl Drop for Copier {
    fn drop(&mut self) {
        // The thread issues every handed over swap before it exits, and the copies land before the slots can go
        self.tx = None;
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
        let _ = self.copy.synchronize();
    }
}

fn copier_loop(ctx: Arc<CudaContext>, copy: Arc<CudaStream>, rx: std::sync::mpsc::Receiver<Swap>, pending: &AtomicUsize) -> Result<()> {
    ctx.bind_to_thread()?;
    let mut staging = unsafe { ctx.alloc_pinned_with_flags::<u8>(STAGING_BYTES, 0) }.context("pinned expert staging")?;
    kf_core::event::alloc(kf_core::Pool::Host, AllocKind::Experts, STAGING_BYTES as u64);
    // The ring is written in order, a copy's bytes reused once the DMA that read them is complete
    let mut head = 0usize;
    let mut landed: std::collections::VecDeque<(usize, usize, CudaEvent)> = std::collections::VecDeque::new();
    let base = staging.as_mut_ptr().context("staging pointer")?;
    while let Ok(sw) = rx.recv() {
        copy.wait(&sw.after).context("wait unmap")?;
        for (&(src, len), &dst) in sw.srcs.iter().zip(&sw.dsts) {
            ensure!(len <= STAGING_BYTES, "expert of {len} bytes past the staging ring");
            if head + len > STAGING_BYTES {
                head = 0;
            }
            // The copies still reading the bytes this one overwrites complete first
            while let Some((start, end, ev)) = landed.front() {
                if *start < head + len && *end > head {
                    ev.synchronize().context("staging reuse")?;
                    landed.pop_front();
                } else {
                    break;
                }
            }
            let stage = unsafe { base.add(head) };
            unsafe { std::ptr::copy_nonoverlapping(src, stage, len) };
            unsafe { cudarc::driver::result::memcpy_htod_async(dst, std::slice::from_raw_parts(stage, len), copy.cu_stream()) }.context("expert copy")?;
            let ev = copy.record_event(None).context("record expert copy")?;
            landed.push_back((head, head + len, ev));
            head += len;
        }
        pending.fetch_sub(1, Ordering::AcqRel);
    }
    Ok(())
}

/// The copier of the backend, made with the first cache
fn copier(b: &mut CudaBackend) -> Result<&Copier> {
    if b.res.copier.is_none() {
        b.res.copier = Some(Copier::new(b.ctx.clone(), b.copy_stream.clone())?);
    }
    b.res.copier.as_ref().context("expert copier")
}

/// Hands the copy of expert e into slot s of a layer to the copier, after the compute stream reached the event
fn hand_over(b: &mut CudaBackend, layer: usize, e: usize, s: usize, after: CudaEvent, weights: &dyn WeightSource) -> Result<()> {
    let (srcs, dsts) = {
        let cache = b.res.experts.get_mut(&layer).context("expert cache")?;
        let mut srcs = Vec::with_capacity(3);
        let mut dsts = Vec::with_capacity(3);
        for ct in cache.tensors.iter().flatten() {
            let all = weights.bytes(ct.wid)?;
            let src = all.get(e * ct.eb..(e + 1) * ct.eb).ok_or_else(|| anyhow!("expert {e} past the end of {}", weights.name(ct.wid)))?;
            srcs.push((src.as_ptr(), src.len()));
            dsts.push(ct.ptr + (s * ct.eb) as u64);
        }
        cache.landing.push((e, s));
        (srcs, dsts)
    };
    copier(b)?.send(Swap { srcs, dsts, after })
}

/// Maps every slot whose copy was handed over at an earlier boundary, once the copies have landed, on the compute
/// stream between passes so every pass sees one table
fn map_landed(b: &mut CudaBackend) -> Result<()> {
    if let Some(cp) = &b.res.copier {
        cp.drain()?;
        b.copy_stream.synchronize().context("expert copies")?;
    }
    let compute = b.stream.clone();
    let landed: Vec<(usize, Vec<(usize, usize)>)> = b.res.experts.iter_mut().filter(|(_, c)| !c.landing.is_empty()).map(|(l, c)| (*l, std::mem::take(&mut c.landing))).collect();
    for (layer, list) in landed {
        let table = b.res.experts.get(&layer).context("expert cache")?.table_ptr;
        for (e, s) in list {
            // The slot still belongs to the expert unless a later boundary gave it away
            if b.res.experts.get(&layer).context("expert cache")?.owner[s] == Some(e) {
                write_table(b, &compute, table + 4 * e as u64, s as i32)?;
            }
        }
    }
    Ok(())
}

/// Passes between halvings of the selection counts, so the slots follow the recent routing
const DECAY_PASSES: u32 = 4096;

/// Two buffers a layer's experts stream through chunk by chunk during passes of many rows, with the events that
/// order the copies into each against the kernels that read it
pub(crate) struct ExpertStage {
    _bufs: [CudaSlice<u8>; 2],
    ptr: [u64; 2],
    bytes: usize,
    ready: [Option<CudaEvent>; 2],
    free: [Option<CudaEvent>; 2],
}

struct CachedTensor {
    wid: Wid,
    eb: usize,
    _dev: CudaSlice<u8>,
    ptr: u64,
}

/// The slots of one layer, who holds them, and the selection counts the replacement follows
pub(crate) struct ExpertCache {
    pub hits: u64,
    pub misses: u64,
    pub counts: Vec<u64>,
    owner: Vec<Option<usize>>,
    slot_of: HashMap<usize, usize>,
    /// Pass each expert was last selected in and last missed in, the replacement's recency
    last_used: Vec<u32>,
    last_missed: Vec<u32>,
    tensors: [Option<CachedTensor>; 3],
    _table: CudaSlice<i32>,
    table_ptr: u64,
    /// Experts a prefill pass used most, swapped in at the pass boundary
    warm_wanted: Vec<usize>,
    /// Experts whose copies into slots are handed over and whose table entries the next boundary writes
    landing: Vec<(usize, usize)>,
}

impl ExpertCache {
    pub fn slots(&self) -> usize {
        self.owner.len()
    }

    pub fn table_ptr(&self) -> u64 {
        self.table_ptr
    }

    /// The slot buffers as expert tensors of slots many experts
    pub fn tensors(&self, weights: &dyn WeightSource) -> Result<[Option<ExpertT>; 3]> {
        let mut t = [None; 3];
        for (k, ct) in self.tensors.iter().enumerate() {
            if let Some(ct) = ct {
                t[k] = Some(ExpertT::new(ct.ptr, ct.wid, weights)?);
            }
        }
        Ok(t)
    }

    fn free_slot(&self) -> Option<usize> {
        self.owner.iter().position(Option::is_none)
    }

    /// A free slot not already given away this pass
    fn free_slot_past(&self, taken: &HashSet<usize>) -> Option<usize> {
        (0..self.owner.len()).find(|s| self.owner[*s].is_none() && !taken.contains(s))
    }

    /// The held slot whose expert was selected longest ago among those not protected
    fn stalest(&self, protected: &HashSet<usize>) -> Option<usize> {
        (0..self.owner.len()).filter(|&s| self.owner[s].is_some_and(|e| !protected.contains(&e))).min_by_key(|&s| self.last_used[self.owner[s].expect("held slot")])
    }
}

/// Slots of a layer under the plan, zero for a layer whose experts are all resident
pub(crate) fn cache_slots(b: &CudaBackend, layer: Option<usize>, n_expert: usize) -> usize {
    b.placement(layer).2.min(n_expert)
}

/// The layer's expert tensors as the workers read them
fn layer_desc(op: &Op, weights: &dyn WeightSource) -> Result<LayerDesc> {
    let Op::MoeFfn { gate_exps, up_exps, down_exps: Some(down_exps), n_used, act, limit, scale, .. } = op else { bail!("coprocessor layer for {op:?}") };
    let tensor = |w: Wid| -> Result<TensorRef> {
        let d = weights.dims(w);
        ensure!(d.len() >= 3, "{} needs three dims", weights.name(w));
        let bytes = weights.bytes(w)?;
        Ok(TensorRef { ptr: bytes.as_ptr(), len: bytes.len(), ty: weights.ggml_type(w), n_in: d[0], n_out: d[1] })
    };
    let up = tensor(*up_exps)?;
    let gate = gate_exps.map(tensor).transpose()?;
    let down = tensor(*down_exps)?;
    let n_ff = if gate.is_some() { up.n_out } else { up.n_out / 2 };
    Ok(LayerDesc { gate, up, down, n_embd: up.n_in, n_ff, n_used: *n_used, act: *act, limit: *limit, scale: *scale })
}

/// Allocates the slots and the table of a layer once, warms the slots from the routing profile, and registers the
/// layer with the coprocessor
pub(crate) fn ensure_cache(b: &mut CudaBackend, layer: Option<usize>, op: &Op, weights: &dyn WeightSource) -> Result<()> {
    let layer = layer.context("expert cache on a node outside any block")?;
    if b.res.experts.contains_key(&layer) {
        return Ok(());
    }
    let Op::MoeFfn { gate_exps, up_exps, down_exps, n_expert, .. } = op else { bail!("expert cache for {op:?}") };
    let slots = cache_slots(b, Some(layer), *n_expert).max(1);
    let mut tensors = [None, None, None];
    for (k, w) in [*gate_exps, Some(*up_exps), *down_exps].into_iter().enumerate() {
        let Some(w) = w else { continue };
        let d = weights.dims(w);
        let eb = row_bytes(weights.ggml_type(w), d[0])? * d[1];
        let dev = b.alloc::<u8>(slots * eb + 16, AllocKind::Experts, weights.name(w))?;
        let ptr = dptr(&dev, &b.stream);
        tensors[k] = Some(CachedTensor { wid: w, eb, _dev: dev, ptr });
    }
    let table = b.alloc::<i32>(*n_expert, AllocKind::Experts, "expert slot table")?;
    let table_ptr = dptr(&table, &b.stream);
    htod(&b.stream, table_ptr, &vec![-1i32; *n_expert])?;
    let counts = match b.profile.get(&layer) {
        Some(p) if p.len() == *n_expert => p.clone(),
        _ => vec![0; *n_expert],
    };
    b.res.experts.insert(layer, ExpertCache { hits: 0, misses: 0, counts, owner: vec![None; slots], slot_of: HashMap::new(), last_used: vec![0; *n_expert], last_missed: vec![0; *n_expert], tensors, _table: table, table_ptr, warm_wanted: Vec::new(), landing: Vec::new() });
    b.res.epoch += 1;
    let n_layers = b.layers.keys().max().map_or(layer + 1, |m| (m + 1).max(layer + 1));
    if b.coproc.is_none() {
        b.coproc = Some(coproc::Coproc::new(&b.ctx, &b.stream, b.pool(), n_layers)?);
    }
    let desc = layer_desc(op, weights)?;
    b.coproc.as_mut().context("coprocessor")?.ensure_layer(layer, desc)?;
    let cache = &b.res.experts[&layer];
    let mut order: Vec<usize> = (0..*n_expert).filter(|&e| cache.counts[e] > 0).collect();
    order.sort_by(|&x, &y| cache.counts[y].cmp(&cache.counts[x]).then(x.cmp(&y)));
    order.truncate(slots);
    fill_free(b, layer, &order, weights)?;
    map_landed(b)
}

/// Copies experts into free slots on the copy stream and maps them, before any pass reads the layer
fn fill_free(b: &mut CudaBackend, layer: usize, order: &[usize], weights: &dyn WeightSource) -> Result<()> {
    for &e in order {
        let cache = b.res.experts.get_mut(&layer).context("expert cache")?;
        if cache.slot_of.contains_key(&e) {
            continue;
        }
        let Some(s) = cache.free_slot() else { break };
        cache.owner[s] = Some(e);
        cache.slot_of.insert(e, s);
        let after = b.stream.record_event(None).context("record free fill")?;
        hand_over(b, layer, e, s, after, weights)?;
    }
    Ok(())
}

/// At a pass boundary, unmaps the holder of slot s on the compute stream, then copies expert e into the slot and maps
/// it on the copy stream, so the next pass misses either expert until the bytes are in place and never reads a slot
/// being written
fn replace(b: &mut CudaBackend, layer: usize, e: usize, s: usize, weights: &dyn WeightSource) -> Result<()> {
    let compute = b.stream.clone();
    let (table, old) = {
        let cache = b.res.experts.get_mut(&layer).context("expert cache")?;
        let old = cache.owner[s];
        if let Some(old) = old {
            cache.slot_of.remove(&old);
        }
        (cache.table_ptr, old)
    };
    if let Some(old) = old {
        write_table(b, &compute, table + 4 * old as u64, -1)?;
    }
    let unmapped = compute.record_event(None).context("record unmap")?;
    let cache = b.res.experts.get_mut(&layer).context("expert cache")?;
    cache.owner[s] = Some(e);
    cache.slot_of.insert(e, s);
    let _ = table;
    hand_over(b, layer, e, s, unmapped, weights)
}

/// Closes a pass for the coprocessor, reads what every layer posted, counts the selections, and swaps the most
/// selected missing experts into the coldest slots, at most MAX_SWAPS copies that land during the next pass
pub(crate) fn end_pass(b: &mut CudaBackend, weights: &dyn WeightSource) -> Result<()> {
    let Some(c) = b.coproc.as_mut() else { return Ok(()) };
    let report = c.end_pass()?;
    let (busy_us, jobs, max_us) = c.pass_stats();
    let seq = c.seq;
    // The swaps of the last boundary have landed and take their table entries before the tables change again
    map_landed(b)?;
    let trace = std::env::var_os("KF_TRACE_EXPERTS").is_some();
    let mut fills: Vec<(usize, usize, usize)> = Vec::new();
    let mut evictions = 0usize;
    let mut n_missed = 0usize;
    for pl in &report {
        let cache = b.res.experts.get_mut(&pl.layer).context("expert cache")?;
        let protected: HashSet<usize> = pl.sel.iter().filter(|e| **e >= 0).map(|e| *e as usize).collect();
        for &e in &protected {
            cache.counts[e] += pl.sel.iter().filter(|x| **x as usize == e).count() as u64;
            cache.last_used[e] = seq;
        }
        cache.misses += pl.missed.len() as u64;
        cache.hits += pl.sel.len() as u64 - pl.missed.len() as u64;
        n_missed += pl.missed.len();
        let mut candidates: Vec<usize> = pl.missed.clone();
        candidates.sort_unstable();
        candidates.dedup();
        candidates.sort_by(|&x, &y| cache.counts[y].cmp(&cache.counts[x]));
        let mut taken: HashSet<usize> = HashSet::new();
        for e in candidates {
            if cache.slot_of.contains_key(&e) {
                continue;
            }
            // A free slot takes any miss, a held one only an expert missed twice within the window, replacing the stalest
            let again = cache.last_missed[e] != 0 && seq.wrapping_sub(cache.last_missed[e]) <= ADMIT_WINDOW;
            cache.last_missed[e] = seq;
            if let Some(s) = cache.free_slot_past(&taken) {
                taken.insert(s);
                fills.push((pl.layer, e, s));
            } else if again && evictions < MAX_SWAPS {
                let keep: HashSet<usize> = protected.union(&taken.iter().filter_map(|s| cache.owner[*s]).collect()).copied().collect();
                if let Some(s) = cache.stalest(&keep) {
                    taken.insert(s);
                    evictions += 1;
                    fills.push((pl.layer, e, s));
                }
            }
        }
    }
    // Prefill passes name the experts they used most, which take the coldest slots not chosen this pass
    let wanted: Vec<(usize, Vec<usize>)> = b.res.experts.iter_mut().filter(|(_, c)| !c.warm_wanted.is_empty()).map(|(l, c)| (*l, std::mem::take(&mut c.warm_wanted))).collect();
    for (layer, list) in wanted {
        let cache = b.res.experts.get_mut(&layer).context("expert cache")?;
        let wanted: HashSet<usize> = list.iter().copied().collect();
        for &e in &list {
            cache.last_used[e] = seq;
        }
        let mut taken: HashSet<usize> = HashSet::new();
        for e in list {
            if cache.slot_of.contains_key(&e) {
                continue;
            }
            if let Some(s) = cache.free_slot_past(&taken) {
                taken.insert(s);
                fills.push((layer, e, s));
            } else if evictions < MAX_SWAPS {
                let keep: HashSet<usize> = wanted.union(&taken.iter().filter_map(|s| cache.owner[*s]).collect()).copied().collect();
                if let Some(s) = cache.stalest(&keep) {
                    taken.insert(s);
                    evictions += 1;
                    fills.push((layer, e, s));
                }
            }
        }
    }
    let n_swaps = fills.len();
    for (layer, e, s) in fills {
        replace(b, layer, e, s, weights)?;
    }
    if seq.is_multiple_of(DECAY_PASSES) {
        for cache in b.res.experts.values_mut() {
            cache.counts.iter_mut().for_each(|c| *c >>= 1);
        }
    }
    if trace {
        eprintln!("TRACE pass={seq} coproc layers={} missed={n_missed} jobs={jobs} worker_us={busy_us} longest_us={max_us} fills={n_swaps} evictions={evictions}", report.len());
    }
    Ok(())
}

/// A pass of many rows over a layer with slots, the pairs whose experts hold slots run from the slots, the experts
/// missing from them stream through two device buffers a chunk at a time, a chunk copied on the copy stream while
/// the device computes the chunk before it, then one combine over every pair. The experts the pass used most are
/// asked to take slots at the pass boundary.
fn prefill_experts(b: &mut CudaBackend, p: &mut Prepared, i: usize, d: &MoeDims, layer: usize, sel: &[i32], weights: &dyn WeightSource) -> Result<()> {
    let total = d.rows * d.n_used;
    let slots = p.scratch_ptr(i, S_SLOTS);
    let (sum_eb, resident_slots, slot_of_res, missing) = {
        let cache = b.res.experts.get(&layer).context("expert cache")?;
        let sum_eb: usize = cache.tensors.iter().flatten().map(|ct| ct.eb).sum();
        let slot_of_res: Vec<i32> = sel.iter().map(|&e| usize::try_from(e).ok().and_then(|e| cache.slot_of.get(&e)).map_or(-1, |s| *s as i32)).collect();
        let mut missing: Vec<usize> = sel.iter().filter_map(|&e| usize::try_from(e).ok()).filter(|e| !cache.slot_of.contains_key(e) && *e < d.n_expert).collect();
        missing.sort_unstable();
        missing.dedup();
        (sum_eb, cache.slots(), slot_of_res, missing)
    };
    let chunk = (EXPERT_STAGE_BYTES / sum_eb).clamp(1, d.n_expert);
    let bytes = chunk * sum_eb;
    if b.res.expert_stage.as_ref().is_none_or(|st| st.bytes < bytes) {
        b.res.expert_stage = None;
        let a = b.alloc::<u8>(bytes + 16, AllocKind::Compute, "expert stage")?;
        let c = b.alloc::<u8>(bytes + 16, AllocKind::Compute, "expert stage")?;
        let ptr = [dptr(&a, &b.stream), dptr(&c, &b.stream)];
        // The zero fill of the new buffers runs on the compute stream, so the copies into them wait for it
        let zeroed = b.stream.record_event(None).context("record stage zero fill")?;
        b.copy_stream.wait(&zeroed).context("wait stage zero fill")?;
        b.res.expert_stage = Some(ExpertStage { _bufs: [a, c], ptr, bytes, ready: [None, None], free: [None, None] });
        b.res.epoch += 1;
    }
    let stage_ptr = b.res.expert_stage.as_ref().map(|st| st.ptr).context("expert stage")?;
    // The expert tensors as they sit in each stage buffer, one after another, and as they sit in the slots
    let mut t_stage: [[Option<ExpertT>; 3]; 2] = [[None; 3]; 2];
    let mut offs = Vec::with_capacity(3);
    let t_slots = {
        let cache = b.res.experts.get(&layer).context("expert cache")?;
        let mut off = 0usize;
        for (k, ct) in cache.tensors.iter().enumerate() {
            let Some(ct) = ct else { continue };
            offs.push(off);
            for buf in 0..2 {
                t_stage[buf][k] = Some(ExpertT::new(stage_ptr[buf] + off as u64, ct.wid, weights)?);
            }
            off += chunk * ct.eb;
        }
        cache.tensors(weights)?
    };
    let int8 = exec::expert_int8_all(&b.kernels, d, &t_stage[0]);
    let mut v = Vec::new();
    let acts = exec::expert_input(b, p, i, d, &t_stage[0], int8, &mut v)?;
    for l in &v {
        l.run(&b.stream)?;
    }
    if slot_of_res.iter().any(|s| *s >= 0) {
        htod(&b.stream, slots, &slot_of_res)?;
        let dr = MoeDims { n_slots: resident_slots, ..*d };
        let mut body = Vec::new();
        exec::expert_body(b, p, i, &dr, &t_slots, slots, acts, int8, &mut body)?;
        for l in &body {
            l.run(&b.stream)?;
        }
    }
    let copy = b.copy_stream.clone();
    for (c, group) in missing.chunks(chunk).enumerate() {
        let buf = c % 2;
        // Copy the chunk once the device is done reading this buffer's last chunk
        if let Some(ev) = b.res.expert_stage.as_mut().and_then(|st| st.free[buf].take()) {
            copy.wait(&ev).context("wait stage free")?;
        }
        {
            let cache = b.res.experts.get(&layer).context("expert cache")?;
            for (k, ct) in cache.tensors.iter().flatten().enumerate() {
                let all = weights.bytes(ct.wid)?;
                for (j, &e) in group.iter().enumerate() {
                    let src = all.get(e * ct.eb..(e + 1) * ct.eb).ok_or_else(|| anyhow!("expert {e} past the end of {}", weights.name(ct.wid)))?;
                    htod(&copy, stage_ptr[buf] + (offs[k] + j * ct.eb) as u64, src)?;
                }
            }
        }
        let ready = copy.record_event(None).context("record stage copies")?;
        let slot_of: Vec<i32> = sel.iter().map(|&e| group.iter().position(|g| *g as i32 == e).map_or(-1, |j| j as i32)).collect();
        htod(&b.stream, slots, &slot_of)?;
        b.stream.wait(&ready).context("wait stage copies")?;
        let dc = MoeDims { n_slots: group.len(), ..*d };
        let mut body = Vec::new();
        exec::expert_body(b, p, i, &dc, &t_stage[buf], slots, acts, int8, &mut body)?;
        for l in &body {
            l.run(&b.stream)?;
        }
        let free = b.stream.record_event(None).context("record stage reads")?;
        if let Some(st) = b.res.expert_stage.as_mut() {
            st.free[buf] = Some(free);
            st.ready[buf] = Some(ready);
        }
    }
    let all: Vec<i32> = vec![0; total];
    htod(&b.stream, slots, &all)?;
    let combine = exec::expert_combine(b, p, i, d, slots, 0)?;
    combine.run(&b.stream)?;
    if std::mem::take(&mut b.res.kf8.scratch_grew) {
        b.res.epoch += 1;
    }
    let cache = b.res.experts.get_mut(&layer).context("expert cache")?;
    let mut tally = vec![0usize; d.n_expert];
    for &e in sel {
        if let Some(t) = usize::try_from(e).ok().and_then(|e| tally.get_mut(e)) {
            *t += 1;
        }
    }
    for (e, t) in tally.iter().enumerate() {
        cache.counts[e] += *t as u64;
    }
    cache.hits += sel.iter().zip(&slot_of_res).filter(|(e, s)| **e >= 0 && **s >= 0).count() as u64;
    cache.misses += sel.iter().zip(&slot_of_res).filter(|(e, s)| **e >= 0 && **s < 0).count() as u64;
    let mut order: Vec<usize> = (0..d.n_expert).filter(|&e| tally[e] > 0).collect();
    order.sort_by(|a, b2| tally[*b2].cmp(&tally[*a]).then(a.cmp(b2)));
    order.truncate(cache.slots());
    cache.warm_wanted = order;
    Ok(())
}

/// A pass of many rows over a layer with slots, routes on the device, then runs the slots and streams the rest
pub(crate) fn run(b: &mut CudaBackend, p: &mut Prepared, i: usize, _inputs: &Inputs, weights: &dyn WeightSource) -> Result<()> {
    let trace = std::env::var_os("KF_TRACE_EXPERTS").is_some();
    let t0 = std::time::Instant::now();
    let node = p.nodes[i].clone();
    let Op::MoeFfn { router, router_bias, probs_bias, n_expert, .. } = &node.op else { bail!("expert step on {:?}", node.op) };
    let layer = node.layer.context("expert cache on a node outside any block")?;
    exec::ensure_weight(b, *router, weights)?;
    for v in [router_bias, probs_bias].into_iter().flatten() {
        exec::ensure_vec(b, *v, weights)?;
    }
    ensure_cache(b, node.layer, &node.op, weights)?;
    // The slots this pass reads hold every expert handed over at the last boundary once its copy has landed
    if let Some(cp) = &b.res.copier {
        cp.drain()?;
        let ev = b.copy_stream.record_event(None).context("record expert copies")?;
        b.stream.wait(&ev).context("wait expert copies")?;
    }
    if p.shapes[node.out.0 as usize].is_none() {
        exec::plan_device_node(b, p, i, weights)?;
    }
    let slots = cache_slots(b, Some(layer), *n_expert).max(1);
    let d = exec::moe_dims(&node.op, p.shape(node.inputs[0])?, slots, weights)?;
    ensure!(d.rows > COPROC_ROWS, "a pass of {} rows over cached experts belongs to the coprocessor path", d.rows);
    for l in exec::route_launches(b, p, i, weights)? {
        l.run(&b.stream)?;
    }
    let mut sel = vec![0i32; d.rows * d.n_used];
    dtoh(&b.stream, &mut sel, p.scratch_ptr(i, S_SEL))?;
    prefill_experts(b, p, i, &d, layer, &sel, weights)?;
    if trace {
        b.stream.synchronize().ok();
        eprintln!("TRACE layer={layer} rows={} streamed total={:?}", d.rows, t0.elapsed());
    }
    Ok(())
}

//! CUDA backend, kernels are CUDA C compiled through NVRTC and cached as PTX
mod exec;
mod experts;
mod registry;
pub mod sources;

use anyhow::{Context, Result, anyhow};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use kf_backend_cpu::CpuBackend;
use kf_core::{AllocKind, Backend, Graph, Inputs, LayerPlan, Op, Plan, Pool, WeightSource, Wid};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

pub use registry::{KernelRef, KernelRegistry, cuda_kernel_for_op};

#[derive(Clone, Debug)]
pub struct DeviceInfo {
    pub ordinal: usize,
    pub name: String,
    pub total_bytes: usize,
    pub free_bytes: usize,
}

/// Every CUDA device the driver reports
pub fn devices() -> Result<Vec<DeviceInfo>> {
    let n = CudaContext::device_count().context("cuda device count")? as usize;
    let mut out = Vec::with_capacity(n);
    for ordinal in 0..n {
        let ctx = CudaContext::new(ordinal).with_context(|| format!("cuda device {ordinal}"))?;
        ctx.bind_to_thread()?;
        let (free_bytes, total_bytes) = cudarc::driver::result::mem_get_info()?;
        out.push(DeviceInfo { ordinal, name: ctx.name()?, total_bytes, free_bytes });
    }
    Ok(out)
}

pub struct CudaBackend {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    /// Copies that overlap the compute stream, streamed cache rows and prefetched experts
    pub copy_stream: Arc<CudaStream>,
    pub kernels: KernelRegistry,
    host: CpuBackend,
    layers: HashMap<usize, LayerPlan>,
    pinned: HashSet<Wid>,
    res: exec::Resources,
    pool: exec::BufPool,
    prepared: Vec<exec::Prepared>,
    last: Option<exec::Output>,
    pending: Option<anyhow::Error>,
    captures: u64,
    replays: u64,
    n_ctx: usize,
    n_batch: usize,
    /// Counts per expert by layer that warm each expert cache when it is made
    pub(crate) profile: HashMap<usize, Vec<u64>>,
}

impl CudaBackend {
    /// Opens a device, PTX compiled through NVRTC is cached under the cache directory when one is given
    pub fn new(ordinal: usize, cache_dir: Option<&Path>) -> Result<Self> {
        let ctx = CudaContext::new(ordinal).with_context(|| format!("cuda device {ordinal}"))?;
        unsafe { ctx.disable_event_tracking() };
        let stream = ctx.new_stream().context("cuda stream")?;
        let copy_stream = ctx.new_stream().context("cuda copy stream")?;
        let mut kernels = KernelRegistry::new(ctx.clone(), cache_dir)?;
        kernels.ensure("elementwise")?;
        Ok(Self {
            ctx,
            stream,
            copy_stream,
            kernels,
            host: CpuBackend::new(),
            layers: HashMap::new(),
            pinned: HashSet::new(),
            res: exec::Resources::default(),
            pool: exec::BufPool::new(Pool::Device(ordinal as u32)),
            prepared: Vec::new(),
            last: None,
            pending: None,
            captures: 0,
            replays: 0,
            n_ctx: 0,
            n_batch: 0,
            profile: HashMap::new(),
        })
    }

    /// Greedy token of the last row of the most recent graph output, computed on the device
    pub fn argmax_last(&mut self) -> Result<u32> {
        let out = self.last.ok_or_else(|| anyhow!("argmax before any graph ran"))?;
        exec::argmax_last(self, out)
    }

    /// A weight expanded to f32 on the device through the dequant kernel of its quant type
    pub fn dequant(&mut self, w: Wid, weights: &dyn WeightSource) -> Result<Vec<f32>> {
        exec::dequant(self, w, weights)
    }

    /// Slot count, hits, and misses of the expert cache of a layer
    pub fn expert_cache_stats(&self, layer: usize) -> (usize, u64, u64) {
        match self.res.experts.get(&layer) {
            Some(c) => (c.slots(), c.hits, c.misses),
            None => (self.layers.get(&layer).map_or(0, |l| l.expert_cache_slots), 0, 0),
        }
    }

    /// Experts copied ahead of their layer and how many of them the layer then chose, per layer
    pub fn prefetch_stats(&self) -> Vec<(usize, u64, u64)> {
        let mut v: Vec<(usize, u64, u64)> = self.res.experts.iter().map(|(l, c)| (*l, c.prefetches, c.prefetch_hits)).collect();
        v.sort();
        v
    }

    /// Decode launch sequences captured into CUDA graphs and replays of captured graphs
    pub fn graph_stats(&self) -> (u64, u64) {
        (self.captures, self.replays)
    }

    /// The device as a pool for the allocation events
    pub fn pool(&self) -> Pool {
        Pool::Device(self.ctx.ordinal() as u32)
    }

    /// A zeroed device buffer of n elements accounted to a kind
    pub(crate) fn alloc<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits>(&self, n: usize, kind: AllocKind, what: &str) -> Result<CudaSlice<T>> {
        exec::dev_alloc(&self.stream, self.pool(), n, kind, what)
    }

    /// A device copy of host data accounted to a kind
    pub(crate) fn upload<T: cudarc::driver::DeviceRepr + Unpin>(&self, data: &[T], kind: AllocKind, what: &str) -> Result<CudaSlice<T>> {
        exec::dev_upload(&self.stream, self.pool(), data, kind, what)
    }

    /// Weights pool, state pool, and expert slots of a block, the embedding and head live on the device
    fn placement(&self, layer: Option<usize>) -> (Pool, Pool, usize) {
        let dev = Pool::Device(self.ctx.ordinal() as u32);
        match layer.and_then(|l| self.layers.get(&l)) {
            Some(l) => (l.weights, l.state, l.expert_cache_slots),
            None => (dev, dev, 0),
        }
    }
}

impl Backend for CudaBackend {
    fn name(&self) -> &'static str {
        "cuda"
    }

    fn supports(&self, op: &Op) -> bool {
        self.kernels.supports(op)
    }

    fn apply_plan(&mut self, plan: &Plan) -> Result<()> {
        exec::apply_plan(self, plan)
    }

    fn run(&mut self, g: &Graph, inputs: &Inputs, weights: &dyn WeightSource) -> Result<Vec<f32>> {
        cuda_run_graph(self, g, inputs, weights)
    }

    fn truncate(&mut self, seq: usize, n: usize) -> Result<()> {
        cuda_truncate_state(self, seq, n)
    }

    fn reset(&mut self) {
        cuda_reset_state(self)
    }

    fn reset_seq(&mut self, seq: usize) {
        if let Err(e) = exec::truncate(self, Some(seq), 0) {
            self.pending.get_or_insert(e);
        }
    }

    fn save_state(&mut self, seq: usize) -> Result<Vec<u8>> {
        exec::save_state(self, seq)
    }

    fn load_state(&mut self, seq: usize, bytes: &[u8]) -> Result<()> {
        exec::load_state(self, seq, bytes)
    }

    fn expert_cache_stats(&self) -> Vec<(usize, usize, u64, u64, u64, u64)> {
        let mut v: Vec<(usize, usize, u64, u64, u64, u64)> = self.layers.values().filter(|l| l.expert_cache_slots > 0).map(|l| {
            let (slots, hits, misses) = self.expert_cache_stats(l.layer);
            let (pre, pre_hits) = self.res.experts.get(&l.layer).map_or((0, 0), |c| (c.prefetches, c.prefetch_hits));
            (l.layer, slots, hits, misses, pre, pre_hits)
        }).collect();
        v.sort();
        v
    }

    fn routing_profile(&self) -> Vec<(usize, Vec<u64>)> {
        let mut v: Vec<(usize, Vec<u64>)> = self.res.experts.iter().filter(|(_, c)| c.counts.iter().any(|n| *n > 0)).map(|(l, c)| (*l, c.counts.clone())).collect();
        v.sort_by_key(|(l, _)| *l);
        v
    }

    fn set_routing_profile(&mut self, profile: Vec<(usize, Vec<u64>)>) {
        self.profile = profile.into_iter().collect();
    }
}

/// Whether the backend has kernels for a ggml type
pub fn has_quant_module(ggml_type: u32) -> bool {
    sources::quant_module(ggml_type).is_some()
}

/// Runs a graph with activations resident on the device and decode launches replayed from CUDA graphs
pub fn cuda_run_graph(b: &mut CudaBackend, g: &Graph, inputs: &Inputs, weights: &dyn WeightSource) -> Result<Vec<f32>> {
    exec::run(b, g, inputs, weights)
}

/// Drops device and host state of one slot past its first n positions
pub fn cuda_truncate_state(b: &mut CudaBackend, seq: usize, n: usize) -> Result<()> {
    exec::truncate(b, Some(seq), n)
}

pub fn cuda_reset_state(b: &mut CudaBackend) {
    if let Err(e) = exec::truncate(b, None, 0) {
        b.pending.get_or_insert(e);
    }
}

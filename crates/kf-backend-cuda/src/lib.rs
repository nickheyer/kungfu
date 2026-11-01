//! CUDA backend, kernels are CUDA C compiled through NVRTC and cached as PTX
pub mod bench;
mod coproc;
pub mod drafter;
mod exec;
mod experts;
pub mod features;
mod kf8;
mod persist;
mod registry;
pub mod sources;
pub mod train;

use anyhow::{Context, Result};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use kf_backend_cpu::CpuBackend;
use kf_core::{AllocKind, Backend, Graph, Inputs, LayerPlan, Op, Plan, Pool, WeightSource, Wid};
use std::collections::HashMap;
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
    /// Expert tensors the plan keeps in host memory for the slots, page locked once a cache is made
    pinned: Vec<Wid>,
    res: exec::Resources,
    /// Host workers computing the experts the device slots miss, made with the first expert cache
    coproc: Option<coproc::Coproc>,
    pool: exec::BufPool,
    prepared: Vec<exec::Prepared>,
    last: Option<exec::Output>,
    /// The rows the last pass's final top k read, when it ended in one
    dense: Option<exec::Output>,
    /// The residual rows before the final norm the last pass kept
    hidden: Option<exec::Output>,
    /// The rows the passes scored, for the draft head, made with the first pass that keeps hidden rows
    pub ring: Option<features::FeatureRing>,
    /// The slot and token of every row of the last tree pass, which compaction appends to the ring along the kept path
    tree_tokens: Vec<(u32, u32)>,
    /// The draft head, once started
    drafter: Option<drafter::Drafter>,
    /// The model's vocabulary and the draft vocabulary's size, set when the drafter starts
    draft_vocab: (usize, usize),
    pending: Option<anyhow::Error>,
    captures: u64,
    replays: u64,
    n_ctx: usize,
    n_batch: usize,
    /// Positions the rope angle tables cover, the context or the furthest position a pass has used
    rope_positions: usize,
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
            pinned: Vec::new(),
            res: exec::Resources::default(),
            coproc: None,
            pool: exec::BufPool::new(Pool::Device(ordinal as u32)),
            prepared: Vec::new(),
            last: None,
            dense: None,
            hidden: None,
            ring: None,
            tree_tokens: Vec::new(),
            drafter: None,
            draft_vocab: (0, 0),
            pending: None,
            captures: 0,
            replays: 0,
            n_ctx: 0,
            n_batch: 0,
            rope_positions: 0,
            profile: HashMap::new(),
        })
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

    /// Decode launch sequences captured into CUDA graphs and replays of captured graphs
    pub fn graph_stats(&self) -> (u64, u64) {
        (self.captures, self.replays)
    }

    /// The device as a pool for the allocation events
    pub fn pool(&self) -> Pool {
        Pool::Device(self.ctx.ordinal() as u32)
    }

    /// Makes the feature ring that every pass's hidden rows of this width join from now on
    pub fn start_ring(&mut self, cols: usize) -> Result<()> {
        let ring = features::FeatureRing::new(self, features::RING_ROWS, cols, self.host.n_seq())?;
        self.ring = Some(ring);
        Ok(())
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

    fn dense_row(&mut self, row: usize) -> Result<Vec<f32>> {
        exec::dense_row(self, row)
    }

    fn hidden_rows(&mut self) -> Result<Vec<f32>> {
        exec::hidden_rows(self)
    }

    fn drafter_start(&mut self, cfg: kf_core::HeadConfig, embed: kf_core::Wid, head: kf_core::Wid, n_vocab: usize, draft_vocab: usize, batch: usize, weights: &dyn WeightSource) -> Result<()> {
        self.ctx.bind_to_thread()?;
        let n_seq = self.host.n_seq();
        self.start_ring(cfg.d)?;
        let d = drafter::Drafter::new(self, cfg, batch, n_seq, embed, head, weights)?;
        self.drafter = Some(d);
        self.draft_vocab = (n_vocab, draft_vocab.clamp(1, n_vocab));
        self.drafter_refresh_vocab()
    }

    fn drafter_step(&mut self) -> Result<Option<kf_core::TrainStep>> {
        let Some(mut d) = self.drafter.take() else { return Ok(None) };
        let r = d.step(self);
        self.drafter = Some(d);
        r
    }

    fn drafter_rows(&self) -> u64 {
        self.ring.as_ref().map_or(0, |r| r.appended)
    }

    fn drafter_hold(&mut self, seq: usize, row: usize, pos: u32) -> Result<()> {
        let Some(mut d) = self.drafter.take() else { return Ok(()) };
        let r = (|| {
            let h = self.hidden.ok_or_else(|| anyhow::anyhow!("the last pass kept no hidden rows"))?;
            anyhow::ensure!(row < h.rows, "row {row} of {} hidden rows", h.rows);
            d.hold(self, seq, h.ptr + (row * h.cols * 4) as u64, pos)
        })();
        self.drafter = Some(d);
        r
    }

    fn drafter_sync(&mut self, seq: usize) -> Result<()> {
        let Some(mut d) = self.drafter.take() else { return Ok(()) };
        let r = d.sync(self, seq);
        self.drafter = Some(d);
        r
    }

    fn drafter_draft(&mut self, seq: usize, own: u32, budget: usize) -> Result<kf_core::Tree> {
        let Some(mut d) = self.drafter.take() else { return Ok(kf_core::Tree::root(own)) };
        let r = d.draft(self, seq, own, budget);
        self.drafter = Some(d);
        r
    }

    fn drafter_refresh_vocab(&mut self) -> Result<()> {
        let Some(mut d) = self.drafter.take() else { return Ok(()) };
        let (n_vocab, v) = self.draft_vocab;
        let vocab = drafter::Drafter::vocab_from_ring(self, n_vocab, v);
        let r = d.set_vocab(self, &vocab);
        self.drafter = Some(d);
        r
    }

    fn drafter_save(&mut self) -> Result<Vec<u8>> {
        let d = self.drafter.as_ref().ok_or_else(|| anyhow::anyhow!("no draft head to save"))?;
        Ok(kf_core::head_save(&d.trainer.head.cfg, &d.vocab, &d.trainer.head.params_host(self)?))
    }

    fn drafter_load(&mut self, bytes: &[u8]) -> Result<()> {
        let Some(mut d) = self.drafter.take() else { anyhow::bail!("no draft head to load into") };
        let r = (|| {
            let (vocab, params) = kf_core::head_load(&d.trainer.head.cfg, bytes)?;
            d.trainer.head.set_params(self, &params)?;
            d.set_vocab(self, &vocab)
        })();
        self.drafter = Some(d);
        r
    }

    fn compact(&mut self, seq: usize, base: usize, keep: &[usize]) -> Result<()> {
        exec::compact(self, seq, base, keep)
    }

    fn truncate(&mut self, seq: usize, n: usize) -> Result<()> {
        if let Some(r) = self.ring.as_mut() {
            r.roll_back(seq);
        }
        if let Some(d) = self.drafter.as_mut() {
            d.forget(seq);
        }
        cuda_truncate_state(self, seq, n)
    }

    fn reset(&mut self) {
        if let Some(r) = self.ring.as_mut() {
            r.roll_back_all();
        }
        if let Some(d) = self.drafter.as_mut() {
            for seq in 0..self.host.n_seq() {
                d.forget(seq);
            }
        }
        cuda_reset_state(self)
    }

    fn reset_seq(&mut self, seq: usize) {
        if let Some(r) = self.ring.as_mut() {
            r.roll_back(seq);
        }
        if let Some(d) = self.drafter.as_mut() {
            d.forget(seq);
        }
        if let Err(e) = exec::truncate(self, Some(seq), 0) {
            self.pending.get_or_insert(e);
        }
    }

    fn save_state(&mut self, seq: usize) -> Result<Vec<u8>> {
        exec::save_state(self, seq)
    }

    fn load_state(&mut self, seq: usize, bytes: &[u8]) -> Result<()> {
        if let Some(r) = self.ring.as_mut() {
            r.roll_back(seq);
        }
        if let Some(d) = self.drafter.as_mut() {
            d.forget(seq);
        }
        exec::load_state(self, seq, bytes)
    }

    fn expert_cache_stats(&self) -> Vec<(usize, usize, u64, u64)> {
        let mut v: Vec<(usize, usize, u64, u64)> = self.layers.values().filter(|l| l.expert_cache_slots > 0).map(|l| {
            let (slots, hits, misses) = self.expert_cache_stats(l.layer);
            (l.layer, slots, hits, misses)
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

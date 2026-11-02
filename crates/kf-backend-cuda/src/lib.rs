//! CUDA backend, kernels are CUDA C compiled through NVRTC and cached as PTX
pub mod bench;
mod coproc;
pub mod draft;
mod exec;
mod experts;
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

pub use exec::Multi;
pub use registry::{KernelRef, KernelRegistry, cuda_kernel_for_op};

/// The device address of a slice on the backend's stream
pub fn device_ptr<T>(s: &CudaSlice<T>, b: &CudaBackend) -> u64 {
    exec::dptr(s, &b.stream)
}

/// A device slice of integers brought to the host after the stream drains
pub fn download_i32(s: &CudaSlice<i32>, b: &CudaBackend) -> Result<Vec<i32>> {
    let mut out = vec![0i32; s.len()];
    exec::dtoh(&b.stream, &mut out, exec::dptr(s, &b.stream))?;
    Ok(out)
}

/// A device slice of bytes brought to the host after the stream drains
pub fn download_u8(s: &CudaSlice<u8>, b: &CudaBackend) -> Result<Vec<u8>> {
    let mut out = vec![0u8; s.len()];
    exec::dtoh(&b.stream, &mut out, exec::dptr(s, &b.stream))?;
    Ok(out)
}

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
    /// The residual rows after every tap layer the last pass kept, in the graph's order
    taps: Vec<exec::Output>,
    /// Layer statistics the tap selection collects, while it does
    cal: Option<draft::Calibration>,
    /// The packed candidate rows the last pass ended in, when it ended in a top k
    cands: Option<exec::Output>,
    /// The draft block, once started
    draft: Option<draft::Drafter>,
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
    /// The gradients of the draft block's first learner over one batch of its ring, what parity tests compare
    #[doc(hidden)]
    pub fn draft_probe_grads(&mut self) -> Result<Option<Vec<Vec<f32>>>> {
        self.ctx.bind_to_thread()?;
        let Some(mut d) = self.draft.take() else { anyhow::bail!("no draft block") };
        let r = d.probe_grads(self);
        self.draft = Some(d);
        r
    }

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
            taps: Vec::new(), cal: None,
            cands: None,
            draft: None,
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

    /// Ends the draft block's runs and history of one slot or of every slot, after their rows rolled back
    fn draft_roll_back(&mut self, seq: Option<usize>) -> Result<()> {
        let Some(mut d) = self.draft.take() else { return Ok(()) };
        let r = match seq {
            Some(s) => d.roll_back(self, s),
            None => (0..self.host.n_seq()).try_for_each(|s| d.roll_back(self, s)),
        };
        self.draft = Some(d);
        r
    }

    /// A zeroed device buffer of n elements accounted to a kind
    pub(crate) fn alloc<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits>(&self, n: usize, kind: AllocKind, what: &str) -> Result<CudaSlice<T>> {
        exec::dev_alloc(&self.stream, self.pool(), n, kind, what)
    }

    /// A device copy of host data accounted to a kind
    pub fn upload<T: cudarc::driver::DeviceRepr + Unpin>(&self, data: &[T], kind: AllocKind, what: &str) -> Result<CudaSlice<T>> {
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

    fn tap_rows(&mut self) -> Result<Vec<f32>> {
        exec::tap_rows(self)
    }

    fn draft_calibrate(&mut self, n_layers: usize) -> Result<()> {
        anyhow::ensure!(n_layers >= 1, "a tap selection over no layers");
        self.cal = Some(draft::Calibration::new(n_layers));
        Ok(())
    }

    fn draft_calibrated(&self) -> u64 {
        self.cal.as_ref().map_or(0, |c| c.retained)
    }

    fn draft_select_taps(&mut self, m: usize, cka: bool) -> Result<Vec<usize>> {
        self.ctx.bind_to_thread()?;
        let cal = self.cal.take().ok_or_else(|| anyhow::anyhow!("no tap statistics were collected"))?;
        anyhow::ensure!(cal.retained >= kf_core::DRAFT_CAL_BLOCK as u64, "{} rows kept for the tap selection, at least {} are needed", cal.retained, kf_core::DRAFT_CAL_BLOCK);
        cal.select(self, m, cka)
    }

    fn draft_start(&mut self, cfg: kf_core::DraftConfig, tap_layers: &[usize], embed: kf_core::Wid, head: kf_core::Wid, norm: kf_core::Wid, n_vocab: usize, eos: Option<u32>, draft_vocab: usize, batch: usize, weights: &dyn kf_core::WeightSource) -> Result<()> {
        self.ctx.bind_to_thread()?;
        let n_seq = self.host.n_seq();
        let d = draft::Drafter::new(self, cfg, tap_layers.to_vec(), batch, n_seq, embed, head, norm, n_vocab, eos, draft_vocab, weights)?;
        self.draft = Some(d);
        Ok(())
    }

    fn draft_step(&mut self) -> Result<Option<kf_core::TrainStep>> {
        let Some(mut d) = self.draft.take() else { return Ok(None) };
        let r = d.step(self);
        self.draft = Some(d);
        r
    }

    fn draft_rows(&self) -> u64 {
        self.draft.as_ref().map_or(0, |d| d.ring.meta.appended)
    }

    fn draft_tree(&mut self, seq: usize, path: &[usize], root: u32, rows: usize) -> Result<kf_core::Tree> {
        let Some(mut d) = self.draft.take() else { anyhow::bail!("no draft block") };
        let r = d.draft_tree(self, seq, path, root, rows);
        self.draft = Some(d);
        r
    }

    fn draft_refresh_vocab(&mut self) -> Result<()> {
        let Some(mut d) = self.draft.take() else { anyhow::bail!("no draft block") };
        let r = d.refresh_vocab(self);
        self.draft = Some(d);
        r
    }

    fn draft_save(&mut self) -> Result<Vec<u8>> {
        self.draft.as_ref().ok_or_else(|| anyhow::anyhow!("no draft block to save"))?.save(self)
    }

    fn draft_load(&mut self, bytes: &[u8]) -> Result<()> {
        let Some(mut d) = self.draft.take() else { anyhow::bail!("no draft block to load into") };
        let r = d.load(self, bytes);
        self.draft = Some(d);
        r
    }

    fn compact(&mut self, seq: usize, base: usize, keep: &[usize]) -> Result<()> {
        exec::compact(self, seq, base, keep)
    }

    fn truncate(&mut self, seq: usize, n: usize) -> Result<()> {
        self.draft_roll_back(Some(seq))?;
        cuda_truncate_state(self, seq, n)
    }

    fn reset(&mut self) {
        if let Err(e) = self.draft_roll_back(None) {
            kf_core::event::error("drafter", &[("detail", format!("{e:#}"))]);
        }
        cuda_reset_state(self)
    }

    fn reset_seq(&mut self, seq: usize) {
        if let Err(e) = self.draft_roll_back(Some(seq)).and_then(|()| exec::truncate(self, Some(seq), 0)) {
            self.pending.get_or_insert(e);
        }
    }

    fn save_state(&mut self, seq: usize) -> Result<Vec<u8>> {
        exec::save_state(self, seq)
    }

    fn load_state(&mut self, seq: usize, bytes: &[u8]) -> Result<()> {
        self.draft_roll_back(Some(seq))?;
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

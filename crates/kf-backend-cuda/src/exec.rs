use crate::CudaBackend;
use crate::coproc;
use crate::experts;
use crate::persist;
use crate::registry::{KernelRegistry, cuda_kernel_for_op};
use crate::sources;

use anyhow::{Context, Result, anyhow, bail, ensure};
use cudarc::driver::{CudaEvent, CudaFunction, CudaGraph, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PinnedHostSlice, PushKernelArg, result, sys};
use kf_core::{Act, AllocKind, Gating, Graph, Inputs, Node, Op, Plan, Pool, RopeKind, RopeParams, Tid, WeightNorm, WeightSource, Wid};
use kf_quant::ActKind;
use std::collections::HashMap;
use std::f32::consts::PI;
use std::ops::Range;
use std::sync::Arc;

/// One kernel argument, pointers travel as raw device addresses
#[derive(Clone, Copy, Debug)]
pub(crate) enum Arg {
    P(u64),
    I(i32),
    L(i64),
    F(f32),
}

#[derive(Clone, Debug)]
pub(crate) struct Launch {
    pub func: CudaFunction,
    pub grid: (u32, u32, u32),
    pub block: u32,
    pub smem: u32,
    pub args: Vec<Arg>,
}

impl Launch {
    pub fn run(&self, stream: &CudaStream) -> Result<()> {
        let mut b = stream.launch_builder(&self.func);
        for a in &self.args {
            match a {
                Arg::P(v) => b.arg(v),
                Arg::I(v) => b.arg(v),
                Arg::L(v) => b.arg(v),
                Arg::F(v) => b.arg(v),
            };
        }
        let cfg = LaunchConfig { grid_dim: self.grid, block_dim: (self.block, 1, 1), shared_mem_bytes: self.smem };
        unsafe { b.launch(cfg) }.context("kernel launch")?;
        Ok(())
    }

    /// Launches with every block of the grid guaranteed resident at once, refused when the grid does not fit the device
    pub fn run_cooperative(&self, stream: &CudaStream) -> Result<()> {
        let mut b = stream.launch_builder(&self.func);
        for a in &self.args {
            match a {
                Arg::P(v) => b.arg(v),
                Arg::I(v) => b.arg(v),
                Arg::L(v) => b.arg(v),
                Arg::F(v) => b.arg(v),
            };
        }
        let cfg = LaunchConfig { grid_dim: self.grid, block_dim: (self.block, 1, 1), shared_mem_bytes: self.smem };
        unsafe { b.launch_cooperative(cfg) }.context("cooperative kernel launch")?;
        Ok(())
    }
}

/// A grid stride launch over n elements in blocks of 256
pub(crate) fn flat(func: CudaFunction, n: usize, args: Vec<Arg>) -> Launch {
    Launch { func, grid: (n.div_ceil(256).clamp(1, 1 << 20) as u32, 1, 1), block: 256, smem: 0, args }
}

pub(crate) fn dptr<T>(s: &CudaSlice<T>, stream: &CudaStream) -> u64 {
    s.device_ptr(stream).0
}

/// Copies host data to a device address on the stream, pageable sources are staged before return
pub(crate) fn htod<T>(stream: &CudaStream, ptr: u64, data: &[T]) -> Result<()> {
    if data.is_empty() {
        return Ok(());
    }
    unsafe { result::memcpy_htod_async(ptr, data, stream.cu_stream()) }.context("copy to device")?;
    Ok(())
}

/// Copies a device range to the host and waits for it
pub(crate) fn dtoh<T>(stream: &CudaStream, data: &mut [T], ptr: u64) -> Result<()> {
    if !data.is_empty() {
        unsafe { result::memcpy_dtoh_async(data, ptr, stream.cu_stream()) }.context("copy to host")?;
    }
    stream.synchronize().context("stream sync")?;
    Ok(())
}

/// Bytes of one row of n elements of a ggml type, failing when n is not a whole number of blocks
pub(crate) fn row_bytes(ty: u32, n: usize) -> Result<usize> {
    let info = kf_quant::info(ty).ok_or_else(|| anyhow!("unknown ggml type {ty}"))?;
    ensure!(n.is_multiple_of(info.block), "{} rows of {n} elements are not whole blocks of {}", info.name, info.block);
    Ok(info.row_bytes(n))
}

/// Code of the activation rounding the reference pairs with a weight type, zero when none
pub(crate) fn round_code(ty: u32) -> Result<i32> {
    let q = kf_quant::by_id(ty).ok_or_else(|| anyhow!("unknown ggml type {ty}"))?;
    Ok(match kf_quant::act_kind(q) {
        ActKind::F32 => 0,
        ActKind::F16 => 1,
        ActKind::Bf16 => 2,
        ActKind::Q8_0 | ActKind::Q8_1 => 3,
        ActKind::Q8K => 4,
    })
}

/// Rounds rows of n activations into out as the reference does before a product against the type
pub(crate) fn round_launch(k: &mut KernelRegistry, ty: u32, x: u64, out: u64, rows: usize, n: usize) -> Result<Option<Launch>> {
    let code = round_code(ty)?;
    if code == 0 {
        return Ok(None);
    }
    Ok(Some(Launch { func: k.function("elementwise", "round_act")?, grid: (n.div_ceil(256) as u32, rows as u32, 1), block: 256, smem: 0, args: vec![Arg::P(x), Arg::P(out), Arg::I(n as i32), Arg::I(code)] }))
}

/// The rounded copy of x for a weight type with its launch pushed, or x itself when none is needed
fn rounded(k: &mut KernelRegistry, ty: u32, x: u64, out: u64, rows: usize, n: usize, v: &mut Vec<Launch>) -> Result<u64> {
    match round_launch(k, ty, x, out, rows, n)? {
        Some(l) => {
            v.push(l);
            Ok(out)
        }
        None => Ok(x),
    }
}

/// A zeroed device buffer of n elements accounted to a kind, an alloc_failed event when the device is out of memory
pub(crate) fn dev_alloc<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits>(stream: &Arc<CudaStream>, pool: Pool, n: usize, kind: AllocKind, what: &str) -> Result<CudaSlice<T>> {
    let bytes = (n * std::mem::size_of::<T>()) as u64;
    match stream.alloc_zeros::<T>(n) {
        Ok(s) => {
            kf_core::event::alloc(pool, kind, bytes);
            Ok(s)
        }
        Err(e) => {
            let free = result::mem_get_info().map_or(0, |(f, _)| f as u64);
            kf_core::event::alloc_failed(pool, kind, bytes, free);
            Err(anyhow!("{what}: {bytes} bytes of {} with {free} free: {e}", kind.name()))
        }
    }
}

/// A device copy of host data accounted to a kind, an alloc_failed event when the device is out of memory
pub(crate) fn dev_upload<T: cudarc::driver::DeviceRepr + Unpin>(stream: &Arc<CudaStream>, pool: Pool, data: &[T], kind: AllocKind, what: &str) -> Result<CudaSlice<T>> {
    let bytes = std::mem::size_of_val(data) as u64;
    match stream.clone_htod(data) {
        Ok(s) => {
            kf_core::event::alloc(pool, kind, bytes);
            Ok(s)
        }
        Err(e) => {
            let free = result::mem_get_info().map_or(0, |(f, _)| f as u64);
            kf_core::event::alloc_failed(pool, kind, bytes, free);
            Err(anyhow!("{what}: {bytes} bytes of {} with {free} free: {e}", kind.name()))
        }
    }
}

/// Free device buffers kept across prepared graphs so shape changes reuse memory
pub(crate) struct BufPool {
    free: Vec<CudaSlice<f32>>,
    pool: Pool,
}

impl BufPool {
    pub fn new(pool: Pool) -> Self {
        Self { free: Vec::new(), pool }
    }

    fn take(&mut self, stream: &Arc<CudaStream>, n: usize) -> Result<CudaSlice<f32>> {
        let best = self.free.iter().enumerate().filter(|(_, s)| s.len() >= n).min_by_key(|(_, s)| s.len()).map(|(i, _)| i);
        match best {
            Some(i) => Ok(self.free.swap_remove(i)),
            None => dev_alloc::<f32>(stream, self.pool, n.max(256).next_power_of_two(), AllocKind::Compute, "activation buffer"),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Output {
    pub ptr: u64,
    pub rows: usize,
    pub cols: usize,
}

/// Words per row of the staged integer inputs: token, position, slot, attend bound, four rotary streams, and the
/// three words of a tree row
const IN_WORDS: usize = 11;

/// Threads of the top k kernel, KF_TOPK_THREADS of sample.cu
const TOPK_THREADS: u32 = 1024;

/// Most candidates the top k kernel writes per row, KF_TOPK_MAX of sample.cu
const TOPK_MAX: usize = 128;

/// Slices a row of the top k is cut into, one block each of about eight thousand columns, at most KF_TOPK_SLICES of sample.cu
pub(crate) fn topk_slices(cols: usize) -> usize {
    (cols / 8192).clamp(1, 16)
}

/// Scratch of a top k node in floats, the slices' candidate keys as pairs of words then the slices' maxima and sums
fn topk_scratch(rows: usize, cols: usize, k: usize) -> Vec<usize> {
    let s = topk_slices(cols);
    vec![rows * s * k * 2, rows * s * 2]
}

/// The two launches of a top k over rows of cols logits, the slices then the merge, keys and stats being its scratch
pub(crate) fn topk_launches(kernels: &mut KernelRegistry, module: &str, x: u64, out: u64, keys: u64, stats: u64, rows: usize, cols: usize, k: usize) -> Result<Vec<Launch>> {
    use Arg::{I, P};
    let s = topk_slices(cols);
    Ok(vec![
        Launch { func: kernels.function(module, "topk_slices")?, grid: (s as u32, rows as u32, 1), block: TOPK_THREADS, smem: 0, args: vec![P(x), P(keys), P(stats), I(cols as i32), I(k as i32), I(s as i32)] },
        Launch { func: kernels.function(module, "topk_merge")?, grid: (rows as u32, 1, 1), block: TOPK_THREADS, smem: 0, args: vec![P(keys), P(stats), P(out), I(cols as i32), I(k as i32), I(s as i32)] },
    ])
}

pub(crate) struct RopeTable {
    _buf: CudaSlice<f32>,
    ptr: u64,
    /// The cosine and sine of every pair angle at the first n_pos positions, half pairs per position
    _cs: CudaSlice<f32>,
    cs_ptr: u64,
    n_pos: usize,
    half: usize,
    mscale: f32,
}

/// The angle table of a rope over n_pos positions from its frequencies
fn rope_angles(b: &mut CudaBackend, freq: u64, half: usize, n_pos: usize) -> Result<(CudaSlice<f32>, u64)> {
    let cs = b.alloc::<f32>((2 * n_pos * half).max(2), AllocKind::Weights, "rope angles")?;
    let cs_ptr = dptr(&cs, &b.stream);
    if half > 0 {
        flat(b.kernels.function("rope", "rope_angles")?, n_pos * half, vec![Arg::P(freq), Arg::P(cs_ptr), Arg::I(half as i32), Arg::I(n_pos as i32)]).run(&b.stream)?;
    }
    Ok((cs, cs_ptr))
}

/// Grows every rope angle table to cover positions below n_pos, which stales the launches holding the old tables
fn ensure_rope_positions(b: &mut CudaBackend, n_pos: usize) -> Result<()> {
    let short: Vec<String> = b.res.rope.iter().filter(|(_, t)| t.n_pos < n_pos).map(|(k, _)| k.clone()).collect();
    for key in short {
        let (ptr, half) = b.res.rope.get(&key).map(|t| (t.ptr, t.half)).context("rope table")?;
        let (cs, cs_ptr) = rope_angles(b, ptr, half, n_pos)?;
        let t = b.res.rope.get_mut(&key).context("rope table")?;
        t._cs = cs;
        t.cs_ptr = cs_ptr;
        t.n_pos = n_pos;
        b.res.epoch += 1;
    }
    Ok(())
}

/// Key and value rows of one layer in half precision, one ring of resident rows per sequence slot with the rows each slot holds, older rows pinned on the host when the ring is shorter than the cache
pub(crate) struct DevKv {
    k: CudaSlice<u16>,
    v: CudaSlice<u16>,
    /// Rows the cache logically holds per slot
    cap: usize,
    /// Rows of the device ring per slot, the cache streams from the host when fewer than cap
    resident: usize,
    /// Host rows per slot for the positions that age out of the ring, cap less resident
    host_k: Option<PinnedHostSlice<u16>>,
    host_v: Option<PinnedHostSlice<u16>>,
    pub n: Vec<usize>,
    kw: usize,
    vw: usize,
}

impl DevKv {
    fn streams(&self) -> bool {
        self.resident < self.cap
    }

    /// Rows per slot the host keeps, positions below this never leave the ring
    fn host_rows(&self) -> usize {
        self.cap - self.resident
    }
}

/// Rows one streamed chunk carries from the host
pub(crate) const STREAM_CHUNK: usize = 1024;

/// Two device buffers streamed chunks land in, with the events that order copies against the kernels reading them
pub(crate) struct Stage {
    k: [CudaSlice<u16>; 2],
    v: [CudaSlice<u16>; 2],
    kw: usize,
    vw: usize,
    ready: [CudaEvent; 2],
    free: [CudaEvent; 2],
}

/// Device copies of weights, state, and tables, epoch counts allocations that move pointers
#[derive(Default)]
pub(crate) struct Resources {
    /// The thread copying experts into slots behind the passes, declared first so it stops before any buffer it
    /// writes is freed
    pub copier: Option<experts::Copier>,
    weights: HashMap<Wid, (CudaSlice<u8>, u64)>,
    vecs: HashMap<Wid, (CudaSlice<f32>, u64)>,
    rope: HashMap<String, RopeTable>,
    pub kv: HashMap<usize, DevKv>,
    pub experts: HashMap<usize, experts::ExpertCache>,
    /// Buffers experts stream through during passes of many rows
    pub expert_stage: Option<experts::ExpertStage>,
    /// The pinned ring the host's slot table writes go through
    pub table_stage: Option<(PinnedHostSlice<i32>, usize)>,
    /// KF8 copies of weights for the tensor core product
    pub kf8: crate::kf8::Store,
    stage: Option<Stage>,
    /// Partial softmaxes of a streamed attention pass, grown to the largest pass seen
    parts: Option<CudaSlice<f32>>,
    /// Arrival counts of the decode attention's part blocks, one per row and kv head then one per row, zero between passes
    counters: Option<CudaSlice<i32>>,
    pub epoch: u64,
}

impl Resources {
    pub fn weight(&self, w: Wid) -> Result<u64> {
        self.weights.get(&w).map(|x| x.1).ok_or_else(|| anyhow!("weight {} is not on the device", w.0))
    }
    pub fn vec(&self, w: Wid) -> Result<u64> {
        self.vecs.get(&w).map(|x| x.1).ok_or_else(|| anyhow!("vector {} is not on the device", w.0))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Place {
    Device,
    Host,
    Experts,
    /// Attention over a cache streamed from the host past its resident window
    Stream,
}

#[derive(Clone, Debug)]
pub(crate) enum Segment {
    Device(Range<usize>),
    Host(usize),
    Experts(usize),
    Stream(usize),
}

struct Captured(CudaGraph);

// The graph is only touched through the owning backend, which needs exclusive access
unsafe impl Send for Captured {}

/// Prepared graphs kept across passes, the shapes a chain head and a decode loop alternate between
pub(crate) const PREPARED_KEPT: usize = 8;

/// How a node's launch folds into its neighbours, decided once per prepared graph before buffers are placed
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) enum Fusion {
    #[default]
    None,
    /// Work done by another node's launch
    Skip,
    /// A norm or gated activation writing the int8 activations of its product consumers into its own output buffer
    QuantOut { code: i32 },
    /// A norm whose only reader adds the residual res, writing the sum into that add node's output
    NormAdd { add: usize, res: Tid },
    /// A product reading int8 activations from node act's output buffer, adding the residual tensor res into its own output
    MatMul { act: Option<usize>, res: Option<Tid> },
    /// A rope reading src with the per head norm of node norm folded in, appending its keys with the value tensor to the cache of the attention node
    Rope { src: Tid, norm: Option<usize>, append: Option<(usize, Tid)> },
    /// An attention whose keys and values the rope before it already appended, quantizing its output for the
    /// product that follows when quant names the activation rounding, and norming, rotating, and appending the
    /// query and key itself when rope describes the layer's two rope nodes
    Attention { appended: bool, quant: Option<i32>, rope: Option<RopeFold> },
}

/// The two ropes a decode attention folds in: the query rope node with its source and per head norm, and the key
/// rope's source and norm with the value tensor it appended
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RopeFold {
    pub qi: usize,
    pub qsrc: Tid,
    pub qnorm: Option<usize>,
    pub ksrc: Tid,
    pub knorm: Option<usize>,
    pub v: Tid,
}

/// A graph signature with its buffer plan, launches, and decode CUDA graphs, filled as nodes first run
pub(crate) struct Prepared {
    pub nodes: Vec<Node>,
    output: Tid,
    n_tok: usize,
    n_pos: usize,
    pub place: Vec<Place>,
    segments: Vec<Segment>,
    pub shapes: Vec<Option<(usize, usize)>>,
    loc: Vec<Option<(usize, usize)>>,
    pub scratch: Vec<Vec<usize>>,
    bufs: Vec<CudaSlice<f32>>,
    ptrs: Vec<u64>,
    free: Vec<usize>,
    buf_last: Vec<usize>,
    last: Vec<usize>,
    rope_keys: Vec<Option<String>>,
    /// The integer inputs of a pass, tokens, positions, slots, attend bounds, and the four rotary streams of each row, in one buffer
    _inputs: CudaSlice<i32>,
    /// Pinned host staging of the integer inputs, filled then copied in one transfer
    stage_in: PinnedHostSlice<i32>,
    /// Pinned host landing of the output, made at the output's shape on the first pass
    stage_out: Option<PinnedHostSlice<f32>>,
    _hid: CudaSlice<f32>,
    rope_ptr: u64,
    /// Query tiles of the tensor core attention, a count then row start, rows, and slot per tile
    _tiles: CudaSlice<i32>,
    tiles_ptr: u64,
    n_tiles_max: usize,
    /// Segment tables of grouped products, filled as launches are built
    _segs: CudaSlice<i64>,
    segs_ptr: u64,
    segs_used: std::cell::Cell<usize>,
    pub tok_ptr: u64,
    pos_ptr: u64,
    seq_ptr: u64,
    att_ptr: u64,
    hid_ptr: u64,
    n_hid: usize,
    hidden_cols: usize,
    /// Whether the pass is a tree, its rows seeing their ancestors by mask, which keeps its own kernels and graphs
    tree: bool,
    tree_ptr: u64,
    /// The residual rows before the final norm, kept through the pass
    hidden: Option<Tid>,
    epoch: Option<u64>,
    launches: Vec<Option<Vec<Launch>>>,
    graphs: Vec<Option<Captured>>,
    /// The persistent form of each device segment's launches, built with its graph for single row decode passes
    persist: Vec<Option<persist::PersistPlan>>,
    /// Consumers of every tensor
    uses: Vec<Vec<usize>>,
    pub fused: Vec<Fusion>,
    /// Nodes whose output shares the buffer of another tensor, the add folded into a product
    alias: Vec<Option<Tid>>,
    /// Leader of the product group each grouped product belongs to, the leader launching for all
    grouped: Vec<Option<usize>>,
    /// Outputs of other nodes a node writes, placed when it is planned so nothing live then shares their buffers
    early: Vec<Vec<Tid>>,
    /// The other rope of a layer a fused rope launches with, the later of the two launching both
    pair: Vec<Option<usize>>,
    /// Scratch buffers that outlive their node, freed when the node named runs
    deferred: Vec<(usize, usize)>,
}

/// Words of segment table a prepared graph can hold
const SEGS_WORDS: usize = 8192;

impl Prepared {
    fn new(b: &mut CudaBackend, g: &Graph, inputs: &Inputs, weights: &dyn WeightSource) -> Result<Self> {
        let output = g.output.ok_or_else(|| anyhow!("graph has no output"))?;
        let n_t = g.n_tensors();
        ensure!(g.nodes.iter().any(|n| n.out == output), "tensor {} used before it was computed", output.0);
        let mut last = vec![0usize; n_t];
        for (i, node) in g.nodes.iter().enumerate() {
            last[node.out.0 as usize] = last[node.out.0 as usize].max(i);
            for t in &node.inputs {
                ensure!((t.0 as usize) < n_t, "tensor {} is outside the graph", t.0);
                last[t.0 as usize] = last[t.0 as usize].max(i);
            }
        }
        last[output.0 as usize] = usize::MAX;
        if let Some(h) = g.hidden {
            ensure!((h.0 as usize) < n_t && g.nodes.iter().any(|n| n.out == h), "hidden tensor {} is not computed by the graph", h.0);
            last[h.0 as usize] = usize::MAX;
        }
        let place: Vec<Place> = g.nodes.iter().map(|n| place_node(b, n, weights, inputs.tokens.len())).collect();
        let mut segments = Vec::new();
        let mut i = 0;
        while i < place.len() {
            match place[i] {
                Place::Device => {
                    let start = i;
                    while i < place.len() && place[i] == Place::Device {
                        i += 1;
                    }
                    segments.push(Segment::Device(start..i));
                }
                Place::Host => {
                    segments.push(Segment::Host(i));
                    i += 1;
                }
                Place::Experts => {
                    segments.push(Segment::Experts(i));
                    i += 1;
                }
                Place::Stream => {
                    segments.push(Segment::Stream(i));
                    i += 1;
                }
            }
        }
        let rope_keys = g.nodes.iter().map(|n| if let Op::Rope { params, head_dim, .. } = &n.op { Some(format!("{params:?}/{head_dim}")) } else { None }).collect();
        let n_in = inputs.tokens.len().max(1);
        let inputs_dev = b.alloc::<i32>(IN_WORDS * n_in, AllocKind::Compute, "pass input buffer")?;
        // Plain page locked memory, the write combined default being slow to read back and to fill piecemeal
        let stage_in = unsafe { b.ctx.alloc_pinned_with_flags::<i32>(IN_WORDS * n_in, 0) }.context("pinned pass inputs")?;
        let in_ptr = dptr(&inputs_dev, &b.stream);
        let hid = b.alloc::<f32>(inputs.hidden.len().max(1), AllocKind::Compute, "hidden buffer")?;
        let n_tiles_max = inputs.tokens.len().div_ceil(FLASH_BQ) + b.host.n_seq();
        let tiles = b.alloc::<i32>(4 + 4 * n_tiles_max, AllocKind::Compute, "attention tile buffer")?;
        let segs = b.alloc::<i64>(SEGS_WORDS, AllocKind::Compute, "product segment tables")?;
        let n_nodes = g.nodes.len();
        let mut uses = vec![Vec::new(); n_t];
        for (i, node) in g.nodes.iter().enumerate() {
            for t in &node.inputs {
                uses[t.0 as usize].push(i);
            }
        }
        Ok(Self {
            nodes: g.nodes.clone(),
            output,
            n_tok: inputs.tokens.len(),
            n_pos: inputs.positions.len(),
            place,
            graphs: (0..segments.len()).map(|_| None).collect(),
            persist: (0..segments.len()).map(|_| None).collect(),
            segments,
            shapes: vec![None; n_t],
            loc: vec![None; n_t],
            scratch: vec![Vec::new(); n_nodes],
            bufs: Vec::new(),
            ptrs: Vec::new(),
            free: Vec::new(),
            buf_last: Vec::new(),
            last,
            rope_keys,
            tok_ptr: in_ptr,
            pos_ptr: in_ptr + (4 * n_in) as u64,
            seq_ptr: in_ptr + (8 * n_in) as u64,
            att_ptr: in_ptr + (12 * n_in) as u64,
            hid_ptr: dptr(&hid, &b.stream),
            tiles_ptr: dptr(&tiles, &b.stream),
            n_tiles_max,
            _tiles: tiles,
            segs_ptr: dptr(&segs, &b.stream),
            segs_used: std::cell::Cell::new(0),
            _segs: segs,
            n_hid: inputs.hidden.len(),
            hidden_cols: inputs.hidden_cols,
            tree: !inputs.tree.is_empty(),
            tree_ptr: in_ptr + (32 * n_in) as u64,
            hidden: g.hidden,
            _inputs: inputs_dev,
            stage_in,
            stage_out: None,
            _hid: hid,
            epoch: None,
            launches: vec![None; n_nodes],
            uses,
            fused: vec![Fusion::None; n_nodes],
            alias: vec![None; n_nodes],
            grouped: vec![None; n_nodes],
            early: vec![Vec::new(); n_nodes],
            pair: vec![None; n_nodes],
            deferred: Vec::new(),
            rope_ptr: in_ptr + (16 * n_in) as u64,
        })
    }

    /// Uploads a segment table and returns its device address
    fn segment_table(&self, stream: &CudaStream, words: &[i64]) -> Result<u64> {
        let at = self.segs_used.get();
        ensure!(at + words.len() <= SEGS_WORDS, "segment tables past {SEGS_WORDS} words");
        let ptr = self.segs_ptr + (8 * at) as u64;
        htod(stream, ptr, words)?;
        self.segs_used.set(at + words.len());
        Ok(ptr)
    }

    /// Index of the node computing a tensor, none for a graph input
    fn producer(&self, t: Tid) -> Option<usize> {
        self.nodes.iter().position(|n| n.out == t)
    }

    fn matches(&self, g: &Graph, inputs: &Inputs) -> bool {
        self.n_tok == inputs.tokens.len()
            && self.n_pos == inputs.positions.len()
            && self.n_hid == inputs.hidden.len()
            && self.hidden_cols == inputs.hidden_cols
            && self.tree == !inputs.tree.is_empty()
            && self.hidden == g.hidden
            && Some(self.output) == g.output
            && self.nodes.len() == g.nodes.len()
            && self.nodes.iter().zip(&g.nodes).all(|(a, b)| a.op == b.op && a.inputs == b.inputs && a.out == b.out && a.layer == b.layer)
    }

    pub fn release(self, pool: &mut BufPool) {
        drop(self.graphs);
        pool.free.extend(self.bufs);
    }

    pub fn ptr(&self, t: Tid) -> Result<u64> {
        let (b, off) = self.loc[t.0 as usize].ok_or_else(|| anyhow!("tensor {} has no device buffer", t.0))?;
        Ok(self.ptrs[b] + 4 * off as u64)
    }

    pub fn shape(&self, t: Tid) -> Result<(usize, usize)> {
        self.shapes[t.0 as usize].ok_or_else(|| anyhow!("tensor {} used before it was computed", t.0))
    }

    pub fn scratch_ptr(&self, i: usize, k: usize) -> u64 {
        self.ptrs[self.scratch[i][k]]
    }

    /// A buffer holding at least n floats, the smallest free one or a fresh one from the pool
    fn assign(&mut self, pool: &mut BufPool, stream: &Arc<CudaStream>, n: usize, last: usize) -> Result<usize> {
        let best = self.free.iter().enumerate().filter(|(_, b)| self.bufs[**b].len() >= n).min_by_key(|(_, b)| self.bufs[**b].len()).map(|(i, _)| i);
        let b = match best {
            Some(i) => self.free.swap_remove(i),
            None => {
                let s = pool.take(stream, n)?;
                self.ptrs.push(dptr(&s, stream));
                self.bufs.push(s);
                self.buf_last.push(0);
                self.bufs.len() - 1
            }
        };
        self.buf_last[b] = last;
        Ok(b)
    }

    /// Places the output of node i at a shape, aliasing views and drawing scratch, then frees dead buffers
    pub fn plan(&mut self, pool: &mut BufPool, stream: &Arc<CudaStream>, i: usize, shape: (usize, usize), scratch: &[usize]) -> Result<()> {
        let node = &self.nodes[i];
        let out = node.out.0 as usize;
        let tensors: Vec<usize> = node.inputs.iter().chain(std::iter::once(&node.out)).map(|t| t.0 as usize).collect();
        let last = self.last[out];
        let view = match &node.op {
            _ if self.alias[i].is_some() => Some((self.alias[i].expect("alias"), 0)),
            Op::Reshape { .. } => Some((node.inputs[0], 0)),
            Op::LastRows { .. } => {
                let (rows, cols) = self.shape(node.inputs[0])?;
                Some((node.inputs[0], rows.saturating_sub(shape.0) * cols))
            }
            Op::Rows { start, .. } => {
                let (_, cols) = self.shape(node.inputs[0])?;
                Some((node.inputs[0], start * cols))
            }
            _ => None,
        };
        self.shapes[out] = Some(shape);
        match view {
            Some((src, skip)) => {
                let (b, off) = self.loc[src.0 as usize].ok_or_else(|| anyhow!("view of tensor {} with no buffer", src.0))?;
                self.loc[out] = Some((b, off + skip));
                self.buf_last[b] = self.buf_last[b].max(last);
            }
            // A grouped product's output was placed when its leader was planned
            None if self.loc[out].is_some() => {}
            None => {
                let b = self.assign(pool, stream, shape.0 * shape.1, last)?;
                self.loc[out] = Some((b, 0));
            }
        }
        let early = std::mem::take(&mut self.early[i]);
        for t in &early {
            let (r, c) = self.shape(*t)?;
            let b = self.assign(pool, stream, r * c, self.last[t.0 as usize])?;
            self.loc[t.0 as usize] = Some((b, 0));
        }
        self.early[i] = early;
        // A quantizing attention's scratch feeds the products reading its output, so it lives as long as the output
        let scratch_last = if matches!(self.fused[i], Fusion::Attention { quant: Some(_), .. }) { last } else { i };
        let mut taken = Vec::with_capacity(scratch.len());
        for &n in scratch {
            taken.push(self.assign(pool, stream, n.max(1), scratch_last)?);
        }
        if scratch_last == i {
            self.free.extend(taken.iter().copied());
        } else {
            self.deferred.extend(taken.iter().map(|b| (scratch_last, *b)));
        }
        self.scratch[i] = taken;
        for t in tensors {
            if let Some((b, _)) = self.loc[t]
                && self.buf_last[b] == i
                && !self.free.contains(&b)
            {
                self.free.push(b);
            }
        }
        let deferred = std::mem::take(&mut self.deferred);
        for (k, b) in deferred {
            if k == i {
                if !self.free.contains(&b) {
                    self.free.push(b);
                }
            } else {
                self.deferred.push((k, b));
            }
        }
        Ok(())
    }
}

/// Weights a node reads through device kernels, rope factors excluded since the frequency table is built on the host
fn op_weights(op: &Op) -> Vec<Wid> {
    match op {
        Op::Embed { weight } | Op::MulWeight { weight } | Op::AddWeight { weight } | Op::DivWeight { weight } => vec![*weight],
        Op::MatMul { weight, bias, scale } => [Some(*weight), *bias, *scale].into_iter().flatten().collect(),
        Op::RmsNorm { weight, bias, .. } | Op::LayerNorm { weight, bias, .. } => [*weight, *bias].into_iter().flatten().collect(),
        Op::Attention { sinks, .. } => sinks.iter().copied().collect(),
        Op::MoeFfn { router, router_bias, probs_bias, gate_exps, up_exps, down_exps, .. } => {
            [Some(*router), *router_bias, *probs_bias, *gate_exps, Some(*up_exps), *down_exps].into_iter().flatten().collect()
        }
        _ => Vec::new(),
    }
}

/// Device, host, or expert cache execution for a node from kernels, quant modules, and the plan
fn place_node(b: &CudaBackend, node: &Node, weights: &dyn WeightSource, n_tok: usize) -> Place {
    let Some(kref) = cuda_kernel_for_op(&b.kernels, &node.op) else { return Place::Host };
    if !kref.quant.is_empty() && !op_weights(&node.op).iter().all(|w| b.kernels.has_quant(weights.ggml_type(*w))) {
        return Place::Host;
    }
    let (wpool, _, slots) = b.placement(node.layer);
    match &node.op {
        Op::Embed { .. } | Op::MatMul { .. } | Op::MoeFfn { .. } if wpool == Pool::Host => Place::Host,
        Op::MoeFfn { .. } if slots > 0 && n_tok > coproc::COPROC_ROWS => Place::Experts,
        Op::Attention { layer, .. } if b.placement(Some(*layer)).1 == Pool::Host => Place::Host,
        Op::Attention { layer, window: None, .. } if b.layers.get(layer).is_some_and(|l| l.kv_resident > 0) => Place::Stream,
        _ => Place::Device,
    }
}

/// Output shape of a device node, failing where the reference backend fails
#[allow(unreachable_patterns)]
fn infer(op: &Op, ins: &[(usize, usize)], n_tok: usize, n_pos: usize, hidden: (usize, usize), weights: &dyn WeightSource) -> Result<(usize, usize)> {
    let x = ins.first().copied().unwrap_or((0, 0));
    let numel = |w: Wid| weights.dims(w).iter().product::<usize>();
    let broadcast = |y: (usize, usize)| -> Result<(usize, usize)> {
        ensure!(y == x || (y.0 == x.0 && y.1 == 1) || (y.0 == 1 && y.1 == x.1) || y.0 * y.1 == 1 || y.0 * y.1 == x.0 * x.1, "binary op on {}x{} and {}x{}", x.0, x.1, y.0, y.1);
        Ok(x)
    };
    Ok(match op {
        Op::Input { name } => match *name {
            "tokens" => (n_tok, 1),
            "positions" => (n_pos, 1),
            "hidden" => {
                ensure!(hidden.1 > 0 && hidden.0 * hidden.1 > 0, "hidden input of {} values in rows of {}", hidden.0 * hidden.1, hidden.1);
                hidden
            }
            other => bail!("unknown graph input {other}"),
        },
        Op::Embed { weight } => (n_tok, weights.dims(*weight)[0]),
        Op::MatMul { weight, .. } => {
            let d = weights.dims(*weight);
            ensure!(x.1 == d[0], "{}: {} columns against {} inputs", weights.name(*weight), x.1, d[0]);
            (x.0, d.iter().skip(1).product::<usize>().max(1))
        }
        Op::RmsNorm { weight, bias, .. } | Op::LayerNorm { weight, bias, .. } => {
            for w in [weight, bias].into_iter().flatten() {
                ensure!(numel(*w) >= x.1, "norm weight of {} against {} columns", numel(*w), x.1);
            }
            x
        }
        Op::Rope { n_head, head_dim, .. } => {
            ensure!(x.1 == n_head * head_dim, "rope over {} columns, expected {} heads of {}", x.1, n_head, head_dim);
            ensure!(n_pos == x.0, "rope with {} positions for {} rows", n_pos, x.0);
            x
        }
        Op::Attention { n_head, n_head_kv, head_dim, head_dim_v, sinks, .. } => {
            ensure!(ins.len() >= 3, "attention with {} inputs", ins.len());
            let (q, k, v) = (ins[0], ins[1], ins[2]);
            ensure!(
                q.1 == n_head * head_dim && k.1 == n_head_kv * head_dim && v.1 == n_head_kv * head_dim_v,
                "attention shapes q {} k {} v {} for {} heads of {} and {} kv heads",
                q.1,
                k.1,
                v.1,
                n_head,
                head_dim,
                n_head_kv
            );
            ensure!(k.0 == q.0 && v.0 == q.0, "attention with {} query rows against {} key and {} value rows", q.0, k.0, v.0);
            ensure!(*n_head_kv > 0 && n_head.is_multiple_of(*n_head_kv), "attention with {} heads over {} kv heads", n_head, n_head_kv);
            if let Some(s) = sinks {
                ensure!(numel(*s) >= *n_head, "attention sinks of {} for {} heads", numel(*s), n_head);
            }
            (q.0, n_head * head_dim_v)
        }
        Op::AttentionFull { n_head, head_dim, .. } => {
            ensure!(ins.len() >= 3, "full attention with {} inputs", ins.len());
            ensure!(x.1 == n_head * head_dim && ins[1] == x && ins[2] == x, "full attention shapes q {}x{} k {}x{} v {}x{} for {n_head} heads of {head_dim}", x.0, x.1, ins[1].0, ins[1].1, ins[2].0, ins[2].1);
            x
        }
        Op::GeluQuick => x,
        Op::PoolGrid { side, kernel } => {
            ensure!(x.0 == side * side && *kernel > 0 && side.is_multiple_of(*kernel), "grid pool of {} rows as a {side} by {side} grid in blocks of {kernel}", x.0);
            ((side / kernel) * (side / kernel), x.1)
        }
        Op::MoeFfn { router, n_expert, .. } => {
            let d = weights.dims(*router);
            let n_out = d.iter().skip(1).product::<usize>().max(1);
            ensure!(d[0] == x.1, "{}: {} columns against {} inputs", weights.name(*router), x.1, d[0]);
            ensure!(n_out == *n_expert, "router gives {} experts, model says {}", n_out, n_expert);
            x
        }
        Op::GatedAct { .. } => {
            let up = ins[1];
            ensure!(x.0 * x.1 == up.0 * up.1, "gated activation on {}x{} and {}x{}", x.0, x.1, up.0, up.1);
            x
        }
        Op::GatedActFused { .. } => (x.0, x.1 / 2),
        Op::Add | Op::Sub | Op::Mul | Op::Div => broadcast(ins[1])?,
        Op::MulWeight { weight } | Op::AddWeight { weight } | Op::DivWeight { weight } => {
            let n = numel(*weight);
            ensure!(n == x.1 || n == 1, "row vector of {} against {} columns", n, x.1);
            x
        }
        Op::Scale(_)
        | Op::ScaleBias { .. }
        | Op::Clamp { .. }
        | Op::Silu
        | Op::Gelu
        | Op::GeluErf
        | Op::Relu
        | Op::ReluSqr
        | Op::Sigmoid
        | Op::Tanh
        | Op::Exp
        | Op::Sqrt
        | Op::Sqr
        | Op::Abs
        | Op::Sign
        | Op::Log
        | Op::Round
        | Op::Softplus { .. }
        | Op::Softmax
        | Op::Softcap(_)
        | Op::L2Norm { .. }
        | Op::MaskCols { .. } => x,
        Op::SumRows => (x.0, 1),
        Op::TopK { k } => {
            ensure!(*k >= 1 && *k <= x.1 && *k <= TOPK_MAX, "top {k} of {} columns, at most {TOPK_MAX}", x.1);
            (x.0, 2 * k + 1)
        }
        Op::Reshape { rows, cols } => {
            ensure!(x.0 * x.1 == rows * cols, "reshape {}x{} onto {} elements", rows, cols, x.0 * x.1);
            (*rows, *cols)
        }
        Op::Cols { start, len } => {
            ensure!(start + len <= x.1, "columns {}..{} of a {} column tensor", start, start + len, x.1);
            (x.0, *len)
        }
        Op::Rows { start, len } => {
            ensure!(start + len <= x.0, "rows {}..{} of a {} row tensor", start, start + len, x.0);
            (*len, x.1)
        }
        Op::LastRows { n } => (x.0.min(*n), x.1),
        Op::Concat => {
            let y = ins[1];
            ensure!(y.0 == x.0, "concat of {} rows with {} rows", x.0, y.0);
            (x.0, x.1 + y.1)
        }
        Op::Tile { n } => (x.0, x.1 * n),
        Op::PadCols { to } => {
            ensure!(*to >= x.1, "pad {} columns down to {to}", x.1);
            (x.0, *to)
        }
        Op::Fill { rows, cols, .. } => (*rows, *cols),
        other => bail!("{other:?} has no device shape rule"),
    })
}

/// Shape facts of a routed expert layer read from its tensors
pub(crate) struct MoeDims {
    pub rows: usize,
    pub n_embd: usize,
    pub n_ff: usize,
    pub n_expert: usize,
    pub n_used: usize,
    pub n_slots: usize,
}

pub(crate) fn moe_dims(op: &Op, x: (usize, usize), slots: usize, weights: &dyn WeightSource) -> Result<MoeDims> {
    let Op::MoeFfn { gate_exps, up_exps, down_exps: Some(down_exps), n_expert, n_used, .. } = op else { bail!("expert dims of {op:?}") };
    let up = weights.dims(*up_exps);
    let down = weights.dims(*down_exps);
    let n_ff = match gate_exps {
        Some(g) => {
            ensure!(weights.dims(*g) == up, "expert gate {:?} against up {:?}", weights.dims(*g), up);
            up[1]
        }
        None => up[1] / 2,
    };
    ensure!(up.len() >= 3 && down.len() >= 3, "expert tensors need three dims");
    ensure!(up[0] == x.1 && down[1] == x.1 && down[0] == n_ff, "expert down projection {}x{} against hidden {} and width {}", down[0], down[1], n_ff, x.1);
    ensure!(up[2] >= *n_expert && down[2] >= *n_expert, "expert tensors hold {} experts, router has {}", up[2].min(down[2]), n_expert);
    Ok(MoeDims { rows: x.0, n_embd: x.1, n_ff, n_expert: *n_expert, n_used: *n_used, n_slots: if slots > 0 { slots } else { *n_expert } })
}

/// Scratch lengths of an expert layer, in the order the expert launches index them
pub(crate) fn moe_scratch(d: &MoeDims) -> Vec<usize> {
    let rj = d.rows * d.n_used;
    let xe = d.rows * d.n_embd;
    vec![d.rows * d.n_expert, rj, rj, rj, rj, d.n_slots + 1, rj * 2 * d.n_ff, rj * d.n_ff, rj * d.n_ff, rj * d.n_embd, xe, xe, xe, xe, rj * d.n_ff, 1]
}

pub(crate) const S_LOGITS: usize = 0;
pub(crate) const S_SEL: usize = 1;
pub(crate) const S_WTS: usize = 2;
pub(crate) const S_SLOTS: usize = 3;
const S_ORDER: usize = 4;
const S_OFFSETS: usize = 5;
const S_GATE: usize = 6;
const S_UP: usize = 7;
const S_HIDDEN: usize = 8;
const S_PART: usize = 9;
pub(crate) const S_HOST: usize = 10;
const S_XR: usize = 11;
const S_XG: usize = 12;
const S_XU: usize = 13;
const S_HQ: usize = 14;
/// Misses the dispatch kernel counted for the await kernel
const S_NMISS: usize = 15;

/// Shape of node i from the shape rules and the scratch lengths it needs, expert layers theirs and products one activation copy
pub(crate) fn infer_node(b: &CudaBackend, p: &Prepared, i: usize, weights: &dyn WeightSource) -> Result<((usize, usize), Vec<usize>)> {
    let node = &p.nodes[i];
    let ins: Vec<(usize, usize)> = node.inputs.iter().map(|t| p.shape(*t)).collect::<Result<_>>()?;
    let hidden = if p.hidden_cols > 0 { (p.n_hid / p.hidden_cols, p.hidden_cols) } else { (0, 0) };
    let shape = infer(&node.op, &ins, p.n_tok, p.n_pos, hidden, weights)?;
    let scratch = match &node.op {
        Op::MoeFfn { n_expert, .. } => {
            let slots = experts::cache_slots(b, node.layer, *n_expert);
            moe_scratch(&moe_dims(&node.op, ins[0], slots, weights)?)
        }
        Op::MatMul { weight, .. } if round_code(weights.ggml_type(*weight))? != 0 => vec![ins[0].0 * ins[0].1],
        // The decode attention, which a tree pass runs at any row count, quantizes its output for the product after it
        // into scratch of its own, since rows finish at different times and the int8 rows of one would overwrite the
        // float rows of another in place
        Op::Attention { n_head, head_dim_v, .. } if ins[0].0 < FLASH_ROWS || p.tree => vec![ins[0].0 * n_head * head_dim_v],
        Op::TopK { k } => topk_scratch(ins[0].0, ins[0].1, *k),
        _ => Vec::new(),
    };
    Ok((shape, scratch))
}

/// Plans node i on the device side, its shape from the shape rules and scratch for expert layers
pub(crate) fn plan_device_node(b: &mut CudaBackend, p: &mut Prepared, i: usize, weights: &dyn WeightSource) -> Result<()> {
    let (shape, scratch) = infer_node(b, p, i, weights)?;
    p.plan(&mut b.pool, &b.stream, i, shape, &scratch)
}

/// The activation rounding a product against a weight type consumes as int8, none when it takes no int8 path
fn act_form(k: &KernelRegistry, weights: &dyn WeightSource, w: Wid, rows: usize) -> Option<i32> {
    let ty = weights.ggml_type(w);
    let n_in = weights.dims(w)[0];
    if !int8_path(k, ty, rows, n_in) {
        return None;
    }
    round_code(ty).ok()
}

/// Decides the fusions of the pending nodes of a device range from their ops, shapes, and weight types, before buffers are placed
fn plan_fusions(b: &CudaBackend, p: &mut Prepared, pending: &[usize], weights: &dyn WeightSource) -> Result<()> {
    if std::env::var_os("KF_NO_FUSION").is_some() {
        return Ok(());
    }
    // Bits of the fusions allowed: quantizing producers, ropes, residual adds, product groups, and the quantizing attention, every one by default
    let mask: u32 = std::env::var("KF_FUSION_MASK").ok().and_then(|v| v.parse().ok()).unwrap_or(127);
    let mut pend = vec![false; p.nodes.len()];
    for &i in pending {
        pend[i] = true;
    }
    let device = |p: &Prepared, j: usize| pend[j] && p.place[j] == Place::Device;
    // The graph output is read back as f32, so the node computing it keeps its own launch
    let output = p.output;
    for &i in pending {
        if p.place[i] != Place::Device || p.nodes[i].out == output {
            continue;
        }
        match p.nodes[i].op.clone() {
            // A norm or gated activation feeding only int8 products of one activation form quantizes for them
            Op::RmsNorm { .. } | Op::GatedAct { limit: None, .. } | Op::GatedActFused { limit: None, .. } if mask & 1 != 0 => {
                let out = p.nodes[i].out;
                let rows = p.shape(out)?.0;
                let consumers = p.uses[out.0 as usize].clone();
                let mut form = None;
                let all = !consumers.is_empty()
                    && consumers.iter().all(|&j| {
                        let Op::MatMul { weight, .. } = &p.nodes[j].op else { return false };
                        let Some(f) = act_form(&b.kernels, weights, *weight, rows) else { return false };
                        device(p, j) && form.is_none_or(|have| have == f) && {
                            form = Some(f);
                            true
                        }
                    });
                if let (true, Some(code)) = (all, form) {
                    p.fused[i] = Fusion::QuantOut { code };
                    for j in consumers {
                        p.fused[j] = Fusion::MatMul { act: Some(i), res: None };
                    }
                }
            }
            Op::Rope { params, n_head, head_dim, offset: 0, inverse: false } if mask & 2 != 0 => {
                let half = params.dims.min(head_dim) / 2;
                let neox = params.kind != RopeKind::Normal;
                if head_dim % 32 != 0 || head_dim > 256 || (neox && half % 32 != 0) {
                    continue;
                }
                let t = p.nodes[i].inputs[0];
                let rows = p.shape(t)?.0;
                // The shared head norm chain: reshape to rows of one head, rms norm, reshape back
                let mut src = t;
                let mut norm = None;
                let mut chain = Vec::new();
                if let Some(r2) = p.producer(t)
                    && let Op::Reshape { rows: rr2, cols: cc2 } = p.nodes[r2].op
                    && rr2 == rows
                    && cc2 == n_head * head_dim
                    && p.uses[t.0 as usize].len() == 1
                    && t != output
                    && let Some(n) = p.producer(p.nodes[r2].inputs[0])
                    && let Op::RmsNorm { weight, .. } = &p.nodes[n].op
                    && weight.is_none_or(|w| weights.dims(w).iter().product::<usize>() == head_dim)
                    && p.uses[p.nodes[r2].inputs[0].0 as usize].len() == 1
                    && p.nodes[r2].inputs[0] != output
                    && let Some(r1) = p.producer(p.nodes[n].inputs[0])
                    && let Op::Reshape { rows: rr1, cols: cc1 } = p.nodes[r1].op
                    && rr1 == rows * n_head
                    && cc1 == head_dim
                    && p.uses[p.nodes[n].inputs[0].0 as usize].len() == 1
                    && p.nodes[n].inputs[0] != output
                    && device(p, r2)
                    && device(p, n)
                    && device(p, r1)
                {
                    src = p.nodes[r1].inputs[0];
                    norm = Some(n);
                    chain = vec![r1, n, r2];
                }
                // Keys whose only reader is the attention after them go straight into its cache with the values
                let out = p.nodes[i].out;
                let mut append = None;
                if let [a] = p.uses[out.0 as usize].as_slice()
                    && let Op::Attention { kv_source: None, .. } = &p.nodes[*a].op
                    && device(p, *a)
                    && p.nodes[*a].inputs.len() >= 3
                    && p.nodes[*a].inputs[1] == out
                {
                    let v = p.nodes[*a].inputs[2];
                    if p.producer(v).is_none_or(|pv| pv < i) {
                        append = Some((*a, v));
                        p.fused[*a] = Fusion::Attention { appended: true, quant: None, rope: None };
                    }
                }
                for c in chain {
                    p.fused[c] = Fusion::Skip;
                }
                // The rope reads the projection past the chain it replaces, so that tensor lives until here
                p.last[src.0 as usize] = p.last[src.0 as usize].max(i);
                p.fused[i] = Fusion::Rope { src, norm, append };
            }
            _ => {}
        }
    }
    // A norm whose only reader is an add with a full shape residual writes the sum itself into the add's output
    for &i in pending {
        if mask & 32 == 0 {
            break;
        }
        let Op::RmsNorm { .. } = p.nodes[i].op else { continue };
        if p.place[i] != Place::Device || p.fused[i] != Fusion::None {
            continue;
        }
        let out = p.nodes[i].out;
        let [a] = p.uses[out.0 as usize][..] else { continue };
        if !device(p, a) || p.nodes[a].op != Op::Add || out == output || p.nodes[a].out == output {
            continue;
        }
        let ins = p.nodes[a].inputs.clone();
        let res = if ins[0] == out { ins[1] } else { ins[0] };
        if res == out || p.shape(res)? != p.shape(out)? || p.producer(res).is_some_and(|pr| pr >= i) {
            continue;
        }
        let aout = p.nodes[a].out;
        p.early[i].push(aout);
        p.last[res.0 as usize] = p.last[res.0 as usize].max(i);
        p.fused[a] = Fusion::Skip;
        p.fused[i] = Fusion::NormAdd { add: a, res };
    }
    // A product whose only reader is an add with a full shape residual writes the sum itself and the add aliases it
    for &i in pending {
        if mask & 4 == 0 {
            break;
        }
        let Op::MatMul { .. } = p.nodes[i].op else { continue };
        if p.place[i] != Place::Device {
            continue;
        }
        let out = p.nodes[i].out;
        let [a] = p.uses[out.0 as usize].as_slice() else { continue };
        let a = *a;
        if !device(p, a) || p.nodes[a].op != Op::Add || out == output {
            continue;
        }
        let ins = p.nodes[a].inputs.clone();
        let res = if ins[0] == out { ins[1] } else { ins[0] };
        if res == out || p.shape(res)? != p.shape(out)? || p.producer(res).is_some_and(|pr| pr >= i) {
            continue;
        }
        let aout = p.nodes[a].out.0 as usize;
        p.last[out.0 as usize] = p.last[out.0 as usize].max(p.last[aout]);
        p.alias[a] = Some(out);
        p.fused[a] = Fusion::Skip;
        p.fused[i] = match p.fused[i] {
            Fusion::MatMul { act, .. } => Fusion::MatMul { act, res: Some(res) },
            _ => Fusion::MatMul { act: None, res: Some(res) },
        };
    }
    // A decode attention feeding only int8 products of one form quantizes its joined rows for them
    for &i in pending {
        if mask & 16 == 0 {
            break;
        }
        let Op::Attention { n_head, n_head_kv, head_dim, head_dim_v, .. } = &p.nodes[i].op else { continue };
        let out = p.nodes[i].out;
        if p.place[i] != Place::Device || out == output || n_head * head_dim_v * 4 > 48 * 1024 {
            continue;
        }
        let rows = p.shape(out)?.0;
        if !matches!(attention_kind(&b.kernels, rows, *n_head, *n_head_kv, *head_dim, *head_dim_v, 0, p.tree), AttnKind::Decode { .. }) {
            continue;
        }
        let consumers = p.uses[out.0 as usize].clone();
        let mut form = None;
        let all = !consumers.is_empty()
            && consumers.iter().all(|&j| {
                let Op::MatMul { weight, .. } = &p.nodes[j].op else { return false };
                let Some(f) = act_form(&b.kernels, weights, *weight, rows) else { return false };
                device(p, j) && form.is_none_or(|have| have == f) && {
                    form = Some(f);
                    true
                }
            });
        if let (true, Some(form)) = (all, form) {
            let (appended, rope) = match p.fused[i] {
                Fusion::Attention { appended, rope, .. } => (appended, rope),
                _ => (false, None),
            };
            p.fused[i] = Fusion::Attention { appended, quant: Some(form), rope };
            for j in consumers {
                p.fused[j] = match p.fused[j] {
                    Fusion::MatMul { res, .. } => Fusion::MatMul { act: Some(i), res },
                    _ => Fusion::MatMul { act: Some(i), res: None },
                };
            }
        }
    }
    // The query rope and the appending key rope of one layer launch together at the later of the two
    let ropes: Vec<usize> = pending.iter().copied().filter(|&i| matches!(p.fused[i], Fusion::Rope { .. })).collect();
    for &k in &ropes {
        let Fusion::Rope { append: Some(_), .. } = p.fused[k] else { continue };
        let Op::Rope { head_dim: hk, .. } = &p.nodes[k].op else { continue };
        let partner = ropes.iter().copied().find(|&q| {
            let Fusion::Rope { append: None, .. } = p.fused[q] else { return false };
            let Op::Rope { head_dim: hq, .. } = &p.nodes[q].op else { return false };
            p.pair[q].is_none() && q != k && p.nodes[q].layer == p.nodes[k].layer && hq == hk && p.rope_keys[q] == p.rope_keys[k]
        });
        if let Some(q) = partner {
            p.pair[q] = Some(k);
            p.pair[k] = Some(q);
            let leader = q.max(k);
            for n in [q, k] {
                if let Fusion::Rope { src, .. } = p.fused[n] {
                    p.last[src.0 as usize] = p.last[src.0 as usize].max(leader);
                }
            }
        }
    }
    // A decode attention fed by the layer's two fused ropes norms, rotates, and appends the query and key itself
    if mask & 64 != 0 {
        for &i in pending {
            let Op::Attention { n_head, n_head_kv, head_dim, head_dim_v, .. } = &p.nodes[i].op else { continue };
            let Fusion::Attention { appended: true, quant, rope: None } = p.fused[i] else { continue };
            if p.place[i] != Place::Device {
                continue;
            }
            // One row only: the block of a row appends its key as other rows' blocks would be reading it
            let rows = p.shape(p.nodes[i].out)?.0;
            if rows != 1 || !matches!(attention_kind(&b.kernels, rows, *n_head, *n_head_kv, *head_dim, *head_dim_v, 0, p.tree), AttnKind::Decode { .. }) {
                continue;
            }
            let (qt, kt) = (p.nodes[i].inputs[0], p.nodes[i].inputs[1]);
            let (Some(qi), Some(ki)) = (p.producer(qt), p.producer(kt)) else { continue };
            let (Fusion::Rope { src: qsrc, norm: qnorm, .. }, Fusion::Rope { src: ksrc, norm: knorm, append: Some((a, v)) }) = (p.fused[qi], p.fused[ki]) else { continue };
            if a != i || p.pair[qi] != Some(ki) || p.rope_keys[qi] != p.rope_keys[ki] || p.uses[qt.0 as usize].len() != 1 {
                continue;
            }
            let Op::Rope { params, head_dim: hd, .. } = &p.nodes[qi].op else { continue };
            let half = params.dims.min(*hd) / 2;
            if params.kind != RopeKind::Normal && half % 8 != 0 {
                continue;
            }
            p.fused[qi] = Fusion::Skip;
            p.fused[ki] = Fusion::Skip;
            p.last[qsrc.0 as usize] = p.last[qsrc.0 as usize].max(i);
            p.last[ksrc.0 as usize] = p.last[ksrc.0 as usize].max(i);
            p.fused[i] = Fusion::Attention { appended: true, quant, rope: Some(RopeFold { qi, qsrc, qnorm, ksrc, knorm, v }) };
        }
    }
    // Int8 products of one input, type, and width launch together, the first of them for all, when their
    // row counts align with the tile the launch uses and every residual is ready at the first
    if mask & 8 != 0 {
        let mut by_input: Vec<((Tid, u32, usize), Vec<usize>)> = Vec::new();
        for &i in pending {
            let Op::MatMul { weight, scale: None, .. } = &p.nodes[i].op else { continue };
            if p.place[i] != Place::Device {
                continue;
            }
            let x = p.nodes[i].inputs[0];
            let rows = p.shape(x)?.0;
            if act_form(&b.kernels, weights, *weight, rows).is_none() {
                continue;
            }
            let key = (x, weights.ggml_type(*weight), weights.dims(*weight)[0]);
            match by_input.iter_mut().find(|(k, _)| *k == key) {
                Some((_, v)) => v.push(i),
                None => by_input.push((key, vec![i])),
            }
        }
        for ((x, ty, n_in), mut members) in by_input {
            members.sort_unstable();
            let leader = members[0];
            let rows = p.shape(x)?.0;
            let unit = if rows > matvec_rows(ty, n_in, &[]) { 128 } else if rows > 1 { 8 } else { 1 };
            let keep: Vec<usize> = members
                .into_iter()
                .filter(|&m| {
                    let Op::MatMul { weight, .. } = &p.nodes[m].op else { return false };
                    let n_out: usize = weights.dims(*weight).iter().skip(1).product::<usize>().max(1);
                    let res_ready = match p.fused[m] {
                        Fusion::MatMul { res: Some(r), .. } => p.producer(r).is_none_or(|pr| pr < leader),
                        _ => true,
                    };
                    n_out.is_multiple_of(unit) && res_ready
                })
                .take(4)
                .collect();
            if keep.len() >= 2 && keep[0] == leader {
                for &m in &keep {
                    p.grouped[m] = Some(leader);
                    if m != leader {
                        p.early[leader].push(p.nodes[m].out);
                    }
                }
            }
        }
    }
    Ok(())
}

/// Where a quantizing producer writes the int8 form of its output, an attention into its scratch and a norm or
/// activation into its own output buffer
fn quant_out_ptr(p: &Prepared, i: usize) -> Result<u64> {
    match p.nodes[i].op {
        Op::Attention { .. } => {
            ensure!(!p.scratch[i].is_empty(), "attention node {i} quantizes without scratch");
            Ok(p.scratch_ptr(i, 0))
        }
        _ => p.ptr(p.nodes[i].out),
    }
}

/// One product of a group: its weight, bias, residual, and output
struct GroupMember {
    w: Wid,
    w_ptr: u64,
    bias: u64,
    res: u64,
    y: u64,
    n_out: usize,
}

/// Launches of a group of int8 products sharing one input: the quantization once, then one product over the stacked
/// weights whose column tiles or warps write through a segment table, or each product alone when no stacked copy fits
#[allow(clippy::too_many_arguments)]
fn matmul_group_launch(b: &mut CudaBackend, p: &Prepared, weights: &dyn WeightSource, members: &[GroupMember], x: u64, xq: u64, pre: Option<u64>, rows: usize) -> Result<Vec<Launch>> {
    let ty = weights.ggml_type(members[0].w);
    let n_in = weights.dims(members[0].w)[0];
    let rb = row_bytes(ty, n_in)? as i64;
    let code = round_code(ty)?;
    let total: usize = members.iter().map(|m| m.n_out).sum();
    let f = crate::kf8::Form::of(ty, n_in, total).context("kf8 form")?;
    let mut v = Vec::new();
    let base = match pre {
        Some(base) => base,
        None => {
            v.push(quant_launch(&mut b.kernels, x, xq, rows, n_in, code)?);
            xq
        }
    };
    let bases: Vec<usize> = members.iter().scan(0usize, |start, m| { let at = *start; *start += m.n_out; Some(at) }).collect();
    if rows <= matvec_rows(ty, n_in, &bases) {
        // Every member's pointers as byte offsets from the leader's weights and output, so the kernel's loads stay global
        let lead = &members[0];
        let off = |ptr: u64, base: u64| if ptr == 0 { SEG_NONE } else { ptr.wrapping_sub(base) as i64 };
        let mut words = Vec::with_capacity(6 * members.len());
        for (m, &start) in members.iter().zip(&bases) {
            words.extend_from_slice(&[off(m.w_ptr, lead.w_ptr), off(m.y, lead.y), off(m.bias, lead.y), off(m.res, lead.y), start as i64, m.n_out as i64]);
        }
        let segs = p.segment_table(&b.stream, &words)?;
        v.push(matvec_i8_launch(&mut b.kernels, ty, lead.w_ptr, base, 0, 0, lead.y, rows, n_in, total, rb, segs, members.len(), mm_i8_fits(ty, n_in, &bases))?);
        return Ok(v);
    }
    let (q, scales, sums) = crate::kf8::act_scratch(base, rows, n_in, f.g);
    let pool = Pool::Device(b.ctx.ordinal() as u32);
    let stacked: Vec<(Wid, u64, i64, usize)> = members.iter().map(|m| (m.w, m.w_ptr, rb, m.n_out)).collect();
    match b.res.kf8.ensure_group(&mut b.kernels, &b.stream, pool, &stacked, ty, &f)? {
        Some(ptrs) => {
            let mut words = Vec::with_capacity(5 * members.len());
            let mut start = 0usize;
            for m in members {
                words.extend_from_slice(&[m.y as i64, m.bias as i64, m.res as i64, start as i64, m.n_out as i64]);
                start += m.n_out;
            }
            let segs = p.segment_table(&b.stream, &words)?;
            v.push(b.res.kf8.gemm(&mut b.kernels, &f, ptrs, q, scales, sums, 0, 0, 0, rows, segs, members.len())?);
        }
        None => {
            for m in members {
                v.extend(matmul_launch(b, weights, m.w, m.w_ptr, m.bias, x, xq, Some(base), m.res, m.y, rows)?);
            }
        }
    }
    Ok(v)
}

pub(crate) fn ensure_weight(b: &mut CudaBackend, w: Wid, weights: &dyn WeightSource) -> Result<u64> {
    if let Some(x) = b.res.weights.get(&w) {
        return Ok(x.1);
    }
    let bytes = weights.bytes(w)?;
    // Sixteen spare bytes, the aligned window the matvec stages a row's last block through
    let buf = b.alloc::<u8>(bytes.len() + 16, AllocKind::Weights, weights.name(w))?;
    let ptr = dptr(&buf, &b.stream);
    htod(&b.stream, ptr, bytes)?;
    b.res.weights.insert(w, (buf, ptr));
    b.res.epoch += 1;
    Ok(ptr)
}

/// Device f32 copy of a whole weight through the dequant kernel of its type, with its element count
fn expand(b: &mut CudaBackend, w: Wid, weights: &dyn WeightSource) -> Result<(CudaSlice<f32>, u64, usize)> {
    let src = ensure_weight(b, w, weights)?;
    let dims = weights.dims(w);
    let ty = weights.ggml_type(w);
    let inner = dims.first().copied().unwrap_or(1).max(1);
    let n: usize = dims.iter().product::<usize>().max(1);
    let rb = row_bytes(ty, inner)?;
    let buf = b.alloc::<f32>(n, AllocKind::Weights, weights.name(w))?;
    let ptr = dptr(&buf, &b.stream);
    let func = b.kernels.quant_function(ty, "dequant")?;
    flat(func, n.div_ceil(8), vec![Arg::P(src), Arg::P(ptr), Arg::I(inner as i32), Arg::L((n / inner) as i64), Arg::L(rb as i64)]).run(&b.stream)?;
    Ok((buf, ptr, n))
}

/// A small weight such as a norm or bias, uploaded raw and expanded to f32 on the device
pub(crate) fn ensure_vec(b: &mut CudaBackend, w: Wid, weights: &dyn WeightSource) -> Result<u64> {
    if let Some(x) = b.res.vecs.get(&w) {
        return Ok(x.1);
    }
    let (buf, ptr, _) = expand(b, w, weights)?;
    b.res.vecs.insert(w, (buf, ptr));
    b.res.epoch += 1;
    Ok(ptr)
}

/// Frequency of rotary pair i exactly as the reference computes it before multiplying by the position
fn rope_table(p: &RopeParams, head_dim: usize, factors: Option<&[f32]>) -> Result<(Vec<f32>, usize, f32)> {
    let dims = p.dims.min(head_dim);
    let half = dims / 2;
    if let Some(f) = factors {
        ensure!(f.len() >= half, "rope factors of {} for {} frequencies", f.len(), half);
    }
    let (low, high) = match p.yarn {
        Some(y) => {
            let corr = |rot: f32| (dims as f32 * (p.orig_ctx as f32 / (rot * 2.0 * PI)).ln()) / (2.0 * p.base.ln());
            (corr(y.beta_fast).floor().max(0.0), corr(y.beta_slow).ceil().min(dims as f32 - 1.0))
        }
        None => (0.0, 0.0),
    };
    let mscale = match p.yarn {
        Some(y) if y.ext_factor != 0.0 => p.attn_factor * (1.0 + 0.1 * (1.0 / p.scale).ln()),
        _ => p.attn_factor,
    };
    if p.kind == RopeKind::Vision {
        // Two streams over the halves of the pairs, each with the plain frequencies of its own half
        let s0 = p.sections[0].max(1);
        let table = (0..half).map(|i| p.base.powf(-2.0 * (if i < s0 { i } else { i - s0 }) as f32 / half as f32)).collect();
        return Ok((table, half, mscale));
    }
    let table = (0..half)
        .map(|i| {
            let freq_extra = p.base.powf(-(2.0 * i as f32) / dims as f32) / factors.map_or(1.0, |f| f[i]);
            let freq_inter = freq_extra * p.scale;
            match p.yarn {
                Some(y) if y.ext_factor != 0.0 => {
                    let span = if (high - low).abs() < 1e-3 { 1e-3 } else { high - low };
                    let ramp = ((i as f32 - low) / span).clamp(0.0, 1.0);
                    let mix = (1.0 - ramp) * y.ext_factor;
                    freq_inter * (1.0 - mix) + freq_extra * mix
                }
                _ => freq_inter,
            }
        })
        .collect();
    Ok((table, half, mscale))
}

/// The per head norm folded into a rope as kernel arguments: a flag, the weight, the bias, and the epsilon
fn norm_args(b: &CudaBackend, p: &Prepared, norm: Option<usize>) -> Result<Vec<Arg>> {
    Ok(match norm.map(|n| p.nodes[n].op.clone()) {
        Some(Op::RmsNorm { weight, bias, eps }) => vec![Arg::I(1), Arg::P(weight.map(|w| b.res.vec(w)).transpose()?.unwrap_or(0)), Arg::P(bias.map(|w| b.res.vec(w)).transpose()?.unwrap_or(0)), Arg::F(eps)],
        Some(other) => bail!("{other:?} folded into rope as a norm"),
        None => vec![Arg::I(0), Arg::P(0), Arg::P(0), Arg::F(0.0)],
    })
}

/// The stream selection of a rope kind as the kernels take it: the mode, three section widths, and the section period
fn rope_mode(p: &RopeParams) -> [i32; 5] {
    let s = p.sections;
    let (mode, s0) = match p.kind {
        RopeKind::Normal | RopeKind::Neox => (0, s[0]),
        RopeKind::MRope => (1, s[0]),
        RopeKind::IMRope => (2, s[0]),
        RopeKind::Vision => (3, s[0].max(1)),
    };
    [mode, s0 as i32, s[1] as i32, s[2] as i32, s.iter().sum::<usize>().max(1) as i32]
}

fn ensure_rope(b: &mut CudaBackend, key: &str, params: &RopeParams, head_dim: usize, weights: &dyn WeightSource) -> Result<()> {
    if b.res.rope.contains_key(key) {
        return Ok(());
    }
    let factors = params.factors.map(|f| kf_backend_cpu::dequant_all(f, weights)).transpose()?;
    let (table, half, mscale) = rope_table(params, head_dim, factors.as_deref())?;
    let buf = if table.is_empty() { b.alloc::<f32>(1, AllocKind::Weights, "rope table")? } else { b.upload(&table, AllocKind::Weights, "rope table")? };
    let ptr = dptr(&buf, &b.stream);
    let n_pos = b.n_ctx.max(b.rope_positions);
    let (cs, cs_ptr) = rope_angles(b, ptr, half, n_pos)?;
    b.res.rope.insert(key.to_string(), RopeTable { _buf: buf, ptr, _cs: cs, cs_ptr, n_pos, half, mscale });
    b.res.epoch += 1;
    Ok(())
}

/// Makes the caches of a layer, one ring per slot at the rows the plan gives its window or its resident rows, host rows pinned for what streams, and checks the rows each slot appends fit the context
fn ensure_kv(b: &mut CudaBackend, layer: usize, kw: usize, vw: usize, extra: &[(usize, usize)], window: Option<usize>) -> Result<()> {
    ensure!(b.n_ctx > 0 && b.n_batch > 0, "layer {layer} cache before a plan set the context and batch");
    let n_seq = b.host.n_seq();
    for (seq, rows) in extra {
        ensure!(*rows <= b.n_batch, "attention fed {rows} rows past the batch of {}", b.n_batch);
        ensure!(*seq < n_seq, "sequence slot {seq} of {n_seq}");
    }
    if let Some(kv) = b.res.kv.get(&layer) {
        ensure!(kv.kw == kw && kv.vw == vw, "layer {layer} cache widths {} and {} against {kw} and {vw}", kv.kw, kv.vw);
        for (seq, rows) in extra {
            ensure!(kv.n[*seq] + rows <= b.n_ctx, "{} positions past a context of {}", kv.n[*seq] + rows, b.n_ctx);
        }
        return Ok(());
    }
    let cap = kf_core::cache_rows(b.n_ctx, b.n_batch, window);
    let planned = b.layers.get(&layer).map_or(0, |l| l.kv_resident);
    let resident = if window.is_none() && planned > 0 && planned < cap { planned } else { cap };
    ensure!(resident >= 2 * b.n_batch, "layer {layer} keeps {resident} resident rows, at least two batches of {} needed", b.n_batch);
    let k = b.alloc::<u16>(n_seq * resident * kw, AllocKind::State, "key cache")?;
    let v = b.alloc::<u16>(n_seq * resident * vw, AllocKind::State, "value cache")?;
    let (host_k, host_v) = if resident < cap {
        let rows = n_seq * (cap - resident);
        let hk = unsafe { b.ctx.alloc_pinned::<u16>(rows * kw) }.context("pinned key rows")?;
        let hv = unsafe { b.ctx.alloc_pinned::<u16>(rows * vw) }.context("pinned value rows")?;
        kf_core::event::alloc(Pool::Host, AllocKind::State, (rows * (kw + vw) * 2) as u64);
        (Some(hk), Some(hv))
    } else {
        (None, None)
    };
    b.res.kv.insert(layer, DevKv { k, v, cap, resident, host_k, host_v, n: vec![0; n_seq], kw, vw });
    b.res.epoch += 1;
    Ok(())
}

/// The staging buffers for streamed rows of at least the widths given, made or regrown
fn ensure_stage(b: &mut CudaBackend, kw: usize, vw: usize) -> Result<()> {
    if b.res.stage.as_ref().is_some_and(|s| s.kw >= kw && s.vw >= vw) {
        return Ok(());
    }
    let (kw, vw) = match &b.res.stage {
        Some(s) => (s.kw.max(kw), s.vw.max(vw)),
        None => (kw, vw),
    };
    let mk = |b: &CudaBackend, w: usize| -> Result<[CudaSlice<u16>; 2]> { Ok([b.alloc::<u16>(STREAM_CHUNK * w, AllocKind::Compute, "stream staging")?, b.alloc::<u16>(STREAM_CHUNK * w, AllocKind::Compute, "stream staging")?]) };
    let k = mk(b, kw)?;
    let v = mk(b, vw)?;
    let ev = |b: &CudaBackend| -> Result<[CudaEvent; 2]> { Ok([b.stream.record_event(None).context("stage event")?, b.stream.record_event(None).context("stage event")?]) };
    let ready = ev(b)?;
    let free = ev(b)?;
    b.res.stage = Some(Stage { k, v, kw, vw, ready, free });
    b.res.epoch += 1;
    Ok(())
}

/// Keys one block of resident attention covers when a pass has too few rows to fill the device
const AT_PART: usize = 128;
/// Rows times heads from which resident attention runs undivided, one block per row and head
const AT_BLOCKS: usize = 256;
/// Widest value row the attention kernels accumulate in registers
const AT_V_MAX: usize = 512;

/// Parts the key range of resident attention splits into, one for a pass with enough blocks and the ring in parts otherwise
fn attention_parts(rows: usize, n_head: usize, cap: usize) -> usize {
    if rows * n_head >= AT_BLOCKS { 1 } else { cap.div_ceil(AT_PART).max(1) }
}

/// A partials buffer of at least n floats
/// The arrival counters of the decode attention, zero, at least n of them
fn ensure_counters(b: &mut CudaBackend, n: usize) -> Result<u64> {
    if b.res.counters.as_ref().is_none_or(|c| c.len() < n) {
        b.res.counters = Some(b.alloc::<i32>(n.max(1024).next_power_of_two(), AllocKind::Compute, "attention counters")?);
        b.res.epoch += 1;
    }
    Ok(dptr(b.res.counters.as_ref().context("counters")?, &b.stream))
}

fn ensure_parts(b: &mut CudaBackend, n: usize) -> Result<u64> {
    if b.res.parts.as_ref().is_none_or(|p| p.len() < n) {
        b.res.parts = Some(b.alloc::<f32>(n.max(1024).next_power_of_two(), AllocKind::Compute, "attention partials")?);
        b.res.epoch += 1;
    }
    Ok(dptr(b.res.parts.as_ref().context("partials")?, &b.stream))
}

/// Host row region of one slot of a streamed cache as a mutable slice, a DMA target the stream orders
unsafe fn host_rows_mut(h: &mut PinnedHostSlice<u16>, start: usize, len: usize) -> Result<&mut [u16]> {
    let ptr = h.as_mut_ptr().context("pinned rows")?;
    ensure!(start + len <= h.len(), "host rows {}..{} of {}", start, start + len, h.len());
    Ok(unsafe { std::slice::from_raw_parts_mut(ptr.add(start), len) })
}

unsafe fn host_rows(h: &PinnedHostSlice<u16>, start: usize, len: usize) -> Result<&[u16]> {
    let ptr = h.as_ptr().context("pinned rows")?;
    ensure!(start + len <= h.len(), "host rows {}..{} of {}", start, start + len, h.len());
    Ok(unsafe { std::slice::from_raw_parts(ptr.add(start), len) })
}

/// Copies positions p0 through p1 of a slot between the device ring and the host rows, towards the host or back, each wrap of the ring its own copy
fn ring_host_copy(stream: &CudaStream, kv: &mut DevKv, seq: usize, p0: usize, p1: usize, to_host: bool) -> Result<()> {
    let host_rows_per = kv.host_rows();
    let p1 = p1.min(host_rows_per.saturating_sub(1));
    if p0 > p1 {
        return Ok(());
    }
    let res = kv.resident;
    let mut p = p0;
    while p <= p1 {
        let slot = p % res;
        let len = (res - slot).min(p1 - p + 1);
        for (dev, host, w) in [(&kv.k, &mut kv.host_k, kv.kw), (&kv.v, &mut kv.host_v, kv.vw)] {
            let host = host.as_mut().context("host rows of a streamed cache")?;
            let dptr = dptr(dev, stream) + (((seq * res + slot) * w) * 2) as u64;
            let hstart = (seq * host_rows_per + p) * w;
            if to_host {
                let dst = unsafe { host_rows_mut(host, hstart, len * w)? };
                unsafe { result::memcpy_dtoh_async(dst, dptr, stream.cu_stream()) }.context("copy rows to host")?;
            } else {
                let src = unsafe { host_rows(host, hstart, len * w)? };
                unsafe { result::memcpy_htod_async(dptr, src, stream.cu_stream()) }.context("copy rows to device")?;
            }
        }
        p += len;
    }
    Ok(())
}

/// Restores the ring rows of a slot that later positions may have overwritten, after a truncation to n
fn refill_ring(stream: &CudaStream, kv: &mut DevKv, seq: usize, n: usize) -> Result<()> {
    if !kv.streams() || n == 0 {
        return Ok(());
    }
    let lo = n.saturating_sub(kv.resident);
    let hi = n.min(kv.host_rows());
    if lo < hi {
        ring_host_copy(stream, kv, seq, lo, hi - 1, false)?;
    }
    Ok(())
}

/// Attention over a streamed cache, the ring appended and mirrored to the host, older rows attended chunk by chunk as they arrive, every part merged
fn run_stream_attention(b: &mut CudaBackend, p: &mut Prepared, i: usize, inputs: &Inputs, weights: &dyn WeightSource) -> Result<()> {
    use Arg::{F, I, P};
    let node = p.nodes[i].clone();
    let Op::Attention { layer, n_head, n_head_kv, head_dim, head_dim_v, scale, window, softcap, sinks, .. } = &node.op else { bail!("stream segment on {:?}", node.op) };
    if p.shapes[node.out.0 as usize].is_none() {
        plan_device_node(b, p, i, weights)?;
    }
    let mut appended = HashMap::new();
    ensure_node(b, p, i, inputs, weights, &mut appended)?;
    let (kw, vw) = (n_head_kv * head_dim, n_head_kv * head_dim_v);
    ensure_stage(b, kw, vw)?;
    let rows = p.shape(node.out)?.0;
    let runs = inputs.runs();
    let (q, k, v, out) = (p.ptr(node.inputs[0])?, p.ptr(node.inputs[1])?, p.ptr(node.inputs[2])?, p.ptr(node.out)?);
    let sk = sinks.map(|w| b.res.vec(w)).transpose()?.unwrap_or(0);
    let stream = b.stream.clone();
    let copy = b.copy_stream.clone();
    let (kc, vc, resident) = {
        let kv = b.res.kv.get(layer).context("streamed cache")?;
        (dptr(&kv.k, &stream), dptr(&kv.v, &stream), kv.resident)
    };
    let append = flat(b.kernels.function("attention", "kv_append")?, rows * (kw + vw), vec![P(k), P(v), P(kc), P(vc), P(p.pos_ptr), P(p.seq_ptr), I(rows as i32), I(kw as i32), I(vw as i32), I(resident as i32)]);
    append.run(&stream)?;
    let mut after: Vec<(usize, usize)> = Vec::new();
    {
        let kv = b.res.kv.get_mut(layer).context("streamed cache")?;
        for (seq, range) in &runs {
            let (p0, p1) = (inputs.positions[range.start] as usize, inputs.positions[range.end - 1] as usize);
            ring_host_copy(&stream, kv, *seq, p0, p1, true)?;
            after.push((*seq, p1 + 1));
        }
    }
    let width = 2 + head_dim_v;
    let n_parts = after.iter().map(|(_, n)| n.saturating_sub(resident).div_ceil(STREAM_CHUNK) + 1).max().unwrap_or(1);
    let parts = ensure_parts(b, n_parts * rows * n_head * width)?;
    let clear = flat(b.kernels.function("attention", "attention_clear")?, n_parts * rows * n_head * width, vec![P(parts), Arg::L((n_parts * rows * n_head * width) as i64), I(width as i32)]);
    clear.run(&stream)?;
    let part_fn = b.kernels.function("attention", "attention_part")?;
    ensure!(*head_dim_v <= AT_V_MAX, "attention value width {head_dim_v} past the {AT_V_MAX} the device kernel holds");
    let smem = ((head_dim + 8 + 4 * AT_V_MAX) * 4) as u32;
    let launch_part = |kb: u64, vb: u64, row0: usize, n_rows: usize, base: usize, ring: usize, jlo: usize, jhi: usize, part: usize| -> Result<()> {
        let l = Launch {
            func: part_fn.clone(),
            grid: (n_rows as u32, *n_head as u32, 1),
            block: 128,
            smem,
            args: vec![P(q), P(kb), P(vb), P(p.pos_ptr), P(p.seq_ptr), P(p.att_ptr), I(row0 as i32), I(base as i32), I(ring as i32), I(jlo as i32), I(jhi as i32), P(parts), I(part as i32), I(rows as i32), I(*n_head as i32), I(*n_head_kv as i32), I(*head_dim as i32), I(*head_dim_v as i32), F(*scale), I(window.map_or(0, |w| w.size) as i32), F(*softcap), P(if p.tree { p.tree_ptr } else { 0 })],
        };
        l.run(&stream)
    };
    let mut chunk_no = 0usize;
    for ((seq, range), (_, n_after)) in runs.iter().zip(&after) {
        let old_end = n_after.saturating_sub(resident);
        let mut part = 0;
        let mut j0 = 0;
        while j0 < old_end {
            let j1 = (j0 + STREAM_CHUNK).min(old_end) - 1;
            let si = chunk_no % 2;
            chunk_no += 1;
            let (sk_ptr, sv_ptr) = {
                let stage = b.res.stage.as_ref().context("staging")?;
                copy.wait(&stage.free[si]).context("wait staging free")?;
                let kv = b.res.kv.get(layer).context("streamed cache")?;
                let host_rows_per = kv.host_rows();
                let len = j1 - j0 + 1;
                for (host, dst, w) in [(kv.host_k.as_ref(), &stage.k[si], kw), (kv.host_v.as_ref(), &stage.v[si], vw)] {
                    let host = host.context("host rows of a streamed cache")?;
                    let src = unsafe { host_rows(host, (seq * host_rows_per + j0) * w, len * w)? };
                    unsafe { result::memcpy_htod_async(dptr(dst, &stream), src, copy.cu_stream()) }.context("stream rows to device")?;
                }
                stage.ready[si].record(&copy).context("record staging ready")?;
                stream.wait(&stage.ready[si]).context("wait staging ready")?;
                (dptr(&stage.k[si], &stream), dptr(&stage.v[si], &stream))
            };
            launch_part(sk_ptr, sv_ptr, range.start, range.len(), j0, 0, j0, j1, part)?;
            let stage = b.res.stage.as_ref().context("staging")?;
            stage.free[si].record(&stream).context("record staging free")?;
            part += 1;
            j0 = j1 + 1;
        }
        launch_part(kc, vc, range.start, range.len(), 0, resident, old_end, n_after - 1, part)?;
    }
    let merge = Launch { func: b.kernels.function("attention", "attention_merge")?, grid: (rows as u32, *n_head as u32, 1), block: 256, smem: 0, args: vec![P(parts), I(n_parts as i32), I(rows as i32), P(sk), P(out), I(*n_head as i32), I(*head_dim_v as i32)] };
    merge.run(&stream)?;
    let kv = b.res.kv.get_mut(layer).context("streamed cache")?;
    for (seq, n_after) in after {
        kv.n[seq] = n_after;
    }
    Ok(())
}

/// Uploads and allocates what a device node reads, checking tokens and cache room for this pass
fn ensure_node(b: &mut CudaBackend, p: &Prepared, i: usize, inputs: &Inputs, weights: &dyn WeightSource, appended: &mut HashMap<usize, Vec<(usize, usize)>>) -> Result<()> {
    let node = &p.nodes[i];
    match &node.op {
        Op::Embed { weight } => {
            ensure_weight(b, *weight, weights)?;
            let n_vocab = weights.dims(*weight).get(1).copied().unwrap_or(1);
            if let Some(t) = inputs.tokens.iter().find(|t| **t as usize >= n_vocab) {
                bail!("token {t} past vocabulary of {n_vocab}");
            }
        }
        Op::MatMul { weight, bias, scale } => {
            ensure_weight(b, *weight, weights)?;
            for v in [bias, scale].into_iter().flatten() {
                ensure_vec(b, *v, weights)?;
            }
        }
        Op::RmsNorm { weight, bias, .. } | Op::LayerNorm { weight, bias, .. } => {
            for v in [weight, bias].into_iter().flatten() {
                ensure_vec(b, *v, weights)?;
            }
        }
        Op::MulWeight { weight } | Op::AddWeight { weight } | Op::DivWeight { weight } => {
            ensure_vec(b, *weight, weights)?;
        }
        Op::Rope { params, head_dim, .. } => {
            let key = p.rope_keys[i].as_deref().context("rope key")?;
            ensure_rope(b, key, params, *head_dim, weights)?;
        }
        Op::Attention { layer, n_head, n_head_kv, head_dim, head_dim_v, sinks, window, .. } => {
            let rows = p.shape(node.inputs[0])?.0;
            ensure!(rows == inputs.tokens.len(), "attention over {rows} rows for a pass of {} tokens", inputs.tokens.len());
            let add = appended.entry(*layer).or_default();
            for (seq, range) in inputs.runs() {
                match add.iter_mut().find(|(s, _)| *s == seq) {
                    Some(e) => e.1 += range.len(),
                    None => add.push((seq, range.len())),
                }
            }
            let extra = add.clone();
            ensure_kv(b, *layer, n_head_kv * head_dim, n_head_kv * head_dim_v, &extra, window.map(|w| w.size))?;
            let cap = b.res.kv.get(layer).map_or(0, |kv| kv.cap);
            let kind = attention_kind(&b.kernels, rows, *n_head, *n_head_kv, *head_dim, *head_dim_v, cap, p.tree);
            let n_parts = match kind {
                AttnKind::Flash => 1,
                AttnKind::Decode { parts } | AttnKind::Classic { parts } => parts,
            };
            if n_parts > 1 {
                ensure_parts(b, n_parts * rows * n_head * (2 + head_dim_v))?;
            }
            if matches!(kind, AttnKind::Decode { .. }) {
                ensure_counters(b, rows * (n_head_kv * head_splits(*n_head, *n_head_kv) + 1))?;
            }
            if let Some(s) = sinks {
                ensure_vec(b, *s, weights)?;
            }
        }
        Op::MoeFfn { router, router_bias, probs_bias, gate_exps, up_exps, down_exps, n_expert, .. } => {
            ensure_weight(b, *router, weights)?;
            for v in [router_bias, probs_bias].into_iter().flatten() {
                ensure_vec(b, *v, weights)?;
            }
            if experts::cache_slots(b, node.layer, *n_expert) > 0 {
                experts::ensure_cache(b, node.layer, &node.op, weights)?;
            } else {
                for w in [*gate_exps, Some(*up_exps), *down_exps].into_iter().flatten() {
                    ensure_weight(b, w, weights)?;
                }
            }
        }
        _ => {}
    }
    Ok(())
}

/// Query rows per block of the tensor core attention
pub(crate) const FLASH_BQ: usize = 64;

/// Passes of this many rows and more run attention on the tensor cores
const FLASH_ROWS: usize = 16;

/// Head sizes the tensor core attention is compiled for
const FLASH_HEAD_DIMS: &[usize] = &[64, 80, 96, 112, 128, 256];

/// Head sizes the decode attention is compiled for, those whose eighths divide a warp
const DECODE_HEAD_DIMS: &[usize] = &[64, 128, 256];

/// Blocks the query heads of one kv head split over in the decode attention, at most four heads each
fn head_splits(n_head: usize, n_head_kv: usize) -> usize {
    (n_head / n_head_kv.max(1)).div_ceil(4).max(1)
}

/// Whether an attention pass runs on the tensor cores: enough rows, an Ampere device, and a compiled head size shared by keys and values
fn flash_path(k: &KernelRegistry, rows: usize, hd: usize, hdv: usize) -> bool {
    rows >= FLASH_ROWS && k.arch() >= INT8_ARCH && hd == hdv && FLASH_HEAD_DIMS.contains(&hd)
}

/// How an attention pass runs and how many key parts it splits into
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AttnKind {
    Flash,
    Decode { parts: usize },
    Classic { parts: usize },
}

/// The kernel of an attention pass: the tensor core tile for many rows, the per key decode kernel split so the
/// grid fills every SM twice, else the per row kernel split over the cache. A tree pass runs the decode kernel, the
/// one that reads the ancestor masks.
fn attention_kind(k: &KernelRegistry, rows: usize, n_head: usize, n_head_kv: usize, hd: usize, hdv: usize, cap: usize, tree: bool) -> AttnKind {
    if flash_path(k, rows, hd, hdv) && !tree {
        AttnKind::Flash
    } else if hd == hdv && DECODE_HEAD_DIMS.contains(&hd) && n_head.is_multiple_of(n_head_kv) && n_head / n_head_kv <= 8 {
        let parts = std::env::var("KF_DECODE_PARTS").ok().and_then(|v| v.parse().ok()).unwrap_or((2 * k.sm_count()).div_ceil(rows * n_head_kv));
        AttnKind::Decode { parts: parts.clamp(2, 64) }
    } else {
        AttnKind::Classic { parts: attention_parts(rows, n_head, cap) }
    }
}

/// Dynamic shared memory of the tensor core attention for a head size, the query tile and two stages of key and value tiles
fn flash_smem(hd: usize) -> u32 {
    let nch = (hd / 8).div_ceil(8) * 8;
    let rowb = nch * 16;
    let bk = if hd <= 128 { 64 } else { 32 };
    ((FLASH_BQ + 4 * bk) * rowb) as u32
}

/// The query tiles of a pass, 64 rows of one slot each, heaviest keys first so the long tiles do not trail the grid
fn flash_tiles(inputs: &Inputs, n_max: usize) -> Vec<i32> {
    let mut tiles: Vec<(i32, i32, i32, i32)> = Vec::new();
    for (seq, range) in inputs.runs() {
        let mut r = range.start;
        while r < range.end {
            let n = (range.end - r).min(FLASH_BQ);
            let last = r + n - 1;
            let hi = inputs.positions.get(last).copied().unwrap_or(0).max(inputs.attend_of(last));
            tiles.push((hi as i32, r as i32, n as i32, seq as i32));
            r += n;
        }
    }
    tiles.sort_by(|a, b| b.0.cmp(&a.0));
    let n = tiles.len().min(n_max);
    let mut out = vec![0i32; 4 + 4 * n_max];
    out[0] = n as i32;
    for (i, t) in tiles.iter().take(n).enumerate() {
        out[4 + 4 * i] = t.1;
        out[5 + 4 * i] = t.2;
        out[6 + 4 * i] = t.3;
    }
    out
}

/// Rows a warp level product handles before the tiled product takes over
pub(crate) const MATVEC_ROWS: usize = 8;

/// Compute capability from which the int8 tensor core product runs, the mma shape it uses is Ampere's
const INT8_ARCH: u32 = 80;

/// Whether a product against a weight type takes the int8 path on this device, the tensor core tile past
/// MATVEC_ROWS and the dp4a matvec up to it, both reading the same quantized activations
pub(crate) fn int8_path(k: &KernelRegistry, ty: u32, _rows: usize, n_in: usize) -> bool {
    k.arch() >= INT8_ARCH && crate::kf8::Form::of(ty, n_in, 1).is_some_and(|f| f.fits())
}

/// Threads of an int8 matvec block, eight warps split across its output rows
pub(crate) const MATVEC_THREADS: u32 = 256;

/// Activation rows up to which the tensor core product on native bytes beats the KF8 tile, measured on Llama-3.2-1B
pub(crate) const MM_ROWS: usize = 32;

/// Bytes of activation rows one window of the tensor core product holds in shared memory
const MM_WINDOW: usize = 16384;

/// Most dynamic shared memory a block of the tensor core product may take on the device
const MM_SMEM_MAX: usize = 99 * 1024;

/// Whether the tensor core product runs a shape: a step of four lane widths divides the columns and every stacked
/// member starts on an eight row tile
pub(crate) fn mm_i8_fits(ty: u32, n_in: usize, bases: &[usize]) -> bool {
    n_in.is_multiple_of(4 * sources::matvec_cols(ty)) && bases.iter().all(|b| b.is_multiple_of(8))
}

/// Rows up to which a product of a type and width runs as a matvec family kernel rather than the KF8 tile
pub(crate) fn matvec_rows(ty: u32, n_in: usize, bases: &[usize]) -> usize {
    if mm_i8_fits(ty, n_in, bases) { MM_ROWS } else { MATVEC_ROWS }
}

/// Warps per eight row group of the tensor core product, doubled while the grid is under two blocks per SM and
/// every warp keeps a step of its own
fn mm_wpr(k: &KernelRegistry, n_in: usize, n_out: usize, cols: usize) -> usize {
    let steps = n_in / (4 * cols);
    let mut wpr = 1;
    while wpr < 8 && wpr * 2 <= steps && n_out.div_ceil(64 / wpr) < 2 * k.sm_count() {
        wpr *= 2;
    }
    wpr
}

/// Steps of the activation window of the tensor core product, every warp of a row group taking at least two from it
/// so the block refills it less often than a warp steps, zero when no such window fits beside the tiles
fn mm_wsteps(ty: u32, cols: usize, r: usize, wpr: usize) -> usize {
    let nwin = (MM_WINDOW / (r * 4 * cols * wpr)).min(8);
    if nwin < 2 || mm_smem(ty, cols, r, wpr * nwin) > MM_SMEM_MAX / 2 { 0 } else { wpr * nwin }
}

/// Shared memory of the tensor core product: eight tiles of KfMmShape or the join scratch of eight warps of r rows,
/// then the window of r activation rows over wsteps steps with their sums and scales per run
fn mm_smem(ty: u32, cols: usize, r: usize, wsteps: usize) -> usize {
    let tiles = (8 * 8 * (4 * cols + 16)).max(8 * 32 * (r / 8) * 2 * 4);
    let runs = wsteps * 4 * cols / sources::matvec_run(ty);
    tiles + if wsteps > 0 { r * (wsteps * 4 * cols + 16) + 2 * r * runs * 4 } else { 0 }
}

/// Warp steps over a row of a weight type, a step being 32 lanes of the type's lane width
fn matvec_steps(ty: u32, n_in: usize) -> usize {
    n_in.div_ceil(32 * sources::matvec_cols(ty)).max(1)
}

/// Warps of an int8 matvec block per output row for a weight type and width: one for rows of under four warp steps,
/// two up to eight, four beyond
pub(crate) fn matvec_wpr(ty: u32, n_in: usize) -> usize {
    let steps = matvec_steps(ty, n_in);
    if steps >= 8 { 4 } else if steps >= 4 { 2 } else { 1 }
}

/// Shared memory of the one row int8 matvec, the staged activation row with its scales and sums per sixteen
pub(crate) fn matvec_smem(rows: usize, n_in: usize) -> u32 {
    if rows == 1 { (n_in + 2 * (n_in / 16) * 4) as u32 } else { 0 }
}

/// The segment table word marking a member without a bias or residual
pub(crate) const SEG_NONE: i64 = i64::MIN;

/// The int8 matvec launch of rows of quantized activations in the scratch at act against native weight rows, or against
/// the nseg stacked products of the segment table at segs when it is nonzero, whose pointers are offsets from w_ptr and y.
/// Several rows run the tensor core product when mm says the shape fits it, see mm_i8_fits.
#[allow(clippy::too_many_arguments)]
pub(crate) fn matvec_i8_launch(k: &mut KernelRegistry, ty: u32, w_ptr: u64, act: u64, bias: u64, res: u64, y: u64, rows: usize, n_in: usize, n_out: usize, rb: i64, segs: u64, nseg: usize, mm: bool) -> Result<Launch> {
    let module = sources::quant_module(ty).ok_or_else(|| anyhow!("no quant kernels for ggml type {ty}"))?;
    let [q, s16, m16, s32, m32] = crate::kf8::act_scratch_all(act, rows, n_in);
    if rows > 1 && mm {
        let cols = sources::matvec_cols(ty);
        let wpr = mm_wpr(k, n_in, n_out, cols);
        ensure!(rows <= MM_ROWS, "{rows} rows for the tensor core product of at most {MM_ROWS}");
        let (entry, r) = if rows <= 8 { ("mm_i8_8", 8) } else if rows <= 16 { ("mm_i8_16", 16) } else { ("mm_i8_32", 32) };
        let wsteps = mm_wsteps(ty, cols, r, wpr);
        let smem = mm_smem(ty, cols, r, wsteps);
        return Ok(Launch {
            func: k.function_smem(&module, entry, smem as u32)?,
            grid: (n_out.div_ceil(64 / wpr) as u32, rows.div_ceil(r) as u32, 1),
            block: MATVEC_THREADS,
            smem: smem as u32,
            args: vec![Arg::P(w_ptr), Arg::P(q), Arg::P(s16), Arg::P(m16), Arg::P(s32), Arg::P(m32), Arg::P(bias), Arg::P(res), Arg::P(y), Arg::I(rows as i32), Arg::I(n_in as i32), Arg::I(n_out as i32), Arg::L(rb), Arg::P(segs), Arg::I(nseg as i32), Arg::I(wpr as i32), Arg::I(wsteps as i32)],
        });
    }
    ensure!(rows <= MATVEC_ROWS, "{rows} rows for the dp4a matvec of at most {MATVEC_ROWS}");
    let wpr = matvec_wpr(ty, n_in);
    let rpb = 8 / wpr;
    // One row's kernel, the one capped at four blocks per SM when a warp has a single step and the grid fits the
    // device in one wave at four blocks but not at the three the uncapped one allows
    let blocks = n_out.div_ceil(rpb);
    let one_wave = blocks > 3 * k.sm_count() && blocks <= 4 * k.sm_count();
    let entry = match (rows, matvec_steps(ty, n_in), one_wave) {
        (1, 1, true) => "matvec_i8_1s",
        (1, _, _) => "matvec_i8_1",
        _ => "matvec_i8",
    };
    let smem = matvec_smem(rows, n_in);
    Ok(Launch {
        func: k.function_smem(&module, entry, smem)?,
        grid: (n_out.div_ceil(rpb) as u32, 1, 1),
        block: MATVEC_THREADS,
        smem,
        args: vec![Arg::P(w_ptr), Arg::P(q), Arg::P(s16), Arg::P(m16), Arg::P(s32), Arg::P(m32), Arg::P(bias), Arg::P(res), Arg::P(y), Arg::I(rows as i32), Arg::I(n_in as i32), Arg::I(n_out as i32), Arg::L(rb), Arg::P(segs), Arg::I(nseg as i32), Arg::I(wpr as i32)],
    })
}

/// Columns one block of the quantizing kernels covers, the KF_QUANT_SPAN of the fused module
pub(crate) const QUANT_SPAN: usize = 2048;

/// The grid of a quantizing kernel over rows of n columns, a block per row and span
pub(crate) fn quant_grid(rows: usize, n: usize) -> (u32, u32, u32) {
    (rows as u32, n.div_ceil(QUANT_SPAN).max(1) as u32, 1)
}

/// The five scratch pointers as kernel arguments in the order the quantizers take them
pub(crate) fn act_args(base: u64, rows: usize, n: usize) -> Vec<Arg> {
    crate::kf8::act_scratch_all(base, rows, n).into_iter().map(Arg::P).collect()
}

/// The quantization launch of rows of n activations into the int8 form at base
pub(crate) fn quant_launch(k: &mut KernelRegistry, x: u64, base: u64, rows: usize, n: usize, code: i32) -> Result<Launch> {
    Ok(Launch { func: k.function("fused", "quant_act")?, grid: quant_grid(rows, n), block: 256, smem: 0, args: [vec![Arg::P(x)], act_args(base, rows, n), vec![Arg::I(n as i32), Arg::I(code)]].concat() })
}

/// Activation quantization then the KF8 tensor core product, or activation rounding then a matrix vector product
/// for a few rows or a shared memory tiled product. pre is the int8 activations a fused producer already wrote, res
/// the residual rows added into the output.
#[allow(clippy::too_many_arguments)]
pub(crate) fn matmul_launch(b: &mut CudaBackend, weights: &dyn WeightSource, w: Wid, w_ptr: u64, bias: u64, x: u64, xq: u64, pre: Option<u64>, res: u64, y: u64, rows: usize) -> Result<Vec<Launch>> {
    let d = weights.dims(w);
    let ty = weights.ggml_type(w);
    let n_in = d[0];
    let n_out: usize = d.iter().skip(1).product::<usize>().max(1);
    let rb = row_bytes(ty, n_in)? as i64;
    let mut v = Vec::with_capacity(3);
    let k = &mut b.kernels;
    if int8_path(k, ty, rows, n_in) {
        let code = round_code(ty)?;
        ensure!(matches!(code, 3 | 4), "int8 product against a type with activation rounding {code}");
        let f = crate::kf8::Form::of(ty, n_in, n_out).context("kf8 form")?;
        let base = match pre {
            Some(base) => base,
            None => {
                v.push(quant_launch(k, x, xq, rows, n_in, code)?);
                xq
            }
        };
        if rows <= matvec_rows(ty, n_in, &[]) {
            v.push(matvec_i8_launch(k, ty, w_ptr, base, bias, res, y, rows, n_in, n_out, rb, 0, 0, mm_i8_fits(ty, n_in, &[]))?);
            return Ok(v);
        }
        let (q, scales, sums) = crate::kf8::act_scratch(base, rows, n_in, f.g);
        let pool = Pool::Device(b.ctx.ordinal() as u32);
        let was = b.res.kf8.is_resident(w);
        let p = b.res.kf8.ensure(k, &b.stream, pool, w, ty, w_ptr, rb, f, &mut v)?;
        if !was && b.res.kf8.is_resident(w) {
            b.res.epoch += 1;
        }
        v.push(b.res.kf8.gemm(k, &f, p, q, scales, sums, bias, res, y, rows, 0, 0)?);
        return Ok(v);
    }
    let x = rounded(k, ty, x, xq, rows, n_in, &mut v)?;
    v.push(if rows <= MATVEC_ROWS {
        Launch {
            func: k.quant_function(ty, "matvec")?,
            grid: (n_out.div_ceil(4) as u32, 1, 1),
            block: 128,
            smem: 0,
            args: vec![Arg::P(w_ptr), Arg::P(x), Arg::P(bias), Arg::P(res), Arg::P(y), Arg::I(rows as i32), Arg::I(n_in as i32), Arg::I(n_out as i32), Arg::L(rb)],
        }
    } else {
        Launch {
            func: k.quant_function(ty, "matmul")?,
            grid: (n_out.div_ceil(64) as u32, rows.div_ceil(32) as u32, 1),
            block: 256,
            smem: 0,
            args: vec![Arg::P(w_ptr), Arg::P(x), Arg::P(bias), Arg::P(res), Arg::P(y), Arg::I(rows as i32), Arg::I(n_in as i32), Arg::I(n_out as i32), Arg::L(rb)],
        }
    });
    Ok(v)
}

/// Stacked expert tensor as the expert kernels address it, base plus slot times expert bytes
#[derive(Clone, Copy, Debug)]
pub(crate) struct ExpertT {
    pub ptr: u64,
    pub ty: u32,
    pub eb: i64,
    pub rb: i64,
    pub n_in: usize,
    pub n_out: usize,
}

impl ExpertT {
    pub fn new(ptr: u64, w: Wid, weights: &dyn WeightSource) -> Result<Self> {
        let d = weights.dims(w);
        let ty = weights.ggml_type(w);
        let rb = row_bytes(ty, d[0])?;
        Ok(Self { ptr, ty, eb: (rb * d[1]) as i64, rb: rb as i64, n_in: d[0], n_out: d[1] })
    }
}

pub(crate) fn act_code(a: Act) -> Result<i32> {
    Ok(match a {
        Act::Silu => 0,
        Act::Gelu => 1,
        Act::Relu => 2,
        Act::GeluErf => 3,
        Act::ReluSqr => 4,
        Act::SwigluOai => bail!("the swiglu oai activation has no device kernel"),
    })
}

/// Router product and routing kernel writing chosen experts and weights into scratch
pub(crate) fn route_launches(b: &mut CudaBackend, p: &Prepared, i: usize, weights: &dyn WeightSource) -> Result<Vec<Launch>> {
    let (mut v, route) = route_launches_split(b, p, i, weights)?;
    v.push(route);
    Ok(v)
}

/// The router product launches apart from the routing kernel, whose arguments and per row shared memory a fused
/// launch takes over
fn route_launches_split(b: &mut CudaBackend, p: &Prepared, i: usize, weights: &dyn WeightSource) -> Result<(Vec<Launch>, Launch)> {
    let node = &p.nodes[i];
    let Op::MoeFfn { router, router_bias, probs_bias, n_expert, n_used, gating, norm_weights, groups, .. } = &node.op else { bail!("routing of {:?}", node.op) };
    let rows = p.shape(node.out)?.0;
    let x = p.ptr(node.inputs[0])?;
    let rb = router_bias.map(|w| b.res.vec(w)).transpose()?.unwrap_or(0);
    let pb = probs_bias.map(|w| b.res.vec(w)).transpose()?.unwrap_or(0);
    let wr = b.res.weight(*router)?;
    let logits = p.scratch_ptr(i, S_LOGITS);
    let v = matmul_launch(b, weights, *router, wr, rb, x, p.scratch_ptr(i, S_XR), None, 0, logits, rows)?;
    let (n_group, n_group_used) = groups.map_or((0, 0), |g| (g.n_group, g.n_used));
    let route = Launch {
        func: b.kernels.function("moe", "moe_route")?,
        grid: (rows as u32, 1, 1),
        block: 32,
        smem: ((3 * n_expert + 2 * n_group.max(1)) * 4) as u32,
        args: vec![
            Arg::P(logits),
            Arg::P(pb),
            Arg::P(p.scratch_ptr(i, S_SEL)),
            Arg::P(p.scratch_ptr(i, S_WTS)),
            Arg::I(*n_expert as i32),
            Arg::I(*n_used as i32),
            Arg::I((*gating == Gating::Sigmoid) as i32),
            Arg::I(n_group as i32),
            Arg::I(n_group_used as i32),
            Arg::I((*norm_weights == WeightNorm::Clamp) as i32),
        ],
    };
    Ok((v, route))
}

/// Whether the routed expert products of a type run on int8 activations, a type with the integer form
fn expert_int8(k: &KernelRegistry, ty: u32, rows: usize, n_in: usize) -> bool {
    int8_path(k, ty, rows, n_in)
}

/// Device bytes the expert unpack scratch may take for one chunk of experts
const EXPERT_SCRATCH: usize = 96 << 20;

/// Rows up to which routed pairs take the dp4a matvec, reading each expert once per pair, past which unpacking every
/// expert for the tensor core product costs less than the rereads
const MOE_MATVEC_ROWS: usize = 64;

/// Products of routed pairs of many rows against the experts of slots: the experts unpacked into the KF8 scratch a
/// chunk at a time, each chunk one tensor core product over its pairs
#[allow(clippy::too_many_arguments)]
fn expert_gemm_chunks(b: &mut CudaBackend, t: &ExpertT, d: &MoeDims, xq: u64, xd: u64, xs: u64, order: u64, offsets: u64, per_slot: bool, y: u64, g: usize, v: &mut Vec<Launch>) -> Result<()> {
    let one = crate::kf8::Form::of(t.ty, t.n_in, t.n_out).context("expert kf8 form")?;
    ensure!(one.g == g, "expert type group {} against activations grouped by {g}", one.g);
    let per = one.buffer_bytes();
    let chunk = (EXPERT_SCRATCH / per).clamp(1, d.n_slots);
    let pool = Pool::Device(b.ctx.ordinal() as u32);
    let form = crate::kf8::Form::of(t.ty, t.n_in, t.n_out * chunk).context("expert kf8 form")?;
    let (base, grew) = b.res.kf8.scratch_for(&b.stream, pool, form.buffer_bytes())?;
    b.res.kf8.scratch_grew |= grew;
    let rows_max = if per_slot { d.rows * d.n_used } else { d.rows };
    let rows_per_expert = (d.rows * d.n_used).div_ceil(d.n_expert.max(1));
    let mut slot0 = 0;
    while slot0 < d.n_slots {
        let n = chunk.min(d.n_slots - slot0);
        let f = crate::kf8::Form::of(t.ty, t.n_in, t.n_out * n).context("expert kf8 form")?;
        let ptrs = crate::kf8::Ptrs::at(base, &f);
        v.push(crate::kf8::Store::unpack_launch(&mut b.kernels, t.ty, t.ptr + (slot0 as i64 * t.eb) as u64, t.rb, &f, ptrs)?);
        v.push(b.res.kf8.gemm_moe(&mut b.kernels, &one, ptrs, xq, xd, xs, order, offsets, per_slot, y, rows_max, d.n_used, slot0, n, rows_per_expert)?);
        slot0 += n;
    }
    Ok(())
}

/// Expert product for every routed pair against the quantized activations of the scratch at act, a warp per output row
#[allow(clippy::too_many_arguments)]
fn expert_mv_i8(k: &mut KernelRegistry, t: &ExpertT, d: &MoeDims, slots: u64, act: u64, rows: usize, per_slot: bool, y: u64) -> Result<Launch> {
    let [q, s16, m16, s32, m32] = crate::kf8::act_scratch_all(act, rows, t.n_in);
    Ok(Launch {
        func: k.quant_function(t.ty, "moe_matvec_i8")?,
        grid: (t.n_out.div_ceil(8) as u32, (d.n_used * d.rows) as u32, 1),
        block: 256,
        smem: 0,
        args: vec![Arg::P(t.ptr), Arg::L(t.eb), Arg::L(t.rb), Arg::P(slots), Arg::P(q), Arg::P(s16), Arg::P(m16), Arg::P(s32), Arg::P(m32), Arg::I(per_slot as i32), Arg::P(y), Arg::I(t.n_in as i32), Arg::I(t.n_out as i32), Arg::I(d.n_used as i32)],
    })
}

/// Gate and up products of every routed pair with the gated activation, one launch writing the hidden rows, for a
/// gate and up of one type or one tensor holding both
fn expert_gate_up_i8(k: &mut KernelRegistry, gate: Option<&ExpertT>, up: &ExpertT, d: &MoeDims, slots: u64, act: u64, rows: usize, h: u64, act_code: i32) -> Result<Launch> {
    let [q, s16, m16, s32, m32] = crate::kf8::act_scratch_all(act, rows, up.n_in);
    let (gp, up_ptr, eb_g, eb_u) = match gate {
        Some(g) => (g.ptr, up.ptr, g.eb, up.eb),
        None => (up.ptr, 0, up.eb, 0),
    };
    Ok(Launch {
        func: k.quant_function(up.ty, "moe_gate_up_i8")?,
        grid: (d.n_ff.div_ceil(8) as u32, (d.n_used * d.rows) as u32, 1),
        block: 256,
        smem: 0,
        args: vec![Arg::P(gp), Arg::P(up_ptr), Arg::L(eb_g), Arg::L(eb_u), Arg::L(up.rb), Arg::P(slots), Arg::P(q), Arg::P(s16), Arg::P(m16), Arg::P(s32), Arg::P(m32), Arg::P(h), Arg::I(up.n_in as i32), Arg::I(d.n_ff as i32), Arg::I(d.n_used as i32), Arg::I(act_code)],
    })
}

/// Expert product for every routed pair, a matrix vector product per pair for a few rows and grouped tiles otherwise
#[allow(clippy::too_many_arguments)]
fn expert_mm(k: &mut KernelRegistry, t: &ExpertT, d: &MoeDims, slots: u64, order: u64, offsets: u64, x: u64, per_slot: bool, y: u64) -> Result<Launch> {
    Ok(if d.rows <= MATVEC_ROWS {
        Launch {
            func: k.quant_function(t.ty, "moe_matvec")?,
            grid: (t.n_out.div_ceil(4) as u32, d.n_used as u32, d.rows as u32),
            block: 128,
            smem: 0,
            args: vec![Arg::P(t.ptr), Arg::L(t.eb), Arg::L(t.rb), Arg::P(slots), Arg::P(x), Arg::I(per_slot as i32), Arg::P(y), Arg::I(t.n_in as i32), Arg::I(t.n_out as i32), Arg::I(d.n_used as i32)],
        }
    } else {
        Launch {
            func: k.quant_function(t.ty, "moe_matmul")?,
            grid: (t.n_out.div_ceil(64) as u32, d.rows.div_ceil(32) as u32, d.n_slots as u32),
            block: 256,
            smem: 0,
            args: vec![
                Arg::P(t.ptr),
                Arg::L(t.eb),
                Arg::L(t.rb),
                Arg::P(order),
                Arg::P(offsets),
                Arg::P(x),
                Arg::I(per_slot as i32),
                Arg::P(y),
                Arg::I(t.n_in as i32),
                Arg::I(t.n_out as i32),
                Arg::I(d.n_used as i32),
            ],
        }
    })
}

/// Activations of one side of the expert products, the int8 scratch of rows with scales and sums per group or rounded f32
#[derive(Clone, Copy, Debug)]
pub(crate) enum ActForm {
    Int8 { base: u64, rows: usize, g: usize },
    F32 { x: u64 },
}

/// Whether every expert tensor of a layer runs the int8 path on this device
pub(crate) fn expert_int8_all(k: &KernelRegistry, d: &MoeDims, t: &[Option<ExpertT>; 3]) -> bool {
    let [gate, up, down] = t;
    up.is_some_and(|u| expert_int8(k, u.ty, d.rows, d.n_embd)) && gate.is_none_or(|g| expert_int8(k, g.ty, d.rows, d.n_embd)) && down.is_some_and(|w| expert_int8(k, w.ty, d.rows, d.n_ff))
}

/// The activation form rows of n take for a weight type, quantized or rounded into base, launches pushed
fn expert_act(k: &mut KernelRegistry, ty: u32, int8: bool, x: u64, base: u64, rows: usize, n: usize, v: &mut Vec<Launch>) -> Result<ActForm> {
    if int8 {
        let f = crate::kf8::Form::of(ty, n, 1).context("expert kf8 form")?;
        v.push(quant_launch(k, x, base, rows, n, round_code(ty)?)?);
        Ok(ActForm::Int8 { base, rows, g: f.g })
    } else {
        Ok(ActForm::F32 { x: rounded(k, ty, x, base, rows, n, v)? })
    }
}

/// The expert input in the forms of the gate and up tensors, each quantized or rounded once per pass
pub(crate) fn expert_input(b: &mut CudaBackend, p: &Prepared, i: usize, d: &MoeDims, t: &[Option<ExpertT>; 3], int8: bool, v: &mut Vec<Launch>) -> Result<(ActForm, ActForm)> {
    let node = &p.nodes[i];
    let x = p.ptr(node.inputs[0])?;
    let s = |n: usize| p.scratch_ptr(i, n);
    let up = t[1].context("up experts")?;
    let first_ty = t[0].map_or(up.ty, |g| g.ty);
    let xg = expert_act(&mut b.kernels, first_ty, int8, x, s(S_XG), d.rows, d.n_embd, v)?;
    let xu = if t[0].is_none_or(|g| g.ty == up.ty) { xg } else { expert_act(&mut b.kernels, up.ty, int8, x, s(S_XU), d.rows, d.n_embd, v)? };
    Ok((xg, xu))
}

/// One expert product over the pairs of the slot table in a given activation form, by dp4a for few rows and by the
/// tensor core tile over unpacked experts otherwise
#[allow(clippy::too_many_arguments)]
fn expert_product(b: &mut CudaBackend, p: &Prepared, i: usize, d: &MoeDims, t: &ExpertT, slots: u64, act: ActForm, per_slot: bool, y: u64, v: &mut Vec<Launch>) -> Result<()> {
    let s = |n: usize| p.scratch_ptr(i, n);
    match act {
        ActForm::Int8 { base, rows, g } => {
            if d.rows <= MOE_MATVEC_ROWS {
                v.push(expert_mv_i8(&mut b.kernels, t, d, slots, base, rows, per_slot, y)?);
            } else {
                let (q, xd, xs) = crate::kf8::act_scratch(base, rows, t.n_in, g);
                expert_gemm_chunks(b, t, d, q, xd, xs, s(S_ORDER), s(S_OFFSETS), per_slot, y, g, v)?;
            }
        }
        ActForm::F32 { x } => v.push(expert_mm(&mut b.kernels, t, d, slots, s(S_ORDER), s(S_OFFSETS), x, per_slot, y)?),
    }
    Ok(())
}

/// Gate and up products, the gated activation, the hidden activations in the down tensor's form, and the down
/// product, for the pairs of the slot table of d.n_slots slots, writing their rows of the part scratch
#[allow(clippy::too_many_arguments)]
pub(crate) fn expert_body(b: &mut CudaBackend, p: &Prepared, i: usize, d: &MoeDims, t: &[Option<ExpertT>; 3], slots: u64, acts: (ActForm, ActForm), int8: bool, v: &mut Vec<Launch>) -> Result<()> {
    let node = &p.nodes[i];
    let Op::MoeFfn { act, .. } = &node.op else { bail!("experts of {:?}", node.op) };
    let [gate, up, down] = *t;
    let up = up.context("up experts")?;
    let down = down.context("down experts")?;
    let s = |n: usize| p.scratch_ptr(i, n);
    let total = d.rows * d.n_used;
    if d.rows > MATVEC_ROWS {
        v.push(Launch {
            func: b.kernels.function("moe", "moe_group")?,
            grid: (1, 1, 1),
            block: 1024,
            smem: (d.n_slots * 4) as u32,
            args: vec![Arg::P(slots), Arg::I(total as i32), Arg::I(d.n_slots as i32), Arg::P(s(S_ORDER)), Arg::P(s(S_OFFSETS))],
        });
    }
    // Few rows on int8 activations of one form take the fused gate and up kernel, the rest the two products and the activation
    let fused = match (acts.0, gate) {
        (ActForm::Int8 { base, rows, .. }, Some(g)) if d.rows <= MOE_MATVEC_ROWS && g.ty == up.ty => Some((base, rows)),
        (ActForm::Int8 { base, rows, .. }, None) if d.rows <= MOE_MATVEC_ROWS => Some((base, rows)),
        _ => None,
    };
    if let Some((base, arows)) = fused {
        v.push(expert_gate_up_i8(&mut b.kernels, gate.as_ref(), &up, d, slots, base, arows, s(S_HIDDEN), act_code(*act)?)?);
    } else {
        let (g_ptr, u_ptr, stride) = match gate {
            Some(g) => {
                expert_product(b, p, i, d, &g, slots, acts.0, false, s(S_GATE), v)?;
                expert_product(b, p, i, d, &up, slots, acts.1, false, s(S_UP), v)?;
                (s(S_GATE), s(S_UP), d.n_ff)
            }
            None => {
                expert_product(b, p, i, d, &up, slots, acts.0, false, s(S_GATE), v)?;
                (s(S_GATE), s(S_GATE) + 4 * d.n_ff as u64, 2 * d.n_ff)
            }
        };
        v.push(flat(
            b.kernels.function("moe", "moe_glu")?,
            total * d.n_ff,
            vec![Arg::P(g_ptr), Arg::P(u_ptr), Arg::I(stride as i32), Arg::P(s(S_HIDDEN)), Arg::P(slots), Arg::I(d.n_ff as i32), Arg::I(total as i32), Arg::I(act_code(*act)?)],
        ));
    }
    let hidden = expert_act(&mut b.kernels, down.ty, int8, s(S_HIDDEN), s(S_HQ), total, d.n_ff, v)?;
    expert_product(b, p, i, d, &down, slots, hidden, true, s(S_PART), v)
}

/// The weighted sum of the pairs of the slot table plus the host partial when present
pub(crate) fn expert_combine(b: &mut CudaBackend, p: &Prepared, i: usize, d: &MoeDims, slots: u64, host: u64) -> Result<Launch> {
    let node = &p.nodes[i];
    let Op::MoeFfn { scale, .. } = &node.op else { bail!("experts of {:?}", node.op) };
    let s = |n: usize| p.scratch_ptr(i, n);
    Ok(flat(
        b.kernels.function("moe", "moe_combine")?,
        d.rows * d.n_embd,
        vec![Arg::P(s(S_PART)), Arg::P(s(S_WTS)), Arg::P(slots), Arg::P(host), Arg::P(p.ptr(node.out)?), Arg::I(d.rows as i32), Arg::I(d.n_used as i32), Arg::I(d.n_embd as i32), Arg::F(*scale)],
    ))
}

/// Expert products, gated activation and down projection for routed pairs, plus the weighted combine
pub(crate) fn expert_launches(b: &mut CudaBackend, p: &Prepared, i: usize, d: &MoeDims, t: [Option<ExpertT>; 3], slots: u64, host: u64) -> Result<(Vec<Launch>, Launch)> {
    let int8 = expert_int8_all(&b.kernels, d, &t);
    let mut v = Vec::new();
    let acts = expert_input(b, p, i, d, &t, int8, &mut v)?;
    expert_body(b, p, i, d, &t, slots, acts, int8, &mut v)?;
    let combine = expert_combine(b, p, i, d, slots, host)?;
    Ok((v, combine))
}

/// Kernel launches of a device node with its buffers and resources bound
#[allow(unreachable_patterns)]
fn device_launches(b: &mut CudaBackend, p: &Prepared, i: usize, weights: &dyn WeightSource) -> Result<Vec<Launch>> {
    use Arg::{F, I, L, P};
    let node = &p.nodes[i];
    let (rows, cols) = p.shape(node.out)?;
    if rows * cols == 0 {
        return Ok(Vec::new());
    }
    let out = p.ptr(node.out)?;
    let n = (rows * cols) as i32;
    let x = |k: usize| p.ptr(node.inputs[k]);
    let fused = p.fused[i];
    if fused == Fusion::Skip {
        return Ok(Vec::new());
    }
    if let Fusion::QuantOut { code } = fused {
        let tail = [vec![I(code)], act_args(out, rows, cols)].concat();
        let grid = quant_grid(rows, cols);
        return Ok(vec![match &node.op {
            Op::RmsNorm { weight, bias, eps } => {
                let w = weight.map(|w| b.res.vec(w)).transpose()?.unwrap_or(0);
                let bb = bias.map(|w| b.res.vec(w)).transpose()?.unwrap_or(0);
                Launch { func: b.kernels.function("fused", "rms_norm_q8")?, grid, block: 256, smem: 0, args: [vec![P(x(0)?), P(w), P(bb), I(cols as i32), F(*eps)], tail.clone()].concat() }
            }
            Op::GatedAct { act, .. } => Launch { func: b.kernels.function("fused", "gated_act_q8")?, grid, block: 256, smem: 0, args: [vec![P(x(0)?), P(x(1)?), I(cols as i32), I(cols as i32), I(act_code(*act)?)], tail.clone()].concat() },
            Op::GatedActFused { act, .. } => {
                let xc = p.shape(node.inputs[0])?.1;
                Launch { func: b.kernels.function("fused", "gated_act_q8")?, grid, block: 256, smem: 0, args: [vec![P(x(0)?), P(x(0)? + 4 * cols as u64), I(cols as i32), I(xc as i32), I(act_code(*act)?)], tail.clone()].concat() }
            }
            other => bail!("{other:?} cannot quantize its output"),
        }]);
    }
    let ew = |b: &mut CudaBackend, name: &str, args: Vec<Arg>| -> Result<Vec<Launch>> { Ok(vec![flat(b.kernels.function("elementwise", name)?, rows * cols, args)]) };
    let un = |b: &mut CudaBackend, op: i32, p0: f32, p1: f32| -> Result<Vec<Launch>> { Ok(vec![flat(b.kernels.function("elementwise", "unary")?, rows * cols, vec![P(x(0)?), P(out), I(n), I(op), F(p0), F(p1)])]) };
    let row_op = |b: &mut CudaBackend, w: Wid, op: i32| -> Result<Vec<Launch>> {
        let wn = weights.dims(w).iter().product::<usize>() as i32;
        let wp = b.res.vec(w)?;
        Ok(vec![flat(b.kernels.function("elementwise", "row_op")?, rows * cols, vec![P(x(0)?), P(wp), P(out), I(rows as i32), I(cols as i32), I(wn), I(op)])])
    };
    let binary = |b: &mut CudaBackend, op: i32| -> Result<Vec<Launch>> {
        let (br, bc) = p.shape(node.inputs[1])?;
        Ok(vec![flat(b.kernels.function("elementwise", "binary")?, rows * cols, vec![P(x(0)?), P(x(1)?), P(out), I(rows as i32), I(cols as i32), I(br as i32), I(bc as i32), I(op)])])
    };
    match &node.op {
        Op::Input { name } => match *name {
            "hidden" => ew(b, "scale", vec![P(p.hid_ptr), F(1.0), P(out), I(n)]),
            "tokens" => ew(b, "input_f32", vec![P(p.tok_ptr), P(out), I(n)]),
            _ => ew(b, "input_f32", vec![P(p.pos_ptr), P(out), I(n)]),
        },
        Op::Embed { weight } => {
            let d = weights.dims(*weight);
            let ty = weights.ggml_type(*weight);
            let rb = row_bytes(ty, d[0])? as i64;
            let w = b.res.weight(*weight)?;
            Ok(vec![flat(b.kernels.quant_function(ty, "embed")?, d[0].div_ceil(8) * rows, vec![P(w), P(p.tok_ptr), P(out), I(d[0] as i32), I(rows as i32), L(rb)])])
        }
        Op::MatMul { weight, bias, scale } => {
            let bias = bias.map(|w| b.res.vec(w)).transpose()?.unwrap_or(0);
            let w = b.res.weight(*weight)?;
            let xq = if p.scratch[i].is_empty() { 0 } else { p.scratch_ptr(i, 0) };
            let (pre, res) = match fused {
                Fusion::MatMul { act, res } => (act.map(|a| quant_out_ptr(p, a)).transpose()?, res.map(|r| p.ptr(r)).transpose()?.unwrap_or(0)),
                _ => (None, 0),
            };
            if let Some(leader) = p.grouped[i] {
                if leader != i {
                    return Ok(Vec::new());
                }
                let mut members = Vec::new();
                for (m, l) in p.grouped.iter().enumerate() {
                    if *l != Some(i) {
                        continue;
                    }
                    let Op::MatMul { weight, bias, .. } = &p.nodes[m].op else { bail!("{:?} in a product group", p.nodes[m].op) };
                    let res = match p.fused[m] {
                        Fusion::MatMul { res: Some(r), .. } => p.ptr(r)?,
                        _ => 0,
                    };
                    members.push(GroupMember { w: *weight, w_ptr: b.res.weight(*weight)?, bias: bias.map(|w| b.res.vec(w)).transpose()?.unwrap_or(0), res, y: p.ptr(p.nodes[m].out)?, n_out: weights.dims(*weight).iter().skip(1).product::<usize>().max(1) });
                }
                return matmul_group_launch(b, p, weights, &members, x(0)?, xq, pre, rows);
            }
            let mut v = matmul_launch(b, weights, *weight, w, bias, x(0)?, xq, pre, res, out, rows)?;
            if let Some(sw) = scale {
                let sv = kf_backend_cpu::dequant_all(*sw, weights)?.first().copied().context("empty scale weight")?;
                v.push(flat(b.kernels.function("elementwise", "scale")?, rows * cols, vec![P(out), F(sv), P(out), I(n)]));
            }
            Ok(v)
        }
        Op::RmsNorm { weight, bias, eps } => {
            let w = weight.map(|w| b.res.vec(w)).transpose()?.unwrap_or(0);
            let bb = bias.map(|w| b.res.vec(w)).transpose()?.unwrap_or(0);
            if let Fusion::NormAdd { add, res } = fused {
                return Ok(vec![Launch { func: b.kernels.function("norm", "rms_norm_add")?, grid: (rows as u32, 1, 1), block: 256, smem: 0, args: vec![P(x(0)?), P(w), P(bb), P(p.ptr(res)?), P(p.ptr(p.nodes[add].out)?), I(cols as i32), F(*eps)] }]);
            }
            Ok(vec![Launch { func: b.kernels.function("norm", "rms_norm")?, grid: (rows as u32, 1, 1), block: 256, smem: 0, args: vec![P(x(0)?), P(w), P(bb), P(out), I(cols as i32), F(*eps)] }])
        }
        Op::LayerNorm { weight, bias, eps } => {
            let w = weight.map(|w| b.res.vec(w)).transpose()?.unwrap_or(0);
            let bb = bias.map(|w| b.res.vec(w)).transpose()?.unwrap_or(0);
            Ok(vec![Launch { func: b.kernels.function("norm", "layer_norm")?, grid: (rows as u32, 1, 1), block: 256, smem: 0, args: vec![P(x(0)?), P(w), P(bb), P(out), I(cols as i32), F(*eps)] }])
        }
        Op::L2Norm { eps } => Ok(vec![Launch { func: b.kernels.function("norm", "l2_norm")?, grid: (rows as u32, 1, 1), block: 256, smem: 0, args: vec![P(x(0)?), P(out), I(cols as i32), F(*eps)] }]),
        Op::SumRows => {
            let xc = p.shape(node.inputs[0])?.1;
            Ok(vec![Launch { func: b.kernels.function("norm", "sum_rows")?, grid: (rows as u32, 1, 1), block: 256, smem: 0, args: vec![P(x(0)?), P(out), I(xc as i32)] }])
        }
        Op::Rope { params, n_head, head_dim, offset, inverse } => {
            let key = p.rope_keys[i].as_deref().context("rope key")?;
            let t = b.res.rope.get(key).context("rope table")?;
            let (ptr, half, mscale) = (t.ptr, t.half, t.mscale);
            let neox = (params.kind != RopeKind::Normal) as i32;
            let mode: Vec<Arg> = rope_mode(params).into_iter().map(I).collect();
            let Fusion::Rope { src, norm, append } = fused else {
                let args = [vec![P(x(0)?), P(out), P(p.rope_ptr), P(ptr), I(rows as i32), I(*n_head as i32), I(*head_dim as i32), I(half as i32), I(neox), F(mscale)], mode, vec![I(*offset as i32), F(if *inverse { -1.0 } else { 1.0 })]].concat();
                return Ok(vec![flat(b.kernels.function("rope", "rope")?, rows * cols, args)]);
            };
            let norm_of = |b: &CudaBackend, norm: Option<usize>| -> Result<Vec<Arg>> { norm_args(b, p, norm) };
            if let Some(other) = p.pair[i] {
                if other > i {
                    return Ok(Vec::new());
                }
                // This is the later rope of the pair: gather the query side and the key side and launch both
                let (qi, ki) = if append.is_some() { (other, i) } else { (i, other) };
                let (Fusion::Rope { src: qsrc, norm: qnorm, .. }, Fusion::Rope { src: ksrc, norm: knorm, append: Some((a, v)) }) = (p.fused[qi], p.fused[ki]) else { bail!("rope pair without an appending side") };
                let Op::Attention { layer, n_head_kv, head_dim_v, .. } = &p.nodes[a].op else { bail!("rope appending into {:?}", p.nodes[a].op) };
                let Op::Rope { n_head: nq, .. } = &p.nodes[qi].op else { bail!("rope pair of {:?}", p.nodes[qi].op) };
                let kv = b.res.kv.get(layer).with_context(|| format!("no cache for layer {layer}"))?;
                let (kc, vc, cap) = (dptr(&kv.k, &b.stream), dptr(&kv.v, &b.stream), kv.cap);
                let qn = norm_of(b, qnorm)?;
                let kn = norm_of(b, knorm)?;
                let warps = rows * (nq + n_head_kv);
                return Ok(vec![Launch {
                    func: b.kernels.function("fused", &format!("norm_rope_pair_hd{head_dim}"))?,
                    grid: (warps.div_ceil(4) as u32, 1, 1),
                    block: 128,
                    smem: 0,
                    args: [vec![P(p.ptr(qsrc)?), P(p.ptr(p.nodes[qi].out)?)], qn, vec![P(p.ptr(ksrc)?), P(p.ptr(v)?), P(kc), P(vc), I(cap as i32)], kn, vec![P(p.rope_ptr), P(p.pos_ptr), P(p.seq_ptr), P(ptr), I(rows as i32), I(*nq as i32), I(*n_head_kv as i32), I(*head_dim_v as i32), I(half as i32), I(neox), F(mscale)], mode].concat(),
                }]);
            }
            let norm_args = norm_of(b, norm)?;
            let src = p.ptr(src)?;
            Ok(vec![match append {
                Some((a, v)) => {
                    let Op::Attention { layer, n_head_kv, head_dim_v, .. } = &p.nodes[a].op else { bail!("rope appending into {:?}", p.nodes[a].op) };
                    let kv = b.res.kv.get(layer).with_context(|| format!("no cache for layer {layer}"))?;
                    let (kc, vc, cap) = (dptr(&kv.k, &b.stream), dptr(&kv.v, &b.stream), kv.cap);
                    let warps = rows * n_head_kv;
                    Launch {
                        func: b.kernels.function("fused", &format!("norm_rope_append_hd{head_dim}"))?,
                        grid: (warps.div_ceil(4) as u32, 1, 1),
                        block: 128,
                        smem: 0,
                        args: [vec![P(src), P(p.ptr(v)?), P(kc), P(vc), P(p.rope_ptr), P(p.pos_ptr), P(p.seq_ptr), I(cap as i32), P(ptr), I(rows as i32), I(*n_head_kv as i32), I(*head_dim_v as i32), I(half as i32), I(neox), F(mscale)], norm_args, mode].concat(),
                    }
                }
                None => {
                    let warps = rows * n_head;
                    Launch {
                        func: b.kernels.function("fused", &format!("norm_rope_hd{head_dim}"))?,
                        grid: (warps.div_ceil(4) as u32, 1, 1),
                        block: 128,
                        smem: 0,
                        args: [vec![P(src), P(out), P(p.rope_ptr), P(ptr), I(rows as i32), I(*n_head as i32), I(half as i32), I(neox), F(mscale)], norm_args, mode].concat(),
                    }
                }
            }])
        }
        Op::Attention { layer, n_head, n_head_kv, head_dim, head_dim_v, scale, window, softcap, sinks, .. } => {
            ensure!(*head_dim_v <= AT_V_MAX, "attention value width {head_dim_v} past the {AT_V_MAX} the device kernel holds");
            let kv = b.res.kv.get(layer).with_context(|| format!("no cache for layer {layer}"))?;
            let (kc, vc) = (dptr(&kv.k, &b.stream), dptr(&kv.v, &b.stream));
            let (kw, vw, cap) = (kv.kw, kv.vw, kv.cap);
            let sk = sinks.map(|w| b.res.vec(w)).transpose()?.unwrap_or(0);
            let (appended, quant, rope) = match fused {
                Fusion::Attention { appended, quant, rope } => (appended, quant, rope),
                _ => (false, None, None),
            };
            let mut v = Vec::with_capacity(3);
            if !appended {
                v.push(flat(b.kernels.function("attention", "kv_append")?, rows * (kw + vw), vec![P(x(1)?), P(x(2)?), P(kc), P(vc), P(p.pos_ptr), P(p.seq_ptr), I(rows as i32), I(kw as i32), I(vw as i32), I(cap as i32)]));
            }
            let kind = attention_kind(&b.kernels, rows, *n_head, *n_head_kv, *head_dim, *head_dim_v, cap, p.tree);
            if let AttnKind::Decode { parts } = kind {
                let parts_ptr = dptr(b.res.parts.as_ref().context("attention partials")?, &b.stream);
                let counters = dptr(b.res.counters.as_ref().context("attention counters")?, &b.stream);
                let n = n_head * head_dim_v;
                let (code, act, smem) = match quant {
                    Some(code) => (code, act_args(quant_out_ptr(p, i)?, rows, n), (n * 4) as u32),
                    None => (0, vec![P(0); 5], 0),
                };
                // With the ropes folded in, the query and key arrive from the projections and the norm and table of the ropes come along
                let (q_ptr, rope_args) = match rope {
                    Some(f) => {
                        let Op::Rope { params, .. } = &p.nodes[f.qi].op else { bail!("attention folding {:?} as a rope", p.nodes[f.qi].op) };
                        let key = p.rope_keys[f.qi].as_deref().context("rope key")?;
                        let t = b.res.rope.get(key).context("rope table")?;
                        let (cs_ptr, half, mscale) = (t.cs_ptr, t.half, t.mscale);
                        let mode: Vec<Arg> = rope_mode(params).into_iter().map(I).collect();
                        let qn = norm_args(b, p, f.qnorm)?;
                        let kn = norm_args(b, p, f.knorm)?;
                        (p.ptr(f.qsrc)?, [vec![P(p.ptr(f.ksrc)?), P(p.ptr(f.v)?), P(p.rope_ptr), P(cs_ptr), I(half as i32), I((params.kind != RopeKind::Normal) as i32), F(mscale)], mode, qn, kn].concat())
                    }
                    None => (x(0)?, [vec![P(0), P(0), P(0), P(0), I(0), I(0), F(0.0)], vec![I(0); 5], vec![I(0), P(0), P(0), F(0.0)], vec![I(0), P(0), P(0), F(0.0)]].concat()),
                };
                v.push(Launch {
                    func: b.kernels.function("attention", &format!("attention_decode_hd{head_dim}"))?,
                    grid: ((rows * n_head_kv) as u32, parts as u32, head_splits(*n_head, *n_head_kv) as u32),
                    block: 256,
                    smem,
                    args: [vec![P(q_ptr), P(kc), P(vc), P(p.pos_ptr), P(p.seq_ptr), P(p.att_ptr), P(sk), P(out), P(parts_ptr), I(*n_head as i32), I(*n_head_kv as i32), F(*scale), I(window.map_or(0, |w| w.size) as i32), F(*softcap), I(cap as i32), I(rows as i32), P(counters), I(code)], act, rope_args, vec![P(if p.tree { p.tree_ptr } else { 0 })]].concat(),
                });
                return Ok(v);
            }
            if kind == AttnKind::Flash {
                let smem = flash_smem(*head_dim);
                let flash = Launch {
                    func: b.kernels.function_smem("flash", &format!("flash_prefill_hd{head_dim}"), smem)?,
                    grid: (p.n_tiles_max as u32, *n_head as u32, 1),
                    block: 128,
                    smem,
                    args: vec![P(x(0)?), P(kc), P(vc), P(p.pos_ptr), P(p.seq_ptr), P(p.att_ptr), P(p.tiles_ptr), P(sk), P(out), I(*n_head as i32), I(*n_head_kv as i32), F(*scale), I(window.map_or(0, |w| w.size) as i32), F(*softcap), I(cap as i32)],
                };
                v.push(flash);
                return Ok(v);
            }
            let n_parts = attention_parts(rows, *n_head, cap);
            let parts = if n_parts > 1 { dptr(b.res.parts.as_ref().context("attention partials")?, &b.stream) } else { 0 };
            let attend = Launch {
                func: b.kernels.function("attention", "attention")?,
                grid: (rows as u32, *n_head as u32, n_parts as u32),
                block: 128,
                smem: ((head_dim + 8 + 4 * AT_V_MAX) * 4) as u32,
                args: vec![
                    P(x(0)?),
                    P(kc),
                    P(vc),
                    P(p.pos_ptr),
                    P(p.seq_ptr),
                    P(p.att_ptr),
                    P(sk),
                    P(out),
                    P(parts),
                    I(AT_PART as i32),
                    I(*n_head as i32),
                    I(*n_head_kv as i32),
                    I(*head_dim as i32),
                    I(*head_dim_v as i32),
                    F(*scale),
                    I(window.map_or(0, |w| w.size) as i32),
                    F(*softcap),
                    I(cap as i32),
                    P(if p.tree { p.tree_ptr } else { 0 }),
                ],
            };
            v.push(attend);
            if n_parts > 1 {
                v.push(Launch { func: b.kernels.function("attention", "attention_merge")?, grid: (rows as u32, *n_head as u32, 1), block: 256, smem: 0, args: vec![P(parts), I(n_parts as i32), I(rows as i32), P(sk), P(out), I(*n_head as i32), I(*head_dim_v as i32)] });
            }
            Ok(v)
        }
        Op::MoeFfn { gate_exps, up_exps, down_exps, n_expert, .. } => {
            let slots = experts::cache_slots(b, node.layer, *n_expert);
            if slots == 0 {
                let d = moe_dims(&node.op, p.shape(node.inputs[0])?, 0, weights)?;
                let mut t = [None; 3];
                for (slot, w) in [*gate_exps, Some(*up_exps), *down_exps].into_iter().enumerate() {
                    if let Some(w) = w {
                        t[slot] = Some(ExpertT::new(b.res.weight(w)?, w, weights)?);
                    }
                }
                let mut v = route_launches(b, p, i, weights)?;
                let (body, combine) = expert_launches(b, p, i, &d, t, p.scratch_ptr(i, S_SEL), 0)?;
                v.extend(body);
                v.push(combine);
                return Ok(v);
            }
            // Cached experts of a few rows: the dispatch kernel resolves slots and posts the misses to the coprocessor,
            // the slot products run on the device, and the await kernel brings the host partial in for the combine
            let layer = node.layer.context("cached experts outside any block")?;
            let d = moe_dims(&node.op, p.shape(node.inputs[0])?, slots, weights)?;
            ensure!(d.rows <= coproc::COPROC_ROWS, "{} rows on the coprocessor path", d.rows);
            let cache = b.res.experts.get(&layer).context("expert cache")?;
            let t = cache.tensors(weights)?;
            let table = cache.table_ptr();
            let c = b.coproc.as_ref().context("coprocessor")?;
            let mb = c.mailbox(layer).context("coprocessor mailbox")?;
            let seq_ptr = c.seq_ptr;
            let s = |k: usize| p.scratch_ptr(i, k);
            let (mut v, route) = route_launches_split(b, p, i, weights)?;
            v.push(Launch {
                func: b.kernels.function("moe", "moe_route_dispatch")?,
                grid: (1, 1, 1),
                block: 256,
                smem: route.smem * rows as u32,
                args: [route.args, vec![P(table), P(s(S_SLOTS)), P(x(0)?), I(rows as i32), I(d.n_embd as i32), P(mb.header), P(mb.sel), P(mb.miss), P(mb.x), P(seq_ptr), P(s(S_NMISS))]].concat(),
            });
            let (body, combine) = expert_launches(b, p, i, &d, t, s(S_SLOTS), s(S_HOST))?;
            v.extend(body);
            v.push(Launch { func: b.kernels.function("moe", "moe_await")?, grid: (1, 1, 1), block: 256, smem: 0, args: vec![P(mb.header), P(mb.partial), P(seq_ptr), P(s(S_NMISS)), P(s(S_HOST)), I((d.rows * d.n_embd) as i32)] });
            v.push(combine);
            Ok(v)
        }
        Op::GatedAct { act, .. } => ew(b, "gated_act", vec![P(x(0)?), P(x(1)?), P(out), I(n), I(act_code(*act)?)]),
        Op::GatedActFused { act, .. } => {
            let xc = p.shape(node.inputs[0])?.1;
            ew(b, "gated_act_fused", vec![P(x(0)?), P(out), I(rows as i32), I(xc as i32), I(cols as i32), I(act_code(*act)?)])
        }
        Op::Add => binary(b, 0),
        Op::Sub => binary(b, 1),
        Op::Mul => binary(b, 2),
        Op::Div => binary(b, 3),
        Op::MulWeight { weight } => row_op(b, *weight, 0),
        Op::AddWeight { weight } => row_op(b, *weight, 1),
        Op::DivWeight { weight } => row_op(b, *weight, 2),
        Op::Scale(s) => ew(b, "scale", vec![P(x(0)?), F(*s), P(out), I(n)]),
        Op::Silu => ew(b, "silu", vec![P(x(0)?), P(out), I(n)]),
        Op::Gelu => ew(b, "gelu", vec![P(x(0)?), P(out), I(n)]),
        Op::Relu => ew(b, "relu", vec![P(x(0)?), P(out), I(n)]),
        Op::Sigmoid => ew(b, "sigmoid", vec![P(x(0)?), P(out), I(n)]),
        Op::Tanh => un(b, 0, 0.0, 0.0),
        Op::Exp => un(b, 1, 0.0, 0.0),
        Op::Sqrt => un(b, 2, 0.0, 0.0),
        Op::Sqr => un(b, 3, 0.0, 0.0),
        Op::Abs => un(b, 4, 0.0, 0.0),
        Op::Sign => un(b, 5, 0.0, 0.0),
        Op::Log => un(b, 6, 0.0, 0.0),
        Op::Round => un(b, 7, 0.0, 0.0),
        Op::Softplus { beta } => un(b, 8, *beta, 0.0),
        Op::Clamp { min, max } => un(b, 9, *min, *max),
        Op::ScaleBias { scale, bias } => un(b, 10, *scale, *bias),
        Op::GeluErf => un(b, 11, 0.0, 0.0),
        Op::ReluSqr => un(b, 12, 0.0, 0.0),
        Op::GeluQuick => un(b, 13, 0.0, 0.0),
        Op::AttentionFull { n_head, head_dim, scale } => Ok(vec![Launch {
            func: b.kernels.function("attention", "attention_full")?,
            grid: (rows as u32, *n_head as u32, head_dim.div_ceil(512) as u32),
            block: 128,
            smem: ((head_dim + 8 + 4 * 512) * 4) as u32,
            args: vec![P(x(0)?), P(x(1)?), P(x(2)?), P(out), I(rows as i32), I(*n_head as i32), I(*head_dim as i32), F(*scale)],
        }]),
        Op::PoolGrid { side, kernel } => ew(b, "pool_grid", vec![P(x(0)?), P(out), I(*side as i32), I(*kernel as i32), I(cols as i32)]),
        Op::Fill { value, .. } => Ok(vec![flat(b.kernels.function("elementwise", "unary")?, rows * cols, vec![P(out), P(out), I(n), I(10), F(0.0), F(*value)])]),
        Op::Softmax => Ok(vec![Launch { func: b.kernels.function("norm", "softmax")?, grid: (rows as u32, 1, 1), block: 256, smem: 0, args: vec![P(x(0)?), P(out), I(cols as i32)] }]),
        Op::TopK { k } => {
            let xc = p.shape(node.inputs[0])?.1;
            topk_launches(&mut b.kernels, "sample", x(0)?, out, p.scratch_ptr(i, 0), p.scratch_ptr(i, 1), rows, xc, *k)
        }
        Op::Softcap(c) => ew(b, "softcap", vec![P(x(0)?), F(*c), P(out), I(n)]),
        Op::Reshape { .. } | Op::LastRows { .. } | Op::Rows { .. } => Ok(Vec::new()),
        Op::Cols { start, len } => {
            let xc = p.shape(node.inputs[0])?.1;
            ew(b, "cols", vec![P(x(0)?), P(out), I(rows as i32), I(xc as i32), I(*start as i32), I(*len as i32)])
        }
        Op::Concat => {
            let ca = p.shape(node.inputs[0])?.1;
            let cb = p.shape(node.inputs[1])?.1;
            ew(b, "concat", vec![P(x(0)?), P(x(1)?), P(out), I(rows as i32), I(ca as i32), I(cb as i32)])
        }
        Op::Tile { n: times } => {
            let xc = p.shape(node.inputs[0])?.1;
            ew(b, "tile", vec![P(x(0)?), P(out), I(rows as i32), I(xc as i32), I(*times as i32)])
        }
        Op::PadCols { to } => {
            let xc = p.shape(node.inputs[0])?.1;
            ew(b, "pad_cols", vec![P(x(0)?), P(out), I(rows as i32), I(xc as i32), I(*to as i32)])
        }
        Op::MaskCols { start, len, value } => ew(b, "mask_cols", vec![P(x(0)?), P(out), I(rows as i32), I(cols as i32), I(*start as i32), I(*len as i32), F(*value)]),
        other => bail!("{other:?} has no device launch"),
    }
}

/// Runs a contiguous device segment, captured into a CUDA graph on first decode and replayed after
fn run_device_segment(b: &mut CudaBackend, p: &mut Prepared, si: usize, range: Range<usize>, inputs: &Inputs, weights: &dyn WeightSource, decode: bool) -> Result<()> {
    let pending: Vec<usize> = range.clone().filter(|&i| p.shapes[p.nodes[i].out.0 as usize].is_none()).collect();
    if !pending.is_empty() {
        let mut planned = Vec::with_capacity(pending.len());
        for &i in &pending {
            let (shape, scratch) = infer_node(b, p, i, weights)?;
            p.shapes[p.nodes[i].out.0 as usize] = Some(shape);
            planned.push((shape, scratch));
        }
        plan_fusions(b, p, &pending, weights)?;
        for (&i, (shape, scratch)) in pending.iter().zip(&planned) {
            p.plan(&mut b.pool, &b.stream, i, *shape, scratch)?;
        }
    }
    let mut appended = HashMap::new();
    for i in range.clone() {
        ensure_node(b, p, i, inputs, weights, &mut appended)?;
    }
    if p.epoch != Some(b.res.epoch) {
        p.epoch = Some(b.res.epoch);
        p.launches.iter_mut().for_each(|l| *l = None);
        p.graphs.iter_mut().for_each(|g| *g = None);
        p.persist.iter_mut().for_each(|g| *g = None);
        p.segs_used.set(0);
    }
    // Launches hold the address of the unpack scratch, so a build that moved it starts the range over
    loop {
        for i in range.clone() {
            if p.launches[i].is_none() {
                p.launches[i] = Some(device_launches(b, p, i, weights)?);
            }
        }
        if !std::mem::take(&mut b.res.kf8.scratch_grew) {
            break;
        }
        b.res.epoch += 1;
        p.epoch = Some(b.res.epoch);
        p.launches.iter_mut().for_each(|l| *l = None);
        p.graphs.iter_mut().for_each(|g| *g = None);
        p.persist.iter_mut().for_each(|g| *g = None);
        p.segs_used.set(0);
    }
    let issue = |p: &Prepared, stream: &CudaStream| range.clone().try_for_each(|i| p.launches[i].iter().flatten().try_for_each(|l| l.run(stream)));
    // Decode steps, verify passes of a few rows, and whole prefill chunks repeat their launch sequence exactly, so all replay from a graph
    let graphed = decode || p.n_tok <= coproc::COPROC_ROWS || (p.tree && p.n_tok <= kf_core::TREE_ROWS) || p.n_tok == b.n_batch;
    if range.clone().any(|i| p.launches[i].as_ref().is_some_and(|l| !l.is_empty())) {
        if let Ok(trace) = std::env::var("KF_TRACE_KERNELS") {
            // Per kernel device time between events, each kernel repeated the number of times the variable names so
            // the device never waits on the host inside the bracket, every kernel of a pass being safe to rerun in place
            let reps: usize = trace.parse::<usize>().ok().filter(|n| *n > 0).unwrap_or(1);
            let mut timed = Vec::new();
            for i in range.clone() {
                for l in p.launches[i].iter().flatten() {
                    let e0 = b.stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
                    for _ in 0..reps {
                        l.run(&b.stream)?;
                    }
                    let e1 = b.stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
                    timed.push((i, l, e0, e1));
                }
            }
            b.stream.synchronize()?;
            for (i, l, e0, e1) in timed {
                eprintln!("kernel {} node {i:4} grid {:?} block {} smem {}: {:.1} us", b.kernels.kernel_name(&l.func), l.grid, l.block, l.smem, e0.elapsed_ms(&e1)? * 1000.0 / reps as f32);
            }
        } else if !graphed {
            issue(p, &b.stream)?;
        } else if let Some(Captured(graph)) = &p.graphs[si] {
            graph.launch().context("graph launch")?;
            b.replays += 1;
            if let Some(plan) = &p.persist[si] {
                plan.report(b)?;
            }
        } else {
            // Under KF_PERSIST a single row decode pass runs as one persistent launch when every kernel of the segment has a stage form
            if decode && p.n_tok == 1 && std::env::var_os("KF_PERSIST").is_some() && p.persist[si].is_none() {
                let launches: Vec<&Launch> = range.clone().flat_map(|i| p.launches[i].iter().flatten()).collect();
                p.persist[si] = persist::plan(b, &launches)?;
            }
            b.stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL).context("begin capture")?;
            let issued = match &p.persist[si] {
                Some(plan) => plan.run(&b.stream),
                None => issue(p, &b.stream),
            };
            let graph = b.stream.end_capture(sys::CUgraphInstantiate_flags(0)).context("end capture");
            issued?;
            let graph = graph?.context("capture recorded no work")?;
            graph.launch().context("graph launch")?;
            b.captures += 1;
            p.graphs[si] = Some(Captured(graph));
        }
    }
    for (layer, runs) in appended {
        if let Some(kv) = b.res.kv.get_mut(&layer) {
            for (seq, rows) in runs {
                kv.n[seq] += rows;
            }
        }
    }
    Ok(())
}

/// Runs a node through the reference backend on host copies of its inputs and uploads the result
fn run_host_node(b: &mut CudaBackend, p: &mut Prepared, i: usize, inputs: &Inputs, weights: &dyn WeightSource) -> Result<()> {
    b.stream.synchronize().context("stream sync")?;
    let node = p.nodes[i].clone();
    let mut ins = Vec::with_capacity(node.inputs.len());
    for t in &node.inputs {
        let (rows, cols) = p.shape(*t)?;
        let mut data = vec![0.0f32; rows * cols];
        dtoh(&b.stream, &mut data, p.ptr(*t)?)?;
        ins.push(kf_backend_cpu::Buf { data, rows, cols });
    }
    let out = b.host.exec_node(&node.op, &ins, inputs, weights)?;
    match p.shapes[node.out.0 as usize] {
        Some(s) => ensure!(s == (out.rows, out.cols), "host node {i} gave {}x{} where {}x{} was planned", out.rows, out.cols, s.0, s.1),
        None => p.plan(&mut b.pool, &b.stream, i, (out.rows, out.cols), &[])?,
    }
    htod(&b.stream, p.ptr(node.out)?, &out.data)
}

/// Executes a prepared graph segment by segment and copies the output back
fn run_prepared(b: &mut CudaBackend, p: &mut Prepared, inputs: &Inputs, weights: &dyn WeightSource, decode: bool) -> Result<Vec<f32>> {
    let n = inputs.tokens.len();
    ensure!(inputs.positions.len() == n, "{} positions for {} tokens", inputs.positions.len(), n);
    ensure!(n == p.n_tok, "a pass of {n} tokens on a graph prepared for {}", p.n_tok);
    // The rope angle tables cover every position of the pass
    let top = inputs.positions.iter().copied().chain(inputs.rope.iter().flat_map(|s| s[..3].iter().copied())).max().unwrap_or(0) as usize + 1;
    if top > b.rope_positions {
        b.rope_positions = top;
        ensure_rope_positions(b, top)?;
    }
    // The integer inputs staged in pinned memory in one layout: tokens, positions, slots, attend bounds, then the four
    // rotary streams of each row, a text row's position in the first three and zero in the fourth, as the reference reads them
    {
        let st = p.stage_in.as_mut_slice().context("pinned pass inputs")?;
        for (r, t) in inputs.tokens.iter().enumerate() {
            st[r] = *t as i32;
            st[n + r] = inputs.positions[r] as i32;
            st[2 * n + r] = inputs.seq_of(r) as i32;
            st[3 * n + r] = inputs.attend_of(r) as i32;
            let streams = match inputs.rope.get(r) {
                Some(s) => [s[0] as i32, s[1] as i32, s[2] as i32, s[3] as i32],
                None => {
                    let p = inputs.positions[r] as i32;
                    [p, p, p, 0]
                }
            };
            st[4 * n + 4 * r..4 * n + 4 * r + 4].copy_from_slice(&streams);
            let t = inputs.tree_of(r).unwrap_or_default();
            st[8 * n + 3 * r] = t.mask as u32 as i32;
            st[8 * n + 3 * r + 1] = (t.mask >> 32) as u32 as i32;
            st[8 * n + 3 * r + 2] = t.position as i32;
        }
        htod(&b.stream, p.tok_ptr, &st[..IN_WORDS * n])?;
    }
    ensure!(inputs.tree.is_empty() || inputs.tree.len() == n, "{} tree rows for {n} tokens", inputs.tree.len());
    htod(&b.stream, p.hid_ptr, &inputs.hidden)?;
    if !decode {
        htod(&b.stream, p.tiles_ptr, &flash_tiles(inputs, p.n_tiles_max))?;
    }
    if std::env::var_os("KF_TRACE_KERNELS").is_some() {
        eprintln!("pass rows={} decode={decode}", inputs.tokens.len());
    }
    let trace_pass = std::env::var_os("KF_TRACE_PASS").is_some();
    let t0 = std::time::Instant::now();
    if let Some(c) = b.coproc.as_mut() {
        c.begin_pass(&b.stream)?;
    }
    for si in 0..p.segments.len() {
        match p.segments[si].clone() {
            Segment::Device(range) => run_device_segment(b, p, si, range, inputs, weights, decode)?,
            Segment::Host(i) => run_host_node(b, p, i, inputs, weights)?,
            Segment::Experts(i) => experts::run(b, p, i, inputs, weights)?,
            Segment::Stream(i) => run_stream_attention(b, p, i, inputs, weights)?,
        }
    }
    let t_issue = t0.elapsed();
    let (rows, cols) = p.shape(p.output)?;
    let ptr = p.ptr(p.output)?;
    // The output lands in pinned memory, one transfer at full speed, then one copy out
    if p.stage_out.as_ref().is_none_or(|s| s.len() < rows * cols) {
        p.stage_out = Some(unsafe { b.ctx.alloc_pinned_with_flags::<f32>(rows * cols, 0) }.context("pinned output")?);
    }
    let landing = p.stage_out.as_mut().context("pinned output")?;
    dtoh(&b.stream, &mut landing.as_mut_slice().context("pinned output")?[..rows * cols], ptr)?;
    let data = landing.as_slice().context("pinned output")?[..rows * cols].to_vec();
    experts::end_pass(b, weights)?;
    if trace_pass {
        let launches: usize = p.launches.iter().flatten().map(Vec::len).sum();
        let stages: usize = p.persist.iter().flatten().map(|pl| pl.n_stages).sum();
        eprintln!("pass rows={} decode={decode} launches={launches} graphs={} persistent_stages={stages} issue={:.1} ms wall={:.1} ms", inputs.tokens.len(), p.graphs.iter().flatten().count(), t_issue.as_secs_f64() * 1e3, t0.elapsed().as_secs_f64() * 1e3);
    }
    b.host.account_state();
    b.last = Some(Output { ptr, rows, cols });
    let topk = p.nodes.iter().find(|n| n.out == p.output).filter(|n| matches!(n.op, Op::TopK { .. })).map(|n| n.inputs[0]);
    b.dense = match topk {
        Some(t) => {
            let (r, c) = p.shape(t)?;
            Some(Output { ptr: p.ptr(t)?, rows: r, cols: c })
        }
        None => None,
    };
    b.hidden = match p.hidden {
        Some(t) => {
            let (r, c) = p.shape(t)?;
            Some(Output { ptr: p.ptr(t)?, rows: r, cols: c })
        }
        None => None,
    };
    if let Some(h) = b.hidden {
        // The kept rows are the last ones of the pass, a tree's rows going to the ring with the path compaction keeps
        ensure!(h.rows <= n, "{} hidden rows for a pass of {n} tokens", h.rows);
        let first = n - h.rows;
        if p.tree {
            b.tree_tokens = (first..n).map(|r| (inputs.seq_of(r) as u32, inputs.tokens[r])).collect();
        } else {
            let rows: Vec<(u32, u32, u32)> = (first..n).map(|r| (inputs.seq_of(r) as u32, inputs.positions[r], inputs.tokens[r])).collect();
            ring_append(b, h.ptr, &rows, h.cols)?;
        }
    }
    Ok(data)
}

/// Appends scored rows to the feature ring once a draft head has made one
pub(crate) fn ring_append(b: &mut CudaBackend, src: u64, rows: &[(u32, u32, u32)], cols: usize) -> Result<()> {
    let Some(ring) = b.ring.as_mut() else { return Ok(()) };
    ensure!(ring.cols == cols, "hidden rows of {cols} for a feature ring of {}", ring.cols);
    ring.append(&b.stream, src, rows)
}

/// The residual rows before the final norm the last pass kept
pub(crate) fn hidden_rows(b: &mut CudaBackend) -> Result<Vec<f32>> {
    let h = b.hidden.ok_or_else(|| anyhow!("the last pass kept no hidden rows"))?;
    b.ctx.bind_to_thread()?;
    let mut out = vec![0.0f32; h.rows * h.cols];
    dtoh(&b.stream, &mut out, h.ptr)?;
    Ok(out)
}

/// One row of the tensor the last pass's final top k read
pub(crate) fn dense_row(b: &mut CudaBackend, row: usize) -> Result<Vec<f32>> {
    let d = b.dense.ok_or_else(|| anyhow!("the last pass ended in no top k"))?;
    ensure!(row < d.rows, "row {row} of {} dense rows", d.rows);
    b.ctx.bind_to_thread()?;
    let mut out = vec![0.0f32; d.cols];
    dtoh(&b.stream, &mut out, d.ptr + (row * d.cols * 4) as u64)?;
    Ok(out)
}

pub(crate) fn run(b: &mut CudaBackend, g: &Graph, inputs: &Inputs, weights: &dyn WeightSource) -> Result<Vec<f32>> {
    if let Some(e) = b.pending.take() {
        return Err(e);
    }
    b.ctx.bind_to_thread()?;
    b.last = None;
    let decode = inputs.runs().iter().all(|(_, r)| r.len() == 1);
    let mut p = match b.prepared.iter().position(|p| p.matches(g, inputs)) {
        Some(i) => b.prepared.swap_remove(i),
        None => {
            if b.prepared.len() >= PREPARED_KEPT {
                b.prepared.remove(0).release(&mut b.pool);
            }
            Prepared::new(b, g, inputs, weights)?
        }
    };
    let out = run_prepared(b, &mut p, inputs, weights, decode);
    b.prepared.push(p);
    out
}

/// Keeps the first n positions of one slot's caches, or every slot's none for a reset, on device and host
pub(crate) fn truncate(b: &mut CudaBackend, seq: Option<usize>, n: usize) -> Result<()> {
    b.ctx.bind_to_thread()?;
    match seq {
        Some(seq) => {
            ensure!(seq < b.host.n_seq(), "sequence slot {seq} of {}", b.host.n_seq());
            let stream = b.stream.clone();
            for kv in b.res.kv.values_mut() {
                if kv.n[seq] > n {
                    kv.n[seq] = n;
                    refill_ring(&stream, kv, seq, n)?;
                }
            }
            kf_core::Backend::truncate(&mut b.host, seq, n)
        }
        None => {
            for kv in b.res.kv.values_mut() {
                kv.n.iter_mut().for_each(|n| *n = 0);
            }
            kf_core::Backend::reset(&mut b.host);
            Ok(())
        }
    }
}

/// Moves the cache rows base plus keep[i] of one slot to base plus i in every device layer and the host layers, then
/// drops the rows past them
pub(crate) fn compact(b: &mut CudaBackend, seq: usize, base: usize, keep: &[usize]) -> Result<()> {
    b.ctx.bind_to_thread()?;
    ensure!(seq < b.host.n_seq(), "sequence slot {seq} of {}", b.host.n_seq());
    ensure!(!keep.is_empty() && keep.len() <= kf_core::TREE_ROWS, "{} rows kept of a tree pass", keep.len());
    for (i, k) in keep.iter().enumerate() {
        ensure!(*k >= i, "kept row {k} stands before its place {i}");
    }
    let idx: Vec<i32> = keep.iter().map(|k| *k as i32).collect();
    let idx_dev = b.upload(&idx, AllocKind::Compute, "compaction index")?;
    let idx_ptr = dptr(&idx_dev, &b.stream);
    let func = b.kernels.function("attention", "kv_compact")?;
    for kv in b.res.kv.values() {
        let last = base + keep.iter().copied().max().unwrap_or(0);
        ensure!(last < kv.n[seq], "kept row {last} past the {} rows of the cache", kv.n[seq]);
        ensure!(kv.n[seq] - base <= kv.resident, "a tree pass of {} rows past the {} resident rows of the ring", kv.n[seq] - base, kv.resident);
        let smem = (keep.len() * (kv.kw + kv.vw) * 2) as u32;
        let l = Launch { func: func.clone(), grid: (1, 1, 1), block: 256, smem, args: vec![Arg::P(dptr(&kv.k, &b.stream)), Arg::P(dptr(&kv.v, &b.stream)), Arg::I(kv.cap as i32), Arg::I(kv.kw as i32), Arg::I(kv.vw as i32), Arg::I(seq as i32), Arg::I(base as i32), Arg::P(idx_ptr), Arg::I(keep.len() as i32)] };
        if smem > 48 * 1024 {
            l.func.set_attribute(cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem as i32).context("compaction shared memory limit")?;
        }
        l.run(&b.stream)?;
    }
    b.stream.synchronize().context("compaction")?;
    kf_core::Backend::compact(&mut b.host, seq, base, keep)?;
    // The kept path's rows join the ring in order, where their positions now run
    if let Some(h) = b.hidden
        && b.tree_tokens.len() == h.rows
    {
        let tokens = std::mem::take(&mut b.tree_tokens);
        for (i, k) in keep.iter().enumerate() {
            ensure!(*k < h.rows, "kept row {k} of a tree pass of {} rows", h.rows);
            let (s, token) = tokens[*k];
            ensure!(s as usize == seq, "kept row {k} belongs to slot {s}, not {seq}");
            ring_append(b, h.ptr + (*k * h.cols * 4) as u64, &[(seq as u32, (base + i) as u32, token)], h.cols)?;
        }
    }
    truncate(b, Some(seq), base + keep.len())
}

/// Copies one slot's device cache rings down into the host backend's layer states and saves them in the shared form
pub(crate) fn save_state(b: &mut CudaBackend, seq: usize) -> Result<Vec<u8>> {
    b.ctx.bind_to_thread()?;
    b.stream.synchronize()?;
    let mut saved = b.host.saved(seq)?;
    saved.n_ctx = b.n_ctx;
    saved.n_batch = b.n_batch;
    for (layer, d) in &b.res.kv {
        let n = d.n[seq];
        let rows = n.min(d.cap);
        let mut ring = kf_backend_cpu::KvLayer::with_capacity(0, d.kw, d.vw);
        ring.cap = d.cap;
        ring.n = n;
        let mut k = vec![0u16; rows * d.kw];
        let mut v = vec![0u16; rows * d.vw];
        if d.streams() {
            let host_rows_per = d.host_rows();
            let from_host = rows.min(host_rows_per);
            for (host, dst, w) in [(d.host_k.as_ref(), &mut k, d.kw), (d.host_v.as_ref(), &mut v, d.vw)] {
                let host = host.context("host rows of a streamed cache")?;
                dst[..from_host * w].copy_from_slice(unsafe { host_rows(host, seq * host_rows_per * w, from_host * w)? });
            }
            for pos in from_host..rows {
                let slot = pos % d.resident;
                dtoh(&b.stream, &mut k[pos * d.kw..(pos + 1) * d.kw], dptr(&d.k, &b.stream) + (((seq * d.resident + slot) * d.kw) * 2) as u64)?;
                dtoh(&b.stream, &mut v[pos * d.vw..(pos + 1) * d.vw], dptr(&d.v, &b.stream) + (((seq * d.resident + slot) * d.vw) * 2) as u64)?;
            }
        } else {
            dtoh(&b.stream, &mut k, dptr(&d.k, &b.stream) + (seq * d.cap * d.kw * 2) as u64)?;
            dtoh(&b.stream, &mut v, dptr(&d.v, &b.stream) + (seq * d.cap * d.vw * 2) as u64)?;
        }
        ring.k = k.into_iter().map(half::f16::from_bits).collect();
        ring.v = v.into_iter().map(half::f16::from_bits).collect();
        saved.layers.entry(*layer).or_default().kv = Some(ring);
    }
    saved.to_bytes()
}

/// Fills one slot's cache rings of device placed layers from a saved state and hands the rest to the host backend
pub(crate) fn load_state(b: &mut CudaBackend, seq: usize, bytes: &[u8]) -> Result<()> {
    let mut saved = kf_backend_cpu::SavedState::from_bytes(bytes)?;
    ensure!(saved.n_ctx == b.n_ctx && saved.n_batch == b.n_batch, "state saved under a context of {} and batch of {}, this plan has {} and {}", saved.n_ctx, saved.n_batch, b.n_ctx, b.n_batch);
    ensure!(seq < b.host.n_seq(), "sequence slot {seq} of {}", b.host.n_seq());
    b.ctx.bind_to_thread()?;
    b.stream.synchronize()?;
    truncate(b, Some(seq), 0)?;
    for (layer, st) in saved.layers.iter_mut() {
        if b.placement(Some(*layer)).1 != b.pool() {
            continue;
        }
        let Some(ring) = st.kv.take() else { continue };
        if !b.res.kv.contains_key(layer) {
            let full = ring.cap == kf_core::cache_rows(b.n_ctx, b.n_batch, None);
            let window = (!full).then_some(ring.cap.saturating_sub(b.n_batch));
            ensure_kv(b, *layer, ring.k_width, ring.v_width, &[], window)?;
        }
        let stream = b.stream.clone();
        let d = b.res.kv.get_mut(layer).context("cache")?;
        ensure!(d.cap == ring.cap && d.kw == ring.k_width && d.vw == ring.v_width, "layer {layer} state of {} rows by {} and {} against a ring of {} by {} and {}", ring.cap, ring.k_width, ring.v_width, d.cap, d.kw, d.vw);
        let rows = ring.n.min(ring.cap);
        ensure!(ring.k.len() == rows * ring.k_width && ring.v.len() == rows * ring.v_width, "layer {layer} state carries {} key and {} value elements for {rows} rows", ring.k.len(), ring.v.len());
        let k: Vec<u16> = ring.k.iter().map(|x| x.to_bits()).collect();
        let v: Vec<u16> = ring.v.iter().map(|x| x.to_bits()).collect();
        if d.streams() {
            let host_rows_per = d.host_rows();
            let to_host = rows.min(host_rows_per);
            for (host, src, w) in [(d.host_k.as_mut(), &k, d.kw), (d.host_v.as_mut(), &v, d.vw)] {
                let host = host.context("host rows of a streamed cache")?;
                unsafe { host_rows_mut(host, seq * host_rows_per * w, to_host * w)? }.copy_from_slice(&src[..to_host * w]);
            }
            for pos in rows.saturating_sub(d.resident)..rows {
                let slot = pos % d.resident;
                htod(&stream, dptr(&d.k, &stream) + (((seq * d.resident + slot) * d.kw) * 2) as u64, &k[pos * d.kw..(pos + 1) * d.kw])?;
                htod(&stream, dptr(&d.v, &stream) + (((seq * d.resident + slot) * d.vw) * 2) as u64, &v[pos * d.vw..(pos + 1) * d.vw])?;
            }
        } else {
            htod(&stream, dptr(&d.k, &stream) + (seq * d.cap * d.kw * 2) as u64, &k)?;
            htod(&stream, dptr(&d.v, &stream) + (seq * d.cap * d.vw * 2) as u64, &v)?;
        }
        d.n[seq] = ring.n;
    }
    b.stream.synchronize()?;
    b.host.restore(seq, saved)
}

pub(crate) fn apply_plan(b: &mut CudaBackend, plan: &Plan) -> Result<()> {
    let ord = b.ctx.ordinal() as u32;
    for l in &plan.layers {
        for pool in [l.weights, l.state] {
            if let Pool::Device(n) = pool {
                ensure!(n == ord, "layer {} placed on device {n}, backend runs device {ord}", l.layer);
            }
        }
    }
    b.ctx.bind_to_thread()?;
    b.stream.synchronize()?;
    for p in b.prepared.drain(..) {
        p.release(&mut b.pool);
    }
    ensure!(plan.n_ctx > 0 && plan.n_batch > 0 && plan.n_seq > 0, "plan without a context length, batch, or sequence slots");
    b.last = None;
    b.res.weights.clear();
    b.res.vecs.clear();
    b.res.rope.clear();
    b.res.copier = None;
    b.res.experts.clear();
    b.coproc = None;
    b.res.expert_stage = None;
    b.res.table_stage = None;
    b.res.kf8.clear();
    b.res.kv.clear();
    b.res.stage = None;
    b.res.parts = None;
    b.res.counters = None;
    b.res.epoch += 1;
    let pool = b.pool();
    for kind in [AllocKind::Weights, AllocKind::State, AllocKind::Experts, AllocKind::Compute] {
        kf_core::event::release(pool, kind, kf_core::event::total(pool, kind));
    }
    kf_core::event::release(Pool::Host, AllocKind::Experts, kf_core::event::total(Pool::Host, AllocKind::Experts));
    kf_core::event::release(Pool::Host, AllocKind::State, kf_core::event::total(Pool::Host, AllocKind::State));
    kf_core::Backend::reset(&mut b.host);
    b.host.set_limits(plan.n_ctx, plan.n_batch, plan.n_seq);
    b.n_ctx = plan.n_ctx;
    b.n_batch = plan.n_batch;
    b.res.kf8.planned = plan.weights_bytes + plan.state_bytes;
    b.layers = plan.layers.iter().map(|l| (l.layer, l.clone())).collect();
    b.pinned = plan.pinned.clone();
    Ok(())
}

/// Whole weight expanded to f32 by the device dequant kernel of its type
pub(crate) fn dequant(b: &mut CudaBackend, w: Wid, weights: &dyn WeightSource) -> Result<Vec<f32>> {
    b.ctx.bind_to_thread()?;
    let (_buf, ptr, n) = expand(b, w, weights)?;
    let mut out = vec![0.0f32; n];
    dtoh(&b.stream, &mut out, ptr)?;
    kf_core::event::release(b.pool(), AllocKind::Weights, (n * 4) as u64);
    Ok(out)
}

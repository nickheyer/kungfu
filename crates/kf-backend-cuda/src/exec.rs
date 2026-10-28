use crate::CudaBackend;
use crate::experts;
use crate::registry::{KernelRegistry, cuda_kernel_for_op};
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

pub(crate) struct RopeTable {
    _buf: CudaSlice<f32>,
    ptr: u64,
    half: usize,
    mscale: f32,
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
    weights: HashMap<Wid, (CudaSlice<u8>, u64)>,
    vecs: HashMap<Wid, (CudaSlice<f32>, u64)>,
    rope: HashMap<String, RopeTable>,
    pub kv: HashMap<usize, DevKv>,
    pub experts: HashMap<usize, experts::ExpertCache>,
    argmax: Option<CudaSlice<i32>>,
    stage: Option<Stage>,
    /// Partial softmaxes of a streamed attention pass, grown to the largest pass seen
    parts: Option<CudaSlice<f32>>,
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
    _tok: CudaSlice<i32>,
    _pos: CudaSlice<i32>,
    _seq: CudaSlice<i32>,
    _att: CudaSlice<i32>,
    _hid: CudaSlice<f32>,
    pub tok_ptr: u64,
    pos_ptr: u64,
    seq_ptr: u64,
    att_ptr: u64,
    hid_ptr: u64,
    n_hid: usize,
    hidden_cols: usize,
    epoch: Option<u64>,
    launches: Vec<Option<Vec<Launch>>>,
    graphs: Vec<Option<Captured>>,
}

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
        let place: Vec<Place> = g.nodes.iter().map(|n| place_node(b, n, weights)).collect();
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
        let tok = b.alloc::<i32>(inputs.tokens.len().max(1), AllocKind::Compute, "token buffer")?;
        let pos = b.alloc::<i32>(inputs.positions.len().max(1), AllocKind::Compute, "position buffer")?;
        let seq = b.alloc::<i32>(inputs.tokens.len().max(1), AllocKind::Compute, "sequence buffer")?;
        let att = b.alloc::<i32>(inputs.tokens.len().max(1), AllocKind::Compute, "attend buffer")?;
        let hid = b.alloc::<f32>(inputs.hidden.len().max(1), AllocKind::Compute, "hidden buffer")?;
        let n_nodes = g.nodes.len();
        Ok(Self {
            nodes: g.nodes.clone(),
            output,
            n_tok: inputs.tokens.len(),
            n_pos: inputs.positions.len(),
            place,
            graphs: (0..segments.len()).map(|_| None).collect(),
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
            tok_ptr: dptr(&tok, &b.stream),
            pos_ptr: dptr(&pos, &b.stream),
            seq_ptr: dptr(&seq, &b.stream),
            att_ptr: dptr(&att, &b.stream),
            hid_ptr: dptr(&hid, &b.stream),
            n_hid: inputs.hidden.len(),
            hidden_cols: inputs.hidden_cols,
            _tok: tok,
            _pos: pos,
            _seq: seq,
            _att: att,
            _hid: hid,
            epoch: None,
            launches: vec![None; n_nodes],
        })
    }

    fn matches(&self, g: &Graph, inputs: &Inputs) -> bool {
        self.n_tok == inputs.tokens.len()
            && self.n_pos == inputs.positions.len()
            && self.n_hid == inputs.hidden.len()
            && self.hidden_cols == inputs.hidden_cols
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
            None => {
                let b = self.assign(pool, stream, shape.0 * shape.1, last)?;
                self.loc[out] = Some((b, 0));
            }
        }
        let mut taken = Vec::with_capacity(scratch.len());
        for &n in scratch {
            taken.push(self.assign(pool, stream, n.max(1), i)?);
        }
        self.free.extend(taken.iter().copied());
        self.scratch[i] = taken;
        for t in tensors {
            if let Some((b, _)) = self.loc[t]
                && self.buf_last[b] == i
                && !self.free.contains(&b)
            {
                self.free.push(b);
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
fn place_node(b: &CudaBackend, node: &Node, weights: &dyn WeightSource) -> Place {
    let Some(kref) = cuda_kernel_for_op(&b.kernels, &node.op) else { return Place::Host };
    if !kref.quant.is_empty() && !op_weights(&node.op).iter().all(|w| b.kernels.has_quant(weights.ggml_type(*w))) {
        return Place::Host;
    }
    let (wpool, _, slots) = b.placement(node.layer);
    match &node.op {
        Op::Embed { .. } | Op::MatMul { .. } | Op::MoeFfn { .. } if wpool == Pool::Host => Place::Host,
        Op::MoeFfn { .. } if slots > 0 => Place::Experts,
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
    vec![d.rows * d.n_expert, rj, rj, rj, rj, d.n_slots + 1, rj * 2 * d.n_ff, rj * d.n_ff, rj * d.n_ff, rj * d.n_embd, xe, xe, xe, xe, rj * d.n_ff]
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

/// Plans node i on the device side, its shape from the shape rules and scratch for expert layers
pub(crate) fn plan_device_node(b: &mut CudaBackend, p: &mut Prepared, i: usize, weights: &dyn WeightSource) -> Result<()> {
    let node = &p.nodes[i];
    let ins: Vec<(usize, usize)> = node.inputs.iter().map(|t| p.shape(*t)).collect::<Result<_>>()?;
    let hidden = if p.hidden_cols > 0 { (p.n_hid / p.hidden_cols, p.hidden_cols) } else { (0, 0) };
    let shape = infer(&node.op, &ins, p.n_tok, p.n_pos, hidden, weights)?;
    let scratch = match &node.op {
        Op::MoeFfn { n_expert, .. } => {
            let slots = if p.place[i] == Place::Experts { b.placement(node.layer).2.min(*n_expert).max(1) } else { 0 };
            moe_scratch(&moe_dims(&node.op, ins[0], slots, weights)?)
        }
        Op::MatMul { weight, .. } if round_code(weights.ggml_type(*weight))? != 0 => vec![ins[0].0 * ins[0].1],
        _ => Vec::new(),
    };
    p.plan(&mut b.pool, &b.stream, i, shape, &scratch)
}

pub(crate) fn ensure_weight(b: &mut CudaBackend, w: Wid, weights: &dyn WeightSource) -> Result<u64> {
    if let Some(x) = b.res.weights.get(&w) {
        return Ok(x.1);
    }
    let bytes = weights.bytes(w)?;
    let buf = b.upload(bytes, AllocKind::Weights, weights.name(w))?;
    let ptr = dptr(&buf, &b.stream);
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

fn ensure_rope(b: &mut CudaBackend, key: &str, params: &RopeParams, head_dim: usize, weights: &dyn WeightSource) -> Result<()> {
    if b.res.rope.contains_key(key) {
        return Ok(());
    }
    let factors = params.factors.map(|f| kf_backend_cpu::dequant_all(f, weights)).transpose()?;
    let (table, half, mscale) = rope_table(params, head_dim, factors.as_deref())?;
    let buf = if table.is_empty() { b.alloc::<f32>(1, AllocKind::Weights, "rope table")? } else { b.upload(&table, AllocKind::Weights, "rope table")? };
    let ptr = dptr(&buf, &b.stream);
    b.res.rope.insert(key.to_string(), RopeTable { _buf: buf, ptr, half, mscale });
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

/// A partials buffer of at least n floats
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
    let smem = ((head_dim + 8 + 4 * 512) * 4) as u32;
    let launch_part = |kb: u64, vb: u64, row0: usize, n_rows: usize, base: usize, ring: usize, jlo: usize, jhi: usize, part: usize| -> Result<()> {
        let l = Launch {
            func: part_fn.clone(),
            grid: (n_rows as u32, *n_head as u32, head_dim_v.div_ceil(512) as u32),
            block: 128,
            smem,
            args: vec![P(q), P(kb), P(vb), P(p.pos_ptr), P(p.seq_ptr), P(p.att_ptr), I(row0 as i32), I(base as i32), I(ring as i32), I(jlo as i32), I(jhi as i32), P(parts), I(part as i32), I(rows as i32), I(*n_head as i32), I(*n_head_kv as i32), I(*head_dim as i32), I(*head_dim_v as i32), F(*scale), I(window.map_or(0, |w| w.size) as i32), F(*softcap)],
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
        Op::Attention { layer, n_head_kv, head_dim, head_dim_v, sinks, window, .. } => {
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
            if let Some(s) = sinks {
                ensure_vec(b, *s, weights)?;
            }
        }
        Op::MoeFfn { router, router_bias, probs_bias, gate_exps, up_exps, down_exps, .. } => {
            ensure_weight(b, *router, weights)?;
            for v in [router_bias, probs_bias].into_iter().flatten() {
                ensure_vec(b, *v, weights)?;
            }
            if p.place[i] == Place::Experts {
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

/// Rows a warp level product handles before the tiled product takes over
pub(crate) const MATVEC_ROWS: usize = 8;

/// Activation rounding then a matrix vector product for a few rows or a shared memory tiled product
#[allow(clippy::too_many_arguments)]
pub(crate) fn matmul_launch(k: &mut KernelRegistry, weights: &dyn WeightSource, w: Wid, w_ptr: u64, bias: u64, x: u64, xq: u64, y: u64, rows: usize) -> Result<Vec<Launch>> {
    let d = weights.dims(w);
    let ty = weights.ggml_type(w);
    let n_in = d[0];
    let n_out: usize = d.iter().skip(1).product::<usize>().max(1);
    let rb = row_bytes(ty, n_in)? as i64;
    let mut v = Vec::with_capacity(2);
    let x = rounded(k, ty, x, xq, rows, n_in, &mut v)?;
    v.push(if rows <= MATVEC_ROWS {
        Launch {
            func: k.quant_function(ty, "matvec")?,
            grid: (n_out.div_ceil(4) as u32, 1, 1),
            block: 128,
            smem: 0,
            args: vec![Arg::P(w_ptr), Arg::P(x), Arg::P(bias), Arg::P(y), Arg::I(rows as i32), Arg::I(n_in as i32), Arg::I(n_out as i32), Arg::L(rb)],
        }
    } else {
        Launch {
            func: k.quant_function(ty, "matmul")?,
            grid: (n_out.div_ceil(64) as u32, rows.div_ceil(32) as u32, 1),
            block: 256,
            smem: 0,
            args: vec![Arg::P(w_ptr), Arg::P(x), Arg::P(bias), Arg::P(y), Arg::I(rows as i32), Arg::I(n_in as i32), Arg::I(n_out as i32), Arg::L(rb)],
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
    let node = &p.nodes[i];
    let Op::MoeFfn { router, router_bias, probs_bias, n_expert, n_used, gating, norm_weights, groups, .. } = &node.op else { bail!("routing of {:?}", node.op) };
    let rows = p.shape(node.out)?.0;
    let x = p.ptr(node.inputs[0])?;
    let rb = router_bias.map(|w| b.res.vec(w)).transpose()?.unwrap_or(0);
    let pb = probs_bias.map(|w| b.res.vec(w)).transpose()?.unwrap_or(0);
    let wr = b.res.weight(*router)?;
    let logits = p.scratch_ptr(i, S_LOGITS);
    let mut v = matmul_launch(&mut b.kernels, weights, *router, wr, rb, x, p.scratch_ptr(i, S_XR), logits, rows)?;
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
    v.push(route);
    Ok(v)
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

/// Expert products, gated activation and down projection for routed pairs, plus the weighted combine
pub(crate) fn expert_launches(b: &mut CudaBackend, p: &Prepared, i: usize, d: &MoeDims, t: [Option<ExpertT>; 3], slots: u64, host: u64) -> Result<(Vec<Launch>, Launch)> {
    let node = &p.nodes[i];
    let Op::MoeFfn { scale, act, .. } = &node.op else { bail!("experts of {:?}", node.op) };
    let [gate, up, down] = t;
    let up = up.context("up experts")?;
    let down = down.context("down experts")?;
    let k = &mut b.kernels;
    let s = |n: usize| p.scratch_ptr(i, n);
    let x = p.ptr(node.inputs[0])?;
    let total = d.rows * d.n_used;
    let mut v = Vec::new();
    if d.rows > MATVEC_ROWS {
        v.push(Launch {
            func: k.function("moe", "moe_group")?,
            grid: (1, 1, 1),
            block: 1024,
            smem: (d.n_slots * 4) as u32,
            args: vec![Arg::P(slots), Arg::I(total as i32), Arg::I(d.n_slots as i32), Arg::P(s(S_ORDER)), Arg::P(s(S_OFFSETS))],
        });
    }
    let xg = match &gate {
        Some(g) => rounded(k, g.ty, x, s(S_XG), d.rows, d.n_embd, &mut v)?,
        None => x,
    };
    let xu = if gate.as_ref().is_some_and(|g| g.ty == up.ty) { xg } else { rounded(k, up.ty, x, s(S_XU), d.rows, d.n_embd, &mut v)? };
    let (g_ptr, u_ptr, stride) = match gate {
        Some(g) => {
            v.push(expert_mm(k, &g, d, slots, s(S_ORDER), s(S_OFFSETS), xg, false, s(S_GATE))?);
            v.push(expert_mm(k, &up, d, slots, s(S_ORDER), s(S_OFFSETS), xu, false, s(S_UP))?);
            (s(S_GATE), s(S_UP), d.n_ff)
        }
        None => {
            v.push(expert_mm(k, &up, d, slots, s(S_ORDER), s(S_OFFSETS), xu, false, s(S_GATE))?);
            (s(S_GATE), s(S_GATE) + 4 * d.n_ff as u64, 2 * d.n_ff)
        }
    };
    v.push(flat(
        k.function("moe", "moe_glu")?,
        total * d.n_ff,
        vec![Arg::P(g_ptr), Arg::P(u_ptr), Arg::I(stride as i32), Arg::P(s(S_HIDDEN)), Arg::P(slots), Arg::I(d.n_ff as i32), Arg::I(total as i32), Arg::I(act_code(*act)?)],
    ));
    let hq = rounded(k, down.ty, s(S_HIDDEN), s(S_HQ), total, d.n_ff, &mut v)?;
    v.push(expert_mm(k, &down, d, slots, s(S_ORDER), s(S_OFFSETS), hq, true, s(S_PART))?);
    let combine = flat(
        k.function("moe", "moe_combine")?,
        d.rows * d.n_embd,
        vec![
            Arg::P(s(S_PART)),
            Arg::P(s(S_WTS)),
            Arg::P(slots),
            Arg::P(host),
            Arg::P(p.ptr(node.out)?),
            Arg::I(d.rows as i32),
            Arg::I(d.n_used as i32),
            Arg::I(d.n_embd as i32),
            Arg::F(*scale),
        ],
    );
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
            let mut v = matmul_launch(&mut b.kernels, weights, *weight, w, bias, x(0)?, xq, out, rows)?;
            if let Some(sw) = scale {
                let sv = kf_backend_cpu::dequant_all(*sw, weights)?.first().copied().context("empty scale weight")?;
                v.push(flat(b.kernels.function("elementwise", "scale")?, rows * cols, vec![P(out), F(sv), P(out), I(n)]));
            }
            Ok(v)
        }
        Op::RmsNorm { weight, bias, eps } => {
            let w = weight.map(|w| b.res.vec(w)).transpose()?.unwrap_or(0);
            let bb = bias.map(|w| b.res.vec(w)).transpose()?.unwrap_or(0);
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
        Op::Rope { params, n_head, head_dim, .. } => {
            let key = p.rope_keys[i].as_deref().context("rope key")?;
            let t = b.res.rope.get(key).context("rope table")?;
            let neox = (params.kind != RopeKind::Normal) as i32;
            let args = vec![P(x(0)?), P(out), P(p.pos_ptr), P(t.ptr), I(rows as i32), I(*n_head as i32), I(*head_dim as i32), I(t.half as i32), I(neox), F(t.mscale)];
            Ok(vec![flat(b.kernels.function("rope", "rope")?, rows * cols, args)])
        }
        Op::Attention { layer, n_head, n_head_kv, head_dim, head_dim_v, scale, window, softcap, sinks, .. } => {
            let kv = b.res.kv.get(layer).with_context(|| format!("no cache for layer {layer}"))?;
            let (kc, vc) = (dptr(&kv.k, &b.stream), dptr(&kv.v, &b.stream));
            let (kw, vw, cap) = (kv.kw, kv.vw, kv.cap);
            let sk = sinks.map(|w| b.res.vec(w)).transpose()?.unwrap_or(0);
            let append = flat(b.kernels.function("attention", "kv_append")?, rows * (kw + vw), vec![P(x(1)?), P(x(2)?), P(kc), P(vc), P(p.pos_ptr), P(p.seq_ptr), I(rows as i32), I(kw as i32), I(vw as i32), I(cap as i32)]);
            let attend = Launch {
                func: b.kernels.function("attention", "attention")?,
                grid: (rows as u32, *n_head as u32, head_dim_v.div_ceil(512) as u32),
                block: 128,
                smem: ((head_dim + 8 + 4 * 512) * 4) as u32,
                args: vec![
                    P(x(0)?),
                    P(kc),
                    P(vc),
                    P(p.pos_ptr),
                    P(p.seq_ptr),
                    P(p.att_ptr),
                    P(sk),
                    P(out),
                    I(*n_head as i32),
                    I(*n_head_kv as i32),
                    I(*head_dim as i32),
                    I(*head_dim_v as i32),
                    F(*scale),
                    I(window.map_or(0, |w| w.size) as i32),
                    F(*softcap),
                    I(cap as i32),
                ],
            };
            Ok(vec![append, attend])
        }
        Op::MoeFfn { gate_exps, up_exps, down_exps, .. } => {
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
    for i in range.clone() {
        if p.shapes[p.nodes[i].out.0 as usize].is_none() {
            plan_device_node(b, p, i, weights)?;
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
    }
    for i in range.clone() {
        if p.launches[i].is_none() {
            p.launches[i] = Some(device_launches(b, p, i, weights)?);
        }
    }
    let issue = |p: &Prepared, stream: &CudaStream| range.clone().try_for_each(|i| p.launches[i].iter().flatten().try_for_each(|l| l.run(stream)));
    if range.clone().any(|i| p.launches[i].as_ref().is_some_and(|l| !l.is_empty())) {
        if !decode {
            issue(p, &b.stream)?;
        } else if let Some(Captured(graph)) = &p.graphs[si] {
            graph.launch().context("graph launch")?;
            b.replays += 1;
        } else {
            b.stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL).context("begin capture")?;
            let issued = issue(p, &b.stream);
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
    let tok: Vec<i32> = inputs.tokens.iter().map(|t| *t as i32).collect();
    let pos: Vec<i32> = inputs.positions.iter().map(|t| *t as i32).collect();
    let seq: Vec<i32> = (0..inputs.tokens.len()).map(|r| inputs.seq_of(r) as i32).collect();
    let att: Vec<i32> = (0..inputs.tokens.len()).map(|r| inputs.attend_of(r) as i32).collect();
    ensure!(inputs.positions.len() == inputs.tokens.len(), "{} positions for {} tokens", inputs.positions.len(), inputs.tokens.len());
    htod(&b.stream, p.tok_ptr, &tok)?;
    htod(&b.stream, p.pos_ptr, &pos)?;
    htod(&b.stream, p.seq_ptr, &seq)?;
    htod(&b.stream, p.att_ptr, &att)?;
    htod(&b.stream, p.hid_ptr, &inputs.hidden)?;
    for si in 0..p.segments.len() {
        match p.segments[si].clone() {
            Segment::Device(range) => run_device_segment(b, p, si, range, inputs, weights, decode)?,
            Segment::Host(i) => run_host_node(b, p, i, inputs, weights)?,
            Segment::Experts(i) => experts::run(b, p, i, inputs, weights)?,
            Segment::Stream(i) => run_stream_attention(b, p, i, inputs, weights)?,
        }
    }
    let (rows, cols) = p.shape(p.output)?;
    let ptr = p.ptr(p.output)?;
    let mut data = vec![0.0f32; rows * cols];
    dtoh(&b.stream, &mut data, ptr)?;
    b.host.account_state();
    b.last = Some(Output { ptr, rows, cols });
    Ok(data)
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
    b.res.experts.clear();
    b.res.kv.clear();
    b.res.stage = None;
    b.res.parts = None;
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
    b.layers = plan.layers.iter().map(|l| (l.layer, l.clone())).collect();
    b.pinned = plan.pinned.iter().copied().collect();
    Ok(())
}

pub(crate) fn argmax_last(b: &mut CudaBackend, out: Output) -> Result<u32> {
    ensure!(out.rows > 0 && out.cols > 0, "argmax of an empty output");
    b.ctx.bind_to_thread()?;
    if b.res.argmax.is_none() {
        b.res.argmax = Some(b.alloc::<i32>(1, AllocKind::Compute, "argmax buffer")?);
    }
    let idx = dptr(b.res.argmax.as_ref().context("argmax buffer")?, &b.stream);
    let row = out.ptr + ((out.rows - 1) * out.cols * 4) as u64;
    let l = Launch { func: b.kernels.function("sample", "argmax")?, grid: (1, 1, 1), block: 1024, smem: 0, args: vec![Arg::P(row), Arg::I(out.cols as i32), Arg::P(idx)] };
    l.run(&b.stream)?;
    let mut v = [0i32];
    dtoh(&b.stream, &mut v, idx)?;
    Ok(v[0] as u32)
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

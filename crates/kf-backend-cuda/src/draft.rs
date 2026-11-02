//! The draft block on the device, see kf_core::DraftConfig and the host reference in kf_backend_cpu::draft
use crate::CudaBackend;
use crate::exec::{Arg, Launch, dptr, flat, row_bytes};
use crate::train::{self, ATTN_T, Tensor};
use anyhow::{Result, anyhow, ensure};
use cudarc::driver::CudaSlice;
use cudarc::driver::result::memcpy_dtod_async;
use kf_backend_cpu::draft::{FC_E, FC_G, FUSE, MASK, MATRICES, N1, N2, WD, WG, WK, WO, WQ, WU, WV};
use kf_core::{AllocKind, DRAFT_CAL_BLOCK, DRAFT_CAL_KEEP, DRAFT_CANDS, DRAFT_DEPTH, DraftConfig, Objective, RingMeta, TrainStep, Tree, WeightSource, Wid, tree_of_block};

/// Floats one packed candidate row holds
const CAND_W: usize = 2 * DRAFT_CANDS + 1;

/// Rows the drafting work space holds, a prefill chunk or a path with its mask rows
const DRAFT_WORK_ROWS: usize = 512;

/// The ggml type id of q8_0, the form of the snapshot's rows
const Q8_0: u32 = 8;

fn copy(b: &CudaBackend, dst: u64, src: u64, bytes: usize) -> Result<()> {
    unsafe { memcpy_dtod_async(dst, src, bytes, b.stream.cu_stream()) }?;
    Ok(())
}

/// c = op(a) op(b) + beta c over row major operands as the host reference's mm
#[allow(clippy::too_many_arguments)]
fn mm(b: &mut CudaBackend, a: u64, bm: u64, c: u64, m: usize, n: usize, k: usize, ta: bool, tb: bool, beta: f32) -> Result<()> {
    let (lda, ldb) = (if ta { m } else { k }, if tb { k } else { n });
    train::gemm(b, a, bm, c, m, n, k, lda, ldb, n, ta, tb, 1.0, beta)
}

/// The block's matrices as the drafting pass reads them, every one transposed into q8_0 rows, with the int8
/// activation scratch the products round their rows into
pub struct Snapshot {
    rows: Vec<CudaSlice<u8>>,
    shape: Vec<(usize, usize)>,
    act: CudaSlice<u8>,
}

impl Snapshot {
    fn new(b: &CudaBackend, cfg: &DraftConfig) -> Result<Self> {
        let shapes = cfg.param_shapes();
        let mut rows = Vec::with_capacity(MATRICES.len());
        let mut shape = Vec::with_capacity(MATRICES.len());
        let mut kmax = 0;
        for which in MATRICES {
            let (_, k, n) = shapes[which];
            ensure!(k.is_multiple_of(32), "a matrix of {k} inputs for int8 rows of 32");
            rows.push(crate::exec::dev_alloc::<u8>(&b.stream, b.pool(), n * (k / 32) * 34, AllocKind::Weights, "draft snapshot")?);
            shape.push((k, n));
            kmax = kmax.max(k);
        }
        let act = crate::exec::dev_alloc::<u8>(&b.stream, b.pool(), crate::kf8::act_scratch_bytes(crate::exec::MM_ROWS, kmax), AllocKind::Compute, "draft snapshot activations")?;
        Ok(Self { rows, shape, act })
    }

    /// Requantizes every matrix from the mixture's parameters
    fn refresh(&mut self, b: &mut CudaBackend, mixture: &[Tensor]) -> Result<()> {
        for (slot, which) in MATRICES.iter().enumerate() {
            let (k, n) = self.shape[slot];
            train::quant_q8_0_t(b, mixture[*which].ptr(), dptr(&self.rows[slot], &b.stream), k, n)?;
        }
        Ok(())
    }

    /// y = x w for m rows of x through the int8 rows of matrix which, the rows rounded to int8 blocks first
    fn product(&self, b: &mut CudaBackend, which: usize, x: u64, m: usize, y: u64) -> Result<()> {
        let slot = MATRICES.iter().position(|w| *w == which).ok_or_else(|| anyhow!("no int8 rows for parameter {which}"))?;
        let (k, n) = self.shape[slot];
        ensure!(crate::exec::int8_path(&b.kernels, Q8_0, m, k), "the int8 product does not run q8_0 rows of {k} on this device");
        let code = crate::exec::round_code(Q8_0)?;
        let per = crate::exec::matvec_rows(Q8_0, k, &[]);
        let kind = crate::exec::multi_kind(Q8_0, k, &[]);
        let w = dptr(&self.rows[slot], &b.stream);
        let act = dptr(&self.act, &b.stream);
        let rb = (k / 32 * 34) as i64;
        for r0 in (0..m).step_by(per) {
            let r = (m - r0).min(per);
            crate::exec::quant_launch(&mut b.kernels, x + (r0 * k * 4) as u64, act, r, k, code)?.run(&b.stream)?;
            crate::exec::matvec_i8_launch(&mut b.kernels, Q8_0, w, act, 0, 0, y + (r0 * n * 4) as u64, r, k, n, rb, 0, 0, kind)?.run(&b.stream)?;
        }
        Ok(())
    }
}

/// How a forward reads the block, through a learner's single precision parameters or through the snapshot
#[derive(Clone, Copy)]
enum Mats<'a> {
    Exact(&'a [Tensor]),
    Snap(&'a Snapshot),
}

impl Mats<'_> {
    #[allow(clippy::too_many_arguments)]
    fn product(&self, b: &mut CudaBackend, which: usize, x: u64, m: usize, k: usize, n: usize, y: u64) -> Result<()> {
        match self {
            Mats::Exact(p) => mm(b, x, p[which].ptr(), y, m, n, k, false, false, 0.0),
            Mats::Snap(s) => s.product(b, which, x, m, y),
        }
    }

    fn small(&self, which: usize, mixture: &[Tensor]) -> u64 {
        match self {
            Mats::Exact(p) => p[which].ptr(),
            Mats::Snap(_) => mixture[which].ptr(),
        }
    }
}

/// One learner, the block with its parameters, gradients, optimizer moments, last directions, and rate
pub struct Block {
    pub cfg: DraftConfig,
    pub params: Vec<Tensor>,
    grads: Vec<Tensor>,
    m: Vec<Tensor>,
    v: Vec<Tensor>,
    u: Vec<Tensor>,
    pub step: i32,
    pub lr: f32,
}

impl Block {
    pub fn new(b: &CudaBackend, cfg: DraftConfig, lr: f32) -> Result<Self> {
        cfg.check()?;
        let values = kf_backend_cpu::draft::init_params(&cfg);
        let (mut params, mut grads, mut m, mut v, mut u) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for ((name, rows, cols), vals) in cfg.param_shapes().iter().zip(values) {
            params.push(Tensor::upload(b, &vals, *rows, *cols, name)?);
            grads.push(Tensor::zeros(b, *rows, *cols, name)?);
            m.push(Tensor::zeros(b, *rows, *cols, name)?);
            v.push(Tensor::zeros(b, *rows, *cols, name)?);
            u.push(Tensor::zeros(b, *rows, *cols, name)?);
        }
        Ok(Self { cfg, params, grads, m, v, u, step: 0, lr })
    }

    fn p(&self, i: usize) -> u64 {
        self.params[i].ptr()
    }

    /// One optimistic AdamW step over every parameter, the gradients cleared
    pub fn adam(&mut self, b: &mut CudaBackend) -> Result<()> {
        self.step += 1;
        for i in 0..self.params.len() {
            let n = self.params[i].len();
            train::adamw_opt(b, self.params[i].ptr(), self.grads[i].ptr(), self.m[i].ptr(), self.v[i].ptr(), self.u[i].ptr(), n, self.lr, 0.9, 0.999, 1e-8, self.cfg.weight_decay, self.step)?;
            self.grads[i].zero(b)?;
        }
        Ok(())
    }

    pub fn params_host(&self, b: &CudaBackend) -> Result<Vec<Vec<f32>>> {
        self.params.iter().map(|t| t.download(b)).collect()
    }

    pub fn grads_host(&self, b: &CudaBackend) -> Result<Vec<Vec<f32>>> {
        self.grads.iter().map(|t| t.download(b)).collect()
    }

    /// Replaces the parameters, the last directions starting over
    pub fn set_params(&mut self, b: &CudaBackend, values: &[Vec<f32>]) -> Result<()> {
        ensure!(values.len() == self.params.len(), "{} parameter tensors for a block of {}", values.len(), self.params.len());
        for ((t, u), v) in self.params.iter_mut().zip(&self.u).zip(values) {
            t.set(b, v)?;
            u.zero(b)?;
        }
        Ok(())
    }

    /// Gradients of the second half given dout in w.dout, the parameter gradients added, dx1 and da left in the work
    fn layer_out_bwd(&mut self, b: &mut CudaBackend, w: &Rows, m: usize) -> Result<()> {
        let (d, ff) = (self.cfg.d, self.cfg.ff);
        let g: Vec<u64> = self.grads.iter().map(|t| t.ptr()).collect();
        mm(b, w.mm, w.dout, g[WD], ff, d, m, true, false, 1.0)?;
        mm(b, w.dout, self.p(WD), w.dm, m, ff, d, false, true, 0.0)?;
        train::silu_gate_bwd(b, w.g, w.u, w.dm, w.dg, w.du, m * ff)?;
        mm(b, w.h2, w.dg, g[WG], d, ff, m, true, false, 1.0)?;
        mm(b, w.h2, w.du, g[WU], d, ff, m, true, false, 1.0)?;
        mm(b, w.dg, self.p(WG), w.dh2, m, d, ff, false, true, 0.0)?;
        mm(b, w.du, self.p(WU), w.dh2, m, d, ff, false, true, 1.0)?;
        train::rms_bwd(b, w.x1, self.p(N2), w.inv2, w.dh2, w.dx1n, g[N2], m, d)?;
        copy(b, w.dx1, w.dout, m * d * 4)?;
        train::axpy(b, w.dx1n, w.dx1, m * d, 1.0)?;
        mm(b, w.a, w.dx1, g[WO], d, d, m, true, false, 1.0)?;
        mm(b, w.dx1, self.p(WO), w.da, m, d, d, false, true, 0.0)
    }

    /// Gradients of the first half given dx1 and the attention's dq dk dv in the work, the parameter gradients
    /// added, dg left in w.dg_in and de in w.de
    fn layer_in_bwd(&mut self, b: &mut CudaBackend, w: &Rows, g: u64, e: u64, m: usize) -> Result<()> {
        let d = self.cfg.d;
        let gr: Vec<u64> = self.grads.iter().map(|t| t.ptr()).collect();
        mm(b, w.h1, w.dq, gr[WQ], d, d, m, true, false, 1.0)?;
        mm(b, w.h1, w.dk, gr[WK], d, d, m, true, false, 1.0)?;
        mm(b, w.h1, w.dv, gr[WV], d, d, m, true, false, 1.0)?;
        mm(b, w.dq, self.p(WQ), w.dh1, m, d, d, false, true, 0.0)?;
        mm(b, w.dk, self.p(WK), w.dh1, m, d, d, false, true, 1.0)?;
        mm(b, w.dv, self.p(WV), w.dh1, m, d, d, false, true, 1.0)?;
        train::rms_bwd(b, w.x0, self.p(N1), w.inv1, w.dh1, w.dx0n, gr[N1], m, d)?;
        copy(b, w.dx0, w.dx1, m * d * 4)?;
        train::axpy(b, w.dx0n, w.dx0, m * d, 1.0)?;
        mm(b, g, w.dx0, gr[FC_G], d, d, m, true, false, 1.0)?;
        mm(b, e, w.dx0, gr[FC_E], d, d, m, true, false, 1.0)?;
        mm(b, w.dx0, self.p(FC_G), w.dg_in, m, d, d, false, true, 0.0)?;
        mm(b, w.dx0, self.p(FC_E), w.de, m, d, d, false, true, 0.0)
    }

    /// Gradient of the fusion given dg over m rows of tap features
    fn fuse_bwd(&mut self, b: &mut CudaBackend, taps: u64, dg: u64, m: usize) -> Result<()> {
        let (d, td) = (self.cfg.d, self.cfg.taps * self.cfg.d);
        mm(b, taps, dg, self.grads[FUSE].ptr(), td, d, m, true, false, 1.0)
    }
}

/// g = taps fuse for m rows of tap features through the given matrices
fn fuse(b: &mut CudaBackend, mats: Mats<'_>, cfg: &DraftConfig, taps: u64, g: u64, m: usize) -> Result<()> {
    mats.product(b, FUSE, taps, m, cfg.taps * cfg.d, cfg.d, g)
}

/// The layer up to its queries, keys, and values for m rows of g and e into the work's rows
#[allow(clippy::too_many_arguments)]
fn layer_in(b: &mut CudaBackend, mats: Mats<'_>, mixture: &[Tensor], cfg: &DraftConfig, w: &Rows, g: u64, e: u64, m: usize) -> Result<()> {
    let d = cfg.d;
    mats.product(b, FC_G, g, m, d, d, w.x0)?;
    mats.product(b, FC_E, e, m, d, d, w.tmp)?;
    train::axpy(b, w.tmp, w.x0, m * d, 1.0)?;
    train::rms_fwd(b, w.x0, mats.small(N1, mixture), w.h1, w.inv1, m, d, cfg.eps)?;
    mats.product(b, WQ, w.h1, m, d, d, w.q)?;
    mats.product(b, WK, w.h1, m, d, d, w.k)?;
    mats.product(b, WV, w.h1, m, d, d, w.v)
}

/// The layer from the attention output to the block's output rows
fn layer_out(b: &mut CudaBackend, mats: Mats<'_>, mixture: &[Tensor], cfg: &DraftConfig, w: &Rows, m: usize) -> Result<()> {
    let (d, ff) = (cfg.d, cfg.ff);
    copy(b, w.x1, w.x0, m * d * 4)?;
    mats.product(b, WO, w.a, m, d, d, w.tmp)?;
    train::axpy(b, w.tmp, w.x1, m * d, 1.0)?;
    train::rms_fwd(b, w.x1, mats.small(N2, mixture), w.h2, w.inv2, m, d, cfg.eps)?;
    mats.product(b, WG, w.h2, m, d, ff, w.g)?;
    mats.product(b, WU, w.h2, m, d, ff, w.u)?;
    train::silu_gate_fwd(b, w.g, w.u, w.mm, m * ff)?;
    copy(b, w.out, w.x1, m * d * 4)?;
    mats.product(b, WD, w.mm, m, ff, d, w.tmp)?;
    train::axpy(b, w.tmp, w.out, m * d, 1.0)
}

/// Pointers of one step's rows through the layer, forward and backward
#[derive(Clone, Copy)]
struct Rows {
    x0: u64,
    h1: u64,
    inv1: u64,
    q: u64,
    k: u64,
    v: u64,
    a: u64,
    x1: u64,
    h2: u64,
    inv2: u64,
    g: u64,
    u: u64,
    mm: u64,
    out: u64,
    tmp: u64,
    dout: u64,
    dm: u64,
    dg: u64,
    du: u64,
    dh2: u64,
    dx1: u64,
    dx1n: u64,
    da: u64,
    dq: u64,
    dk: u64,
    dv: u64,
    dh1: u64,
    dx0: u64,
    dx0n: u64,
    dg_in: u64,
    de: u64,
}

/// Work space of the training step, both query groups' rows at every drafted step side by side so the chained
/// attention reads them at a stride, the group's steps in order
pub struct Work {
    n: usize,
    taps: Tensor,
    g0: Tensor,
    e: Tensor,
    x0: Tensor,
    h1: Tensor,
    inv1: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    a: Tensor,
    x1: Tensor,
    h2: Tensor,
    inv2: Tensor,
    g: Tensor,
    u: Tensor,
    mm: Tensor,
    out: Tensor,
    tmp: Tensor,
    probs: Tensor,
    probs_stride: usize,
    z: Tensor,
    dz: Tensor,
    n_out: Tensor,
    inv_out: Tensor,
    dn: Tensor,
    dout: Tensor,
    dm: Tensor,
    dg: Tensor,
    du: Tensor,
    dh2: Tensor,
    dx1: Tensor,
    dx1n: Tensor,
    da: Tensor,
    dq: Tensor,
    dk: Tensor,
    dv: Tensor,
    dh1: Tensor,
    dx0: Tensor,
    dx0n: Tensor,
    dg_in: Tensor,
    de: Tensor,
    dg0: Tensor,
    dw_unused: Tensor,
    loss: Tensor,
    losses: Tensor,
    weights: Tensor,
    counted: CudaSlice<i32>,
    tokens: CudaSlice<i32>,
    idx: CudaSlice<i32>,
    mask_idx: CudaSlice<i32>,
    truth: CudaSlice<i32>,
    cands: Tensor,
    acc: Tensor,
    cost: Tensor,
    omega: Tensor,
    dvalue: Tensor,
    topk: Tensor,
    keys: Tensor,
    stats: Tensor,
}

impl Work {
    fn new(b: &CudaBackend, cfg: &DraftConfig, batch: usize, v: usize) -> Result<Self> {
        let (d, ff, h, k, td) = (cfg.d, cfg.ff, cfg.heads, cfg.depth, cfg.taps * cfg.d);
        let n = batch * ATTN_T;
        let steps = 2 * k;
        let t = |rows: usize, cols: usize| Tensor::zeros(b, rows, cols, "draft work");
        let probs_stride = batch * h * ATTN_T * (ATTN_T + k);
        let s = crate::exec::topk_slices(v);
        let fan = cfg.fanout[0];
        let mut mask_idx = Vec::with_capacity((k - 1) * n);
        for s in 1..k {
            mask_idx.extend(std::iter::repeat_n((s - 1) as i32, n));
        }
        Ok(Self {
            n,
            taps: t(n, td)?,
            g0: t(n, d)?,
            e: t(steps * n, d)?,
            x0: t(steps * n, d)?,
            h1: t(steps * n, d)?,
            inv1: t(steps, n)?,
            q: t(steps * n, d)?,
            k: t(steps * n, d)?,
            v: t(steps * n, d)?,
            a: t(steps * n, d)?,
            x1: t(steps * n, d)?,
            h2: t(steps * n, d)?,
            inv2: t(steps, n)?,
            g: t(steps * n, ff)?,
            u: t(steps * n, ff)?,
            mm: t(steps * n, ff)?,
            out: t(steps * n, d)?,
            tmp: t(n, d.max(ff))?,
            probs: t(steps, probs_stride)?,
            probs_stride,
            z: t(steps * n, v)?,
            dz: t(steps * n, v)?,
            n_out: t(n, d)?,
            inv_out: t(steps, n)?,
            dn: t(n, d)?,
            dout: t(steps * n, d)?,
            dm: t(n, ff)?,
            dg: t(n, ff)?,
            du: t(n, ff)?,
            dh2: t(n, d)?,
            dx1: t(n, d)?,
            dx1n: t(n, d)?,
            da: t(steps * n, d)?,
            dq: t(steps * n, d)?,
            dk: t(steps * n, d)?,
            dv: t(steps * n, d)?,
            dh1: t(n, d)?,
            dx0: t(n, d)?,
            dx0n: t(n, d)?,
            dg_in: t(n, d)?,
            de: t(n, d)?,
            dg0: t(n, d)?,
            dw_unused: t(1, d)?,
            loss: t(1, 1)?,
            losses: t(steps, n)?,
            weights: t(steps, n)?,
            counted: crate::exec::dev_alloc::<i32>(&b.stream, b.pool(), 1, AllocKind::Compute, "draft counted")?,
            tokens: crate::exec::dev_alloc::<i32>(&b.stream, b.pool(), 2 * n, AllocKind::Compute, "draft tokens")?,
            idx: crate::exec::dev_alloc::<i32>(&b.stream, b.pool(), (1 + steps) * n, AllocKind::Compute, "draft index")?,
            mask_idx: crate::exec::dev_upload(&b.stream, b.pool(), &mask_idx, AllocKind::Compute, "draft mask index")?,
            truth: crate::exec::dev_alloc::<i32>(&b.stream, b.pool(), k * n, AllocKind::Compute, "draft truth")?,
            cands: t(steps * n, CAND_W)?,
            acc: t(k, n)?,
            cost: t(k, n)?,
            omega: t(k, n)?,
            dvalue: t(k, n)?,
            topk: t(n, 2 * fan + 1)?,
            keys: t(n * s * fan * 2, 1)?,
            stats: t(n * s * 2, 1)?,
        })
    }

    /// The pointers of the rows of group g at step s, the branch group second
    fn rows(&self, cfg: &DraftConfig, g: usize, s: usize) -> Rows {
        let (d, ff) = (self.taps.cols / cfg.taps, self.g.cols);
        let step = g * cfg.depth + s;
        let (sd, sf, sn) = ((step * self.n * d * 4) as u64, (step * self.n * ff * 4) as u64, (step * self.n * 4) as u64);
        Rows {
            x0: self.x0.ptr() + sd,
            h1: self.h1.ptr() + sd,
            inv1: self.inv1.ptr() + sn,
            q: self.q.ptr() + sd,
            k: self.k.ptr() + sd,
            v: self.v.ptr() + sd,
            a: self.a.ptr() + sd,
            x1: self.x1.ptr() + sd,
            h2: self.h2.ptr() + sd,
            inv2: self.inv2.ptr() + sn,
            g: self.g.ptr() + sf,
            u: self.u.ptr() + sf,
            mm: self.mm.ptr() + sf,
            out: self.out.ptr() + sd,
            tmp: self.tmp.ptr(),
            dout: self.dout.ptr() + sd,
            dm: self.dm.ptr(),
            dg: self.dg.ptr(),
            du: self.du.ptr(),
            dh2: self.dh2.ptr(),
            dx1: self.dx1.ptr(),
            dx1n: self.dx1n.ptr(),
            da: self.da.ptr() + sd,
            dq: self.dq.ptr() + sd,
            dk: self.dk.ptr() + sd,
            dv: self.dv.ptr() + sd,
            dh1: self.dh1.ptr(),
            dx0: self.dx0.ptr(),
            dx0n: self.dx0n.ptr(),
            dg_in: self.dg_in.ptr(),
            de: self.de.ptr(),
        }
    }

    /// Offset of the per row values of group g at step s in a [steps, n] tensor, in floats
    fn at(&self, cfg: &DraftConfig, g: usize, s: usize) -> usize {
        (g * cfg.depth + s) * self.n
    }
}

/// The model's head over the draft vocabulary, native rows for drafting, single precision rows for training, the
/// final norm the outputs pass through, and the lookup from token id to draft column
pub struct Vocab {
    pub ids: Vec<u32>,
    native: CudaSlice<u8>,
    rows: Tensor,
    lut: CudaSlice<i32>,
    norm: Tensor,
}

/// The rows the passes scored, every tap row side by side with the packed candidates, kept on the device for
/// training, the shared bookkeeping deciding what trains, and a sentinel candidate row past the ring for targets
/// that do not exist
pub struct Ring {
    taps: Tensor,
    cands: Tensor,
    pub meta: RingMeta,
}

impl Ring {
    fn new(b: &CudaBackend, cap: usize, td: usize, n_seq: usize, seed: u64) -> Result<Self> {
        let cands = Tensor::zeros(b, cap + 1, CAND_W, "draft ring candidates")?;
        let mut sentinel = vec![0.0f32; CAND_W];
        sentinel[CAND_W - 1] = f32::INFINITY;
        crate::exec::htod(&b.stream, cands.ptr() + (cap * CAND_W * 4) as u64, &sentinel)?;
        Ok(Self { taps: Tensor::zeros(b, cap, td, "draft ring taps")?, cands, meta: RingMeta::new(cap, n_seq, seed) })
    }

    /// Appends rows whose tap rows stand contiguous per tap in taps, d columns each, and whose candidate rows
    /// stand contiguous at cands, row i standing for rows[i] with its slot, position, token, path flag, and parent,
    /// returning every row's ring index
    #[allow(clippy::too_many_arguments)]
    fn append(&mut self, b: &mut CudaBackend, taps: &[u64], d: usize, cands: u64, rows: &[(u32, u32, u32, bool, i32)]) -> Result<Vec<u32>> {
        let n = rows.len();
        let cap = self.meta.cap;
        ensure!(n <= cap, "{n} rows into a ring of {cap}");
        let td = self.taps.cols;
        let mut out = Vec::with_capacity(n);
        let mut at = 0;
        while at < n {
            let head = self.meta.append(rows[at].0, rows[at].1, rows[at].2, rows[at].3, rows[at].4) as usize;
            out.push(head as u32);
            let span = (n - at).min(cap - head);
            for (l, src) in taps.iter().enumerate() {
                train::copy_cols(b, src + (at * d * 4) as u64, self.taps.ptr() + (head * td * 4) as u64, span, d, td, l * d)?;
            }
            copy(b, self.cands.ptr() + (head * CAND_W * 4) as u64, cands + (at * CAND_W * 4) as u64, span * CAND_W * 4)?;
            for i in 1..span {
                out.push(self.meta.append(rows[at + i].0, rows[at + i].1, rows[at + i].2, rows[at + i].3, rows[at + i].4));
            }
            at += span;
        }
        Ok(out)
    }

    /// The candidate row index of a target, the sentinel past the ring for none
    fn target(&self, i: i32) -> i32 {
        if i < 0 { self.meta.cap as i32 } else { i }
    }
}

/// One slot's history of step zero keys and values by position, a ring twice the window deep, with the rows of
/// the last pass over the slot
struct Slot {
    kring: Tensor,
    vring: Tensor,
    hpos: CudaSlice<i32>,
    cap: usize,
    next_pos: Option<u32>,
    /// The rows and positions of one write, staged here
    rows_dev: CudaSlice<i32>,
    pos_dev: CudaSlice<i32>,
    /// The tap rows of the last pass's rows, side by side
    last_taps: Tensor,
    /// Position and token of every row of the last pass, and each row's ring index
    nodes: Vec<(u32, u32)>,
    ring: Vec<u32>,
    /// The ring index of the slot's last row on the model's own path
    last_main: i32,
}

impl Slot {
    fn new(b: &CudaBackend, cfg: &DraftConfig) -> Result<Self> {
        let cap = 2 * cfg.window;
        let hpos = crate::exec::dev_upload(&b.stream, b.pool(), &vec![-1i32; cap], AllocKind::Compute, "draft history positions")?;
        let stage = |what: &str| crate::exec::dev_alloc::<i32>(&b.stream, b.pool(), cap, AllocKind::Compute, what);
        Ok(Self { kring: Tensor::zeros(b, cap, cfg.d, "draft history keys")?, vring: Tensor::zeros(b, cap, cfg.d, "draft history values")?, hpos, cap, next_pos: None, rows_dev: stage("draft history rows")?, pos_dev: stage("draft history write")?, last_taps: Tensor::zeros(b, DRAFT_WORK_ROWS, cfg.taps * cfg.d, "draft last taps")?, nodes: Vec::new(), ring: Vec::new(), last_main: -1 })
    }

    fn clear(&mut self, b: &CudaBackend) -> Result<()> {
        crate::exec::htod(&b.stream, dptr(&self.hpos, &b.stream), &vec![-1i32; self.cap])?;
        self.next_pos = None;
        self.nodes.clear();
        self.ring.clear();
        self.last_main = -1;
        Ok(())
    }

    /// Writes rows of keys and values at consecutive positions, the history starting over when they do not follow,
    /// only the last cap of them since the ring holds no more and two rows a ring apart would race for a slot
    fn write(&mut self, b: &mut CudaBackend, k: u64, v: u64, rows: &[i32], pos: &[u32], d: usize) -> Result<()> {
        if rows.is_empty() {
            return Ok(());
        }
        ensure!(rows.len() == pos.len() && pos.windows(2).all(|w| w[0] + 1 == w[1]), "history rows out of order");
        if self.next_pos != Some(pos[0]) {
            crate::exec::htod(&b.stream, dptr(&self.hpos, &b.stream), &vec![-1i32; self.cap])?;
        }
        let skip = rows.len().saturating_sub(self.cap);
        let (rows, pos) = (&rows[skip..], &pos[skip..]);
        let pos_i: Vec<i32> = pos.iter().map(|p| *p as i32).collect();
        let (rows_ptr, pos_ptr) = (dptr(&self.rows_dev, &b.stream), dptr(&self.pos_dev, &b.stream));
        crate::exec::htod(&b.stream, rows_ptr, rows)?;
        crate::exec::htod(&b.stream, pos_ptr, &pos_i)?;
        train::ring_write(b, k, v, rows_ptr, pos_ptr, rows.len(), self.cap, d, self.kring.ptr(), self.vring.ptr(), dptr(&self.hpos, &b.stream))?;
        self.next_pos = Some(pos[pos.len() - 1] + 1);
        Ok(())
    }
}

/// Space the drafting and prefill passes work in, every row's inputs and layer rows by row index, the ancestor
/// lists, and the scorer
struct DraftWork {
    taps: Tensor,
    g: Tensor,
    e: Tensor,
    rows: Rows,
    _bufs: Vec<Tensor>,
    anc: CudaSlice<i32>,
    pos: CudaSlice<i32>,
    tokens: CudaSlice<i32>,
    nodes: CudaSlice<i32>,
    normed: Tensor,
    inv: Tensor,
    act: Tensor,
    logits: Tensor,
    topk: Tensor,
    keys: Tensor,
    stats: Tensor,
}

impl DraftWork {
    fn new(b: &CudaBackend, cfg: &DraftConfig, v: usize) -> Result<Self> {
        let (d, ff, td) = (cfg.d, cfg.ff, cfg.taps * cfg.d);
        let n = DRAFT_WORK_ROWS;
        let t = |rows: usize, cols: usize| Tensor::zeros(b, rows, cols, "draft rows work");
        let bufs = vec![t(n, d)?, t(n, d)?, t(1, n)?, t(n, d)?, t(n, d)?, t(n, d)?, t(n, d)?, t(n, d)?, t(n, d)?, t(1, n)?, t(n, ff)?, t(n, ff)?, t(n, ff)?, t(n, d)?, t(n, d.max(ff))?];
        let p = |i: usize| bufs[i].ptr();
        let rows = Rows { x0: p(0), h1: p(1), inv1: p(2), q: p(3), k: p(4), v: p(5), a: p(6), x1: p(7), h2: p(8), inv2: p(9), g: p(10), u: p(11), mm: p(12), out: p(13), tmp: p(14), dout: 0, dm: 0, dg: 0, du: 0, dh2: 0, dx1: 0, dx1n: 0, da: 0, dq: 0, dk: 0, dv: 0, dh1: 0, dx0: 0, dx0n: 0, dg_in: 0, de: 0 };
        let fan = cfg.fanout.iter().copied().max().unwrap_or(1);
        let s = crate::exec::topk_slices(v);
        let alloc = |n: usize, what: &str| crate::exec::dev_alloc::<i32>(&b.stream, b.pool(), n, AllocKind::Compute, what);
        Ok(Self {
            taps: t(n, td)?,
            g: t(n, d)?,
            e: t(n, d)?,
            rows,
            _bufs: bufs,
            anc: alloc(n * train::DRAFT_ANCESTORS, "draft ancestors")?,
            pos: alloc(n, "draft positions")?,
            tokens: alloc(n, "draft tokens")?,
            nodes: alloc(n, "draft nodes")?,
            normed: t(DRAFT_DEPTH, d)?,
            inv: t(1, DRAFT_DEPTH)?,
            act: t(DRAFT_DEPTH, d)?,
            logits: t(DRAFT_DEPTH, v)?,
            topk: t(DRAFT_DEPTH, 2 * fan + 1)?,
            keys: t(DRAFT_DEPTH * s * fan * 2, 1)?,
            stats: t(DRAFT_DEPTH * s * 2, 1)?,
        })
    }
}

/// Layer statistics the tap selection collects on the device, block averaged tap rows of every layer over the
/// rows of highest predictive entropy, see kf_backend_cpu::draft::Calibration
pub struct Calibration {
    pub n_layers: usize,
    d: usize,
    z: Option<Tensor>,
    counts: Vec<u64>,
    pub retained: u64,
}

impl Calibration {
    pub fn new(n_layers: usize) -> Self {
        Self { n_layers, d: 0, z: None, counts: vec![0; DRAFT_CAL_BLOCK], retained: 0 }
    }

    /// Takes a pass's rows, the rows of highest predictive entropy, the kept share of the pass, join the block
    /// averages in order
    pub fn observe(&mut self, b: &mut CudaBackend, taps: &[(u64, usize)], logits: u64, rows: usize, n_vocab: usize) -> Result<()> {
        if rows == 0 {
            return Ok(());
        }
        ensure!(taps.len() == self.n_layers, "{} tap tensors for {} layers", taps.len(), self.n_layers);
        if self.z.is_none() {
            self.d = taps[0].1;
            self.z = Some(Tensor::zeros(b, self.n_layers * DRAFT_CAL_BLOCK, self.d, "tap statistics")?);
        }
        let z = self.z.as_ref().ok_or_else(|| anyhow!("tap statistics unallocated"))?.ptr();
        let ent = Tensor::zeros(b, 1, rows, "row entropies")?;
        train::row_entropy(b, logits, n_vocab, ent.ptr(), rows)?;
        let ent = ent.download(b)?;
        let keep = ((rows as f32 * DRAFT_CAL_KEEP).ceil() as usize).clamp(1, rows);
        let mut order: Vec<(f32, usize)> = ent.iter().copied().zip(0..rows).collect();
        order.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        let mut chosen: Vec<i32> = order.iter().take(keep).map(|(_, r)| *r as i32).collect();
        chosen.sort_unstable();
        let idx = crate::exec::dev_upload(&b.stream, b.pool(), &chosen, AllocKind::Compute, "kept rows")?;
        let off = self.retained as usize % DRAFT_CAL_BLOCK;
        for (l, (src, d)) in taps.iter().enumerate() {
            ensure!(*d == self.d, "a tap of width {d} among taps of {}", self.d);
            train::cal_accum(b, *src, dptr(&idx, &b.stream), chosen.len(), off, DRAFT_CAL_BLOCK, z + (l * DRAFT_CAL_BLOCK * self.d * 4) as u64, self.d)?;
        }
        for k in 0..chosen.len() {
            self.counts[(off + k) % DRAFT_CAL_BLOCK] += 1;
        }
        self.retained += chosen.len() as u64;
        Ok(())
    }

    /// The m layers the statistics select, in rising order
    pub fn select(&self, b: &CudaBackend, m: usize, cka: bool) -> Result<Vec<usize>> {
        let z = self.z.as_ref().ok_or_else(|| anyhow!("no tap statistics were collected"))?.download(b)?;
        let (bl, d) = (DRAFT_CAL_BLOCK, self.d);
        let grams: Vec<Vec<Vec<f64>>> = (0..self.n_layers)
            .map(|l| {
                let row = |i: usize| -> Vec<f64> {
                    let c = self.counts[i].max(1) as f64;
                    z[(l * bl + i) * d..(l * bl + i + 1) * d].iter().map(|v| f64::from(*v) / c).collect()
                };
                let rows: Vec<Vec<f64>> = (0..bl).map(row).collect();
                (0..bl).map(|i| (0..bl).map(|j| rows[i].iter().zip(&rows[j]).map(|(a, b)| a * b).sum()).collect()).collect()
            })
            .collect();
        let mut taps = kf_backend_cpu::train::select_taps(&grams, m, kf_backend_cpu::draft::TAP_BEAM, kf_backend_cpu::draft::TAP_EXPAND, cka);
        taps.sort_unstable();
        Ok(taps)
    }
}

/// The block ensemble at work on the device, the learners, the hedge's mixture and the reference, the snapshot the
/// drafts read, the ring, every slot's history, the vocabulary, and the work spaces
pub struct Drafter {
    pub cfg: DraftConfig,
    pub tap_layers: Vec<usize>,
    pub learners: Vec<Block>,
    pub cum: Vec<f64>,
    pub weights: Vec<f32>,
    mixture: Vec<Tensor>,
    reference: Vec<Tensor>,
    snapshot: Snapshot,
    pub ring: Ring,
    pub vocab: Vocab,
    slots: Vec<Slot>,
    work: Work,
    dwork: DraftWork,
    batch: usize,
    draft_vocab: usize,
    n_vocab: usize,
    eos: Option<u32>,
    embed: Wid,
    embed_ty: u32,
    embed_rb: i64,
    head: Wid,
    head_ty: u32,
    head_rb: usize,
    head_f32: Vec<f32>,
    pub steps: usize,
}

/// What one learner's step found before its update
struct Learned {
    loss: f32,
    ce: f32,
    hit: f32,
}

impl Drafter {
    /// A block ensemble for a model whose embedding, head, and final norm weights are given, the head also kept
    /// dequantized on the host for the training logits
    #[allow(clippy::too_many_arguments)]
    pub fn new(b: &mut CudaBackend, cfg: DraftConfig, tap_layers: Vec<usize>, batch: usize, n_seq: usize, embed: Wid, head: Wid, norm: Wid, n_vocab: usize, eos: Option<u32>, draft_vocab: usize, weights: &dyn WeightSource) -> Result<Self> {
        cfg.check()?;
        ensure!(tap_layers.len() == cfg.taps && tap_layers.windows(2).all(|w| w[0] < w[1]), "tap layers {tap_layers:?} for a block of {} taps", cfg.taps);
        ensure!(crate::exec::int8_path(&b.kernels, Q8_0, 1, cfg.d), "the learned block drafts through the int8 product, which this device does not run");
        crate::exec::ensure_weight(b, embed, weights)?;
        crate::exec::ensure_weight(b, head, weights)?;
        let d = cfg.d;
        ensure!(weights.dims(embed).first().copied() == Some(d) && weights.dims(head).first().copied() == Some(d), "embedding and head widths against a block of {d}");
        let embed_ty = weights.ggml_type(embed);
        let embed_rb = row_bytes(embed_ty, d)? as i64;
        let head_ty = weights.ggml_type(head);
        let head_rb = row_bytes(head_ty, d)?;
        let head_f32 = kf_backend_cpu::dequant_all(head, weights)?;
        ensure!(head_f32.len() == n_vocab * d, "a head of {} values for {n_vocab} tokens of {d}", head_f32.len());
        let norm_vals = kf_backend_cpu::dequant_all(norm, weights)?;
        ensure!(norm_vals.len() == d, "a final norm of {} values for a block of {d}", norm_vals.len());
        let draft_vocab = draft_vocab.clamp(1, n_vocab);
        let learners: Vec<Block> = (0..cfg.learners).map(|i| Block::new(b, cfg, cfg.learner_lr(i))).collect::<Result<_>>()?;
        let shapes = cfg.param_shapes();
        let clone_of = |b: &CudaBackend, src: &[Tensor], what: &str| -> Result<Vec<Tensor>> {
            shapes
                .iter()
                .zip(src)
                .map(|((_, r, c), t)| {
                    let out = Tensor::zeros(b, *r, *c, what)?;
                    copy(b, out.ptr(), t.ptr(), r * c * 4)?;
                    Ok(out)
                })
                .collect()
        };
        let mixture = clone_of(b, &learners[0].params, "draft mixture")?;
        let reference = clone_of(b, &learners[0].params, "draft reference")?;
        let mut snapshot = Snapshot::new(b, &cfg)?;
        snapshot.refresh(b, &mixture)?;
        let ring = Ring::new(b, kf_core::DRAFT_RING_ROWS, cfg.taps * d, n_seq, cfg.seed)?;
        let slots = (0..n_seq.max(1)).map(|_| Slot::new(b, &cfg)).collect::<Result<_>>()?;
        let norm_t = Tensor::upload(b, &norm_vals, 1, d, "draft final norm")?;
        let mut s = Self {
            cfg,
            tap_layers,
            cum: vec![0.0; cfg.learners],
            weights: vec![1.0 / cfg.learners as f32; cfg.learners],
            learners,
            mixture,
            reference,
            snapshot,
            ring,
            vocab: Vocab { ids: Vec::new(), native: crate::exec::dev_alloc::<u8>(&b.stream, b.pool(), 1, AllocKind::Weights, "draft head rows")?, rows: Tensor::zeros(b, 1, 1, "draft head f32")?, lut: crate::exec::dev_alloc::<i32>(&b.stream, b.pool(), n_vocab, AllocKind::Compute, "draft lookup")?, norm: norm_t },
            slots,
            work: Work::new(b, &cfg, batch.max(1), draft_vocab)?,
            dwork: DraftWork::new(b, &cfg, draft_vocab)?,
            batch: batch.max(1),
            draft_vocab,
            n_vocab,
            eos,
            embed,
            embed_ty,
            embed_rb,
            head,
            head_ty,
            head_rb,
            head_f32,
            steps: 0,
        };
        let ids: Vec<u32> = (0..draft_vocab as u32).collect();
        s.set_vocab(b, &ids)?;
        Ok(s)
    }

    /// Sets the draft vocabulary, gathering its head rows natively for drafting and in single precision for training
    pub fn set_vocab(&mut self, b: &mut CudaBackend, ids: &[u32]) -> Result<()> {
        let d = self.cfg.d;
        ensure!(!ids.is_empty() && ids.len() <= self.draft_vocab && ids.windows(2).all(|w| w[0] < w[1]) && ids.iter().all(|t| (*t as usize) < self.n_vocab), "a draft vocabulary of {} ids for {} of {}", ids.len(), self.draft_vocab, self.n_vocab);
        let v = ids.len();
        let idx: Vec<i32> = ids.iter().map(|t| *t as i32).collect();
        let idx_dev = crate::exec::dev_upload(&b.stream, b.pool(), &idx, AllocKind::Compute, "vocab index")?;
        let native = crate::exec::dev_alloc::<u8>(&b.stream, b.pool(), v * self.head_rb, AllocKind::Weights, "draft head rows")?;
        let src = b.res.weight(self.head)?;
        train::gather_bytes(b, src, dptr(&idx_dev, &b.stream), dptr(&native, &b.stream), v, self.head_rb)?;
        let f32_rows: Vec<f32> = ids.iter().flat_map(|t| self.head_f32[*t as usize * d..(*t as usize + 1) * d].iter().copied()).collect();
        let rows = Tensor::upload(b, &f32_rows, v, d, "draft head f32")?;
        let mut lut = vec![-1i32; self.n_vocab];
        for (c, t) in ids.iter().enumerate() {
            lut[*t as usize] = c as i32;
        }
        crate::exec::htod(&b.stream, dptr(&self.vocab.lut, &b.stream), &lut)?;
        if v != self.work.z.cols {
            self.work = Work::new(b, &self.cfg, self.batch, v)?;
            self.dwork = DraftWork::new(b, &self.cfg, v)?;
        }
        self.vocab.ids = ids.to_vec();
        self.vocab.native = native;
        self.vocab.rows = rows;
        Ok(())
    }

    /// Embeddings of m tokens into dst through the model's embedding table
    fn embed(&self, b: &mut CudaBackend, tokens: u64, dst: u64, m: usize) -> Result<()> {
        let d = self.cfg.d;
        let w = b.res.weight(self.embed)?;
        flat(b.kernels.quant_function(self.embed_ty, "embed")?, d.div_ceil(8) * m, vec![Arg::P(w), Arg::P(tokens), Arg::P(dst), Arg::I(d as i32), Arg::I(m as i32), Arg::L(self.embed_rb)]).run(&b.stream)
    }

    /// Column of a token in the draft vocabulary
    fn col(&self, token: u32) -> i32 {
        self.vocab.ids.binary_search(&token).map_or(-1, |c| c as i32)
    }

    /// The forward of one query group through the given matrices, every step's rows from the fused features and
    /// the step's embedding rows, through the attention over the main group's window, to the logits over the draft
    /// vocabulary
    fn forward(&self, b: &mut CudaBackend, mats: Mats<'_>, group: usize) -> Result<()> {
        let cfg = self.cfg;
        let (d, h, hd, k, v) = (cfg.d, cfg.heads, cfg.head_dim(), cfg.depth, self.vocab.ids.len());
        let n = self.work.n;
        let stride = n * d;
        let scale = 1.0 / (hd as f32).sqrt();
        let tok_ptr = dptr(&self.work.tokens, &b.stream) + (group * n * 4) as u64;
        let mask_ptr = mats.small(MASK, &self.mixture);
        let win = self.work.rows(&cfg, 0, 0);
        let first = if group == 0 { 1 } else { 0 };
        for s in 0..k {
            let w = self.work.rows(&cfg, group, s);
            let e = self.work.e.ptr() + ((group * k + s) * stride * 4) as u64;
            if s == 0 {
                self.embed(b, tok_ptr, e, n)?;
            } else {
                train::gather_rows(b, mask_ptr, dptr(&self.work.mask_idx, &b.stream) + ((s - 1) * n * 4) as u64, e, n, d)?;
            }
            layer_in(b, mats, &self.mixture, &cfg, &w, self.work.g0.ptr(), e, n)?;
            let base = self.work.q.ptr() + (group * k * stride * 4) as u64;
            let (kb, vb) = (self.work.k.ptr() + (group * k * stride * 4) as u64, self.work.v.ptr() + (group * k * stride * 4) as u64);
            let probs = self.work.probs.ptr() + ((group * k + s) * self.work.probs_stride * 4) as u64;
            let out = self.work.a.ptr() + (group * k * stride * 4) as u64;
            train::chain_attn_fwd(b, base, kb, vb, stride, s, win.k, win.v, first, group == 0, out, probs, self.batch, h, hd, cfg.window, scale)?;
            layer_out(b, mats, &self.mixture, &cfg, &w, n)?;
            let inv = self.work.inv_out.ptr() + (self.work.at(&cfg, group, s) * 4) as u64;
            train::rms_fwd(b, w.out, self.vocab.norm.ptr(), self.work.n_out.ptr(), inv, n, d, cfg.eps)?;
            let z = self.work.z.ptr() + (self.work.at(&cfg, group, s) * v * 4) as u64;
            train::gemm(b, self.work.n_out.ptr(), self.vocab.rows.ptr(), z, n, v, d, d, d, v, false, true, 1.0, 0.0)?;
        }
        Ok(())
    }

    /// The cross entropy of every row of every step of both groups against its candidates, the gradient left
    /// zero, returning the losses as [2 depth, n]
    fn ce_values(&self, b: &mut CudaBackend) -> Result<Vec<f32>> {
        let cfg = self.cfg;
        let (k, v, n) = (cfg.depth, self.vocab.ids.len(), self.work.n);
        self.work.weights.zero(b)?;
        self.work.loss.zero(b)?;
        crate::exec::htod(&b.stream, dptr(&self.work.counted, &b.stream), &[0i32])?;
        for step in 0..2 * k {
            let at = step * n;
            train::soft_ce(b, self.work.z.ptr() + (at * v * 4) as u64, v, self.work.cands.ptr() + (at * CAND_W * 4) as u64, dptr(&self.vocab.lut, &b.stream), self.work.weights.ptr() + (at * 4) as u64, self.work.dz.ptr() + (at * v * 4) as u64, self.work.losses.ptr() + (at * 4) as u64, self.work.loss.ptr(), dptr(&self.work.counted, &b.stream), n)?;
        }
        self.work.losses.download(b)
    }

    /// The backward of one group given every step's gradient over the logits in dz, the window key gradients of
    /// the branch group added into the main group's step zero accumulators, dg summed into dg0
    fn backward(&mut self, b: &mut CudaBackend, li: usize, group: usize) -> Result<()> {
        let cfg = self.cfg;
        let (d, h, hd, k, v) = (cfg.d, cfg.heads, cfg.head_dim(), cfg.depth, self.vocab.ids.len());
        let n = self.work.n;
        let stride = n * d;
        let scale = 1.0 / (hd as f32).sqrt();
        let win = self.work.rows(&cfg, 0, 0);
        let first = if group == 0 { 1 } else { 0 };
        for s in (0..k).rev() {
            let w = self.work.rows(&cfg, group, s);
            let at = self.work.at(&cfg, group, s);
            let dz = self.work.dz.ptr() + (at * v * 4) as u64;
            train::gemm(b, dz, self.vocab.rows.ptr(), self.work.dn.ptr(), n, d, v, v, d, d, false, false, 1.0, 0.0)?;
            let inv = self.work.inv_out.ptr() + (at * 4) as u64;
            train::rms_bwd(b, w.out, self.vocab.norm.ptr(), inv, self.work.dn.ptr(), w.dout, self.work.dw_unused.ptr(), n, d)?;
            self.learners[li].layer_out_bwd(b, &w, n)?;
            let base = self.work.q.ptr() + (group * k * stride * 4) as u64;
            let (kb, vb) = (self.work.k.ptr() + (group * k * stride * 4) as u64, self.work.v.ptr() + (group * k * stride * 4) as u64);
            let (dqb, dkb, dvb) = (self.work.dq.ptr() + (group * k * stride * 4) as u64, self.work.dk.ptr() + (group * k * stride * 4) as u64, self.work.dv.ptr() + (group * k * stride * 4) as u64);
            let probs = self.work.probs.ptr() + ((group * k + s) * self.work.probs_stride * 4) as u64;
            let (dkwin, dvwin) = (self.work.dk.ptr(), self.work.dv.ptr());
            let dab = self.work.da.ptr() + (group * k * stride * 4) as u64;
            train::chain_attn_bwd(b, base, kb, vb, stride, s, win.k, win.v, first, probs, dab, dqb, dkb, dvb, dkwin, dvwin, self.batch, h, hd, scale)?;
            let e = self.work.e.ptr() + ((group * k + s) * stride * 4) as u64;
            self.learners[li].layer_in_bwd(b, &w, self.work.g0.ptr(), e, n)?;
            train::axpy(b, w.dg_in, self.work.dg0.ptr(), n * d, 1.0)?;
            if s > 0 {
                train::col_sum(b, w.de, n, d, self.learners[li].grads[MASK].ptr() + ((s - 1) * d * 4) as u64)?;
            }
        }
        Ok(())
    }

    /// The loss and gradients of one learner over the staged chunks, the gradients left in it
    fn learn(&mut self, b: &mut CudaBackend, li: usize, ref_ce: Option<&[f32]>, batch: usize) -> Result<Learned> {
        let cfg = self.cfg;
        let (k, v, n) = (cfg.depth, self.vocab.ids.len(), self.work.n);
        let t = ATTN_T;
        let steps = 2 * k;
        // The learner's forward over both groups, then its cross entropies, minus one where a row has no target
        fuse(b, Mats::Exact(&self.learners[li].params), &cfg, self.work.taps.ptr(), self.work.g0.ptr(), n)?;
        self.forward(b, Mats::Exact(&self.learners[li].params), 0)?;
        self.forward(b, Mats::Exact(&self.learners[li].params), 1)?;
        let ce = self.ce_values(b)?;
        let present: Vec<bool> = ce.iter().map(|c| *c >= 0.0).collect();
        // The kept share of the targets, every target when the share is whole, else the widest gaps over the reference
        let threshold = ref_ce.map(|rc| {
            let mut gaps: Vec<f32> = (0..steps * n).filter(|i| present[*i]).map(|i| ce[i] - rc[i]).collect();
            let keep = ((gaps.len() as f32 * cfg.keep_frac).ceil() as usize).clamp(1, gaps.len().max(1));
            gaps.sort_by(|a, b| b.total_cmp(a));
            gaps.get(keep - 1).copied().unwrap_or(f32::NEG_INFINITY)
        });
        let mut mask = vec![false; steps * n];
        let mut kept = 0usize;
        for i in 0..steps * n {
            if present[i] && threshold.is_none_or(|th| ce[i] - ref_ce.map_or(0.0, |rc| rc[i]) >= th) {
                mask[i] = true;
                kept += 1;
            }
        }
        ensure!(kept > 0, "no row of the batch has a target in the draft vocabulary");
        let inv = 1.0 / kept as f32;
        let mut w: Vec<f32> = mask.iter().map(|m| if *m { inv } else { 0.0 }).collect();
        // Rows of the main group under the objective, the expected rounds weigh the chunk, the block verification the row
        match cfg.objective {
            Objective::Ce => {}
            Objective::Edr => {
                for wi in w.iter_mut().take(k * n) {
                    *wi *= kept as f32 / batch as f32;
                }
            }
            Objective::Bv => {
                for (i, wi) in w.iter_mut().enumerate().take(k * n) {
                    *wi = if i < n { *wi * kept as f32 / n as f32 } else { 0.0 };
                }
            }
        }
        self.work.weights.set(b, &w)?;
        self.work.loss.zero(b)?;
        crate::exec::htod(&b.stream, dptr(&self.work.counted, &b.stream), &[0i32])?;
        let lut = dptr(&self.vocab.lut, &b.stream);
        let eos_col = self.eos.map_or(-1, |e| self.col(e));
        let step_of = |step: usize| -> (u64, u64, u64, u64, u64) {
            let at = step * n;
            (self.work.z.ptr() + (at * v * 4) as u64, self.work.cands.ptr() + (at * CAND_W * 4) as u64, self.work.weights.ptr() + (at * 4) as u64, self.work.dz.ptr() + (at * v * 4) as u64, self.work.losses.ptr() + (at * 4) as u64)
        };
        let mut loss = 0.0f64;
        for step in k..steps {
            let (z, c, wp, dz, l) = step_of(step);
            train::soft_ce(b, z, v, c, lut, wp, dz, l, self.work.loss.ptr(), dptr(&self.work.counted, &b.stream), n)?;
        }
        match cfg.objective {
            Objective::Ce => {
                for step in 0..k {
                    let (z, c, wp, dz, l) = step_of(step);
                    train::soft_ce(b, z, v, c, lut, wp, dz, l, self.work.loss.ptr(), dptr(&self.work.counted, &b.stream), n)?;
                }
                loss += f64::from(self.work.loss.download(b)?[0]);
            }
            Objective::Edr => {
                loss += f64::from(self.work.loss.download(b)?[0]);
                for step in 0..k {
                    let (z, c, _, _, _) = step_of(step);
                    train::edr_terms(b, z, v, c, lut, eos_col, self.work.acc.ptr() + (step * n * 4) as u64, self.work.cost.ptr() + (step * n * 4) as u64, n)?;
                }
                let (acc, cost) = (self.work.acc.download(b)?, self.work.cost.download(b)?);
                let (mut omega, mut dvalue) = (vec![0.0f32; k * n], vec![0.0f32; k * n]);
                for c in 0..batch {
                    let a: Vec<Vec<f32>> = (0..t).map(|i| (0..k).map(|s| acc[s * n + c * t + i]).collect()).collect();
                    let cc: Vec<Vec<f32>> = (0..t).map(|i| (0..k).map(|s| cost[s * n + c * t + i]).collect()).collect();
                    let (om, dv, rounds) = kf_backend_cpu::train::edr_recursion(&a, &cc, t, k);
                    loss += f64::from(rounds) / batch as f64;
                    for i in 0..t {
                        for s in 0..k {
                            omega[s * n + c * t + i] = om[i][s];
                            dvalue[s * n + c * t + i] = dv[i][s];
                        }
                    }
                }
                self.work.omega.set(b, &omega)?;
                self.work.dvalue.set(b, &dvalue)?;
                for step in 0..k {
                    let (z, c, wp, dz, _) = step_of(step);
                    unsafe { cudarc::driver::result::memset_d8_async(dz, 0, n * v * 4, b.stream.cu_stream()) }?;
                    train::edr_grad(b, z, v, c, lut, eos_col, self.work.omega.ptr() + (step * n * 4) as u64, self.work.dvalue.ptr() + (step * n * 4) as u64, wp, dz, n)?;
                }
            }
            Objective::Bv => {
                loss += f64::from(self.work.loss.download(b)?[0]);
                let (z, c, wp, dz, l) = step_of(0);
                unsafe { cudarc::driver::result::memset_d8_async(dz, 0, k * n * v * 4, b.stream.cu_stream()) }?;
                train::bv_row(b, z, v, n * v, c, n * CAND_W, dptr(&self.work.truth, &b.stream), n, lut, k, wp, dz, l, n)?;
                let rows = self.work.losses.download(b)?;
                loss += rows[..n].iter().map(|x| f64::from(*x)).sum::<f64>() / n as f64;
            }
        }
        let (mut ce_sum, mut ce_n) = (0.0f64, 0usize);
        for i in 0..k * n {
            if present[i] {
                ce_sum += f64::from(ce[i]);
                ce_n += 1;
            }
        }
        // The hit rate, how many first positions name the realized next token among their first fanout children
        let fan = cfg.fanout[0];
        for l in crate::exec::topk_launches(&mut b.kernels, "sample", self.work.z.ptr(), self.work.topk.ptr(), self.work.keys.ptr(), self.work.stats.ptr(), n, v, fan)? {
            l.run(&b.stream)?;
        }
        let packed = self.work.topk.download(b)?;
        let mut truth0 = vec![0i32; n];
        crate::exec::dtoh(&b.stream, &mut truth0, dptr(&self.work.truth, &b.stream))?;
        let width = 2 * fan + 1;
        let hits = (0..n).filter(|r| truth0[*r] >= 0 && (0..fan).any(|i| packed[r * width + i] as i32 == truth0[*r])).count();
        self.work.dk.zero(b)?;
        self.work.dv.zero(b)?;
        self.work.dg0.zero(b)?;
        self.backward(b, li, 1)?;
        self.backward(b, li, 0)?;
        let (taps, dg0) = (self.work.taps.ptr(), self.work.dg0.ptr());
        self.learners[li].fuse_bwd(b, taps, dg0, n)?;
        Ok(Learned { loss: loss as f32, ce: (ce_sum / ce_n.max(1) as f64) as f32, hit: hits as f32 / n as f32 })
    }

    /// Stages the chunks' rows, the tap rows, both groups' first tokens, every step's candidate rows, and the
    /// realized next tokens as draft columns
    fn stage(&self, b: &mut CudaBackend, chunks: &[kf_core::ChunkIdx]) -> Result<()> {
        let cfg = self.cfg;
        let (k, td, n, t) = (cfg.depth, cfg.taps * cfg.d, self.work.n, ATTN_T);
        let batch = chunks.len();
        ensure!(batch * t == n, "{batch} chunks for a work space of {n} rows");
        let mut rows: Vec<i32> = Vec::with_capacity(n);
        let mut tokens: Vec<i32> = Vec::with_capacity(2 * n);
        let mut targets: Vec<i32> = Vec::with_capacity(2 * k * n);
        let mut truth: Vec<i32> = Vec::with_capacity(k * n);
        for c in chunks {
            rows.extend(c.rows.iter().map(|r| *r as i32));
        }
        for c in chunks {
            tokens.extend(c.main_tokens.iter().map(|t| *t as i32));
        }
        for c in chunks {
            tokens.extend(c.branch_tokens.iter().map(|t| if *t < 0 { 0 } else { *t }));
        }
        for group in 0..2 {
            for s in 0..k {
                for (ci, c) in chunks.iter().enumerate() {
                    let src = if group == 0 { &c.main_targets[s] } else { &c.branch_targets[s] };
                    ensure!(src.len() == t, "chunk {ci} holds {} targets at step {s} for {t} rows", src.len());
                    targets.extend(src.iter().map(|idx| self.ring.target(*idx)));
                }
            }
        }
        for s in 0..k {
            for c in chunks {
                truth.extend(c.main_truth[s].iter().map(|tok| if *tok < 0 { -1 } else { self.col(*tok as u32) }));
            }
        }
        let idx_ptr = dptr(&self.work.idx, &b.stream);
        crate::exec::htod(&b.stream, idx_ptr, &rows)?;
        crate::exec::htod(&b.stream, idx_ptr + (n * 4) as u64, &targets)?;
        crate::exec::htod(&b.stream, dptr(&self.work.tokens, &b.stream), &tokens)?;
        crate::exec::htod(&b.stream, dptr(&self.work.truth, &b.stream), &truth)?;
        train::gather_rows(b, self.ring.taps.ptr(), idx_ptr, self.work.taps.ptr(), n, td)?;
        for step in 0..2 * k {
            train::gather_rows(b, self.ring.cands.ptr(), idx_ptr + ((1 + step) * n * 4) as u64, self.work.cands.ptr() + (step * n * CAND_W * 4) as u64, n, CAND_W)?;
        }
        Ok(())
    }

    /// One training step on the ring, every learner steps on the same chunks, the hedge reweighs them, and the
    /// mixture, reference, and snapshot follow, none while the ring holds no run of the chunk's rows
    pub fn step(&mut self, b: &mut CudaBackend) -> Result<Option<TrainStep>> {
        let cfg = self.cfg;
        let Some(chunks) = self.ring.meta.chunks(self.batch, ATTN_T, cfg.depth) else { return Ok(None) };
        self.stage(b, &chunks)?;
        let n = self.work.n;
        let ref_ce = if cfg.keep_frac < 1.0 {
            fuse(b, Mats::Exact(&self.reference), &cfg, self.work.taps.ptr(), self.work.g0.ptr(), n)?;
            self.forward(b, Mats::Exact(&self.reference), 0)?;
            self.forward(b, Mats::Exact(&self.reference), 1)?;
            Some(self.ce_values(b)?)
        } else {
            None
        };
        let mut report = TrainStep { loss: 0.0, hit: 0.0 };
        for li in 0..self.learners.len() {
            let l = self.learn(b, li, ref_ce.as_deref(), chunks.len())?;
            self.cum[li] += f64::from(l.ce);
            report.loss += self.weights[li] * l.loss;
            report.hit += self.weights[li] * l.hit;
            self.learners[li].adam(b)?;
        }
        self.rehedge(b)?;
        self.steps += 1;
        Ok(Some(report))
    }

    /// The gradients of the first learner over one batch of the ring, left zero after, none while no run is long
    /// enough, what parity tests compare across backends
    pub fn probe_grads(&mut self, b: &mut CudaBackend) -> Result<Option<Vec<Vec<f32>>>> {
        let cfg = self.cfg;
        let Some(chunks) = self.ring.meta.chunks(self.batch, ATTN_T, cfg.depth) else { return Ok(None) };
        self.stage(b, &chunks)?;
        let n = self.work.n;
        let ref_ce = if cfg.keep_frac < 1.0 {
            fuse(b, Mats::Exact(&self.reference), &cfg, self.work.taps.ptr(), self.work.g0.ptr(), n)?;
            self.forward(b, Mats::Exact(&self.reference), 0)?;
            self.forward(b, Mats::Exact(&self.reference), 1)?;
            Some(self.ce_values(b)?)
        } else {
            None
        };
        self.learn(b, 0, ref_ce.as_deref(), chunks.len())?;
        let grads = self.learners[0].grads_host(b)?;
        for g in &self.learners[0].grads {
            g.zero(b)?;
        }
        Ok(Some(grads))
    }

    /// Reweighs the learners by their running losses and rebuilds the mixture, the reference, and the snapshot
    fn rehedge(&mut self, b: &mut CudaBackend) -> Result<()> {
        let best = self.cum.iter().copied().fold(f64::INFINITY, f64::min);
        let raw: Vec<f64> = self.cum.iter().map(|c| (-f64::from(self.cfg.hedge) * (c - best)).exp()).collect();
        let total: f64 = raw.iter().sum();
        self.weights = raw.iter().map(|r| (r / total) as f32).collect();
        for pi in 0..self.mixture.len() {
            let srcs: Vec<u64> = self.learners.iter().map(|l| l.params[pi].ptr()).collect();
            let n = self.mixture[pi].len();
            train::mix4(b, self.mixture[pi].ptr(), &srcs, &self.weights, n)?;
            train::ema(b, self.reference[pi].ptr(), self.mixture[pi].ptr(), n, self.cfg.ema)?;
        }
        self.snapshot.refresh(b, &self.mixture)
    }

    /// The logits of m output rows at out over the draft vocabulary into the drafting work's logits, the rows
    /// through the model's final norm, then the int8 product when the head's type takes it and the type's own
    /// product otherwise
    fn logits(&self, b: &mut CudaBackend, out: u64, m: usize) -> Result<()> {
        use Arg::{I, L, P};
        let (d, v, ty, rb) = (self.cfg.d, self.vocab.ids.len(), self.head_ty, self.head_rb as i64);
        ensure!(m <= DRAFT_DEPTH, "{m} rows to score in space for {DRAFT_DEPTH}");
        let (act, logits, pred) = (self.dwork.act.ptr(), self.dwork.logits.ptr(), self.dwork.normed.ptr());
        train::rms_fwd(b, out, self.vocab.norm.ptr(), pred, self.dwork.inv.ptr(), m, d, self.cfg.eps)?;
        let w = dptr(&self.vocab.native, &b.stream);
        if crate::exec::int8_path(&b.kernels, ty, m, d) {
            let code = crate::exec::round_code(ty)?;
            let per = crate::exec::matvec_rows(ty, d, &[]);
            let kind = crate::exec::multi_kind(ty, d, &[]);
            for r0 in (0..m).step_by(per) {
                let r = (m - r0).min(per);
                crate::exec::quant_launch(&mut b.kernels, pred + (r0 * d * 4) as u64, act, r, d, code)?.run(&b.stream)?;
                crate::exec::matvec_i8_launch(&mut b.kernels, ty, w, act, 0, 0, logits + (r0 * v * 4) as u64, r, d, v, rb, 0, 0, kind)?.run(&b.stream)?;
            }
            return Ok(());
        }
        let x = match crate::exec::round_launch(&mut b.kernels, ty, pred, act, m, d)? {
            Some(l) => {
                l.run(&b.stream)?;
                act
            }
            None => pred,
        };
        let (entry, grid, block) = if m <= crate::exec::MATVEC_ROWS { ("matvec", (v.div_ceil(4) as u32, 1, 1), 128) } else { ("matmul", (v.div_ceil(64) as u32, m.div_ceil(32) as u32, 1), 256) };
        Launch { func: b.kernels.quant_function(ty, entry)?, grid, block, smem: 0, args: vec![P(w), P(x), P(0), P(0), P(logits), I(m as i32), I(d as i32), I(v as i32), L(rb)] }.run(&b.stream)
    }

    /// The k most likely draft tokens of each of m scored rows with their probabilities, best first
    fn top(&self, b: &mut CudaBackend, m: usize, k: usize) -> Result<Vec<Vec<(u32, f32)>>> {
        let v = self.vocab.ids.len();
        ensure!(k >= 1 && k <= v && 2 * k < self.dwork.topk.cols, "{k} children of a vocabulary of {v}");
        for l in crate::exec::topk_launches(&mut b.kernels, "sample", self.dwork.logits.ptr(), self.dwork.topk.ptr(), self.dwork.keys.ptr(), self.dwork.stats.ptr(), m, v, k)? {
            l.run(&b.stream)?;
        }
        let width = self.dwork.topk.cols;
        let mut packed = vec![0.0f32; m * width];
        crate::exec::dtoh(&b.stream, &mut packed, self.dwork.topk.ptr())?;
        Ok((0..m).map(|r| (0..k).map(|i| (self.vocab.ids[packed[r * width + i] as usize], (packed[r * width + k + i] - packed[r * width + 2 * k]).exp())).collect()).collect())
    }

    /// Runs the snapshot over the drafting rows r0 to r0 plus m whose g and e stand ready, each attending over the
    /// slot's history and the rows its ancestor list names, itself first
    fn run_rows(&self, b: &mut CudaBackend, seq: usize, r0: usize, m: usize, hist: bool) -> Result<()> {
        let cfg = self.cfg;
        let (d, h, hd) = (cfg.d, cfg.heads, cfg.head_dim());
        ensure!(r0 + m <= DRAFT_WORK_ROWS, "{} draft rows past {DRAFT_WORK_ROWS}", r0 + m);
        let w = rows_at(&self.dwork, r0, cfg.d, cfg.ff);
        let g = self.dwork.g.ptr() + (r0 * d * 4) as u64;
        let e = self.dwork.e.ptr() + (r0 * d * 4) as u64;
        layer_in(b, Mats::Snap(&self.snapshot), &self.mixture, &cfg, &w, g, e, m)?;
        let slot = &self.slots[seq];
        let cap = if hist { slot.cap } else { 0 };
        let anc = dptr(&self.dwork.anc, &b.stream) + (r0 * train::DRAFT_ANCESTORS * 4) as u64;
        let pos = dptr(&self.dwork.pos, &b.stream) + (r0 * 4) as u64;
        train::draft_attn(b, w.q, slot.kring.ptr(), slot.vring.ptr(), dptr(&slot.hpos, &b.stream), cap, self.dwork.rows.k, self.dwork.rows.v, anc, pos, cfg.window, w.a, m, h, hd, 1.0 / (hd as f32).sqrt())?;
        layer_out(b, Mats::Snap(&self.snapshot), &self.mixture, &cfg, &w, m)
    }

    /// Uploads the tokens, positions, and ancestor lists of drafting rows r0 to r0 plus m and embeds the tokens,
    /// rows past the tokens taking the mask rows in order
    fn stage_rows(&self, b: &mut CudaBackend, r0: usize, toks: &[u32], masks: usize, pos: &[u32], anc: &[Vec<i32>]) -> Result<()> {
        let d = self.cfg.d;
        const A: usize = train::DRAFT_ANCESTORS;
        let m = pos.len();
        ensure!(toks.len() + masks == m && anc.len() == m, "{} tokens and {masks} masks for {m} rows", toks.len());
        let mut anc_flat = vec![-1i32; m * A];
        for (j, a) in anc.iter().enumerate() {
            ensure!(a.len() <= A, "{} ancestors of a draft row past {A}", a.len());
            anc_flat[j * A..j * A + a.len()].copy_from_slice(a);
        }
        let toks_i: Vec<i32> = toks.iter().map(|t| *t as i32).collect();
        let pos_i: Vec<i32> = pos.iter().map(|p| *p as i32).collect();
        let tok_ptr = dptr(&self.dwork.tokens, &b.stream) + (r0 * 4) as u64;
        crate::exec::htod(&b.stream, tok_ptr, &toks_i)?;
        crate::exec::htod(&b.stream, dptr(&self.dwork.pos, &b.stream) + (r0 * 4) as u64, &pos_i)?;
        crate::exec::htod(&b.stream, dptr(&self.dwork.anc, &b.stream) + (r0 * A * 4) as u64, &anc_flat)?;
        let e = self.dwork.e.ptr() + (r0 * d * 4) as u64;
        self.embed(b, tok_ptr, e, toks.len())?;
        if masks > 0 {
            let idx: Vec<i32> = (0..masks as i32).collect();
            let idx_ptr = dptr(&self.dwork.nodes, &b.stream);
            crate::exec::htod(&b.stream, idx_ptr, &idx)?;
            train::gather_rows(b, self.mixture[MASK].ptr(), idx_ptr, e + (toks.len() * d * 4) as u64, masks, d)?;
        }
        Ok(())
    }

    /// Takes the rows of one slot's pass, their tap rows contiguous per tap and their candidate rows contiguous on
    /// the device, a chain joins the ring on the model's path and the history, a tree joins the ring below the
    /// slot's last row as branches until compaction puts its kept path on the model's path
    #[allow(clippy::too_many_arguments)]
    pub fn pass(&mut self, b: &mut CudaBackend, seq: usize, taps: &[u64], cands: u64, tokens: &[u32], pos: &[u32], parents: &[i32], tree: bool) -> Result<()> {
        let cfg = self.cfg;
        let (d, td) = (cfg.d, cfg.taps * cfg.d);
        let n = tokens.len();
        ensure!(seq < self.slots.len() && (1..=DRAFT_WORK_ROWS).contains(&n) && pos.len() == n && parents.len() == n && taps.len() == cfg.taps, "a pass of {n} rows with {} taps over slot {seq} of {}", taps.len(), self.slots.len());
        let last_main = self.slots[seq].last_main;
        for (l, src) in taps.iter().enumerate() {
            train::copy_cols(b, *src, self.slots[seq].last_taps.ptr(), n, d, td, l * d)?;
        }
        if tree {
            let mut depth = vec![0u32; n];
            for i in 1..n {
                ensure!(parents[i] >= 0 && (parents[i] as usize) < i, "a pass whose rows do not form a tree");
                depth[i] = depth[parents[i] as usize] + 1;
            }
            // Every row's parent among the ring's rows, the root below the slot's last main row, the rest below the
            // ring row their tree parent takes, which follows the next index in order
            let (base, cap) = (self.ring.meta.next_index() as usize, self.ring.meta.cap);
            let rows: Vec<(u32, u32, u32, bool, i32)> = (0..n).map(|i| (seq as u32, pos[0] + depth[i], tokens[i], false, if parents[i] < 0 { last_main } else { ((base + parents[i] as usize) % cap) as i32 })).collect();
            let ring_idx = self.ring.append(b, taps, d, cands, &rows)?;
            let slot = &mut self.slots[seq];
            slot.nodes = pos.iter().copied().zip(tokens.iter().copied()).collect();
            slot.ring = ring_idx;
            return Ok(());
        }
        if n > 1 {
            // The rows before the last join the history with the tokens that followed them, their keys and values
            // being all the history holds
            let m = n - 1;
            fuse(b, Mats::Snap(&self.snapshot), &cfg, self.slots[seq].last_taps.ptr(), self.dwork.g.ptr(), m)?;
            let anc: Vec<Vec<i32>> = vec![Vec::new(); m];
            self.stage_rows(b, 0, &tokens[1..n], 0, &pos[..m], &anc)?;
            let w = rows_at(&self.dwork, 0, cfg.d, cfg.ff);
            layer_in(b, Mats::Snap(&self.snapshot), &self.mixture, &cfg, &w, self.dwork.g.ptr(), self.dwork.e.ptr(), m)?;
            let rows_idx: Vec<i32> = (0..m as i32).collect();
            let (k, v) = (self.dwork.rows.k, self.dwork.rows.v);
            self.slots[seq].write(b, k, v, &rows_idx, &pos[..m], d)?;
        }
        let (base, cap) = (self.ring.meta.next_index() as usize, self.ring.meta.cap);
        let rows: Vec<(u32, u32, u32, bool, i32)> = (0..n).map(|i| (seq as u32, pos[i], tokens[i], true, if i == 0 { last_main } else { ((base + i - 1) % cap) as i32 })).collect();
        let ring_idx = self.ring.append(b, taps, d, cands, &rows)?;
        let slot = &mut self.slots[seq];
        slot.last_main = ring_idx[n - 1] as i32;
        slot.nodes = vec![(pos[n - 1], tokens[n - 1])];
        slot.ring = vec![ring_idx[n - 1]];
        // The last row's tap row moves to the front of the slot's last rows so the path indexes it as row zero
        if n > 1 {
            let (src, dst) = (slot.last_taps.ptr() + ((n - 1) * td * 4) as u64, slot.last_taps.ptr());
            copy(b, dst, src, td * 4)?;
        }
        Ok(())
    }

    /// Writes the accepted path's step zero rows into the slot's history and drafts a tree of up to rows below the
    /// root picked at its end, the path rows attending to the history and the rows before them, the mask rows to
    /// all of them
    pub fn draft_tree(&mut self, b: &mut CudaBackend, seq: usize, path: &[usize], root: u32, rows: usize) -> Result<Tree> {
        let cfg = self.cfg;
        let (d, td) = (cfg.d, cfg.taps * cfg.d);
        ensure!(seq < self.slots.len(), "slot {seq} of {}", self.slots.len());
        if self.slots[seq].nodes.is_empty() {
            return Ok(Tree::root(root));
        }
        let nodes = self.slots[seq].nodes.clone();
        ensure!(!path.is_empty() && path.iter().all(|i| *i < nodes.len()), "a path of {:?} over {} rows", path, nodes.len());
        let masks = if rows > 1 { cfg.depth - 1 } else { 0 };
        let m = path.len() + masks;
        ensure!(m <= DRAFT_WORK_ROWS, "{m} draft rows past {DRAFT_WORK_ROWS}");
        let leaf = path.len() - 1;
        let mut gather: Vec<i32> = path.iter().map(|i| *i as i32).collect();
        gather.extend(std::iter::repeat_n(path[leaf] as i32, masks));
        let idx_ptr = dptr(&self.dwork.nodes, &b.stream);
        crate::exec::htod(&b.stream, idx_ptr, &gather)?;
        train::gather_rows(b, self.slots[seq].last_taps.ptr(), idx_ptr, self.dwork.taps.ptr(), m, td)?;
        fuse(b, Mats::Snap(&self.snapshot), &cfg, self.dwork.taps.ptr(), self.dwork.g.ptr(), m)?;
        let base = nodes[path[0]].0;
        let toks: Vec<u32> = (0..path.len()).map(|a| if a + 1 < path.len() { nodes[path[a + 1]].1 } else { root }).collect();
        let mut pos: Vec<u32> = (0..path.len()).map(|a| base + a as u32).collect();
        pos.extend(std::iter::repeat_n(base + leaf as u32, masks));
        let mut anc: Vec<Vec<i32>> = (0..path.len()).map(|a| (0..=a as i32).collect()).collect();
        anc.extend((0..masks).map(|j| (0..=(path.len() + j) as i32).collect()));
        self.stage_rows(b, 0, &toks, masks, &pos, &anc)?;
        let hist = self.slots[seq].next_pos == Some(base);
        self.run_rows(b, seq, 0, m, hist)?;
        let rows_idx: Vec<i32> = (0..path.len() as i32).collect();
        let (k, v) = (self.dwork.rows.k, self.dwork.rows.v);
        self.slots[seq].write(b, k, v, &rows_idx, &pos[..path.len()], d)?;
        if masks == 0 {
            return Ok(Tree::root(root));
        }
        self.logits(b, self.dwork.rows.out + (leaf * d * 4) as u64, masks + 1)?;
        let fan = cfg.fanout.iter().take(masks + 1).copied().max().unwrap_or(1);
        let tops_all = self.top(b, masks + 1, fan)?;
        let tops: Vec<Vec<(u32, f32)>> = tops_all.into_iter().enumerate().map(|(j, t)| t.into_iter().take(cfg.fanout[j]).collect()).collect();
        Ok(tree_of_block(root, &tops, rows))
    }

    /// Puts the rows of the last tree pass that compaction kept on the model's path at the positions they now hold
    pub fn compacted(&mut self, seq: usize, base: usize, keep: &[usize]) {
        let Some(slot) = self.slots.get_mut(seq) else { return };
        for (i, k) in keep.iter().enumerate() {
            if let Some(idx) = slot.ring.get(*k) {
                self.ring.meta.promote(*idx, (base + i) as u32);
                slot.last_main = *idx as i32;
            }
        }
    }

    /// Ends a slot's runs and history after its rows rolled back
    pub fn roll_back(&mut self, b: &CudaBackend, seq: usize) -> Result<()> {
        self.ring.meta.roll_back(seq);
        match self.slots.get_mut(seq) {
            Some(s) => s.clear(b),
            None => Ok(()),
        }
    }

    pub fn refresh_vocab(&mut self, b: &mut CudaBackend) -> Result<()> {
        let ids = self.ring.meta.frequent(self.n_vocab, self.draft_vocab);
        self.set_vocab(b, &ids)
    }

    pub fn save(&self, b: &CudaBackend) -> Result<Vec<u8>> {
        let params: Vec<Vec<f32>> = self.mixture.iter().map(|t| t.download(b)).collect::<Result<_>>()?;
        Ok(kf_core::draft_save(&self.cfg, &self.tap_layers, &self.vocab.ids, &params))
    }

    /// Replaces every learner, the mixture, and the reference with saved parameters, the running losses starting over
    pub fn load(&mut self, b: &mut CudaBackend, bytes: &[u8]) -> Result<()> {
        let (taps, ids, params) = kf_core::draft_load(&self.cfg, bytes)?;
        ensure!(taps == self.tap_layers, "a saved block over layers {taps:?}, this block taps {:?}", self.tap_layers);
        for l in &mut self.learners {
            l.set_params(b, &params)?;
        }
        for (t, p) in self.mixture.iter_mut().zip(&params) {
            t.set(b, p)?;
        }
        for (t, p) in self.reference.iter_mut().zip(&params) {
            t.set(b, p)?;
        }
        self.cum.iter_mut().for_each(|c| *c = 0.0);
        self.weights = vec![1.0 / self.learners.len() as f32; self.learners.len()];
        self.snapshot.refresh(b, &self.mixture)?;
        self.set_vocab(b, &ids)
    }

    /// The hedge's mixture of the learners as host values, what the parity tests read
    pub fn mixture_host(&self, b: &CudaBackend) -> Result<Vec<Vec<f32>>> {
        self.mixture.iter().map(|t| t.download(b)).collect()
    }
}

/// The int8 snapshot's product of matrix which over m rows of x, built from the given parameters, what the parity
/// tests compare with the host reference's rows
#[doc(hidden)]
pub fn probe_product(b: &mut CudaBackend, cfg: &DraftConfig, params: &[Vec<f32>], which: usize, x: &[f32], m: usize) -> Result<Vec<f32>> {
    let shapes = cfg.param_shapes();
    let tensors: Vec<Tensor> = shapes.iter().zip(params).map(|((_, r, c), p)| Tensor::upload(b, p, *r, *c, "probe")).collect::<Result<_>>()?;
    let mut snap = Snapshot::new(b, cfg)?;
    snap.refresh(b, &tensors)?;
    let (_, k, n) = shapes[which];
    let xt = Tensor::upload(b, x, m, k, "probe x")?;
    let y = Tensor::zeros(b, m, n, "probe y")?;
    snap.product(b, which, xt.ptr(), m, y.ptr())?;
    y.download(b)
}

/// The drafting work's layer rows from row offset r0
fn rows_at(w: &DraftWork, r0: usize, d: usize, ff: usize) -> Rows {
    let (sd, sf, sn) = ((r0 * d * 4) as u64, (r0 * ff * 4) as u64, (r0 * 4) as u64);
    let r = w.rows;
    Rows { x0: r.x0 + sd, h1: r.h1 + sd, inv1: r.inv1 + sn, q: r.q + sd, k: r.k + sd, v: r.v + sd, a: r.a + sd, x1: r.x1 + sd, h2: r.h2 + sd, inv2: r.inv2 + sn, g: r.g + sf, u: r.u + sf, mm: r.mm + sf, out: r.out + sd, ..r }
}

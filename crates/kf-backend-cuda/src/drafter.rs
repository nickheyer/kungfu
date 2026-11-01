//! The draft head on the device, see kf_core::HeadConfig and the host reference in kf_backend_cpu::drafter
use crate::CudaBackend;
use crate::exec::{Arg, Launch, dev_upload, dptr, flat, row_bytes};
use crate::train::{self, ATTN_T, Tensor};
use anyhow::{Result, ensure};
use cudarc::driver::CudaSlice;
use kf_core::{AllocKind, HeadConfig, TrainStep, WeightSource, Wid};
use std::collections::HashMap;

const FC_F: usize = 0;
const FC_E: usize = 1;
const WQ: usize = 2;
const WK: usize = 3;
const WV: usize = 4;
const WO: usize = 5;
const N1: usize = 6;
const WG: usize = 7;
const WU: usize = 8;
const WD: usize = 9;
const N2: usize = 10;

/// A head with its parameters, gradients, and optimizer moments on the device
pub struct Head {
    pub cfg: HeadConfig,
    params: Vec<Tensor>,
    grads: Vec<Tensor>,
    m: Vec<Tensor>,
    v: Vec<Tensor>,
    pub step: i32,
}

/// The intermediates of one forward and backward pass over n rows
pub struct Work {
    pub n: usize,
    x0: Tensor,
    h1: Tensor,
    inv1: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    a: Tensor,
    probs: Tensor,
    x1: Tensor,
    h2: Tensor,
    inv2: Tensor,
    g: Tensor,
    u: Tensor,
    mm: Tensor,
    pub pred: Tensor,
    dx2: Tensor,
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
    loss: Tensor,
}

/// c = op(a) op(b) + beta c over row major operands as the host reference's mm, a few untransposed rows going through
/// the skinny product when parts names its scratch
#[allow(clippy::too_many_arguments)]
fn mm(b: &mut CudaBackend, a: u64, bm: u64, c: u64, m: usize, n: usize, k: usize, ta: bool, tb: bool, beta: f32, parts: u64) -> Result<()> {
    let (lda, ldb) = (if ta { m } else { k }, if tb { k } else { n });
    if parts != 0 && m <= train::SKINNY_ROWS && !ta && !tb {
        return train::gemm_skinny(b, a, bm, c, m, n, k, lda, ldb, n, parts, 1.0, beta);
    }
    train::gemm(b, a, bm, c, m, n, k, lda, ldb, n, ta, tb, 1.0, beta)
}

impl Head {
    /// A head with the reference's initial parameters
    pub fn new(b: &CudaBackend, cfg: HeadConfig) -> Result<Self> {
        ensure!(cfg.d.is_multiple_of(cfg.heads) && cfg.heads > 0, "a head width of {} over {} heads", cfg.d, cfg.heads);
        let shapes = cfg.param_shapes();
        let values = kf_backend_cpu::drafter::init_params(&cfg);
        let mut params = Vec::new();
        let (mut grads, mut m, mut v) = (Vec::new(), Vec::new(), Vec::new());
        for ((name, rows, cols), vals) in shapes.iter().zip(values) {
            params.push(Tensor::upload(b, &vals, *rows, *cols, name)?);
            grads.push(Tensor::zeros(b, *rows, *cols, name)?);
            m.push(Tensor::zeros(b, *rows, *cols, name)?);
            v.push(Tensor::zeros(b, *rows, *cols, name)?);
        }
        Ok(Self { cfg, params, grads, m, v, step: 0 })
    }

    /// Work space for batch chunks of ATTN_T rows
    pub fn work(&self, b: &CudaBackend, batch: usize) -> Result<Work> {
        let (d, ff, h) = (self.cfg.d, self.cfg.ff, self.cfg.heads);
        let n = batch * ATTN_T;
        let t = |rows: usize, cols: usize| Tensor::zeros(b, rows, cols, "head work");
        Ok(Work {
            n,
            x0: t(n, d)?,
            h1: t(n, d)?,
            inv1: t(1, n)?,
            q: t(n, d)?,
            k: t(n, d)?,
            v: t(n, d)?,
            a: t(n, d)?,
            probs: t(batch * h * ATTN_T, ATTN_T)?,
            x1: t(n, d)?,
            h2: t(n, d)?,
            inv2: t(1, n)?,
            g: t(n, ff)?,
            u: t(n, ff)?,
            mm: t(n, ff)?,
            pred: t(n, d)?,
            dx2: t(n, d)?,
            dm: t(n, ff)?,
            dg: t(n, ff)?,
            du: t(n, ff)?,
            dh2: t(n, d)?,
            dx1: t(n, d)?,
            dx1n: t(n, d)?,
            da: t(n, d)?,
            dq: t(n, d)?,
            dk: t(n, d)?,
            dv: t(n, d)?,
            dh1: t(n, d)?,
            dx0: t(n, d)?,
            dx0n: t(n, d)?,
            loss: t(1, 1)?,
        })
    }

    /// Predicts the next rows of the chunks from rows f and the next tokens' embeddings e, both [n, d] on the device
    pub fn forward(&self, b: &mut CudaBackend, w: &Work, f: u64, e: u64, batch: usize) -> Result<()> {
        let (d, ff, h) = (self.cfg.d, self.cfg.ff, self.cfg.heads);
        let n = w.n;
        ensure!(n == batch * ATTN_T, "work for {n} rows against {batch} chunks");
        let p: Vec<u64> = self.params.iter().map(|t| t.ptr()).collect();
        let scale = 1.0 / (self.cfg.head_dim() as f32).sqrt();
        mm(b, f, p[FC_F], w.x0.ptr(), n, d, d, false, false, 0.0, 0)?;
        mm(b, e, p[FC_E], w.x0.ptr(), n, d, d, false, false, 1.0, 0)?;
        train::rms_fwd(b, w.x0.ptr(), p[N1], w.h1.ptr(), w.inv1.ptr(), n, d, self.cfg.eps)?;
        mm(b, w.h1.ptr(), p[WQ], w.q.ptr(), n, d, d, false, false, 0.0, 0)?;
        mm(b, w.h1.ptr(), p[WK], w.k.ptr(), n, d, d, false, false, 0.0, 0)?;
        mm(b, w.h1.ptr(), p[WV], w.v.ptr(), n, d, d, false, false, 0.0, 0)?;
        train::head_attn_fwd(b, w.q.ptr(), w.k.ptr(), w.v.ptr(), w.a.ptr(), w.probs.ptr(), batch, h, d / h, self.cfg.window, scale)?;
        unsafe { cudarc::driver::result::memcpy_dtod_async(w.x1.ptr(), w.x0.ptr(), n * d * 4, b.stream.cu_stream()) }?;
        mm(b, w.a.ptr(), p[WO], w.x1.ptr(), n, d, d, false, false, 1.0, 0)?;
        train::rms_fwd(b, w.x1.ptr(), p[N2], w.h2.ptr(), w.inv2.ptr(), n, d, self.cfg.eps)?;
        mm(b, w.h2.ptr(), p[WG], w.g.ptr(), n, ff, d, false, false, 0.0, 0)?;
        mm(b, w.h2.ptr(), p[WU], w.u.ptr(), n, ff, d, false, false, 0.0, 0)?;
        train::silu_gate_fwd(b, w.g.ptr(), w.u.ptr(), w.mm.ptr(), n * ff)?;
        unsafe { cudarc::driver::result::memcpy_dtod_async(w.pred.ptr(), w.x1.ptr(), n * d * 4, b.stream.cu_stream()) }?;
        mm(b, w.mm.ptr(), p[WD], w.pred.ptr(), n, d, ff, false, false, 1.0, 0)
    }

    /// The mean smooth L1 loss against the target rows, the gradients of every parameter added into grads
    pub fn backward(&mut self, b: &mut CudaBackend, w: &Work, f: u64, e: u64, target: u64, batch: usize) -> Result<f32> {
        let (d, ff, h) = (self.cfg.d, self.cfg.ff, self.cfg.heads);
        let n = w.n;
        let p: Vec<u64> = self.params.iter().map(|t| t.ptr()).collect();
        let g: Vec<u64> = self.grads.iter().map(|t| t.ptr()).collect();
        let scale = 1.0 / (self.cfg.head_dim() as f32).sqrt();
        w.loss.zero(b)?;
        train::smooth_l1(b, w.pred.ptr(), target, w.dx2.ptr(), w.loss.ptr(), n * d, 1.0)?;
        mm(b, w.mm.ptr(), w.dx2.ptr(), g[WD], ff, d, n, true, false, 1.0, 0)?;
        mm(b, w.dx2.ptr(), p[WD], w.dm.ptr(), n, ff, d, false, true, 0.0, 0)?;
        train::silu_gate_bwd(b, w.g.ptr(), w.u.ptr(), w.dm.ptr(), w.dg.ptr(), w.du.ptr(), n * ff)?;
        mm(b, w.h2.ptr(), w.dg.ptr(), g[WG], d, ff, n, true, false, 1.0, 0)?;
        mm(b, w.h2.ptr(), w.du.ptr(), g[WU], d, ff, n, true, false, 1.0, 0)?;
        mm(b, w.dg.ptr(), p[WG], w.dh2.ptr(), n, d, ff, false, true, 0.0, 0)?;
        mm(b, w.du.ptr(), p[WU], w.dh2.ptr(), n, d, ff, false, true, 1.0, 0)?;
        train::rms_bwd(b, w.x1.ptr(), p[N2], w.inv2.ptr(), w.dh2.ptr(), w.dx1n.ptr(), g[N2], n, d)?;
        unsafe { cudarc::driver::result::memcpy_dtod_async(w.dx1.ptr(), w.dx2.ptr(), n * d * 4, b.stream.cu_stream()) }?;
        train::axpy(b, w.dx1n.ptr(), w.dx1.ptr(), n * d, 1.0)?;
        mm(b, w.a.ptr(), w.dx1.ptr(), g[WO], d, d, n, true, false, 1.0, 0)?;
        mm(b, w.dx1.ptr(), p[WO], w.da.ptr(), n, d, d, false, true, 0.0, 0)?;
        train::head_attn_bwd(b, w.q.ptr(), w.k.ptr(), w.v.ptr(), w.probs.ptr(), w.da.ptr(), w.dq.ptr(), w.dk.ptr(), w.dv.ptr(), batch, h, d / h, scale)?;
        mm(b, w.h1.ptr(), w.dq.ptr(), g[WQ], d, d, n, true, false, 1.0, 0)?;
        mm(b, w.h1.ptr(), w.dk.ptr(), g[WK], d, d, n, true, false, 1.0, 0)?;
        mm(b, w.h1.ptr(), w.dv.ptr(), g[WV], d, d, n, true, false, 1.0, 0)?;
        mm(b, w.dq.ptr(), p[WQ], w.dh1.ptr(), n, d, d, false, true, 0.0, 0)?;
        mm(b, w.dk.ptr(), p[WK], w.dh1.ptr(), n, d, d, false, true, 1.0, 0)?;
        mm(b, w.dv.ptr(), p[WV], w.dh1.ptr(), n, d, d, false, true, 1.0, 0)?;
        train::rms_bwd(b, w.x0.ptr(), p[N1], w.inv1.ptr(), w.dh1.ptr(), w.dx0n.ptr(), g[N1], n, d)?;
        unsafe { cudarc::driver::result::memcpy_dtod_async(w.dx0.ptr(), w.dx1.ptr(), n * d * 4, b.stream.cu_stream()) }?;
        train::axpy(b, w.dx0n.ptr(), w.dx0.ptr(), n * d, 1.0)?;
        mm(b, f, w.dx0.ptr(), g[FC_F], d, d, n, true, false, 1.0, 0)?;
        mm(b, e, w.dx0.ptr(), g[FC_E], d, d, n, true, false, 1.0, 0)?;
        Ok(w.loss.download(b)?[0])
    }

    /// One AdamW step over every parameter, the gradients cleared
    pub fn step(&mut self, b: &mut CudaBackend) -> Result<()> {
        self.step += 1;
        for i in 0..self.params.len() {
            let n = self.params[i].len();
            train::adamw(b, self.params[i].ptr(), self.grads[i].ptr(), self.m[i].ptr(), self.v[i].ptr(), n, self.cfg.lr, 0.9, 0.999, 1e-8, self.cfg.weight_decay, self.step)?;
            self.grads[i].zero(b)?;
        }
        Ok(())
    }

    /// Every parameter on the host, in the configuration's order
    pub fn params_host(&self, b: &CudaBackend) -> Result<Vec<Vec<f32>>> {
        self.params.iter().map(|t| t.download(b)).collect()
    }

    /// Every gradient on the host, in the configuration's order
    pub fn grads_host(&self, b: &CudaBackend) -> Result<Vec<Vec<f32>>> {
        self.grads.iter().map(|t| t.download(b)).collect()
    }

    /// Overwrites every parameter from host values in the configuration's order
    pub fn set_params(&mut self, b: &CudaBackend, values: &[Vec<f32>]) -> Result<()> {
        ensure!(values.len() == self.params.len(), "{} parameter tensors for a head of {}", values.len(), self.params.len());
        for (t, v) in self.params.iter_mut().zip(values) {
            t.set(b, v)?;
        }
        Ok(())
    }
}

/// Trains the head on the feature ring: a step gathers batch runs of ATTN_T plus two rows of one slot and epoch, the
/// first ATTN_T as the inputs, the next ATTN_T as the targets with their tokens' embeddings as the second input, and
/// the tokens after those as the truth the predictions are scored against
pub struct Trainer {
    pub head: Head,
    pub(crate) work: Work,
    pub batch: usize,
    embed: Wid,
    embed_ty: u32,
    embed_rb: i64,
    f: Tensor,
    e: Tensor,
    y: Tensor,
    idx: CudaSlice<i32>,
    tokens: CudaSlice<i32>,
    /// The token after each target row of the last batch, chunk by chunk
    pub(crate) truth: Vec<u32>,
    rng: u64,
    pub steps: usize,
    pub last_loss: f32,
}

impl Trainer {
    /// A trainer over the model's embedding table for the head's inputs
    pub fn new(b: &CudaBackend, cfg: HeadConfig, batch: usize, embed: Wid, weights: &dyn WeightSource) -> Result<Self> {
        ensure!(batch >= 1, "a batch of no chunks");
        let d = weights.dims(embed);
        ensure!(d.first().copied() == Some(cfg.d), "an embedding of width {:?} for a head of {}", d.first(), cfg.d);
        let embed_ty = weights.ggml_type(embed);
        let embed_rb = row_bytes(embed_ty, cfg.d)? as i64;
        let head = Head::new(b, cfg)?;
        let work = head.work(b, batch)?;
        let n = batch * ATTN_T;
        let idx = crate::exec::dev_alloc::<i32>(&b.stream, b.pool(), 2 * n, AllocKind::Compute, "trainer index")?;
        let tokens = crate::exec::dev_alloc::<i32>(&b.stream, b.pool(), n, AllocKind::Compute, "trainer tokens")?;
        Ok(Self { head, work, batch, embed, embed_ty, embed_rb, f: Tensor::zeros(b, n, cfg.d, "trainer f")?, e: Tensor::zeros(b, n, cfg.d, "trainer e")?, y: Tensor::zeros(b, n, cfg.d, "trainer y")?, idx, tokens, truth: Vec::new(), rng: cfg.seed.max(1), steps: 0, last_loss: f32::NAN })
    }

    fn next(&mut self) -> u64 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        self.rng
    }

    /// Chunk starts for one batch, each a run and an offset leaving ATTN_T plus two rows, none while the ring holds no run that long
    fn sample(&mut self, runs: &[Vec<u32>]) -> Option<Vec<(usize, usize)>> {
        let spans: Vec<(usize, usize)> = runs.iter().enumerate().filter(|(_, r)| r.len() > ATTN_T + 1).map(|(i, r)| (i, r.len() - ATTN_T - 1)).collect();
        let total: usize = spans.iter().map(|(_, n)| *n).sum();
        if total == 0 {
            return None;
        }
        let mut out = Vec::with_capacity(self.batch);
        for _ in 0..self.batch {
            let mut pick = (self.next() % total as u64) as usize;
            for (i, n) in &spans {
                if pick < *n {
                    out.push((*i, pick));
                    break;
                }
                pick -= n;
            }
        }
        Some(out)
    }

    /// One optimizer step on a batch from the ring, the loss before the update, none while the ring is too short
    pub fn step(&mut self, b: &mut CudaBackend) -> Result<Option<f32>> {
        let (runs, ring_ptr, tokens_of): (Vec<Vec<u32>>, u64, Vec<u32>) = match b.ring.as_ref() {
            Some(r) => (r.runs(), r.ptr(), (0..r.cap).map(|i| r.entry(i).token).collect()),
            None => return Ok(None),
        };
        let Some(picks) = self.sample(&runs) else { return Ok(None) };
        let n = self.batch * ATTN_T;
        let mut idx = Vec::with_capacity(2 * n);
        let mut tokens = Vec::with_capacity(n);
        let mut truth = Vec::with_capacity(n);
        for (run, at) in &picks {
            idx.extend(runs[*run][*at..*at + ATTN_T].iter().map(|i| *i as i32));
        }
        for (run, at) in &picks {
            for i in &runs[*run][*at + 1..*at + 1 + ATTN_T] {
                idx.push(*i as i32);
                tokens.push(tokens_of[*i as usize] as i32);
            }
            truth.extend(runs[*run][*at + 2..*at + 2 + ATTN_T].iter().map(|i| tokens_of[*i as usize]));
        }
        crate::exec::htod(&b.stream, dptr(&self.idx, &b.stream), &idx)?;
        crate::exec::htod(&b.stream, dptr(&self.tokens, &b.stream), &tokens)?;
        let d = self.head.cfg.d;
        let idx_ptr = dptr(&self.idx, &b.stream);
        train::gather_rows(b, ring_ptr, idx_ptr, self.f.ptr(), n, d)?;
        train::gather_rows(b, ring_ptr, idx_ptr + (n * 4) as u64, self.y.ptr(), n, d)?;
        let w = b.res.weight(self.embed)?;
        let tok_ptr = dptr(&self.tokens, &b.stream);
        flat(b.kernels.quant_function(self.embed_ty, "embed")?, d.div_ceil(8) * n, vec![Arg::P(w), Arg::P(tok_ptr), Arg::P(self.e.ptr()), Arg::I(d as i32), Arg::I(n as i32), Arg::L(self.embed_rb)]).run(&b.stream)?;
        self.head.forward(b, &self.work, self.f.ptr(), self.e.ptr(), self.batch)?;
        let loss = self.head.backward(b, &self.work, self.f.ptr(), self.e.ptr(), self.y.ptr(), self.batch)?;
        self.head.step(b)?;
        self.steps += 1;
        self.last_loss = loss;
        self.truth = truth;
        Ok(Some(loss))
    }
}

/// Uploads host values as a device slice, for small index and token arrays
pub fn upload_i32(b: &CudaBackend, values: &[i32], what: &str) -> Result<CudaSlice<i32>> {
    dev_upload(&b.stream, b.pool(), values, AllocKind::Compute, what)
}

/// Children per depth of a drafted tree, the first depth widest
pub const DRAFT_FANOUT: [usize; 5] = kf_backend_cpu::drafter::DRAFT_FANOUT;

/// The head at work: a trainer over the ring, per slot windows of the head's keys and values kept in step with the
/// ring, a held row per slot for the coming draft, and the draft vocabulary gathered from the model's head weight
pub struct Drafter {
    pub trainer: Trainer,
    n_seq: usize,
    /// Pairs a slot's window holds, one under the head's window so a row and its pairs make the window the head trained on
    window: usize,
    /// Head keys and values of each slot's last window pairs as rings, [n_seq, window, d]
    kwin: Tensor,
    vwin: Tensor,
    /// Pairs cached per slot, the position after the last cached pair, the position the window's ring counts from,
    /// and the ring epoch the window follows
    cached: Vec<usize>,
    next_pos: Vec<u32>,
    base_pos: Vec<u32>,
    epoch_seen: Vec<u32>,
    /// The hidden row each slot holds for its coming draft, with its position
    held: Tensor,
    held_pos: Vec<Option<u32>>,
    /// The draft vocabulary: token of every draft id, and the model's head rows of those tokens as native bytes
    pub vocab: Vec<u32>,
    vocab_dev: Option<CudaSlice<u8>>,
    head_w: Wid,
    head_ty: u32,
    head_rb: usize,
    rows: DraftWork,
    /// Scoring space for the rows of one depth and for the trainer's batch
    draft: Scorer,
    eval: Scorer,
    /// Partial sums of the skinny product
    parts: Tensor,
}

/// Device space one depth of drafting works in: rows of the depth, then buffers by tree node index
struct DraftWork {
    f: Tensor,
    e: Tensor,
    x0: Tensor,
    h1: Tensor,
    inv: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    a: Tensor,
    x1: Tensor,
    h2: Tensor,
    g: Tensor,
    u: Tensor,
    mm: Tensor,
    pred: Tensor,
    anc: CudaSlice<i32>,
    tokens: CudaSlice<i32>,
    knode: Tensor,
    vnode: Tensor,
    prednode: Tensor,
}

/// Space for scoring predicted rows against the draft vocabulary: the rounded or quantized activations, the logits,
/// and the top k with its scratch
struct Scorer {
    rows: usize,
    act: Tensor,
    logits: Tensor,
    topk: Tensor,
    keys: Tensor,
    stats: Tensor,
}

impl Scorer {
    fn new(b: &CudaBackend, rows: usize, d: usize, v: usize, k: usize) -> Result<Self> {
        let s = crate::exec::topk_slices(v);
        Ok(Self {
            rows,
            act: Tensor::zeros(b, rows, d, "scoring activations")?,
            logits: Tensor::zeros(b, rows, v, "draft logits")?,
            topk: Tensor::zeros(b, rows, 2 * k + 1, "draft top k")?,
            keys: Tensor::zeros(b, rows * s * k * 2, 1, "draft keys")?,
            stats: Tensor::zeros(b, rows * s * 2, 1, "draft stats")?,
        })
    }
}

impl Drafter {
    /// A drafter for a model whose embedding and head weights are given, both brought to the device
    pub fn new(b: &mut CudaBackend, cfg: HeadConfig, batch: usize, n_seq: usize, embed: Wid, head: Wid, weights: &dyn WeightSource) -> Result<Self> {
        crate::exec::ensure_weight(b, embed, weights)?;
        crate::exec::ensure_weight(b, head, weights)?;
        let trainer = Trainer::new(b, cfg, batch, embed, weights)?;
        let d = cfg.d;
        let hd = weights.dims(head);
        ensure!(hd.first().copied() == Some(d), "a head weight of width {:?} for rows of {d}", hd.first());
        let head_ty = weights.ggml_type(head);
        let head_rb = row_bytes(head_ty, d)?;
        let n = kf_core::TREE_ROWS;
        let t = |rows: usize, cols: usize| Tensor::zeros(b, rows, cols, "draft work");
        let rows = DraftWork {
            f: t(n, d)?,
            e: t(n, d)?,
            x0: t(n, d)?,
            h1: t(n, d)?,
            inv: t(1, n)?,
            q: t(n, d)?,
            k: t(n, d)?,
            v: t(n, d)?,
            a: t(n, d)?,
            x1: t(n, d)?,
            h2: t(n, d)?,
            g: t(n, cfg.ff)?,
            u: t(n, cfg.ff)?,
            mm: t(n, cfg.ff)?,
            pred: t(n, d)?,
            anc: crate::exec::dev_alloc::<i32>(&b.stream, b.pool(), n * train::DRAFT_ANCESTORS, AllocKind::Compute, "draft ancestors")?,
            tokens: crate::exec::dev_alloc::<i32>(&b.stream, b.pool(), n, AllocKind::Compute, "draft tokens")?,
            knode: t(n, d)?,
            vnode: t(n, d)?,
            prednode: t(n, d)?,
        };
        ensure!(cfg.window >= 2, "a head window of {} positions", cfg.window);
        let window = cfg.window - 1;
        let k = DRAFT_FANOUT[0];
        Ok(Self {
            draft: Scorer::new(b, n, d, 1, k)?,
            eval: Scorer::new(b, batch * ATTN_T, d, 1, k)?,
            parts: Tensor::zeros(b, train::SKINNY_SPLITS * train::SKINNY_ROWS, d.max(cfg.ff), "skinny parts")?,
            trainer,
            n_seq,
            window,
            kwin: Tensor::zeros(b, n_seq * window, d, "head key window")?,
            vwin: Tensor::zeros(b, n_seq * window, d, "head value window")?,
            cached: vec![0; n_seq],
            next_pos: vec![0; n_seq],
            base_pos: vec![0; n_seq],
            epoch_seen: vec![u32::MAX; n_seq],
            held: Tensor::zeros(b, n_seq, d, "held rows")?,
            held_pos: vec![None; n_seq],
            vocab: Vec::new(),
            vocab_dev: None,
            head_w: head,
            head_ty,
            head_rb,
            rows,
        })
    }

    /// Sets the draft vocabulary to these tokens, whose head rows are gathered on the device
    pub fn set_vocab(&mut self, b: &mut CudaBackend, tokens: &[u32]) -> Result<()> {
        ensure!(!tokens.is_empty(), "an empty draft vocabulary");
        let v = tokens.len();
        let ids: Vec<i32> = tokens.iter().map(|t| *t as i32).collect();
        let idx = upload_i32(b, &ids, "vocab index")?;
        let dst = crate::exec::dev_alloc::<u8>(&b.stream, b.pool(), v * self.head_rb, AllocKind::Weights, "draft head rows")?;
        let src = b.res.weight(self.head_w)?;
        train::gather_bytes(b, src, dptr(&idx, &b.stream), dptr(&dst, &b.stream), v, self.head_rb)?;
        let (d, k) = (self.trainer.head.cfg.d, DRAFT_FANOUT[0]);
        self.draft = Scorer::new(b, kf_core::TREE_ROWS, d, v, k)?;
        self.eval = Scorer::new(b, self.trainer.batch * ATTN_T, d, v, k)?;
        self.vocab = tokens.to_vec();
        self.vocab_dev = Some(dst);
        Ok(())
    }

    /// The most frequent tokens of the ring as a draft vocabulary of at most v tokens, in rising id order
    pub fn vocab_from_ring(b: &CudaBackend, n_vocab: usize, v: usize) -> Vec<u32> {
        let mut counts = vec![0u64; n_vocab];
        if let Some(r) = b.ring.as_ref() {
            for i in 0..r.cap {
                let e = r.entry(i);
                if e.live && (e.token as usize) < n_vocab {
                    counts[e.token as usize] += 1;
                }
            }
        }
        let mut ids: Vec<u32> = (0..n_vocab as u32).collect();
        ids.sort_by(|a, b| counts[*b as usize].cmp(&counts[*a as usize]).then(a.cmp(b)));
        ids.truncate(v.min(n_vocab));
        ids.sort_unstable();
        ids
    }

    /// Keeps a slot's hidden row of the last pass, at its position, for the coming draft
    pub fn hold(&mut self, b: &CudaBackend, seq: usize, src: u64, pos: u32) -> Result<()> {
        let d = self.trainer.head.cfg.d;
        ensure!(seq < self.n_seq, "slot {seq} of {}", self.n_seq);
        unsafe { cudarc::driver::result::memcpy_dtod_async(self.held.ptr() + (seq * d * 4) as u64, src, d * 4, b.stream.cu_stream()) }?;
        self.held_pos[seq] = Some(pos);
        Ok(())
    }

    /// Forgets a slot's cached window and held row, after its rows rolled back
    pub fn forget(&mut self, seq: usize) {
        if seq < self.n_seq {
            self.cached[seq] = 0;
            self.next_pos[seq] = 0;
            self.base_pos[seq] = 0;
            self.epoch_seen[seq] = u32::MAX;
            self.held_pos[seq] = None;
        }
    }

    /// Brings a slot's key and value window up to the ring: the window follows the slot's latest run of consecutive
    /// positions in its current epoch, starts over when a new run or epoch begins, and adds the keys and values of
    /// every pair of that run it does not hold yet
    pub fn sync(&mut self, b: &mut CudaBackend, seq: usize) -> Result<()> {
        let d = self.trainer.head.cfg.d;
        ensure!(seq < self.n_seq, "slot {seq} of {}", self.n_seq);
        let (pairs, ring_ptr) = {
            let Some(ring) = b.ring.as_ref() else { return Ok(()) };
            let epoch = ring.epoch(seq);
            let mut by_pos: HashMap<u32, (u32, u32)> = HashMap::new();
            for i in 0..ring.cap {
                let e = ring.entry(i);
                if e.live && e.seq as usize == seq && e.epoch == epoch {
                    by_pos.insert(e.pos, (i as u32, e.token));
                }
            }
            let Some(hi) = by_pos.keys().copied().max() else { return Ok(()) };
            let mut lo = hi;
            while lo > 0 && by_pos.contains_key(&(lo - 1)) {
                lo -= 1;
            }
            if self.epoch_seen[seq] != epoch || self.next_pos[seq] < lo || self.next_pos[seq] > hi + 1 {
                self.epoch_seen[seq] = epoch;
                self.base_pos[seq] = lo;
                self.next_pos[seq] = lo;
                self.cached[seq] = 0;
            }
            let pairs: Vec<(u32, u32, u32)> = (self.next_pos[seq]..hi).map(|p| (by_pos[&p].0, by_pos[&(p + 1)].1, p)).collect();
            (pairs, ring.ptr())
        };
        let base = self.base_pos[seq];
        for chunk in pairs.chunks(kf_core::TREE_ROWS) {
            let m = chunk.len();
            let idx = upload_i32(b, &chunk.iter().map(|(i, _, _)| *i as i32).collect::<Vec<i32>>(), "sync index")?;
            let toks = upload_i32(b, &chunk.iter().map(|(_, t, _)| *t as i32).collect::<Vec<i32>>(), "sync tokens")?;
            train::gather_rows(b, ring_ptr, dptr(&idx, &b.stream), self.rows.f.ptr(), m, d)?;
            self.embed(b, dptr(&toks, &b.stream), self.rows.e.ptr(), m)?;
            self.project_kv(b, m)?;
            for (j, (_, _, p)) in chunk.iter().enumerate() {
                let dst = ((seq * self.window + ((*p - base) as usize) % self.window) * d * 4) as u64;
                unsafe {
                    cudarc::driver::result::memcpy_dtod_async(self.kwin.ptr() + dst, self.rows.k.ptr() + (j * d * 4) as u64, d * 4, b.stream.cu_stream())?;
                    cudarc::driver::result::memcpy_dtod_async(self.vwin.ptr() + dst, self.rows.v.ptr() + (j * d * 4) as u64, d * 4, b.stream.cu_stream())?;
                }
                self.cached[seq] = (self.cached[seq] + 1).min(self.window);
                self.next_pos[seq] = p + 1;
            }
        }
        Ok(())
    }

    /// Embeddings of m tokens into dst through the model's embedding table
    fn embed(&self, b: &mut CudaBackend, tokens: u64, dst: u64, m: usize) -> Result<()> {
        let d = self.trainer.head.cfg.d;
        let w = b.res.weight(self.trainer.embed)?;
        flat(b.kernels.quant_function(self.trainer.embed_ty, "embed")?, d.div_ceil(8) * m, vec![Arg::P(w), Arg::P(tokens), Arg::P(dst), Arg::I(d as i32), Arg::I(m as i32), Arg::L(self.trainer.embed_rb)]).run(&b.stream)
    }

    /// The head up to its keys and values for the m rows of f and e in the work, into k and v, x0 and h1 kept
    fn project_kv(&mut self, b: &mut CudaBackend, m: usize) -> Result<()> {
        let d = self.trainer.head.cfg.d;
        let p: Vec<u64> = self.trainer.head.params.iter().map(|t| t.ptr()).collect();
        let (w, parts) = (&self.rows, self.parts.ptr());
        mm(b, w.f.ptr(), p[FC_F], w.x0.ptr(), m, d, d, false, false, 0.0, parts)?;
        mm(b, w.e.ptr(), p[FC_E], w.x0.ptr(), m, d, d, false, false, 1.0, parts)?;
        train::rms_fwd(b, w.x0.ptr(), p[N1], w.h1.ptr(), w.inv.ptr(), m, d, self.trainer.head.cfg.eps)?;
        mm(b, w.h1.ptr(), p[WK], w.k.ptr(), m, d, d, false, false, 0.0, parts)?;
        mm(b, w.h1.ptr(), p[WV], w.v.ptr(), m, d, d, false, false, 0.0, parts)
    }

    /// The rest of the head for m rows whose x0 and h1 stand in the work, attending over nwin rows of the slot's
    /// window and the node indexed keys and values the ancestor lists name, the predictions into pred
    fn finish_rows(&mut self, b: &mut CudaBackend, seq: usize, m: usize, nwin: usize) -> Result<()> {
        let cfg = self.trainer.head.cfg;
        let (d, ff, h) = (cfg.d, cfg.ff, cfg.heads);
        let p: Vec<u64> = self.trainer.head.params.iter().map(|t| t.ptr()).collect();
        let (w, parts) = (&self.rows, self.parts.ptr());
        mm(b, w.h1.ptr(), p[WQ], w.q.ptr(), m, d, d, false, false, 0.0, parts)?;
        let kwin = self.kwin.ptr() + (seq * self.window * d * 4) as u64;
        let vwin = self.vwin.ptr() + (seq * self.window * d * 4) as u64;
        train::head_attn_draft(b, w.q.ptr(), kwin, vwin, nwin, w.knode.ptr(), w.vnode.ptr(), dptr(&w.anc, &b.stream), w.a.ptr(), m, h, d / h, 1.0 / (cfg.head_dim() as f32).sqrt())?;
        unsafe { cudarc::driver::result::memcpy_dtod_async(w.x1.ptr(), w.x0.ptr(), m * d * 4, b.stream.cu_stream()) }?;
        mm(b, w.a.ptr(), p[WO], w.x1.ptr(), m, d, d, false, false, 1.0, parts)?;
        train::rms_fwd(b, w.x1.ptr(), p[N2], w.h2.ptr(), w.inv.ptr(), m, d, cfg.eps)?;
        mm(b, w.h2.ptr(), p[WG], w.g.ptr(), m, ff, d, false, false, 0.0, parts)?;
        mm(b, w.h2.ptr(), p[WU], w.u.ptr(), m, ff, d, false, false, 0.0, parts)?;
        train::silu_gate_fwd(b, w.g.ptr(), w.u.ptr(), w.mm.ptr(), m * ff)?;
        unsafe { cudarc::driver::result::memcpy_dtod_async(w.pred.ptr(), w.x1.ptr(), m * d * 4, b.stream.cu_stream()) }?;
        mm(b, w.mm.ptr(), p[WD], w.pred.ptr(), m, d, ff, false, false, 1.0, parts)
    }

    /// The logits of m predicted rows over the draft vocabulary into the scorer, through the int8 product when the
    /// head's type takes it and the type's own product otherwise, as the model's products run
    fn logits(&self, b: &mut CudaBackend, pred: u64, m: usize, sc: &Scorer) -> Result<()> {
        use Arg::{I, L, P};
        let (d, v, ty, rb) = (self.trainer.head.cfg.d, self.vocab.len(), self.head_ty, self.head_rb as i64);
        ensure!(m <= sc.rows, "{m} rows to score in space for {}", sc.rows);
        let w = dptr(self.vocab_dev.as_ref().ok_or_else(|| anyhow::anyhow!("no draft vocabulary"))?, &b.stream);
        if crate::exec::int8_path(&b.kernels, ty, m, d) {
            let code = crate::exec::round_code(ty)?;
            let per = crate::exec::matvec_rows(ty, d, &[]);
            let fits = crate::exec::mm_i8_fits(ty, d, &[]);
            for r0 in (0..m).step_by(per) {
                let r = (m - r0).min(per);
                crate::exec::quant_launch(&mut b.kernels, pred + (r0 * d * 4) as u64, sc.act.ptr(), r, d, code)?.run(&b.stream)?;
                crate::exec::matvec_i8_launch(&mut b.kernels, ty, w, sc.act.ptr(), 0, 0, sc.logits.ptr() + (r0 * v * 4) as u64, r, d, v, rb, 0, 0, fits && r > 1)?.run(&b.stream)?;
            }
            return Ok(());
        }
        let x = match crate::exec::round_launch(&mut b.kernels, ty, pred, sc.act.ptr(), m, d)? {
            Some(l) => {
                l.run(&b.stream)?;
                sc.act.ptr()
            }
            None => pred,
        };
        let (entry, grid, block) = if m <= crate::exec::MATVEC_ROWS { ("matvec", (v.div_ceil(4) as u32, 1, 1), 128) } else { ("matmul", (v.div_ceil(64) as u32, m.div_ceil(32) as u32, 1), 256) };
        Launch { func: b.kernels.quant_function(ty, entry)?, grid, block, smem: 0, args: vec![P(w), P(x), P(0), P(0), P(sc.logits.ptr()), I(m as i32), I(d as i32), I(v as i32), L(rb)] }.run(&b.stream)
    }

    /// The k most likely draft vocabulary tokens of each of m predicted rows under the model's head, with their logits
    fn score(&self, b: &mut CudaBackend, pred: u64, m: usize, sc: &Scorer, k: usize) -> Result<Vec<Vec<(u32, f32)>>> {
        let v = self.vocab.len();
        ensure!(k >= 1 && k <= DRAFT_FANOUT[0] && k <= v, "{k} children of a vocabulary of {v}");
        self.logits(b, pred, m, sc)?;
        for l in crate::exec::topk_launches(&mut b.kernels, "sample", sc.logits.ptr(), sc.topk.ptr(), sc.keys.ptr(), sc.stats.ptr(), m, v, k)? {
            l.run(&b.stream)?;
        }
        let packed = sc.topk.download(b)?;
        let width = 2 * k + 1;
        Ok((0..m).map(|r| (0..k).map(|i| (self.vocab[packed[r * width + i] as usize], packed[r * width + k + i])).collect()).collect())
    }

    /// One training step on the ring, the loss before the update and the share of the batch's target rows whose
    /// following token stood among the children the first depth drafts, none while the ring is too short
    pub fn step(&mut self, b: &mut CudaBackend) -> Result<Option<TrainStep>> {
        let Some(loss) = self.trainer.step(b)? else { return Ok(None) };
        let n = self.trainer.batch * ATTN_T;
        let drafted = self.score(b, self.trainer.work.pred.ptr(), n, &self.eval, DRAFT_FANOUT[0])?;
        let hits = drafted.iter().zip(&self.trainer.truth).filter(|(c, t)| c.iter().any(|(tok, _)| tok == *t)).count();
        Ok(Some(TrainStep { loss, hit: hits as f32 / n as f32 }))
    }

    /// The prediction the last draft made for its root, the next hidden row after the held row and own
    pub fn root_prediction(&self, b: &CudaBackend) -> Result<Vec<f32>> {
        let d = self.trainer.head.cfg.d;
        let mut out = vec![0.0f32; d];
        crate::exec::dtoh(&b.stream, &mut out, self.rows.prednode.ptr())?;
        Ok(out)
    }

    /// The native bytes of the draft vocabulary's head rows, in draft id order
    pub fn vocab_bytes(&self, b: &CudaBackend) -> Result<Vec<u8>> {
        let dev = self.vocab_dev.as_ref().ok_or_else(|| anyhow::anyhow!("no draft vocabulary"))?;
        let mut out = vec![0u8; self.vocab.len() * self.head_rb];
        crate::exec::dtoh(&b.stream, &mut out, dptr(dev, &b.stream))?;
        Ok(out)
    }

    /// The draft logits of the last depth drafted, rows by draft vocabulary id
    pub fn last_logits(&self, b: &CudaBackend, rows: usize) -> Result<Vec<f32>> {
        let v = self.vocab.len();
        let mut out = vec![0.0f32; rows * v];
        crate::exec::dtoh(&b.stream, &mut out, self.draft.logits.ptr())?;
        Ok(out)
    }

    /// Pairs cached in a slot's window
    pub fn cached(&self, seq: usize) -> usize {
        self.cached.get(seq).copied().unwrap_or(0)
    }

    /// Whether a slot holds a row to draft from and the vocabulary is set
    pub fn ready(&self, seq: usize) -> bool {
        seq < self.n_seq && self.held_pos[seq].is_some() && self.vocab_dev.is_some()
    }

    /// A tree rooted at own, the token chosen after the held row, grown depth by depth with DRAFT_FANOUT children of
    /// every row, at most budget rows in all, the window read only while its pairs run up to the held row
    pub fn draft(&mut self, b: &mut CudaBackend, seq: usize, own: u32, budget: usize) -> Result<kf_core::Tree> {
        let d = self.trainer.head.cfg.d;
        let mut tree = kf_core::Tree::root(own);
        let budget = budget.min(kf_core::TREE_ROWS);
        if budget <= 1 || !self.ready(seq) {
            return Ok(tree);
        }
        let nwin = if self.held_pos[seq] == Some(self.next_pos[seq]) { self.cached[seq] } else { 0 };
        const A: usize = train::DRAFT_ANCESTORS;
        // Node zero's input row is the held row, every other node's the prediction of its parent
        unsafe { cudarc::driver::result::memcpy_dtod_async(self.rows.prednode.ptr(), self.held.ptr() + (seq * d * 4) as u64, d * 4, b.stream.cu_stream()) }?;
        let mut parent_pred: Vec<usize> = vec![0];
        let mut frontier: Vec<usize> = vec![0];
        let mut anc = vec![-1i32; kf_core::TREE_ROWS * A];
        anc[0] = 0;
        let mut depth = 0;
        while !frontier.is_empty() && tree.len() < budget && depth < DRAFT_FANOUT.len() {
            let m = frontier.len();
            for (j, src) in parent_pred.iter().enumerate() {
                unsafe { cudarc::driver::result::memcpy_dtod_async(self.rows.f.ptr() + (j * d * 4) as u64, self.rows.prednode.ptr() + (src * d * 4) as u64, d * 4, b.stream.cu_stream()) }?;
            }
            let tokens: Vec<i32> = frontier.iter().map(|n| tree.tokens[*n] as i32).collect();
            crate::exec::htod(&b.stream, dptr(&self.rows.tokens, &b.stream), &tokens)?;
            self.embed(b, dptr(&self.rows.tokens, &b.stream), self.rows.e.ptr(), m)?;
            let anc_rows: Vec<i32> = frontier.iter().flat_map(|n| anc[n * A..(n + 1) * A].iter().copied()).collect();
            crate::exec::htod(&b.stream, dptr(&self.rows.anc, &b.stream), &anc_rows)?;
            self.project_kv(b, m)?;
            for (j, n) in frontier.iter().enumerate() {
                unsafe {
                    cudarc::driver::result::memcpy_dtod_async(self.rows.knode.ptr() + (n * d * 4) as u64, self.rows.k.ptr() + (j * d * 4) as u64, d * 4, b.stream.cu_stream())?;
                    cudarc::driver::result::memcpy_dtod_async(self.rows.vnode.ptr() + (n * d * 4) as u64, self.rows.v.ptr() + (j * d * 4) as u64, d * 4, b.stream.cu_stream())?;
                }
            }
            self.finish_rows(b, seq, m, nwin)?;
            for (j, n) in frontier.iter().enumerate() {
                unsafe { cudarc::driver::result::memcpy_dtod_async(self.rows.prednode.ptr() + (n * d * 4) as u64, self.rows.pred.ptr() + (j * d * 4) as u64, d * 4, b.stream.cu_stream()) }?;
            }
            let fan = DRAFT_FANOUT[depth].min(budget - tree.len()).max(1);
            let kids = self.score(b, self.rows.pred.ptr(), m, &self.draft, fan)?;
            let mut next_frontier = Vec::new();
            let mut next_parent = Vec::new();
            for (j, n) in frontier.iter().enumerate() {
                for (tok, _) in &kids[j] {
                    if tree.len() >= budget {
                        break;
                    }
                    if tree.has_child(*n, *tok) {
                        continue;
                    }
                    let Some(row) = tree.push(*n, *tok) else { break };
                    let mut list = anc[n * A..(n + 1) * A].to_vec();
                    if let Some(slot) = list.iter().position(|a| *a < 0) {
                        list[slot] = row as i32;
                    }
                    anc[row * A..(row + 1) * A].copy_from_slice(&list);
                    next_frontier.push(row);
                    next_parent.push(*n);
                }
            }
            frontier = next_frontier;
            parent_pred = next_parent;
            depth += 1;
        }
        Ok(tree)
    }
}

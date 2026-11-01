//! The expert coprocessor, host worker threads that compute the routed experts missing from a layer's device slots,
//! driven by the device itself through a mailbox per layer in mapped pinned memory, so a decode pass never waits on
//! the host thread. The dispatch kernel of a layer posts its selections, its misses and the input rows and raises
//! req_seq to the pass number, the leader worker turns that into a job every worker takes chunks of, and the await
//! kernel of the layer polls done_seq for the pass number before it reads the partial sum back.
use crate::exec::{dev_alloc, dptr, htod};
use anyhow::{Context, Result, bail, ensure};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, PinnedHostSlice, sys};
use kf_backend_cpu::experts::{ExpertTensor, Rows, down_cols, hidden_rows};
use kf_core::{Act, AllocKind, GluLimit, Pool};
use kf_quant::ActKind;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

/// Rows up to which a pass with cached experts takes the coprocessor path, more rows stream every expert through the device
pub(crate) const COPROC_ROWS: usize = 8;

/// Routed pairs one mailbox holds, rows times experts used
pub(crate) const MAX_PAIRS: usize = 256;

/// Byte offsets inside a layer's mailbox, the header then the selections, the misses, the input rows and the partial
const OFF_SEL: usize = 128;
const OFF_MISS: usize = OFF_SEL + 4 * MAX_PAIRS;
const OFF_X: usize = OFF_MISS + 12 * MAX_PAIRS;

/// Header offsets, see KfMbox in moe.cu
const H_REQ: usize = 0;
const H_NMISS: usize = 4;
const H_ROWS: usize = 8;
const H_DONE: usize = 64;
const H_ERR: usize = 68;

/// Hidden rows one phase one unit covers and output columns one phase two unit covers
const CHUNK_FF: usize = 32;
const CHUNK_EMBD: usize = 64;

/// One stacked expert tensor the workers read straight from the mapped file
#[derive(Clone, Copy)]
pub(crate) struct TensorRef {
    pub ptr: *const u8,
    pub len: usize,
    pub ty: u32,
    pub n_in: usize,
    pub n_out: usize,
}

unsafe impl Send for TensorRef {}
unsafe impl Sync for TensorRef {}

impl TensorRef {
    fn tensor(&self) -> Result<ExpertTensor<'static>> {
        // The file mapping these bytes come from lives as long as the loaded model, which outlives every backend
        let bytes = unsafe { std::slice::from_raw_parts(self.ptr, self.len) };
        ExpertTensor::new(self.ty, bytes, self.n_in, self.n_out)
    }
}

/// What the workers need to compute one layer's missing experts
pub(crate) struct LayerDesc {
    pub gate: Option<TensorRef>,
    pub up: TensorRef,
    pub down: TensorRef,
    pub n_embd: usize,
    pub n_ff: usize,
    pub n_used: usize,
    pub act: Act,
    pub limit: Option<GluLimit>,
    pub scale: f32,
}

/// Device addresses of the regions of one mailbox, the arguments of the dispatch and await kernels
#[derive(Clone, Copy, Debug)]
pub(crate) struct MailboxPtrs {
    pub header: u64,
    pub sel: u64,
    pub miss: u64,
    pub x: u64,
    pub partial: u64,
}

struct Mailbox {
    _pinned: PinnedHostSlice<u8>,
    host: *mut u8,
    dev: u64,
    rows_max: usize,
    n_embd: usize,
}

unsafe impl Send for Mailbox {}
unsafe impl Sync for Mailbox {}

impl Mailbox {
    fn header_u32(&self, off: usize) -> &AtomicU32 {
        unsafe { &*(self.host.add(off) as *const AtomicU32) }
    }

    fn header_i32(&self, off: usize) -> i32 {
        unsafe { (self.host.add(off) as *const AtomicU32).as_ref().expect("header").load(Ordering::Acquire) as i32 }
    }

    fn sel(&self, n: usize) -> &[i32] {
        unsafe { std::slice::from_raw_parts(self.host.add(OFF_SEL) as *const i32, n) }
    }

    /// Row, expert and weight of miss k
    fn miss(&self, k: usize) -> (usize, usize, f32) {
        let p = unsafe { self.host.add(OFF_MISS + 12 * k) as *const i32 };
        unsafe { ((*p) as usize, (*p.add(1)) as usize, f32::from_bits(*(p.add(2) as *const u32))) }
    }

    fn x(&self, rows: usize) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.host.add(OFF_X) as *const f32, rows * self.n_embd) }
    }

    fn partial_mut(&self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.host.add(OFF_X + 4 * self.rows_max * self.n_embd) as *mut f32, self.rows_max * self.n_embd) }
    }

    fn ptrs(&self) -> MailboxPtrs {
        let x = self.dev + OFF_X as u64;
        MailboxPtrs { header: self.dev, sel: self.dev + OFF_SEL as u64, miss: self.dev + OFF_MISS as u64, x, partial: x + (4 * self.rows_max * self.n_embd) as u64 }
    }
}

struct Layer {
    mbox: Mailbox,
    desc: LayerDesc,
}

/// One pass's misses of one layer as the workers compute them
struct Job {
    layer: usize,
    seq: u32,
    rows: usize,
    /// Distinct missed experts with the pass rows and routing weights of each
    experts: Vec<(usize, Vec<usize>, Vec<f32>)>,
    xg: Vec<Rows>,
    xu: Vec<Rows>,
    hidden: Vec<Mutex<Vec<f32>>>,
    hq: Vec<OnceLock<Rows>>,
    units1: usize,
    units2: usize,
    /// Claim and completion counts of the hidden units, the quantizations, and the output column units
    next1: AtomicUsize,
    done1: AtomicUsize,
    nextq: AtomicUsize,
    doneq: AtomicUsize,
    next2: AtomicUsize,
    done2: AtomicUsize,
}

impl Job {
    /// Spins until count reaches target, the end of a phase every unit of which some worker finished
    fn wait_for(count: &AtomicUsize, target: usize) {
        while count.load(Ordering::Acquire) < target {
            std::hint::spin_loop();
        }
    }
}

struct Shared {
    layers: Vec<OnceLock<Layer>>,
    job: Mutex<Option<Arc<Job>>>,
    job_gen: AtomicUsize,
    active: AtomicBool,
    stop: AtomicBool,
    wake: Mutex<()>,
    cv: Condvar,
    /// Microseconds the workers spent on jobs this pass, the jobs they ran, and the longest job
    busy_us: AtomicUsize,
    jobs: AtomicUsize,
    max_us: AtomicUsize,
}

/// The worker pool with the pass counter the kernels compare against
pub(crate) struct Coproc {
    shared: Arc<Shared>,
    workers: Vec<std::thread::JoinHandle<()>>,
    ctx: Arc<CudaContext>,
    pub seq: u32,
    _seq_dev: CudaSlice<u32>,
    pub seq_ptr: u64,
}

impl Drop for Coproc {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Release);
        self.shared.active.store(true, Ordering::Release);
        self.shared.cv.notify_all();
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

/// Worker threads for the coprocessor, half the hardware threads unless KF_COPROC_THREADS says otherwise, so the
/// workers, the copier and the pass thread never outnumber the cores and no worker loses a scheduler slice mid job
fn thread_count() -> usize {
    if let Some(n) = std::env::var("KF_COPROC_THREADS").ok().and_then(|v| v.parse::<usize>().ok()).filter(|n| *n >= 1) {
        return n;
    }
    (std::thread::available_parallelism().map_or(4, |n| n.get()) / 2).max(2)
}

impl Coproc {
    pub fn new(ctx: &Arc<CudaContext>, stream: &Arc<CudaStream>, pool: Pool, n_layers: usize) -> Result<Self> {
        let threads = thread_count();
        let seq_dev = dev_alloc::<u32>(stream, pool, 1, AllocKind::Compute, "coprocessor pass counter")?;
        let seq_ptr = dptr(&seq_dev, stream);
        // The pool is born inside a pass, which runs as pass one with the workers awake
        htod(stream, seq_ptr, &[1u32])?;
        let shared = Arc::new(Shared {
            layers: (0..n_layers).map(|_| OnceLock::new()).collect(),
            job: Mutex::new(None),
            job_gen: AtomicUsize::new(0),
            active: AtomicBool::new(true),
            stop: AtomicBool::new(false),
            wake: Mutex::new(()),
            cv: Condvar::new(),
            busy_us: AtomicUsize::new(0),
            jobs: AtomicUsize::new(0),
            max_us: AtomicUsize::new(0),
        });
        let workers = (0..threads)
            .map(|t| {
                let s = shared.clone();
                std::thread::Builder::new().name(format!("kf-expert-{t}")).spawn(move || worker(s, t)).context("spawn coprocessor worker")
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { shared, workers, ctx: ctx.clone(), seq: 1, _seq_dev: seq_dev, seq_ptr })
    }

    pub fn has_layer(&self, layer: usize) -> bool {
        self.shared.layers.get(layer).is_some_and(|l| l.get().is_some())
    }

    /// Allocates the mailbox of a layer and hands the workers its description
    pub fn ensure_layer(&mut self, layer: usize, desc: LayerDesc) -> Result<()> {
        ensure!(layer < self.shared.layers.len(), "layer {layer} past the {} layers of the coprocessor", self.shared.layers.len());
        if self.has_layer(layer) {
            return Ok(());
        }
        ensure!(COPROC_ROWS * desc.n_used <= MAX_PAIRS, "{} experts per row over {COPROC_ROWS} rows exceed the mailbox of {MAX_PAIRS} pairs", desc.n_used);
        let bytes = OFF_X + 2 * 4 * COPROC_ROWS * desc.n_embd;
        let mut pinned = unsafe { self.ctx.alloc_pinned_with_flags::<u8>(bytes, sys::CU_MEMHOSTALLOC_DEVICEMAP) }.context("mapped mailbox")?;
        let host = pinned.as_mut_ptr().context("mailbox pointer")?;
        unsafe { std::ptr::write_bytes(host, 0, bytes) };
        let mut dev: sys::CUdeviceptr = 0;
        unsafe { sys::cuMemHostGetDevicePointer_v2(&mut dev, host as *mut std::ffi::c_void, 0) }.result().context("mailbox device pointer")?;
        kf_core::event::alloc(Pool::Host, AllocKind::Compute, bytes as u64);
        let mbox = Mailbox { _pinned: pinned, host, dev, rows_max: COPROC_ROWS, n_embd: desc.n_embd };
        self.shared.layers[layer].set(Layer { mbox, desc }).map_err(|_| anyhow::anyhow!("layer {layer} registered twice"))?;
        Ok(())
    }

    pub fn mailbox(&self, layer: usize) -> Option<MailboxPtrs> {
        self.shared.layers.get(layer)?.get().map(|l| l.mbox.ptrs())
    }

    /// Counts the pass, writes its number for the kernels, and wakes the workers
    pub fn begin_pass(&mut self, stream: &CudaStream) -> Result<()> {
        self.seq = self.seq.wrapping_add(1);
        if self.seq == 0 {
            self.seq = 1;
        }
        htod(stream, self.seq_ptr, &[self.seq])?;
        self.shared.busy_us.store(0, Ordering::Relaxed);
        self.shared.jobs.store(0, Ordering::Relaxed);
        self.shared.max_us.store(0, Ordering::Relaxed);
        let _g = self.shared.wake.lock().expect("wake lock");
        self.shared.active.store(true, Ordering::Release);
        self.shared.cv.notify_all();
        Ok(())
    }

    /// Puts the workers to sleep and reads back what every layer posted this pass, an error when a layer timed out
    pub fn end_pass(&mut self) -> Result<Vec<PassLayer>> {
        self.shared.active.store(false, Ordering::Release);
        let mut out = Vec::new();
        for (layer, slot) in self.shared.layers.iter().enumerate() {
            let Some(l) = slot.get() else { continue };
            let mb = &l.mbox;
            if mb.header_u32(H_ERR).load(Ordering::Acquire) != 0 {
                mb.header_u32(H_ERR).store(0, Ordering::Release);
                bail!("the expert coprocessor did not answer layer {layer} within two seconds");
            }
            if mb.header_u32(H_REQ).load(Ordering::Acquire) != self.seq {
                continue;
            }
            let rows = mb.header_i32(H_ROWS) as usize;
            let n_miss = mb.header_i32(H_NMISS) as usize;
            let sel: Vec<i32> = mb.sel(rows * l.desc.n_used).to_vec();
            let missed: Vec<usize> = (0..n_miss).map(|k| mb.miss(k).1).collect();
            out.push(PassLayer { layer, sel, missed });
        }
        Ok(out)
    }

    /// Microseconds of worker time, jobs run, and the longest job since the pass began
    pub fn pass_stats(&self) -> (usize, usize, usize) {
        (self.shared.busy_us.load(Ordering::Relaxed), self.shared.jobs.load(Ordering::Relaxed), self.shared.max_us.load(Ordering::Relaxed))
    }
}

/// What one layer posted in a pass, every routed expert and the ones its slots did not hold
pub(crate) struct PassLayer {
    pub layer: usize,
    pub sel: Vec<i32>,
    pub missed: Vec<usize>,
}

/// Activation kind two tensors share for their input, the gate's and the up's
fn kinds(desc: &LayerDesc) -> (ActKind, ActKind, ActKind) {
    let kind = |t: &TensorRef| kf_quant::act_kind(kf_quant::by_id(t.ty).expect("quant type"));
    (desc.gate.as_ref().map_or_else(|| kind(&desc.up), kind), kind(&desc.up), kind(&desc.down))
}

/// Builds the job of a request the leader found, the misses grouped by expert with their rows quantized
fn build_job(layer: usize, l: &Layer, seq: u32) -> Result<Job> {
    let mb = &l.mbox;
    let rows = mb.header_i32(H_ROWS) as usize;
    let n_miss = mb.header_i32(H_NMISS) as usize;
    ensure!(rows <= mb.rows_max && n_miss <= MAX_PAIRS, "mailbox of layer {layer} names {rows} rows and {n_miss} misses");
    let mut experts: Vec<(usize, Vec<usize>, Vec<f32>)> = Vec::new();
    for k in 0..n_miss {
        let (r, e, w) = mb.miss(k);
        match experts.iter_mut().find(|x| x.0 == e) {
            Some(x) => {
                x.1.push(r);
                x.2.push(w * l.desc.scale);
            }
            None => experts.push((e, vec![r], vec![w * l.desc.scale])),
        }
    }
    let (kg, ku, _) = kinds(&l.desc);
    let x = mb.x(rows);
    let xg_all = Rows::new(kg, x, rows).with_context(|| format!("layer {layer} input"))?;
    let xu_all = if ku == kg { None } else { Some(Rows::new(ku, x, rows).with_context(|| format!("layer {layer} input"))?) };
    let xg: Vec<Rows> = experts.iter().map(|(_, r, _)| xg_all.gather(r)).collect();
    let xu: Vec<Rows> = experts.iter().map(|(_, r, _)| xu_all.as_ref().unwrap_or(&xg_all).gather(r)).collect();
    let hidden = experts.iter().map(|(_, r, _)| Mutex::new(vec![0.0f32; r.len() * l.desc.n_ff])).collect();
    let hq = experts.iter().map(|_| OnceLock::new()).collect();
    let per_expert = l.desc.n_ff.div_ceil(CHUNK_FF);
    Ok(Job {
        layer,
        seq,
        rows,
        units1: experts.len() * per_expert,
        units2: l.desc.n_embd.div_ceil(CHUNK_EMBD),
        experts,
        xg,
        xu,
        hidden,
        hq,
        next1: AtomicUsize::new(0),
        done1: AtomicUsize::new(0),
        nextq: AtomicUsize::new(0),
        doneq: AtomicUsize::new(0),
        next2: AtomicUsize::new(0),
        done2: AtomicUsize::new(0),
    })
}

/// Phase one, hidden rows of every missed expert in chunks any worker claims
fn phase_hidden(job: &Job, l: &Layer) -> Result<()> {
    let desc = &l.desc;
    let gate = desc.gate.as_ref().map(|g| g.tensor()).transpose()?;
    let up = desc.up.tensor()?;
    let per_expert = desc.n_ff.div_ceil(CHUNK_FF);
    loop {
        let u = job.next1.fetch_add(1, Ordering::AcqRel);
        if u >= job.units1 {
            break;
        }
        let (k, c) = (u / per_expert, u % per_expert);
        let (e, rows, _) = &job.experts[k];
        let (o0, o1) = (c * CHUNK_FF, ((c + 1) * CHUNK_FF).min(desc.n_ff));
        let mut tmp = vec![0.0f32; rows.len() * desc.n_ff];
        hidden_rows(gate.as_ref(), &up, *e, o0, o1, &job.xg[k], &job.xu[k], desc.act, desc.limit, desc.n_ff, &mut tmp);
        {
            let mut h = job.hidden[k].lock().expect("hidden lock");
            for r in 0..rows.len() {
                h[r * desc.n_ff + o0..r * desc.n_ff + o1].copy_from_slice(&tmp[r * desc.n_ff + o0..r * desc.n_ff + o1]);
            }
        }
        job.done1.fetch_add(1, Ordering::AcqRel);
    }
    Ok(())
}

/// Between the phases, the hidden rows of expert k quantized for the down projection by whichever worker claims k
fn phase_quant(job: &Job, l: &Layer) -> Result<()> {
    let (_, _, kd) = kinds(&l.desc);
    loop {
        let k = job.nextq.fetch_add(1, Ordering::AcqRel);
        if k >= job.experts.len() {
            break;
        }
        let q = {
            let h = job.hidden[k].lock().expect("hidden lock");
            Rows::new(kd, &h, job.experts[k].1.len()).with_context(|| format!("layer {} hidden", job.layer))?
        };
        job.hq[k].set(q).map_err(|_| anyhow::anyhow!("hidden of expert {k} quantized twice"))?;
        job.doneq.fetch_add(1, Ordering::AcqRel);
    }
    Ok(())
}

/// Phase two, output columns in chunks any worker claims, each summing every missed expert into the partial rows
fn phase_down(job: &Job, l: &Layer) -> Result<()> {
    let desc = &l.desc;
    let down = desc.down.tensor()?;
    let partial = l.mbox.partial_mut();
    loop {
        let u = job.next2.fetch_add(1, Ordering::AcqRel);
        if u >= job.units2 {
            break;
        }
        let (c0, c1) = (u * CHUNK_EMBD, ((u + 1) * CHUNK_EMBD).min(desc.n_embd));
        for r in 0..job.rows {
            partial[r * desc.n_embd + c0..r * desc.n_embd + c1].fill(0.0);
        }
        for (k, (e, rows, w)) in job.experts.iter().enumerate() {
            let hq = job.hq[k].get().context("hidden not quantized")?;
            down_cols(&down, *e, c0, c1, hq, w, rows, desc.n_embd, partial);
        }
        job.done2.fetch_add(1, Ordering::AcqRel);
    }
    Ok(())
}

/// Runs the phases of a job alongside the other workers, each phase starting once every unit of the one before is
/// finished by someone, so a worker the scheduler holds back delays only the unit it claimed. The leader signals the
/// device once the last column unit is done.
fn run_job(s: &Shared, job: &Arc<Job>, leader: bool) {
    let l = s.layers[job.layer].get().expect("job layer");
    let mut err = phase_hidden(job, l).err();
    Job::wait_for(&job.done1, job.units1);
    err = err.or(phase_quant(job, l).err());
    Job::wait_for(&job.doneq, job.experts.len());
    err = err.or(phase_down(job, l).err());
    if let Some(e) = err {
        eprintln!("kungfu event=error code=coproc layer={} detail={e:#}", job.layer);
        l.mbox.header_u32(H_ERR).store(1, Ordering::Release);
        job.done2.store(job.units2, Ordering::Release);
    }
    if leader {
        Job::wait_for(&job.done2, job.units2);
        l.mbox.header_u32(H_DONE).store(job.seq, Ordering::Release);
    }
}

/// The loop of worker t, the first thread also scanning the mailboxes for requests and publishing jobs
fn worker(s: Arc<Shared>, t: usize) {
    let leader = t == 0;
    let mut seen: Vec<u32> = vec![0; s.layers.len()];
    let mut seen_gen = 0usize;
    loop {
        if s.stop.load(Ordering::Acquire) {
            return;
        }
        if !s.active.load(Ordering::Acquire) {
            let g = s.wake.lock().expect("wake lock");
            if !s.active.load(Ordering::Acquire) && !s.stop.load(Ordering::Acquire) {
                let _ = s.cv.wait_timeout(g, std::time::Duration::from_millis(20)).expect("wake wait");
            }
            continue;
        }
        if leader {
            let mut posted = false;
            for (layer, slot) in s.layers.iter().enumerate() {
                let Some(l) = slot.get() else { continue };
                let req = l.mbox.header_u32(H_REQ).load(Ordering::Acquire);
                if req == seen[layer] {
                    continue;
                }
                seen[layer] = req;
                if l.mbox.header_i32(H_NMISS) == 0 {
                    continue;
                }
                let t0 = std::time::Instant::now();
                match build_job(layer, l, req) {
                    Ok(job) => {
                        let job = Arc::new(job);
                        *s.job.lock().expect("job lock") = Some(job.clone());
                        s.job_gen.fetch_add(1, Ordering::AcqRel);
                        seen_gen += 1;
                        run_job(&s, &job, true);
                        let us = t0.elapsed().as_micros() as usize;
                        s.busy_us.fetch_add(us, Ordering::Relaxed);
                        s.max_us.fetch_max(us, Ordering::Relaxed);
                        s.jobs.fetch_add(1, Ordering::Relaxed);
                        posted = true;
                    }
                    Err(e) => {
                        eprintln!("kungfu event=error code=coproc layer={layer} detail={e:#}");
                        l.mbox.header_u32(H_ERR).store(1, Ordering::Release);
                        l.mbox.header_u32(H_DONE).store(req, Ordering::Release);
                    }
                }
            }
            if !posted {
                std::hint::spin_loop();
            }
        } else {
            let g = s.job_gen.load(Ordering::Acquire);
            if g == seen_gen {
                std::hint::spin_loop();
                continue;
            }
            seen_gen = g;
            let job = s.job.lock().expect("job lock").clone().expect("published job");
            run_job(&s, &job, false);
        }
    }
}

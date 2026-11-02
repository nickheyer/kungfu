//! Cooperative launches for a whole single row decode pass, the launch list of a device segment run as the stages
//! of kernels/persist.cuh, each stage waiting only for the latest stage whose buffers it touches. Each launch owns a
//! module instance whose constant bank holds its stage table and arguments, a pass taking as many launches as its
//! stages need banks. Launches past the last one with a stage form run as themselves after the stages.
use crate::CudaBackend;
use crate::exec::{Arg, Launch, dev_alloc, dptr, dtoh, htod};
use crate::sources;
use anyhow::{Context, Result, bail, ensure};
use cudarc::driver::{CudaModule, CudaSlice, DevicePtr};
use kf_core::AllocKind;
use sha2::{Digest, Sha256};
use std::sync::Arc;

const INPUT: i32 = 0;
const EMBED: i32 = 1;
const RMS_Q8: i32 = 2;
const MATVEC: i32 = 3;
const ATTENTION: i32 = 4;
const GATED_Q8: i32 = 5;

/// Bytes of the constant bank a module holds for its stages, KF_PS_DATA of the kernel
const DATA: usize = 61440;

/// Threads of a persistent block, KF_PS_THREADS of the kernel, the block size of every launch a stage stands for
const THREADS: u32 = 256;

/// A stage header as the kernel reads it, the arguments at argoff in the same bank, dep the stage it waits for
#[repr(C)]
#[derive(Clone, Copy)]
struct Stage {
    kind: i32,
    sel: i32,
    items: i32,
    gx: i32,
    gy: i32,
    gz: i32,
    smem: i32,
    argoff: i32,
    dep: i32,
}

const HEADER: usize = std::mem::size_of::<Stage>();

/// One stage before placement, its header, the arguments of the launch it stands for, and the activation buffers
/// it reads and writes
struct Pending {
    stage: Stage,
    args: Vec<Arg>,
    name: String,
    reads: Vec<u64>,
    writes: Vec<u64>,
}

/// Prologue kinds a product stage starts with, KF_PS_PRE_* of the kernel
const PRE_NONE: i32 = 0;
const PRE_RMS: i32 = 1;
const PRE_GATED: i32 = 2;

/// Bytes of shared memory a product's staged activation of n columns takes, kf_ps_act_bytes of the kernel
fn act_bytes(n: usize) -> usize {
    n + (n / 16) * 8 + (n / 32) * 8
}

/// Bytes of shared memory a block fills with the rows of its coming items before it waits on the stage before,
/// KF_PERSIST_WARM overriding it for tuning
fn warm_bytes() -> u32 {
    std::env::var("KF_PERSIST_WARM").ok().and_then(|v| v.parse().ok()).unwrap_or(8192)
}

fn ptr(args: &[Arg], i: usize) -> Result<u64> {
    match args.get(i) {
        Some(Arg::P(v)) => Ok(*v),
        other => anyhow::bail!("argument {i} is {other:?} where a pointer was expected"),
    }
}

fn int(args: &[Arg], i: usize) -> Result<i64> {
    match args.get(i) {
        Some(Arg::I(v)) => Ok(*v as i64),
        Some(Arg::L(v)) => Ok(*v),
        other => anyhow::bail!("argument {i} is {other:?} where an integer was expected"),
    }
}

/// Pointers of the arguments at the given positions that are not null
fn ptrs(args: &[Arg], at: &[usize]) -> Vec<u64> {
    at.iter().filter_map(|&i| match args.get(i) {
        Some(Arg::P(v)) if *v != 0 => Some(*v),
        _ => None,
    }).collect()
}

/// The activation buffers a stage reads and writes, by the argument layout of the kernel it stands for, caches and
/// counters counted on both sides, a product with segments adding every member's output and residual
fn io_of(b: &CudaBackend, kind: i32, args: &[Arg]) -> Result<(Vec<u64>, Vec<u64>)> {
    Ok(match kind {
        INPUT => (ptrs(args, &[0]), ptrs(args, &[1])),
        EMBED => (ptrs(args, &[1]), ptrs(args, &[2])),
        RMS_Q8 => (ptrs(args, &[0]), ptrs(args, &[6, 7, 8, 9, 10])),
        GATED_Q8 => (ptrs(args, &[0, 1]), ptrs(args, &[6, 7, 8, 9, 10])),
        MATVEC => {
            let mut reads = ptrs(args, &[1, 2, 3, 4, 5, 7]);
            let mut writes = ptrs(args, &[8]);
            let (y, segs, nseg) = (ptr(args, 8)?, ptr(args, 13)?, int(args, 14)?);
            if segs != 0 {
                let mut words = vec![0i64; 6 * nseg as usize];
                dtoh(&b.stream, &mut words, segs)?;
                for e in words.chunks_exact(6) {
                    writes.push(y.wrapping_add(e[1] as u64));
                    if e[3] != i64::MIN {
                        reads.push(y.wrapping_add(e[3] as u64));
                    }
                }
            }
            (reads, writes)
        }
        ATTENTION => {
            let both = ptrs(args, &[1, 2, 8, 16]);
            let mut reads = ptrs(args, &[0, 3, 4, 5, 23, 24, 25, 43]);
            let mut writes = ptrs(args, &[7, 18, 19, 20, 21, 22]);
            reads.extend_from_slice(&both);
            writes.extend_from_slice(&both);
            (reads, writes)
        }
        other => bail!("stage kind {other} has no buffer description"),
    })
}

/// Rows of f32 a product's prologue may read per block before the redundant reads of every block cost more L2
/// bandwidth than one block's epilogue would
const PRE_COLS: usize = 4096;

/// Folds every norm or gated activation of one row whose quantized output only products read into those products as
/// their prologue, each product staging the activation itself from the f32 rows, and drops the stage
fn fold_prologues(stages: &mut Vec<Pending>) -> Result<()> {
    let n = stages.len();
    let mut dropped = vec![false; n];
    let mut pre: Vec<Option<(i32, Vec<Arg>, Vec<u64>)>> = (0..n).map(|_| None).collect();
    for i in 0..n {
        let st = &stages[i];
        let kind = match st.stage.kind {
            RMS_Q8 => PRE_RMS,
            GATED_Q8 => PRE_GATED,
            _ => continue,
        };
        let cols = int(&st.args, if kind == PRE_RMS { 3 } else { 2 })? as usize;
        if cols * if kind == PRE_RMS { 1 } else { 2 } > PRE_COLS {
            continue;
        }
        if st.stage.gx != 1 {
            continue;
        }
        let q = ptr(&st.args, 6)?;
        // The readers of the quantized output up to the stage that next writes it
        let mut readers = Vec::new();
        let mut folds = true;
        for j in i + 1..n {
            if stages[j].writes.contains(&q) {
                break;
            }
            if stages[j].reads.contains(&q) {
                let is_product = stages[j].stage.kind == MATVEC && ptr(&stages[j].args, 1)? == q && pre[j].is_none() && int(&stages[j].args, 9)? == 1;
                if !is_product {
                    folds = false;
                    break;
                }
                readers.push(j);
            }
        }
        if !folds || readers.is_empty() {
            continue;
        }
        let inputs = match kind {
            PRE_RMS => ptrs(&st.args, &[0]),
            _ => ptrs(&st.args, &[0, 1]),
        };
        for j in readers {
            pre[j] = Some((kind, st.args.clone(), inputs.clone()));
        }
        dropped[i] = true;
    }
    let mut out = Vec::with_capacity(n);
    for (i, mut st) in std::mem::take(stages).into_iter().enumerate() {
        if dropped[i] {
            continue;
        }
        if st.stage.kind == MATVEC {
            match pre[i].take() {
                Some((kind, args, inputs)) => {
                    let scratch = ptrs(&st.args, &[1, 2, 3, 4, 5]);
                    st.reads.retain(|p| !scratch.contains(p));
                    st.reads.extend(inputs);
                    st.stage.smem = st.stage.smem.max(act_bytes(int(&st.args, 10)? as usize) as i32);
                    st.name = format!("{} with {}", st.name, stages_name(kind));
                    st.args.push(Arg::I(kind));
                    st.args.extend(args);
                }
                None => st.args.push(Arg::I(PRE_NONE)),
            }
        }
        out.push(st);
    }
    *stages = out;
    Ok(())
}

fn stages_name(pre: i32) -> &'static str {
    if pre == PRE_RMS { "norm prologue" } else { "gated prologue" }
}

/// The latest earlier stage each stage must wait for, one whose writes meet its reads or writes or whose reads
/// meet its writes, minus one when none
fn dependencies(stages: &mut [Pending]) {
    for t in 0..stages.len() {
        let mut dep = -1i32;
        for s in (0..t).rev() {
            let (ws, rs) = (&stages[s].writes, &stages[s].reads);
            let (wt, rt) = (&stages[t].writes, &stages[t].reads);
            if ws.iter().any(|p| rt.contains(p) || wt.contains(p)) || rs.iter().any(|p| wt.contains(p)) {
                dep = s as i32;
                break;
            }
        }
        stages[t].stage.dep = dep;
    }
}

/// One launch of the pass, the module instance holding its stages and the counters of its barriers
struct Chunk {
    _module: Arc<CudaModule>,
    _sync: CudaSlice<i32>,
    launch: Launch,
    names: Vec<String>,
    /// Arrival stamps of every block at every stage after a row of start stamps, under KF_PERSIST_TRACE
    trace: Option<CudaSlice<i64>>,
}

/// The launches of a pass in order, the stages then the launches without a stage form
pub(crate) struct PersistPlan {
    chunks: Vec<Chunk>,
    pub n_stages: usize,
    tail: Vec<Launch>,
}

impl PersistPlan {
    /// Issues every launch of the pass on the stream
    pub fn run(&self, stream: &cudarc::driver::CudaStream) -> Result<()> {
        self.chunks.iter().try_for_each(|c| c.launch.run_cooperative(stream))?;
        self.tail.iter().try_for_each(|l| l.run(stream))
    }

    /// Prints each stage's time from the last arrival at the stage before to the last arrival at it, the spread
    /// between the first and last block to arrive, and the time its blocks spent waiting, staging, and on items
    pub fn report(&self, b: &CudaBackend) -> Result<()> {
        let mut total = 0.0;
        for c in &self.chunks {
            let Some(buf) = &c.trace else { return Ok(()) };
            let g = c.launch.grid.0 as usize;
            let n = c.names.len();
            let mut t = vec![0i64; (4 * n + 1) * g];
            dtoh(&b.stream, &mut t, dptr(buf, &b.stream))?;
            let row = |s: usize| {
                let r = &t[s * g..(s + 1) * g];
                (*r.iter().filter(|v| **v != 0).min().unwrap_or(&0), *r.iter().max().expect("stamps"))
            };
            let us = |a: i64, b: i64| (b - a) as f64 / 1e3;
            for (i, name) in c.names.iter().enumerate() {
                let before = if i == 0 { row(0).1 } else { row(3 * n + i).1 };
                let (lo, hi) = row(3 * n + 1 + i);
                let waited = row(3 * i + 1).1;
                let staged = row(3 * i + 2).1;
                let looped = row(3 * i + 3).1;
                if staged > 0 {
                    eprintln!("stage {i:4} {name:44}: {:7.1} us  spread {:6.1}  wait {:5.1} stage {:5.1} items {:5.1} tail {:5.1}", us(before, hi), us(lo, hi), us(before, waited), us(waited, staged), us(staged, looped), us(looped, hi));
                } else {
                    eprintln!("stage {i:4} {name:44}: {:7.1} us  spread {:6.1}  wait {:5.1} work {:5.1}", us(before, hi), us(lo, hi), us(before, waited), us(waited, hi));
                }
            }
            total += us(row(0).1, row(4 * n).1);
        }
        let grid = self.chunks.first().map_or(0, |c| c.launch.grid.0);
        let smem = self.chunks.first().map_or(0, |c| c.launch.smem);
        eprintln!("persistent pass of {} stages in {} launches of {grid} blocks with {smem} bytes of shared memory: {total:.1} us", self.n_stages, self.chunks.len());
        Ok(())
    }
}

/// Kernel arguments laid out as the members of a C struct in order, each aligned to its own size
fn pack_args(args: &[Arg]) -> Vec<u8> {
    let mut out = Vec::with_capacity(args.len() * 8);
    for a in args {
        let (bytes, size): ([u8; 8], usize) = match a {
            Arg::P(v) => (v.to_le_bytes(), 8),
            Arg::L(v) => (v.to_le_bytes(), 8),
            Arg::I(v) => {
                let mut b = [0u8; 8];
                b[..4].copy_from_slice(&v.to_le_bytes());
                (b, 4)
            }
            Arg::F(v) => {
                let mut b = [0u8; 8];
                b[..4].copy_from_slice(&v.to_le_bytes());
                (b, 4)
            }
        };
        out.resize(out.len().next_multiple_of(size), 0);
        out.extend_from_slice(&bytes[..size]);
    }
    out.resize(out.len().next_multiple_of(8), 0);
    out
}

/// Quant type name of a module name, the part past quant_
fn quant_type(module: &str) -> Option<&str> {
    module.strip_prefix("quant_")
}

/// Index of a type name in the list, appended when new
fn type_index(types: &mut Vec<String>, name: &str) -> i32 {
    match types.iter().position(|t| t == name) {
        Some(i) => i as i32,
        None => {
            types.push(name.to_string());
            types.len() as i32 - 1
        }
    }
}

/// The stage kind and selector a launch stands for, none for a kernel without a persistent form
fn classify(name: &str, mv_types: &mut Vec<String>, embed_types: &mut Vec<String>, hds: &mut Vec<usize>) -> Option<(i32, i32)> {
    let (module, entry) = name.split_once('.')?;
    match (module, entry) {
        ("elementwise", "input_f32") => Some((INPUT, 0)),
        ("fused", "rms_norm_q8") => Some((RMS_Q8, 0)),
        ("fused", "gated_act_q8") => Some((GATED_Q8, 0)),
        ("attention", e) => {
            let hd: usize = e.strip_prefix("attention_decode_hd")?.parse().ok()?;
            if !hds.contains(&hd) {
                hds.push(hd);
            }
            Some((ATTENTION, hd as i32))
        }
        (m, "matvec_i8_1" | "matvec_i8_1s") => Some((MATVEC, type_index(mv_types, quant_type(m)?))),
        (m, "embed") => Some((EMBED, type_index(embed_types, quant_type(m)?))),
        _ => None,
    }
}

/// Name of the quant struct a type's kernel file defines, the first struct of the file
fn struct_name(ty: &str) -> Result<&'static str> {
    let src = sources::module_source(&format!("quant_{ty}")).with_context(|| format!("no quant module for {ty}"))?;
    let rest = src.split_once("\nstruct ").with_context(|| format!("quant_{ty} defines no struct"))?.1;
    Ok(rest.split([' ', '{']).next().context("struct name")?)
}

/// The persistent kernel source for the quant types and head sizes of a pass, the quant kernel files pasted without
/// their kernel entries so one translation unit holds every type
fn compose(mv_types: &[String], embed_types: &[String], hds: &[usize]) -> Result<String> {
    let mut src = String::from("#include \"quant.cuh\"\n#undef KF_QUANT_KERNELS_I8\n#define KF_QUANT_KERNELS_I8(T)\n#undef KF_QUANT_KERNELS\n#define KF_QUANT_KERNELS(T)\n#undef KF_FRAG_KERNELS\n#define KF_FRAG_KERNELS(T)\n");
    // Blocks per SM the kernel is bound to, and further defines separated by bars, both for tuning
    if let Some(n) = std::env::var("KF_PERSIST_BLOCKS").ok().and_then(|v| v.parse::<u32>().ok()).filter(|n| *n >= 1) {
        src.push_str(&format!("#define KF_PS_BLOCKS {n}\n"));
    }
    if let Ok(defs) = std::env::var("KF_PERSIST_DEFINES") {
        for d in defs.split('|').filter(|d| !d.trim().is_empty()) {
            src.push_str(&format!("#define {}\n", d.trim()));
        }
    }
    let mut all: Vec<&String> = mv_types.iter().collect();
    for t in embed_types {
        if !all.contains(&t) {
            all.push(t);
        }
    }
    for t in &all {
        src.push_str(sources::module_source(&format!("quant_{t}")).with_context(|| format!("no quant module for {t}"))?);
        src.push('\n');
    }
    let list = |types: &[String]| -> Result<String> {
        let mut s = String::new();
        for (i, t) in types.iter().enumerate() {
            s.push_str(&format!(" X({i}, {})", struct_name(t)?));
        }
        Ok(s)
    };
    src.push_str(&format!("#define KF_PERSIST_MV_TYPES(X){}\n", list(mv_types)?));
    src.push_str(&format!("#define KF_PERSIST_EMBED_TYPES(X){}\n", list(embed_types)?));
    let hd_list: String = hds.iter().map(|h| format!(" X({h})")).collect();
    src.push_str(&format!("#define KF_PERSIST_HDS(X){hd_list}\n#include \"persist.cuh\"\n"));
    Ok(src)
}

/// Bytes of the constant bank a run of stages takes, headers then arguments
fn bank_bytes(stages: &[Pending]) -> usize {
    (HEADER * stages.len()).next_multiple_of(8) + stages.iter().map(|p| pack_args(&p.args).len().next_multiple_of(8)).sum::<usize>()
}

/// The constant bank image of a run of stages starting at stage first, every header's argoff pointing into the bank
/// and every dependency relative to the run, minus one for a stage of an earlier launch
fn bank(stages: &[Pending], first: usize) -> Vec<u8> {
    let mut out = vec![0u8; (HEADER * stages.len()).next_multiple_of(8)];
    for (i, p) in stages.iter().enumerate() {
        let mut st = p.stage;
        st.argoff = out.len() as i32;
        st.dep = if st.dep >= first as i32 { st.dep - first as i32 } else { -1 };
        out.extend_from_slice(&pack_args(&p.args));
        out.resize(out.len().next_multiple_of(8), 0);
        let hp = &st as *const Stage as *const u8;
        // Plain old data of a repr C struct of nine ints
        out[HEADER * i..HEADER * (i + 1)].copy_from_slice(unsafe { std::slice::from_raw_parts(hp, HEADER) });
    }
    out
}

/// The persistent form of a launch list, the longest prefix of launches with a stage kind as stages and the rest as
/// they are, none when the first launch has no stage kind. The kernel is composed for the types of the list and
/// sized to the device by its occupancy, each launch's module instance filled with its stages.
pub(crate) fn plan(b: &mut CudaBackend, launches: &[&Launch]) -> Result<Option<PersistPlan>> {
    let mut mv_types = Vec::new();
    let mut embed_types = Vec::new();
    let mut hds = Vec::new();
    let mut pending = Vec::with_capacity(launches.len());
    let mut smem = 0u32;
    let mut covered = launches.len();
    for (i, l) in launches.iter().enumerate() {
        let name = b.kernels.kernel_name(&l.func).to_string();
        let Some((kind, sel)) = classify(&name, &mut mv_types, &mut embed_types, &mut hds) else {
            covered = i;
            break;
        };
        ensure!(l.block == THREADS, "{name} launches {} threads where the persistent pass runs {THREADS}", l.block);
        let (gx, gy, gz) = l.grid;
        let items = gx as usize * gy as usize * gz as usize;
        ensure!(items > 0 && items <= i32::MAX as usize, "{name} grid of {items} blocks");
        let (reads, writes) = io_of(b, kind, &l.args)?;
        pending.push(Pending { stage: Stage { kind, sel, items: items as i32, gx: gx as i32, gy: gy as i32, gz: gz as i32, smem: l.smem as i32, argoff: 0, dep: -1 }, args: l.args.clone(), name, reads, writes });
    }
    if pending.is_empty() {
        return Ok(None);
    }
    if std::env::var_os("KF_PERSIST_NO_FOLD").is_none() {
        fold_prologues(&mut pending)?;
    } else {
        pending.iter_mut().filter(|p| p.stage.kind == MATVEC).for_each(|p| p.args.push(Arg::I(PRE_NONE)));
    }
    dependencies(&mut pending);
    for p in &pending {
        smem = smem.max(p.stage.smem as u32);
    }
    // The warm buffer sits behind the widest stage's own shared memory
    let warm_off = smem.next_multiple_of(16);
    let warm = warm_bytes();
    smem = warm_off + warm;
    let src = compose(&mv_types, &embed_types, &hds)?;
    let mut h = Sha256::new();
    h.update(src.as_bytes());
    let name = format!("persist_{:x}", u64::from_le_bytes(h.finalize()[..8].try_into().expect("hash bytes")));
    let ptx = b.kernels.ptx(&name, &src)?;
    let pool = b.pool();
    let trace_on = std::env::var_os("KF_PERSIST_TRACE").is_some();
    // Every launch is one instance of the module, the first loaded now to size the grid by its occupancy
    let load = |b: &CudaBackend| -> Result<(Arc<CudaModule>, cudarc::driver::CudaFunction)> {
        let module = b.ctx.load_module(ptx.clone()).context("load persistent module")?;
        let func = module.load_function("persist").context("persistent kernel")?;
        if smem > 48 * 1024 {
            func.set_attribute(cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem as i32).context("persistent shared memory limit")?;
        }
        Ok((module, func))
    };
    let mut first = Some(load(b)?);
    let per_sm = first.as_ref().expect("module").1.occupancy_max_active_blocks_per_multiprocessor(THREADS, smem as usize, None).context("persistent kernel occupancy")?;
    ensure!(per_sm >= 1, "the persistent kernel fits no block on an SM with {smem} bytes of shared memory");
    let grid = per_sm as usize * b.kernels.sm_count();
    for p in &pending {
        let packed = pack_args(&p.args).len();
        ensure!(HEADER + packed <= DATA, "{} arguments of {packed} bytes past the {DATA} byte bank", p.name);
    }
    let mut chunks = Vec::new();
    let mut at = 0;
    while at < pending.len() {
        // The longest run of stages from here whose headers and arguments fit one bank
        let mut end = at + 1;
        while end < pending.len() && bank_bytes(&pending[at..end + 1]) <= DATA {
            end += 1;
        }
        let run = &pending[at..end];
        let image = bank(run, at);
        let (module, func) = match first.take() {
            Some(m) => m,
            None => load(b)?,
        };
        {
            let global = module.get_global("kf_ps_data", &b.stream).context("persistent constant bank")?;
            let (ptr, _guard) = global.device_ptr(&b.stream);
            htod(&b.stream, ptr, &image)?;
        }
        let sync = dev_alloc::<i32>(&b.stream, pool, run.len() + 1, AllocKind::Compute, "persistent counters")?;
        let sp = dptr(&sync, &b.stream);
        let trace = if trace_on { Some(dev_alloc::<i64>(&b.stream, pool, (4 * run.len() + 1) * grid, AllocKind::Compute, "persistent trace")?) } else { None };
        let trace_ptr = trace.as_ref().map_or(0, |t| dptr(t, &b.stream));
        let launch = Launch { func, grid: (grid as u32, 1, 1), block: THREADS, smem, args: vec![Arg::I(run.len() as i32), Arg::I(warm_off as i32), Arg::I(warm as i32), Arg::P(sp), Arg::P(sp + 4 * run.len() as u64), Arg::P(trace_ptr)] };
        chunks.push(Chunk { _module: module, _sync: sync, launch, names: run.iter().map(|p| p.name.clone()).collect(), trace });
        at = end;
    }
    b.stream.synchronize().context("persistent bank upload")?;
    let tail = launches[covered..].iter().map(|l| (*l).clone()).collect();
    Ok(Some(PersistPlan { chunks, n_stages: pending.len(), tail }))
}

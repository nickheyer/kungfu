//! Device side of the kernels a draft head learns with, see kernels/train.cu, over single precision device tensors
use crate::CudaBackend;
use crate::exec::{Arg, Launch, dev_alloc, dev_upload, dptr, dtoh};
use anyhow::{Result, ensure};
use cudarc::driver::CudaSlice;
use kf_core::AllocKind;

/// A single precision tensor on the device with its row major shape and address
pub struct Tensor {
    _buf: CudaSlice<f32>,
    ptr: u64,
    pub rows: usize,
    pub cols: usize,
}

impl Tensor {
    /// Zeros of a shape
    pub fn zeros(b: &CudaBackend, rows: usize, cols: usize, what: &str) -> Result<Self> {
        let buf = dev_alloc::<f32>(&b.stream, b.pool(), rows * cols, AllocKind::Compute, what)?;
        Ok(Self { ptr: dptr(&buf, &b.stream), _buf: buf, rows, cols })
    }

    /// A copy of host values in a shape
    pub fn upload(b: &CudaBackend, values: &[f32], rows: usize, cols: usize, what: &str) -> Result<Self> {
        ensure!(values.len() == rows * cols, "{} values for a {rows} by {cols} tensor", values.len());
        let buf = dev_upload(&b.stream, b.pool(), values, AllocKind::Compute, what)?;
        Ok(Self { ptr: dptr(&buf, &b.stream), _buf: buf, rows, cols })
    }

    pub fn ptr(&self) -> u64 {
        self.ptr
    }

    pub fn len(&self) -> usize {
        self.rows * self.cols
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The values on the host, after the stream drains
    pub fn download(&self, b: &CudaBackend) -> Result<Vec<f32>> {
        let mut out = vec![0.0f32; self.len()];
        dtoh(&b.stream, &mut out, self.ptr)?;
        Ok(out)
    }

    /// Zeros every value on the stream
    pub fn zero(&self, b: &CudaBackend) -> Result<()> {
        unsafe { cudarc::driver::result::memset_d8_async(self.ptr, 0, self.len() * 4, b.stream.cu_stream()) }?;
        Ok(())
    }

    /// Overwrites the values from the host
    pub fn set(&mut self, b: &CudaBackend, values: &[f32]) -> Result<()> {
        ensure!(values.len() == self.len(), "{} values for a tensor of {}", values.len(), self.len());
        crate::exec::htod(&b.stream, self.ptr, values)?;
        Ok(())
    }
}

fn grid_for(n: usize) -> u32 {
    n.div_ceil(256).clamp(1, 4096) as u32
}

fn launch(b: &mut CudaBackend, entry: &str, grid: (u32, u32, u32), block: u32, args: Vec<Arg>) -> Result<()> {
    let func = b.kernels.function("train", entry)?;
    Launch { func, grid, block, smem: 0, args }.run(&b.stream)
}

/// c = alpha op(a) op(b) + beta c, the product of an m by k and a k by n operand after the transposes, every operand
/// row major with its own leading dimension
#[allow(clippy::too_many_arguments)]
pub fn gemm(b: &mut CudaBackend, a: u64, bm: u64, c: u64, m: usize, n: usize, k: usize, lda: usize, ldb: usize, ldc: usize, ta: bool, tb: bool, alpha: f32, beta: f32) -> Result<()> {
    use Arg::{F, I, P};
    let grid = (n.div_ceil(64) as u32, m.div_ceil(64) as u32, 1);
    launch(b, "gemm_f32", grid, 256, vec![P(a), P(bm), P(c), I(m as i32), I(n as i32), I(k as i32), I(lda as i32), I(ldb as i32), I(ldc as i32), I(ta as i32), I(tb as i32), F(alpha), F(beta)])
}

/// Rows up to which the skinny product serves, KF_SK_ROWS of train.cu
pub const SKINNY_ROWS: usize = 16;

/// Splits of k the skinny product takes at most, the rows of its parts scratch being this many times SKINNY_ROWS
pub const SKINNY_SPLITS: usize = 32;

const SKINNY_THREADS: usize = 128;
const SKINNY_K: usize = 64;

/// c = alpha a b + beta c for at most SKINNY_ROWS rows of a, b read once in coalesced columns over splits of k whose
/// partial sums land in parts, SKINNY_SPLITS by SKINNY_ROWS by n floats, and are added in a fixed order
#[allow(clippy::too_many_arguments)]
pub fn gemm_skinny(b: &mut CudaBackend, a: u64, bm: u64, c: u64, m: usize, n: usize, k: usize, lda: usize, ldb: usize, ldc: usize, parts: u64, alpha: f32, beta: f32) -> Result<()> {
    use Arg::{F, I, P};
    ensure!((1..=SKINNY_ROWS).contains(&m), "{m} rows for the skinny product of at most {SKINNY_ROWS}");
    let col_blocks = n.div_ceil(SKINNY_THREADS);
    let want = (4 * b.kernels.sm_count()).div_ceil(col_blocks).max(1);
    let kper = k.div_ceil(want.min(SKINNY_SPLITS).min(k.div_ceil(SKINNY_K))).next_multiple_of(SKINNY_K);
    let splits = k.div_ceil(kper);
    launch(b, "gemm_skinny", (col_blocks as u32, splits as u32, 1), SKINNY_THREADS as u32, vec![P(a), P(bm), P(parts), I(m as i32), I(n as i32), I(k as i32), I(lda as i32), I(ldb as i32), I(kper as i32)])?;
    launch(b, "gemm_skinny_reduce", (grid_for(m * n), 1, 1), 256, vec![P(parts), P(c), I(m as i32), I(n as i32), I(ldc as i32), I(splits as i32), F(alpha), F(beta)])
}

/// h = silu(g) u over n values
pub fn silu_gate_fwd(b: &mut CudaBackend, g: u64, u: u64, h: u64, n: usize) -> Result<()> {
    use Arg::{L, P};
    launch(b, "silu_gate_fwd", (grid_for(n), 1, 1), 256, vec![P(g), P(u), P(h), L(n as i64)])
}

/// dg and du of h = silu(g) u given dh
pub fn silu_gate_bwd(b: &mut CudaBackend, g: u64, u: u64, dh: u64, dg: u64, du: u64, n: usize) -> Result<()> {
    use Arg::{L, P};
    launch(b, "silu_gate_bwd", (grid_for(n), 1, 1), 256, vec![P(g), P(u), P(dh), P(dg), P(du), L(n as i64)])
}

/// y = x w / rms(x) over rows of cols, the inverse rms of every row into inv
pub fn rms_fwd(b: &mut CudaBackend, x: u64, w: u64, y: u64, inv: u64, rows: usize, cols: usize, eps: f32) -> Result<()> {
    use Arg::{F, I, P};
    launch(b, "rms_fwd", (rows as u32, 1, 1), 256, vec![P(x), P(w), P(y), P(inv), I(cols as i32), F(eps)])
}

/// dx of the rms norm given dy, and dw summed over the rows added into dw
#[allow(clippy::too_many_arguments)]
pub fn rms_bwd(b: &mut CudaBackend, x: u64, w: u64, inv: u64, dy: u64, dx: u64, dw: u64, rows: usize, cols: usize) -> Result<()> {
    use Arg::{I, P};
    launch(b, "rms_bwd", (rows as u32, 1, 1), 256, vec![P(x), P(w), P(inv), P(dy), P(dx), I(cols as i32)])?;
    launch(b, "rms_bwd_w", (grid_for(cols), 1, 1), 256, vec![P(x), P(inv), P(dy), P(dw), I(rows as i32), I(cols as i32)])
}

/// One AdamW step over n parameters at step t
#[allow(clippy::too_many_arguments)]
pub fn adamw(b: &mut CudaBackend, p: u64, g: u64, m: u64, v: u64, n: usize, lr: f32, b1: f32, b2: f32, eps: f32, wd: f32, t: i32) -> Result<()> {
    use Arg::{F, I, L, P};
    launch(b, "adamw", (grid_for(n), 1, 1), 256, vec![P(p), P(g), P(m), P(v), L(n as i64), F(lr), F(b1), F(b2), F(eps), F(wd), I(t)])
}

/// y += alpha x over n values
pub fn axpy(b: &mut CudaBackend, x: u64, y: u64, n: usize, alpha: f32) -> Result<()> {
    use Arg::{F, L, P};
    launch(b, "axpy", (grid_for(n), 1, 1), 256, vec![P(x), P(y), L(n as i64), F(alpha)])
}

/// Rows of a single precision matrix gathered by index into dst
pub fn gather_rows(b: &mut CudaBackend, src: u64, idx: u64, dst: u64, rows: usize, cols: usize) -> Result<()> {
    use Arg::{I, P};
    launch(b, "gather_rows", (grid_for(rows * cols), 1, 1), 256, vec![P(src), P(idx), P(dst), I(rows as i32), I(cols as i32)])
}

/// Rows of a half precision matrix gathered by index into single precision dst
pub fn gather_rows_h(b: &mut CudaBackend, src: u64, idx: u64, dst: u64, rows: usize, cols: usize) -> Result<()> {
    use Arg::{I, P};
    launch(b, "gather_rows_h", (grid_for(rows * cols), 1, 1), 256, vec![P(src), P(idx), P(dst), I(rows as i32), I(cols as i32)])
}

/// Chunk length of the block's training attention, KF_TA_T of train.cu
pub const ATTN_T: usize = kf_core::DRAFT_CHUNK;

/// Ancestors a draft row may list, KF_DA_ANC of train.cu, a path of every tree row with the mask rows after it
pub const DRAFT_ANCESTORS: usize = 80;

/// Candidates a packed row holds for the soft target loss, KF_SOFT_K of train.cu
pub const SOFT_K: usize = kf_core::DRAFT_CANDS;

/// Shared memory of the chained attention kernels for a chain length and head size
fn chain_smem(hd: usize, chain: usize) -> u32 {
    ((2 * ATTN_T * hd + ATTN_T * (ATTN_T + chain)) * 4) as u32
}

/// Attention of one query group at drafted step `step` over b chunks of ATTN_T rows and h heads of hd, every row
/// sees the window keys kwin and vwin at or before it within the window, strictly before it when self_key is
/// false, and its own row at every chain step first to step, the chain keys and values of step m lying m times
/// stepstride floats into k and v, the probabilities kept as [b, h, T, T + chain] for the backward pass
#[allow(clippy::too_many_arguments)]
pub fn chain_attn_fwd(b: &mut CudaBackend, q: u64, k: u64, v: u64, stepstride: usize, step: usize, kwin: u64, vwin: u64, first: usize, self_key: bool, out: u64, probs: u64, batch: usize, h: usize, hd: usize, window: usize, scale: f32) -> Result<()> {
    use Arg::{F, I, L, P};
    ensure!(first <= step + 1, "a chain from step {first} to {step}");
    let smem = chain_smem(hd, step + 1 - first);
    let func = b.kernels.function_smem("train", "chain_attn_fwd", smem)?;
    Launch { func, grid: ((batch * h) as u32, 1, 1), block: 256, smem, args: vec![P(q), P(k), P(v), L(stepstride as i64), I(step as i32), P(kwin), P(vwin), I(first as i32), I(self_key as i32), P(out), P(probs), I(ATTN_T as i32), I(h as i32), I(hd as i32), I(window as i32), F(scale)] }.run(&b.stream)
}

/// Gradients of chain_attn_fwd given dout and the kept probabilities, the window keys' gradients added into dkwin
/// and dvwin and the chain rows' into every step's rows of dk and dv
#[allow(clippy::too_many_arguments)]
pub fn chain_attn_bwd(b: &mut CudaBackend, q: u64, k: u64, v: u64, stepstride: usize, step: usize, kwin: u64, vwin: u64, first: usize, probs: u64, dout: u64, dq: u64, dk: u64, dv: u64, dkwin: u64, dvwin: u64, batch: usize, h: usize, hd: usize, scale: f32) -> Result<()> {
    use Arg::{F, I, L, P};
    ensure!(first <= step + 1, "a chain from step {first} to {step}");
    let smem = chain_smem(hd, step + 1 - first);
    let func = b.kernels.function_smem("train", "chain_attn_bwd", smem)?;
    Launch { func, grid: ((batch * h) as u32, 1, 1), block: 256, smem, args: vec![P(q), P(k), P(v), L(stepstride as i64), I(step as i32), P(kwin), P(vwin), I(first as i32), P(probs), P(dout), P(dq), P(dk), P(dv), P(dkwin), P(dvwin), I(ATTN_T as i32), I(h as i32), I(hd as i32), F(scale)] }.run(&b.stream)
}

/// Attention of m draft rows over a slot's history ring of cap rows with their positions, the node rows their
/// ancestor lists name, and themselves, see draft_attn in train.cu
#[allow(clippy::too_many_arguments)]
pub fn draft_attn(b: &mut CudaBackend, q: u64, kring: u64, vring: u64, hpos: u64, cap: usize, knode: u64, vnode: u64, anc: u64, pos: u64, window: usize, out: u64, m: usize, h: usize, hd: usize, scale: f32) -> Result<()> {
    use Arg::{F, I, P};
    let smem = ((hd + cap + DRAFT_ANCESTORS) * 4) as u32;
    let func = b.kernels.function_smem("train", "draft_attn", smem)?;
    Launch { func, grid: ((m * h) as u32, 1, 1), block: 256, smem, args: vec![P(q), P(kring), P(vring), P(hpos), I(cap as i32), P(knode), P(vnode), P(anc), P(pos), I(window as i32), P(out), I(h as i32), I(hd as i32), F(scale)] }.run(&b.stream)
}

/// Soft target cross entropy of rows of v logits against packed candidate rows through a vocabulary lookup,
/// weighted per row, every row's own loss into losses, the weighted loss added into loss, the rows with a target
/// into counted, and the weight times softmax minus target into dz
#[allow(clippy::too_many_arguments)]
pub fn soft_ce(b: &mut CudaBackend, z: u64, v: usize, cands: u64, lut: u64, w: u64, dz: u64, losses: u64, loss: u64, counted: u64, rows: usize) -> Result<()> {
    use Arg::{I, P};
    launch(b, "soft_ce", (rows as u32, 1, 1), 256, vec![P(z), I(v as i32), P(cands), I(SOFT_K as i32), P(lut), P(w), P(dz), P(losses), P(loss), P(counted)])
}

/// Acceptance and rejection cost of every row's drafted position against its packed candidates, the end column
/// left out of the cost, see edr_terms in train.cu
#[allow(clippy::too_many_arguments)]
pub fn edr_terms(b: &mut CudaBackend, z: u64, v: usize, cands: u64, lut: u64, eos: i32, a: u64, c: u64, rows: usize) -> Result<()> {
    use Arg::{I, P};
    launch(b, "edr_terms", (rows as u32, 1, 1), 256, vec![P(z), I(v as i32), P(cands), I(SOFT_K as i32), P(lut), I(eos), P(a), P(c)])
}

/// Gradient of every row's share of the expected rounds into dz, scaled by w, given its occupancy and the rounds
/// its acceptance saves
#[allow(clippy::too_many_arguments)]
pub fn edr_grad(b: &mut CudaBackend, z: u64, v: usize, cands: u64, lut: u64, eos: i32, omega: u64, dv: u64, w: u64, dz: u64, rows: usize) -> Result<()> {
    use Arg::{I, P};
    launch(b, "edr_grad", (rows as u32, 1, 1), 256, vec![P(z), I(v as i32), P(cands), I(SOFT_K as i32), P(lut), I(eos), P(omega), P(dv), P(w), P(dz)])
}

/// The block verification loss of every row's depth drafted positions, the positions' logits step_stride floats
/// apart and their candidates cand_stride apart, the path columns truth_stride apart, the gradient times w added
/// into dz and the loss into losses
#[allow(clippy::too_many_arguments)]
pub fn bv_row(b: &mut CudaBackend, z: u64, v: usize, step_stride: usize, cands: u64, cand_stride: usize, truth: u64, truth_stride: usize, lut: u64, depth: usize, w: u64, dz: u64, losses: u64, rows: usize) -> Result<()> {
    use Arg::{I, L, P};
    ensure!(depth <= kf_core::DRAFT_DEPTH, "{depth} drafted positions past {}", kf_core::DRAFT_DEPTH);
    launch(b, "bv_row", (rows as u32, 1, 1), 256, vec![P(z), I(v as i32), L(step_stride as i64), P(cands), I(SOFT_K as i32), L(cand_stride as i64), P(truth), I(truth_stride as i32), P(lut), I(depth as i32), P(w), P(dz), P(losses)])
}

/// One optimistic AdamW step over n parameters at step t, u holding the last direction
#[allow(clippy::too_many_arguments)]
pub fn adamw_opt(b: &mut CudaBackend, p: u64, g: u64, m: u64, v: u64, u: u64, n: usize, lr: f32, b1: f32, b2: f32, eps: f32, wd: f32, t: i32) -> Result<()> {
    use Arg::{F, I, L, P};
    launch(b, "adamw_opt", (grid_for(n), 1, 1), 256, vec![P(p), P(g), P(m), P(v), P(u), L(n as i64), F(lr), F(b1), F(b2), F(eps), F(wd), I(t)])
}

/// out = the weighted sum of up to four buffers of n values
pub fn mix4(b: &mut CudaBackend, out: u64, srcs: &[u64], weights: &[f32], n: usize) -> Result<()> {
    use Arg::{F, L, P};
    ensure!(!srcs.is_empty() && srcs.len() <= 4 && srcs.len() == weights.len(), "{} buffers to mix", srcs.len());
    let at = |i: usize| srcs.get(i).copied().unwrap_or(0);
    let wt = |i: usize| weights.get(i).copied().unwrap_or(0.0);
    launch(b, "mix4", (grid_for(n), 1, 1), 256, vec![P(out), P(at(0)), P(at(1)), P(at(2)), P(at(3)), F(wt(0)), F(wt(1)), F(wt(2)), F(wt(3)), L(n as i64)])
}

/// y = a y + (1 - a) x over n values
pub fn ema(b: &mut CudaBackend, y: u64, x: u64, n: usize, a: f32) -> Result<()> {
    use Arg::{F, L, P};
    launch(b, "ema", (grid_for(n), 1, 1), 256, vec![P(y), P(x), L(n as i64), F(a)])
}

/// The columns of a row major k by n matrix as n q8_0 rows of k into dst
pub fn quant_q8_0_t(b: &mut CudaBackend, src: u64, dst: u64, k: usize, n: usize) -> Result<()> {
    use Arg::{I, P};
    ensure!(k.is_multiple_of(32), "{k} inputs for int8 rows of 32");
    launch(b, "quant_q8_0_t", (n.div_ceil(128) as u32, (k / 32) as u32, 1), 128, vec![P(src), P(dst), I(k as i32), I(n as i32)])
}

/// Predictive entropy of every row of v logits into out
pub fn row_entropy(b: &mut CudaBackend, z: u64, v: usize, out: u64, rows: usize) -> Result<()> {
    use Arg::{I, P};
    launch(b, "row_entropy", (rows as u32, 1, 1), 256, vec![P(z), I(v as i32), P(out)])
}

/// Adds n_sel rows of src, those idx names in order, into the block rows of z from slot off
#[allow(clippy::too_many_arguments)]
pub fn cal_accum(b: &mut CudaBackend, src: u64, idx: u64, n_sel: usize, off: usize, block: usize, z: u64, d: usize) -> Result<()> {
    use Arg::{I, P};
    launch(b, "cal_accum", (grid_for(d), 1, 1), 256, vec![P(src), P(idx), I(n_sel as i32), I(off as i32), I(block as i32), P(z), I(d as i32)])
}

/// out[c] += the sum over rows of x[r][c]
pub fn col_sum(b: &mut CudaBackend, x: u64, rows: usize, cols: usize, out: u64) -> Result<()> {
    use Arg::{I, P};
    launch(b, "col_sum", (grid_for(cols), 1, 1), 256, vec![P(x), I(rows as i32), I(cols as i32), P(out)])
}

/// dst[r][off + c] = src[r][c] for rows of cols, dst rows stride floats apart
pub fn copy_cols(b: &mut CudaBackend, src: u64, dst: u64, rows: usize, cols: usize, stride: usize, off: usize) -> Result<()> {
    use Arg::{I, L, P};
    launch(b, "copy_cols", (grid_for(rows * cols), 1, 1), 256, vec![P(src), P(dst), I(rows as i32), I(cols as i32), L(stride as i64), I(off as i32)])
}

/// Copies n node rows' keys and values into a slot's position ring at the positions given
#[allow(clippy::too_many_arguments)]
pub fn ring_write(b: &mut CudaBackend, k: u64, v: u64, rows: u64, pos: u64, n: usize, cap: usize, d: usize, kring: u64, vring: u64, hpos: u64) -> Result<()> {
    use Arg::{I, P};
    launch(b, "ring_write", (grid_for(n * d), 1, 1), 256, vec![P(k), P(v), P(rows), P(pos), I(n as i32), I(cap as i32), I(d as i32), P(kring), P(vring), P(hpos)])
}

/// Rows of bytes gathered by index
pub fn gather_bytes(b: &mut CudaBackend, src: u64, idx: u64, dst: u64, rows: usize, row_bytes: usize) -> Result<()> {
    use Arg::{I, L, P};
    launch(b, "gather_bytes", (grid_for(rows * row_bytes).min(4096), 1, 1), 256, vec![P(src), P(idx), P(dst), I(rows as i32), L(row_bytes as i64)])
}

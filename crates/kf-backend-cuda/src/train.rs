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

/// The mean smooth L1 loss added into loss, one float, and the gradient of the mean into grad
pub fn smooth_l1(b: &mut CudaBackend, pred: u64, target: u64, grad: u64, loss: u64, n: usize, beta: f32) -> Result<()> {
    use Arg::{F, L, P};
    launch(b, "smooth_l1", (grid_for(n).min(1024), 1, 1), 256, vec![P(pred), P(target), P(grad), P(loss), L(n as i64), F(beta)])
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

/// Chunk length of the head's attention, KF_TA_T of train.cu
pub const ATTN_T: usize = kf_core::HEAD_CHUNK;

/// Shared memory of the head's attention kernels for a head size
fn attn_smem(hd: usize) -> u32 {
    ((2 * ATTN_T * hd + ATTN_T * ATTN_T) * 4) as u32
}

/// Attention of the head over b chunks of ATTN_T positions and h heads of hd, every position seeing the keys at or before it within
/// the window, the probabilities kept for the backward pass
#[allow(clippy::too_many_arguments)]
pub fn head_attn_fwd(b: &mut CudaBackend, q: u64, k: u64, v: u64, out: u64, probs: u64, batch: usize, h: usize, hd: usize, window: usize, scale: f32) -> Result<()> {
    use Arg::{F, I, P};
    let smem = attn_smem(hd);
    let func = b.kernels.function_smem("train", "head_attn_fwd", smem)?;
    Launch { func, grid: ((batch * h) as u32, 1, 1), block: 256, smem, args: vec![P(q), P(k), P(v), P(out), P(probs), I(ATTN_T as i32), I(h as i32), I(hd as i32), I(window as i32), F(scale)] }.run(&b.stream)
}

/// Gradients of the head's attention given dout and the kept probabilities
#[allow(clippy::too_many_arguments)]
pub fn head_attn_bwd(b: &mut CudaBackend, q: u64, k: u64, v: u64, probs: u64, dout: u64, dq: u64, dk: u64, dv: u64, batch: usize, h: usize, hd: usize, scale: f32) -> Result<()> {
    use Arg::{F, I, P};
    let smem = attn_smem(hd);
    let func = b.kernels.function_smem("train", "head_attn_bwd", smem)?;
    Launch { func, grid: ((batch * h) as u32, 1, 1), block: 256, smem, args: vec![P(q), P(k), P(v), P(probs), P(dout), P(dq), P(dk), P(dv), I(ATTN_T as i32), I(h as i32), I(hd as i32), F(scale)] }.run(&b.stream)
}

/// Ancestors a draft row may list, KF_TD_ANC of train.cu
pub const DRAFT_ANCESTORS: usize = 8;

/// Attention of m draft rows over their slot's window of nwin cached keys and values and their listed ancestors
#[allow(clippy::too_many_arguments)]
pub fn head_attn_draft(b: &mut CudaBackend, q: u64, kwin: u64, vwin: u64, nwin: usize, kanc: u64, vanc: u64, anc: u64, out: u64, m: usize, h: usize, hd: usize, scale: f32) -> Result<()> {
    use Arg::{F, I, P};
    let smem = ((hd + nwin + DRAFT_ANCESTORS + 1) * 4) as u32;
    let func = b.kernels.function_smem("train", "head_attn_draft", smem)?;
    Launch { func, grid: ((m * h) as u32, 1, 1), block: 256, smem, args: vec![P(q), P(kwin), P(vwin), I(nwin as i32), P(kanc), P(vanc), P(anc), P(out), I(h as i32), I(hd as i32), F(scale)] }.run(&b.stream)
}

/// Rows of bytes gathered by index
pub fn gather_bytes(b: &mut CudaBackend, src: u64, idx: u64, dst: u64, rows: usize, row_bytes: usize) -> Result<()> {
    use Arg::{I, L, P};
    launch(b, "gather_bytes", (grid_for(rows * row_bytes).min(4096), 1, 1), 256, vec![P(src), P(idx), P(dst), I(rows as i32), L(row_bytes as i64)])
}

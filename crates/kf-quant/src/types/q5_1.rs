use crate::types::q5_0::qh_bytes;
use crate::types::{f16_at, u32_at};
use crate::{ActI8, Quant, TypeInfo};

pub static INFO: TypeInfo = TypeInfo { id: 7, name: "q5_1", block: 32, size: 24 };
pub struct Q5_1;

/// Rows r0 on of the activations against one row, adding the min times each block's activation sum separately
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2<const R: usize>(src: &[u8], x: &ActI8, r0: usize, y: &mut [f32]) {
    use crate::simd::hsum_ps;
    use std::arch::x86_64::*;
    let (n, n16) = (x.n, x.n / 16);
    let m4 = _mm_set1_epi8(0xf);
    let bit4 = _mm256_set1_epi8(0x10);
    let ones = _mm256_set1_epi16(1);
    let mut acc = [_mm256_setzero_ps(); R];
    let mut accm = [0.0f32; R];
    for (b, blk) in src.chunks_exact(24).enumerate() {
        let (dw, mw) = (f16_at(blk, 0), f16_at(blk, 2));
        let q = unsafe { _mm_loadu_si128(blk.as_ptr().add(8) as *const __m128i) };
        let nib = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(q, 4), m4), _mm_and_si128(q, m4));
        let w = _mm256_or_si256(nib, _mm256_and_si256(qh_bytes(u32_at(blk, 4)), bit4));
        for r in 0..R {
            let at = (r0 + r) * n16 + b * 2;
            let q8 = unsafe { _mm256_loadu_si256(x.q.as_ptr().add((r0 + r) * n + b * 32) as *const __m256i) };
            let dot = _mm256_madd_epi16(_mm256_maddubs_epi16(w, q8), ones);
            acc[r] = _mm256_fmadd_ps(_mm256_set1_ps(dw * x.d[at]), _mm256_cvtepi32_ps(dot), acc[r]);
            accm[r] += mw * x.d[at] * (x.sum[at] + x.sum[at + 1]) as f32;
        }
    }
    for r in 0..R {
        y[r0 + r] = hsum_ps(acc[r]) + accm[r];
    }
}

impl Quant for Q5_1 {
    fn info(&self) -> &'static TypeInfo {
        &INFO
    }
    fn dequant_row(&self, src: &[u8], dst: &mut [f32]) {
        for (b, out) in dst.chunks_exact_mut(32).enumerate() {
            let blk = &src[b * 24..(b + 1) * 24];
            let d = f16_at(blk, 0);
            let m = f16_at(blk, 2);
            let qh = u32_at(blk, 4);
            for j in 0..16 {
                let xh0 = ((qh >> j) << 4) & 0x10;
                let xh1 = (qh >> (j + 12)) & 0x10;
                let x0 = (blk[8 + j] & 0x0F) as u32 | xh0;
                let x1 = (blk[8 + j] >> 4) as u32 | xh1;
                out[j] = x0 as f32 * d + m;
                out[j + 16] = x1 as f32 * d + m;
            }
        }
    }
    fn dot_i8(&self, src: &[u8], x: &ActI8, y: &mut [f32]) {
        #[cfg(target_arch = "x86_64")]
        if crate::simd::have() {
            return crate::simd::avx2_rows!(dot_avx2, src, x, y);
        }
        crate::dot_i8_generic(self, src, x, y)
    }
}

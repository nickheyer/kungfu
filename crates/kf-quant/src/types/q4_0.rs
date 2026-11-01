use crate::types::f16_at;
use crate::{ActI8, Quant, TypeInfo};

pub static INFO: TypeInfo = TypeInfo { id: 2, name: "q4_0", block: 32, size: 18 };
pub struct Q4_0;

/// Rows r0 on of the activations against one row, each block's nibbles offset by eight once for every row
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2<const R: usize>(src: &[u8], x: &ActI8, r0: usize, y: &mut [f32]) {
    use crate::simd::hsum_ps;
    use std::arch::x86_64::*;
    let (n, n16) = (x.n, x.n / 16);
    let m4 = _mm_set1_epi8(0xf);
    let eight = _mm256_set1_epi8(8);
    let ones = _mm256_set1_epi16(1);
    let mut acc = [_mm256_setzero_ps(); R];
    for (b, blk) in src.chunks_exact(18).enumerate() {
        let dw = f16_at(blk, 0);
        let q = unsafe { _mm_loadu_si128(blk.as_ptr().add(2) as *const __m128i) };
        let nib = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(q, 4), m4), _mm_and_si128(q, m4));
        let w = _mm256_sub_epi8(nib, eight);
        let wabs = _mm256_sign_epi8(w, w);
        for r in 0..R {
            let q8 = unsafe { _mm256_loadu_si256(x.q.as_ptr().add((r0 + r) * n + b * 32) as *const __m256i) };
            let dot = _mm256_madd_epi16(_mm256_maddubs_epi16(wabs, _mm256_sign_epi8(q8, w)), ones);
            acc[r] = _mm256_fmadd_ps(_mm256_set1_ps(dw * x.d[(r0 + r) * n16 + b * 2]), _mm256_cvtepi32_ps(dot), acc[r]);
        }
    }
    for r in 0..R {
        y[r0 + r] = hsum_ps(acc[r]);
    }
}

impl Quant for Q4_0 {
    fn info(&self) -> &'static TypeInfo {
        &INFO
    }
    fn dequant_row(&self, src: &[u8], dst: &mut [f32]) {
        for (b, out) in dst.chunks_exact_mut(32).enumerate() {
            let blk = &src[b * 18..(b + 1) * 18];
            let d = f16_at(blk, 0);
            for j in 0..16 {
                let q = blk[2 + j];
                out[j] = ((q & 0x0F) as i32 - 8) as f32 * d;
                out[j + 16] = ((q >> 4) as i32 - 8) as f32 * d;
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

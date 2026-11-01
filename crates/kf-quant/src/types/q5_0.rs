use crate::types::{f16_at, u32_at};
use crate::{ActI8, Quant, TypeInfo};

pub static INFO: TypeInfo = TypeInfo { id: 6, name: "q5_0", block: 32, size: 22 };
pub struct Q5_0;

/// Spreads the thirty two bits of qh into bytes, 0xff where the bit is set and zero otherwise
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) fn qh_bytes(qh: u32) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;
    let shuf = _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101, 0);
    let bytes = _mm256_shuffle_epi8(_mm256_set1_epi32(qh as i32), shuf);
    let bits = _mm256_or_si256(bytes, _mm256_set1_epi64x(0x7fbfdfeff7fbfdfe));
    _mm256_cmpeq_epi8(bits, _mm256_set1_epi8(-1))
}

/// Rows r0 on of the activations against one row, each block's five bit weights unpacked once for every row
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2<const R: usize>(src: &[u8], x: &ActI8, r0: usize, y: &mut [f32]) {
    use crate::simd::hsum_ps;
    use std::arch::x86_64::*;
    let (n, n16) = (x.n, x.n / 16);
    let m4 = _mm_set1_epi8(0xf);
    let high = _mm256_set1_epi8(0xf0u8 as i8);
    let ones = _mm256_set1_epi16(1);
    let mut acc = [_mm256_setzero_ps(); R];
    for (b, blk) in src.chunks_exact(22).enumerate() {
        let dw = f16_at(blk, 0);
        let q = unsafe { _mm_loadu_si128(blk.as_ptr().add(6) as *const __m128i) };
        let nib = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(q, 4), m4), _mm_and_si128(q, m4));
        let w = _mm256_or_si256(nib, _mm256_andnot_si256(qh_bytes(u32_at(blk, 2)), high));
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

impl Quant for Q5_0 {
    fn info(&self) -> &'static TypeInfo {
        &INFO
    }
    fn dequant_row(&self, src: &[u8], dst: &mut [f32]) {
        for (b, out) in dst.chunks_exact_mut(32).enumerate() {
            let blk = &src[b * 22..(b + 1) * 22];
            let d = f16_at(blk, 0);
            let qh = u32_at(blk, 2);
            for j in 0..16 {
                let xh0 = ((qh >> j) << 4) & 0x10;
                let xh1 = (qh >> (j + 12)) & 0x10;
                let x0 = ((blk[6 + j] & 0x0F) as u32 | xh0) as i32 - 16;
                let x1 = ((blk[6 + j] >> 4) as u32 | xh1) as i32 - 16;
                out[j] = x0 as f32 * d;
                out[j + 16] = x1 as f32 * d;
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

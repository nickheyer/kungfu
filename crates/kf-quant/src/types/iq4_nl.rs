use crate::types::f16_at;
use crate::types::grids::KVALUES_IQ4NL;
use crate::{ActI8, Quant, TypeInfo};

pub static INFO: TypeInfo = TypeInfo { id: 20, name: "iq4_nl", block: 32, size: 18 };
pub struct Iq4Nl;

/// Rows r0 on of the activations against one row, nibbles mapped through the value table once for every row
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2<const R: usize>(src: &[u8], x: &ActI8, r0: usize, y: &mut [f32]) {
    use crate::simd::hsum_ps;
    use std::arch::x86_64::*;
    let (n, n16) = (x.n, x.n / 16);
    let kv = unsafe { _mm_loadu_si128(KVALUES_IQ4NL.as_ptr() as *const __m128i) };
    let m4 = _mm_set1_epi8(0xf);
    let ones = _mm256_set1_epi16(1);
    let mut acc = [_mm256_setzero_ps(); R];
    for (b, blk) in src.chunks_exact(18).enumerate() {
        let dw = f16_at(blk, 0);
        let q = unsafe { _mm_loadu_si128(blk.as_ptr().add(2) as *const __m128i) };
        let lo = _mm_shuffle_epi8(kv, _mm_and_si128(q, m4));
        let hi = _mm_shuffle_epi8(kv, _mm_and_si128(_mm_srli_epi16(q, 4), m4));
        let w = _mm256_set_m128i(hi, lo);
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

impl Quant for Iq4Nl {
    fn info(&self) -> &'static TypeInfo {
        &INFO
    }
    fn dequant_row(&self, src: &[u8], dst: &mut [f32]) {
        for (b, out) in dst.chunks_exact_mut(32).enumerate() {
            let blk = &src[b * 18..(b + 1) * 18];
            let d = f16_at(blk, 0);
            let (lo, hi) = out.split_at_mut(16);
            for ((y0, y1), &q) in lo.iter_mut().zip(hi.iter_mut()).zip(&blk[2..18]) {
                *y0 = d * KVALUES_IQ4NL[(q & 0xf) as usize] as f32;
                *y1 = d * KVALUES_IQ4NL[(q >> 4) as usize] as f32;
            }
        }
    }
    fn dot_row(&self, src: &[u8], x: &[f32]) -> f32 {
        let mut acc = 0.0f32;
        for (b, xs) in x.chunks_exact(32).enumerate() {
            let blk = &src[b * 18..(b + 1) * 18];
            let d = f16_at(blk, 0);
            let (x0, x1) = xs.split_at(16);
            let mut s = 0.0f32;
            for ((&a, &c), &q) in x0.iter().zip(x1).zip(&blk[2..18]) {
                s += KVALUES_IQ4NL[(q & 0xf) as usize] as f32 * a;
                s += KVALUES_IQ4NL[(q >> 4) as usize] as f32 * c;
            }
            acc += d * s;
        }
        acc
    }
    fn dot_i8(&self, src: &[u8], x: &ActI8, y: &mut [f32]) {
        #[cfg(target_arch = "x86_64")]
        if crate::simd::have() {
            return crate::simd::avx2_rows!(dot_avx2, src, x, y);
        }
        crate::dot_i8_generic(self, src, x, y)
    }
}

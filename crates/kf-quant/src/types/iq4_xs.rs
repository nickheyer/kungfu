use crate::types::grids::KVALUES_IQ4NL;
use crate::types::{f16_at, u16_at};
use crate::{ActI8, Quant, TypeInfo};

pub static INFO: TypeInfo = TypeInfo { id: 23, name: "iq4_xs", block: 256, size: 136 };
pub struct Iq4Xs;

/// Signed 6 bit scale of sub block ib
fn sub_scale(blk: &[u8], ib: usize) -> i32 {
    let scales_h = u16_at(blk, 2) as i32;
    let lo = ((blk[4 + ib / 2] >> (4 * (ib % 2))) & 0xf) as i32;
    (lo | (((scales_h >> (2 * ib)) & 3) << 4)) - 32
}

/// Rows r0 on of the activations against one row, each thirty two of nibbles mapped through the value table once for every row
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2<const R: usize>(src: &[u8], x: &ActI8, r0: usize, y: &mut [f32]) {
    use crate::simd::hsum_ps;
    use std::arch::x86_64::*;
    let (n, n16) = (x.n, x.n / 16);
    let kv = unsafe { _mm_loadu_si128(KVALUES_IQ4NL.as_ptr() as *const __m128i) };
    let m4 = _mm_set1_epi8(0xf);
    let mut acc = [_mm256_setzero_ps(); R];
    for (b, blk) in src.chunks_exact(136).enumerate() {
        let dw = f16_at(blk, 0);
        let mut sumi = [_mm256_setzero_si256(); R];
        for ib in 0..8 {
            let q = unsafe { _mm_loadu_si128(blk.as_ptr().add(8 + 16 * ib) as *const __m128i) };
            let lo = _mm_shuffle_epi8(kv, _mm_and_si128(q, m4));
            let hi = _mm_shuffle_epi8(kv, _mm_and_si128(_mm_srli_epi16(q, 4), m4));
            let w = _mm256_set_m128i(hi, lo);
            let wabs = _mm256_sign_epi8(w, w);
            let scale = _mm256_set1_epi16(sub_scale(blk, ib) as i16);
            for r in 0..R {
                let q8 = unsafe { _mm256_loadu_si256(x.q.as_ptr().add((r0 + r) * n + b * 256 + ib * 32) as *const __m256i) };
                let dot = _mm256_maddubs_epi16(wabs, _mm256_sign_epi8(q8, w));
                sumi[r] = _mm256_add_epi32(sumi[r], _mm256_madd_epi16(dot, scale));
            }
        }
        for r in 0..R {
            acc[r] = _mm256_fmadd_ps(_mm256_set1_ps(dw * x.d[(r0 + r) * n16 + b * 16]), _mm256_cvtepi32_ps(sumi[r]), acc[r]);
        }
    }
    for r in 0..R {
        y[r0 + r] = hsum_ps(acc[r]);
    }
}

impl Quant for Iq4Xs {
    fn info(&self) -> &'static TypeInfo {
        &INFO
    }
    fn dequant_row(&self, src: &[u8], dst: &mut [f32]) {
        for (b, out) in dst.chunks_exact_mut(256).enumerate() {
            let blk = &src[b * 136..(b + 1) * 136];
            let d = f16_at(blk, 0);
            for (ib, ys) in out.chunks_exact_mut(32).enumerate() {
                let dl = d * sub_scale(blk, ib) as f32;
                let qs = &blk[8 + 16 * ib..24 + 16 * ib];
                let (lo, hi) = ys.split_at_mut(16);
                for ((y0, y1), &q) in lo.iter_mut().zip(hi.iter_mut()).zip(qs) {
                    *y0 = dl * KVALUES_IQ4NL[(q & 0xf) as usize] as f32;
                    *y1 = dl * KVALUES_IQ4NL[(q >> 4) as usize] as f32;
                }
            }
        }
    }
    fn dot_row(&self, src: &[u8], x: &[f32]) -> f32 {
        let mut acc = 0.0f32;
        for (b, xs) in x.chunks_exact(256).enumerate() {
            let blk = &src[b * 136..(b + 1) * 136];
            let d = f16_at(blk, 0);
            for (ib, xb) in xs.chunks_exact(32).enumerate() {
                let qs = &blk[8 + 16 * ib..24 + 16 * ib];
                let (x0, x1) = xb.split_at(16);
                let mut s = 0.0f32;
                for ((&a, &c), &q) in x0.iter().zip(x1).zip(qs) {
                    s += KVALUES_IQ4NL[(q & 0xf) as usize] as f32 * a;
                    s += KVALUES_IQ4NL[(q >> 4) as usize] as f32 * c;
                }
                acc += d * sub_scale(blk, ib) as f32 * s;
            }
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

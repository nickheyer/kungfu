use crate::types::f16_at;
use crate::{ActI8, Quant, TypeInfo};

pub static INFO: TypeInfo = TypeInfo { id: 10, name: "q2_K", block: 256, size: 84 };
pub struct Q2K;

/// Rows r0 on of the activations against one row, each half's two bit quants unpacked once for every row
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2<const R: usize>(src: &[u8], x: &ActI8, r0: usize, y: &mut [f32]) {
    use crate::simd::hsum_ps;
    use std::arch::x86_64::*;
    let (n, n16) = (x.n, x.n / 16);
    let m3 = _mm256_set1_epi8(3);
    let mut acc = [_mm256_setzero_ps(); R];
    for (b, blk) in src.chunks_exact(84).enumerate() {
        let d = f16_at(blk, 80);
        let dmin = f16_at(blk, 82);
        let sm = unsafe { _mm_loadu_si128(blk.as_ptr() as *const __m128i) };
        let mins = _mm_and_si128(_mm_srli_epi16(sm, 4), _mm_set1_epi8(0xf));
        let mlo = _mm256_cvtepu8_epi32(mins);
        let mhi = _mm256_cvtepu8_epi32(_mm_srli_si128(mins, 8));
        let mut sumi = [_mm256_setzero_si256(); R];
        for h in 0..2 {
            let mut qv = unsafe { _mm256_loadu_si256(blk.as_ptr().add(16 + 32 * h) as *const __m256i) };
            for k in 0..4 {
                let g = 8 * h + 2 * k;
                let w = _mm256_and_si256(qv, m3);
                qv = _mm256_srli_epi16(qv, 2);
                let scale = _mm256_set_m128i(_mm_set1_epi16((blk[g + 1] & 0xf) as i16), _mm_set1_epi16((blk[g] & 0xf) as i16));
                for r in 0..R {
                    let q8 = unsafe { _mm256_loadu_si256(x.q.as_ptr().add((r0 + r) * n + b * 256 + 16 * g) as *const __m256i) };
                    sumi[r] = _mm256_add_epi32(sumi[r], _mm256_madd_epi16(_mm256_maddubs_epi16(w, q8), scale));
                }
            }
        }
        for r in 0..R {
            let at = (r0 + r) * n16 + b * 16;
            let s0 = unsafe { _mm256_loadu_si256(x.sum.as_ptr().add(at) as *const __m256i) };
            let s1 = unsafe { _mm256_loadu_si256(x.sum.as_ptr().add(at + 8) as *const __m256i) };
            let summ = _mm256_add_epi32(_mm256_mullo_epi32(s0, mlo), _mm256_mullo_epi32(s1, mhi));
            let dx = x.d[at];
            acc[r] = _mm256_fmadd_ps(_mm256_set1_ps(d * dx), _mm256_cvtepi32_ps(sumi[r]), acc[r]);
            acc[r] = _mm256_fnmadd_ps(_mm256_set1_ps(dmin * dx), _mm256_cvtepi32_ps(summ), acc[r]);
        }
    }
    for r in 0..R {
        y[r0 + r] = hsum_ps(acc[r]);
    }
}

impl Quant for Q2K {
    fn info(&self) -> &'static TypeInfo {
        &INFO
    }
    fn dequant_row(&self, src: &[u8], dst: &mut [f32]) {
        for (b, out) in dst.chunks_exact_mut(256).enumerate() {
            let blk = &src[b * 84..(b + 1) * 84];
            let d = f16_at(blk, 80);
            let min = f16_at(blk, 82);
            let mut y = 0;
            let mut is = 0;
            for n in 0..2 {
                let q = &blk[16 + 32 * n..48 + 32 * n];
                for shift in [0, 2, 4, 6] {
                    for part in q.chunks_exact(16) {
                        let sc = blk[is];
                        is += 1;
                        let dl = d * (sc & 0xF) as f32;
                        let ml = min * (sc >> 4) as f32;
                        for &v in part {
                            out[y] = dl * ((v >> shift) & 3) as f32 - ml;
                            y += 1;
                        }
                    }
                }
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

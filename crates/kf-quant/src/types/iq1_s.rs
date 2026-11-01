use crate::types::grids::IQ1S_GRID;
use crate::types::{IQ1S_DELTA, f16_at, u16_at};
use crate::{ActI8, Quant, TypeInfo};

pub static INFO: TypeInfo = TypeInfo { id: 19, name: "iq1_s", block: 256, size: 50 };
pub struct Iq1S;

/// Rows r0 on of the activations against one row, weights held as eight times grid value plus delta sign
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2<const R: usize>(src: &[u8], x: &ActI8, r0: usize, y: &mut [f32]) {
    use crate::simd::hsum_ps;
    use std::arch::x86_64::*;
    let (n, n16) = (x.n, x.n / 16);
    let eight = _mm256_set1_epi8(8);
    let mut acc = [_mm256_setzero_ps(); R];
    for (b, blk) in src.chunks_exact(50).enumerate() {
        let dw = f16_at(blk, 0);
        let mut sumi = [_mm256_setzero_si256(); R];
        for ib in 0..8 {
            let qs = &blk[2 + 4 * ib..6 + 4 * ib];
            let qh = u16_at(blk, 34 + 2 * ib) as usize;
            let grid = |l: usize| IQ1S_GRID[qs[l] as usize | (((qh >> (3 * l)) & 7) << 8)] as i64;
            let g = _mm256_set_epi64x(grid(3), grid(2), grid(1), grid(0));
            let delta = _mm256_set1_epi8(if qh & 0x8000 != 0 { -1 } else { 1 });
            let w = _mm256_add_epi8(_mm256_sign_epi8(eight, g), delta);
            let wabs = _mm256_sign_epi8(w, w);
            let scale = _mm256_set1_epi16((2 * ((qh >> 12) & 7) + 1) as i16);
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
        y[r0 + r] = 0.125 * hsum_ps(acc[r]);
    }
}

impl Quant for Iq1S {
    fn info(&self) -> &'static TypeInfo {
        &INFO
    }
    fn dequant_row(&self, src: &[u8], dst: &mut [f32]) {
        for (b, out) in dst.chunks_exact_mut(256).enumerate() {
            let blk = &src[b * 50..(b + 1) * 50];
            let d = f16_at(blk, 0);
            for (ib, ys) in out.chunks_exact_mut(32).enumerate() {
                let qs = &blk[2 + 4 * ib..6 + 4 * ib];
                let qh = u16_at(blk, 34 + 2 * ib) as usize;
                let dl = d * (2 * ((qh >> 12) & 7) + 1) as f32;
                let delta = if qh & 0x8000 != 0 { -IQ1S_DELTA } else { IQ1S_DELTA };
                for (l, y8) in ys.chunks_exact_mut(8).enumerate() {
                    let grid =
                        IQ1S_GRID[qs[l] as usize | (((qh >> (3 * l)) & 7) << 8)].to_le_bytes();
                    for (y, &g) in y8.iter_mut().zip(&grid) {
                        *y = dl * ((g as i8) as f32 + delta);
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

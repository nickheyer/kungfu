use crate::types::grids::{IQ2S_GRID, KMASK_IQ2XS};
use crate::types::{f16_at, signed};
use crate::{ActI8, Quant, TypeInfo};

pub static INFO: TypeInfo = TypeInfo { id: 22, name: "iq2_s", block: 256, size: 82 };
pub struct Iq2S;

/// Rows r0 on of the activations against one row, the grid and sign words of each thirty two looked up once for every row
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2<const R: usize>(src: &[u8], x: &ActI8, r0: usize, y: &mut [f32]) {
    use crate::simd::{MASK_SIGNS, hsum_ps};
    use std::arch::x86_64::*;
    let (n, n16) = (x.n, x.n / 16);
    let mut acc = [_mm256_setzero_ps(); R];
    for (b, blk) in src.chunks_exact(82).enumerate() {
        let dw = f16_at(blk, 0);
        let (qs, signs, qh, scales) = (&blk[2..34], &blk[34..66], &blk[66..74], &blk[74..82]);
        let mut sumi = [_mm256_setzero_si256(); R];
        for ib32 in 0..8 {
            let h = qh[ib32] as usize;
            let idx = |l: usize| qs[4 * ib32 + l] as usize | ((h << (8 - 2 * l)) & 0x300);
            let g = _mm256_set_epi64x(IQ2S_GRID[idx(3)] as i64, IQ2S_GRID[idx(2)] as i64, IQ2S_GRID[idx(1)] as i64, IQ2S_GRID[idx(0)] as i64);
            let sg = &signs[4 * ib32..4 * ib32 + 4];
            let s = _mm256_set_epi64x(MASK_SIGNS[sg[3] as usize] as i64, MASK_SIGNS[sg[2] as usize] as i64, MASK_SIGNS[sg[1] as usize] as i64, MASK_SIGNS[sg[0] as usize] as i64);
            let sc = scales[ib32];
            let scale = _mm256_set_m128i(_mm_set1_epi16(2 * (sc >> 4) as i16 + 1), _mm_set1_epi16(2 * (sc & 0xf) as i16 + 1));
            for r in 0..R {
                let q8 = unsafe { _mm256_loadu_si256(x.q.as_ptr().add((r0 + r) * n + b * 256 + ib32 * 32) as *const __m256i) };
                let dot = _mm256_maddubs_epi16(g, _mm256_sign_epi8(q8, s));
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

impl Quant for Iq2S {
    fn info(&self) -> &'static TypeInfo {
        &INFO
    }
    fn dequant_row(&self, src: &[u8], dst: &mut [f32]) {
        for (b, out) in dst.chunks_exact_mut(256).enumerate() {
            let blk = &src[b * 82..(b + 1) * 82];
            let d = f16_at(blk, 0);
            for (ib32, ys) in out.chunks_exact_mut(32).enumerate() {
                let qs = &blk[2 + 4 * ib32..6 + 4 * ib32];
                let signs = &blk[34 + 4 * ib32..38 + 4 * ib32];
                let qh = blk[66 + ib32] as usize;
                let sc = blk[74 + ib32];
                let db = [d * (0.5 + (sc & 0xf) as f32) * 0.25, d * (0.5 + (sc >> 4) as f32) * 0.25];
                for (l, y8) in ys.chunks_exact_mut(8).enumerate() {
                    let dl = db[l / 2];
                    let idx = qs[l] as usize | ((qh << (8 - 2 * l)) & 0x300);
                    let grid = IQ2S_GRID[idx].to_le_bytes();
                    for ((y, &g), &mask) in y8.iter_mut().zip(&grid).zip(&KMASK_IQ2XS) {
                        *y = signed(dl * g as f32, signs[l] & mask != 0);
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

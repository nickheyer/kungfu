use crate::types::grids::{IQ3S_GRID, KMASK_IQ2XS};
use crate::types::{f16_at, signed};
use crate::{ActI8, Quant, TypeInfo};

pub static INFO: TypeInfo = TypeInfo { id: 21, name: "iq3_s", block: 256, size: 110 };
pub struct Iq3S;

/// Rows r0 on of the activations against one row, the grid and sign words of each thirty two looked up once for every row
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2<const R: usize>(src: &[u8], x: &ActI8, r0: usize, y: &mut [f32]) {
    use crate::simd::{MASK_SIGNS, hsum_ps};
    use std::arch::x86_64::*;
    let (n, n16) = (x.n, x.n / 16);
    let mut acc = [_mm256_setzero_ps(); R];
    for (b, blk) in src.chunks_exact(110).enumerate() {
        let dw = f16_at(blk, 0);
        let (qs, qh, signs, scales) = (&blk[2..66], &blk[66..74], &blk[74..106], &blk[106..110]);
        let mut sumi = [_mm256_setzero_si256(); R];
        for ib in 0..8 {
            let h = qh[ib] as usize;
            let idx = |k: usize| qs[8 * ib + k] as usize | ((h << (8 - k)) & 256);
            let g = _mm256_set_epi32(IQ3S_GRID[idx(7)] as i32, IQ3S_GRID[idx(6)] as i32, IQ3S_GRID[idx(5)] as i32, IQ3S_GRID[idx(4)] as i32, IQ3S_GRID[idx(3)] as i32, IQ3S_GRID[idx(2)] as i32, IQ3S_GRID[idx(1)] as i32, IQ3S_GRID[idx(0)] as i32);
            let sg = &signs[4 * ib..4 * ib + 4];
            let s = _mm256_set_epi64x(MASK_SIGNS[sg[3] as usize] as i64, MASK_SIGNS[sg[2] as usize] as i64, MASK_SIGNS[sg[1] as usize] as i64, MASK_SIGNS[sg[0] as usize] as i64);
            let sc = (scales[ib / 2] >> (4 * (ib % 2))) & 0xf;
            let scale = _mm256_set1_epi16(1 + 2 * sc as i16);
            for r in 0..R {
                let q8 = unsafe { _mm256_loadu_si256(x.q.as_ptr().add((r0 + r) * n + b * 256 + ib * 32) as *const __m256i) };
                let dot = _mm256_maddubs_epi16(g, _mm256_sign_epi8(q8, s));
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

impl Quant for Iq3S {
    fn info(&self) -> &'static TypeInfo {
        &INFO
    }
    fn dequant_row(&self, src: &[u8], dst: &mut [f32]) {
        for (b, out) in dst.chunks_exact_mut(256).enumerate() {
            let blk = &src[b * 110..(b + 1) * 110];
            let d = f16_at(blk, 0);
            for (ib, ys) in out.chunks_exact_mut(32).enumerate() {
                let qs = &blk[2 + 8 * ib..10 + 8 * ib];
                let qh = blk[66 + ib] as usize;
                let signs = &blk[74 + 4 * ib..78 + 4 * ib];
                let sc = (blk[106 + ib / 2] >> (4 * (ib % 2))) & 0xf;
                let db = d * (1 + 2 * sc as i32) as f32;
                for (l, y8) in ys.chunks_exact_mut(8).enumerate() {
                    let g1 = IQ3S_GRID[qs[2 * l] as usize | ((qh << (8 - 2 * l)) & 256)].to_le_bytes();
                    let g2 = IQ3S_GRID[qs[2 * l + 1] as usize | ((qh << (7 - 2 * l)) & 256)].to_le_bytes();
                    for j in 0..4 {
                        y8[j] = signed(db * g1[j] as f32, signs[l] & KMASK_IQ2XS[j] != 0);
                        y8[j + 4] = signed(db * g2[j] as f32, signs[l] & KMASK_IQ2XS[j + 4] != 0);
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

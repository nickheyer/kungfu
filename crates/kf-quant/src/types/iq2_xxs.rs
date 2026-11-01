use crate::types::grids::{IQ2XXS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS};
use crate::types::{f16_at, signed, u32_at};
use crate::{ActI8, Quant, TypeInfo};

pub static INFO: TypeInfo = TypeInfo { id: 16, name: "iq2_xxs", block: 256, size: 66 };
pub struct Iq2Xxs;

/// Rows r0 on of the activations against one row, the grid and sign words of each thirty two looked up once for every row
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2<const R: usize>(src: &[u8], x: &ActI8, r0: usize, y: &mut [f32]) {
    use crate::simd::{KEVEN_SIGNS, hsum_ps};
    use std::arch::x86_64::*;
    let (n, n16) = (x.n, x.n / 16);
    let mut acc = [_mm256_setzero_ps(); R];
    for (b, blk) in src.chunks_exact(66).enumerate() {
        let dw = f16_at(blk, 0);
        let mut sumi = [_mm256_setzero_si256(); R];
        for ib32 in 0..8 {
            let aux0 = u32_at(blk, 2 + 8 * ib32);
            let aux1 = u32_at(blk, 6 + 8 * ib32);
            let g = _mm256_set_epi64x(IQ2XXS_GRID[(aux0 >> 24) as usize] as i64, IQ2XXS_GRID[((aux0 >> 16) & 255) as usize] as i64, IQ2XXS_GRID[((aux0 >> 8) & 255) as usize] as i64, IQ2XXS_GRID[(aux0 & 255) as usize] as i64);
            let s = _mm256_set_epi64x(KEVEN_SIGNS[((aux1 >> 21) & 127) as usize] as i64, KEVEN_SIGNS[((aux1 >> 14) & 127) as usize] as i64, KEVEN_SIGNS[((aux1 >> 7) & 127) as usize] as i64, KEVEN_SIGNS[(aux1 & 127) as usize] as i64);
            let scale = _mm256_set1_epi16((2 * (aux1 >> 28) + 1) as i16);
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

impl Quant for Iq2Xxs {
    fn info(&self) -> &'static TypeInfo {
        &INFO
    }
    fn dequant_row(&self, src: &[u8], dst: &mut [f32]) {
        for (b, out) in dst.chunks_exact_mut(256).enumerate() {
            let blk = &src[b * 66..(b + 1) * 66];
            let d = f16_at(blk, 0);
            for (ib32, ys) in out.chunks_exact_mut(32).enumerate() {
                let aux0 = u32_at(blk, 2 + 8 * ib32);
                let aux1 = u32_at(blk, 6 + 8 * ib32);
                let db = d * (0.5 + (aux1 >> 28) as f32) * 0.25;
                for (l, (y8, &idx)) in ys.chunks_exact_mut(8).zip(&aux0.to_le_bytes()).enumerate() {
                    let grid = IQ2XXS_GRID[idx as usize].to_le_bytes();
                    let signs = KSIGNS_IQ2XS[((aux1 >> (7 * l)) & 127) as usize];
                    for ((y, &g), &mask) in y8.iter_mut().zip(&grid).zip(&KMASK_IQ2XS) {
                        *y = signed(db * g as f32, signs & mask != 0);
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

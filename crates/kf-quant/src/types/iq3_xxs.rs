use crate::types::grids::{IQ3XXS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS};
use crate::types::{f16_at, signed, u32_at};
use crate::{ActI8, Quant, TypeInfo};

pub static INFO: TypeInfo = TypeInfo { id: 18, name: "iq3_xxs", block: 256, size: 98 };
pub struct Iq3Xxs;

/// Rows r0 on of the activations against one row, the grid and sign words of each thirty two looked up once for every row
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2<const R: usize>(src: &[u8], x: &ActI8, r0: usize, y: &mut [f32]) {
    use crate::simd::{KEVEN_SIGNS, hsum_ps};
    use std::arch::x86_64::*;
    let (n, n16) = (x.n, x.n / 16);
    let mut acc = [_mm256_setzero_ps(); R];
    for (b, blk) in src.chunks_exact(98).enumerate() {
        let dw = f16_at(blk, 0);
        let mut sumi = [_mm256_setzero_si256(); R];
        for ib32 in 0..8 {
            let qs = &blk[2 + 8 * ib32..10 + 8 * ib32];
            let aux = u32_at(blk, 66 + 4 * ib32);
            let g = _mm256_set_epi32(IQ3XXS_GRID[qs[7] as usize] as i32, IQ3XXS_GRID[qs[6] as usize] as i32, IQ3XXS_GRID[qs[5] as usize] as i32, IQ3XXS_GRID[qs[4] as usize] as i32, IQ3XXS_GRID[qs[3] as usize] as i32, IQ3XXS_GRID[qs[2] as usize] as i32, IQ3XXS_GRID[qs[1] as usize] as i32, IQ3XXS_GRID[qs[0] as usize] as i32);
            let s = _mm256_set_epi64x(KEVEN_SIGNS[((aux >> 21) & 127) as usize] as i64, KEVEN_SIGNS[((aux >> 14) & 127) as usize] as i64, KEVEN_SIGNS[((aux >> 7) & 127) as usize] as i64, KEVEN_SIGNS[(aux & 127) as usize] as i64);
            let scale = _mm256_set1_epi16((2 * (aux >> 28) + 1) as i16);
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
        y[r0 + r] = 0.25 * hsum_ps(acc[r]);
    }
}

impl Quant for Iq3Xxs {
    fn info(&self) -> &'static TypeInfo {
        &INFO
    }
    fn dequant_row(&self, src: &[u8], dst: &mut [f32]) {
        for (b, out) in dst.chunks_exact_mut(256).enumerate() {
            let blk = &src[b * 98..(b + 1) * 98];
            let d = f16_at(blk, 0);
            for (ib32, ys) in out.chunks_exact_mut(32).enumerate() {
                let qs = &blk[2 + 8 * ib32..10 + 8 * ib32];
                let aux = u32_at(blk, 66 + 4 * ib32);
                let db = d * (0.5 + (aux >> 28) as f32) * 0.5;
                for (l, y8) in ys.chunks_exact_mut(8).enumerate() {
                    let signs = KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize];
                    let g1 = IQ3XXS_GRID[qs[2 * l] as usize].to_le_bytes();
                    let g2 = IQ3XXS_GRID[qs[2 * l + 1] as usize].to_le_bytes();
                    for j in 0..4 {
                        y8[j] = signed(db * g1[j] as f32, signs & KMASK_IQ2XS[j] != 0);
                        y8[j + 4] = signed(db * g2[j] as f32, signs & KMASK_IQ2XS[j + 4] != 0);
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

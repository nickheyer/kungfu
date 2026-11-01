use crate::types::grids::IQ1S_GRID;
use crate::types::{IQ1S_DELTA, u16_at};
use crate::{ActI8, Quant, TypeInfo};

pub static INFO: TypeInfo = TypeInfo { id: 29, name: "iq1_m", block: 256, size: 56 };
pub struct Iq1M;

/// Rows r0 on of the activations against one row, weights held as eight times grid value plus delta sign
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2<const R: usize>(src: &[u8], x: &ActI8, r0: usize, y: &mut [f32]) {
    use crate::simd::hsum_ps;
    use std::arch::x86_64::*;
    let (n, n16) = (x.n, x.n / 16);
    let eight = _mm256_set1_epi8(8);
    let mut acc = [_mm256_setzero_ps(); R];
    for (b, blk) in src.chunks_exact(56).enumerate() {
        let sc = [u16_at(blk, 48), u16_at(blk, 50), u16_at(blk, 52), u16_at(blk, 54)];
        let dw = half::f16::from_bits((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000)).to_f32();
        let mut sumi = [_mm256_setzero_si256(); R];
        for ib in 0..8 {
            let qs = &blk[4 * ib..4 * ib + 4];
            let qh = [blk[32 + 2 * ib] as usize, blk[33 + 2 * ib] as usize];
            let grid = |l: usize| IQ1S_GRID[qs[l] as usize | ((qh[l / 2] << (8 - 4 * (l % 2))) & 0x700)] as i64;
            let g = _mm256_set_epi64x(grid(3), grid(2), grid(1), grid(0));
            let sign = |l: usize| if qh[l / 2] & (0x08 << (4 * (l % 2))) != 0 { -1i64 } else { 0x0101010101010101 };
            let delta = _mm256_set_epi64x(sign(3), sign(2), sign(1), sign(0));
            let w = _mm256_add_epi8(_mm256_sign_epi8(eight, g), delta);
            let wabs = _mm256_sign_epi8(w, w);
            let s = (sc[ib / 2] >> (6 * (ib % 2))) as i16;
            let scale = _mm256_set_m128i(_mm_set1_epi16(2 * ((s >> 3) & 7) + 1), _mm_set1_epi16(2 * (s & 7) + 1));
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

impl Quant for Iq1M {
    fn info(&self) -> &'static TypeInfo {
        &INFO
    }
    fn dequant_row(&self, src: &[u8], dst: &mut [f32]) {
        for (b, out) in dst.chunks_exact_mut(256).enumerate() {
            let blk = &src[b * 56..(b + 1) * 56];
            let sc = [u16_at(blk, 48), u16_at(blk, 50), u16_at(blk, 52), u16_at(blk, 54)];
            let scale = (sc[0] >> 12)
                | ((sc[1] >> 8) & 0x00f0)
                | ((sc[2] >> 4) & 0x0f00)
                | (sc[3] & 0xf000);
            let d = half::f16::from_bits(scale).to_f32();
            for (ib, ys) in out.chunks_exact_mut(32).enumerate() {
                let s = sc[ib / 2] as i32;
                let dl1 = d * (2 * ((s >> (6 * (ib % 2))) & 0x7) + 1) as f32;
                let dl2 = d * (2 * ((s >> (6 * (ib % 2) + 3)) & 0x7) + 1) as f32;
                let qs = &blk[4 * ib..4 * ib + 4];
                let qh = [blk[32 + 2 * ib] as usize, blk[33 + 2 * ib] as usize];
                let idx = [
                    qs[0] as usize | ((qh[0] << 8) & 0x700),
                    qs[1] as usize | ((qh[0] << 4) & 0x700),
                    qs[2] as usize | ((qh[1] << 8) & 0x700),
                    qs[3] as usize | ((qh[1] << 4) & 0x700),
                ];
                let delta = [
                    if qh[0] & 0x08 != 0 { -IQ1S_DELTA } else { IQ1S_DELTA },
                    if qh[0] & 0x80 != 0 { -IQ1S_DELTA } else { IQ1S_DELTA },
                    if qh[1] & 0x08 != 0 { -IQ1S_DELTA } else { IQ1S_DELTA },
                    if qh[1] & 0x80 != 0 { -IQ1S_DELTA } else { IQ1S_DELTA },
                ];
                for (l, y8) in ys.chunks_exact_mut(8).enumerate() {
                    let dl = if l < 2 { dl1 } else { dl2 };
                    let grid = IQ1S_GRID[idx[l]].to_le_bytes();
                    for (y, &g) in y8.iter_mut().zip(&grid) {
                        *y = dl * ((g as i8) as f32 + delta[l]);
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

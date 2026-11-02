use crate::types::f16_at;
use crate::{ActI8, Quant, TypeInfo};

pub static INFO: TypeInfo = TypeInfo { id: 8, name: "q8_0", block: 32, size: 34 };
pub struct Q8_0;

/// Rows r0 on of the activations against one row, each block's thirty two weights loaded once for every row
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2<const R: usize>(src: &[u8], x: &ActI8, r0: usize, y: &mut [f32]) {
    use crate::simd::hsum_ps;
    use std::arch::x86_64::*;
    let (n, n16) = (x.n, x.n / 16);
    let ones = _mm256_set1_epi16(1);
    let mut acc = [_mm256_setzero_ps(); R];
    for (b, blk) in src.chunks_exact(34).enumerate() {
        let dw = f16_at(blk, 0);
        let w = unsafe { _mm256_loadu_si256(blk.as_ptr().add(2) as *const __m256i) };
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

/// Quantizes one row of whole blocks of 32 into q8_0 bytes, each block's scale the largest magnitude over 127 as
/// f16 and its integers the values over that scale rounded to nearest even
pub fn quantize_row(x: &[f32], dst: &mut [u8]) {
    for (blk, out) in x.chunks_exact(32).zip(dst.chunks_exact_mut(34)) {
        let amax = blk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out[..2].copy_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        for (o, v) in out[2..].iter_mut().zip(blk) {
            *o = ((v * id).round_ties_even() as i32).clamp(-127, 127) as i8 as u8;
        }
    }
}

impl Quant for Q8_0 {
    fn info(&self) -> &'static TypeInfo {
        &INFO
    }
    fn dequant_row(&self, src: &[u8], dst: &mut [f32]) {
        for (b, out) in dst.chunks_exact_mut(32).enumerate() {
            let blk = &src[b * 34..(b + 1) * 34];
            let d = f16_at(blk, 0);
            for j in 0..32 {
                out[j] = (blk[2 + j] as i8) as f32 * d;
            }
        }
    }
    fn dot_row(&self, src: &[u8], x: &[f32]) -> f32 {
        let mut acc = 0.0f32;
        for (b, xs) in x.chunks_exact(32).enumerate() {
            let blk = &src[b * 34..(b + 1) * 34];
            let d = f16_at(blk, 0);
            let mut s = 0.0f32;
            for j in 0..32 {
                s += (blk[2 + j] as i8) as f32 * xs[j];
            }
            acc += s * d;
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

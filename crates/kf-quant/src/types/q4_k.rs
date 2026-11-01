use crate::types::{f16_at, scale_min_k4};
use crate::{ActI8, Quant, TypeInfo};

pub static INFO: TypeInfo = TypeInfo { id: 12, name: "q4_K", block: 256, size: 144 };
pub struct Q4K;

/// Rows r0 on of the activations against one row, each pair of nibble sub blocks unpacked once for every row
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2<const R: usize>(src: &[u8], x: &ActI8, r0: usize, y: &mut [f32]) {
    use crate::simd::hsum_ps;
    use std::arch::x86_64::*;
    let (n, n16) = (x.n, x.n / 16);
    let m4 = _mm256_set1_epi8(0xf);
    let mut acc = [_mm256_setzero_ps(); R];
    for (b, blk) in src.chunks_exact(144).enumerate() {
        let d = f16_at(blk, 0);
        let dmin = f16_at(blk, 2);
        let mut sc = [0i16; 8];
        let mut mn = [0i32; 16];
        for s in 0..8 {
            let (a, m) = scale_min_k4(s, &blk[4..16]);
            sc[s] = a as i16;
            mn[2 * s] = m as i32;
            mn[2 * s + 1] = m as i32;
        }
        let mlo = unsafe { _mm256_loadu_si256(mn.as_ptr() as *const __m256i) };
        let mhi = unsafe { _mm256_loadu_si256(mn.as_ptr().add(8) as *const __m256i) };
        let mut sumi = [_mm256_setzero_si256(); R];
        for j in 0..4 {
            let q = unsafe { _mm256_loadu_si256(blk.as_ptr().add(16 + 32 * j) as *const __m256i) };
            let lo = _mm256_and_si256(q, m4);
            let hi = _mm256_and_si256(_mm256_srli_epi16(q, 4), m4);
            let slo = _mm256_set1_epi16(sc[2 * j]);
            let shi = _mm256_set1_epi16(sc[2 * j + 1]);
            for r in 0..R {
                let at = (r0 + r) * n + b * 256 + 64 * j;
                let q0 = unsafe { _mm256_loadu_si256(x.q.as_ptr().add(at) as *const __m256i) };
                let q1 = unsafe { _mm256_loadu_si256(x.q.as_ptr().add(at + 32) as *const __m256i) };
                let p0 = _mm256_madd_epi16(_mm256_maddubs_epi16(lo, q0), slo);
                let p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(hi, q1), shi);
                sumi[r] = _mm256_add_epi32(sumi[r], _mm256_add_epi32(p0, p1));
            }
        }
        for r in 0..R {
            let at = (r0 + r) * n16 + b * 16;
            let s0 = unsafe { _mm256_loadu_si256(x.sum.as_ptr().add(at) as *const __m256i) };
            let s1 = unsafe { _mm256_loadu_si256(x.sum.as_ptr().add(at + 8) as *const __m256i) };
            let mins = _mm256_add_epi32(_mm256_mullo_epi32(s0, mlo), _mm256_mullo_epi32(s1, mhi));
            let dx = x.d[at];
            acc[r] = _mm256_fmadd_ps(_mm256_set1_ps(d * dx), _mm256_cvtepi32_ps(sumi[r]), acc[r]);
            acc[r] = _mm256_fnmadd_ps(_mm256_set1_ps(dmin * dx), _mm256_cvtepi32_ps(mins), acc[r]);
        }
    }
    for r in 0..R {
        y[r0 + r] = hsum_ps(acc[r]);
    }
}

impl Quant for Q4K {
    fn info(&self) -> &'static TypeInfo {
        &INFO
    }
    fn dequant_row(&self, src: &[u8], dst: &mut [f32]) {
        for (b, out) in dst.chunks_exact_mut(256).enumerate() {
            let blk = &src[b * 144..(b + 1) * 144];
            let d = f16_at(blk, 0);
            let min = f16_at(blk, 2);
            let scales = &blk[4..16];
            for (j, ys) in out.chunks_exact_mut(64).enumerate() {
                let q = &blk[16 + 32 * j..48 + 32 * j];
                let (sc, m) = scale_min_k4(2 * j, scales);
                let d1 = d * sc as f32;
                let m1 = min * m as f32;
                let (sc, m) = scale_min_k4(2 * j + 1, scales);
                let d2 = d * sc as f32;
                let m2 = min * m as f32;
                let (lo, hi) = ys.split_at_mut(32);
                for ((y0, y1), &v) in lo.iter_mut().zip(hi.iter_mut()).zip(q) {
                    *y0 = d1 * (v & 0xF) as f32 - m1;
                    *y1 = d2 * (v >> 4) as f32 - m2;
                }
            }
        }
    }
    fn dot_row(&self, src: &[u8], x: &[f32]) -> f32 {
        let mut acc = 0.0f32;
        for (b, xs) in x.chunks_exact(256).enumerate() {
            let blk = &src[b * 144..(b + 1) * 144];
            let d = f16_at(blk, 0);
            let min = f16_at(blk, 2);
            let scales = &blk[4..16];
            for (j, xj) in xs.chunks_exact(64).enumerate() {
                let q = &blk[16 + 32 * j..48 + 32 * j];
                let (sc1, m1) = scale_min_k4(2 * j, scales);
                let (sc2, m2) = scale_min_k4(2 * j + 1, scales);
                let (x1, x2) = xj.split_at(32);
                let (mut qx1, mut sx1, mut qx2, mut sx2) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                for ((&a, &c), &v) in x1.iter().zip(x2).zip(q) {
                    qx1 += (v & 0xF) as f32 * a;
                    sx1 += a;
                    qx2 += (v >> 4) as f32 * c;
                    sx2 += c;
                }
                acc += d * sc1 as f32 * qx1 - min * m1 as f32 * sx1;
                acc += d * sc2 as f32 * qx2 - min * m2 as f32 * sx2;
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

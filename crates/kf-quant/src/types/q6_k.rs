use crate::types::f16_at;
use crate::{ActI8, Quant, TypeInfo};

pub static INFO: TypeInfo = TypeInfo { id: 14, name: "q6_K", block: 256, size: 210 };
pub struct Q6K;

/// Four signed 6 bit quants at offset l of the 128 element half using its ql and qh slices
fn quads(ql: &[u8], qh: &[u8], l: usize) -> [i8; 4] {
    [
        ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i8 - 32,
        ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i8 - 32,
        ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8 - 32,
        ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8 - 32,
    ]
}

/// Rows r0 on of the activations against one row, each half's six bit quants unpacked once for every row
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2<const R: usize>(src: &[u8], x: &ActI8, r0: usize, y: &mut [f32]) {
    use crate::simd::hsum_ps;
    use std::arch::x86_64::*;
    let (n, n16) = (x.n, x.n / 16);
    let m4 = _mm256_set1_epi8(0xf);
    let m30 = _mm256_set1_epi8(0x30);
    let mut acc = [_mm256_setzero_ps(); R];
    for (b, blk) in src.chunks_exact(210).enumerate() {
        let d = f16_at(blk, 208);
        let sc = &blk[192..208];
        let scv = unsafe { _mm_loadu_si128(sc.as_ptr() as *const __m128i) };
        let clo = _mm256_slli_epi32(_mm256_cvtepi8_epi32(scv), 5);
        let chi = _mm256_slli_epi32(_mm256_cvtepi8_epi32(_mm_srli_si128(scv, 8)), 5);
        let mut sumi = [_mm256_setzero_si256(); R];
        for h in 0..2 {
            let qa = unsafe { _mm256_loadu_si256(blk.as_ptr().add(64 * h) as *const __m256i) };
            let qb = unsafe { _mm256_loadu_si256(blk.as_ptr().add(64 * h + 32) as *const __m256i) };
            let qh = unsafe { _mm256_loadu_si256(blk.as_ptr().add(128 + 32 * h) as *const __m256i) };
            let w = [
                _mm256_or_si256(_mm256_and_si256(qa, m4), _mm256_and_si256(_mm256_slli_epi16(qh, 4), m30)),
                _mm256_or_si256(_mm256_and_si256(qb, m4), _mm256_and_si256(_mm256_slli_epi16(qh, 2), m30)),
                _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(qa, 4), m4), _mm256_and_si256(qh, m30)),
                _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(qb, 4), m4), _mm256_and_si256(_mm256_srli_epi16(qh, 2), m30)),
            ];
            for (k, wk) in w.into_iter().enumerate() {
                let g = 8 * h + 2 * k;
                let scale = _mm256_set_m128i(_mm_set1_epi16(sc[g + 1] as i8 as i16), _mm_set1_epi16(sc[g] as i8 as i16));
                for r in 0..R {
                    let q8 = unsafe { _mm256_loadu_si256(x.q.as_ptr().add((r0 + r) * n + b * 256 + 16 * g) as *const __m256i) };
                    sumi[r] = _mm256_add_epi32(sumi[r], _mm256_madd_epi16(_mm256_maddubs_epi16(wk, q8), scale));
                }
            }
        }
        for r in 0..R {
            let at = (r0 + r) * n16 + b * 16;
            let s0 = unsafe { _mm256_loadu_si256(x.sum.as_ptr().add(at) as *const __m256i) };
            let s1 = unsafe { _mm256_loadu_si256(x.sum.as_ptr().add(at + 8) as *const __m256i) };
            let bias = _mm256_add_epi32(_mm256_mullo_epi32(s0, clo), _mm256_mullo_epi32(s1, chi));
            let dot = _mm256_sub_epi32(sumi[r], bias);
            acc[r] = _mm256_fmadd_ps(_mm256_set1_ps(d * x.d[at]), _mm256_cvtepi32_ps(dot), acc[r]);
        }
    }
    for r in 0..R {
        y[r0 + r] = hsum_ps(acc[r]);
    }
}

impl Quant for Q6K {
    fn info(&self) -> &'static TypeInfo {
        &INFO
    }
    fn dequant_row(&self, src: &[u8], dst: &mut [f32]) {
        for (b, out) in dst.chunks_exact_mut(256).enumerate() {
            let blk = &src[b * 210..(b + 1) * 210];
            let d = f16_at(blk, 208);
            for (n, ys) in out.chunks_exact_mut(128).enumerate() {
                let ql = &blk[64 * n..64 * n + 64];
                let qh = &blk[128 + 32 * n..160 + 32 * n];
                let sc = &blk[192 + 8 * n..200 + 8 * n];
                for l in 0..32 {
                    let is = l / 16;
                    for (k, q) in quads(ql, qh, l).into_iter().enumerate() {
                        ys[l + 32 * k] = d * (sc[is + 2 * k] as i8) as f32 * q as f32;
                    }
                }
            }
        }
    }
    fn dot_row(&self, src: &[u8], x: &[f32]) -> f32 {
        let mut acc = 0.0f32;
        for (b, xs) in x.chunks_exact(256).enumerate() {
            let blk = &src[b * 210..(b + 1) * 210];
            let d = f16_at(blk, 208);
            let mut sums = [0.0f32; 16];
            for (n, xn) in xs.chunks_exact(128).enumerate() {
                let ql = &blk[64 * n..64 * n + 64];
                let qh = &blk[128 + 32 * n..160 + 32 * n];
                for l in 0..32 {
                    let is = 8 * n + l / 16;
                    for (k, q) in quads(ql, qh, l).into_iter().enumerate() {
                        sums[is + 2 * k] += q as f32 * xn[l + 32 * k];
                    }
                }
            }
            for (s, &sc) in sums.iter().zip(&blk[192..208]) {
                acc += d * (sc as i8) as f32 * s;
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

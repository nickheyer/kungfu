use crate::types::{f16_at, u32_at};
use crate::{ActI8, Quant, TypeInfo};

pub static INFO: TypeInfo = TypeInfo { id: 11, name: "q3_K", block: 256, size: 110 };
pub struct Q3K;

const KMASK1: u32 = 0x0303_0303;
const KMASK2: u32 = 0x0f0f_0f0f;

/// Unpacks the twelve scale bytes into sixteen 6 bit scales stored as bytes
fn scales(blk: &[u8]) -> [i8; 16] {
    let a0 = u32_at(blk, 96);
    let a1 = u32_at(blk, 100);
    let tmp = u32_at(blk, 104);
    let aux = [
        (a0 & KMASK2) | ((tmp & KMASK1) << 4),
        (a1 & KMASK2) | (((tmp >> 2) & KMASK1) << 4),
        ((a0 >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4),
        ((a1 >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4),
    ];
    let mut out = [0i8; 16];
    for (dst, a) in out.chunks_exact_mut(4).zip(aux) {
        for (o, byte) in dst.iter_mut().zip(a.to_le_bytes()) {
            *o = byte as i8;
        }
    }
    out
}

/// Rows r0 on of the activations against one row, each half's three bit quants unpacked once for every row
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2<const R: usize>(src: &[u8], x: &ActI8, r0: usize, y: &mut [f32]) {
    use crate::simd::hsum_ps;
    use std::arch::x86_64::*;
    let (n, n16) = (x.n, x.n / 16);
    let m3 = _mm256_set1_epi8(3);
    let m4 = _mm256_set1_epi8(4);
    let mut acc = [_mm256_setzero_ps(); R];
    for (b, blk) in src.chunks_exact(110).enumerate() {
        let d = f16_at(blk, 108);
        let mut sc = [0i16; 16];
        let mut bias = [0i32; 16];
        for (g, s) in scales(blk).into_iter().enumerate() {
            sc[g] = s as i16 - 32;
            bias[g] = 4 * (s as i32 - 32);
        }
        let blo = unsafe { _mm256_loadu_si256(bias.as_ptr() as *const __m256i) };
        let bhi = unsafe { _mm256_loadu_si256(bias.as_ptr().add(8) as *const __m256i) };
        let mut hb = unsafe { _mm256_loadu_si256(blk.as_ptr() as *const __m256i) };
        let mut sumi = [_mm256_setzero_si256(); R];
        for h in 0..2 {
            let mut qv = unsafe { _mm256_loadu_si256(blk.as_ptr().add(32 + 32 * h) as *const __m256i) };
            for k in 0..4 {
                let g = 8 * h + 2 * k;
                let w = _mm256_or_si256(_mm256_and_si256(qv, m3), _mm256_and_si256(_mm256_slli_epi16(hb, 2), m4));
                qv = _mm256_srli_epi16(qv, 2);
                hb = _mm256_srli_epi16(hb, 1);
                let scale = _mm256_set_m128i(_mm_set1_epi16(sc[g + 1]), _mm_set1_epi16(sc[g]));
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
            let off = _mm256_add_epi32(_mm256_mullo_epi32(s0, blo), _mm256_mullo_epi32(s1, bhi));
            let dot = _mm256_sub_epi32(sumi[r], off);
            acc[r] = _mm256_fmadd_ps(_mm256_set1_ps(d * x.d[at]), _mm256_cvtepi32_ps(dot), acc[r]);
        }
    }
    for r in 0..R {
        y[r0 + r] = hsum_ps(acc[r]);
    }
}

impl Quant for Q3K {
    fn info(&self) -> &'static TypeInfo {
        &INFO
    }
    fn dequant_row(&self, src: &[u8], dst: &mut [f32]) {
        for (b, out) in dst.chunks_exact_mut(256).enumerate() {
            let blk = &src[b * 110..(b + 1) * 110];
            let d_all = f16_at(blk, 108);
            let hm = &blk[0..32];
            let sc = scales(blk);
            let mut m = 1u8;
            let mut y = 0;
            let mut is = 0;
            for n in 0..2 {
                let q = &blk[32 + 32 * n..64 + 32 * n];
                for shift in [0, 2, 4, 6] {
                    for h in 0..2 {
                        let dl = d_all * (sc[is] as i32 - 32) as f32;
                        is += 1;
                        for l in 16 * h..16 * h + 16 {
                            let hi = if hm[l] & m != 0 { 0 } else { 4 };
                            out[y] = dl * (((q[l] >> shift) & 3) as i32 - hi) as f32;
                            y += 1;
                        }
                    }
                    m <<= 1;
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

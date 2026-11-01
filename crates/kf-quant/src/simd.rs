//! AVX2 helpers shared by the int8 dot kernels of the quant types
use std::arch::x86_64::*;

/// Eight sign bytes per seven bit sign index with its parity bit, 0xff for a negated weight and 0x01 otherwise
pub(crate) static KEVEN_SIGNS: [u64; 128] = {
    let mut t = [0u64; 128];
    let mut i = 0;
    while i < 128 {
        let s = i as u8 | (((i as u8).count_ones() as u8 & 1) << 7);
        t[i] = sign_bytes(s);
        i += 1;
    }
    t
};

/// Eight sign bytes per eight bit sign mask, bit j of the mask negating weight j
pub(crate) static MASK_SIGNS: [u64; 256] = {
    let mut t = [0u64; 256];
    let mut i = 0;
    while i < 256 {
        t[i] = sign_bytes(i as u8);
        i += 1;
    }
    t
};

const fn sign_bytes(mask: u8) -> u64 {
    let mut v = 0u64;
    let mut j = 0;
    while j < 8 {
        v |= (if (mask >> j) & 1 == 1 { 0xffu64 } else { 0x01u64 }) << (8 * j);
        j += 1;
    }
    v
}

/// Whether the AVX2 and FMA kernels can run on this machine
pub(crate) fn have() -> bool {
    std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
}

/// Sum of the eight lanes
#[target_feature(enable = "avx2")]
pub(crate) fn hsum_ps(v: __m256) -> f32 {
    let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    _mm_cvtss_f32(s)
}

/// Runs a row kernel generic over its row count on every activation row, four, two, or one rows at a time
macro_rules! avx2_rows {
    ($f:ident, $src:expr, $x:expr, $y:expr) => {{
        let mut r = 0;
        while r < $x.rows {
            let left = $x.rows - r;
            if left >= 4 {
                unsafe { $f::<4>($src, $x, r, $y) };
                r += 4;
            } else if left >= 2 {
                unsafe { $f::<2>($src, $x, r, $y) };
                r += 2;
            } else {
                unsafe { $f::<1>($src, $x, r, $y) };
                r += 1;
            }
        }
    }};
}
pub(crate) use avx2_rows;

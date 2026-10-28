use crate::Quant;

pub mod bf16;
pub mod f16;
pub mod f32;
pub mod grids;
pub mod iq1_m;
pub mod iq1_s;
pub mod iq2_s;
pub mod iq2_xs;
pub mod iq2_xxs;
pub mod iq3_s;
pub mod iq3_xxs;
pub mod iq4_nl;
pub mod iq4_xs;
pub mod mxfp4;
pub mod nvfp4;
pub mod q1_0;
pub mod q2_0;
pub mod q2_k;
pub mod q3_k;
pub mod q4_0;
pub mod q4_1;
pub mod q4_k;
pub mod q5_0;
pub mod q5_1;
pub mod q5_k;
pub mod q6_k;
pub mod q8_0;
pub mod q8_1;
pub mod q8_k;
pub mod tq1_0;
pub mod tq2_0;

/// The only list of quant types, one entry per module
pub fn all() -> &'static [&'static dyn Quant] {
    &[
        &f32::F32,
        &f16::F16,
        &bf16::Bf16,
        &q4_0::Q4_0,
        &q4_1::Q4_1,
        &q5_0::Q5_0,
        &q5_1::Q5_1,
        &q8_0::Q8_0,
        &q8_1::Q8_1,
        &q2_k::Q2K,
        &q3_k::Q3K,
        &q4_k::Q4K,
        &q5_k::Q5K,
        &q6_k::Q6K,
        &q8_k::Q8K,
        &iq2_xxs::Iq2Xxs,
        &iq2_xs::Iq2Xs,
        &iq3_xxs::Iq3Xxs,
        &iq1_s::Iq1S,
        &iq4_nl::Iq4Nl,
        &iq3_s::Iq3S,
        &iq2_s::Iq2S,
        &iq4_xs::Iq4Xs,
        &iq1_m::Iq1M,
        &tq1_0::Tq1_0,
        &tq2_0::Tq2_0,
        &mxfp4::Mxfp4,
        &nvfp4::Nvfp4,
        &q1_0::Q1_0,
        &q2_0::Q2_0,
    ]
}

pub(crate) const IQ1S_DELTA: core::primitive::f32 = 0.125;

pub(crate) fn f16_at(src: &[u8], at: usize) -> core::primitive::f32 {
    half::f16::from_le_bytes([src[at], src[at + 1]]).to_f32()
}

pub(crate) fn u32_at(src: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([src[at], src[at + 1], src[at + 2], src[at + 3]])
}

pub(crate) fn u16_at(src: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([src[at], src[at + 1]])
}

/// Unpacks the 6 bit scale and min of sub block j from a 12 byte k quant scale array
pub(crate) fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4))
    }
}

/// Negates v when neg is set unless v is NaN, matching a hardware multiply by minus one
pub(crate) fn signed(v: core::primitive::f32, neg: bool) -> core::primitive::f32 {
    if neg && !v.is_nan() { -v } else { v }
}

use crate::types::grids::KVALUES_FP4;
use crate::{Quant, TypeInfo};

pub static INFO: TypeInfo = TypeInfo { id: 39, name: "mxfp4", block: 32, size: 17 };
pub struct Mxfp4;

/// Half the power of two encoded by an e8m0 exponent byte
fn e8m0_to_fp32_half(x: u8) -> f32 {
    let bits = if x < 2 { 0x0020_0000u32 << x } else { ((x - 1) as u32) << 23 };
    f32::from_bits(bits)
}

impl Quant for Mxfp4 {
    fn info(&self) -> &'static TypeInfo {
        &INFO
    }
    fn dequant_row(&self, src: &[u8], dst: &mut [f32]) {
        for (b, out) in dst.chunks_exact_mut(32).enumerate() {
            let blk = &src[b * 17..(b + 1) * 17];
            let d = e8m0_to_fp32_half(blk[0]);
            let (lo, hi) = out.split_at_mut(16);
            for ((y0, y1), &q) in lo.iter_mut().zip(hi.iter_mut()).zip(&blk[1..17]) {
                *y0 = KVALUES_FP4[(q & 0x0F) as usize] as f32 * d;
                *y1 = KVALUES_FP4[(q >> 4) as usize] as f32 * d;
            }
        }
    }
}

use crate::types::grids::KVALUES_FP4;
use crate::{Quant, TypeInfo};

pub static INFO: TypeInfo = TypeInfo { id: 40, name: "nvfp4", block: 64, size: 36 };
pub struct Nvfp4;

/// Exact power of two for exponents in the normal f32 range
fn pow2(e: i32) -> f32 {
    f32::from_bits(((e + 127) as u32) << 23)
}

/// Half the value of a ue4m3 scale byte
fn ue4m3_to_fp32(x: u8) -> f32 {
    if x == 0 || x == 0x7F {
        return 0.0;
    }
    let exp = ((x >> 3) & 0xF) as i32;
    let man = (x & 0x7) as i32;
    let raw =
        if exp == 0 { man as f32 * pow2(-9) } else { (1.0 + man as f32 / 8.0) * pow2(exp - 7) };
    raw * 0.5
}

impl Quant for Nvfp4 {
    fn info(&self) -> &'static TypeInfo {
        &INFO
    }
    fn dequant_row(&self, src: &[u8], dst: &mut [f32]) {
        for (b, out) in dst.chunks_exact_mut(64).enumerate() {
            let blk = &src[b * 36..(b + 1) * 36];
            for (s, ys) in out.chunks_exact_mut(16).enumerate() {
                let d = ue4m3_to_fp32(blk[s]);
                let (lo, hi) = ys.split_at_mut(8);
                for ((y0, y1), &q) in
                    lo.iter_mut().zip(hi.iter_mut()).zip(&blk[4 + 8 * s..12 + 8 * s])
                {
                    *y0 = KVALUES_FP4[(q & 0x0F) as usize] as f32 * d;
                    *y1 = KVALUES_FP4[(q >> 4) as usize] as f32 * d;
                }
            }
        }
    }
}

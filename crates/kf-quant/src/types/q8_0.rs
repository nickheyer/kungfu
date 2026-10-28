use crate::types::f16_at;
use crate::{Quant, TypeInfo};

pub static INFO: TypeInfo = TypeInfo { id: 8, name: "q8_0", block: 32, size: 34 };
pub struct Q8_0;

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
}

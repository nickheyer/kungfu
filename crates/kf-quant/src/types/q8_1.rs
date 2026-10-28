use crate::types::f16_at;
use crate::{Quant, TypeInfo};

pub static INFO: TypeInfo = TypeInfo { id: 9, name: "q8_1", block: 32, size: 36 };
pub struct Q8_1;

impl Quant for Q8_1 {
    fn info(&self) -> &'static TypeInfo {
        &INFO
    }
    fn dequant_row(&self, src: &[u8], dst: &mut [f32]) {
        for (b, out) in dst.chunks_exact_mut(32).enumerate() {
            let blk = &src[b * 36..(b + 1) * 36];
            let d = f16_at(blk, 0);
            for j in 0..32 {
                out[j] = (blk[4 + j] as i8) as f32 * d;
            }
        }
    }
}

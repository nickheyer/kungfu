use crate::types::f16_at;
use crate::{Quant, TypeInfo};

pub static INFO: TypeInfo = TypeInfo { id: 2, name: "q4_0", block: 32, size: 18 };
pub struct Q4_0;

impl Quant for Q4_0 {
    fn info(&self) -> &'static TypeInfo {
        &INFO
    }
    fn dequant_row(&self, src: &[u8], dst: &mut [f32]) {
        for (b, out) in dst.chunks_exact_mut(32).enumerate() {
            let blk = &src[b * 18..(b + 1) * 18];
            let d = f16_at(blk, 0);
            for j in 0..16 {
                let q = blk[2 + j];
                out[j] = ((q & 0x0F) as i32 - 8) as f32 * d;
                out[j + 16] = ((q >> 4) as i32 - 8) as f32 * d;
            }
        }
    }
}

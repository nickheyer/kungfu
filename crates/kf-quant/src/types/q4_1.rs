use crate::types::f16_at;
use crate::{Quant, TypeInfo};

pub static INFO: TypeInfo = TypeInfo { id: 3, name: "q4_1", block: 32, size: 20 };
pub struct Q4_1;

impl Quant for Q4_1 {
    fn info(&self) -> &'static TypeInfo {
        &INFO
    }
    fn dequant_row(&self, src: &[u8], dst: &mut [f32]) {
        for (b, out) in dst.chunks_exact_mut(32).enumerate() {
            let blk = &src[b * 20..(b + 1) * 20];
            let d = f16_at(blk, 0);
            let m = f16_at(blk, 2);
            for j in 0..16 {
                let q = blk[4 + j];
                out[j] = (q & 0x0F) as f32 * d + m;
                out[j + 16] = (q >> 4) as f32 * d + m;
            }
        }
    }
}

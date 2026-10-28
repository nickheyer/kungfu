use crate::types::{f16_at, u32_at};
use crate::{Quant, TypeInfo};

pub static INFO: TypeInfo = TypeInfo { id: 7, name: "q5_1", block: 32, size: 24 };
pub struct Q5_1;

impl Quant for Q5_1 {
    fn info(&self) -> &'static TypeInfo {
        &INFO
    }
    fn dequant_row(&self, src: &[u8], dst: &mut [f32]) {
        for (b, out) in dst.chunks_exact_mut(32).enumerate() {
            let blk = &src[b * 24..(b + 1) * 24];
            let d = f16_at(blk, 0);
            let m = f16_at(blk, 2);
            let qh = u32_at(blk, 4);
            for j in 0..16 {
                let xh0 = ((qh >> j) << 4) & 0x10;
                let xh1 = (qh >> (j + 12)) & 0x10;
                let x0 = (blk[8 + j] & 0x0F) as u32 | xh0;
                let x1 = (blk[8 + j] >> 4) as u32 | xh1;
                out[j] = x0 as f32 * d + m;
                out[j + 16] = x1 as f32 * d + m;
            }
        }
    }
}

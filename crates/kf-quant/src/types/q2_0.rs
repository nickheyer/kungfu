use crate::types::f16_at;
use crate::{Quant, TypeInfo};

pub static INFO: TypeInfo = TypeInfo { id: 42, name: "q2_0", block: 64, size: 18 };
pub struct Q2_0;

impl Quant for Q2_0 {
    fn info(&self) -> &'static TypeInfo {
        &INFO
    }
    fn dequant_row(&self, src: &[u8], dst: &mut [f32]) {
        for (b, out) in dst.chunks_exact_mut(64).enumerate() {
            let blk = &src[b * 18..(b + 1) * 18];
            let d = f16_at(blk, 0);
            for (j, y) in out.iter_mut().enumerate() {
                let q = (blk[2 + j / 4] >> ((j % 4) * 2)) & 0x03;
                *y = (q as i32 - 1) as f32 * d;
            }
        }
    }
}

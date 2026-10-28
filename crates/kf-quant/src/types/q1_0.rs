use crate::types::f16_at;
use crate::{Quant, TypeInfo};

pub static INFO: TypeInfo = TypeInfo { id: 41, name: "q1_0", block: 128, size: 18 };
pub struct Q1_0;

impl Quant for Q1_0 {
    fn info(&self) -> &'static TypeInfo {
        &INFO
    }
    fn dequant_row(&self, src: &[u8], dst: &mut [f32]) {
        for (b, out) in dst.chunks_exact_mut(128).enumerate() {
            let blk = &src[b * 18..(b + 1) * 18];
            let d = f16_at(blk, 0);
            let neg_d = -d;
            for (j, y) in out.iter_mut().enumerate() {
                *y = if (blk[2 + j / 8] >> (j % 8)) & 1 != 0 { d } else { neg_d };
            }
        }
    }
}

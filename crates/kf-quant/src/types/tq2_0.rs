use crate::types::f16_at;
use crate::{Quant, TypeInfo};

pub static INFO: TypeInfo = TypeInfo { id: 35, name: "tq2_0", block: 256, size: 66 };
pub struct Tq2_0;

impl Quant for Tq2_0 {
    fn info(&self) -> &'static TypeInfo {
        &INFO
    }
    fn dequant_row(&self, src: &[u8], dst: &mut [f32]) {
        for (b, out) in dst.chunks_exact_mut(256).enumerate() {
            let blk = &src[b * 66..(b + 1) * 66];
            let d = f16_at(blk, 64);
            let mut y = 0;
            for qs in blk[0..64].chunks_exact(32) {
                for l in 0..4 {
                    for &q in qs {
                        out[y] = (((q >> (l * 2)) & 3) as i32 - 1) as f32 * d;
                        y += 1;
                    }
                }
            }
        }
    }
}

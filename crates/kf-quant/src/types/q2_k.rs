use crate::types::f16_at;
use crate::{Quant, TypeInfo};

pub static INFO: TypeInfo = TypeInfo { id: 10, name: "q2_K", block: 256, size: 84 };
pub struct Q2K;

impl Quant for Q2K {
    fn info(&self) -> &'static TypeInfo {
        &INFO
    }
    fn dequant_row(&self, src: &[u8], dst: &mut [f32]) {
        for (b, out) in dst.chunks_exact_mut(256).enumerate() {
            let blk = &src[b * 84..(b + 1) * 84];
            let d = f16_at(blk, 80);
            let min = f16_at(blk, 82);
            let mut y = 0;
            let mut is = 0;
            for n in 0..2 {
                let q = &blk[16 + 32 * n..48 + 32 * n];
                for shift in [0, 2, 4, 6] {
                    for part in q.chunks_exact(16) {
                        let sc = blk[is];
                        is += 1;
                        let dl = d * (sc & 0xF) as f32;
                        let ml = min * (sc >> 4) as f32;
                        for &v in part {
                            out[y] = dl * ((v >> shift) & 3) as f32 - ml;
                            y += 1;
                        }
                    }
                }
            }
        }
    }
}

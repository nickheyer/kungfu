use crate::types::grids::{IQ3XXS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS};
use crate::types::{f16_at, signed, u32_at};
use crate::{Quant, TypeInfo};

pub static INFO: TypeInfo = TypeInfo { id: 18, name: "iq3_xxs", block: 256, size: 98 };
pub struct Iq3Xxs;

impl Quant for Iq3Xxs {
    fn info(&self) -> &'static TypeInfo {
        &INFO
    }
    fn dequant_row(&self, src: &[u8], dst: &mut [f32]) {
        for (b, out) in dst.chunks_exact_mut(256).enumerate() {
            let blk = &src[b * 98..(b + 1) * 98];
            let d = f16_at(blk, 0);
            for (ib32, ys) in out.chunks_exact_mut(32).enumerate() {
                let qs = &blk[2 + 8 * ib32..10 + 8 * ib32];
                let aux = u32_at(blk, 66 + 4 * ib32);
                let db = d * (0.5 + (aux >> 28) as f32) * 0.5;
                for (l, y8) in ys.chunks_exact_mut(8).enumerate() {
                    let signs = KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize];
                    let g1 = IQ3XXS_GRID[qs[2 * l] as usize].to_le_bytes();
                    let g2 = IQ3XXS_GRID[qs[2 * l + 1] as usize].to_le_bytes();
                    for j in 0..4 {
                        y8[j] = signed(db * g1[j] as f32, signs & KMASK_IQ2XS[j] != 0);
                        y8[j + 4] = signed(db * g2[j] as f32, signs & KMASK_IQ2XS[j + 4] != 0);
                    }
                }
            }
        }
    }
}

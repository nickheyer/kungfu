use crate::types::grids::{IQ2XXS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS};
use crate::types::{f16_at, signed, u32_at};
use crate::{Quant, TypeInfo};

pub static INFO: TypeInfo = TypeInfo { id: 16, name: "iq2_xxs", block: 256, size: 66 };
pub struct Iq2Xxs;

impl Quant for Iq2Xxs {
    fn info(&self) -> &'static TypeInfo {
        &INFO
    }
    fn dequant_row(&self, src: &[u8], dst: &mut [f32]) {
        for (b, out) in dst.chunks_exact_mut(256).enumerate() {
            let blk = &src[b * 66..(b + 1) * 66];
            let d = f16_at(blk, 0);
            for (ib32, ys) in out.chunks_exact_mut(32).enumerate() {
                let aux0 = u32_at(blk, 2 + 8 * ib32);
                let aux1 = u32_at(blk, 6 + 8 * ib32);
                let db = d * (0.5 + (aux1 >> 28) as f32) * 0.25;
                for (l, (y8, &idx)) in ys.chunks_exact_mut(8).zip(&aux0.to_le_bytes()).enumerate() {
                    let grid = IQ2XXS_GRID[idx as usize].to_le_bytes();
                    let signs = KSIGNS_IQ2XS[((aux1 >> (7 * l)) & 127) as usize];
                    for ((y, &g), &mask) in y8.iter_mut().zip(&grid).zip(&KMASK_IQ2XS) {
                        *y = signed(db * g as f32, signs & mask != 0);
                    }
                }
            }
        }
    }
}

use crate::types::grids::{IQ2XS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS};
use crate::types::{f16_at, signed, u16_at};
use crate::{Quant, TypeInfo};

pub static INFO: TypeInfo = TypeInfo { id: 17, name: "iq2_xs", block: 256, size: 74 };
pub struct Iq2Xs;

impl Quant for Iq2Xs {
    fn info(&self) -> &'static TypeInfo {
        &INFO
    }
    fn dequant_row(&self, src: &[u8], dst: &mut [f32]) {
        for (b, out) in dst.chunks_exact_mut(256).enumerate() {
            let blk = &src[b * 74..(b + 1) * 74];
            let d = f16_at(blk, 0);
            for (ib32, ys) in out.chunks_exact_mut(32).enumerate() {
                let sc = blk[66 + ib32];
                let db =
                    [d * (0.5 + (sc & 0xf) as f32) * 0.25, d * (0.5 + (sc >> 4) as f32) * 0.25];
                for (l, y8) in ys.chunks_exact_mut(8).enumerate() {
                    let q = u16_at(blk, 2 + 2 * (4 * ib32 + l));
                    let grid = IQ2XS_GRID[(q & 511) as usize].to_le_bytes();
                    let signs = KSIGNS_IQ2XS[(q >> 9) as usize];
                    let dl = db[l / 2];
                    for ((y, &g), &mask) in y8.iter_mut().zip(&grid).zip(&KMASK_IQ2XS) {
                        *y = signed(dl * g as f32, signs & mask != 0);
                    }
                }
            }
        }
    }
}

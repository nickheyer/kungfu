use crate::types::grids::{IQ2S_GRID, KMASK_IQ2XS};
use crate::types::{f16_at, signed};
use crate::{Quant, TypeInfo};

pub static INFO: TypeInfo = TypeInfo { id: 22, name: "iq2_s", block: 256, size: 82 };
pub struct Iq2S;

impl Quant for Iq2S {
    fn info(&self) -> &'static TypeInfo {
        &INFO
    }
    fn dequant_row(&self, src: &[u8], dst: &mut [f32]) {
        for (b, out) in dst.chunks_exact_mut(256).enumerate() {
            let blk = &src[b * 82..(b + 1) * 82];
            let d = f16_at(blk, 0);
            for (ib32, ys) in out.chunks_exact_mut(32).enumerate() {
                let qs = &blk[2 + 4 * ib32..6 + 4 * ib32];
                let signs = &blk[34 + 4 * ib32..38 + 4 * ib32];
                let qh = blk[66 + ib32] as usize;
                let sc = blk[74 + ib32];
                let db =
                    [d * (0.5 + (sc & 0xf) as f32) * 0.25, d * (0.5 + (sc >> 4) as f32) * 0.25];
                for (l, y8) in ys.chunks_exact_mut(8).enumerate() {
                    let dl = db[l / 2];
                    let idx = qs[l] as usize | ((qh << (8 - 2 * l)) & 0x300);
                    let grid = IQ2S_GRID[idx].to_le_bytes();
                    for ((y, &g), &mask) in y8.iter_mut().zip(&grid).zip(&KMASK_IQ2XS) {
                        *y = signed(dl * g as f32, signs[l] & mask != 0);
                    }
                }
            }
        }
    }
}

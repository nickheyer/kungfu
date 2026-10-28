use crate::types::grids::{IQ3S_GRID, KMASK_IQ2XS};
use crate::types::{f16_at, signed};
use crate::{Quant, TypeInfo};

pub static INFO: TypeInfo = TypeInfo { id: 21, name: "iq3_s", block: 256, size: 110 };
pub struct Iq3S;

impl Quant for Iq3S {
    fn info(&self) -> &'static TypeInfo {
        &INFO
    }
    fn dequant_row(&self, src: &[u8], dst: &mut [f32]) {
        for (b, out) in dst.chunks_exact_mut(256).enumerate() {
            let blk = &src[b * 110..(b + 1) * 110];
            let d = f16_at(blk, 0);
            for (ib, ys) in out.chunks_exact_mut(32).enumerate() {
                let qs = &blk[2 + 8 * ib..10 + 8 * ib];
                let qh = blk[66 + ib] as usize;
                let signs = &blk[74 + 4 * ib..78 + 4 * ib];
                let sc = (blk[106 + ib / 2] >> (4 * (ib % 2))) & 0xf;
                let db = d * (1 + 2 * sc as i32) as f32;
                for (l, y8) in ys.chunks_exact_mut(8).enumerate() {
                    let g1 =
                        IQ3S_GRID[qs[2 * l] as usize | ((qh << (8 - 2 * l)) & 256)].to_le_bytes();
                    let g2 = IQ3S_GRID[qs[2 * l + 1] as usize | ((qh << (7 - 2 * l)) & 256)]
                        .to_le_bytes();
                    for j in 0..4 {
                        y8[j] = signed(db * g1[j] as f32, signs[l] & KMASK_IQ2XS[j] != 0);
                        y8[j + 4] = signed(db * g2[j] as f32, signs[l] & KMASK_IQ2XS[j + 4] != 0);
                    }
                }
            }
        }
    }
}

use crate::types::grids::IQ1S_GRID;
use crate::types::{IQ1S_DELTA, f16_at, u16_at};
use crate::{Quant, TypeInfo};

pub static INFO: TypeInfo = TypeInfo { id: 19, name: "iq1_s", block: 256, size: 50 };
pub struct Iq1S;

impl Quant for Iq1S {
    fn info(&self) -> &'static TypeInfo {
        &INFO
    }
    fn dequant_row(&self, src: &[u8], dst: &mut [f32]) {
        for (b, out) in dst.chunks_exact_mut(256).enumerate() {
            let blk = &src[b * 50..(b + 1) * 50];
            let d = f16_at(blk, 0);
            for (ib, ys) in out.chunks_exact_mut(32).enumerate() {
                let qs = &blk[2 + 4 * ib..6 + 4 * ib];
                let qh = u16_at(blk, 34 + 2 * ib) as usize;
                let dl = d * (2 * ((qh >> 12) & 7) + 1) as f32;
                let delta = if qh & 0x8000 != 0 { -IQ1S_DELTA } else { IQ1S_DELTA };
                for (l, y8) in ys.chunks_exact_mut(8).enumerate() {
                    let grid =
                        IQ1S_GRID[qs[l] as usize | (((qh >> (3 * l)) & 7) << 8)].to_le_bytes();
                    for (y, &g) in y8.iter_mut().zip(&grid) {
                        *y = dl * ((g as i8) as f32 + delta);
                    }
                }
            }
        }
    }
}

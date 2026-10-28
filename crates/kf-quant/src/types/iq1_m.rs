use crate::types::grids::IQ1S_GRID;
use crate::types::{IQ1S_DELTA, u16_at};
use crate::{Quant, TypeInfo};

pub static INFO: TypeInfo = TypeInfo { id: 29, name: "iq1_m", block: 256, size: 56 };
pub struct Iq1M;

impl Quant for Iq1M {
    fn info(&self) -> &'static TypeInfo {
        &INFO
    }
    fn dequant_row(&self, src: &[u8], dst: &mut [f32]) {
        for (b, out) in dst.chunks_exact_mut(256).enumerate() {
            let blk = &src[b * 56..(b + 1) * 56];
            let sc = [u16_at(blk, 48), u16_at(blk, 50), u16_at(blk, 52), u16_at(blk, 54)];
            let scale = (sc[0] >> 12)
                | ((sc[1] >> 8) & 0x00f0)
                | ((sc[2] >> 4) & 0x0f00)
                | (sc[3] & 0xf000);
            let d = half::f16::from_bits(scale).to_f32();
            for (ib, ys) in out.chunks_exact_mut(32).enumerate() {
                let s = sc[ib / 2] as i32;
                let dl1 = d * (2 * ((s >> (6 * (ib % 2))) & 0x7) + 1) as f32;
                let dl2 = d * (2 * ((s >> (6 * (ib % 2) + 3)) & 0x7) + 1) as f32;
                let qs = &blk[4 * ib..4 * ib + 4];
                let qh = [blk[32 + 2 * ib] as usize, blk[33 + 2 * ib] as usize];
                let idx = [
                    qs[0] as usize | ((qh[0] << 8) & 0x700),
                    qs[1] as usize | ((qh[0] << 4) & 0x700),
                    qs[2] as usize | ((qh[1] << 8) & 0x700),
                    qs[3] as usize | ((qh[1] << 4) & 0x700),
                ];
                let delta = [
                    if qh[0] & 0x08 != 0 { -IQ1S_DELTA } else { IQ1S_DELTA },
                    if qh[0] & 0x80 != 0 { -IQ1S_DELTA } else { IQ1S_DELTA },
                    if qh[1] & 0x08 != 0 { -IQ1S_DELTA } else { IQ1S_DELTA },
                    if qh[1] & 0x80 != 0 { -IQ1S_DELTA } else { IQ1S_DELTA },
                ];
                for (l, y8) in ys.chunks_exact_mut(8).enumerate() {
                    let dl = if l < 2 { dl1 } else { dl2 };
                    let grid = IQ1S_GRID[idx[l]].to_le_bytes();
                    for (y, &g) in y8.iter_mut().zip(&grid) {
                        *y = dl * ((g as i8) as f32 + delta[l]);
                    }
                }
            }
        }
    }
}

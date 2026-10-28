use crate::types::grids::KVALUES_IQ4NL;
use crate::types::{f16_at, u16_at};
use crate::{Quant, TypeInfo};

pub static INFO: TypeInfo = TypeInfo { id: 23, name: "iq4_xs", block: 256, size: 136 };
pub struct Iq4Xs;

/// Signed 6 bit scale of sub block ib
fn sub_scale(blk: &[u8], ib: usize) -> i32 {
    let scales_h = u16_at(blk, 2) as i32;
    let lo = ((blk[4 + ib / 2] >> (4 * (ib % 2))) & 0xf) as i32;
    (lo | (((scales_h >> (2 * ib)) & 3) << 4)) - 32
}

impl Quant for Iq4Xs {
    fn info(&self) -> &'static TypeInfo {
        &INFO
    }
    fn dequant_row(&self, src: &[u8], dst: &mut [f32]) {
        for (b, out) in dst.chunks_exact_mut(256).enumerate() {
            let blk = &src[b * 136..(b + 1) * 136];
            let d = f16_at(blk, 0);
            for (ib, ys) in out.chunks_exact_mut(32).enumerate() {
                let dl = d * sub_scale(blk, ib) as f32;
                let qs = &blk[8 + 16 * ib..24 + 16 * ib];
                let (lo, hi) = ys.split_at_mut(16);
                for ((y0, y1), &q) in lo.iter_mut().zip(hi.iter_mut()).zip(qs) {
                    *y0 = dl * KVALUES_IQ4NL[(q & 0xf) as usize] as f32;
                    *y1 = dl * KVALUES_IQ4NL[(q >> 4) as usize] as f32;
                }
            }
        }
    }
    fn dot_row(&self, src: &[u8], x: &[f32]) -> f32 {
        let mut acc = 0.0f32;
        for (b, xs) in x.chunks_exact(256).enumerate() {
            let blk = &src[b * 136..(b + 1) * 136];
            let d = f16_at(blk, 0);
            for (ib, xb) in xs.chunks_exact(32).enumerate() {
                let qs = &blk[8 + 16 * ib..24 + 16 * ib];
                let (x0, x1) = xb.split_at(16);
                let mut s = 0.0f32;
                for ((&a, &c), &q) in x0.iter().zip(x1).zip(qs) {
                    s += KVALUES_IQ4NL[(q & 0xf) as usize] as f32 * a;
                    s += KVALUES_IQ4NL[(q >> 4) as usize] as f32 * c;
                }
                acc += d * sub_scale(blk, ib) as f32 * s;
            }
        }
        acc
    }
}

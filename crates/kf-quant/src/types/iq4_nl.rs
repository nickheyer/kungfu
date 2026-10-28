use crate::types::f16_at;
use crate::types::grids::KVALUES_IQ4NL;
use crate::{Quant, TypeInfo};

pub static INFO: TypeInfo = TypeInfo { id: 20, name: "iq4_nl", block: 32, size: 18 };
pub struct Iq4Nl;

impl Quant for Iq4Nl {
    fn info(&self) -> &'static TypeInfo {
        &INFO
    }
    fn dequant_row(&self, src: &[u8], dst: &mut [f32]) {
        for (b, out) in dst.chunks_exact_mut(32).enumerate() {
            let blk = &src[b * 18..(b + 1) * 18];
            let d = f16_at(blk, 0);
            let (lo, hi) = out.split_at_mut(16);
            for ((y0, y1), &q) in lo.iter_mut().zip(hi.iter_mut()).zip(&blk[2..18]) {
                *y0 = d * KVALUES_IQ4NL[(q & 0xf) as usize] as f32;
                *y1 = d * KVALUES_IQ4NL[(q >> 4) as usize] as f32;
            }
        }
    }
    fn dot_row(&self, src: &[u8], x: &[f32]) -> f32 {
        let mut acc = 0.0f32;
        for (b, xs) in x.chunks_exact(32).enumerate() {
            let blk = &src[b * 18..(b + 1) * 18];
            let d = f16_at(blk, 0);
            let (x0, x1) = xs.split_at(16);
            let mut s = 0.0f32;
            for ((&a, &c), &q) in x0.iter().zip(x1).zip(&blk[2..18]) {
                s += KVALUES_IQ4NL[(q & 0xf) as usize] as f32 * a;
                s += KVALUES_IQ4NL[(q >> 4) as usize] as f32 * c;
            }
            acc += d * s;
        }
        acc
    }
}

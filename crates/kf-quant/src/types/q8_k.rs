use crate::types::u32_at;
use crate::{Quant, TypeInfo};

pub static INFO: TypeInfo = TypeInfo { id: 15, name: "q8_K", block: 256, size: 292 };
pub struct Q8K;

impl Quant for Q8K {
    fn info(&self) -> &'static TypeInfo {
        &INFO
    }
    fn dequant_row(&self, src: &[u8], dst: &mut [f32]) {
        for (b, out) in dst.chunks_exact_mut(256).enumerate() {
            let blk = &src[b * 292..(b + 1) * 292];
            let d = f32::from_bits(u32_at(blk, 0));
            for (y, &q) in out.iter_mut().zip(&blk[4..260]) {
                *y = d * (q as i8) as f32;
            }
        }
    }
    fn dot_row(&self, src: &[u8], x: &[f32]) -> f32 {
        let mut acc = 0.0f32;
        for (b, xs) in x.chunks_exact(256).enumerate() {
            let blk = &src[b * 292..(b + 1) * 292];
            let d = f32::from_bits(u32_at(blk, 0));
            let mut s = 0.0f32;
            for (&v, &q) in xs.iter().zip(&blk[4..260]) {
                s += (q as i8) as f32 * v;
            }
            acc += d * s;
        }
        acc
    }
}

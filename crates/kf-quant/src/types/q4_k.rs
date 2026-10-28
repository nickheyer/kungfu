use crate::types::{f16_at, scale_min_k4};
use crate::{Quant, TypeInfo};

pub static INFO: TypeInfo = TypeInfo { id: 12, name: "q4_K", block: 256, size: 144 };
pub struct Q4K;

impl Quant for Q4K {
    fn info(&self) -> &'static TypeInfo {
        &INFO
    }
    fn dequant_row(&self, src: &[u8], dst: &mut [f32]) {
        for (b, out) in dst.chunks_exact_mut(256).enumerate() {
            let blk = &src[b * 144..(b + 1) * 144];
            let d = f16_at(blk, 0);
            let min = f16_at(blk, 2);
            let scales = &blk[4..16];
            for (j, ys) in out.chunks_exact_mut(64).enumerate() {
                let q = &blk[16 + 32 * j..48 + 32 * j];
                let (sc, m) = scale_min_k4(2 * j, scales);
                let d1 = d * sc as f32;
                let m1 = min * m as f32;
                let (sc, m) = scale_min_k4(2 * j + 1, scales);
                let d2 = d * sc as f32;
                let m2 = min * m as f32;
                let (lo, hi) = ys.split_at_mut(32);
                for ((y0, y1), &v) in lo.iter_mut().zip(hi.iter_mut()).zip(q) {
                    *y0 = d1 * (v & 0xF) as f32 - m1;
                    *y1 = d2 * (v >> 4) as f32 - m2;
                }
            }
        }
    }
    fn dot_row(&self, src: &[u8], x: &[f32]) -> f32 {
        let mut acc = 0.0f32;
        for (b, xs) in x.chunks_exact(256).enumerate() {
            let blk = &src[b * 144..(b + 1) * 144];
            let d = f16_at(blk, 0);
            let min = f16_at(blk, 2);
            let scales = &blk[4..16];
            for (j, xj) in xs.chunks_exact(64).enumerate() {
                let q = &blk[16 + 32 * j..48 + 32 * j];
                let (sc1, m1) = scale_min_k4(2 * j, scales);
                let (sc2, m2) = scale_min_k4(2 * j + 1, scales);
                let (x1, x2) = xj.split_at(32);
                let (mut qx1, mut sx1, mut qx2, mut sx2) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                for ((&a, &c), &v) in x1.iter().zip(x2).zip(q) {
                    qx1 += (v & 0xF) as f32 * a;
                    sx1 += a;
                    qx2 += (v >> 4) as f32 * c;
                    sx2 += c;
                }
                acc += d * sc1 as f32 * qx1 - min * m1 as f32 * sx1;
                acc += d * sc2 as f32 * qx2 - min * m2 as f32 * sx2;
            }
        }
        acc
    }
}

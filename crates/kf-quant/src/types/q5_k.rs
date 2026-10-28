use crate::types::{f16_at, scale_min_k4};
use crate::{Quant, TypeInfo};

pub static INFO: TypeInfo = TypeInfo { id: 13, name: "q5_K", block: 256, size: 176 };
pub struct Q5K;

impl Quant for Q5K {
    fn info(&self) -> &'static TypeInfo {
        &INFO
    }
    fn dequant_row(&self, src: &[u8], dst: &mut [f32]) {
        for (b, out) in dst.chunks_exact_mut(256).enumerate() {
            let blk = &src[b * 176..(b + 1) * 176];
            let d = f16_at(blk, 0);
            let min = f16_at(blk, 2);
            let scales = &blk[4..16];
            let qh = &blk[16..48];
            let mut u1 = 1u8;
            let mut u2 = 2u8;
            for (j, ys) in out.chunks_exact_mut(64).enumerate() {
                let ql = &blk[48 + 32 * j..80 + 32 * j];
                let (sc, m) = scale_min_k4(2 * j, scales);
                let d1 = d * sc as f32;
                let m1 = min * m as f32;
                let (sc, m) = scale_min_k4(2 * j + 1, scales);
                let d2 = d * sc as f32;
                let m2 = min * m as f32;
                let (lo, hi) = ys.split_at_mut(32);
                for (l, (y0, y1)) in lo.iter_mut().zip(hi.iter_mut()).enumerate() {
                    let h1 = if qh[l] & u1 != 0 { 16 } else { 0 };
                    let h2 = if qh[l] & u2 != 0 { 16 } else { 0 };
                    *y0 = d1 * ((ql[l] & 0xF) as i32 + h1) as f32 - m1;
                    *y1 = d2 * ((ql[l] >> 4) as i32 + h2) as f32 - m2;
                }
                u1 <<= 2;
                u2 <<= 2;
            }
        }
    }
    fn dot_row(&self, src: &[u8], x: &[f32]) -> f32 {
        let mut acc = 0.0f32;
        for (b, xs) in x.chunks_exact(256).enumerate() {
            let blk = &src[b * 176..(b + 1) * 176];
            let d = f16_at(blk, 0);
            let min = f16_at(blk, 2);
            let scales = &blk[4..16];
            let qh = &blk[16..48];
            for (j, xj) in xs.chunks_exact(64).enumerate() {
                let ql = &blk[48 + 32 * j..80 + 32 * j];
                let (sc1, m1) = scale_min_k4(2 * j, scales);
                let (sc2, m2) = scale_min_k4(2 * j + 1, scales);
                let (x1, x2) = xj.split_at(32);
                let (mut qx1, mut sx1, mut qx2, mut sx2) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                for l in 0..32 {
                    let h1 = ((qh[l] >> (2 * j)) & 1) << 4;
                    let h2 = ((qh[l] >> (2 * j + 1)) & 1) << 4;
                    qx1 += ((ql[l] & 0xF) | h1) as f32 * x1[l];
                    sx1 += x1[l];
                    qx2 += ((ql[l] >> 4) | h2) as f32 * x2[l];
                    sx2 += x2[l];
                }
                acc += d * sc1 as f32 * qx1 - min * m1 as f32 * sx1;
                acc += d * sc2 as f32 * qx2 - min * m2 as f32 * sx2;
            }
        }
        acc
    }
}

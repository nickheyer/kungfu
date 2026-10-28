use crate::types::f16_at;
use crate::{Quant, TypeInfo};

pub static INFO: TypeInfo = TypeInfo { id: 14, name: "q6_K", block: 256, size: 210 };
pub struct Q6K;

/// Four signed 6 bit quants at offset l of the 128 element half using its ql and qh slices
fn quads(ql: &[u8], qh: &[u8], l: usize) -> [i8; 4] {
    [
        ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i8 - 32,
        ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i8 - 32,
        ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8 - 32,
        ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8 - 32,
    ]
}

impl Quant for Q6K {
    fn info(&self) -> &'static TypeInfo {
        &INFO
    }
    fn dequant_row(&self, src: &[u8], dst: &mut [f32]) {
        for (b, out) in dst.chunks_exact_mut(256).enumerate() {
            let blk = &src[b * 210..(b + 1) * 210];
            let d = f16_at(blk, 208);
            for (n, ys) in out.chunks_exact_mut(128).enumerate() {
                let ql = &blk[64 * n..64 * n + 64];
                let qh = &blk[128 + 32 * n..160 + 32 * n];
                let sc = &blk[192 + 8 * n..200 + 8 * n];
                for l in 0..32 {
                    let is = l / 16;
                    for (k, q) in quads(ql, qh, l).into_iter().enumerate() {
                        ys[l + 32 * k] = d * (sc[is + 2 * k] as i8) as f32 * q as f32;
                    }
                }
            }
        }
    }
    fn dot_row(&self, src: &[u8], x: &[f32]) -> f32 {
        let mut acc = 0.0f32;
        for (b, xs) in x.chunks_exact(256).enumerate() {
            let blk = &src[b * 210..(b + 1) * 210];
            let d = f16_at(blk, 208);
            let mut sums = [0.0f32; 16];
            for (n, xn) in xs.chunks_exact(128).enumerate() {
                let ql = &blk[64 * n..64 * n + 64];
                let qh = &blk[128 + 32 * n..160 + 32 * n];
                for l in 0..32 {
                    let is = 8 * n + l / 16;
                    for (k, q) in quads(ql, qh, l).into_iter().enumerate() {
                        sums[is + 2 * k] += q as f32 * xn[l + 32 * k];
                    }
                }
            }
            for (s, &sc) in sums.iter().zip(&blk[192..208]) {
                acc += d * (sc as i8) as f32 * s;
            }
        }
        acc
    }
}

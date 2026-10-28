use crate::types::{f16_at, u32_at};
use crate::{Quant, TypeInfo};

pub static INFO: TypeInfo = TypeInfo { id: 11, name: "q3_K", block: 256, size: 110 };
pub struct Q3K;

const KMASK1: u32 = 0x0303_0303;
const KMASK2: u32 = 0x0f0f_0f0f;

/// Unpacks the twelve scale bytes into sixteen 6 bit scales stored as bytes
fn scales(blk: &[u8]) -> [i8; 16] {
    let a0 = u32_at(blk, 96);
    let a1 = u32_at(blk, 100);
    let tmp = u32_at(blk, 104);
    let aux = [
        (a0 & KMASK2) | ((tmp & KMASK1) << 4),
        (a1 & KMASK2) | (((tmp >> 2) & KMASK1) << 4),
        ((a0 >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4),
        ((a1 >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4),
    ];
    let mut out = [0i8; 16];
    for (dst, a) in out.chunks_exact_mut(4).zip(aux) {
        for (o, byte) in dst.iter_mut().zip(a.to_le_bytes()) {
            *o = byte as i8;
        }
    }
    out
}

impl Quant for Q3K {
    fn info(&self) -> &'static TypeInfo {
        &INFO
    }
    fn dequant_row(&self, src: &[u8], dst: &mut [f32]) {
        for (b, out) in dst.chunks_exact_mut(256).enumerate() {
            let blk = &src[b * 110..(b + 1) * 110];
            let d_all = f16_at(blk, 108);
            let hm = &blk[0..32];
            let sc = scales(blk);
            let mut m = 1u8;
            let mut y = 0;
            let mut is = 0;
            for n in 0..2 {
                let q = &blk[32 + 32 * n..64 + 32 * n];
                for shift in [0, 2, 4, 6] {
                    for h in 0..2 {
                        let dl = d_all * (sc[is] as i32 - 32) as f32;
                        is += 1;
                        for l in 16 * h..16 * h + 16 {
                            let hi = if hm[l] & m != 0 { 0 } else { 4 };
                            out[y] = dl * (((q[l] >> shift) & 3) as i32 - hi) as f32;
                            y += 1;
                        }
                    }
                    m <<= 1;
                }
            }
        }
    }
}

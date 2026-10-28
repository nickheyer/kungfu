use crate::types::f16_at;
use crate::{Quant, TypeInfo};

pub static INFO: TypeInfo = TypeInfo { id: 34, name: "tq1_0", block: 256, size: 54 };
pub struct Tq1_0;

const POW3: [u8; 6] = [1, 3, 9, 27, 81, 243];

/// Dequantized base 3 digit n of a packed ternary byte
fn trit(q: u8, n: usize, d: f32) -> f32 {
    let q = q.wrapping_mul(POW3[n]);
    let xi = ((q as u16 * 3) >> 8) as i16;
    (xi - 1) as f32 * d
}

impl Quant for Tq1_0 {
    fn info(&self) -> &'static TypeInfo {
        &INFO
    }
    fn dequant_row(&self, src: &[u8], dst: &mut [f32]) {
        for (b, out) in dst.chunks_exact_mut(256).enumerate() {
            let blk = &src[b * 54..(b + 1) * 54];
            let d = f16_at(blk, 52);
            let mut y = 0;
            for (start, width) in [(0, 32), (32, 16)] {
                for n in 0..5 {
                    for &q in &blk[start..start + width] {
                        out[y] = trit(q, n, d);
                        y += 1;
                    }
                }
            }
            for n in 0..4 {
                for &q in &blk[48..52] {
                    out[y] = trit(q, n, d);
                    y += 1;
                }
            }
        }
    }
}

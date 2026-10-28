use crate::{Quant, TypeInfo};

pub static INFO: TypeInfo = TypeInfo { id: 30, name: "bf16", block: 1, size: 2 };
pub struct Bf16;

impl Quant for Bf16 {
    fn info(&self) -> &'static TypeInfo {
        &INFO
    }
    fn dequant_row(&self, src: &[u8], dst: &mut [f32]) {
        for (i, d) in dst.iter_mut().enumerate() {
            *d = half::bf16::from_le_bytes([src[2 * i], src[2 * i + 1]]).to_f32();
        }
    }
}

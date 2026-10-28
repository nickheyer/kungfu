use crate::{Quant, TypeInfo};

pub static INFO: TypeInfo = TypeInfo { id: 0, name: "f32", block: 1, size: 4 };
pub struct F32;

impl Quant for F32 {
    fn info(&self) -> &'static TypeInfo {
        &INFO
    }
    fn dequant_row(&self, src: &[u8], dst: &mut [f32]) {
        for (i, d) in dst.iter_mut().enumerate() {
            *d = f32::from_le_bytes([src[4 * i], src[4 * i + 1], src[4 * i + 2], src[4 * i + 3]]);
        }
    }
}

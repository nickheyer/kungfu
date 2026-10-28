use crate::types::f16_at;
use crate::{Quant, TypeInfo};

pub static INFO: TypeInfo = TypeInfo { id: 1, name: "f16", block: 1, size: 2 };
pub struct F16;

impl Quant for F16 {
    fn info(&self) -> &'static TypeInfo {
        &INFO
    }
    fn dequant_row(&self, src: &[u8], dst: &mut [f32]) {
        for (i, d) in dst.iter_mut().enumerate() {
            *d = f16_at(src, 2 * i);
        }
    }
}

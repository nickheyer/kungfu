pub mod types;

/// Block layout of one ggml tensor type
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TypeInfo {
    pub id: u32,
    pub name: &'static str,
    pub block: usize,
    pub size: usize,
}

impl TypeInfo {
    /// Bytes a row of n elements takes
    pub fn row_bytes(&self, n: usize) -> usize {
        n / self.block * self.size
    }
}

/// One quantization type, a block layout plus the math on it
pub trait Quant: Send + Sync {
    fn info(&self) -> &'static TypeInfo;
    /// Expands one row of whole blocks into f32
    fn dequant_row(&self, src: &[u8], dst: &mut [f32]);
    /// Dot product of one quantized row against f32, dequant then multiply unless overridden
    fn dot_row(&self, src: &[u8], x: &[f32]) -> f32 {
        let mut tmp = vec![0.0f32; x.len()];
        self.dequant_row(src, &mut tmp);
        tmp.iter().zip(x).map(|(a, b)| a * b).sum()
    }
}

/// How the reference rounds activations before a dot product against this weight type
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActKind {
    F32,
    F16,
    Bf16,
    Q8_0,
    Q8_1,
    Q8K,
}

/// The activation rounding the reference pairs with a weight type
pub fn act_kind(q: &dyn Quant) -> ActKind {
    match q.info().name {
        "f32" => ActKind::F32,
        "f16" => ActKind::F16,
        "bf16" => ActKind::Bf16,
        "q4_1" | "q5_1" | "q8_1" => ActKind::Q8_1,
        "q1_0" | "q2_0" | "q4_0" | "q5_0" | "q8_0" | "mxfp4" | "nvfp4" | "iq4_nl" => ActKind::Q8_0,
        _ => ActKind::Q8K,
    }
}

/// Rounds to the nearest integer with ties to even, the reference's nearest_int
fn nearest_int(v: f32) -> i32 {
    v.round_ties_even() as i32
}

/// Activations rounded the way the reference quantizes them before the dot product
pub fn round_activations(kind: ActKind, x: &[f32]) -> Vec<f32> {
    match kind {
        ActKind::F32 => x.to_vec(),
        ActKind::F16 => x.iter().map(|v| half::f16::from_f32(*v).to_f32()).collect(),
        ActKind::Bf16 => x.iter().map(|v| half::bf16::from_f32(*v).to_f32()).collect(),
        ActKind::Q8_0 | ActKind::Q8_1 => {
            let mut out = Vec::with_capacity(x.len());
            for blk in x.chunks(32) {
                let amax = blk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                let d = amax / 127.0;
                let id = if d != 0.0 { 1.0 / d } else { 0.0 };
                let dh = half::f16::from_f32(d).to_f32();
                out.extend(blk.iter().map(|v| dh * (v * id).round()));
            }
            out
        }
        ActKind::Q8K => {
            let mut out = Vec::with_capacity(x.len());
            for blk in x.chunks(256) {
                let mut amax = 0.0f32;
                let mut max = 0.0f32;
                for v in blk {
                    if v.abs() > amax {
                        amax = v.abs();
                        max = *v;
                    }
                }
                if amax == 0.0 {
                    out.extend(std::iter::repeat_n(0.0, blk.len()));
                    continue;
                }
                let iscale = -127.0 / max;
                let d = 1.0 / iscale;
                out.extend(blk.iter().map(|v| d * nearest_int(iscale * v).min(127) as f32));
            }
            out
        }
    }
}

pub fn by_id(id: u32) -> Option<&'static dyn Quant> {
    types::all().iter().copied().find(|q| q.info().id == id)
}

pub fn by_name(name: &str) -> Option<&'static dyn Quant> {
    types::all().iter().copied().find(|q| q.info().name == name)
}

pub fn info(id: u32) -> Option<&'static TypeInfo> {
    by_id(id).map(|q| q.info())
}

/// Bytes of a tensor whose innermost dimension is dims[0]
pub fn nbytes(id: u32, dims: &[usize]) -> Option<usize> {
    let info = info(id)?;
    let inner = *dims.first()?;
    if inner % info.block != 0 {
        return None;
    }
    let rows: usize = dims.iter().skip(1).product::<usize>().max(1);
    Some(info.row_bytes(inner) * rows)
}

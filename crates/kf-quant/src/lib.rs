pub mod types;

#[cfg(target_arch = "x86_64")]
pub(crate) mod simd;

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
    /// Dot products of one quantized row against every row of int8 activations of its kind, y[r] receiving row r
    fn dot_i8(&self, src: &[u8], x: &ActI8, y: &mut [f32]) {
        dot_i8_generic(self, src, x, y)
    }
}

/// The reference dot of one row against every activation row, the integers scaled back to the floats the reference multiplies
pub fn dot_i8_generic<Q: Quant + ?Sized>(q: &Q, src: &[u8], x: &ActI8, y: &mut [f32]) {
    for r in 0..x.rows {
        y[r] = q.dot_row(src, &x.dequant_row(r));
    }
}

/// Activations in the integer form the reference rounds them to, int8 values with a scale and the integer sum per sixteen
pub struct ActI8 {
    pub rows: usize,
    pub n: usize,
    pub q: Vec<i8>,
    pub d: Vec<f32>,
    pub sum: Vec<i32>,
}

impl ActI8 {
    pub fn row_q(&self, r: usize) -> &[i8] {
        &self.q[r * self.n..(r + 1) * self.n]
    }

    /// Scale of every sixteen of row r
    pub fn row_d(&self, r: usize) -> &[f32] {
        let n16 = self.n / 16;
        &self.d[r * n16..(r + 1) * n16]
    }

    /// Integer sum of every sixteen of row r
    pub fn row_sum(&self, r: usize) -> &[i32] {
        let n16 = self.n / 16;
        &self.sum[r * n16..(r + 1) * n16]
    }

    /// Row r as the floats the reference multiplies, scale times integer
    pub fn dequant_row(&self, r: usize) -> Vec<f32> {
        let d = self.row_d(r);
        self.row_q(r).iter().enumerate().map(|(j, q)| d[j / 16] * *q as f32).collect()
    }
}

/// Rows of activations quantized the way round_activations rounds them, the same integers it would multiply back
pub fn quantize_activations(kind: ActKind, x: &[f32], rows: usize) -> Option<ActI8> {
    let n = if rows == 0 { 0 } else { x.len() / rows };
    if !n.is_multiple_of(16) {
        return None;
    }
    let n16 = n / 16;
    let mut out = ActI8 { rows, n, q: vec![0; rows * n], d: vec![0.0; rows * n16], sum: vec![0; rows * n16] };
    let block = match kind {
        ActKind::Q8_0 | ActKind::Q8_1 => 32,
        ActKind::Q8K => 256,
        ActKind::F32 | ActKind::F16 | ActKind::Bf16 => return None,
    };
    if !n.is_multiple_of(block) {
        return None;
    }
    for r in 0..rows {
        let xr = &x[r * n..(r + 1) * n];
        let qr = &mut out.q[r * n..(r + 1) * n];
        for (bi, blk) in xr.chunks_exact(block).enumerate() {
            let qb = &mut qr[bi * block..(bi + 1) * block];
            let d = match kind {
                ActKind::Q8_0 | ActKind::Q8_1 => {
                    let amax = blk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                    let d = amax / 127.0;
                    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
                    for (q, v) in qb.iter_mut().zip(blk) {
                        *q = (v * id).round() as i8;
                    }
                    half::f16::from_f32(d).to_f32()
                }
                _ => {
                    let mut amax = 0.0f32;
                    let mut max = 0.0f32;
                    for v in blk {
                        if v.abs() > amax {
                            amax = v.abs();
                            max = *v;
                        }
                    }
                    if amax == 0.0 {
                        qb.fill(0);
                        0.0
                    } else {
                        let iscale = -127.0 / max;
                        for (q, v) in qb.iter_mut().zip(blk) {
                            *q = nearest_int(iscale * v).min(127) as i8;
                        }
                        1.0 / iscale
                    }
                }
            };
            for g in 0..block / 16 {
                let at = r * n16 + bi * (block / 16) + g;
                out.d[at] = d;
                out.sum[at] = qb[g * 16..g * 16 + 16].iter().map(|v| *v as i32).sum();
            }
        }
    }
    Some(out)
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

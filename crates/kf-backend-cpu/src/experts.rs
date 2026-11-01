//! Routed expert products on int8 activations, the host side of the expert coprocessor
use crate::ops;
use anyhow::{Result, anyhow, ensure};
use kf_core::{Act, GluLimit};
use kf_quant::{ActI8, ActKind, Quant};

/// A stacked expert tensor as the host reads it, every expert's rows back to back
#[derive(Clone, Copy)]
pub struct ExpertTensor<'a> {
    pub q: &'static dyn Quant,
    pub bytes: &'a [u8],
    pub n_in: usize,
    pub n_out: usize,
}

impl<'a> ExpertTensor<'a> {
    pub fn new(ty: u32, bytes: &'a [u8], n_in: usize, n_out: usize) -> Result<Self> {
        let q = kf_quant::by_id(ty).ok_or_else(|| anyhow!("unknown ggml type {ty}"))?;
        ensure!(n_in.is_multiple_of(q.info().block), "{} columns against blocks of {}", n_in, q.info().block);
        Ok(Self { q, bytes, n_in, n_out })
    }

    pub fn row_bytes(&self) -> usize {
        self.q.info().row_bytes(self.n_in)
    }

    pub fn expert_bytes(&self) -> usize {
        self.row_bytes() * self.n_out
    }

    /// Row o of expert e
    pub fn row(&self, e: usize, o: usize) -> &'a [u8] {
        let rb = self.row_bytes();
        let at = e * self.expert_bytes() + o * rb;
        &self.bytes[at..at + rb]
    }
}

/// Activation rows in the form a weight type's dot consumes, int8 blocks for block types and rounded floats for float types
pub enum Rows {
    I8(ActI8),
    F32 { rows: usize, n: usize, x: Vec<f32> },
}

impl Rows {
    /// Rows of n columns rounded or quantized the way the reference feeds them to a weight of this kind
    pub fn new(kind: ActKind, x: &[f32], rows: usize) -> Result<Self> {
        match kind {
            ActKind::F32 | ActKind::F16 | ActKind::Bf16 => Ok(Rows::F32 { rows, n: if rows == 0 { 0 } else { x.len() / rows }, x: kf_quant::round_activations(kind, x) }),
            _ => kf_quant::quantize_activations(kind, x, rows).map(Rows::I8).ok_or_else(|| anyhow!("{} columns have no int8 form for {kind:?} activations", if rows == 0 { 0 } else { x.len() / rows })),
        }
    }

    pub fn rows(&self) -> usize {
        match self {
            Rows::I8(a) => a.rows,
            Rows::F32 { rows, .. } => *rows,
        }
    }

    /// The listed rows as rows of their own
    pub fn gather(&self, rows: &[usize]) -> Rows {
        match self {
            Rows::I8(x) => {
                let n16 = x.n / 16;
                let mut out = ActI8 { rows: rows.len(), n: x.n, q: Vec::with_capacity(rows.len() * x.n), d: Vec::with_capacity(rows.len() * n16), sum: Vec::with_capacity(rows.len() * n16) };
                for &r in rows {
                    out.q.extend_from_slice(x.row_q(r));
                    out.d.extend_from_slice(x.row_d(r));
                    out.sum.extend_from_slice(x.row_sum(r));
                }
                Rows::I8(out)
            }
            Rows::F32 { n, x, .. } => Rows::F32 { rows: rows.len(), n: *n, x: rows.iter().flat_map(|&r| x[r * n..(r + 1) * n].iter().copied()).collect() },
        }
    }

    /// One weight row against every row, y[r] receiving row r
    fn dot(&self, q: &dyn Quant, row: &[u8], y: &mut [f32]) {
        match self {
            Rows::I8(a) => q.dot_i8(row, a, y),
            Rows::F32 { rows, n, x } => {
                for r in 0..*rows {
                    y[r] = q.dot_row(row, &x[r * n..(r + 1) * n]);
                }
            }
        }
    }
}

/// Rows o0 to o1 of expert e's hidden activation for every input row, the gate through the activation times the up
#[allow(clippy::too_many_arguments)]
pub fn hidden_rows(gate: Option<&ExpertTensor>, up: &ExpertTensor, e: usize, o0: usize, o1: usize, xg: &Rows, xu: &Rows, act: Act, limit: Option<GluLimit>, n_ff: usize, hidden: &mut [f32]) {
    let rows = xu.rows();
    let mut g = vec![0.0f32; rows];
    let mut u = vec![0.0f32; rows];
    for o in o0..o1 {
        match gate {
            Some(gt) => {
                xg.dot(gt.q, gt.row(e, o), &mut g);
                xu.dot(up.q, up.row(e, o), &mut u);
            }
            None => {
                xu.dot(up.q, up.row(e, o), &mut g);
                xu.dot(up.q, up.row(e, n_ff + o), &mut u);
            }
        }
        for r in 0..rows {
            hidden[r * n_ff + o] = ops::gated(act, limit, g[r], u[r]);
        }
    }
}

/// Columns c0 to c1 of expert e's down projection of every hidden row, weighted and added into the row's pass row of out
#[allow(clippy::too_many_arguments)]
pub fn down_cols(down: &ExpertTensor, e: usize, c0: usize, c1: usize, hq: &Rows, weights: &[f32], rows: &[usize], n_embd: usize, out: &mut [f32]) {
    let mut y = vec![0.0f32; hq.rows()];
    for c in c0..c1 {
        hq.dot(down.q, down.row(e, c), &mut y);
        for k in 0..hq.rows() {
            out[rows[k] * n_embd + c] += weights[k] * y[k];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::moe::{ExpertWeights, expert_forward};
    use kf_core::{WeightSource, Wid};

    struct Mem(Vec<(u32, Vec<usize>, Vec<u8>)>);

    impl WeightSource for Mem {
        fn bytes(&self, w: Wid) -> Result<&[u8]> {
            Ok(&self.0[w.0 as usize].2)
        }
        fn ggml_type(&self, w: Wid) -> u32 {
            self.0[w.0 as usize].0
        }
        fn dims(&self, w: Wid) -> &[usize] {
            &self.0[w.0 as usize].1
        }
        fn name(&self, _w: Wid) -> &str {
            "t"
        }
        fn len(&self) -> usize {
            self.0.len()
        }
    }

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn unit(&mut self) -> f32 {
            (self.next() >> 40) as f32 / (1u64 << 24) as f32
        }
    }

    /// Random blocks of a type whose f16 scale sits at the block start, scales kept small and finite
    fn synth(rng: &mut Rng, q: &dyn Quant, rows: usize, n: usize) -> Vec<u8> {
        let info = q.info();
        let mut v: Vec<u8> = (0..rows * info.row_bytes(n)).map(|_| rng.next() as u8).collect();
        for blk in v.chunks_exact_mut(info.size) {
            let bits = (((5 + rng.next() % 6) << 10) | (rng.next() & 0x3ff)) as u16;
            blk[..2].copy_from_slice(&bits.to_le_bytes());
            if info.name == "q4_K" {
                blk[2..4].copy_from_slice(&bits.to_le_bytes());
            }
        }
        v
    }

    #[test]
    fn int8_experts_match_reference_rows() {
        for name in ["iq2_xxs", "iq4_xs", "q8_0", "f16"] {
            let q = kf_quant::by_name(name).expect(name);
            let (n_embd, n_ff, n_exp, rows) = (512, 256, 3, 2);
            let mut rng = Rng(0x7e5);
            let gate = synth(&mut rng, q, n_ff * n_exp, n_embd);
            let up = synth(&mut rng, q, n_ff * n_exp, n_embd);
            let down = synth(&mut rng, q, n_embd * n_exp, n_ff);
            let ty = q.info().id;
            let mem = Mem(vec![(ty, vec![n_embd, n_ff, n_exp], gate.clone()), (ty, vec![n_embd, n_ff, n_exp], up.clone()), (ty, vec![n_ff, n_embd, n_exp], down.clone())]);
            let gt = ExpertTensor::new(ty, &gate, n_embd, n_ff).expect("gate");
            let ut = ExpertTensor::new(ty, &up, n_embd, n_ff).expect("up");
            let dt = ExpertTensor::new(ty, &down, n_ff, n_embd).expect("down");
            let x: Vec<f32> = (0..rows * n_embd).map(|_| 2.0 * rng.unit() - 1.0).collect();
            let kind = kf_quant::act_kind(q);
            let xq = Rows::new(kind, &x, rows).expect("activations");
            let ew = ExpertWeights { gate: Some(Wid(0)), up: Wid(1), down: Some(Wid(2)), gate_b: None, up_b: None, down_b: None, fused: false, act: Act::Silu, limit: None };
            for e in 0..n_exp {
                let picked = [1usize, 0];
                let xe = xq.gather(&picked);
                let mut hidden = vec![0.0f32; picked.len() * n_ff];
                hidden_rows(Some(&gt), &ut, e, 0, n_ff, &xe, &xe, Act::Silu, None, n_ff, &mut hidden);
                let hq = Rows::new(kind, &hidden, picked.len()).expect("hidden");
                let weights = [0.75f32, 0.25];
                let mut out = vec![0.0f32; rows * n_embd];
                down_cols(&dt, e, 0, n_embd, &hq, &weights, &picked, n_embd, &mut out);
                for (k, &r) in picked.iter().enumerate() {
                    let want = expert_forward(&ew, &mem, e, &x[r * n_embd..(r + 1) * n_embd]).expect("reference");
                    let got = &out[r * n_embd..(r + 1) * n_embd];
                    let mag: f32 = want.iter().map(|v| v.abs()).sum::<f32>() / n_embd as f32;
                    let worst = got.iter().zip(&want).map(|(g, w)| (g - weights[k] * w).abs()).fold(0.0f32, f32::max);
                    assert!(worst <= 2e-2 * mag + 1e-5, "{name} expert {e} row {r}: largest difference {worst} against a mean magnitude of {mag}");
                }
            }
        }
    }
}

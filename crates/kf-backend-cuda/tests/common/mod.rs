#![allow(dead_code)]
use anyhow::{Result, anyhow};
use kf_backend_cpu::CpuBackend;
use kf_backend_cuda::CudaBackend;
use kf_core::{Backend, Graph, Inputs, WeightSource, Wid};
use std::path::{Path, PathBuf};

pub const MODELS: &str = "/tmp/claude-1000/-home-nick-code-kungfu/d8d84ac4-ebc2-4907-b8d6-c27b30a54cec/scratchpad/models";
pub const QWEN: &str = "Qwen3-0.6B-Q8_0.gguf";
pub const LLAMA: &str = "Llama-3.2-1B-Instruct-Q4_K_M.gguf";

pub struct Tensor {
    pub name: String,
    pub ty: u32,
    pub dims: Vec<usize>,
    pub bytes: Vec<u8>,
}

/// Weights held in memory, real tensors copied out of GGUF files plus synthetic ones
#[derive(Default)]
pub struct Mem {
    pub t: Vec<Tensor>,
}

impl Mem {
    pub fn add(&mut self, name: &str, ty: u32, dims: &[usize], bytes: Vec<u8>) -> Wid {
        assert_eq!(kf_quant::nbytes(ty, dims), Some(bytes.len()), "{name} byte count");
        self.t.push(Tensor { name: name.to_string(), ty, dims: dims.to_vec(), bytes });
        Wid(self.t.len() as u32 - 1)
    }

    pub fn f32(&mut self, name: &str, dims: &[usize], v: &[f32]) -> Wid {
        self.add(name, 0, dims, v.iter().flat_map(|x| x.to_le_bytes()).collect())
    }

    /// A tensor from a GGUF file, the first rows rows of it when rows is given
    pub fn gguf(&mut self, g: &kf_gguf::Gguf, name: &str, rows: Option<usize>) -> Wid {
        let t = g.tensor(name).unwrap_or_else(|| panic!("{name} missing"));
        let bytes = g.bytes(t).expect("tensor bytes");
        let mut dims = t.dims.clone();
        let take = match rows {
            Some(r) => {
                let rb = kf_quant::info(t.ggml_type).expect("type").row_bytes(dims[0]);
                dims[1] = r;
                r * rb
            }
            None => bytes.len(),
        };
        self.add(name, t.ggml_type, &dims, bytes[..take].to_vec())
    }
}

impl WeightSource for Mem {
    fn bytes(&self, w: Wid) -> Result<&[u8]> {
        self.t.get(w.0 as usize).map(|t| t.bytes.as_slice()).ok_or_else(|| anyhow!("no weight {}", w.0))
    }
    fn ggml_type(&self, w: Wid) -> u32 {
        self.t[w.0 as usize].ty
    }
    fn dims(&self, w: Wid) -> &[usize] {
        &self.t[w.0 as usize].dims
    }
    fn name(&self, w: Wid) -> &str {
        &self.t[w.0 as usize].name
    }
    fn len(&self) -> usize {
        self.t.len()
    }
}

/// Small deterministic generator for test data
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    pub fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
    pub fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.unit() * scale).collect()
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    /// Half precision bits with a random sign and mantissa and an exponent near one
    pub fn half(&mut self) -> u16 {
        let sign = (self.next() & 1) as u16;
        let exp = 8 + (self.next() % 7) as u16;
        let man = (self.next() & 0x3ff) as u16;
        (sign << 15) | (exp << 10) | man
    }
}

/// Byte offsets of the half precision scales inside one block of each quantized type
fn half_offsets(name: &str) -> &'static [usize] {
    match name {
        "q4_1" | "q5_1" | "q8_1" | "q4_K" | "q5_K" => &[0, 2],
        "q2_K" => &[80, 82],
        "q3_K" => &[108],
        "q6_K" => &[208],
        "tq1_0" => &[52],
        "tq2_0" => &[64],
        "mxfp4" | "nvfp4" | "iq1_m" | "q8_K" | "f32" | "f16" | "bf16" => &[],
        _ => &[0],
    }
}

/// Random but valid bytes for rows of n elements of a ggml type, scales patched to sane magnitudes
pub fn synth(rng: &mut Rng, ty: u32, rows: usize, n: usize) -> Vec<u8> {
    let info = kf_quant::info(ty).expect("type");
    match info.name {
        "f32" => return (0..rows * n).flat_map(|_| rng.unit().to_le_bytes()).collect(),
        "f16" => return (0..rows * n).flat_map(|_| rng.half().to_le_bytes()).collect(),
        "bf16" => return (0..rows * n).flat_map(|_| ((rng.unit().to_bits() >> 16) as u16).to_le_bytes()).collect(),
        _ => {}
    }
    let nblk = rows * n / info.block;
    let mut out = vec![0u8; nblk * info.size];
    for b in out.iter_mut() {
        *b = rng.next() as u8;
    }
    for blk in out.chunks_exact_mut(info.size) {
        for &o in half_offsets(info.name) {
            blk[o..o + 2].copy_from_slice(&rng.half().to_le_bytes());
        }
        match info.name {
            "q8_K" => blk[0..4].copy_from_slice(&(0.01 + 0.05 * rng.unit().abs()).to_le_bytes()),
            "mxfp4" => blk[0] = 118 + (rng.next() % 8) as u8,
            "iq1_m" => {
                let h = rng.half();
                for k in 0..4 {
                    let at = 48 + 2 * k;
                    let w = u16::from_le_bytes([blk[at], blk[at + 1]]) & 0x0fff;
                    let w = w | (((h >> (4 * k)) & 0xf) << 12);
                    blk[at..at + 2].copy_from_slice(&w.to_le_bytes());
                }
            }
            _ => {}
        }
    }
    out
}

/// Largest absolute difference over the largest reference magnitude
pub fn rel_err(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len(), "lengths differ");
    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-20);
    let diff = got.iter().zip(want).fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(diff.is_finite(), "non finite difference");
    diff / scale
}

pub fn assert_close(what: &str, got: &[f32], want: &[f32], tol: f32) {
    let e = rel_err(got, want);
    assert!(e <= tol, "{what}: relative error {e} above {tol}");
}

pub fn cache_dir() -> PathBuf {
    Path::new(env!("CARGO_TARGET_TMPDIR")).join("kf-cuda-cache")
}

pub fn cuda() -> CudaBackend {
    CudaBackend::new(0, Some(&cache_dir())).expect("cuda backend")
}

pub fn gguf(file: &str) -> kf_gguf::Gguf {
    kf_gguf::Gguf::open(Path::new(MODELS).join(file)).expect("open gguf")
}

pub fn inputs(tokens: &[u32], start: u32) -> Inputs {
    Inputs { tokens: tokens.to_vec(), positions: (start..start + tokens.len() as u32).collect(), ..Default::default() }
}

/// Runs a graph on both backends and checks agreement
pub fn both(what: &str, g: &Graph, inp: &Inputs, w: &dyn WeightSource, cpu: &mut CpuBackend, gpu: &mut CudaBackend, tol: f32) -> Vec<f32> {
    let want = cpu.run(g, inp, w).unwrap_or_else(|e| panic!("{what} cpu: {e:#}"));
    let got = gpu.run(g, inp, w).unwrap_or_else(|e| panic!("{what} cuda: {e:#}"));
    assert_close(what, &got, &want, tol);
    got
}

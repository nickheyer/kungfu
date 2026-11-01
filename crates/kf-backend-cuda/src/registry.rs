use crate::sources;
use anyhow::{Context, Result, anyhow};
use cudarc::driver::{CudaContext, CudaFunction, CudaModule};
use cudarc::nvrtc::{CompileError, CompileOptions, Ptx, compile_ptx_with_opts};
use kf_core::{Act, Gating, Op, Selection, SwaType, WeightNorm};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

mod embedded {
    include!(concat!(env!("OUT_DIR"), "/embedded_ptx.rs"));
}

/// Kernels an op launches, fixed module entries plus entries from each weight's quant module
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KernelRef {
    pub fixed: &'static [(&'static str, &'static str)],
    pub quant: &'static [&'static str],
}

/// Compiled kernels by module name, embedded sources plus any added through load
pub struct KernelRegistry {
    ctx: Arc<CudaContext>,
    arch: u32,
    sms: usize,
    cache_dir: Option<PathBuf>,
    modules: HashMap<String, Arc<CudaModule>>,
    functions: HashMap<(String, String), CudaFunction>,
    extra: HashMap<String, String>,
    /// Module and entry of every loaded function by its driver handle, for traces
    names: HashMap<usize, String>,
    /// Dynamic shared memory limit already raised per function
    raised: HashMap<(String, String), u32>,
}

impl KernelRegistry {
    pub fn new(ctx: Arc<CudaContext>, cache_dir: Option<&Path>) -> Result<Self> {
        let (major, minor) = ctx.compute_capability().context("compute capability")?;
        let sms = ctx.attribute(cudarc::driver::sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT).context("multiprocessor count")? as usize;
        Ok(Self {
            ctx,
            arch: (major * 10 + minor) as u32,
            sms,
            cache_dir: cache_dir.map(Path::to_path_buf),
            modules: HashMap::new(),
            functions: HashMap::new(),
            extra: HashMap::new(),
            names: HashMap::new(),
            raised: HashMap::new(),
        })
    }

    /// Compute capability as major times ten plus minor
    pub fn arch(&self) -> u32 {
        self.arch
    }

    /// Streaming multiprocessors of the device
    pub fn sm_count(&self) -> usize {
        self.sms
    }

    /// PTX of a source for this device, through the disk cache
    pub fn ptx(&self, name: &str, src: &str) -> Result<Ptx> {
        self.compile(name, src)
    }

    /// Compiles CUDA C to PTX for this device and loads it under a module name
    pub fn load(&mut self, name: &str, src: &str) -> Result<()> {
        let ptx = self.compile(name, src)?;
        let module = self.ctx.load_module(ptx).with_context(|| format!("load module {name}"))?;
        self.extra.insert(name.to_string(), src.to_string());
        self.modules.insert(name.to_string(), module);
        self.functions.retain(|(m, _), _| m != name);
        Ok(())
    }

    /// Loads a module from embedded PTX for the highest arch not above the device, else through NVRTC
    pub fn ensure(&mut self, name: &str) -> Result<()> {
        if self.modules.contains_key(name) {
            return Ok(());
        }
        let ptx = match embedded_ptx(name, self.arch) {
            Some(text) => Ptx::from_src(text),
            None => {
                let src = sources::module_source(name).ok_or_else(|| anyhow!("no kernel module {name}"))?;
                self.compile(name, src)?
            }
        };
        let module = self.ctx.load_module(ptx).with_context(|| format!("load module {name}"))?;
        self.modules.insert(name.to_string(), module);
        Ok(())
    }

    pub fn function(&mut self, module: &str, name: &str) -> Result<CudaFunction> {
        if let Some(f) = self.functions.get(&(module.to_string(), name.to_string())) {
            return Ok(f.clone());
        }
        self.ensure(module)?;
        let m = self.modules.get(module).with_context(|| format!("no kernel module {module}"))?;
        let f = m.load_function(name).with_context(|| format!("no kernel {name} in {module}"))?;
        self.names.insert(f.cu_function() as usize, format!("{module}.{name}"));
        self.functions.insert((module.to_string(), name.to_string()), f.clone());
        Ok(f)
    }

    /// A function whose dynamic shared memory limit allows smem bytes, raised past the 48 KB default once per function
    pub fn function_smem(&mut self, module: &str, name: &str, smem: u32) -> Result<CudaFunction> {
        let f = self.function(module, name)?;
        if smem > 48 * 1024 {
            let key = (module.to_string(), name.to_string());
            if self.raised.get(&key).is_none_or(|have| *have < smem) {
                f.set_attribute(cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem as i32).with_context(|| format!("{module}.{name} shared memory limit of {smem}"))?;
                self.raised.insert(key, smem);
            }
        }
        Ok(f)
    }

    /// Module and entry name of a loaded function
    pub fn kernel_name(&self, f: &CudaFunction) -> &str {
        self.names.get(&(f.cu_function() as usize)).map_or("?", String::as_str)
    }

    /// Kernel of a quant family entry for a ggml type
    pub fn quant_function(&mut self, ggml_type: u32, entry: &str) -> Result<CudaFunction> {
        let module = sources::quant_module(ggml_type).ok_or_else(|| anyhow!("no quant kernels for ggml type {ggml_type}"))?;
        self.function(&module, entry)
    }

    /// Whether a module is loaded, added, or embedded as source
    pub fn has_module(&self, name: &str) -> bool {
        self.modules.contains_key(name) || self.extra.contains_key(name) || sources::module_source(name).is_some()
    }

    /// Whether a quant kernel module exists for a ggml type
    pub fn has_quant(&self, ggml_type: u32) -> bool {
        sources::quant_module(ggml_type).is_some()
    }

    pub fn modules(&self) -> Vec<&str> {
        self.modules.keys().map(String::as_str).collect()
    }

    /// Whether a kernel exists for the op on this device
    pub fn supports(&self, op: &Op) -> bool {
        cuda_kernel_for_op(self, op).is_some()
    }

    /// PTX through the disk cache keyed by sha256 of the expanded source and the compute capability
    fn compile(&self, name: &str, src: &str) -> Result<Ptx> {
        let full = sources::expand(src)?;
        let mut h = Sha256::new();
        h.update(full.as_bytes());
        h.update(format!("\0sm_{}", self.arch).as_bytes());
        if std::env::var_os("KF_LINEINFO").is_some() {
            h.update(b"\0lineinfo");
        }
        let mut key = String::with_capacity(64);
        for b in h.finalize() {
            write!(key, "{b:02x}")?;
        }
        let path = self.cache_dir.as_ref().map(|d| d.join("ptx").join(format!("{name}-sm{}-{key}.ptx", self.arch)));
        if let Some(p) = &path
            && let Ok(text) = std::fs::read_to_string(p)
        {
            return Ok(Ptx::from_src(text));
        }
        // KF_LINEINFO adds source line tables to the PTX for the CUDA sanitizers and profilers
        let mut options = vec![format!("--gpu-architecture=compute_{}", self.arch), "-std=c++17".to_string()];
        if std::env::var_os("KF_LINEINFO").is_some() {
            options.push("-lineinfo".to_string());
        }
        let opts = CompileOptions {
            options,
            name: Some(format!("{name}.cu")),
            ..Default::default()
        };
        let ptx = compile_ptx_with_opts(&full, opts).map_err(|e| match e {
            CompileError::CompileError { log, .. } => anyhow!("nvrtc compile of {name}\n{}", log.to_string_lossy()),
            other => anyhow!("nvrtc compile of {name}: {other}"),
        })?;
        if let Some(p) = &path {
            let dir = p.parent().context("ptx cache path has no parent")?;
            std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
            static WRITES: AtomicU64 = AtomicU64::new(0);
            let tmp = p.with_extension(format!("tmp{}-{}", std::process::id(), WRITES.fetch_add(1, Ordering::Relaxed)));
            std::fs::write(&tmp, ptx.to_src()).with_context(|| format!("write {}", tmp.display()))?;
            std::fs::rename(&tmp, p).with_context(|| format!("rename {}", p.display()))?;
        }
        Ok(ptx)
    }
}

/// Embedded PTX of a module for the highest compiled arch not above the device
fn embedded_ptx(name: &str, arch: u32) -> Option<&'static str> {
    embedded::EMBEDDED_PTX.iter().filter(|(m, a, _)| *m == name && *a <= arch).max_by_key(|(_, a, _)| *a).map(|(_, _, p)| *p)
}

/// Whether an expert op uses only the routing and expert forms the device kernels implement
fn moe_on_device(op: &Op) -> bool {
    let Op::MoeFfn { gate_b, up_b, down_b, exps_scale, down_exps, selection, gating, norm_weights, limit, act, weight_before_ffn, index_div, post_norm, router_in, selected_in, .. } = op else { return false };
    gate_b.is_none()
        && up_b.is_none()
        && down_b.is_none()
        && exps_scale.is_none()
        && down_exps.is_some()
        && *selection == Selection::Probs
        && matches!(gating, Gating::Softmax | Gating::Sigmoid)
        && matches!(norm_weights, WeightNorm::None | WeightNorm::Clamp)
        && limit.is_none()
        && matches!(act, Act::Silu | Act::Gelu | Act::Relu)
        && !weight_before_ffn
        && *index_div <= 1
        && post_norm.is_none()
        && !router_in
        && !selected_in
}

fn act_on_device(act: &Act) -> bool {
    matches!(act, Act::Silu | Act::Gelu | Act::Relu | Act::GeluErf | Act::ReluSqr)
}

/// The kernels that run an op on the device, none sends the node to the host
#[allow(unreachable_patterns)]
pub fn cuda_kernel_for_op(registry: &KernelRegistry, op: &Op) -> Option<KernelRef> {
    const NONE: &[&str] = &[];
    let fixed = |fixed: &'static [(&'static str, &'static str)]| Some(KernelRef { fixed, quant: NONE });
    let found = match op {
        Op::Input { .. } => fixed(&[("elementwise", "input_f32")]),
        Op::Embed { .. } => Some(KernelRef { fixed: &[], quant: &["embed"] }),
        Op::MatMul { scale: None, .. } => Some(KernelRef { fixed: &[("elementwise", "round_act"), ("fused", "quant_act")], quant: &["matvec", "matmul", "matvec_i8", "dequant"] }),
        Op::MatMul { scale: Some(_), .. } => Some(KernelRef { fixed: &[("elementwise", "round_act"), ("fused", "quant_act"), ("elementwise", "scale")], quant: &["matvec", "matmul", "matvec_i8", "dequant"] }),
        Op::RmsNorm { .. } => Some(KernelRef { fixed: &[("norm", "rms_norm"), ("norm", "rms_norm_add"), ("fused", "rms_norm_q8")], quant: &["dequant"] }),
        Op::LayerNorm { .. } => Some(KernelRef { fixed: &[("norm", "layer_norm")], quant: &["dequant"] }),
        Op::L2Norm { .. } => fixed(&[("norm", "l2_norm")]),
        Op::Rope { .. } => fixed(&[("rope", "rope"), ("fused", "norm_rope_hd128")]),
        Op::AttentionFull { .. } => fixed(&[("attention", "attention_full")]),
        Op::GeluQuick => fixed(&[("elementwise", "unary")]),
        Op::PoolGrid { .. } => fixed(&[("elementwise", "pool_grid")]),
        Op::Attention { alibi_bias, kv_source: None, top_k: None, window, .. } if *alibi_bias == 0.0 && window.is_none_or(|w| w.kind == SwaType::Standard) => {
            Some(KernelRef { fixed: &[("attention", "kv_append"), ("attention", "attention"), ("flash", "flash_prefill_hd128")], quant: &["dequant"] })
        }
        Op::MoeFfn { .. } if moe_on_device(op) => Some(KernelRef {
            fixed: &[("moe", "moe_route"), ("moe", "moe_group"), ("moe", "moe_glu"), ("moe", "moe_combine"), ("elementwise", "round_act"), ("fused", "quant_act")],
            quant: &["matvec", "matmul", "moe_matvec", "moe_matvec_i8", "moe_matmul", "dequant"],
        }),
        Op::GatedAct { act, limit: None } if act_on_device(act) => fixed(&[("elementwise", "gated_act"), ("fused", "gated_act_q8")]),
        Op::GatedActFused { act, limit: None } if act_on_device(act) => fixed(&[("elementwise", "gated_act_fused"), ("fused", "gated_act_q8")]),
        Op::Add | Op::Sub | Op::Mul | Op::Div => fixed(&[("elementwise", "binary")]),
        Op::MulWeight { .. } | Op::AddWeight { .. } | Op::DivWeight { .. } => Some(KernelRef { fixed: &[("elementwise", "row_op")], quant: &["dequant"] }),
        Op::Scale(_) => fixed(&[("elementwise", "scale")]),
        Op::Silu => fixed(&[("elementwise", "silu")]),
        Op::Gelu => fixed(&[("elementwise", "gelu")]),
        Op::Relu => fixed(&[("elementwise", "relu")]),
        Op::Sigmoid => fixed(&[("elementwise", "sigmoid")]),
        Op::GeluErf
        | Op::ReluSqr
        | Op::Tanh
        | Op::Exp
        | Op::Sqrt
        | Op::Sqr
        | Op::Abs
        | Op::Sign
        | Op::Log
        | Op::Round
        | Op::Softplus { .. }
        | Op::Clamp { .. }
        | Op::ScaleBias { .. } => fixed(&[("elementwise", "unary")]),
        Op::Softmax => fixed(&[("norm", "softmax")]),
        Op::TopK { .. } => fixed(&[("sample", "topk")]),
        Op::SumRows => fixed(&[("norm", "sum_rows")]),
        Op::Softcap(_) => fixed(&[("elementwise", "softcap")]),
        Op::Reshape { .. } | Op::LastRows { .. } | Op::Rows { .. } => fixed(&[]),
        Op::Cols { .. } => fixed(&[("elementwise", "cols")]),
        Op::Concat => fixed(&[("elementwise", "concat")]),
        Op::Tile { .. } => fixed(&[("elementwise", "tile")]),
        Op::PadCols { .. } => fixed(&[("elementwise", "pad_cols")]),
        Op::MaskCols { .. } => fixed(&[("elementwise", "mask_cols")]),
        Op::Fill { .. } => fixed(&[("elementwise", "unary")]),
        _ => None,
    }?;
    found.fixed.iter().all(|(m, _)| registry.has_module(m)).then_some(found)
}

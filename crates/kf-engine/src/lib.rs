//! Loads a model, plans its placement, and runs sessions against a backend
pub mod expert_cache;
pub mod launch;
pub mod loader;
pub mod plan;
pub mod rdma;
pub mod sampler;
pub mod session;
pub mod speculative;
pub mod stage;
pub mod wire;

pub use launch::{CtxChoice, Prepared, RoutingProfile, prepare};
pub use loader::{LoadError, Loaded, Store, load};
pub use plan::{CTX_MIN, CTX_STEP, LayerBytes, PlanDoc, PlacementDoc, PoolDoc, ResidentDoc, Scope, ctx_auto, host_memory, is_expert, layer_bytes, outside_bytes, plan, plan_device_placement, plan_with, pool_memory, state_bytes, state_need_bytes};
pub use sampler::{Rng, Sampling, sample};
pub use session::{Rows, Session};
pub use stage::{Link, StageConn, StageRunner, admitted, serve_stage};
pub use speculative::{Chooser, Drafter, verify};

use anyhow::{Result, bail};
use kf_core::Backend;
use std::path::Path;

/// A device a backend can run on, as the self description lists it
#[derive(Clone, Debug)]
pub struct DeviceInfo {
    pub ordinal: usize,
    pub name: String,
    pub total_bytes: u64,
    pub free_bytes: u64,
}

/// The device backend of this build, every cuda name in the tree passes through here
#[cfg(feature = "cuda")]
pub mod device {
    use super::DeviceInfo;
    use anyhow::Result;
    use kf_core::Backend;
    use std::path::Path;

    pub const NAME: Option<&str> = Some("cuda");

    /// Every device the driver reports, none when no driver is present
    pub fn devices() -> Vec<DeviceInfo> {
        kf_backend_cuda::devices().unwrap_or_default().into_iter().map(|d| DeviceInfo { ordinal: d.ordinal, name: d.name, total_bytes: d.total_bytes as u64, free_bytes: d.free_bytes as u64 }).collect()
    }

    pub fn has_quant(ggml_type: u32) -> bool {
        kf_backend_cuda::has_quant_module(ggml_type)
    }

    pub fn open(ordinal: usize, cache_dir: Option<&Path>) -> Result<Box<dyn Backend>> {
        Ok(Box::new(kf_backend_cuda::CudaBackend::new(ordinal, cache_dir)?))
    }
}

/// The device backend of a build made without one, no devices and no kernels
#[cfg(not(feature = "cuda"))]
pub mod device {
    use super::DeviceInfo;
    use anyhow::{Result, bail};
    use kf_core::Backend;
    use std::path::Path;

    pub const NAME: Option<&str> = None;

    pub fn devices() -> Vec<DeviceInfo> {
        Vec::new()
    }

    pub fn has_quant(_ggml_type: u32) -> bool {
        false
    }

    pub fn open(ordinal: usize, _cache_dir: Option<&Path>) -> Result<Box<dyn Backend>> {
        bail!("this build has no cuda backend for device {ordinal}, it serves on the host")
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendChoice {
    Cpu,
    Cuda { device: usize },
}

impl BackendChoice {
    pub fn parse(name: &str, device: usize) -> Result<Self> {
        Ok(match name {
            "cpu" => Self::Cpu,
            "cuda" => Self::Cuda { device },
            "auto" => Self::auto(device)?,
            other => bail!("unknown backend {other}"),
        })
    }

    /// The device when one is visible, else the host
    pub fn auto(device: usize) -> Result<Self> {
        let visible = device::devices().iter().any(|d| d.ordinal == device);
        Ok(if visible { Self::Cuda { device } } else { Self::Cpu })
    }
}

/// A backend for the choice, the PTX cache of a device backend under the cache directory
pub fn backend(choice: BackendChoice, cache_dir: Option<&Path>) -> Result<Box<dyn Backend>> {
    Ok(match choice {
        BackendChoice::Cpu => Box::new(kf_backend_cpu::CpuBackend::new()),
        BackendChoice::Cuda { device: ordinal } => device::open(ordinal, cache_dir)?,
    })
}

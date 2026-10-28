use anyhow::Result;
use kf_arch::{Architecture, Hparams, Weights};
use std::path::PathBuf;
use kf_core::{WeightSource, Wid};
use kf_gguf::Gguf;
use kf_layers::Model;
use std::path::Path;
use std::sync::Arc;

/// The file as a weight source, ids are positions in its tensor table
pub struct Store {
    pub gguf: Gguf,
}

impl WeightSource for Store {
    fn bytes(&self, w: Wid) -> Result<&[u8]> {
        self.gguf.bytes(&self.gguf.tensors[w.0 as usize])
    }
    fn ggml_type(&self, w: Wid) -> u32 {
        self.gguf.tensors[w.0 as usize].ggml_type
    }
    fn dims(&self, w: Wid) -> &[usize] {
        &self.gguf.tensors[w.0 as usize].dims
    }
    fn name(&self, w: Wid) -> &str {
        &self.gguf.tensors[w.0 as usize].name
    }
    fn len(&self) -> usize {
        self.gguf.tensors.len()
    }
}

pub struct Loaded {
    pub store: Arc<Store>,
    pub arch: &'static dyn Architecture,
    pub hparams: Hparams,
    pub model: Arc<Model>,
}

impl Loaded {
    /// Names of the tensors in the file no layer claims
    pub fn unclaimed(&self) -> Vec<String> {
        let claimed: std::collections::HashSet<Wid> = self.model.weights().into_iter().collect();
        self.store.gguf.tensors.iter().enumerate().filter(|(i, _)| !claimed.contains(&Wid(*i as u32))).map(|(_, t)| t.name.clone()).collect()
    }
}

/// Why a file did not load, each case an error code the self description lists
#[derive(Debug)]
pub enum LoadError {
    Open { path: PathBuf, detail: String },
    UnknownArch { arch: String },
    Compose { arch: String, detail: String },
}

impl LoadError {
    pub fn code(&self) -> &'static str {
        match self {
            Self::Open { .. } => "model_open",
            Self::UnknownArch { .. } => "unknown_arch",
            Self::Compose { .. } => "compose",
        }
    }

    /// The keys the error event carries for this code
    pub fn keys(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::Open { path, detail } => vec![("path", path.display().to_string()), ("detail", detail.clone())],
            Self::UnknownArch { arch } => vec![("arch", arch.clone())],
            Self::Compose { arch, detail } => vec![("arch", arch.clone()), ("detail", detail.clone())],
        }
    }
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Open { path, detail } => write!(f, "{}: {detail}", path.display()),
            Self::UnknownArch { arch } => write!(f, "no architecture module for {arch}"),
            Self::Compose { arch, detail } => write!(f, "{arch}: {detail}"),
        }
    }
}

impl std::error::Error for LoadError {}

/// Opens a GGUF, finds the module that claims its architecture, and composes the model
pub fn load(path: impl AsRef<Path>) -> Result<Loaded, LoadError> {
    let path = path.as_ref();
    let open = |detail: String| LoadError::Open { path: path.to_path_buf(), detail };
    let gguf = Gguf::open(path).map_err(|e| open(format!("{e:#}")))?;
    for t in &gguf.tensors {
        gguf.bytes(t).map_err(|e| open(format!("{e:#}")))?;
    }
    let id = gguf.meta.arch().ok_or_else(|| open("file has no general.architecture".to_string()))?.to_string();
    let arch = kf_arch::find(&id).ok_or_else(|| LoadError::UnknownArch { arch: id.clone() })?;
    let compose = |e: anyhow::Error| LoadError::Compose { arch: id.clone(), detail: format!("{e:#}") };
    let hparams = arch.hparams(&gguf.meta).map_err(compose)?;
    let model = {
        let mut w = Weights::new(&gguf);
        arch.build(&hparams, &mut w).map_err(compose)?
    };
    Ok(Loaded { store: Arc::new(Store { gguf }), arch, hparams, model: Arc::new(model) })
}

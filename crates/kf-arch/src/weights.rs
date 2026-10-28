use anyhow::{Result, anyhow};
use kf_core::Wid;
use kf_gguf::{Gguf, TensorInfo};

/// The tensor table of a file, handing out ids and remembering what was claimed
pub struct Weights<'a> {
    gguf: &'a Gguf,
    used: Vec<bool>,
}

impl<'a> Weights<'a> {
    pub fn new(gguf: &'a Gguf) -> Self {
        Self { gguf, used: vec![false; gguf.tensors.len()] }
    }

    pub fn has(&self, name: &str) -> bool {
        self.gguf.tensor(name).is_some()
    }

    pub fn take_opt(&mut self, name: &str) -> Option<Wid> {
        let idx = self.gguf.tensors.iter().position(|t| t.name == name)?;
        self.used[idx] = true;
        Some(Wid(idx as u32))
    }

    pub fn take(&mut self, name: &str) -> Result<Wid> {
        self.take_opt(name).ok_or_else(|| anyhow!("missing tensor {name}"))
    }

    pub fn info(&self, w: Wid) -> &TensorInfo {
        &self.gguf.tensors[w.0 as usize]
    }

    pub fn dims(&self, w: Wid) -> &[usize] {
        &self.info(w).dims
    }

    /// Marks every tensor under a name prefix used, for layers a graph never reads
    pub fn claim_prefix(&mut self, prefix: &str) {
        for (i, t) in self.gguf.tensors.iter().enumerate() {
            if t.name.starts_with(prefix) {
                self.used[i] = true;
            }
        }
    }

    /// Tensors no layer claimed, a sign the composition missed something
    pub fn unused(&self) -> Vec<&str> {
        self.gguf.tensors.iter().zip(&self.used).filter(|(_, u)| !**u).map(|(t, _)| t.name.as_str()).collect()
    }
}

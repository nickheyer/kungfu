//! One module per architecture, every module answers the same three questions
pub mod archs;
pub mod compose;
pub mod hparams;
pub mod weights;

use anyhow::Result;
use kf_gguf::Metadata;
use kf_layers::Model;

pub use hparams::{Hparams, Keys, RopeScaling};
pub use weights::Weights;

/// The touch points of an architecture, id in the file, hyperparameters, and the layer composition
pub trait Architecture: Send + Sync {
    /// The general.architecture value this module claims
    fn id(&self) -> &'static str;
    fn hparams(&self, meta: &Metadata) -> Result<Hparams>;
    fn build(&self, hp: &Hparams, w: &mut Weights) -> Result<Model>;
}

pub fn registry() -> &'static [&'static dyn Architecture] {
    archs::all()
}

pub fn find(id: &str) -> Option<&'static dyn Architecture> {
    registry().iter().copied().find(|a| a.id() == id)
}

//! The layer API, models compose these and never touch ops, memory, or devices
pub mod attention;
pub mod basic;
pub mod compress;
pub mod ffn;
pub mod gemma;
pub mod hyper;
pub mod kimi;
pub mod lightning;
pub mod mamba;
pub mod mla;
pub mod model;
pub mod nextn;
pub mod ple;
pub mod recurrent;
pub mod rwkv;
pub mod shortconv;
pub mod switch;
pub mod util;

pub use attention::{AttnGate, AttnTemp, Attention, GateKind, QkvProj};
pub use basic::{Branch, EmbedSkip, Embedding, HeadLayout, HeadNorm, Identity, LayerNorm, Linear, MulRms, NormKind, OnCols, Parallel, Residual, RmsNorm, ScaledResidual, SplitResidual, StreamScale};
pub use compress::{CandidateBlocks, CompressedAttention};
pub use ffn::{ActFfn, Activation, ChunkExperts, Ffn, Latent, Moe, RouterInput, SharedExperts};
pub use gemma::{AltUpEntry, AltUpExit, Gemma3nBlock, PerLayerEmbedding};
pub use hyper::{HcHead, HcKind, HcMean, HcMix, HyperConnected};
pub use kimi::{ResMixBlock, ResMixHead};
pub use lightning::LightningAttn;
pub use mamba::{Mamba1, Mamba2, Plamo2Mamba};
pub use mla::{DsaIndexer, IndexerNorm, Mla, MlaQ};
pub use model::{Block, Model, Segment};
pub use nextn::NextN;
pub use ple::{Engram, Ple};
pub use recurrent::{BetaAlpha, DeltaLayout, DeltaNet, HeadMap, Kda, OutGate};
pub use rwkv::{Rwkv6ChannelMix, Rwkv6TimeMix, Rwkv7ChannelMix, Rwkv7TimeMix};
pub use shortconv::ShortConv;
pub use switch::{SwitchedBlock, SwitchedLinear};

pub use kf_core::{Act, Ctx, Gating, GluLimit, GraphBuilder, Groups, IndexerKind, Layer, Op, RopeKind, RopeParams, Selection, StateNeed, SwaType, Tid, WeightNorm, Wid, Window};

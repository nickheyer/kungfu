use crate::graph::Wid;
use serde::{Deserialize, Serialize};

/// Tokens of recurrent, shift, conv and compressed state the host keeps beside the live copy for roll back
pub const STATE_SNAPSHOTS: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Pool {
    Host,
    Device(u32),
}

impl Pool {
    /// The pool name the log events and plan JSON use, host or device:N
    pub fn id(&self) -> String {
        match self {
            Pool::Host => "host".to_string(),
            Pool::Device(n) => format!("device:{n}"),
        }
    }

    pub fn parse(s: &str) -> Option<Pool> {
        if s == "host" {
            return Some(Pool::Host);
        }
        s.strip_prefix("device:")?.parse().ok().map(Pool::Device)
    }
}

/// Placement for one layer instance, set by the planner and read by the layer machinery
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LayerPlan {
    pub layer: usize,
    pub weights: Pool,
    pub state: Pool,
    pub expert_cache_slots: usize,
    /// Rows of each slot's full attention cache resident on the device, the rest streamed from the host, zero for every row
    #[serde(default)]
    pub kv_resident: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Plan {
    pub layers: Vec<LayerPlan>,
    pub pinned: Vec<Wid>,
    /// Weight bytes placed on the device, experts in cache slots counted by their slots
    pub weights_bytes: u64,
    /// State bytes placed on the device at the context length
    pub state_bytes: u64,
    /// Context length every cache is sized for
    pub n_ctx: usize,
    /// Most rows one pass feeds, the prefill chunk and the ring slack of windowed caches
    pub n_batch: usize,
    /// Sequence slots, each with its own state at the context length
    pub n_seq: usize,
}

/// Rows a cache keeps at a context length and batch, the whole context or the window plus one batch of slack
pub fn cache_rows(n_ctx: usize, n_batch: usize, window: Option<usize>) -> usize {
    match window {
        Some(w) => (w + n_batch).min(n_ctx),
        None => n_ctx,
    }
}

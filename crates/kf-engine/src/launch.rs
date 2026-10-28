//! Brings a loaded model to a backend under a plan, the steps every command shares
use crate::plan::{PlanDoc, POOL_FILL, Scope, ctx_auto, layer_bytes, outside_bytes, plan_with, pool_memory, state_bytes};
use crate::{BackendChoice, Loaded};
use anyhow::{Context, Result, bail};
use kf_core::{AllocKind, Backend, Plan, Pool};
use std::path::Path;

/// Context length a launch asks for, a number or the planner's choice
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CtxChoice {
    Auto,
    Tokens(usize),
}

/// A backend with its plan applied and the document the plan event printed
pub struct Prepared {
    pub backend: Box<dyn Backend>,
    pub plan: Plan,
    pub doc: PlanDoc,
}

/// Counts per expert by layer, written at exit and read back to warm the expert caches
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct RoutingProfile {
    pub layers: std::collections::BTreeMap<usize, Vec<u64>>,
}

impl RoutingProfile {
    pub fn read(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| format!("read routing profile {}", path.display()))?;
        serde_json::from_str(&text).with_context(|| format!("parse routing profile {}", path.display()))
    }

    /// The counts a backend gathered, none when it cached no experts
    pub fn of(backend: &dyn Backend) -> Option<Self> {
        let layers: std::collections::BTreeMap<usize, Vec<u64>> = backend.routing_profile().into_iter().collect();
        (!layers.is_empty()).then_some(Self { layers })
    }

    pub fn write(&self, path: &Path) -> Result<()> {
        std::fs::write(path, serde_json::to_string(self)?).with_context(|| format!("write routing profile {}", path.display()))
    }
}

/// Plans against the pool or a plan document, logs the plan event, opens the backend, applies the plan, and hands it the routing profile
#[allow(clippy::too_many_arguments)]
pub fn prepare(loaded: &Loaded, choice: BackendChoice, ctx: CtxChoice, n_batch: usize, n_seq: usize, plan_path: Option<&Path>, cache_dir: Option<&Path>, profile: Option<&Path>, scope: &Scope) -> Result<Prepared> {
    if n_batch == 0 {
        bail!("a batch of zero tokens");
    }
    if n_seq == 0 {
        bail!("no sequence slots");
    }
    let profile = profile.map(RoutingProfile::read).transpose()?;
    let (free, total) = pool_memory(choice)?;
    let plan = match plan_path {
        Some(path) => {
            let text = std::fs::read_to_string(path).with_context(|| format!("read plan {}", path.display()))?;
            let mut doc: PlanDoc = serde_json::from_str(&text).with_context(|| format!("parse plan {}", path.display()))?;
            doc.n_batch = n_batch;
            doc.n_seq = n_seq;
            match ctx {
                CtxChoice::Tokens(n) => doc.n_ctx = n,
                CtxChoice::Auto => doc.n_ctx = ctx_auto_doc(loaded, &doc, choice, free, scope)?,
            }
            doc.to_plan(loaded, choice, scope)?
        }
        None => match ctx {
            CtxChoice::Tokens(n) => plan_with(loaded, choice, n, n_batch, n_seq, free, scope)?,
            CtxChoice::Auto => ctx_auto(loaded, choice, n_batch, n_seq, free, scope)?.1,
        },
    };
    let doc = PlanDoc::of(&plan, loaded, choice, free, total, scope);
    kf_core::event::emit("plan", &[("plan", serde_json::to_string(&doc)?)]);
    let mut backend = crate::backend(choice, cache_dir)?;
    backend.apply_plan(&plan)?;
    if let Some(p) = profile {
        backend.set_routing_profile(p.layers.into_iter().collect());
    }
    account_host_weights(loaded, &plan, choice, scope);
    Ok(Prepared { backend, plan, doc })
}

/// Logs the mapped file bytes of every layer the plan leaves on the host, dense as weights and expert tensors as experts
fn account_host_weights(loaded: &Loaded, plan: &Plan, choice: BackendChoice, scope: &Scope) {
    let (layers, _) = layer_bytes(loaded, plan.n_ctx, plan.n_batch, plan.n_seq);
    let mut weights = 0u64;
    let mut experts = 0u64;
    if choice == BackendChoice::Cpu {
        let (embed, rest) = outside_bytes(loaded);
        weights += if scope.head { embed + rest } else { embed };
    }
    for lp in plan.layers.iter().filter(|l| l.weights == Pool::Host) {
        let lb = layers.get(&lp.layer).cloned().unwrap_or_default();
        weights += lb.dense;
        experts += lb.experts;
    }
    if weights > 0 {
        kf_core::event::alloc(Pool::Host, AllocKind::Weights, weights);
    }
    if experts > 0 {
        kf_core::event::alloc(Pool::Host, AllocKind::Experts, experts);
    }
}

/// The largest context in steps of 256 up to the trained length whose state fits beside the document's weights in the main pool
fn ctx_auto_doc(loaded: &Loaded, doc: &PlanDoc, choice: BackendChoice, free: u64, scope: &Scope) -> Result<usize> {
    let main = match choice {
        BackendChoice::Cpu => Pool::Host,
        BackendChoice::Cuda { device } => Pool::Device(device as u32),
    };
    let pool = doc.pools.iter().find(|p| p.id == main.id());
    let capacity = pool.map(|p| p.capacity).filter(|c| *c > 0).unwrap_or((free as f64 * POOL_FILL) as u64);
    let weights = pool.map_or(0, |p| p.weights + p.experts);
    let trained = loaded.hparams.n_ctx_train.max(crate::CTX_MIN);
    let fits = |n: usize| {
        let mut d = doc.clone();
        d.n_ctx = n;
        let Ok(plan) = d.to_plan(loaded, choice, scope) else { return false };
        let state = if main == Pool::Host { state_bytes(loaded, n, doc.n_batch, doc.n_seq, scope) } else { plan.state_bytes };
        weights + state <= capacity
    };
    let mut n = trained.div_ceil(crate::CTX_STEP) * crate::CTX_STEP;
    while n >= crate::CTX_MIN {
        if fits(n) {
            return Ok(n.min(trained));
        }
        n -= crate::CTX_STEP;
    }
    bail!("no context of {} tokens fits the plan's {capacity} bytes beside {weights} bytes of weights", crate::CTX_MIN)
}

use crate::{BackendChoice, Loaded};
use anyhow::{Result, anyhow, bail};
use kf_core::{Layer, LayerPlan, Plan, Pool, STATE_SNAPSHOTS, StateNeed, WeightSource, Wid, cache_rows};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Smallest context the planner considers and the step it moves in
pub const CTX_MIN: usize = 256;
pub const CTX_STEP: usize = 256;

/// Fraction of a pool's free memory the plan may fill, the rest stays for activations and scratch
pub const POOL_FILL: f64 = 0.9;

/// The layers one process holds in order and whether it holds the head, the whole model for a solo run
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Scope {
    pub layers: Vec<usize>,
    pub head: bool,
}

impl Scope {
    pub fn all(n_layers: usize) -> Self {
        Self { layers: (0..n_layers).collect(), head: true }
    }

    /// One range of layers without the head, what a stage holds
    pub fn range(from: usize, to: usize) -> Self {
        Self { layers: (from..to).collect(), head: false }
    }

    /// Every layer outside the ranges given plus the head, what a chain head holds
    pub fn outside(n_layers: usize, ranges: &[(usize, usize)]) -> Self {
        Self { layers: (0..n_layers).filter(|l| !ranges.iter().any(|(a, b)| l >= a && l < b)).collect(), head: true }
    }

    pub fn layers(&self) -> impl Iterator<Item = usize> + '_ {
        self.layers.iter().copied()
    }

    pub fn contains(&self, l: usize) -> bool {
        self.layers.contains(&l)
    }
}

/// Bytes of the token embedding and of everything else outside the blocks, the head and final norm
pub fn outside_bytes(loaded: &Loaded) -> (u64, u64) {
    let store = loaded.store.as_ref();
    let bytes = |w: Wid| kf_quant::nbytes(store.ggml_type(w), store.dims(w)).unwrap_or(0) as u64;
    let (_, outside) = layer_bytes(loaded, 1, 1, 1);
    let embed: u64 = loaded.model.embed.weights().into_iter().map(bytes).sum();
    (embed, outside.saturating_sub(embed))
}

/// Bytes one state need takes at a context length, half precision ring rows for caches and f32 with roll back snapshots for the rest
pub fn state_need_bytes(s: &StateNeed, n_ctx: usize, n_batch: usize) -> u64 {
    let snaps = (1 + STATE_SNAPSHOTS) as u64;
    let f = |n: usize| (n * 4) as u64;
    let h = |n: usize| (n * 2) as u64;
    match *s {
        StateNeed::KvCache { n_head_kv, head_dim_k, head_dim_v, window } => h(cache_rows(n_ctx, n_batch, window) * n_head_kv * (head_dim_k + head_dim_v)),
        StateNeed::Latent { rank, rope_dim, window } => h(cache_rows(n_ctx, n_batch, window) * (rank + rope_dim)),
        StateNeed::Recurrent { conv_state, ssm_state } => f(conv_state + ssm_state) * snaps,
        StateNeed::TokenShift { n, width } => f(n * width) * snaps,
        StateNeed::IndexerCache { dim } => f(n_ctx * dim),
        StateNeed::Compressed { dim, ratio } => f(n_ctx.div_ceil(ratio.max(1)) * dim) * snaps,
        StateNeed::TokenHistory { n } => f(n),
    }
}

/// Bytes of state at a context length over the layers of a scope for every sequence slot
pub fn state_bytes(loaded: &Loaded, n_ctx: usize, n_batch: usize, n_seq: usize, scope: &Scope) -> u64 {
    loaded.model.states().iter().filter(|(l, _)| scope.contains(*l)).map(|(_, s)| state_need_bytes(s, n_ctx, n_batch)).sum::<u64>() * n_seq.max(1) as u64
}

/// Weight and state bytes of one block, expert tensors counted apart with their count
#[derive(Clone, Debug, Default)]
pub struct LayerBytes {
    pub dense: u64,
    pub experts: u64,
    pub n_expert: usize,
    pub expert_wids: Vec<Wid>,
    pub state: u64,
    /// Bytes one row of every slot adds to the caches that can stream from the host, the full attention caches
    pub stream_row: u64,
    /// Bytes of those caches at the whole context, part of state
    pub stream_full: u64,
}

impl LayerBytes {
    /// State bytes with the streamable caches holding rows resident rows, the whole state at zero
    pub fn state_resident(&self, rows: usize) -> u64 {
        if rows == 0 || self.stream_row == 0 {
            return self.state;
        }
        self.state - self.stream_full + self.stream_row * rows as u64
    }
}

/// Smallest resident window the planner gives a streamed cache, rows in steps of the context step
pub fn min_resident(n_batch: usize) -> usize {
    (2 * n_batch).max(CTX_STEP).div_ceil(CTX_STEP) * CTX_STEP
}

/// Whether a weight is a stacked expert tensor
pub fn is_expert(store: &dyn WeightSource, w: Wid) -> bool {
    store.dims(w).len() >= 3 && store.dims(w)[2] > 1 && store.name(w).contains("_exps")
}

/// Per layer byte accounting from the composed model with state for every sequence slot, plus the bytes outside every block
pub fn layer_bytes(loaded: &Loaded, n_ctx: usize, n_batch: usize, n_seq: usize) -> (BTreeMap<usize, LayerBytes>, u64) {
    let store = loaded.store.as_ref();
    let bytes = |w: Wid| kf_quant::nbytes(store.ggml_type(w), store.dims(w)).unwrap_or(0) as u64;
    let mut layers: BTreeMap<usize, LayerBytes> = BTreeMap::new();
    let mut seen = std::collections::HashSet::new();
    for b in &loaded.model.blocks {
        let entry = layers.entry(b.layer).or_default();
        for p in &b.parts {
            for w in p.weights() {
                if !seen.insert(w) {
                    continue;
                }
                if is_expert(store, w) {
                    entry.experts += bytes(w);
                    entry.n_expert = entry.n_expert.max(store.dims(w)[2]);
                    entry.expert_wids.push(w);
                } else {
                    entry.dense += bytes(w);
                }
            }
        }
        for s in b.states() {
            entry.state += state_need_bytes(&s, n_ctx, n_batch) * n_seq.max(1) as u64;
            if let StateNeed::KvCache { n_head_kv, head_dim_k, head_dim_v, window: None } = s {
                entry.stream_row += (2 * n_head_kv * (head_dim_k + head_dim_v)) as u64 * n_seq.max(1) as u64;
                entry.stream_full += state_need_bytes(&s, n_ctx, n_batch) * n_seq.max(1) as u64;
            }
        }
    }
    let outside: u64 = loaded.model.weights().into_iter().filter(|w| !seen.contains(w)).map(bytes).sum();
    (layers, outside)
}

/// Free and total bytes of the pool a backend choice fills
pub fn pool_memory(choice: BackendChoice) -> Result<(u64, u64)> {
    match choice {
        BackendChoice::Cpu => Ok(host_memory()),
        BackendChoice::Cuda { device } => {
            let dev = crate::device::devices().into_iter().find(|d| d.ordinal == device).ok_or_else(|| anyhow!("no cuda device {device}"))?;
            Ok((dev.free_bytes, dev.total_bytes))
        }
    }
}

/// Available and total host memory
pub fn host_memory() -> (u64, u64) {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    (sys.available_memory(), sys.total_memory())
}

/// Where every layer's weights and state live for a backend at a context length, planned against the pool's free memory
pub fn plan(loaded: &Loaded, choice: BackendChoice, n_ctx: usize, n_batch: usize, n_seq: usize, scope: &Scope) -> Result<Plan> {
    let (free, _) = pool_memory(choice)?;
    plan_with(loaded, choice, n_ctx, n_batch, n_seq, free, scope)
}

/// Weight bytes a scope holds outside the blocks, the embedding and the head when the scope has it
fn scope_outside(loaded: &Loaded, scope: &Scope) -> u64 {
    let (embed, rest) = outside_bytes(loaded);
    if scope.head { embed + rest } else { embed }
}

/// The plan for a backend given the free bytes of its pool
pub fn plan_with(loaded: &Loaded, choice: BackendChoice, n_ctx: usize, n_batch: usize, n_seq: usize, free_bytes: u64, scope: &Scope) -> Result<Plan> {
    if n_ctx == 0 || n_batch == 0 || n_seq == 0 {
        bail!("a plan needs a context length, a batch, and sequence slots");
    }
    let n_layers = loaded.model.n_layers();
    if let Some(l) = scope.layers().find(|l| *l >= n_layers) {
        bail!("scope holds layer {l} of a {n_layers} layer model");
    }
    match choice {
        BackendChoice::Cpu => {
            let (layers, _) = layer_bytes(loaded, n_ctx, n_batch, n_seq);
            let weights = scope_outside(loaded, scope) + scope.layers().map(|l| layers.get(&l).map_or(0, |lb| lb.dense + lb.experts)).sum::<u64>();
            let state = state_bytes(loaded, n_ctx, n_batch, n_seq, scope);
            let budget = (free_bytes as f64 * POOL_FILL) as u64;
            if weights + state > budget {
                bail!("host has {budget} bytes free for {weights} bytes of weights and {state} of state at {n_ctx} tokens");
            }
            Ok(Plan {
                layers: scope.layers().map(|layer| LayerPlan { layer, weights: Pool::Host, state: Pool::Host, expert_cache_slots: 0, kv_resident: 0 }).collect(),
                pinned: Vec::new(),
                weights_bytes: weights,
                state_bytes: state,
                n_ctx,
                n_batch,
                n_seq,
            })
        }
        BackendChoice::Cuda { device } => plan_device_placement(loaded, device, n_ctx, n_batch, n_seq, free_bytes, scope),
    }
}

/// Dense weights and state per layer on the device while the budget lasts, full caches windowed when only part fits, experts resident in order while they fit, the rest sharing cache slots
pub fn plan_device_placement(loaded: &Loaded, device: usize, n_ctx: usize, n_batch: usize, n_seq: usize, free_bytes: u64, scope: &Scope) -> Result<Plan> {
    let (layers, _) = layer_bytes(loaded, n_ctx, n_batch, n_seq);
    let outside = scope_outside(loaded, scope);
    let budget = (free_bytes as f64 * POOL_FILL) as u64;
    if outside > budget {
        bail!("device {device} has {budget} bytes free for a model whose embedding and head need {outside}");
    }
    let mut left = budget - outside;
    let mut device_weights = outside;
    let mut device_state = 0u64;
    let dev = Pool::Device(device as u32);
    let min_rows = min_resident(n_batch);
    let mut plan: Vec<LayerPlan> = Vec::with_capacity(scope.layers.len());
    let mut state_of: BTreeMap<usize, u64> = BTreeMap::new();
    for l in scope.layers() {
        let lb = layers.get(&l).cloned().unwrap_or_default();
        let need = lb.dense + lb.state;
        if need <= left {
            left -= need;
            device_weights += lb.dense;
            device_state += lb.state;
            state_of.insert(l, lb.state);
            plan.push(LayerPlan { layer: l, weights: dev, state: dev, expert_cache_slots: 0, kv_resident: 0 });
            continue;
        }
        let least = lb.dense + lb.state_resident(min_rows);
        if lb.stream_row > 0 && min_rows < n_ctx && least <= left {
            let room = left - lb.dense - (lb.state - lb.stream_full);
            let rows = ((room / lb.stream_row) as usize / CTX_STEP * CTX_STEP).clamp(min_rows, n_ctx - CTX_STEP);
            let state = lb.state_resident(rows);
            left -= lb.dense + state;
            device_weights += lb.dense;
            device_state += state;
            state_of.insert(l, state);
            plan.push(LayerPlan { layer: l, weights: dev, state: dev, expert_cache_slots: 0, kv_resident: rows });
        } else {
            plan.push(LayerPlan { layer: l, weights: Pool::Host, state: Pool::Host, expert_cache_slots: 0, kv_resident: 0 });
        }
    }
    let mut pinned = Vec::new();
    let mut cached: Vec<usize> = Vec::new();
    for lp in plan.iter_mut() {
        let lb = layers.get(&lp.layer).cloned().unwrap_or_default();
        if lb.n_expert == 0 || lp.weights != dev {
            continue;
        }
        if cached.is_empty() && lb.experts <= left {
            left -= lb.experts;
            device_weights += lb.experts;
            continue;
        }
        cached.push(lp.layer);
    }
    let host_state: u64 = plan.iter().filter(|lp| lp.kv_resident > 0).map(|lp| layers.get(&lp.layer).map_or(0, |lb| lb.state - lb.state_resident(lp.kv_resident))).sum();
    if host_state > 0 {
        let (host_free, _) = host_memory();
        let host_budget = (host_free as f64 * POOL_FILL) as u64;
        if host_state > host_budget {
            bail!("the cache rows past the resident windows need {host_state} bytes of host memory, {host_budget} are free");
        }
    }
    let mut i = 0;
    while i < cached.len() {
        let share = left / (cached.len() - i) as u64;
        let lb = layers.get(&cached[i]).cloned().unwrap_or_default();
        let per_expert = (lb.experts / lb.n_expert as u64).max(1);
        let slots = ((share / per_expert) as usize).min(lb.n_expert);
        let lp = plan.iter_mut().find(|l| l.layer == cached[i]).expect("cached layer in the plan");
        if slots == 0 {
            let state = state_of.get(&lp.layer).copied().unwrap_or(0);
            lp.weights = Pool::Host;
            lp.state = Pool::Host;
            lp.kv_resident = 0;
            device_weights -= lb.dense;
            device_state -= state;
            left += lb.dense + state;
        } else {
            left -= slots as u64 * per_expert;
            device_weights += slots as u64 * per_expert;
            lp.expert_cache_slots = slots;
            pinned.extend(lb.expert_wids.iter().copied());
        }
        i += 1;
    }
    Ok(Plan { layers: plan, pinned, weights_bytes: device_weights, state_bytes: device_state, n_ctx, n_batch, n_seq })
}

/// Whether a plan keeps every layer's dense weights and whole state in its own pool, experts cached or resident, nothing streamed
fn plan_fits(plan: &Plan, choice: BackendChoice) -> bool {
    match choice {
        BackendChoice::Cpu => true,
        BackendChoice::Cuda { device } => plan.layers.iter().all(|l| l.weights == Pool::Device(device as u32) && l.kv_resident == 0),
    }
}

/// The largest context in steps of 256 up to the trained length whose plan fits the pool, the plan with it
pub fn ctx_auto(loaded: &Loaded, choice: BackendChoice, n_batch: usize, n_seq: usize, free_bytes: u64, scope: &Scope) -> Result<(usize, Plan)> {
    let trained = loaded.hparams.n_ctx_train.max(CTX_MIN);
    let fits = |n: usize| plan_with(loaded, choice, n, n_batch, n_seq, free_bytes, scope).ok().filter(|p| plan_fits(p, choice));
    let Some(low) = fits(CTX_MIN) else { bail!("no context of {CTX_MIN} tokens fits {free_bytes} free bytes") };
    let mut lo = CTX_MIN;
    let mut best = low;
    let mut hi = trained.div_ceil(CTX_STEP) * CTX_STEP;
    if let Some(p) = fits(trained) {
        return Ok((trained, p));
    }
    while hi - lo > CTX_STEP {
        let mid = (lo + (hi - lo) / 2).div_ceil(CTX_STEP) * CTX_STEP;
        if mid <= lo || mid >= hi {
            break;
        }
        match fits(mid) {
            Some(p) => {
                lo = mid;
                best = p;
            }
            None => hi = mid,
        }
    }
    Ok((lo, best))
}

/// One pool of a plan document with what each kind takes and what the pool offered
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct PoolDoc {
    pub id: String,
    pub kind: String,
    pub weights: u64,
    pub state: u64,
    pub experts: u64,
    /// Bytes held back for activations and scratch, a ceiling the compute allocations stay under
    pub compute_reserve: u64,
    #[serde(default)]
    pub capacity: u64,
    #[serde(default)]
    pub total: u64,
    #[serde(default)]
    pub free: u64,
}

/// Bytes of one tensor group kind in one pool with the layers it covers
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PlacementDoc {
    pub kind: String,
    pub pool: String,
    pub bytes: u64,
    pub count: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SlotDoc {
    pub layer: usize,
    pub slots: usize,
}

/// Rows of a layer's full attention caches kept on the device, the rest on the host
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ResidentDoc {
    pub layer: usize,
    pub rows: usize,
}

/// The plan as the plan event prints it and the plan flag reads it
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct PlanDoc {
    pub n_ctx: usize,
    pub n_batch: usize,
    /// Sequence slots the state is sized for, one when absent
    #[serde(default = "one")]
    pub n_seq: usize,
    pub pools: Vec<PoolDoc>,
    pub placements: Vec<PlacementDoc>,
    #[serde(default)]
    pub expert_slots: Vec<SlotDoc>,
    #[serde(default)]
    pub kv_resident: Vec<ResidentDoc>,
}

fn one() -> usize {
    1
}

impl PlanDoc {
    /// The document of a plan with the bytes each kind takes per pool
    pub fn of(plan: &Plan, loaded: &Loaded, choice: BackendChoice, free: u64, total: u64, scope: &Scope) -> Self {
        let (layers, _) = layer_bytes(loaded, plan.n_ctx, plan.n_batch, plan.n_seq);
        let main = match choice {
            BackendChoice::Cpu => Pool::Host,
            BackendChoice::Cuda { device } => Pool::Device(device as u32),
        };
        let mut pools: BTreeMap<String, PoolDoc> = BTreeMap::new();
        let mut place: BTreeMap<(String, String), PlacementDoc> = BTreeMap::new();
        let mut add = |kind: &str, pool: Pool, b: u64, count: usize| {
            let p = pools.entry(pool.id()).or_insert_with(|| PoolDoc { id: pool.id(), kind: if pool == Pool::Host { "host".into() } else { "device".into() }, ..Default::default() });
            match kind {
                "state" => p.state += b,
                "experts" => p.experts += b,
                _ => p.weights += b,
            }
            let e = place.entry((kind.to_string(), pool.id())).or_insert_with(|| PlacementDoc { kind: kind.to_string(), pool: pool.id(), bytes: 0, count: 0 });
            e.bytes += b;
            e.count += count;
        };
        let (embed, rest) = outside_bytes(loaded);
        add("embedding", main, embed, 1);
        if scope.head {
            add("output", main, rest, 1);
        }
        let mut slots = Vec::new();
        let mut resident = Vec::new();
        for lp in &plan.layers {
            let lb = layers.get(&lp.layer).cloned().unwrap_or_default();
            add("layer", lp.weights, lb.dense, 1);
            if lp.kv_resident > 0 && lp.state != Pool::Host {
                let on_device = lb.state_resident(lp.kv_resident);
                add("state", lp.state, on_device, 1);
                add("state", Pool::Host, lb.state - on_device, 0);
                resident.push(ResidentDoc { layer: lp.layer, rows: lp.kv_resident });
            } else {
                add("state", lp.state, lb.state, 1);
            }
            if lb.n_expert > 0 {
                if lp.expert_cache_slots > 0 {
                    let per = lb.experts / lb.n_expert as u64;
                    add("experts", lp.weights, per * lp.expert_cache_slots as u64, 0);
                    add("experts", Pool::Host, lb.experts, 0);
                    slots.push(SlotDoc { layer: lp.layer, slots: lp.expert_cache_slots });
                } else {
                    add("experts", lp.weights, lb.experts, 1);
                }
            }
        }
        if let Some(p) = pools.get_mut(&main.id()) {
            p.free = free;
            p.total = total;
            p.capacity = (free as f64 * POOL_FILL) as u64;
            p.compute_reserve = free.saturating_sub(p.capacity);
        }
        Self { n_ctx: plan.n_ctx, n_batch: plan.n_batch, n_seq: plan.n_seq, pools: pools.into_values().collect(), placements: place.into_values().collect(), expert_slots: slots, kv_resident: resident }
    }

    /// The plan a document asks for, layers and resident experts taken in order by the counts each pool names, slots per layer as listed
    pub fn to_plan(&self, loaded: &Loaded, choice: BackendChoice, scope: &Scope) -> Result<Plan> {
        if self.n_ctx == 0 || self.n_batch == 0 || self.n_seq == 0 {
            bail!("plan without a context length, batch, or sequence slots");
        }
        let main = match choice {
            BackendChoice::Cpu => Pool::Host,
            BackendChoice::Cuda { device } => Pool::Device(device as u32),
        };
        let count = |kind: &str, pool: Pool| self.placements.iter().filter(|p| p.kind == kind && Pool::parse(&p.pool) == Some(pool)).map(|p| p.count).sum::<usize>();
        for p in &self.placements {
            let Some(pool) = Pool::parse(&p.pool) else { bail!("plan names pool {} which is not host or device:N", p.pool) };
            if pool != Pool::Host && pool != main {
                bail!("plan places {} on {} but this process runs {}", p.kind, p.pool, main.id());
            }
        }
        let n_layers = scope.layers.len();
        let on_device = count("layer", main);
        let state_on_device = count("state", main);
        let experts_on_device = count("experts", main);
        if on_device > n_layers || state_on_device > n_layers {
            bail!("plan places {on_device} layers and {state_on_device} states of the {n_layers} layers this process holds");
        }
        let (layers, _) = layer_bytes(loaded, self.n_ctx, self.n_batch, self.n_seq);
        let outside = scope_outside(loaded, scope);
        let slot_of: BTreeMap<usize, usize> = self.expert_slots.iter().map(|s| (s.layer, s.slots)).collect();
        let resident_of: BTreeMap<usize, usize> = self.kv_resident.iter().map(|r| (r.layer, r.rows)).collect();
        let min_rows = min_resident(self.n_batch);
        let mut weights_bytes = if main == Pool::Host { 0 } else { outside };
        let mut state_bytes = 0;
        let mut pinned = Vec::new();
        let mut resident_seen = 0;
        let mut plan = Vec::with_capacity(n_layers);
        for (i, l) in scope.layers().enumerate() {
            let lb = layers.get(&l).cloned().unwrap_or_default();
            let weights = if i < on_device { main } else { Pool::Host };
            let state = if i < state_on_device { main } else { Pool::Host };
            let mut slots = 0;
            if lb.n_expert > 0 && weights == main && main != Pool::Host {
                match slot_of.get(&l) {
                    Some(&n) if n > 0 => {
                        slots = n.min(lb.n_expert);
                        pinned.extend(lb.expert_wids.iter().copied());
                        weights_bytes += lb.experts / lb.n_expert as u64 * slots as u64;
                    }
                    _ if resident_seen < experts_on_device => {
                        resident_seen += 1;
                        weights_bytes += lb.experts;
                    }
                    _ => {}
                }
            }
            if weights == main && main != Pool::Host {
                weights_bytes += lb.dense;
            }
            let mut kv_resident = 0;
            if state == main && main != Pool::Host {
                match resident_of.get(&l).copied().filter(|r| *r > 0) {
                    Some(rows) => {
                        if lb.stream_row == 0 {
                            bail!("plan keeps {rows} resident rows for layer {l}, which has no full attention cache to stream");
                        }
                        if rows < min_rows || rows >= self.n_ctx {
                            bail!("plan keeps {rows} resident rows for layer {l}, between {min_rows} and the context less {CTX_STEP} expected");
                        }
                        kv_resident = rows;
                        state_bytes += lb.state_resident(rows);
                    }
                    None => state_bytes += lb.state,
                }
            }
            plan.push(LayerPlan { layer: l, weights, state, expert_cache_slots: slots, kv_resident });
        }
        if main == Pool::Host {
            weights_bytes = outside + scope.layers().map(|l| layers.get(&l).map_or(0, |lb| lb.dense + lb.experts)).sum::<u64>();
            state_bytes = crate::plan::state_bytes(loaded, self.n_ctx, self.n_batch, self.n_seq, scope);
        }
        Ok(Plan { layers: plan, pinned, weights_bytes, state_bytes, n_ctx: self.n_ctx, n_batch: self.n_batch, n_seq: self.n_seq })
    }
}

/// Pool and kind totals a plan document expects after load, keyed pool.kind as the alloc events sum
pub fn expected_totals(doc: &PlanDoc) -> BTreeMap<String, u64> {
    let mut m = BTreeMap::new();
    for p in &doc.pools {
        m.insert(format!("{}.weights", p.id), p.weights);
        m.insert(format!("{}.state", p.id), p.state);
        m.insert(format!("{}.experts", p.id), p.experts);
    }
    m
}

//! Reference backend, every op in plain f32 on the host, the parity oracle for every other backend
pub mod attention;
pub mod hyper;
pub mod moe;
pub mod ops;
pub mod recurrent;
pub mod rope;
pub mod sparse;

use anyhow::{Result, anyhow, bail};
use kf_core::{AllocKind, Backend, Graph, Inputs, Op, Plan, Pool, WeightSource, Wid};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A row major f32 tensor
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Buf {
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

impl Buf {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self { data: vec![0.0; rows * cols], rows, cols }
    }
    pub fn from_vec(data: Vec<f32>, rows: usize, cols: usize) -> Self {
        Self { data, rows, cols }
    }
    pub fn row(&self, r: usize) -> &[f32] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }
    pub fn row_mut(&mut self, r: usize) -> &mut [f32] {
        &mut self.data[r * self.cols..(r + 1) * self.cols]
    }
}

/// Key and value rows of one attention layer in half precision, a ring of cap rows indexed by position
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct KvLayer {
    pub k: Vec<half::f16>,
    pub v: Vec<half::f16>,
    pub n: usize,
    pub cap: usize,
    pub k_width: usize,
    pub v_width: usize,
}

impl KvLayer {
    pub fn with_capacity(cap: usize, k_width: usize, v_width: usize) -> Self {
        Self { k: vec![half::f16::ZERO; cap * k_width], v: vec![half::f16::ZERO; cap * v_width], n: 0, cap, k_width, v_width }
    }

    pub fn bytes(&self) -> u64 {
        ((self.k.len() + self.v.len()) * 2) as u64
    }

    pub fn k_row(&self, j: usize) -> &[half::f16] {
        let s = (j % self.cap) * self.k_width;
        &self.k[s..s + self.k_width]
    }

    pub fn v_row(&self, j: usize) -> &[half::f16] {
        let s = (j % self.cap) * self.v_width;
        &self.v[s..s + self.v_width]
    }

    /// The ring with only the live rows kept, every row when positions have wrapped around it
    pub fn trimmed(&self) -> Self {
        let rows = self.n.min(self.cap);
        Self { k: self.k[..rows * self.k_width].to_vec(), v: self.v[..rows * self.v_width].to_vec(), n: self.n, cap: self.cap, k_width: self.k_width, v_width: self.v_width }
    }

    /// The ring at its full size again after a trim, the rows past the live ones zero
    pub fn expanded(mut self) -> Self {
        self.k.resize(self.cap * self.k_width, half::f16::ZERO);
        self.v.resize(self.cap * self.v_width, half::f16::ZERO);
        self
    }

    /// Writes new key and value rows at the positions past n, rounding to half precision
    pub fn append(&mut self, k: &Buf, v: &Buf) -> Result<()> {
        if k.cols != self.k_width || v.cols != self.v_width || k.rows != v.rows {
            bail!("cache append of {}x{} keys and {}x{} values onto widths {} and {}", k.rows, k.cols, v.rows, v.cols, self.k_width, self.v_width);
        }
        for t in 0..k.rows {
            let slot = (self.n + t) % self.cap;
            for (d, s) in self.k[slot * self.k_width..(slot + 1) * self.k_width].iter_mut().zip(k.row(t)) {
                *d = half::f16::from_f32(*s);
            }
            for (d, s) in self.v[slot * self.v_width..(slot + 1) * self.v_width].iter_mut().zip(v.row(t)) {
                *d = half::f16::from_f32(*s);
            }
        }
        self.n += k.rows;
        Ok(())
    }
}

/// Everything one layer keeps between passes
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct LayerState {
    pub kv: Option<KvLayer>,
    /// Causal conv history rows by slot, each (kernel-1)*dilation rows of channels
    pub conv: HashMap<usize, Buf>,
    /// Recurrent matrix state, delta net, ssm, wkv or lightning
    pub ssm: Vec<f32>,
    /// Last normed row by token shift slot
    pub shift: HashMap<usize, Vec<f32>>,
    /// Indexer key rows, one per position or one per compressed block
    pub indexer: Buf,
    pub indexer_ratio: usize,
    /// Compressed rows by compressor slot
    pub compressed: HashMap<usize, Buf>,
    pub ring_kv: HashMap<usize, Buf>,
    pub ring_score: HashMap<usize, Buf>,
    pub ring_start: HashMap<usize, usize>,
    /// Selection of the last indexer pass, reused by shared layers
    pub topk: Buf,
    /// Candidate blocks of the last indexer pass
    pub cand: Buf,
    /// State after each of the last tokens by field, for roll back
    pub snaps: HashMap<String, Vec<(usize, Buf)>>,
}

impl LayerState {
    fn has_recurrent(&self) -> bool {
        !self.ssm.is_empty() || !self.conv.is_empty() || !self.shift.is_empty() || !self.ring_kv.is_empty()
    }

    /// Bytes the state holds counting the capacity reserved for growth and the roll back snapshots
    pub fn bytes(&self) -> u64 {
        let buf = |b: &Buf| (b.data.capacity() * 4) as u64;
        let snaps = (1 + kf_core::STATE_SNAPSHOTS) as u64;
        let mut n = self.kv.as_ref().map_or(0, KvLayer::bytes);
        n += self.conv.values().map(buf).sum::<u64>() * snaps;
        n += (self.ssm.len() * 4) as u64 * snaps;
        n += self.shift.values().map(|v| (v.len() * 4) as u64).sum::<u64>() * snaps;
        n += buf(&self.indexer);
        n += self.compressed.values().map(buf).sum::<u64>() * snaps;
        n += self.ring_kv.values().chain(self.ring_score.values()).map(buf).sum::<u64>() * snaps;
        n
    }
}

/// The state of one sequence slot, every layer's state with the tokens fed so far
#[derive(Clone, Debug, Default)]
pub struct SeqState {
    pub layers: HashMap<usize, LayerState>,
    pub tokens: Vec<u32>,
}

/// Ops that read or write per sequence state and so run one slot run at a time
fn stateful(op: &Op) -> bool {
    matches!(
        op,
        Op::Attention { .. }
            | Op::CompressKv { .. }
            | Op::CompressedAttention { .. }
            | Op::IndexerTopK { .. }
            | Op::CausalConv { .. }
            | Op::SsmScan { .. }
            | Op::GatedDeltaNet { .. }
            | Op::Wkv6 { .. }
            | Op::Gla { .. }
            | Op::Wkv7 { .. }
            | Op::TokenShift { .. }
            | Op::LightningAttn { .. }
            | Op::PleHash { .. }
            | Op::EngramHash { .. }
    )
}

pub struct CpuBackend {
    /// One state per sequence slot
    pub seqs: Vec<SeqState>,
    /// The slot the stateful ops read and write, set per run of rows
    pub cur: usize,
    pub small: HashMap<Wid, Vec<f32>>,
    /// Context length the caches are sized for, zero until a plan sets it
    pub n_ctx: usize,
    /// Most rows one pass may feed, the ring slack of windowed caches
    pub n_batch: usize,
    /// State bytes reported so far, the next report logs the growth since
    state_bytes: u64,
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuBackend {
    pub fn new() -> Self {
        Self { seqs: vec![SeqState::default()], cur: 0, small: HashMap::new(), n_ctx: 0, n_batch: 0, state_bytes: 0 }
    }

    /// Sizes every cache made from now on for a context length and a largest pass, with the slots given
    pub fn set_limits(&mut self, n_ctx: usize, n_batch: usize, n_seq: usize) {
        self.n_ctx = n_ctx;
        self.n_batch = n_batch;
        self.seqs = (0..n_seq.max(1)).map(|_| SeqState::default()).collect();
        self.cur = 0;
    }

    pub fn n_seq(&self) -> usize {
        self.seqs.len()
    }

    /// Layer states of the current slot
    pub fn layers(&self) -> &HashMap<usize, LayerState> {
        &self.seqs[self.cur].layers
    }

    pub fn layer(&mut self, l: usize) -> &mut LayerState {
        let cur = self.cur;
        self.seqs[cur].layers.entry(l).or_default()
    }

    /// Token ids the current slot has seen before this pass
    pub fn tokens(&self) -> &[u32] {
        &self.seqs[self.cur].tokens
    }

    /// Logs the host state bytes grown or released since the last report
    pub fn account_state(&mut self) {
        let now: u64 = self.seqs.iter().flat_map(|s| s.layers.values()).map(LayerState::bytes).sum();
        if now > self.state_bytes {
            kf_core::event::alloc(Pool::Host, AllocKind::State, now - self.state_bytes);
        } else if now < self.state_bytes {
            kf_core::event::release(Pool::Host, AllocKind::State, self.state_bytes - now);
        }
        self.state_bytes = now;
    }

    /// The cache of a layer in the current slot, made at the ring size the window and the limits give when it does not exist
    pub fn cache_for(&mut self, l: usize, k_width: usize, v_width: usize, window: Option<usize>) -> Result<&mut KvLayer> {
        if self.n_ctx == 0 || self.n_batch == 0 {
            bail!("layer {l} cache before a plan set the context and batch");
        }
        let cap = kf_core::cache_rows(self.n_ctx, self.n_batch, window);
        let st = self.layer(l);
        if st.kv.is_none() {
            st.kv = Some(KvLayer::with_capacity(cap, k_width, v_width));
        }
        let kv = st.kv.as_mut().ok_or_else(|| anyhow!("layer {l} lost its cache"))?;
        if kv.k_width != k_width || kv.v_width != v_width {
            bail!("layer {l} cache widths {} and {} against {k_width} and {v_width}", kv.k_width, kv.v_width);
        }
        Ok(kv)
    }

    /// Dequantized copy of a small weight such as a norm or a bias, kept across passes
    pub fn small(&mut self, w: Wid, weights: &dyn WeightSource) -> Result<&[f32]> {
        if !self.small.contains_key(&w) {
            let v = dequant_all(w, weights)?;
            self.small.insert(w, v);
        }
        Ok(self.small[&w].as_slice())
    }

    pub fn small_opt(&mut self, w: Option<Wid>, weights: &dyn WeightSource) -> Result<Option<Vec<f32>>> {
        Ok(match w {
            Some(w) => Some(self.small(w, weights)?.to_vec()),
            None => None,
        })
    }

    /// Runs one op on resolved input buffers, a stateful op once per slot run of rows, the unit a device backend hands back to the host
    pub fn exec_node(&mut self, op: &Op, ins: &[Buf], inputs: &Inputs, weights: &dyn WeightSource) -> Result<Buf> {
        if !stateful(op) {
            return self.exec_one(op, ins, inputs, weights);
        }
        let runs = inputs.runs();
        let n_rows = inputs.tokens.len();
        if let Some(bad) = ins.iter().find(|b| b.rows != n_rows) {
            bail!("stateful op over {} rows for a pass of {n_rows} tokens", bad.rows);
        }
        if let Some((seq, _)) = runs.iter().find(|(s, _)| *s >= self.seqs.len()) {
            bail!("sequence slot {seq} of {}", self.seqs.len());
        }
        if runs.len() == 1 {
            self.cur = runs[0].0;
            return self.exec_one(op, ins, inputs, weights);
        }
        let mut out: Option<Buf> = None;
        for (seq, range) in runs {
            self.cur = seq;
            let sub_ins: Vec<Buf> = ins.iter().map(|b| ops::rows(b, range.start, range.len())).collect::<Result<_>>()?;
            let sub = inputs.slice(range.clone());
            let y = self.exec_one(op, &sub_ins, &sub, weights)?;
            if y.rows != range.len() {
                bail!("stateful op gave {} rows for a run of {}", y.rows, range.len());
            }
            match &mut out {
                Some(o) => {
                    if o.cols != y.cols {
                        bail!("stateful op gave {} columns after {}", y.cols, o.cols);
                    }
                    o.data.extend_from_slice(&y.data);
                    o.rows += y.rows;
                }
                None => out = Some(y),
            }
        }
        self.cur = 0;
        out.ok_or_else(|| anyhow!("stateful op over no rows"))
    }

    fn exec_one(&mut self, op: &Op, ins: &[Buf], inputs: &Inputs, weights: &dyn WeightSource) -> Result<Buf> {
        match op {
            Op::Input { name } => match *name {
                "tokens" => Ok(Buf { data: inputs.tokens.iter().map(|x| *x as f32).collect(), rows: inputs.tokens.len(), cols: 1 }),
                "positions" => Ok(Buf { data: inputs.positions.iter().map(|x| *x as f32).collect(), rows: inputs.positions.len(), cols: 1 }),
                "hidden" => {
                    if inputs.hidden_cols == 0 || !inputs.hidden.len().is_multiple_of(inputs.hidden_cols) {
                        bail!("hidden input of {} values in rows of {}", inputs.hidden.len(), inputs.hidden_cols);
                    }
                    Ok(Buf { data: inputs.hidden.clone(), rows: inputs.hidden.len() / inputs.hidden_cols, cols: inputs.hidden_cols })
                }
                other => bail!("unknown graph input {other}"),
            },
            Op::Rope { .. } | Op::AttnTemp { .. } => rope::exec(self, op, ins, inputs, weights),
            Op::Attention { .. } | Op::AttentionFull { .. } => attention::exec(self, op, ins, inputs, weights),
            Op::MoeFfn { .. } => moe::exec(self, op, ins, inputs, weights),
            Op::CausalConv { .. }
            | Op::SsmScan { .. }
            | Op::GatedDeltaNet { .. }
            | Op::Wkv6 { .. }
            | Op::Gla { .. }
            | Op::Wkv7 { .. }
            | Op::TokenShift { .. }
            | Op::LightningAttn { .. } => recurrent::exec(self, op, ins, inputs, weights),
            Op::IndexerTopK { .. } | Op::CompressKv { .. } | Op::CompressedAttention { .. } => sparse::exec(self, op, ins, inputs, weights),
            Op::HcMix { .. }
            | Op::HcMixGated { .. }
            | Op::HcSinkhorn { .. }
            | Op::HcCombine { .. }
            | Op::PleHash { .. }
            | Op::EngramHash { .. }
            | Op::LoraSwitched { .. }
            | Op::MapTokens { .. } => hyper::exec(self, op, ins, inputs, weights),
            _ => ops::exec(self, op, ins, inputs, weights),
        }
    }
}

/// Whole tensor as f32, row by row
pub fn dequant_all(w: Wid, weights: &dyn WeightSource) -> Result<Vec<f32>> {
    let q = kf_quant::by_id(weights.ggml_type(w)).ok_or_else(|| anyhow!("{}: unknown ggml type {}", weights.name(w), weights.ggml_type(w)))?;
    let dims = weights.dims(w);
    let inner = dims.first().copied().unwrap_or(0);
    let rows: usize = dims.iter().skip(1).product::<usize>().max(1);
    let rb = q.info().row_bytes(inner);
    let bytes = weights.bytes(w)?;
    let mut out = vec![0.0f32; inner * rows];
    for r in 0..rows {
        q.dequant_row(&bytes[r * rb..(r + 1) * rb], &mut out[r * inner..(r + 1) * inner]);
    }
    Ok(out)
}

impl Backend for CpuBackend {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn supports(&self, _op: &Op) -> bool {
        true
    }

    fn apply_plan(&mut self, plan: &Plan) -> Result<()> {
        if let Some(l) = plan.layers.iter().find(|l| l.weights != Pool::Host || l.state != Pool::Host || l.expert_cache_slots != 0) {
            bail!("cpu backend cannot place layer {} on {:?}", l.layer, l.weights);
        }
        if plan.n_ctx == 0 || plan.n_batch == 0 || plan.n_seq == 0 {
            bail!("plan without a context length, batch, or sequence slots");
        }
        self.reset();
        self.set_limits(plan.n_ctx, plan.n_batch, plan.n_seq);
        self.account_state();
        Ok(())
    }

    fn run(&mut self, g: &Graph, inputs: &Inputs, weights: &dyn WeightSource) -> Result<Vec<f32>> {
        let runs = inputs.runs();
        if let Some((seq, _)) = runs.iter().find(|(s, _)| *s >= self.seqs.len()) {
            bail!("sequence slot {seq} of {}", self.seqs.len());
        }
        let mut t: Vec<Option<Buf>> = vec![None; g.n_tensors()];
        for node in &g.nodes {
            let mut ins = Vec::with_capacity(node.inputs.len());
            for id in &node.inputs {
                ins.push(t[id.0 as usize].clone().ok_or_else(|| anyhow!("tensor {} used before it was computed", id.0))?);
            }
            let out = self.exec_node(&node.op, &ins, inputs, weights).map_err(|e| anyhow!("{}: {e}", short_op(&node.op)))?;
            t[node.out.0 as usize] = Some(out);
        }
        for (seq, range) in runs {
            self.seqs[seq].tokens.extend_from_slice(&inputs.tokens[range]);
        }
        self.account_state();
        let out = g.output.ok_or_else(|| anyhow!("graph has no output"))?;
        Ok(t[out.0 as usize].take().ok_or_else(|| anyhow!("tensor {} used before it was computed", out.0))?.data)
    }

    fn truncate(&mut self, seq: usize, n: usize) -> Result<()> {
        let Some(s) = self.seqs.get_mut(seq) else { bail!("sequence slot {seq} of {}", self.seqs.len()) };
        if n == 0 {
            self.reset_seq(seq);
            return Ok(());
        }
        let past = s.tokens.len();
        if n >= past {
            return Ok(());
        }
        for st in s.layers.values_mut() {
            if let Some(kv) = &mut st.kv
                && kv.n > n
            {
                kv.n = n;
            }
            let ratio = st.indexer_ratio.max(1);
            if st.indexer.rows > n / ratio {
                st.indexer.rows = n / ratio;
                st.indexer.data.truncate(st.indexer.rows * st.indexer.cols);
            }
            for c in st.compressed.values_mut() {
                let keep = c.rows.min(n / ratio);
                c.rows = keep;
                c.data.truncate(keep * c.cols);
            }
            if st.has_recurrent() {
                recurrent::roll_back(st, past, n)?;
            }
        }
        s.tokens.truncate(n);
        Ok(())
    }

    fn reset(&mut self) {
        for s in &mut self.seqs {
            *s = SeqState::default();
        }
        self.cur = 0;
        self.account_state();
    }

    fn reset_seq(&mut self, seq: usize) {
        if let Some(s) = self.seqs.get_mut(seq) {
            *s = SeqState::default();
        }
        self.account_state();
    }

    fn save_state(&mut self, seq: usize) -> Result<Vec<u8>> {
        self.saved(seq)?.to_bytes()
    }

    fn load_state(&mut self, seq: usize, bytes: &[u8]) -> Result<()> {
        self.restore(seq, SavedState::from_bytes(bytes)?)
    }
}

impl CpuBackend {
    /// The state of one slot as it travels, every layer's state with the tokens fed, cache rings trimmed to their live rows
    pub fn saved(&self, seq: usize) -> Result<SavedState> {
        let Some(s) = self.seqs.get(seq) else { bail!("sequence slot {seq} of {}", self.seqs.len()) };
        let mut layers = s.layers.clone();
        for st in layers.values_mut() {
            if let Some(kv) = &st.kv {
                st.kv = Some(kv.trimmed());
            }
        }
        Ok(SavedState { n_ctx: self.n_ctx, n_batch: self.n_batch, tokens: s.tokens.clone(), layers })
    }

    /// Replaces one slot's state with a saved one made under the same context and batch
    pub fn restore(&mut self, seq: usize, mut saved: SavedState) -> Result<()> {
        if saved.n_ctx != self.n_ctx || saved.n_batch != self.n_batch {
            bail!("state saved under a context of {} and batch of {}, this plan has {} and {}", saved.n_ctx, saved.n_batch, self.n_ctx, self.n_batch);
        }
        let Some(s) = self.seqs.get_mut(seq) else { bail!("sequence slot {seq} of {}", self.seqs.len()) };
        for st in saved.layers.values_mut() {
            if let Some(kv) = st.kv.take() {
                st.kv = Some(kv.expanded());
            }
        }
        s.layers = saved.layers;
        s.tokens = saved.tokens;
        self.account_state();
        Ok(())
    }
}

/// The state of one sequence as it travels between processes, the same bytes whichever backend made it
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SavedState {
    pub n_ctx: usize,
    pub n_batch: usize,
    pub tokens: Vec<u32>,
    pub layers: HashMap<usize, LayerState>,
}

impl SavedState {
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        bincode::serialize(self).map_err(|e| anyhow!("serialize sequence state: {e}"))
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        bincode::deserialize(bytes).map_err(|e| anyhow!("parse sequence state: {e}"))
    }
}

/// The variant name of an op for error messages
fn short_op(op: &Op) -> String {
    let s = format!("{op:?}");
    s.split([' ', '(', '{']).next().unwrap_or("op").to_string()
}

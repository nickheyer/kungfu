use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Tensor produced by a graph node
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Tid(pub u32);

/// Weight in the model's weight table
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Wid(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum RopeKind {
    Normal,
    #[default]
    Neox,
    MRope,
    IMRope,
    /// Two position streams over the two halves of each head, the vision tower rope of Qwen2 VL
    Vision,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Yarn {
    pub ext_factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RopeParams {
    pub kind: RopeKind,
    pub dims: usize,
    pub base: f32,
    pub scale: f32,
    pub orig_ctx: usize,
    /// Multiplies cos and sin, the yarn mscale folded with rope.scaling.attn_factor
    pub attn_factor: f32,
    pub yarn: Option<Yarn>,
    pub sections: [usize; 4],
    /// Per frequency divisors from the file, such as rope_freqs
    pub factors: Option<Wid>,
}

impl Default for RopeParams {
    fn default() -> Self {
        Self { kind: RopeKind::Neox, dims: 0, base: 10000.0, scale: 1.0, orig_ctx: 0, attn_factor: 1.0, yarn: None, sections: [0; 4], factors: None }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Act {
    Silu,
    Gelu,
    GeluErf,
    Relu,
    ReluSqr,
    SwigluOai,
}

/// Clamp applied inside a gated activation, unfused caps act(gate) and fused caps gate before the act
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum GluLimit {
    Unfused(f32),
    Fused(f32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Gating {
    #[default]
    Softmax,
    Sigmoid,
    SoftmaxWeight,
    SqrtSoftplus,
}

/// Which scores the top k ranks, probs plus bias, raw logits, or sigmoid of the logits
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Selection {
    #[default]
    Probs,
    Logits,
    SigmoidLogits,
}

/// Top k weight normalization, divide by the sum clamped at the f16 minimum or by the sum plus 1e-20
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum WeightNorm {
    #[default]
    None,
    Clamp,
    AddEps,
}

/// Grouped routing, experts split into groups and only some groups eligible
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Groups {
    pub n_group: usize,
    pub n_used: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SwaType {
    #[default]
    Standard,
    Chunked,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Window {
    pub size: usize,
    pub kind: SwaType,
}

/// Key selection an indexer performs, dense top k cells, pooled runs, score blocks, or compressed rows
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum IndexerKind {
    Dense,
    Pool { kpool: usize, tail: usize },
    Block { block: usize, local: usize },
    Compressed { ratio: usize, overlap: bool },
}

/// Softmax pooled key compression of DeepSeek V4, ratio tokens per row, ape optional, norm with its eps
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Compressor {
    pub ratio: usize,
    pub overlap: bool,
    pub ape: Option<Wid>,
    pub norm: Wid,
    pub eps: f32,
}

/// Index key derived from each compressed row, projected, normed and roped at the block start
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IndexKey {
    pub k: Wid,
    pub norm: Wid,
    pub eps: f32,
    pub rope: RopeParams,
    pub head_dim: usize,
}

/// Treatment of pooled indexer keys, softmax pooling with ape or mean pooling followed by a norm and rope
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PoolKeys {
    pub ape: Option<Wid>,
    pub norm: Option<Wid>,
    pub eps: f32,
    pub rope: Option<RopeParams>,
}

/// Two level block selection, computed here when source is none, else restricted to the source layer's blocks
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Candidate {
    pub source: Option<usize>,
    pub top_blocks: usize,
    pub block: usize,
}

/// The op set, the one closed set in the engine and the backend ABI
#[derive(Clone, Debug, PartialEq)]
pub enum Op {
    Input { name: &'static str },
    /// Rows of a table for the token input
    Embed { weight: Wid },
    /// Rows of a table for an index tensor input holding integers as f32
    Gather { weight: Wid },
    /// A vector weight repeated as rows
    RowBroadcast { weight: Wid, rows: usize },
    /// x times the transpose of the weight, bias added, then the scalar scale tensor
    MatMul { weight: Wid, bias: Option<Wid>, scale: Option<Wid> },
    /// Matmul against slice index of a weight stacked along its third dimension
    MatMulSlice { weight: Wid, index: usize },
    /// Head h of the input against slice h of a weight stacked along its third dimension
    MatMulHeads { weight: Wid },
    /// A constant tensor
    Fill { rows: usize, cols: usize, value: f32 },
    /// Inputs x and slot, adds b at slot times a at slot times x
    LoraSwitched { a: Wid, b: Wid },
    RmsNorm { weight: Option<Wid>, bias: Option<Wid>, eps: f32 },
    LayerNorm { weight: Option<Wid>, bias: Option<Wid>, eps: f32 },
    /// Per row x over sqrt of the squared sum plus eps
    L2Norm { eps: f32 },
    /// Inputs x and positions, rotates dims offset to offset plus params dims of each head, inverse undoes it
    Rope { params: RopeParams, n_head: usize, head_dim: usize, offset: usize, inverse: bool },
    /// Inputs x and positions, log form ln(floor((pos+offset)/floor_scale)+1)*scale+1, pow form (1+(pos+offset)/floor_scale)^scale
    AttnTemp { floor_scale: f32, scale: f32, offset: f32, pow: bool },
    /// Inputs q k v, or q alone with kv_source, plus a top k index tensor when top_k is set
    Attention {
        layer: usize,
        n_head: usize,
        n_head_kv: usize,
        head_dim: usize,
        head_dim_v: usize,
        scale: f32,
        window: Option<Window>,
        softcap: f32,
        sinks: Option<Wid>,
        alibi_bias: f32,
        kv_source: Option<usize>,
        top_k: Option<usize>,
    },
    /// Inputs q k v over the rows of the pass alone, every row attending to every other, no cache
    AttentionFull { n_head: usize, head_dim: usize, scale: f32 },
    /// x times sigmoid of 1.702 x
    GeluQuick,
    /// Rows of a side by side grid averaged over kernel by kernel blocks, raster order in and out
    PoolGrid { side: usize, kernel: usize },
    /// Inputs kv rows and score rows, pools each completed block into the slot's compressed rows and returns the new rows
    CompressKv { layer: usize, slot: usize, comp: Compressor, rope: RopeParams, head_dim: usize, index: Option<IndexKey> },
    /// DeepSeek V4 attention, inputs q and kv plus a top k index when set, keys from the raw window and the source layer's compressed rows, K equals V
    CompressedAttention {
        layer: usize,
        n_head: usize,
        head_dim: usize,
        scale: f32,
        window: usize,
        sinks: Wid,
        rope: RopeParams,
        ratio: usize,
        kv_source: Option<usize>,
        top_k: Option<usize>,
    },
    /// Inputs q k w plus the gate rows of an ape pool, appends k to the layer's indexer cache and returns the selected cells or blocks per row
    IndexerTopK { layer: usize, n_head: usize, head_dim: usize, top_k: usize, kind: IndexerKind, source: Option<usize>, keys_from: Option<usize>, pool: Option<PoolKeys>, candidate: Option<Candidate> },
    /// Inputs x or x and router logits
    MoeFfn {
        router: Wid,
        router_bias: Option<Wid>,
        probs_bias: Option<Wid>,
        gate_exps: Option<Wid>,
        up_exps: Wid,
        down_exps: Option<Wid>,
        gate_b: Option<Wid>,
        up_b: Option<Wid>,
        down_b: Option<Wid>,
        exps_scale: Option<Wid>,
        n_expert: usize,
        n_used: usize,
        gating: Gating,
        selection: Selection,
        norm_weights: WeightNorm,
        scale: f32,
        act: Act,
        limit: Option<GluLimit>,
        groups: Option<Groups>,
        fused_gate_up: bool,
        weight_before_ffn: bool,
        index_div: usize,
        /// Each expert output times this weight then rms normed before the routing weight
        post_norm: Option<Wid>,
        post_norm_eps: f32,
        router_in: bool,
        selected_in: bool,
    },
    GatedAct { act: Act, limit: Option<GluLimit> },
    GatedActFused { act: Act, limit: Option<GluLimit> },
    /// Inputs gate and up, beta tanh(g/beta) sigmoid(g) times lb tanh(u/lb) or u
    Situ { beta: f32, linear_beta: f32 },
    Add,
    Sub,
    Mul,
    Div,
    MulWeight { weight: Wid },
    DivWeight { weight: Wid },
    AddWeight { weight: Wid },
    Scale(f32),
    ScaleBias { scale: f32, bias: f32 },
    Clamp { min: f32, max: f32 },
    Silu,
    Gelu,
    GeluErf,
    Relu,
    ReluSqr,
    Sigmoid,
    Tanh,
    Exp,
    Sqrt,
    Sqr,
    Abs,
    Sign,
    Log,
    Softplus { beta: f32 },
    Xielu { alpha_n: f32, alpha_p: f32, beta: f32, eps: f32 },
    Softmax,
    Softcap(f32),
    Reshape { rows: usize, cols: usize },
    Cols { start: usize, len: usize },
    Rows { start: usize, len: usize },
    LastRows { n: usize },
    Concat,
    Tile { n: usize },
    PadCols { to: usize },
    SumRows,
    Round,
    MaskCols { start: usize, len: usize, value: f32 },
    /// Input tokens, maps each id through the table to a scalar, unlisted ids give the default or themselves
    MapTokens { table: Arc<Vec<(u32, f32)>>, default: Option<f32> },
    /// Depthwise causal conv over channels with a carried state row per layer and slot
    CausalConv { layer: usize, slot: usize, kernel: usize, dilation: usize, weight: Wid, bias: Option<Wid> },
    /// Inputs x, dt, B, C with the carried state, Mamba1 when a has d_state rows else Mamba2
    SsmScan { layer: usize, n_head: usize, head_dim: usize, d_state: usize, n_group: usize, a: Wid, mamba2: bool },
    /// Inputs q k v g beta, the gated delta rule over a carried state per value head
    GatedDeltaNet { layer: usize, n_head_k: usize, n_head_v: usize, head_dim: usize, gate_per_channel: bool },
    /// Inputs k v r w with the bonus weight first, rwkv6 recurrence
    Wkv6 { layer: usize, head_size: usize, first: Wid },
    /// Inputs k v q g, gated linear attention recurrence
    Gla { layer: usize, head_size: usize, scale: f32 },
    /// Inputs r w k v a b, rwkv7 recurrence
    Wkv7 { layer: usize, head_size: usize },
    /// Returns the previous rows from the carried slot and stores the last row
    TokenShift { layer: usize, slot: usize },
    /// Inputs q k v, per head decay recurrence of MiniMax lightning attention
    LightningAttn { layer: usize, n_head: usize, head_dim: usize, decay: Arc<Vec<f32>> },
    /// Inputs streams and weights per stream, the weighted sum of the streams
    HcMix { n_streams: usize },
    /// Inputs streams and per element gates, the sum of the streams times sigmoid of the gates, scaled
    HcMixGated { n_streams: usize, scale: f32 },
    /// Input raw comb values, sinkhorn normalized combination matrix per row
    HcSinkhorn { n_streams: usize, iters: usize, eps: f32 },
    /// Inputs streams, out, post weights and an optional comb, the updated streams
    HcCombine { n_streams: usize, comb: bool },
    /// Input tokens, hashed n gram row ids per head from the carried token history
    PleHash { layer: usize, ngram: usize, heads_per_ngram: usize, multipliers: Arc<Vec<u64>>, offsets: Arc<Vec<u32>>, vocab: Arc<Vec<u32>>, eos: u32, image: Option<u32> },
    /// Input tokens, DeepSeek V4.1 engram row ids with a token map and pad
    EngramHash { layer: usize, ngram: usize, heads_per_ngram: usize, multipliers: Arc<Vec<u64>>, offsets: Arc<Vec<u32>>, primes: Arc<Vec<u32>>, token_map: Arc<Vec<u32>>, pad: u32 },
}

#[derive(Clone, Debug)]
pub struct Node {
    pub op: Op,
    pub inputs: Vec<Tid>,
    pub out: Tid,
    /// Block index the node belongs to, none for the embedding and the head
    pub layer: Option<usize>,
}

#[derive(Clone, Debug, Default)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub inputs: Vec<(&'static str, Tid)>,
    pub output: Option<Tid>,
    next: u32,
}

impl Graph {
    pub fn n_tensors(&self) -> usize {
        self.next as usize
    }
}

/// Appends nodes to a graph and hands back tensor ids
pub struct GraphBuilder {
    pub graph: Graph,
    pub layer: Option<usize>,
}

impl Default for GraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphBuilder {
    pub fn new() -> Self {
        Self { graph: Graph::default(), layer: None }
    }

    fn fresh(&mut self) -> Tid {
        let t = Tid(self.graph.next);
        self.graph.next += 1;
        t
    }

    pub fn push(&mut self, op: Op, inputs: &[Tid]) -> Tid {
        let out = self.fresh();
        self.graph.nodes.push(Node { op, inputs: inputs.to_vec(), out, layer: self.layer });
        out
    }

    pub fn input(&mut self, name: &'static str) -> Tid {
        let t = self.push(Op::Input { name }, &[]);
        self.graph.inputs.push((name, t));
        t
    }

    pub fn add(&mut self, a: Tid, b: Tid) -> Tid {
        self.push(Op::Add, &[a, b])
    }

    pub fn sub(&mut self, a: Tid, b: Tid) -> Tid {
        self.push(Op::Sub, &[a, b])
    }

    pub fn mul(&mut self, a: Tid, b: Tid) -> Tid {
        self.push(Op::Mul, &[a, b])
    }

    pub fn div(&mut self, a: Tid, b: Tid) -> Tid {
        self.push(Op::Div, &[a, b])
    }

    pub fn scale(&mut self, a: Tid, s: f32) -> Tid {
        if s == 1.0 { a } else { self.push(Op::Scale(s), &[a]) }
    }

    pub fn reshape(&mut self, a: Tid, rows: usize, cols: usize) -> Tid {
        self.push(Op::Reshape { rows, cols }, &[a])
    }

    pub fn cols(&mut self, a: Tid, start: usize, len: usize) -> Tid {
        self.push(Op::Cols { start, len }, &[a])
    }

    pub fn concat(&mut self, a: Tid, b: Tid) -> Tid {
        self.push(Op::Concat, &[a, b])
    }

    pub fn finish(mut self, output: Tid) -> Graph {
        self.graph.output = Some(output);
        self.graph
    }
}

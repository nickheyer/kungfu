//! The vision tower of a projector file, images preprocessed as the reference does and encoded into the text model's embedding space
pub mod preprocess;

use anyhow::{Context, Result, anyhow, bail};
use kf_core::{Backend, Graph, GraphBuilder, Inputs, Op, RopeKind, RopeParams, Tid, WeightSource, Wid};
use kf_gguf::Gguf;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

/// The projector families this tower composes
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Projector {
    /// Qwen2 VL, two temporal patch convolutions, 2D rotary attention, a 2 by 2 merger through a GELU MLP
    Qwen2Vl,
    /// Gemma 3, SigLIP with learned positions, 4 by 4 average pooling, a soft embedding norm and a projection
    Gemma3,
}

/// Hyperparameters of the vision tower as the file names them
#[derive(Clone, Debug)]
pub struct Hparams {
    pub projector: Projector,
    pub image_size: usize,
    pub patch_size: usize,
    pub n_embd: usize,
    pub n_ff: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub eps: f32,
    pub mean: [f32; 3],
    pub std: [f32; 3],
    pub projection_dim: usize,
    pub n_merge: usize,
    /// Pixel bounds of the dynamic resize, Qwen2 VL only
    pub min_pixels: usize,
    pub max_pixels: usize,
}

/// A tensor the tower reads, from the file or made at load time
enum Source {
    File(usize),
    Made { ty: u32, bytes: Vec<u8>, name: String },
}

/// Offset of every tower weight id, so a backend caching by id never mixes them with the text model's
pub const WID_BASE: u32 = 1 << 30;

/// The projector file as a weight source, patch kernels presented as matrices and the gemma projection transposed
pub struct Store {
    gguf: Gguf,
    sources: Vec<Source>,
    dims: Vec<Vec<usize>>,
    claimed: std::collections::HashSet<usize>,
}

impl Store {
    fn at(&self, w: Wid) -> usize {
        (w.0.wrapping_sub(WID_BASE)) as usize
    }

    fn add_file(&mut self, name: &str) -> Result<Wid> {
        let i = self.gguf.tensors.iter().position(|t| t.name == name).ok_or_else(|| anyhow!("projector file has no tensor {name}"))?;
        self.claimed.insert(i);
        self.sources.push(Source::File(i));
        self.dims.push(self.gguf.tensors[i].dims.clone());
        Ok(Wid(WID_BASE + self.sources.len() as u32 - 1))
    }

    fn file_opt(&mut self, name: &str) -> Result<Option<Wid>> {
        if self.gguf.tensor(name).is_none() {
            return Ok(None);
        }
        self.add_file(name).map(Some)
    }

    /// A file tensor viewed with other dims over the same bytes
    fn add_view(&mut self, name: &str, dims: Vec<usize>) -> Result<Wid> {
        let w = self.add_file(name)?;
        let n: usize = dims.iter().product();
        let at = self.at(w);
        let have: usize = self.dims[at].iter().product();
        if n != have {
            bail!("{name} has {have} elements, a view of {n} asked");
        }
        self.dims[at] = dims;
        Ok(w)
    }

    fn add_made(&mut self, name: &str, dims: Vec<usize>, data: &[f32]) -> Wid {
        let bytes = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.sources.push(Source::Made { ty: 0, bytes, name: name.to_string() });
        self.dims.push(dims);
        Wid(WID_BASE + self.sources.len() as u32 - 1)
    }
}

impl WeightSource for Store {
    fn bytes(&self, w: Wid) -> Result<&[u8]> {
        match self.sources.get(self.at(w)).ok_or_else(|| anyhow!("weight {} is not a tower weight", w.0))? {
            Source::File(i) => self.gguf.bytes(&self.gguf.tensors[*i]),
            Source::Made { bytes, .. } => Ok(bytes),
        }
    }
    fn ggml_type(&self, w: Wid) -> u32 {
        match &self.sources[self.at(w)] {
            Source::File(i) => self.gguf.tensors[*i].ggml_type,
            Source::Made { ty, .. } => *ty,
        }
    }
    fn dims(&self, w: Wid) -> &[usize] {
        &self.dims[self.at(w)]
    }
    fn name(&self, w: Wid) -> &str {
        match &self.sources[self.at(w)] {
            Source::File(i) => &self.gguf.tensors[*i].name,
            Source::Made { name, .. } => name,
        }
    }
    fn len(&self) -> usize {
        self.sources.len()
    }
}

struct Block {
    ln1: (Wid, Wid),
    q: (Wid, Wid),
    k: (Wid, Wid),
    v: (Wid, Wid),
    o: (Wid, Wid),
    ln2: (Wid, Wid),
    up: (Wid, Wid),
    down: (Wid, Wid),
}

/// The tower's weights by role
struct Weights {
    patch: Vec<Wid>,
    patch_bias: Option<Wid>,
    pos: Option<Wid>,
    blocks: Vec<Block>,
    post_ln: (Wid, Wid),
    mm: Vec<(Wid, Option<Wid>)>,
    soft_norm: Option<Wid>,
}

/// A preprocessed image, patch rows in the order the tower reads them with the grid they came from
#[derive(Clone, Debug)]
pub struct ImageInput {
    pub patches: Vec<f32>,
    pub patch_cols: usize,
    pub n_patches_x: usize,
    pub n_patches_y: usize,
    /// Rotary streams per patch, Qwen2 VL's row and column
    pub rope: Vec<[u32; 4]>,
    /// Patches per side that merge into one token
    pub merge: usize,
}

impl ImageInput {
    pub fn n_patches(&self) -> usize {
        self.n_patches_x * self.n_patches_y
    }

    /// Image tokens per row after the tower merges patches
    pub fn n_merged_x(&self) -> usize {
        self.n_patches_x / self.merge
    }

    pub fn n_merged_y(&self) -> usize {
        self.n_patches_y / self.merge
    }

    /// A stable key of the image for prefix matching, over every patch value
    pub fn key(&self) -> u32 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for v in &self.patches {
            h ^= v.to_bits() as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h ^= ((self.n_patches_x as u64) << 32) | self.n_patches_y as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
        (h >> 40) as u32 | 0x8000_0000
    }
}

/// Image embeddings in the text model's space with the grid of tokens they form
#[derive(Clone, Debug)]
pub struct Encoded {
    pub embeds: Vec<f32>,
    pub n_embd: usize,
    pub nx: usize,
    pub ny: usize,
}

impl Encoded {
    pub fn n_tokens(&self) -> usize {
        self.nx * self.ny
    }
}

/// The loaded vision tower
pub struct Clip {
    pub hparams: Hparams,
    store: Store,
    weights: Weights,
    graphs: Mutex<HashMap<(usize, usize), std::sync::Arc<Graph>>>,
    pub bytes: u64,
}

impl Clip {
    /// Opens a projector file and claims its tensors for the projector family it names
    pub fn load(path: impl AsRef<Path>) -> Result<Clip> {
        let gguf = Gguf::open(path.as_ref()).with_context(|| format!("open projector {}", path.as_ref().display()))?;
        let m = &gguf.meta;
        if m.arch() != Some("clip") {
            bail!("projector file has architecture {:?}, clip expected", m.arch());
        }
        let projector = match m.str("clip.projector_type") {
            Some("qwen2vl_merger") => Projector::Qwen2Vl,
            Some("gemma3") => Projector::Gemma3,
            other => bail!("projector type {other:?} is not qwen2vl_merger or gemma3"),
        };
        let u = |k: &str| m.usize(k).ok_or_else(|| anyhow!("projector file lacks {k}"));
        let f3 = |k: &str| -> Result<[f32; 3]> {
            let v = m.arr_f32(k).ok_or_else(|| anyhow!("projector file lacks {k}"))?;
            Ok([v.first().copied().unwrap_or(0.0), v.get(1).copied().unwrap_or(0.0), v.get(2).copied().unwrap_or(0.0)])
        };
        let patch_size = u("clip.vision.patch_size")?;
        let n_merge = match projector {
            Projector::Qwen2Vl => m.usize("clip.vision.spatial_merge_size").unwrap_or(2),
            Projector::Gemma3 => m.usize("clip.vision.projector.scale_factor").unwrap_or(4),
        };
        let patch_area = patch_size * patch_size * n_merge * n_merge;
        let hparams = Hparams {
            projector,
            image_size: u("clip.vision.image_size")?,
            patch_size,
            n_embd: u("clip.vision.embedding_length")?,
            n_ff: u("clip.vision.feed_forward_length")?,
            n_layer: u("clip.vision.block_count")?,
            n_head: u("clip.vision.attention.head_count")?,
            eps: m.f32("clip.vision.attention.layer_norm_epsilon").unwrap_or(1e-6),
            mean: f3("clip.vision.image_mean")?,
            std: f3("clip.vision.image_std")?,
            projection_dim: u("clip.vision.projection_dim")?,
            n_merge,
            min_pixels: m.usize("clip.vision.image_min_pixels").unwrap_or(8 * patch_area),
            max_pixels: m.usize("clip.vision.image_max_pixels").unwrap_or(4096 * patch_area),
        };
        let bytes = gguf.total_bytes();
        let mut store = Store { gguf, sources: Vec::new(), dims: Vec::new(), claimed: std::collections::HashSet::new() };
        let h = &hparams;
        let pcols = 3 * h.patch_size * h.patch_size;
        let mut patch = vec![store.add_view("v.patch_embd.weight", vec![pcols, h.n_embd])?];
        if store.gguf.tensor("v.patch_embd.weight.1").is_some() {
            patch.push(store.add_view("v.patch_embd.weight.1", vec![pcols, h.n_embd])?);
        }
        let patch_bias = store.file_opt("v.patch_embd.bias")?;
        let pos = store.file_opt("v.position_embd.weight")?;
        let pair = |st: &mut Store, w: &str, b: &str| -> Result<(Wid, Wid)> { Ok((st.add_file(w)?, st.add_file(b)?)) };
        let mut blocks = Vec::with_capacity(h.n_layer);
        for l in 0..h.n_layer {
            let p = |n: &str| format!("v.blk.{l}.{n}");
            let mut up = pair(&mut store, &p("ffn_up.weight"), &p("ffn_up.bias"))?;
            let mut down = pair(&mut store, &p("ffn_down.weight"), &p("ffn_down.bias"))?;
            if store.dims(up.0).first() != Some(&h.n_embd) {
                std::mem::swap(&mut up, &mut down);
            }
            if store.dims(up.0).first() != Some(&h.n_embd) || store.dims(down.0).get(1) != Some(&h.n_embd) {
                bail!("layer {l} feed forward tensors have dims {:?} and {:?}, neither reads {} inputs", store.dims(up.0), store.dims(down.0), h.n_embd);
            }
            blocks.push(Block {
                ln1: pair(&mut store, &p("ln1.weight"), &p("ln1.bias"))?,
                q: pair(&mut store, &p("attn_q.weight"), &p("attn_q.bias"))?,
                k: pair(&mut store, &p("attn_k.weight"), &p("attn_k.bias"))?,
                v: pair(&mut store, &p("attn_v.weight"), &p("attn_v.bias"))?,
                o: pair(&mut store, &p("attn_out.weight"), &p("attn_out.bias"))?,
                ln2: pair(&mut store, &p("ln2.weight"), &p("ln2.bias"))?,
                up,
                down,
            });
        }
        let post_ln = pair(&mut store, "v.post_ln.weight", "v.post_ln.bias")?;
        let (mm, soft_norm) = match projector {
            Projector::Qwen2Vl => (vec![(store.add_file("mm.0.weight")?, Some(store.add_file("mm.0.bias")?)), (store.add_file("mm.2.weight")?, Some(store.add_file("mm.2.bias")?))], None),
            Projector::Gemma3 => {
                let at = store.gguf.tensors.iter().position(|t| t.name == "mm.input_projection.weight").ok_or_else(|| anyhow!("projector file has no mm.input_projection.weight"))?;
                store.claimed.insert(at);
                let t = store.gguf.tensors[at].clone();
                if t.dims.len() != 2 || t.dims[1] != h.n_embd {
                    bail!("mm.input_projection.weight has dims {:?}, {} rows of the projection width expected", t.dims, h.n_embd);
                }
                let (n_out, n_in) = (t.dims[0], t.dims[1]);
                let q = kf_quant::by_id(t.ggml_type).ok_or_else(|| anyhow!("mm.input_projection.weight has unknown type {}", t.ggml_type))?;
                let bytes = store.gguf.bytes(&t)?;
                let rb = q.info().row_bytes(n_out);
                let mut rows = vec![0.0f32; n_in * n_out];
                let mut tmp = vec![0.0f32; n_out];
                for r in 0..n_in {
                    q.dequant_row(&bytes[r * rb..(r + 1) * rb], &mut tmp);
                    for (o, v) in tmp.iter().enumerate() {
                        rows[o * n_in + r] = *v;
                    }
                }
                let proj = store.add_made("mm.input_projection.weight.t", vec![n_in, n_out], &rows);
                let soft = store.add_file("mm.soft_emb_norm.weight")?;
                (vec![(proj, None)], Some(soft))
            }
        };
        let unclaimed: Vec<&str> = store.gguf.tensors.iter().enumerate().filter(|(i, _)| !store.claimed.contains(i)).map(|(_, t)| t.name.as_str()).collect();
        if !unclaimed.is_empty() {
            bail!("projector file carries {} tensors the tower does not use: {}", unclaimed.len(), unclaimed.join(", "));
        }
        Ok(Clip { hparams, store, weights: Weights { patch, patch_bias, pos, blocks, post_ln, mm, soft_norm }, graphs: Mutex::new(HashMap::new()), bytes })
    }

    pub fn store(&self) -> &dyn WeightSource {
        &self.store
    }

    /// Whether image tokens see each other through the text model's attention, the gemma rule
    pub fn non_causal(&self) -> bool {
        self.hparams.projector == Projector::Gemma3
    }

    /// Whether the text model places image tokens on a 2D rotary grid
    pub fn mrope(&self) -> bool {
        self.hparams.projector == Projector::Qwen2Vl
    }

    /// The text the prompt carries around an image, the tokens the reference adds before and after the embeddings
    pub fn markers(&self) -> (&'static str, &'static str) {
        match self.hparams.projector {
            Projector::Qwen2Vl => ("<|vision_start|>", "<|vision_end|>"),
            Projector::Gemma3 => ("<start_of_image>", "<end_of_image>"),
        }
    }

    /// Decodes and preprocesses image bytes into patch rows
    pub fn preprocess(&self, bytes: &[u8]) -> Result<ImageInput> {
        preprocess::prepare(&self.hparams, bytes)
    }

    fn graph(&self, img: &ImageInput) -> Result<std::sync::Arc<Graph>> {
        let key = (img.n_patches_x, img.n_patches_y);
        let mut graphs = self.graphs.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(g) = graphs.get(&key) {
            return Ok(g.clone());
        }
        let g = std::sync::Arc::new(self.build(img.n_patches_x, img.n_patches_y)?);
        graphs.insert(key, g.clone());
        Ok(g)
    }

    /// The tower as a graph for a patch grid, patch rows entering through the hidden input
    fn build(&self, nx: usize, ny: usize) -> Result<Graph> {
        let h = &self.hparams;
        let w = &self.weights;
        let n = nx * ny;
        let mut g = GraphBuilder::new();
        let _tokens = g.input("tokens");
        let positions = g.input("positions");
        let patches = g.input("hidden");
        let mut x: Option<Tid> = None;
        for pw in &w.patch {
            let e = g.push(Op::MatMul { weight: *pw, bias: None, scale: None }, &[patches]);
            x = Some(match x {
                Some(acc) => g.add(acc, e),
                None => e,
            });
        }
        let mut x = x.ok_or_else(|| anyhow!("no patch embedding"))?;
        if let Some(b) = w.patch_bias {
            x = g.push(Op::AddWeight { weight: b }, &[x]);
        }
        if let Some(p) = w.pos {
            let pe = g.push(Op::Gather { weight: p }, &[positions]);
            x = g.add(x, pe);
        }
        let head_dim = h.n_embd / h.n_head;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let rope = (h.projector == Projector::Qwen2Vl).then(|| RopeParams { kind: RopeKind::Vision, dims: head_dim, base: 10000.0, scale: 1.0, orig_ctx: 32768, attn_factor: 1.0, yarn: None, sections: [head_dim / 4, head_dim / 4, head_dim / 4, head_dim / 4], factors: None });
        let act = match h.projector {
            Projector::Qwen2Vl => Op::GeluQuick,
            Projector::Gemma3 => Op::Gelu,
        };
        for blk in &w.blocks {
            let res = x;
            let cur = g.push(Op::LayerNorm { weight: Some(blk.ln1.0), bias: Some(blk.ln1.1), eps: h.eps }, &[x]);
            let mut q = g.push(Op::MatMul { weight: blk.q.0, bias: Some(blk.q.1), scale: None }, &[cur]);
            let mut k = g.push(Op::MatMul { weight: blk.k.0, bias: Some(blk.k.1), scale: None }, &[cur]);
            let v = g.push(Op::MatMul { weight: blk.v.0, bias: Some(blk.v.1), scale: None }, &[cur]);
            if let Some(rp) = &rope {
                q = g.push(Op::Rope { params: rp.clone(), n_head: h.n_head, head_dim, offset: 0, inverse: false }, &[q, positions]);
                k = g.push(Op::Rope { params: rp.clone(), n_head: h.n_head, head_dim, offset: 0, inverse: false }, &[k, positions]);
            }
            let a = g.push(Op::AttentionFull { n_head: h.n_head, head_dim, scale }, &[q, k, v]);
            let o = g.push(Op::MatMul { weight: blk.o.0, bias: Some(blk.o.1), scale: None }, &[a]);
            x = g.add(res, o);
            let res = x;
            let cur = g.push(Op::LayerNorm { weight: Some(blk.ln2.0), bias: Some(blk.ln2.1), eps: h.eps }, &[x]);
            let up = g.push(Op::MatMul { weight: blk.up.0, bias: Some(blk.up.1), scale: None }, &[cur]);
            let up = g.push(act.clone(), &[up]);
            let down = g.push(Op::MatMul { weight: blk.down.0, bias: Some(blk.down.1), scale: None }, &[up]);
            x = g.add(res, down);
        }
        x = g.push(Op::LayerNorm { weight: Some(w.post_ln.0), bias: Some(w.post_ln.1), eps: h.eps }, &[x]);
        let out = match h.projector {
            Projector::Qwen2Vl => {
                let merged = g.reshape(x, n / 4, 4 * h.n_embd);
                let (w0, b0) = w.mm[0];
                let (w2, b2) = w.mm[1];
                let y = g.push(Op::MatMul { weight: w0, bias: b0, scale: None }, &[merged]);
                let y = g.push(Op::Gelu, &[y]);
                g.push(Op::MatMul { weight: w2, bias: b2, scale: None }, &[y])
            }
            Projector::Gemma3 => {
                if nx != ny {
                    bail!("gemma pools a square grid, {nx} by {ny} given");
                }
                let pooled = g.push(Op::PoolGrid { side: nx, kernel: h.n_merge }, &[x]);
                let normed = g.push(Op::RmsNorm { weight: w.soft_norm, bias: None, eps: h.eps }, &[pooled]);
                let (wp, bp) = w.mm[0];
                g.push(Op::MatMul { weight: wp, bias: bp, scale: None }, &[normed])
            }
        };
        Ok(g.finish(out))
    }

    /// The tower graph of an image cut after node k, every node when none, for probing intermediate tensors
    pub fn probe(&self, backend: &mut dyn Backend, img: &ImageInput, cut: Option<usize>) -> Result<(Vec<f32>, String)> {
        let full = self.graph(img)?;
        let n = img.n_patches();
        let mut g = (*full).clone();
        let name = match cut {
            Some(k) if k + 1 < g.nodes.len() => {
                g.nodes.truncate(k + 1);
                g.output = Some(g.nodes[k].out);
                format!("{:?}", g.nodes[k].op).split([' ', '(', '{']).next().unwrap_or("op").to_string()
            }
            _ => "output".to_string(),
        };
        let inputs = Inputs { tokens: vec![0; n], positions: (0..n as u32).collect(), seqs: Vec::new(), hidden: img.patches.clone(), hidden_cols: img.patch_cols, rope: img.rope.clone(), attend: Vec::new() };
        Ok((backend.run(&g, &inputs, &self.store)?, name))
    }

    /// Runs the tower on a backend and returns the image tokens' embeddings
    pub fn encode(&self, backend: &mut dyn Backend, img: &ImageInput) -> Result<Encoded> {
        let graph = self.graph(img)?;
        let n = img.n_patches();
        if img.patches.len() != n * img.patch_cols {
            bail!("image carries {} values for {n} patches of {}", img.patches.len(), img.patch_cols);
        }
        let inputs = Inputs { tokens: vec![0; n], positions: (0..n as u32).collect(), seqs: Vec::new(), hidden: img.patches.clone(), hidden_cols: img.patch_cols, rope: img.rope.clone(), attend: Vec::new() };
        let out = backend.run(&graph, &inputs, &self.store)?;
        let (nx, ny) = (img.n_merged_x(), img.n_merged_y());
        let n_tokens = nx * ny;
        if n_tokens == 0 || !out.len().is_multiple_of(n_tokens) {
            bail!("tower gave {} values for {n_tokens} image tokens", out.len());
        }
        Ok(Encoded { n_embd: out.len() / n_tokens, embeds: out, nx, ny })
    }
}

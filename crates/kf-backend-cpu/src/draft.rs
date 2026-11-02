//! The draft block on the host, the reference the device block is checked against, see kf_core::DraftConfig
use crate::train;
use anyhow::{Result, ensure};
use kf_core::{DRAFT_CAL_BLOCK, DRAFT_CAL_KEEP, DraftConfig, Objective, RingMeta, Scored, Soft, TrainStep, Tree, tree_of_block};
use kf_quant::ActKind;
use std::collections::VecDeque;

pub const FUSE: usize = 0;
pub const FC_G: usize = 1;
pub const FC_E: usize = 2;
pub const WQ: usize = 3;
pub const WK: usize = 4;
pub const WV: usize = 5;
pub const WO: usize = 6;
pub const N1: usize = 7;
pub const WG: usize = 8;
pub const WU: usize = 9;
pub const WD: usize = 10;
pub const N2: usize = 11;
pub const MASK: usize = 12;

/// The parameters the drafting pass multiplies through as int8 rows, every matrix of the block
pub const MATRICES: [usize; 10] = [FUSE, FC_G, FC_E, WQ, WK, WV, WO, WG, WU, WD];

/// Beam width, expansion pool, and CKA pool of the tap selection
pub const TAP_BEAM: usize = 10;
pub const TAP_EXPAND: usize = 8;

/// Deterministic uniform noise in a range from a seed
pub fn init_values(n: usize, scale: f32, seed: &mut u64) -> Vec<f32> {
    (0..n)
        .map(|_| {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 7;
            *seed ^= *seed << 17;
            ((*seed >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
        })
        .collect()
}

/// Initial parameters, the fusion passing the last tap through, norms at one, products scaled by their fan in
pub fn init_params(cfg: &DraftConfig) -> Vec<Vec<f32>> {
    let mut seed = cfg.seed.max(1);
    let d = cfg.d;
    cfg.param_shapes()
        .iter()
        .map(|(name, rows, cols)| match *name {
            "fuse" => {
                let mut w = vec![0.0; rows * cols];
                for c in 0..d {
                    w[((cfg.taps - 1) * d + c) * d + c] = 1.0;
                }
                w
            }
            n if n.starts_with('n') => vec![1.0; rows * cols],
            "mask" => init_values(rows * cols, 1.0 / (d as f32).sqrt(), &mut seed),
            _ => init_values(rows * cols, 1.0 / (*rows as f32).sqrt(), &mut seed),
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn mm(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize, ta: bool, tb: bool, beta: f32) {
    let (lda, ldb) = (if ta { m } else { k }, if tb { k } else { n });
    train::gemm(a, b, c, m, n, k, lda, ldb, n, ta, tb, 1.0, beta);
}

/// The block's matrices as the drafting pass reads them, transposed into q8_0 rows, with the norms and mask rows
pub struct Snapshot {
    rows: Vec<Vec<u8>>,
    shape: Vec<(usize, usize)>,
    pub n1: Vec<f32>,
    pub n2: Vec<f32>,
    pub mask: Vec<f32>,
}

impl Snapshot {
    /// The int8 rows of every matrix of params, each output column of the x times w product one row of its inputs
    pub fn of(cfg: &DraftConfig, params: &[Vec<f32>]) -> Self {
        let shapes = cfg.param_shapes();
        let mut rows = Vec::with_capacity(MATRICES.len());
        let mut shape = Vec::with_capacity(MATRICES.len());
        for which in MATRICES {
            let (_, k, n) = shapes[which];
            let p = &params[which];
            let rb = k / 32 * 34;
            let mut out = vec![0u8; n * rb];
            let mut col = vec![0.0f32; k];
            for r in 0..n {
                for j in 0..k {
                    col[j] = p[j * n + r];
                }
                kf_quant::quantize_row_q8_0(&col, &mut out[r * rb..(r + 1) * rb]);
            }
            rows.push(out);
            shape.push((k, n));
        }
        Self { rows, shape, n1: params[N1].clone(), n2: params[N2].clone(), mask: params[MASK].clone() }
    }

    /// y = x w for m rows of x through the int8 rows of matrix which, the rows of x rounded to int8 blocks first
    pub fn product(&self, which: usize, x: &[f32], m: usize) -> Vec<f32> {
        let slot = MATRICES.iter().position(|w| *w == which).expect("a matrix of the block");
        let (k, n) = self.shape[slot];
        let rb = k / 32 * 34;
        let act = kf_quant::quantize_activations(ActKind::Q8_0, x, m).expect("rows of whole int8 blocks");
        let q = kf_quant::by_name("q8_0").expect("q8_0");
        let mut y = vec![0.0f32; m * n];
        let mut tmp = vec![0.0f32; m];
        for r in 0..n {
            q.dot_i8(&self.rows[slot][r * rb..(r + 1) * rb], &act, &mut tmp);
            for row in 0..m {
                y[row * n + r] = tmp[row];
            }
        }
        y
    }
}

/// How a forward reads the block, through single precision parameters or through the int8 snapshot
pub enum Read<'a> {
    Exact(&'a [Vec<f32>]),
    Snap(&'a Snapshot),
}

impl Read<'_> {
    fn product(&self, which: usize, x: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        match self {
            Read::Exact(p) => {
                let mut y = vec![0.0; m * n];
                mm(x, &p[which], &mut y, m, n, k, false, false, 0.0);
                y
            }
            Read::Snap(s) => s.product(which, x, m),
        }
    }

    fn small(&self, which: usize) -> &[f32] {
        match self {
            Read::Exact(p) => &p[which],
            Read::Snap(s) => match which {
                N1 => &s.n1,
                N2 => &s.n2,
                _ => &s.mask,
            },
        }
    }
}

/// One learner of the ensemble, the block with its gradients, optimizer moments, last directions, and rate
pub struct DraftBlock {
    pub cfg: DraftConfig,
    pub params: Vec<Vec<f32>>,
    pub grads: Vec<Vec<f32>>,
    m: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    u: Vec<Vec<f32>>,
    pub step: i32,
    pub lr: f32,
}

/// The first half of the layer for m rows, up to the queries, keys, and values
pub struct In {
    pub x0: Vec<f32>,
    pub h1: Vec<f32>,
    pub inv1: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
}

/// The second half of the layer for m rows, from the attention output to the block's output rows
pub struct Out {
    pub x1: Vec<f32>,
    pub h2: Vec<f32>,
    pub inv2: Vec<f32>,
    pub g: Vec<f32>,
    pub u: Vec<f32>,
    pub mm: Vec<f32>,
    pub out: Vec<f32>,
}

impl DraftBlock {
    pub fn new(cfg: DraftConfig, lr: f32) -> Result<Self> {
        Self::with_params(cfg, init_params(&cfg), lr)
    }

    /// A block over given parameters in the configuration's order
    pub fn with_params(cfg: DraftConfig, params: Vec<Vec<f32>>, lr: f32) -> Result<Self> {
        cfg.check()?;
        let shapes = cfg.param_shapes();
        ensure!(params.len() == shapes.len() && params.iter().zip(&shapes).all(|(p, (_, r, c))| p.len() == r * c), "parameters of the wrong shapes for the block");
        let zeros: Vec<Vec<f32>> = params.iter().map(|p| vec![0.0; p.len()]).collect();
        Ok(Self { cfg, grads: zeros.clone(), m: zeros.clone(), v: zeros.clone(), u: zeros, params, step: 0, lr })
    }

    pub fn scale(&self) -> f32 {
        1.0 / (self.cfg.head_dim() as f32).sqrt()
    }

    /// g = taps fuse for m rows of the tap rows side by side
    pub fn fuse(read: &Read<'_>, cfg: &DraftConfig, taps: &[f32], m: usize) -> Vec<f32> {
        read.product(FUSE, taps, m, cfg.taps * cfg.d, cfg.d)
    }

    /// The layer up to its queries, keys, and values from the feature rows g and the embeddings e
    pub fn layer_in(read: &Read<'_>, cfg: &DraftConfig, g: &[f32], e: &[f32], m: usize) -> In {
        let d = cfg.d;
        let mut x0 = read.product(FC_G, g, m, d, d);
        let xe = read.product(FC_E, e, m, d, d);
        x0.iter_mut().zip(&xe).for_each(|(a, b)| *a += b);
        let mut h1 = vec![0.0; m * d];
        let inv1 = train::rms_fwd(&x0, read.small(N1), &mut h1, d, cfg.eps);
        let q = read.product(WQ, &h1, m, d, d);
        let k = read.product(WK, &h1, m, d, d);
        let v = read.product(WV, &h1, m, d, d);
        In { x0, h1, inv1, q, k, v }
    }

    /// The layer from the attention output a to the block's output rows
    pub fn layer_out(read: &Read<'_>, cfg: &DraftConfig, inp: &In, a: &[f32], m: usize) -> Out {
        let (d, ff) = (cfg.d, cfg.ff);
        let mut x1 = inp.x0.clone();
        let ao = read.product(WO, a, m, d, d);
        x1.iter_mut().zip(&ao).for_each(|(a, b)| *a += b);
        let mut h2 = vec![0.0; m * d];
        let inv2 = train::rms_fwd(&x1, read.small(N2), &mut h2, d, cfg.eps);
        let g = read.product(WG, &h2, m, d, ff);
        let u = read.product(WU, &h2, m, d, ff);
        let mut mmv = vec![0.0; m * ff];
        train::silu_gate_fwd(&g, &u, &mut mmv);
        let mut out = x1.clone();
        let dn = read.product(WD, &mmv, m, ff, d);
        out.iter_mut().zip(&dn).for_each(|(a, b)| *a += b);
        Out { x1, h2, inv2, g, u, mm: mmv, out }
    }

    /// Gradients of the second half given dout, the parameter gradients added, returning dx1 and da
    pub fn layer_out_bwd(&mut self, out: &Out, a: &[f32], dout: &[f32], m: usize) -> (Vec<f32>, Vec<f32>) {
        let (d, ff) = (self.cfg.d, self.cfg.ff);
        let p = &self.params;
        let gr = &mut self.grads;
        mm(&out.mm, dout, &mut gr[WD], ff, d, m, true, false, 1.0);
        let mut dm = vec![0.0; m * ff];
        mm(dout, &p[WD], &mut dm, m, ff, d, false, true, 0.0);
        let (mut dg, mut du) = (vec![0.0; m * ff], vec![0.0; m * ff]);
        train::silu_gate_bwd(&out.g, &out.u, &dm, &mut dg, &mut du);
        mm(&out.h2, &dg, &mut gr[WG], d, ff, m, true, false, 1.0);
        mm(&out.h2, &du, &mut gr[WU], d, ff, m, true, false, 1.0);
        let mut dh2 = vec![0.0; m * d];
        mm(&dg, &p[WG], &mut dh2, m, d, ff, false, true, 0.0);
        mm(&du, &p[WU], &mut dh2, m, d, ff, false, true, 1.0);
        let mut dx1 = dout.to_vec();
        let mut dx1n = vec![0.0; m * d];
        train::rms_bwd(&out.x1, &p[N2], &out.inv2, &dh2, &mut dx1n, &mut gr[N2], d);
        dx1.iter_mut().zip(&dx1n).for_each(|(a, b)| *a += b);
        mm(a, &dx1, &mut gr[WO], d, d, m, true, false, 1.0);
        let mut da = vec![0.0; m * d];
        mm(&dx1, &p[WO], &mut da, m, d, d, false, true, 0.0);
        (dx1, da)
    }

    /// Gradients of the first half given dx1 and the attention's dq dk dv, the parameter gradients added, returning
    /// dg and de
    #[allow(clippy::too_many_arguments)]
    pub fn layer_in_bwd(&mut self, g: &[f32], e: &[f32], inp: &In, dx1: &[f32], dq: &[f32], dk: &[f32], dv: &[f32], m: usize) -> (Vec<f32>, Vec<f32>) {
        let d = self.cfg.d;
        let p = &self.params;
        let gr = &mut self.grads;
        mm(&inp.h1, dq, &mut gr[WQ], d, d, m, true, false, 1.0);
        mm(&inp.h1, dk, &mut gr[WK], d, d, m, true, false, 1.0);
        mm(&inp.h1, dv, &mut gr[WV], d, d, m, true, false, 1.0);
        let mut dh1 = vec![0.0; m * d];
        mm(dq, &p[WQ], &mut dh1, m, d, d, false, true, 0.0);
        mm(dk, &p[WK], &mut dh1, m, d, d, false, true, 1.0);
        mm(dv, &p[WV], &mut dh1, m, d, d, false, true, 1.0);
        let mut dx0 = dx1.to_vec();
        let mut dx0n = vec![0.0; m * d];
        train::rms_bwd(&inp.x0, &p[N1], &inp.inv1, &dh1, &mut dx0n, &mut gr[N1], d);
        dx0.iter_mut().zip(&dx0n).for_each(|(a, b)| *a += b);
        mm(g, &dx0, &mut gr[FC_G], d, d, m, true, false, 1.0);
        mm(e, &dx0, &mut gr[FC_E], d, d, m, true, false, 1.0);
        let mut dg = vec![0.0; m * d];
        mm(&dx0, &p[FC_G], &mut dg, m, d, d, false, true, 0.0);
        let mut de = vec![0.0; m * d];
        mm(&dx0, &p[FC_E], &mut de, m, d, d, false, true, 0.0);
        (dg, de)
    }

    /// Gradient of the fusion given dg
    pub fn fuse_bwd(&mut self, taps: &[f32], dg: &[f32], m: usize) {
        let (d, td) = (self.cfg.d, self.cfg.taps * self.cfg.d);
        mm(taps, dg, &mut self.grads[FUSE], td, d, m, true, false, 1.0);
    }

    /// One optimistic AdamW step over every parameter, the gradients cleared
    pub fn adam(&mut self) {
        self.step += 1;
        for i in 0..self.params.len() {
            train::adamw_opt(&mut self.params[i], &self.grads[i], &mut self.m[i], &mut self.v[i], &mut self.u[i], self.lr, 0.9, 0.999, 1e-8, self.cfg.weight_decay, self.step);
            self.grads[i].iter_mut().for_each(|g| *g = 0.0);
        }
    }

    /// Replaces the parameters, the last directions starting over
    pub fn set_params(&mut self, params: Vec<Vec<f32>>) -> Result<()> {
        let shapes = self.cfg.param_shapes();
        ensure!(params.len() == shapes.len() && params.iter().zip(&shapes).all(|(p, (_, r, c))| p.len() == r * c), "parameters of the wrong shapes for the block");
        self.params = params;
        self.u.iter_mut().for_each(|u| u.iter_mut().for_each(|v| *v = 0.0));
        Ok(())
    }
}

/// The model's head over the draft vocabulary, its rows and the final norm the block's outputs pass through
pub struct Vocab {
    pub ids: Vec<u32>,
    pub rows: Vec<f32>,
    pub norm: Vec<f32>,
    pub eps: f32,
}

impl Vocab {
    pub fn new(ids: Vec<u32>, rows: Vec<f32>, norm: Vec<f32>, eps: f32) -> Result<Self> {
        let d = norm.len();
        ensure!(!ids.is_empty() && rows.len() == ids.len() * d && ids.windows(2).all(|w| w[0] < w[1]), "a draft vocabulary of {} rising ids with {} head values over width {d}", ids.len(), rows.len());
        Ok(Self { ids, rows, norm, eps })
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Column of a token in the draft vocabulary
    pub fn index(&self, token: u32) -> Option<usize> {
        self.ids.binary_search(&token).ok()
    }

    /// Logits of m output rows through the final norm and the head rows, with the normed rows and their inverse rms
    pub fn logits(&self, out: &[f32], m: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let (d, v) = (self.norm.len(), self.len());
        let mut n = vec![0.0; m * d];
        let inv = train::rms_fwd(out, &self.norm, &mut n, d, self.eps);
        let mut z = vec![0.0; m * v];
        mm(&n, &self.rows, &mut z, m, v, d, false, true, 0.0);
        (z, n, inv)
    }

    /// dout of the output rows given dz over the logits
    pub fn logits_bwd(&self, out: &[f32], inv: &[f32], dz: &[f32], m: usize) -> Vec<f32> {
        let (d, v) = (self.norm.len(), self.len());
        let mut dn = vec![0.0; m * d];
        mm(dz, &self.rows, &mut dn, m, d, v, false, false, 0.0);
        let mut dout = vec![0.0; m * d];
        let mut unused = vec![0.0; d];
        train::rms_bwd(out, &self.norm, inv, &dn, &mut dout, &mut unused, d);
        dout
    }

    /// The soft target of a row as draft vocabulary columns with their probabilities, none when no candidate is in the vocabulary
    pub fn target(&self, soft: &Soft) -> Option<Vec<(usize, f32)>> {
        let probs = soft.probs();
        let t: Vec<(usize, f32)> = soft.ids.iter().zip(&probs).filter_map(|(id, p)| self.index(*id).map(|c| (c, *p))).collect();
        let mass: f32 = t.iter().map(|(_, p)| *p).sum();
        (mass > 0.0).then(|| t.into_iter().map(|(c, p)| (c, p / mass)).collect())
    }

    /// The k largest logits of a row as columns with their probabilities, best first
    pub fn top(&self, z: &[f32], k: usize) -> Vec<(usize, f32)> {
        let m = z.iter().fold(f32::NEG_INFINITY, |a, b| a.max(*b));
        let lse = m + z.iter().map(|v| (v - m).exp()).sum::<f32>().ln();
        let mut order: Vec<usize> = (0..z.len()).collect();
        order.sort_by(|a, b| z[*b].total_cmp(&z[*a]).then(a.cmp(b)));
        order.iter().take(k).map(|c| (*c, (z[*c] - lse).exp())).collect()
    }
}

/// One chunk of a training step, the tap rows of t consecutive rows and, per row, the main query group's first
/// token with its targets and realized tokens along the model's own path, and the branch group's first token with
/// its targets along a rejected branch, none where the branch ends
pub struct Chunk {
    pub taps: Vec<f32>,
    pub main_tokens: Vec<u32>,
    pub main_targets: Vec<Vec<Option<Soft>>>,
    pub main_truth: Vec<Vec<Option<u32>>>,
    pub branch_tokens: Vec<Option<u32>>,
    pub branch_targets: Vec<Vec<Option<Soft>>>,
}

/// The targets of one step's rows as draft columns with their probabilities, none where a row has no target
pub type StepTargets = Vec<Option<Vec<(usize, f32)>>>;

/// Both groups' cross entropies per step and row
type GroupCe = (Vec<Vec<f32>>, Vec<Vec<f32>>);

/// The block's candidates at every drafted position, best first
pub type Tops = Vec<Vec<(u32, f32)>>;

/// The forward of one query group through a block, kept for the backward pass
struct GroupFwd {
    es: Vec<Vec<f32>>,
    ins: Vec<In>,
    outs: Vec<Out>,
    atts: Vec<Vec<f32>>,
    probs: Vec<Vec<Vec<f32>>>,
    zs: Vec<Vec<f32>>,
    invs: Vec<Vec<f32>>,
}

/// The block ensemble at work on the host with the vocabulary and embedding table it reads
pub struct CpuDraft {
    pub cfg: DraftConfig,
    pub learners: Vec<DraftBlock>,
    /// Running loss of every learner the hedge weighs
    pub cum: Vec<f64>,
    pub weights: Vec<f32>,
    /// The reference the selective distillation ranks against, a moving average of the mixture
    pub reference: Vec<Vec<f32>>,
    /// The hedge's mixture of the learners, what the block drafts with
    pub mixture: Vec<Vec<f32>>,
    pub snapshot: Snapshot,
    pub vocab: Vocab,
    embd: Vec<f32>,
    eos: Option<u32>,
}

/// A row of an accepted path the drafter runs from, its token, position, and tap rows side by side
#[derive(Clone, Debug, Default)]
pub struct DraftNode {
    pub token: u32,
    pub pos: u32,
    pub taps: Vec<f32>,
}

/// A slot's step zero keys and values by position, the history every draft of the slot attends over
#[derive(Clone)]
pub struct SlotDraft {
    win: usize,
    pub rows: VecDeque<(u32, Vec<f32>, Vec<f32>)>,
    pub next_pos: Option<u32>,
}

impl SlotDraft {
    /// A history keeping the last win positions
    pub fn new(win: usize) -> Self {
        Self { win, rows: VecDeque::new(), next_pos: None }
    }

    /// Adds a position's key and value, the history starting over when the position does not follow
    pub fn push(&mut self, pos: u32, k: Vec<f32>, v: Vec<f32>) {
        if self.next_pos != Some(pos) {
            self.rows.clear();
        }
        self.rows.push_back((pos, k, v));
        while self.rows.len() > self.win {
            self.rows.pop_front();
        }
        self.next_pos = Some(pos + 1);
    }

    pub fn clear(&mut self) {
        self.rows.clear();
        self.next_pos = None;
    }
}

/// Key rows a draft attends over, the slot's history first, then every drafted row in order
struct Keys {
    hist: usize,
    rows: Vec<(u32, Vec<f32>, Vec<f32>)>,
}

/// What one training step found before the update, per learner
struct Learned {
    loss: f32,
    ce: f32,
    hit: f32,
}

impl CpuDraft {
    pub fn new(cfg: DraftConfig, vocab: Vocab, embd: Vec<f32>, eos: Option<u32>) -> Result<Self> {
        cfg.check()?;
        ensure!(embd.len().is_multiple_of(cfg.d) && vocab.norm.len() == cfg.d, "an embedding table of {} values and a norm of {} for a block of {}", embd.len(), vocab.norm.len(), cfg.d);
        let learners: Vec<DraftBlock> = (0..cfg.learners).map(|i| DraftBlock::new(cfg, cfg.learner_lr(i))).collect::<Result<_>>()?;
        let mixture = learners[0].params.clone();
        let snapshot = Snapshot::of(&cfg, &mixture);
        Ok(Self { cfg, cum: vec![0.0; cfg.learners], weights: vec![1.0 / cfg.learners as f32; cfg.learners], reference: mixture.clone(), mixture, snapshot, learners, vocab, embd, eos })
    }

    fn embed(&self, tokens: &[u32]) -> Vec<f32> {
        let d = self.cfg.d;
        tokens.iter().flat_map(|t| self.embd[*t as usize * d..(*t as usize + 1) * d].iter().copied()).collect()
    }

    /// The embedding rows of one step of a group, the first tokens at step zero, rows without one zero, the mask
    /// row of the position after
    fn step_rows(&self, read: &Read<'_>, first: &[Option<u32>], step: usize) -> Vec<f32> {
        let d = self.cfg.d;
        if step == 0 {
            return first.iter().flat_map(|t| match t {
                Some(t) => self.embd[*t as usize * d..(*t as usize + 1) * d].to_vec(),
                None => vec![0.0; d],
            }).collect();
        }
        let mask = &read.small(MASK)[(step - 1) * d..step * d];
        first.iter().flat_map(|_| mask.iter().copied()).collect()
    }

    /// The forward of one query group over b chunks of t rows, every step's rows attend to the window keys of the
    /// main group at or before them, strictly before for the branch group, and to their own chain of steps
    #[allow(clippy::too_many_arguments)]
    fn forward_group(&self, read: &Read<'_>, params: &[Vec<f32>], g0: &[f32], first: &[Option<u32>], b: usize, t: usize, window: Option<&In>, branch: bool) -> GroupFwd {
        let cfg = self.cfg;
        let (d, h, hd, w, steps) = (cfg.d, cfg.heads, cfg.head_dim(), cfg.window, cfg.depth);
        let n = b * t;
        let scale = 1.0 / (hd as f32).sqrt();
        let exact = Read::Exact(params);
        let _ = read;
        let mut out = GroupFwd { es: Vec::new(), ins: Vec::new(), outs: Vec::new(), atts: Vec::new(), probs: Vec::new(), zs: Vec::new(), invs: Vec::new() };
        let chain_first = if branch { 0 } else { 1 };
        for s in 0..steps {
            let e = self.step_rows(&exact, first, s);
            let inp = DraftBlock::layer_in(&exact, &cfg, g0, &e, n);
            let win: &In = match window {
                Some(wn) => wn,
                None => &inp,
            };
            let win = if branch { win } else if s == 0 { &inp } else { &out.ins[0] };
            let nchain = s + 1 - chain_first;
            let nk = t * (1 + nchain);
            let mut a = vec![0.0; n * d];
            let mut pr = Vec::with_capacity(b);
            for c in 0..b {
                let (mut keys, mut vals) = (Vec::with_capacity(nk * d), Vec::with_capacity(nk * d));
                keys.extend_from_slice(&win.k[c * t * d..(c + 1) * t * d]);
                vals.extend_from_slice(&win.v[c * t * d..(c + 1) * t * d]);
                for m in chain_first..=s {
                    let src: &In = if m == s { &inp } else { &out.ins[m] };
                    keys.extend_from_slice(&src.k[c * t * d..(c + 1) * t * d]);
                    vals.extend_from_slice(&src.v[c * t * d..(c + 1) * t * d]);
                }
                let mut mask = vec![false; t * nk];
                for i in 0..t {
                    for j in 0..t {
                        mask[i * nk + j] = (if branch { j < i } else { j <= i }) && i - j < w;
                    }
                    for m in 0..nchain {
                        mask[i * nk + t + m * t + i] = true;
                    }
                }
                let p = train::masked_attn_fwd(&inp.q[c * t * d..(c + 1) * t * d], &keys, &vals, &mask, &mut a[c * t * d..(c + 1) * t * d], t, nk, h, hd, scale);
                pr.push(p);
            }
            let o = DraftBlock::layer_out(&exact, &cfg, &inp, &a, n);
            let (z, _, inv) = self.vocab.logits(&o.out, n);
            out.es.push(e);
            out.ins.push(inp);
            out.outs.push(o);
            out.atts.push(a);
            out.probs.push(pr);
            out.zs.push(z);
            out.invs.push(inv);
        }
        out
    }

    /// Cross entropy of every row of every step of a group against its targets, none where a row has no target
    fn ce_values(&self, fwd: &GroupFwd, targets: &[StepTargets]) -> Vec<Vec<f32>> {
        let v = self.vocab.len();
        let n = targets[0].len();
        (0..self.cfg.depth)
            .map(|s| {
                let mut dz = vec![0.0; n * v];
                let mut losses = vec![0.0; n];
                train::soft_ce_rows(&fwd.zs[s], v, &targets[s], &vec![0.0; n], &mut dz, &mut losses);
                losses
            })
            .collect()
    }

    /// The backward of one group given every step's gradient over the output rows, the window key gradients of
    /// the branch group added into the main group's step zero accumulators, returning dg summed over the steps
    #[allow(clippy::too_many_arguments)]
    fn backward_group(&mut self, li: usize, fwd: &GroupFwd, g0: &[f32], douts: Vec<Vec<f32>>, b: usize, t: usize, window: Option<&In>, branch: bool, dk_win: &mut [f32], dv_win: &mut [f32]) -> Vec<f32> {
        let cfg = self.cfg;
        let (d, h, hd, steps) = (cfg.d, cfg.heads, cfg.head_dim(), cfg.depth);
        let n = b * t;
        let scale = 1.0 / (hd as f32).sqrt();
        let chain_first = if branch { 0 } else { 1 };
        let mut dg0 = vec![0.0; n * d];
        let mut dks: Vec<Vec<f32>> = (0..steps).map(|_| vec![0.0; n * d]).collect();
        let mut dvs: Vec<Vec<f32>> = (0..steps).map(|_| vec![0.0; n * d]).collect();
        for s in (0..steps).rev() {
            let learner = &mut self.learners[li];
            let (dx1, da) = learner.layer_out_bwd(&fwd.outs[s], &fwd.atts[s], &douts[s], n);
            let win: &In = match window {
                Some(wn) => wn,
                None => &fwd.ins[0],
            };
            let nchain = s + 1 - chain_first;
            let nk = t * (1 + nchain);
            let mut dq = vec![0.0; n * d];
            for c in 0..b {
                let (mut keys, mut vals) = (Vec::with_capacity(nk * d), Vec::with_capacity(nk * d));
                keys.extend_from_slice(&win.k[c * t * d..(c + 1) * t * d]);
                vals.extend_from_slice(&win.v[c * t * d..(c + 1) * t * d]);
                for m in chain_first..=s {
                    keys.extend_from_slice(&fwd.ins[m].k[c * t * d..(c + 1) * t * d]);
                    vals.extend_from_slice(&fwd.ins[m].v[c * t * d..(c + 1) * t * d]);
                }
                let (mut dkk, mut dvv) = (vec![0.0; nk * d], vec![0.0; nk * d]);
                train::masked_attn_bwd(&fwd.ins[s].q[c * t * d..(c + 1) * t * d], &keys, &vals, &fwd.probs[s][c], &da[c * t * d..(c + 1) * t * d], &mut dq[c * t * d..(c + 1) * t * d], &mut dkk, &mut dvv, t, nk, h, hd, scale);
                let range = c * t * d..(c + 1) * t * d;
                if branch {
                    dk_win[range.clone()].iter_mut().zip(&dkk[..t * d]).for_each(|(x, y)| *x += y);
                    dv_win[range.clone()].iter_mut().zip(&dvv[..t * d]).for_each(|(x, y)| *x += y);
                } else {
                    dks[0][range.clone()].iter_mut().zip(&dkk[..t * d]).for_each(|(x, y)| *x += y);
                    dvs[0][range.clone()].iter_mut().zip(&dvv[..t * d]).for_each(|(x, y)| *x += y);
                }
                for (ci, m) in (chain_first..=s).enumerate() {
                    let at = (ci + 1) * t * d;
                    dks[m][range.clone()].iter_mut().zip(&dkk[at..at + t * d]).for_each(|(x, y)| *x += y);
                    dvs[m][range.clone()].iter_mut().zip(&dvv[at..at + t * d]).for_each(|(x, y)| *x += y);
                }
            }
            if !branch && s == 0 {
                dks[0].iter_mut().zip(dk_win.iter()).for_each(|(x, y)| *x += y);
                dvs[0].iter_mut().zip(dv_win.iter()).for_each(|(x, y)| *x += y);
            }
            let (dg, de) = self.learners[li].layer_in_bwd(g0, &fwd.es[s], &fwd.ins[s], &dx1, &dq, &dks[s], &dvs[s], n);
            dg0.iter_mut().zip(&dg).for_each(|(a, b)| *a += b);
            if s > 0 {
                let mask_grad = &mut self.learners[li].grads[MASK][(s - 1) * d..s * d];
                for r in 0..n {
                    for c in 0..d {
                        mask_grad[c] += de[r * d + c];
                    }
                }
            }
        }
        dg0
    }

    /// The loss and gradients of one learner over the chunks, the gradients left in it
    fn learn(&mut self, li: usize, chunks: &[Chunk], t: usize, ref_ce: Option<&GroupCe>) -> Result<Learned> {
        let cfg = self.cfg;
        let (d, steps) = (cfg.d, cfg.depth);
        let b = chunks.len();
        let n = b * t;
        let v = self.vocab.len();
        let taps: Vec<f32> = chunks.iter().flat_map(|c| c.taps.iter().copied()).collect();
        let params = self.learners[li].params.clone();
        let exact = Read::Exact(&params);
        let g0 = DraftBlock::fuse(&exact, &cfg, &taps, n);
        let main_first: Vec<Option<u32>> = chunks.iter().flat_map(|c| c.main_tokens.iter().map(|t| Some(*t))).collect();
        let branch_first: Vec<Option<u32>> = chunks.iter().flat_map(|c| c.branch_tokens.iter().copied()).collect();
        let main = self.forward_group(&exact, &params, &g0, &main_first, b, t, None, false);
        let branch = self.forward_group(&exact, &params, &g0, &branch_first, b, t, Some(&main.ins[0]), true);
        let main_targets: Vec<StepTargets> = (0..steps).map(|s| chunks.iter().flat_map(|c| c.main_targets.iter().map(|row| row[s].as_ref().and_then(|soft| self.vocab.target(soft)))).collect()).collect();
        let branch_targets: Vec<StepTargets> = (0..steps).map(|s| chunks.iter().flat_map(|c| c.branch_targets.iter().map(|row| row[s].as_ref().and_then(|soft| self.vocab.target(soft)))).collect()).collect();
        let ce_main = self.ce_values(&main, &main_targets);
        let ce_branch = self.ce_values(&branch, &branch_targets);
        // The kept share of the targets, every target when the share is whole, else the widest gaps over the reference
        let mut gaps: Vec<f32> = Vec::new();
        let mut present = 0usize;
        for s in 0..steps {
            for r in 0..n {
                if main_targets[s][r].is_some() {
                    present += 1;
                    if let Some((rm, _)) = ref_ce {
                        gaps.push(ce_main[s][r] - rm[s][r]);
                    }
                }
                if branch_targets[s][r].is_some() {
                    present += 1;
                    if let Some((_, rb)) = ref_ce {
                        gaps.push(ce_branch[s][r] - rb[s][r]);
                    }
                }
            }
        }
        ensure!(present > 0, "no row of the batch has a target in the draft vocabulary");
        let threshold = match ref_ce {
            Some(_) => {
                let keep = ((present as f32 * cfg.keep_frac).ceil() as usize).clamp(1, present);
                let mut sorted = gaps.clone();
                sorted.sort_by(|a, b| b.total_cmp(a));
                Some(sorted[keep - 1])
            }
            None => None,
        };
        let kept_of = |ce: f32, rc: Option<f32>| -> bool {
            match (threshold, rc) {
                (Some(th), Some(rc)) => ce - rc >= th,
                _ => true,
            }
        };
        let mut w_main = vec![vec![0.0f32; n]; steps];
        let mut w_branch = vec![vec![0.0f32; n]; steps];
        let mut kept = 0usize;
        for s in 0..steps {
            for r in 0..n {
                if main_targets[s][r].is_some() && kept_of(ce_main[s][r], ref_ce.map(|(rm, _)| rm[s][r])) {
                    w_main[s][r] = 1.0;
                    kept += 1;
                }
                if branch_targets[s][r].is_some() && kept_of(ce_branch[s][r], ref_ce.map(|(_, rb)| rb[s][r])) {
                    w_branch[s][r] = 1.0;
                    kept += 1;
                }
            }
        }
        let inv = 1.0 / kept.max(1) as f32;
        w_main.iter_mut().chain(w_branch.iter_mut()).for_each(|ws| ws.iter_mut().for_each(|w| *w *= inv));
        // The objective's gradient over the main group's logits and the cross entropy's over the branch group's
        let mut dz_main: Vec<Vec<f32>> = (0..steps).map(|_| vec![0.0; n * v]).collect();
        let mut dz_branch: Vec<Vec<f32>> = (0..steps).map(|_| vec![0.0; n * v]).collect();
        let mut losses = vec![0.0f32; n];
        let mut loss = 0.0f64;
        for s in 0..steps {
            loss += train::soft_ce_rows(&branch.zs[s], v, &branch_targets[s], &w_branch[s], &mut dz_branch[s], &mut losses);
        }
        let eos = self.eos.and_then(|e| self.vocab.index(e));
        match cfg.objective {
            Objective::Ce => {
                for s in 0..steps {
                    loss += train::soft_ce_rows(&main.zs[s], v, &main_targets[s], &w_main[s], &mut dz_main[s], &mut losses);
                }
            }
            Objective::Edr => {
                for c in 0..b {
                    let mut a = vec![vec![0.0f32; steps]; t];
                    let mut cost = vec![vec![0.0f32; steps]; t];
                    for i in 0..t {
                        for s in 0..steps {
                            let r = c * t + i;
                            if let Some(tg) = &main_targets[s][r] {
                                let (aa, cc) = train::edr_terms(&main.zs[s][r * v..(r + 1) * v], tg, eos);
                                a[i][s] = aa;
                                cost[i][s] = cc;
                            }
                        }
                    }
                    let (omega, dv, rounds) = train::edr_recursion(&a, &cost, t, steps);
                    loss += f64::from(rounds) / b as f64;
                    for i in 0..t {
                        for s in 0..steps {
                            let r = c * t + i;
                            if let Some(tg) = &main_targets[s][r] {
                                let mut dz = vec![0.0f32; v];
                                train::edr_grad(&main.zs[s][r * v..(r + 1) * v], tg, eos, omega[i][s], dv[i][s], &mut dz);
                                let w = w_main[s][r] * kept as f32 / b as f32;
                                dz_main[s][r * v..(r + 1) * v].iter_mut().zip(&dz).for_each(|(x, y)| *x += w * y);
                            }
                        }
                    }
                }
            }
            Objective::Bv => {
                for r in 0..n {
                    let (c, i) = (r / t, r % t);
                    let zs: Vec<&[f32]> = (0..steps).map(|s| &main.zs[s][r * v..(r + 1) * v]).collect();
                    let targets: Vec<Option<Vec<(usize, f32)>>> = (0..steps).map(|s| main_targets[s][r].clone()).collect();
                    let truth: Vec<Option<usize>> = (0..steps).map(|s| chunks[c].main_truth[i][s].and_then(|y| self.vocab.index(y))).collect();
                    let w = w_main[0][r] * kept as f32 / n as f32;
                    let mut dzs: Vec<&mut [f32]> = Vec::with_capacity(steps);
                    let mut rest: &mut [Vec<f32>] = &mut dz_main;
                    for _ in 0..steps {
                        let (head, tail) = rest.split_at_mut(1);
                        dzs.push(&mut head[0][r * v..(r + 1) * v]);
                        rest = tail;
                    }
                    loss += f64::from(train::bv_row(&zs, &targets, &truth, w, &mut dzs)) / n as f64;
                }
            }
        }
        let (mut ce_sum, mut ce_n) = (0.0f64, 0usize);
        for s in 0..steps {
            for r in 0..n {
                if main_targets[s][r].is_some() {
                    ce_sum += f64::from(ce_main[s][r]);
                    ce_n += 1;
                }
            }
        }
        let fan = cfg.fanout[0];
        let mut hits = 0usize;
        for (c, chunk) in chunks.iter().enumerate() {
            for i in 0..t {
                let r = c * t + i;
                let top = self.vocab.top(&main.zs[0][r * v..(r + 1) * v], fan);
                hits += chunk.main_truth[i][0].is_some_and(|y| top.iter().any(|(col, _)| self.vocab.ids[*col] == y)) as usize;
            }
        }
        let douts_of = |fwd: &GroupFwd, dz: &[Vec<f32>]| -> Vec<Vec<f32>> { (0..steps).map(|s| self.vocab.logits_bwd(&fwd.outs[s].out, &fwd.invs[s], &dz[s], n)).collect() };
        let douts_main = douts_of(&main, &dz_main);
        let douts_branch = douts_of(&branch, &dz_branch);
        let (mut dk_win, mut dv_win) = (vec![0.0f32; n * d], vec![0.0f32; n * d]);
        let dg_branch = self.backward_group(li, &branch, &g0, douts_branch, b, t, Some(&main.ins[0]), true, &mut dk_win, &mut dv_win);
        let mut dg = self.backward_group(li, &main, &g0, douts_main, b, t, None, false, &mut dk_win, &mut dv_win);
        dg.iter_mut().zip(&dg_branch).for_each(|(a, b)| *a += b);
        self.learners[li].fuse_bwd(&taps, &dg, n);
        Ok(Learned { loss: loss as f32, ce: (ce_sum / ce_n.max(1) as f64) as f32, hit: hits as f32 / n as f32 })
    }

    /// The reference's cross entropies over both groups of the chunks, what the selective distillation ranks against
    fn reference_ce(&self, chunks: &[Chunk], t: usize) -> GroupCe {
        let cfg = self.cfg;
        let b = chunks.len();
        let n = b * t;
        let taps: Vec<f32> = chunks.iter().flat_map(|c| c.taps.iter().copied()).collect();
        let params = self.reference.clone();
        let exact = Read::Exact(&params);
        let g0 = DraftBlock::fuse(&exact, &cfg, &taps, n);
        let main_first: Vec<Option<u32>> = chunks.iter().flat_map(|c| c.main_tokens.iter().map(|t| Some(*t))).collect();
        let branch_first: Vec<Option<u32>> = chunks.iter().flat_map(|c| c.branch_tokens.iter().copied()).collect();
        let main = self.forward_group(&exact, &params, &g0, &main_first, b, t, None, false);
        let branch = self.forward_group(&exact, &params, &g0, &branch_first, b, t, Some(&main.ins[0]), true);
        let main_targets: Vec<StepTargets> = (0..cfg.depth).map(|s| chunks.iter().flat_map(|c| c.main_targets.iter().map(|row| row[s].as_ref().and_then(|soft| self.vocab.target(soft)))).collect()).collect();
        let branch_targets: Vec<StepTargets> = (0..cfg.depth).map(|s| chunks.iter().flat_map(|c| c.branch_targets.iter().map(|row| row[s].as_ref().and_then(|soft| self.vocab.target(soft)))).collect()).collect();
        (self.ce_values(&main, &main_targets), self.ce_values(&branch, &branch_targets))
    }

    /// The loss and gradients of every learner over the chunks, the gradients left in them, returning what the
    /// hedge's mixture of the learners reports
    pub fn loss_and_grads(&mut self, chunks: &[Chunk], t: usize) -> Result<TrainStep> {
        let ref_ce = (self.cfg.keep_frac < 1.0).then(|| self.reference_ce(chunks, t));
        let mut report = TrainStep { loss: 0.0, hit: 0.0 };
        for li in 0..self.learners.len() {
            let l = self.learn(li, chunks, t, ref_ce.as_ref())?;
            self.cum[li] += f64::from(l.ce);
            report.loss += self.weights[li] * l.loss;
            report.hit += self.weights[li] * l.hit;
        }
        Ok(report)
    }

    /// One training step over the chunks, every learner steps, the hedge reweighs them, the mixture, reference,
    /// and snapshot follow
    pub fn train(&mut self, chunks: &[Chunk], t: usize) -> Result<TrainStep> {
        let step = self.loss_and_grads(chunks, t)?;
        for l in &mut self.learners {
            l.adam();
        }
        self.rehedge();
        Ok(step)
    }

    /// Reweighs the learners by their running losses and rebuilds the mixture, the reference, and the snapshot
    pub fn rehedge(&mut self) {
        let best = self.cum.iter().copied().fold(f64::INFINITY, f64::min);
        let raw: Vec<f64> = self.cum.iter().map(|c| (-f64::from(self.cfg.hedge) * (c - best)).exp()).collect();
        let total: f64 = raw.iter().sum();
        self.weights = raw.iter().map(|r| (r / total) as f32).collect();
        for (pi, mix) in self.mixture.iter_mut().enumerate() {
            mix.iter_mut().for_each(|v| *v = 0.0);
            for (li, l) in self.learners.iter().enumerate() {
                let w = self.weights[li];
                mix.iter_mut().zip(&l.params[pi]).for_each(|(a, b)| *a += w * b);
            }
        }
        let ema = self.cfg.ema;
        for (r, mix) in self.reference.iter_mut().zip(&self.mixture) {
            r.iter_mut().zip(mix).for_each(|(a, b)| *a = ema * *a + (1.0 - ema) * b);
        }
        self.snapshot = Snapshot::of(&self.cfg, &self.mixture);
    }

    /// Replaces every learner, the mixture, and the reference with saved parameters, the running losses starting over
    pub fn set_params(&mut self, params: Vec<Vec<f32>>) -> Result<()> {
        for l in &mut self.learners {
            l.set_params(params.clone())?;
        }
        self.cum.iter_mut().for_each(|c| *c = 0.0);
        self.reference = params.clone();
        self.mixture = params;
        self.weights = vec![1.0 / self.learners.len() as f32; self.learners.len()];
        self.snapshot = Snapshot::of(&self.cfg, &self.mixture);
        Ok(())
    }

    /// The slot's history as the keys a draft begins with, when it runs up to the position the draft starts at
    fn history(&self, slot: &SlotDraft, start: u32) -> Keys {
        let rows: Vec<(u32, Vec<f32>, Vec<f32>)> = if slot.next_pos == Some(start) { slot.rows.iter().cloned().collect() } else { Vec::new() };
        Keys { hist: rows.len(), rows }
    }

    /// Runs the snapshot over rows whose g and e stand ready, each attending over the history within the window
    /// and its listed key rows, every row's key and value appended to the keys, returning the output rows
    fn run_rows(&self, g: &[f32], e: &[f32], pos: &[u32], anc: &[Vec<usize>], keys: &mut Keys) -> Vec<f32> {
        let cfg = self.cfg;
        let (d, h, hd, w) = (cfg.d, cfg.heads, cfg.head_dim(), cfg.window as u32);
        let m = pos.len();
        let read = Read::Snap(&self.snapshot);
        let inp = DraftBlock::layer_in(&read, &cfg, g, e, m);
        let first = keys.rows.len();
        for (r, p) in pos.iter().enumerate() {
            keys.rows.push((*p, inp.k[r * d..(r + 1) * d].to_vec(), inp.v[r * d..(r + 1) * d].to_vec()));
        }
        let nk = keys.rows.len();
        let (mut kk, mut vv) = (Vec::with_capacity(nk * d), Vec::with_capacity(nk * d));
        for (_, k, v) in keys.rows.iter() {
            kk.extend_from_slice(k);
            vv.extend_from_slice(v);
        }
        let mut mask = vec![false; m * nk];
        for r in 0..m {
            for j in 0..nk {
                let kpos = keys.rows[j].0;
                mask[r * nk + j] = (j < keys.hist && kpos <= pos[r] && kpos + w > pos[r]) || anc[r].contains(&j) || j == first + r;
            }
        }
        let mut a = vec![0.0; m * d];
        train::masked_attn_fwd(&inp.q, &kk, &vv, &mask, &mut a, m, nk, h, hd, 1.0 / (hd as f32).sqrt());
        DraftBlock::layer_out(&read, &cfg, &inp, &a, m).out
    }

    /// Writes the step zero rows of an accepted path into the slot's history and drafts a tree below the root
    /// picked at its end, the path rows attend to the history and the rows before them, the mask rows to all of
    /// them, and the tree takes the block's distributions at every drafted position, see docs/drafter.md
    pub fn draft(&self, path: &[DraftNode], root: u32, slot: &mut SlotDraft, rows: usize) -> Result<Tree> {
        let (tree, _) = self.draft_with_tops(path, root, slot, rows)?;
        Ok(tree)
    }

    /// The block's candidates at every drafted position below a root, best first, with the tree they build
    pub fn draft_with_tops(&self, path: &[DraftNode], root: u32, slot: &mut SlotDraft, rows: usize) -> Result<(Tree, Tops)> {
        let cfg = self.cfg;
        let (d, td, v) = (cfg.d, cfg.taps * cfg.d, self.vocab.len());
        ensure!(!path.is_empty() && path.iter().all(|n| n.taps.len() == td), "a path of {} rows of {} features for a block of {td}", path.len(), path.first().map_or(0, |n| n.taps.len()));
        let masks = if rows > 1 { cfg.depth - 1 } else { 0 };
        let mut keys = self.history(slot, path[0].pos);
        let hist = keys.hist;
        let m = path.len() + masks;
        let mut taps = Vec::with_capacity(m * td);
        let mut toks = Vec::with_capacity(path.len());
        let mut pos = Vec::with_capacity(m);
        let mut anc: Vec<Vec<usize>> = Vec::with_capacity(m);
        for (a, node) in path.iter().enumerate() {
            taps.extend_from_slice(&node.taps);
            toks.push(if a + 1 < path.len() { path[a + 1].token } else { root });
            pos.push(path[0].pos + a as u32);
            anc.push((0..a).map(|j| hist + j).collect());
        }
        let leaf = path.len() - 1;
        for j in 0..masks {
            taps.extend_from_slice(&path[leaf].taps);
            pos.push(pos[leaf]);
            anc.push((0..path.len() + j).map(|i| hist + i).collect());
        }
        let read = Read::Snap(&self.snapshot);
        let g = DraftBlock::fuse(&read, &cfg, &taps, m);
        let mut e = self.embed(&toks);
        for j in 0..masks {
            e.extend_from_slice(&self.snapshot.mask[j * d..(j + 1) * d]);
        }
        let out = self.run_rows(&g, &e, &pos, &anc, &mut keys);
        for (a, p) in pos.iter().enumerate().take(path.len()) {
            slot.push(*p, keys.rows[hist + a].1.clone(), keys.rows[hist + a].2.clone());
        }
        if masks == 0 {
            return Ok((Tree::root(root), Vec::new()));
        }
        let (z, _, _) = self.vocab.logits(&out[leaf * d..], masks + 1);
        let tops: Tops = (0..=masks).map(|j| self.vocab.top(&z[j * v..(j + 1) * v], v).into_iter().map(|(c, p)| (self.vocab.ids[c], p)).collect()).collect();
        let narrowed: Tops = tops.iter().enumerate().map(|(j, t)| t.iter().take(cfg.fanout[j]).copied().collect()).collect();
        Ok((tree_of_block(root, &narrowed, rows), tops))
    }

    /// Runs step zero over consecutive rows of a slot whose next tokens are known and writes their keys and values
    /// into the slot's history, as a prompt's rows do before the first draft
    pub fn prefill(&self, rows: &[(u32, Vec<f32>, u32)], slot: &mut SlotDraft) -> Result<()> {
        if rows.is_empty() {
            return Ok(());
        }
        ensure!(rows.windows(2).all(|w| w[0].0 + 1 == w[1].0), "prefill rows out of order");
        let mut keys = self.history(slot, rows[0].0);
        let hist = keys.hist;
        let m = rows.len();
        let taps: Vec<f32> = rows.iter().flat_map(|r| r.1.iter().copied()).collect();
        let read = Read::Snap(&self.snapshot);
        let g = DraftBlock::fuse(&read, &self.cfg, &taps, m);
        let toks: Vec<u32> = rows.iter().map(|r| r.2).collect();
        let pos: Vec<u32> = rows.iter().map(|r| r.0).collect();
        let w = self.cfg.window;
        let anc: Vec<Vec<usize>> = (0..m).map(|r| (r.saturating_sub(w - 1)..r).map(|j| hist + j).collect()).collect();
        self.run_rows(&g, &self.embed(&toks), &pos, &anc, &mut keys);
        for (r, row) in rows.iter().enumerate() {
            slot.push(row.0, keys.rows[hist + r].1.clone(), keys.rows[hist + r].2.clone());
        }
        Ok(())
    }
}

/// Scored rows kept for the block to learn from, the shared bookkeeping with the rows themselves
pub struct ScoredRing {
    pub meta: RingMeta,
    rows: Vec<Option<Scored>>,
}

impl ScoredRing {
    pub fn new(cap: usize, n_seq: usize, seed: u64) -> Self {
        let meta = RingMeta::new(cap, n_seq, seed);
        Self { rows: (0..meta.cap).map(|_| None).collect(), meta }
    }

    pub fn roll_back(&mut self, seq: usize) {
        self.meta.roll_back(seq);
    }

    /// Appends a row below a parent, on the model's own path when main, returning its ring index
    pub fn append(&mut self, row: Scored, main: bool, parent: i32) -> u32 {
        let at = self.meta.append(row.seq, row.pos, row.token, main, parent);
        self.rows[at as usize] = Some(row);
        at
    }

    /// Puts a row on the model's own path at a position, the compaction of a tree pass
    pub fn promote(&mut self, i: u32, pos: u32) {
        self.meta.promote(i, pos);
        if let Some(r) = self.rows[i as usize].as_mut() {
            r.pos = pos;
        }
    }

    pub fn appended(&self) -> u64 {
        self.meta.appended
    }

    fn row(&self, i: i32) -> &Scored {
        self.rows[i as usize].as_ref().expect("a live ring row")
    }

    /// Batch chunks of t rows from the shared sampling, their rows read out, none while no run is long enough
    pub fn chunks(&mut self, batch: usize, t: usize, depth: usize) -> Option<Vec<Chunk>> {
        let picks = self.meta.chunks(batch, t, depth)?;
        Some(picks
            .into_iter()
            .map(|c| Chunk {
                taps: c.rows.iter().flat_map(|r| self.row(*r as i32).taps.iter().copied()).collect(),
                main_tokens: c.main_tokens,
                main_targets: (0..t).map(|i| (0..depth).map(|j| Some(self.row(c.main_targets[j][i]).soft.clone())).collect()).collect(),
                main_truth: (0..t).map(|i| (0..depth).map(|j| Some(c.main_truth[j][i] as u32)).collect()).collect(),
                branch_tokens: c.branch_tokens.iter().map(|t| (*t >= 0).then_some(*t as u32)).collect(),
                branch_targets: (0..t).map(|i| (0..depth).map(|j| (c.branch_targets[j][i] >= 0).then(|| self.row(c.branch_targets[j][i]).soft.clone())).collect()).collect(),
            })
            .collect())
    }

    /// The v most frequent tokens of the rows kept, in rising id order
    pub fn frequent(&self, n_vocab: usize, v: usize) -> Vec<u32> {
        self.meta.frequent(n_vocab, v)
    }
}

/// The rows of the last pass over one slot, what the next draft hangs from
struct LastPass {
    rows: Vec<Scored>,
    ring: Vec<u32>,
}

/// Layer statistics the tap selection collects, block averaged tap rows of every layer over the rows of highest
/// predictive entropy, as kf_core::DRAFT_CAL_BLOCK rows a layer
pub struct Calibration {
    pub n_layers: usize,
    d: usize,
    z: Vec<Vec<f64>>,
    counts: Vec<u64>,
    pub retained: u64,
}

impl Calibration {
    /// Statistics over n_layers layers, sized by the width of the first rows observed
    pub fn new(n_layers: usize) -> Self {
        Self { n_layers, d: 0, z: Vec::new(), counts: vec![0; DRAFT_CAL_BLOCK], retained: 0 }
    }

    /// Takes a pass's rows, the rows of highest predictive entropy, the kept share of the pass, join the block
    /// averages in order
    pub fn observe(&mut self, taps: &[f32], logits: &[f32], rows: usize, n_vocab: usize) {
        if rows == 0 {
            return;
        }
        if self.d == 0 {
            self.d = taps.len() / (rows * self.n_layers.max(1));
            self.z = vec![vec![0.0; DRAFT_CAL_BLOCK * self.d]; self.n_layers];
        }
        let keep = ((rows as f32 * DRAFT_CAL_KEEP).ceil() as usize).clamp(1, rows);
        let mut order: Vec<(f32, usize)> = (0..rows).map(|r| (train::row_entropy(&logits[r * n_vocab..(r + 1) * n_vocab]), r)).collect();
        order.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        let mut chosen: Vec<usize> = order.iter().take(keep).map(|(_, r)| *r).collect();
        chosen.sort_unstable();
        let width = self.n_layers * self.d;
        for (k, r) in chosen.iter().enumerate() {
            let slot = (self.retained as usize + k) % DRAFT_CAL_BLOCK;
            for l in 0..self.n_layers {
                let src = &taps[r * width + l * self.d..r * width + (l + 1) * self.d];
                let dst = &mut self.z[l][slot * self.d..(slot + 1) * self.d];
                dst.iter_mut().zip(src).for_each(|(a, b)| *a += f64::from(*b));
            }
            self.counts[slot] += 1;
        }
        self.retained += chosen.len() as u64;
    }

    /// The gram matrix of every layer's block averaged rows
    pub fn grams(&self) -> Vec<Vec<Vec<f64>>> {
        let b = DRAFT_CAL_BLOCK;
        (0..self.n_layers)
            .map(|l| {
                let row = |i: usize| -> Vec<f64> {
                    let c = self.counts[i].max(1) as f64;
                    self.z[l][i * self.d..(i + 1) * self.d].iter().map(|v| v / c).collect()
                };
                let rows: Vec<Vec<f64>> = (0..b).map(row).collect();
                (0..b).map(|i| (0..b).map(|j| rows[i].iter().zip(&rows[j]).map(|(a, b)| a * b).sum()).collect()).collect()
            })
            .collect()
    }

    /// The m layers the statistics select, in rising order
    pub fn select(&self, m: usize, cka: bool) -> Vec<usize> {
        let mut taps = train::select_taps(&self.grams(), m, TAP_BEAM, TAP_EXPAND, cka);
        taps.sort_unstable();
        taps
    }
}

/// The block at work in the host backend, the ring it learns from, every slot's history, and the last pass's rows
pub struct HostDraft {
    pub draft: CpuDraft,
    pub ring: ScoredRing,
    pub tap_layers: Vec<usize>,
    slots: Vec<SlotDraft>,
    last: Vec<Option<LastPass>>,
    /// The ring index of every slot's last row on the model's own path
    last_main: Vec<i32>,
    batch: usize,
    draft_vocab: usize,
    n_vocab: usize,
    head_rows: Vec<f32>,
}

impl HostDraft {
    /// A block over the model's embedding table, head rows, and final norm, all as f32 tables
    #[allow(clippy::too_many_arguments)]
    pub fn new(cfg: DraftConfig, tap_layers: Vec<usize>, n_seq: usize, embd: Vec<f32>, head_rows: Vec<f32>, norm: Vec<f32>, n_vocab: usize, draft_vocab: usize, batch: usize, eos: Option<u32>) -> Result<Self> {
        ensure!(embd.len() == n_vocab * cfg.d && head_rows.len() == n_vocab * cfg.d && norm.len() == cfg.d, "tables of {} {} and {} values for {n_vocab} tokens of {}", embd.len(), head_rows.len(), norm.len(), cfg.d);
        ensure!(tap_layers.len() == cfg.taps && tap_layers.windows(2).all(|w| w[0] < w[1]), "tap layers {tap_layers:?} for a block of {} taps", cfg.taps);
        let draft_vocab = draft_vocab.clamp(1, n_vocab);
        let ids: Vec<u32> = (0..draft_vocab as u32).collect();
        let vocab = Vocab::new(ids.clone(), gather(&head_rows, &ids, cfg.d), norm, cfg.eps)?;
        let n = n_seq.max(1);
        Ok(Self { draft: CpuDraft::new(cfg, vocab, embd, eos)?, ring: ScoredRing::new(kf_core::DRAFT_RING_ROWS, n, cfg.seed), tap_layers, slots: (0..n).map(|_| SlotDraft::new(cfg.window - 1)).collect(), last: (0..n).map(|_| None).collect(), last_main: vec![-1; n], batch: batch.max(1), draft_vocab, n_vocab, head_rows })
    }

    /// Takes the rows of one slot's pass, a chain joins the ring on the model's path and the history, a tree joins
    /// the ring below the slot's last row as branches until compaction puts its kept path on the model's path
    pub fn pass(&mut self, seq: usize, rows: Vec<Scored>, parents: &[i32], tree: bool) -> Result<()> {
        ensure!(seq < self.slots.len() && !rows.is_empty() && parents.len() == rows.len(), "a pass of {} rows over slot {seq} of {}", rows.len(), self.slots.len());
        let td = self.draft.cfg.taps * self.draft.cfg.d;
        ensure!(rows.iter().all(|r| r.taps.len() == td), "rows of {} features for a block of {td}", rows[0].taps.len());
        if tree {
            let mut ring = Vec::with_capacity(rows.len());
            let mut depth = vec![0u32; rows.len()];
            for (i, r) in rows.iter().enumerate() {
                let parent = if parents[i] < 0 { self.last_main[seq] } else { ring[parents[i] as usize] as i32 };
                if parents[i] >= 0 {
                    depth[i] = depth[parents[i] as usize] + 1;
                }
                let mut row = r.clone();
                row.pos = rows[0].pos + depth[i];
                ring.push(self.ring.append(row, false, parent));
            }
            self.last[seq] = Some(LastPass { rows, ring });
            return Ok(());
        }
        let n = rows.len();
        let history: Vec<(u32, Vec<f32>, u32)> = (0..n - 1).map(|r| (rows[r].pos, rows[r].taps.clone(), rows[r + 1].token)).collect();
        self.draft.prefill(&history, &mut self.slots[seq])?;
        let mut ring = Vec::with_capacity(n);
        for r in &rows {
            let i = self.ring.append(r.clone(), true, self.last_main[seq]);
            self.last_main[seq] = i as i32;
            ring.push(i);
        }
        let last = rows[n - 1].clone();
        self.last[seq] = Some(LastPass { rows: vec![last], ring: vec![ring[n - 1]] });
        Ok(())
    }

    /// Writes the accepted path's step zero rows into the slot's history and drafts below the root picked at its end
    pub fn draft_tree(&mut self, seq: usize, path: &[usize], root: u32, rows: usize) -> Result<Tree> {
        ensure!(seq < self.slots.len(), "slot {seq} of {}", self.slots.len());
        let Some(last) = self.last[seq].as_ref() else { return Ok(Tree::root(root)) };
        ensure!(!path.is_empty() && path.iter().all(|i| *i < last.rows.len()), "a path of {:?} over {} rows", path, last.rows.len());
        let nodes: Vec<DraftNode> = path.iter().map(|i| DraftNode { token: last.rows[*i].token, pos: last.rows[*i].pos, taps: last.rows[*i].taps.clone() }).collect();
        self.draft.draft(&nodes, root, &mut self.slots[seq], rows)
    }

    /// The block's whole distribution at every drafted position below a root, best first, from a copy of the
    /// slot's history, so nothing changes
    pub fn tops(&self, seq: usize, path: &[usize], root: u32) -> Result<Tops> {
        ensure!(seq < self.slots.len(), "slot {seq} of {}", self.slots.len());
        let Some(last) = self.last[seq].as_ref() else { return Ok(Vec::new()) };
        ensure!(!path.is_empty() && path.iter().all(|i| *i < last.rows.len()), "a path of {:?} over {} rows", path, last.rows.len());
        let nodes: Vec<DraftNode> = path.iter().map(|i| DraftNode { token: last.rows[*i].token, pos: last.rows[*i].pos, taps: last.rows[*i].taps.clone() }).collect();
        let mut slot = self.slots[seq].clone();
        Ok(self.draft.draft_with_tops(&nodes, root, &mut slot, 2)?.1)
    }

    /// Puts the rows of the last tree pass that compaction kept on the model's path at the positions they now hold
    pub fn compacted(&mut self, seq: usize, base: usize, keep: &[usize]) {
        let Some(last) = self.last.get(seq).and_then(|l| l.as_ref()) else { return };
        for (i, k) in keep.iter().enumerate() {
            if let Some(idx) = last.ring.get(*k) {
                self.ring.promote(*idx, (base + i) as u32);
                self.last_main[seq] = *idx as i32;
            }
        }
    }

    /// Ends a slot's runs and history after its rows rolled back
    pub fn roll_back(&mut self, seq: usize) {
        self.ring.roll_back(seq);
        if let Some(s) = self.slots.get_mut(seq) {
            s.clear();
        }
        if let Some(l) = self.last.get_mut(seq) {
            *l = None;
        }
        if let Some(m) = self.last_main.get_mut(seq) {
            *m = -1;
        }
    }

    /// One training step on the ring, none while it holds no run long enough
    pub fn step(&mut self) -> Result<Option<TrainStep>> {
        let cfg = self.draft.cfg;
        let Some(chunks) = self.ring.chunks(self.batch, kf_core::DRAFT_CHUNK, cfg.depth) else { return Ok(None) };
        self.draft.train(&chunks, kf_core::DRAFT_CHUNK).map(Some)
    }

    /// The gradients of the first learner over one batch of the ring, left zero after, none while no run is long
    /// enough, what parity tests compare across backends
    pub fn probe_grads(&mut self) -> Result<Option<Vec<Vec<f32>>>> {
        let cfg = self.draft.cfg;
        let Some(chunks) = self.ring.chunks(self.batch, kf_core::DRAFT_CHUNK, cfg.depth) else { return Ok(None) };
        self.draft.loss_and_grads(&chunks, kf_core::DRAFT_CHUNK)?;
        let grads = self.draft.learners[0].grads.clone();
        for l in &mut self.draft.learners {
            l.grads.iter_mut().for_each(|g| g.iter_mut().for_each(|v| *v = 0.0));
        }
        Ok(Some(grads))
    }

    /// Recounts the draft vocabulary from the ring
    pub fn refresh_vocab(&mut self) -> Result<()> {
        let ids = self.ring.frequent(self.n_vocab, self.draft_vocab);
        self.set_vocab(ids)
    }

    fn set_vocab(&mut self, ids: Vec<u32>) -> Result<()> {
        let d = self.draft.cfg.d;
        let rows = gather(&self.head_rows, &ids, d);
        self.draft.vocab = Vocab::new(ids, rows, self.draft.vocab.norm.clone(), self.draft.vocab.eps)?;
        Ok(())
    }

    pub fn save(&self) -> Vec<u8> {
        kf_core::draft_save(&self.draft.cfg, &self.tap_layers, &self.draft.vocab.ids, &self.draft.mixture)
    }

    pub fn load(&mut self, bytes: &[u8]) -> Result<()> {
        let cfg = self.draft.cfg;
        let (taps, ids, params) = kf_core::draft_load(&cfg, bytes)?;
        ensure!(taps == self.tap_layers, "a saved block over layers {taps:?}, this block taps {:?}", self.tap_layers);
        ensure!(ids.iter().all(|t| (*t as usize) < self.n_vocab), "a saved vocabulary past {} tokens", self.n_vocab);
        self.draft.set_params(params)?;
        self.set_vocab(ids)
    }
}

/// Rows of a table by token id
fn gather(table: &[f32], ids: &[u32], d: usize) -> Vec<f32> {
    ids.iter().flat_map(|t| table[*t as usize * d..(*t as usize + 1) * d].iter().copied()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny(objective: Objective) -> DraftConfig {
        DraftConfig { d: 32, heads: 2, ff: 32, window: 3, taps: 2, depth: 3, fanout: [2, 2, 2, 1, 1, 1, 1, 1], learners: 2, objective, keep_frac: 0.6, ema: 0.5, hedge: 1.0, eps: 1e-5, lr: 1e-3, weight_decay: 0.0, seed: 9 }
    }

    /// A vocabulary of every fourth token of a table of forty, with random head rows and a random final norm
    fn vocab(d: usize, seed: &mut u64) -> (Vocab, Vec<f32>) {
        let n_vocab = 40;
        let embd = init_values(n_vocab * d, 1.0, seed);
        let ids: Vec<u32> = (0..n_vocab as u32).step_by(4).collect();
        let rows = init_values(ids.len() * d, 1.0, seed);
        let norm: Vec<f32> = init_values(d, 0.5, seed).iter().map(|v| 1.0 + v).collect();
        (Vocab::new(ids, rows, norm, 1e-5).expect("vocab"), embd)
    }

    fn soft(seed: &mut u64) -> Soft {
        let ids: Vec<u32> = (0..3).map(|_| (init_values(1, 1.0, seed)[0].abs() * 39.0) as u32).collect();
        let logits = init_values(3, 2.0, seed);
        let m = logits.iter().fold(f32::NEG_INFINITY, |a, b| a.max(*b));
        let lse = m + logits.iter().map(|l| (l - m).exp()).sum::<f32>().ln() + 0.3;
        Soft { ids, logits, lse }
    }

    fn chunks(cfg: &DraftConfig, b: usize, t: usize, seed: &mut u64) -> Vec<Chunk> {
        let tok = |seed: &mut u64| (init_values(1, 1.0, seed)[0].abs() * 39.0) as u32;
        (0..b)
            .map(|c| Chunk {
                taps: init_values(t * cfg.taps * cfg.d, 1.0, seed),
                main_tokens: (0..t).map(|_| tok(seed)).collect(),
                main_targets: (0..t).map(|_| (0..cfg.depth).map(|_| Some(soft(seed))).collect()).collect(),
                main_truth: (0..t).map(|_| (0..cfg.depth).map(|_| Some(tok(seed) / 4 * 4)).collect()).collect(),
                branch_tokens: (0..t).map(|i| ((i + c) % 2 == 0).then(|| tok(seed))).collect(),
                branch_targets: (0..t).map(|i| (0..cfg.depth).map(|j| ((i + c) % 2 == 0 && j < 2).then(|| soft(seed))).collect()).collect(),
            })
            .collect()
    }

    /// Every objective's gradients agree with central differences of its loss on every parameter of a learner
    #[test]
    fn draft_gradients_match_finite_differences() {
        for objective in [Objective::Ce, Objective::Edr, Objective::Bv] {
            let mut cfg = tiny(objective);
            cfg.learners = 1;
            cfg.keep_frac = 1.0;
            let (t, b) = (4usize, 2usize);
            let mut seed = 77u64;
            let (vocab, embd) = vocab(cfg.d, &mut seed);
            let mut draft = CpuDraft::new(cfg, vocab, embd, Some(8)).expect("draft");
            let chunks = chunks(&cfg, b, t, &mut seed);
            let step = draft.loss_and_grads(&chunks, t).expect("loss");
            assert!(step.loss.is_finite() && step.loss > 0.0, "{objective:?} loss {}", step.loss);
            let grads = draft.learners[0].grads.clone();
            let eps = 1e-3f32;
            #[allow(clippy::needless_range_loop)]
            for pi in 0..draft.learners[0].params.len() {
                for i in (0..draft.learners[0].params[pi].len()).step_by(11) {
                    let saved = draft.learners[0].params[pi][i];
                    draft.learners[0].params[pi][i] = saved + eps;
                    let up = draft.loss_and_grads(&chunks, t).expect("up").loss;
                    draft.learners[0].params[pi][i] = saved - eps;
                    let down = draft.loss_and_grads(&chunks, t).expect("down").loss;
                    draft.learners[0].params[pi][i] = saved;
                    draft.learners[0].grads.iter_mut().for_each(|g| g.iter_mut().for_each(|v| *v = 0.0));
                    let fd = (f64::from(up) - f64::from(down)) / (2.0 * f64::from(eps));
                    let an = f64::from(grads[pi][i]);
                    assert!((fd - an).abs() < 4e-3 * (1.0 + an.abs()), "{objective:?} parameter {pi} entry {i}: analytic {an} against difference {fd}");
                }
            }
        }
    }

    /// The selective distillation keeps the asked share of the targets and the hedge follows the running losses
    #[test]
    fn learners_hedge_and_filter() {
        let cfg = tiny(Objective::Ce);
        let (t, b) = (4usize, 2usize);
        let mut seed = 5u64;
        let (vocab, embd) = vocab(cfg.d, &mut seed);
        let mut draft = CpuDraft::new(cfg, vocab, embd, None).expect("draft");
        let chunks = chunks(&cfg, b, t, &mut seed);
        for _ in 0..3 {
            let step = draft.train(&chunks, t).expect("step");
            assert!(step.loss.is_finite());
        }
        assert!((draft.weights.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert!(draft.learners[0].lr < draft.learners[1].lr);
        let better = (0..2).min_by(|a, b| draft.cum[*a].total_cmp(&draft.cum[*b])).expect("a learner");
        assert!(draft.weights[better] >= 0.5, "the learner of lower running loss weighs more");
        // The snapshot's products stay close to the exact ones
        let x = init_values(3 * cfg.d, 1.0, &mut seed);
        let exact = Read::Exact(&draft.mixture).product(WQ, &x, 3, cfg.d, cfg.d);
        let snap = draft.snapshot.product(WQ, &x, 3);
        let err: f32 = exact.iter().zip(&snap).map(|(a, b)| (a - b).abs()).sum::<f32>() / exact.iter().map(|a| a.abs()).sum::<f32>();
        assert!(err < 0.03, "int8 rows within {err} of the exact product");
    }

    fn node(token: u32, pos: u32, cfg: &DraftConfig, seed: &mut u64) -> DraftNode {
        DraftNode { token, pos, taps: init_values(cfg.taps * cfg.d, 1.0, seed) }
    }

    /// A draft below a path writes the path into the history, builds its tree from the block's positions, and
    /// reads the history only when it runs up to the path
    #[test]
    fn drafts_follow_the_path_and_the_history() {
        let cfg = tiny(Objective::Ce);
        let mut seed = 31u64;
        let (vocab, embd) = vocab(cfg.d, &mut seed);
        let draft = CpuDraft::new(cfg, vocab, embd, None).expect("draft");
        let path = vec![node(5, 10, &cfg, &mut seed), node(4, 11, &cfg, &mut seed)];
        let mut slot = SlotDraft::new(cfg.window - 1);
        let tree = draft.draft(&path, 8, &mut slot, 6).expect("tree");
        assert_eq!((slot.rows.len(), slot.next_pos), (2, Some(12)), "the path rows joined the history");
        assert_eq!(tree.tokens[0], 8);
        assert!(tree.len() > 1 && tree.len() <= 6);
        let depth = |t: &Tree, mut i: usize| {
            let mut n = 0;
            while t.parents[i] >= 0 {
                i = t.parents[i] as usize;
                n += 1;
            }
            n
        };
        assert!((1..tree.len()).all(|i| depth(&tree, i) <= cfg.depth), "no row past the drafted positions");
        assert_eq!(draft.draft(&path, 8, &mut SlotDraft::new(cfg.window - 1), 1).expect("root"), Tree::root(8));
        let mut loud = SlotDraft::new(cfg.window - 1);
        for p in 8..10 {
            loud.push(p, init_values(cfg.d, 1.0, &mut seed), init_values(cfg.d, 100.0, &mut seed));
        }
        let with = draft.draft(&path, 8, &mut loud, 6).expect("tree");
        assert_ne!(with, tree, "the history reaches the drafts");
        let mut stale = SlotDraft::new(cfg.window - 1);
        stale.push(20, vec![0.0; cfg.d], vec![0.0; cfg.d]);
        assert_eq!(draft.draft(&path, 8, &mut stale, 6).expect("tree"), tree, "a history elsewhere is not read");
    }

    /// The calibration keeps the high entropy rows, block averages them, and selects the layers of richest spectrum
    #[test]
    fn calibration_selects_rich_layers() {
        let (n_layers, d) = (3usize, 8usize);
        let mut cal = Calibration::new(n_layers);
        let mut seed = 12u64;
        let rows = 40;
        let mut taps = vec![0.0f32; rows * n_layers * d];
        for r in 0..rows {
            for l in 0..n_layers {
                let v = init_values(d, 1.0, &mut seed);
                for c in 0..d {
                    // Layer one is one direction, the others full rank
                    taps[r * n_layers * d + l * d + c] = if l == 1 { v[0] } else { v[c] };
                }
            }
        }
        let logits: Vec<f32> = (0..rows).flat_map(|r| (0..5).map(move |c| if r % 2 == 0 { c as f32 * 3.0 } else { 0.0 })).collect();
        cal.observe(&taps, &logits, rows, 5);
        assert_eq!(cal.retained, 12, "three tenths of forty rows");
        let picked = cal.select(2, false);
        assert_eq!(picked.len(), 2);
        assert!(!picked.contains(&1), "the one direction layer is never picked, got {picked:?}");
        assert!(cal.select(1, true).len() == 1);
    }
}

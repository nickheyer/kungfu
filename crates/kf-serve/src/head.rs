//! The learned draft block at the server, when its tap layers are chosen, when it trains, which arm a round takes,
//! and where its weights rest between runs, see docs/drafter.md
use anyhow::{Context, Result, bail};
use kf_core::{Backend, DRAFT_CHUNK, DraftConfig, TrainStep, Tree, Wid};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// Tokens of the draft vocabulary, the most frequent the ring has seen
pub const DRAFT_VOCAB: usize = 32768;

/// Chunks of DRAFT_CHUNK rows one training step takes
pub const TRAIN_BATCH: usize = 4;

/// Rows of highest predictive entropy the tap selection keeps before choosing the block's layers
pub const CAL_ROWS: u64 = 4096;

/// Passes the training makes over every row the ring receives
const EPOCHS: f64 = 2.0;

/// Share of the decode time training may take while requests run
const TRAIN_SHARE: f32 = 0.1;

/// Training steps between recounts of the draft vocabulary
const VOCAB_EVERY: u64 = 256;

/// Time between saves of a block that trained since the last
const SAVE_EVERY: Duration = Duration::from_secs(600);

/// Hit rate from which the block's arms are played at all
const TRY_HIT: f32 = 0.25;

/// Weight of the newest round in the measured seconds and rate of an arm
const RATE_EMA: f32 = 1.0 / 16.0;

/// Weight of the newest step in the running loss and hit rate
const STEP_EMA: f32 = 1.0 / 64.0;

/// The block's weights and vocabulary beside the routing profile, and what they are
pub const FILE: &str = "draft_head.bin";
const META: &str = "draft_head.json";

/// Who proposed a round's tree
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    Seat,
    Head,
    Ngram,
    Union,
    None,
}

impl Source {
    /// The drafters a round can come from, in counter order
    pub const DRAFTERS: [Source; 4] = [Source::Seat, Source::Head, Source::Ngram, Source::Union];

    pub fn name(self) -> &'static str {
        match self {
            Source::Seat => "seat",
            Source::Head => "head",
            Source::Ngram => "ngram",
            Source::Union => "union",
            Source::None => "none",
        }
    }

    /// Position among the drafters, none for a round nothing drafted
    pub fn index(self) -> Option<usize> {
        Source::DRAFTERS.iter().position(|s| *s == self)
    }
}

/// The arms the hedge plays each round, the block's tree, the ngram tree, the block's tree with the ngram chain
/// grafted in, or plain decoding
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arm {
    Head,
    Ngram,
    Union,
    Plain,
}

impl Arm {
    pub const ALL: [Arm; 4] = [Arm::Head, Arm::Ngram, Arm::Union, Arm::Plain];

    pub fn name(self) -> &'static str {
        match self {
            Arm::Head => "head",
            Arm::Ngram => "ngram",
            Arm::Union => "union",
            Arm::Plain => "plain",
        }
    }

    pub fn index(self) -> usize {
        Arm::ALL.iter().position(|a| *a == self).expect("an arm")
    }

    pub fn source(self) -> Source {
        match self {
            Arm::Head => Source::Head,
            Arm::Ngram => Source::Ngram,
            Arm::Union => Source::Union,
            Arm::Plain => Source::None,
        }
    }
}

/// Normal hedge over the arms, parameter free, every arm's regret against the hedge's own loss, and the weights
/// the potential over the regrets implies
#[derive(Clone, Debug)]
pub struct Hedge {
    regret: [f64; 4],
    weights: [f64; 4],
}

impl Default for Hedge {
    fn default() -> Self {
        Self::new()
    }
}

impl Hedge {
    pub fn new() -> Self {
        Self { regret: [0.0; 4], weights: [0.25; 4] }
    }

    /// Takes a round's loss of every awake arm, the hedge's own loss the weighted one over the awake, each awake arm's
    /// regret moving by the difference and a sleeping arm's standing, and reweighs the arms
    pub fn update(&mut self, losses: &[f64; 4], awake: &[bool; 4]) {
        let n = awake.iter().filter(|a| **a).count();
        if n == 0 {
            return;
        }
        let mass: f64 = (0..4).filter(|i| awake[*i]).map(|i| self.weights[i]).sum();
        let own: f64 = (0..4).filter(|i| awake[*i]).map(|i| if mass > 0.0 { self.weights[i] / mass } else { 1.0 / n as f64 } * losses[i]).sum();
        for ((r, l), a) in self.regret.iter_mut().zip(losses).zip(awake) {
            if *a {
                *r += own - l;
            }
        }
        let plus: Vec<f64> = self.regret.iter().map(|r| r.max(0.0)).collect();
        if plus.iter().all(|r| *r <= 0.0) {
            self.weights = [0.25; 4];
            return;
        }
        // The scale c solving mean exp(r^2 / 2c) = e, found by bisection since the mean falls with c
        let potential = |c: f64| plus.iter().map(|r| (r * r / (2.0 * c)).exp()).sum::<f64>() / 4.0 - std::f64::consts::E;
        let top = plus.iter().copied().fold(0.0, f64::max);
        let (mut lo, mut hi) = (top * top * 1e-6 + 1e-12, top * top + 1.0);
        for _ in 0..200 {
            let mid = (lo * hi).sqrt();
            if potential(mid) > 0.0 { lo = mid } else { hi = mid }
        }
        let c = (lo * hi).sqrt();
        let raw: Vec<f64> = plus.iter().map(|r| r / c * (r * r / (2.0 * c)).exp()).collect();
        let total: f64 = raw.iter().sum();
        for (w, r) in self.weights.iter_mut().zip(raw) {
            *w = r / total;
        }
    }

    /// The arm of highest weight among the allowed, of least regret when none weighs, the earlier at a tie
    pub fn pick(&self, allowed: &[bool; 4]) -> Option<Arm> {
        let rank = |a: Arm| (self.weights[a.index()], self.regret[a.index()], usize::MAX - a.index());
        Arm::ALL.iter().zip(allowed).filter(|(_, ok)| **ok).map(|(a, _)| *a).max_by(|a, b| rank(*a).partial_cmp(&rank(*b)).expect("finite ranks"))
    }

    pub fn weights(&self) -> [f64; 4] {
        self.weights
    }
}

/// What the saved block says about itself, so another model's block is never loaded
#[derive(Serialize, Deserialize)]
struct Meta {
    model: String,
    n_embd: usize,
    n_vocab: usize,
    taps: Vec<usize>,
    depth: usize,
    steps: u64,
    loss: f32,
    hit: f32,
    rows: u64,
}

/// The block's schedule and measures at the server, the block itself living on the backend
pub struct Head {
    /// Most rows of a tree the block's drafts may fill
    pub rows: usize,
    pub taps: Vec<usize>,
    pub depth: usize,
    dir: Option<PathBuf>,
    model: String,
    n_embd: usize,
    n_vocab: usize,
    /// Training steps the ring's rows have earned and not yet spent
    credit: f64,
    rows_seen: u64,
    pub steps: u64,
    pub loss: f32,
    pub hit: f32,
    train_secs: f32,
    passes_since: u64,
    last_save: Instant,
    saved_steps: u64,
    pub hedge: Hedge,
    /// Seconds a round of each arm takes as measured, zero before any
    secs: [f32; 4],
    /// Tokens a second each arm's rounds yield as measured, zero before any
    rate: [f32; 4],
}

fn ema(old: f32, new: f32, w: f32) -> f32 {
    old + w * (new - old)
}

/// The expected accepted length of a tree from each depth's acceptance along the committed path, the bonus token
/// counted, every depth reached with the product of the acceptances before it
fn length(gammas: &[f32]) -> f32 {
    let mut total = 0.0f32;
    let mut reach = 1.0f32;
    for (k, g) in gammas.iter().enumerate() {
        total += (k as f32 + 1.0) * (1.0 - g) * reach;
        reach *= g;
    }
    total + (gammas.len() as f32 + 1.0) * reach
}

/// One round's measurement in progress, the arm played and its seconds, which arms were awake, and every drafting
/// arm's tree walked by the tokens committed since the round's root, each depth's acceptance the mass the model's
/// distribution there put on the tree's children, complete once every walk left its tree or reached a leaf
pub struct Measure {
    pub arm: Arm,
    pub secs: f32,
    pub awake: [bool; 4],
    trees: [Option<Tree>; 3],
    node: [Option<usize>; 3],
    gammas: [Vec<f32>; 3],
}

impl Measure {
    /// A measurement at the root of every awake arm's tree, the block's, the ngram's, and the union in that order
    pub fn new(arm: Arm, secs: f32, awake: [bool; 4], trees: [Option<Tree>; 3]) -> Self {
        let node = std::array::from_fn(|k| (awake[k] && trees[k].as_ref().is_some_and(|t| t.len() > 1)).then_some(0));
        Self { arm, secs, awake, trees, node, gammas: Default::default() }
    }

    /// Takes the model's distribution at the next committed position with the token committed there into every walk
    pub fn advance(&mut self, prob: &dyn Fn(u32) -> f32, token: u32) {
        for k in 0..3 {
            let (Some(n), Some(tree)) = (self.node[k], &self.trees[k]) else { continue };
            let children: Vec<u32> = (n + 1..tree.len()).filter(|i| tree.parents[*i] == n as i32).map(|i| tree.tokens[i]).collect();
            if children.is_empty() {
                self.node[k] = None;
                continue;
            }
            self.gammas[k].push(children.iter().map(|c| prob(*c)).sum::<f32>().min(1.0));
            self.node[k] = tree.child(n, token);
        }
    }

    /// Whether every walk has ended, so every arm's expected length is known
    pub fn done(&self) -> bool {
        self.node.iter().all(Option::is_none)
    }

    /// Every arm's expected accepted tokens past the bonus, plain decoding's none
    pub fn taus(&self) -> [f32; 4] {
        [length(&self.gammas[0]) - 1.0, length(&self.gammas[1]) - 1.0, length(&self.gammas[2]) - 1.0, 0.0]
    }
}

impl Head {
    #[allow(clippy::too_many_arguments)]
    pub fn new(rows: usize, taps: Vec<usize>, depth: usize, dir: Option<PathBuf>, model: String, n_embd: usize, n_vocab: usize) -> Self {
        Self { rows, taps, depth, dir, model, n_embd, n_vocab, credit: 0.0, rows_seen: 0, steps: 0, loss: 0.0, hit: 0.0, train_secs: 0.0, passes_since: 0, last_save: Instant::now(), saved_steps: 0, hedge: Hedge::new(), secs: [0.0; 4], rate: [0.0; 4] }
    }

    /// The tap layers and depth of the block saved in a directory for this model, none when there is none
    pub fn saved(dir: Option<&PathBuf>, model: &str, n_embd: usize, n_vocab: usize) -> Result<Option<(Vec<usize>, usize)>> {
        let Some(dir) = dir else { return Ok(None) };
        let meta_path = dir.join(META);
        if !meta_path.exists() {
            return Ok(None);
        }
        let meta: Meta = serde_json::from_str(&std::fs::read_to_string(&meta_path).with_context(|| format!("read {}", meta_path.display()))?).with_context(|| format!("parse {}", meta_path.display()))?;
        if meta.model != model || meta.n_embd != n_embd || meta.n_vocab != n_vocab {
            bail!("the saved block is of {} at width {} over {} tokens, this model is {model} at {n_embd} over {n_vocab}", meta.model, meta.n_embd, meta.n_vocab);
        }
        Ok(Some((meta.taps, meta.depth)))
    }

    /// Loads the block saved in the directory into the backend, true when one of this model was there
    pub fn load(&mut self, backend: &mut dyn Backend) -> Result<bool> {
        let Some(dir) = &self.dir else { return Ok(false) };
        let meta_path = dir.join(META);
        if !meta_path.exists() {
            return Ok(false);
        }
        let meta: Meta = serde_json::from_str(&std::fs::read_to_string(&meta_path).with_context(|| format!("read {}", meta_path.display()))?).with_context(|| format!("parse {}", meta_path.display()))?;
        if meta.model != self.model || meta.n_embd != self.n_embd || meta.n_vocab != self.n_vocab || meta.taps != self.taps || meta.depth != self.depth {
            bail!("the saved block is of {} at width {} over {} tokens tapping {:?} at depth {}, this block is {} at {} over {} tapping {:?} at depth {}", meta.model, meta.n_embd, meta.n_vocab, meta.taps, meta.depth, self.model, self.n_embd, self.n_vocab, self.taps, self.depth);
        }
        let path = dir.join(FILE);
        let bytes = std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
        backend.draft_load(&bytes)?;
        self.steps = meta.steps;
        self.saved_steps = meta.steps;
        self.loss = meta.loss;
        self.hit = meta.hit;
        self.rows_seen = 0;
        Ok(true)
    }

    /// Writes the block's weights and vocabulary with what they are, the weights replaced whole
    pub fn save(&mut self, backend: &mut dyn Backend) -> Result<()> {
        let Some(dir) = &self.dir else { return Ok(()) };
        std::fs::create_dir_all(dir)?;
        let bytes = backend.draft_save()?;
        let path = dir.join(FILE);
        let tmp = dir.join(format!("{FILE}.tmp"));
        std::fs::write(&tmp, &bytes).with_context(|| format!("write {}", tmp.display()))?;
        std::fs::rename(&tmp, &path).with_context(|| format!("replace {}", path.display()))?;
        let meta = Meta { model: self.model.clone(), n_embd: self.n_embd, n_vocab: self.n_vocab, taps: self.taps.clone(), depth: self.depth, steps: self.steps, loss: self.loss, hit: self.hit, rows: self.rows_seen };
        std::fs::write(dir.join(META), serde_json::to_string_pretty(&meta)?)?;
        self.saved_steps = self.steps;
        self.last_save = Instant::now();
        Ok(())
    }

    /// Whether the block trained since its last save and enough time has passed to save again
    pub fn save_due(&self) -> bool {
        self.unsaved() && self.last_save.elapsed() >= SAVE_EVERY
    }

    /// Whether the block trained since its last save
    pub fn unsaved(&self) -> bool {
        self.dir.is_some() && self.steps > self.saved_steps
    }

    /// Credits the steps the rows the ring received since the last call earn, EPOCHS passes over each
    pub fn earn(&mut self, rows: u64) {
        let new = rows.saturating_sub(self.rows_seen);
        self.rows_seen = rows;
        self.credit += new as f64 * EPOCHS / (TRAIN_BATCH * DRAFT_CHUNK) as f64;
    }

    /// Counts a decode pass between training steps
    pub fn passed(&mut self) {
        self.passes_since += 1;
    }

    /// Whether a training step is due, credit stands, and while requests run enough passes went by since the last
    /// step for training to stay under its share of the decode time
    pub fn due(&self, idle: bool, decode_secs: f32) -> bool {
        if self.credit < 1.0 {
            return false;
        }
        if idle {
            return true;
        }
        let every = if decode_secs > 0.0 { (self.train_secs / (TRAIN_SHARE * decode_secs)).ceil().max(1.0) as u64 } else { 1 };
        self.passes_since >= every
    }

    /// Takes a step's report into the running loss, hit rate, and step time
    pub fn trained(&mut self, step: TrainStep, secs: f32) {
        self.steps += 1;
        self.credit -= 1.0;
        self.passes_since = 0;
        let first = self.steps == 1;
        self.loss = if first { step.loss } else { ema(self.loss, step.loss, STEP_EMA) };
        self.hit = if first { step.hit } else { ema(self.hit, step.hit, STEP_EMA) };
        self.train_secs = if self.train_secs == 0.0 { secs } else { ema(self.train_secs, secs, STEP_EMA) };
    }

    /// Whether the step just taken is one the draft vocabulary is recounted after
    pub fn vocab_due(&self) -> bool {
        self.steps.is_multiple_of(VOCAB_EVERY)
    }

    /// Whether the block's hit rate allows its arms to play
    pub fn warm(&self) -> bool {
        self.hit >= TRY_HIT
    }

    /// The arms a round may play, the block's and the union's once it is warm, the ngram's with an ngram drafter,
    /// and plain decoding always
    pub fn awake(&self, ngram: bool) -> [bool; 4] {
        let warm = self.warm();
        [warm, ngram, warm && ngram, true]
    }

    /// The arm of the next round, among the awake arms one not yet measured first so every arm has its seconds,
    /// then the hedge's heaviest
    pub fn arm(&self, ngram: bool) -> Arm {
        let awake = self.awake(ngram);
        if let Some(a) = Arm::ALL.iter().zip(&awake).find(|(a, ok)| **ok && self.secs[a.index()] == 0.0) {
            return *a.0;
        }
        self.hedge.pick(&awake).unwrap_or(Arm::Plain)
    }

    /// Takes a round's complete measurement, the chosen arm's seconds, every awake arm's expected accepted length
    /// against what was committed from the round's root, and the hedge's loss of each, one less its throughput over
    /// the most a round could yield
    pub fn observe(&mut self, chosen: Arm, secs: f32, taus: &[f32; 4], awake: &[bool; 4], decode_secs: f32, most: usize) {
        if secs <= 0.0 || decode_secs <= 0.0 {
            return;
        }
        let c = chosen.index();
        self.secs[c] = if self.secs[c] == 0.0 { secs } else { ema(self.secs[c], secs, RATE_EMA) };
        self.secs[Arm::Plain.index()] = decode_secs;
        let mut losses = [0.0f64; 4];
        for a in Arm::ALL {
            let i = a.index();
            if !awake[i] {
                continue;
            }
            let time = if self.secs[i] > 0.0 { self.secs[i] } else { secs };
            let rate = (1.0 + taus[i]) / time;
            self.rate[i] = if self.rate[i] == 0.0 { rate } else { ema(self.rate[i], rate, RATE_EMA) };
            losses[i] = f64::from((1.0 - rate * decode_secs / (1.0 + most as f32)).clamp(0.0, 1.0));
        }
        self.hedge.update(&losses, awake);
    }

    /// Tokens a second each arm's rounds yield, as measured
    pub fn rates(&self) -> [f32; 4] {
        self.rate
    }
}

/// The tap selection in progress, what the block needs once the layers are chosen
pub struct Calibrating {
    pub cfg: DraftConfig,
    pub cka: bool,
    pub rows: usize,
    pub dir: Option<PathBuf>,
    pub model: String,
    pub n_layers: usize,
    pub n_vocab: usize,
    pub eos: Option<u32>,
    pub embed: Wid,
    pub head: Wid,
    pub norm: Wid,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn credit_follows_rows_and_training_share() {
        let mut h = Head::new(16, vec![1, 2], 5, None, "m".into(), 64, 100);
        h.earn(0);
        assert!(!h.due(true, 0.001));
        h.earn((TRAIN_BATCH * DRAFT_CHUNK) as u64);
        assert!(h.due(true, 0.001), "two epochs over one batch of rows earn two steps");
        h.trained(TrainStep { loss: 0.5, hit: 0.1 }, 0.010);
        assert!(h.due(true, 0.001) && !h.due(false, 0.001), "serving waits for passes");
        for _ in 0..100 {
            h.passed();
        }
        assert!(h.due(false, 0.001), "a ten millisecond step after a hundred one millisecond passes stays under a tenth");
        h.trained(TrainStep { loss: 0.3, hit: 0.3 }, 0.010);
        assert!(!h.due(true, 0.001), "the credit is spent");
        assert!(h.loss < 0.5 && h.loss > 0.3 && h.hit > 0.1);
    }

    #[test]
    fn hedge_weighs_the_arms_by_regret() {
        let mut hd = Hedge::new();
        for _ in 0..30 {
            hd.update(&[0.2, 0.8, 0.5, 0.9], &[true; 4]);
        }
        let w = hd.weights();
        assert!(w[0] > 0.9, "the arm of least loss takes the weight, got {w:?}");
        assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        assert_eq!(hd.pick(&[true, true, true, true]), Some(Arm::Head));
        assert_eq!(hd.pick(&[false, true, true, true]), Some(Arm::Union), "the allowed arm of least regret when none weighs");
        assert_eq!(hd.pick(&[false; 4]), None);
        let mut even = Hedge::new();
        even.update(&[0.5; 4], &[true; 4]);
        assert_eq!(even.weights(), [0.25; 4], "equal losses leave the arms even");
    }

    #[test]
    fn sleeping_arms_keep_their_regret() {
        let mut hd = Hedge::new();
        for _ in 0..30 {
            hd.update(&[0.0, 0.8, 0.0, 0.9], &[false, true, false, true]);
        }
        assert_eq!((hd.regret[0], hd.regret[2]), (0.0, 0.0), "a sleeping arm's regret stands however its loss reads");
        assert!(hd.regret[1] > 0.0 && hd.regret[3] < 0.0, "the awake arms settle it among themselves, got {:?}", hd.regret);
        assert_eq!(hd.pick(&[false, true, false, true]), Some(Arm::Ngram));
        let w = hd.weights();
        assert!(w[1] > 0.99, "got {w:?}");
        // Once awake the block's arm starts level, its regret zero, and a round it wins moves it
        hd.update(&[0.1, 0.8, 0.1, 0.9], &[true; 4]);
        assert!(hd.regret[0] > 0.0 && hd.regret[2] > 0.0);
        hd.update(&[0.5; 4], &[false; 4]);
    }

    #[test]
    fn measure_walks_every_tree_along_the_committed_tokens() {
        // A chain 1 2 3 below root 0 for the block, a chain 1 7 for the ngram, the model putting 0.8 on each next
        // token the chains name and 0.2 elsewhere
        let mut chain = Tree::root(0);
        let a = chain.push(0, 1).expect("row");
        let b = chain.push(a, 2).expect("row");
        chain.push(b, 3).expect("row");
        let mut short = Tree::root(0);
        let a = short.push(0, 1).expect("row");
        short.push(a, 7).expect("row");
        let prob = |tok: u32| if tok > 0 { 0.8f32 } else { 0.0 };
        let mut m = Measure::new(Arm::Head, 0.003, [true; 4], [Some(chain.clone()), Some(short.clone()), None]);
        assert!(!m.done(), "two trees with children wait for the walk");
        m.advance(&prob, 1);
        assert!(!m.done());
        m.advance(&prob, 2);
        assert!(!m.done(), "the chain still holds the path");
        m.advance(&prob, 3);
        assert!(!m.done(), "the chain's leaf reached, its children unknown until the next position");
        m.advance(&prob, 9);
        assert!(m.done());
        let [head, ngram, union, plain] = m.taus();
        let want = 1.0 * 0.2 + 2.0 * 0.8 * 0.2 + 3.0 * 0.64 * 0.2 + 4.0 * 0.512 - 1.0;
        assert!((head - want).abs() < 1e-5, "{head} against {want}");
        // The committed path left the short chain at depth two, the mass there counting and nothing known past it
        let want = 0.2 + 2.0 * 0.8 * 0.2 + 3.0 * 0.64 - 1.0;
        assert!((ngram - want).abs() < 1e-5, "{ngram} against {want}");
        assert_eq!((union, plain), (0.0, 0.0), "an arm without a tree yields its bonus token alone");
        // A bare root, a missing tree, and a sleeping arm's tree need no walk
        let bare = Measure::new(Arm::Plain, 0.001, [false, true, false, true], [Some(chain), Some(Tree::root(0)), None]);
        assert!(bare.done(), "nothing awake with children");
        assert_eq!(bare.taus(), [0.0; 4]);
    }

    #[test]
    fn arms_run_once_unmeasured_then_by_weight() {
        let mut h = Head::new(16, vec![1], 5, None, "m".into(), 64, 100);
        assert_eq!(h.arm(true), Arm::Ngram, "a cold block leaves the ngram arm and plain decoding, the ngram unmeasured first");
        let cold = [false, true, false, true];
        h.observe(Arm::Ngram, 0.003, &[0.0, 2.0, 0.0, 0.0], &cold, 0.001, 20);
        assert_eq!(h.arm(true), Arm::Ngram, "plain decoding needs no measuring round, its seconds are the decode pass, and the two tie");
        h.observe(Arm::Plain, 0.001, &[0.0, 2.0, 0.0, 0.0], &cold, 0.001, 20);
        for _ in 0..40 {
            h.observe(Arm::Ngram, 0.0025, &[0.0, 2.0, 0.0, 0.0], &cold, 0.001, 20);
        }
        assert_eq!(h.rates()[Arm::Head.index()], 0.0, "a sleeping arm is not measured");
        assert_eq!(h.arm(true), Arm::Ngram, "three tokens in two and a half passes beats one a pass");
        for _ in 0..20 {
            h.trained(TrainStep { loss: 0.1, hit: 0.9 }, 0.01);
        }
        assert_eq!(h.arm(true), Arm::Head, "a warm block's arm runs until measured");
        assert_eq!(h.awake(true), [true; 4]);
        h.observe(Arm::Head, 0.003, &[5.0, 2.0, 5.0, 0.0], &[true; 4], 0.001, 20);
        assert_eq!(h.arm(true), Arm::Union, "the union runs until measured");
        for _ in 0..40 {
            h.observe(Arm::Union, 0.0032, &[5.0, 2.0, 5.5, 0.0], &[true; 4], 0.001, 20);
        }
        assert_eq!(h.arm(true), Arm::Union, "the arm of highest throughput wins");
        assert!(h.rates()[Arm::Union.index()] > h.rates()[Arm::Ngram.index()]);
        assert_eq!(Source::Union.index(), Some(3));
        assert_eq!(Arm::Plain.source().index(), None);
    }
}

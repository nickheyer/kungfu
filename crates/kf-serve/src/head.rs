//! The learned draft head at the server: when it trains, which drafter a round takes, and where its weights rest
//! between runs, see docs/drafter.md
use anyhow::{Context, Result, bail};
use kf_core::{Backend, HEAD_CHUNK, TrainStep};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// Tokens of the draft vocabulary, the most frequent the ring has seen
pub const DRAFT_VOCAB: usize = 32768;

/// Chunks of HEAD_CHUNK rows one training step takes
pub const TRAIN_BATCH: usize = 4;

/// Passes the training makes over every row the ring receives
const EPOCHS: f64 = 2.0;

/// Share of the decode time training may take while requests run
const TRAIN_SHARE: f32 = 0.1;

/// Training steps between recounts of the draft vocabulary
const VOCAB_EVERY: u64 = 256;

/// Time between saves of a head that trained since the last
const SAVE_EVERY: Duration = Duration::from_secs(600);

/// Hit rate from which the head's rounds are tried at all
const TRY_HIT: f32 = 0.25;

/// Every so many rounds the drafter not chosen runs, so its measure stays fresh
const TRY_EVERY: u64 = 8;

/// Weight of the newest round in the measured rate of a drafter
const RATE_EMA: f32 = 1.0 / 16.0;

/// Weight of the newest step in the running loss and hit rate
const STEP_EMA: f32 = 1.0 / 64.0;

/// The head's weights and vocabulary beside the routing profile, and what they are
pub const FILE: &str = "draft_head.bin";
const META: &str = "draft_head.json";

/// Who proposed a round's tree
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    Seat,
    Head,
    Ngram,
    None,
}

impl Source {
    /// The drafters a round can come from, in counter order
    pub const DRAFTERS: [Source; 3] = [Source::Seat, Source::Head, Source::Ngram];

    pub fn name(self) -> &'static str {
        match self {
            Source::Seat => "seat",
            Source::Head => "head",
            Source::Ngram => "ngram",
            Source::None => "none",
        }
    }

    /// Position among the drafters, none for a round nothing drafted
    pub fn index(self) -> Option<usize> {
        Source::DRAFTERS.iter().position(|s| *s == self)
    }
}

/// What the saved head says about itself, so another model's head is never loaded
#[derive(Serialize, Deserialize)]
struct Meta {
    model: String,
    n_embd: usize,
    n_vocab: usize,
    steps: u64,
    loss: f32,
    hit: f32,
    rows: u64,
}

/// The head's schedule and measures at the server, the head itself living on the backend
pub struct Head {
    /// Rows of the tree the head drafts
    pub rows: usize,
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
    /// Tokens a second a round yields with the head's tree and with the ngram's, as measured, zero before any round
    rate_head: f32,
    rate_ngram: f32,
    rounds: u64,
}

fn ema(old: f32, new: f32, w: f32) -> f32 {
    old + w * (new - old)
}

impl Head {
    pub fn new(rows: usize, dir: Option<PathBuf>, model: String, n_embd: usize, n_vocab: usize) -> Self {
        Self { rows, dir, model, n_embd, n_vocab, credit: 0.0, rows_seen: 0, steps: 0, loss: 0.0, hit: 0.0, train_secs: 0.0, passes_since: 0, last_save: Instant::now(), saved_steps: 0, rate_head: 0.0, rate_ngram: 0.0, rounds: 0 }
    }

    /// Loads the head saved in the directory into the backend, true when one of this model was there
    pub fn load(&mut self, backend: &mut dyn Backend) -> Result<bool> {
        let Some(dir) = &self.dir else { return Ok(false) };
        let meta_path = dir.join(META);
        if !meta_path.exists() {
            return Ok(false);
        }
        let meta: Meta = serde_json::from_str(&std::fs::read_to_string(&meta_path).with_context(|| format!("read {}", meta_path.display()))?).with_context(|| format!("parse {}", meta_path.display()))?;
        if meta.model != self.model || meta.n_embd != self.n_embd || meta.n_vocab != self.n_vocab {
            bail!("the saved head is of {} at width {} over {} tokens, this model is {} at {} over {}", meta.model, meta.n_embd, meta.n_vocab, self.model, self.n_embd, self.n_vocab);
        }
        let path = dir.join(FILE);
        let bytes = std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
        backend.drafter_load(&bytes)?;
        self.steps = meta.steps;
        self.saved_steps = meta.steps;
        self.loss = meta.loss;
        self.hit = meta.hit;
        self.rows_seen = 0;
        Ok(true)
    }

    /// Writes the head's weights and vocabulary with what they are, the weights replaced whole
    pub fn save(&mut self, backend: &mut dyn Backend) -> Result<()> {
        let Some(dir) = &self.dir else { return Ok(()) };
        std::fs::create_dir_all(dir)?;
        let bytes = backend.drafter_save()?;
        let path = dir.join(FILE);
        let tmp = dir.join(format!("{FILE}.tmp"));
        std::fs::write(&tmp, &bytes).with_context(|| format!("write {}", tmp.display()))?;
        std::fs::rename(&tmp, &path).with_context(|| format!("replace {}", path.display()))?;
        let meta = Meta { model: self.model.clone(), n_embd: self.n_embd, n_vocab: self.n_vocab, steps: self.steps, loss: self.loss, hit: self.hit, rows: self.rows_seen };
        std::fs::write(dir.join(META), serde_json::to_string_pretty(&meta)?)?;
        self.saved_steps = self.steps;
        self.last_save = Instant::now();
        Ok(())
    }

    /// Whether the head trained since its last save and enough time has passed to save again
    pub fn save_due(&self) -> bool {
        self.unsaved() && self.last_save.elapsed() >= SAVE_EVERY
    }

    /// Whether the head trained since its last save
    pub fn unsaved(&self) -> bool {
        self.dir.is_some() && self.steps > self.saved_steps
    }

    /// Credits the steps the rows the ring received since the last call earn, EPOCHS passes over each
    pub fn earn(&mut self, rows: u64) {
        let new = rows.saturating_sub(self.rows_seen);
        self.rows_seen = rows;
        self.credit += new as f64 * EPOCHS / (TRAIN_BATCH * HEAD_CHUNK) as f64;
    }

    /// Counts a decode pass between training steps
    pub fn passed(&mut self) {
        self.passes_since += 1;
    }

    /// Whether a training step is due: credit stands, and while requests run enough passes went by since the last
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

    /// The drafter of the next round: the one measured to yield the most tokens a second, plain decoding counting
    /// as one token a decode pass, a drafter without a measure yet running until it has one, the head only once its
    /// hit rate reaches TRY_HIT, and every TRY_EVERY rounds a drafter not chosen so its measure stays fresh
    pub fn source(&mut self, ngram: bool, decode_secs: f32) -> Source {
        self.rounds += 1;
        let plain = if decode_secs > 0.0 { 1.0 / decode_secs } else { 0.0 };
        let head_ok = self.hit >= TRY_HIT;
        if ngram && self.rate_ngram == 0.0 {
            return Source::Ngram;
        }
        if head_ok && self.rate_head == 0.0 {
            return Source::Head;
        }
        let (mut best, mut best_rate) = (Source::None, plain);
        if ngram && self.rate_ngram > best_rate {
            (best, best_rate) = (Source::Ngram, self.rate_ngram);
        }
        if head_ok && self.rate_head > best_rate {
            best = Source::Head;
        }
        if self.rounds.is_multiple_of(TRY_EVERY) {
            let others: Vec<Source> = [(head_ok, Source::Head), (ngram, Source::Ngram)].into_iter().filter(|(on, s)| *on && *s != best).map(|(_, s)| s).collect();
            if !others.is_empty() {
                return others[(self.rounds / TRY_EVERY) as usize % others.len()];
            }
        }
        best
    }

    /// Takes a drafted round's outcome into its drafter's measured rate: the tokens the round yielded, the
    /// sampler's own, the accepted path, and the model's next, over the seconds the round took
    pub fn observe(&mut self, source: Source, taken: usize, secs: f32) {
        if secs <= 0.0 {
            return;
        }
        let rate = (taken + 2) as f32 / secs;
        match source {
            Source::Head => self.rate_head = if self.rate_head == 0.0 { rate } else { ema(self.rate_head, rate, RATE_EMA) },
            Source::Ngram => self.rate_ngram = if self.rate_ngram == 0.0 { rate } else { ema(self.rate_ngram, rate, RATE_EMA) },
            Source::Seat | Source::None => {}
        }
    }

    /// The measured rates of the head and the ngram drafter, tokens a second
    pub fn rates(&self) -> (f32, f32) {
        (self.rate_head, self.rate_ngram)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn credit_follows_rows_and_training_share() {
        let mut h = Head::new(16, None, "m".into(), 64, 100);
        h.earn(0);
        assert!(!h.due(true, 0.001));
        h.earn((TRAIN_BATCH * HEAD_CHUNK) as u64);
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
    fn source_follows_the_measured_rates() {
        let mut h = Head::new(16, None, "m".into(), 64, 100);
        assert_eq!(h.source(true, 0.001), Source::Ngram, "an unmeasured ngram drafter runs");
        for _ in 0..40 {
            h.observe(Source::Ngram, 0, 0.0025);
        }
        assert_eq!(h.source(true, 0.001), Source::None, "two tokens in two and a half passes loses to one a pass");
        assert_eq!(h.source(false, 0.001), Source::None, "a cold head is never tried");
        for _ in 0..20 {
            h.trained(TrainStep { loss: 0.1, hit: 0.9 }, 0.01);
        }
        assert_eq!(h.source(true, 0.001), Source::Head, "a head whose hit rate allows runs until measured");
        for _ in 0..40 {
            h.observe(Source::Head, 3, 0.003);
        }
        assert_eq!(h.source(true, 0.001), Source::Head, "five tokens in three passes wins");
        let trials = (0..2 * TRY_EVERY).filter(|_| h.source(true, 0.001) == Source::Ngram).count();
        assert_eq!(trials, 2, "the drafter not chosen gets every TRY_EVERY th round");
        assert_eq!(Source::Head.index(), Some(1));
        assert_eq!(Source::None.index(), None);
    }
}

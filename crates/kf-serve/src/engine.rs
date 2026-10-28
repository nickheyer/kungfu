//! The scheduler thread that owns the session, admitting requests to slots and decoding every active slot in one pass
use crate::api::Metrics;
use crate::grammar::Constraint;
use crate::tokenizer::Tok;
use kf_vision::{Clip, ImageInput};
use anyhow::{Result, anyhow, bail, ensure};
use kf_core::event;
use kf_engine::{Rng, RoutingProfile, Sampling, Session, sample};
use serde_json::{Value, json};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc as chan};
use std::time::Instant;
use tokio::sync::{mpsc, oneshot};

/// What a request asks of the sampler beyond the distribution itself
#[derive(Clone, Debug)]
pub struct Limits {
    pub max_tokens: usize,
    pub seed: u64,
    pub stop: Vec<String>,
    pub bias: Vec<(u32, f32)>,
}

/// The log probability of one generated token under the model with the most likely tokens at that position
#[derive(Clone, Debug, PartialEq)]
pub struct TokenLogprob {
    pub token: u32,
    pub logprob: f32,
    /// The most likely tokens with their log probabilities, most likely first
    pub top: Vec<(u32, f32)>,
}

/// What came out of one generation
#[derive(Clone, Debug)]
pub struct Outcome {
    pub text: String,
    pub n_prompt: usize,
    pub cached: usize,
    pub n_out: usize,
    pub finish: &'static str,
    /// The stop string that ended the generation, when one did
    pub stop_sequence: Option<String>,
    /// One entry per generated token kept in the text, empty when the request asked for none
    pub logprobs: Vec<TokenLogprob>,
}

/// A piece of text with the log probabilities of the tokens that completed inside it
pub enum Piece {
    Text(String, Vec<TokenLogprob>),
    Done(Outcome),
    Failed(String),
}

/// One piece of a prompt, tokens or an image whose rows stand in for tokens
pub enum Chunk {
    Text(Vec<u32>),
    Image(ImageInput),
}

/// One generation request as the scheduler takes it
pub struct Job {
    /// Every row of the prompt by key, tokens as themselves and image rows by the image's key, the prefix cache compares these
    pub ids: Vec<u32>,
    /// The prompt in order, the keys of each chunk laid out in ids
    pub chunks: Vec<Chunk>,
    pub sampling: Sampling,
    pub limits: Limits,
    /// The slot the request must run in, any free slot when none
    pub slot: Option<usize>,
    /// The grammar every sampled token must continue
    pub constraint: Option<Constraint>,
    /// Reasoning tokens allowed inside a think block before its closing tag is forced, none for no limit
    pub think_budget: Option<usize>,
    /// Whether the prompt left a think block open
    pub thinking_open: bool,
    /// How many most likely tokens to report beside each token's own log probability, none when not asked
    pub logprobs: Option<usize>,
    pub cancel: Arc<AtomicBool>,
    pub tx: mpsc::Sender<Piece>,
}

pub const THINK_OPEN: &str = "<think>";
pub const THINK_CLOSE: &str = "</think>";

/// One token under every rule a request adds, the bias, the grammar, and the end of sequence, none when the grammar is whole and nothing may follow
#[allow(clippy::too_many_arguments)]
pub fn choose(logits: &[f32], sampling: &Sampling, recent: &[u32], rng: &mut Rng, bias: &[(u32, f32)], constraint: &mut Option<Constraint>, tok: &Tok, first: bool) -> Result<Option<u32>> {
    let mut l = logits.to_vec();
    for (id, b) in bias {
        if let Some(v) = l.get_mut(*id as usize) {
            *v += b;
        }
    }
    let Some(con) = constraint else { return Ok(Some(sample(&l, sampling, recent, rng))) };
    if con.complete() && !con.can_continue() {
        return Ok(None);
    }
    let mut masked = 0usize;
    loop {
        if masked >= l.len() {
            break;
        }
        let t = sample(&l, sampling, recent, rng);
        if l[t as usize] == f32::NEG_INFINITY {
            break;
        }
        if tok.is_eos(t) {
            if con.complete() {
                return Ok(None);
            }
            l[t as usize] = f32::NEG_INFINITY;
            masked += 1;
            continue;
        }
        match con.try_advance(tok.piece(t, first)) {
            Some(next) => {
                *con = next;
                return Ok(Some(t));
            }
            None => {
                l[t as usize] = f32::NEG_INFINITY;
                masked += 1;
            }
        }
    }
    if con.complete() {
        return Ok(None);
    }
    bail!("no token of the vocabulary continues the grammar")
}

/// The log probability of one token under the raw logits with the n_top most likely tokens, ties broken by lower id
pub fn logprob_entry(logits: &[f32], token: u32, n_top: usize) -> TokenLogprob {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let lse = max + (logits.iter().map(|v| f64::from(v - max).exp()).sum::<f64>().ln() as f32);
    let mut top: Vec<(u32, f32)> = Vec::with_capacity(n_top + 1);
    if n_top > 0 {
        for (i, &v) in logits.iter().enumerate() {
            if top.len() < n_top || v > top[top.len() - 1].1 {
                let at = top.partition_point(|(_, t)| *t >= v);
                top.insert(at, (i as u32, v));
                if top.len() > n_top {
                    top.pop();
                }
            }
        }
    }
    let lp = |v: f32| if v == f32::NEG_INFINITY { v } else { v - lse };
    let own = logits.get(token as usize).copied().unwrap_or(f32::NEG_INFINITY);
    TokenLogprob { token, logprob: lp(own), top: top.into_iter().map(|(i, v)| (i, lp(v))).collect() }
}

/// Entries whose token ended at or before byte end, from the first unsent one, moving the sent mark past them
fn take_logprobs(lp: &[(usize, TokenLogprob)], sent: &mut usize, end: usize) -> Vec<TokenLogprob> {
    let n = lp[*sent..].iter().take_while(|(e, _)| *e <= end).count();
    let out = lp[*sent..*sent + n].iter().map(|(_, e)| e.clone()).collect();
    *sent += n;
    out
}

/// Drops the entries of tokens that began at or after byte at, keeping every entry already sent
fn trim_logprobs(lp: &mut Vec<(usize, TokenLogprob)>, sent: usize, at: usize) {
    let mut start = 0;
    let mut keep = 0;
    for (end, _) in lp.iter() {
        if start >= at {
            break;
        }
        keep += 1;
        start = *end;
    }
    lp.truncate(keep.max(sent));
}

pub enum SlotAction {
    Save,
    Restore,
}

pub enum Cmd {
    Generate(Job),
    Draft { tokens: Vec<u32>, n: usize, reply: oneshot::Sender<Result<Vec<u32>>> },
    Slot { id: usize, action: SlotAction, path: PathBuf, name: String, reply: oneshot::Sender<Result<Value>> },
    CacheStats { reply: oneshot::Sender<Vec<(usize, usize, u64, u64, u64, u64)>> },
    Profile { reply: oneshot::Sender<Option<RoutingProfile>> },
}

/// A head's link to its draft seat
pub struct DraftConn {
    pub url: String,
    pub n: usize,
}

impl DraftConn {
    /// Tokens the draft seat proposes after the context, fewer when it ends the sequence
    pub fn draft(&self, context: &[u32]) -> Result<Vec<u32>> {
        let body: Value = ureq::post(&self.url).send_json(json!({ "tokens": context, "n": self.n })).map_err(|e| anyhow!("draft seat: {e}"))?.into_json().map_err(|e| anyhow!("draft seat answer: {e}"))?;
        let tokens = body.get("tokens").and_then(Value::as_array).ok_or_else(|| anyhow!("draft seat answered without tokens"))?;
        tokens.iter().map(|t| t.as_u64().map(|t| t as u32).ok_or_else(|| anyhow!("draft seat answered with {t} for a token"))).collect()
    }
}

/// The handle every request goes through, the scheduler thread behind it
#[derive(Clone)]
pub struct Engine {
    tx: chan::Sender<Cmd>,
}

impl Engine {
    /// Starts the scheduler thread over a session and hands back its handle
    pub fn start(session: Session, tok: Arc<Tok>, draft: Option<DraftConn>, vision: Option<Arc<Clip>>, metrics: Arc<Metrics>) -> Self {
        let (tx, rx) = chan::channel();
        std::thread::Builder::new().name("kungfu-engine".into()).spawn(move || run(session, tok, draft, vision, metrics, rx)).expect("engine thread");
        Self { tx }
    }

    pub fn send(&self, cmd: Cmd) -> Result<()> {
        self.tx.send(cmd).map_err(|_| anyhow!("the engine stopped"))
    }

    pub async fn cache_stats(&self) -> Vec<(usize, usize, u64, u64, u64, u64)> {
        let (reply, rx) = oneshot::channel();
        if self.send(Cmd::CacheStats { reply }).is_err() {
            return Vec::new();
        }
        rx.await.unwrap_or_default()
    }

    pub fn profile(&self) -> Option<RoutingProfile> {
        let (reply, rx) = oneshot::channel();
        self.send(Cmd::Profile { reply }).ok()?;
        rx.blocking_recv().ok().flatten()
    }
}

/// Text that may be sent now, holding back any tail that could grow into a stop string
pub fn safe_len(text: &str, stop: &[String]) -> usize {
    let mut hold = 0;
    for s in stop {
        for n in 1..s.len() {
            if n <= text.len() && text.is_char_boundary(text.len() - n) && text.ends_with(&s[..n]) {
                hold = hold.max(n);
            }
        }
    }
    text.len() - hold
}

/// A request holding a slot, from admission to its finish
struct Active {
    job: Job,
    slot: usize,
    t0: Instant,
    prefill_done: Option<Instant>,
    cached: usize,
    fed: usize,
    logits: Option<Vec<f32>>,
    out: Vec<u32>,
    text: String,
    sent: usize,
    rng: Rng,
    pending: VecDeque<u32>,
    /// Tokens emitted and fed without sampling, the forced close of a think block
    forced: VecDeque<u32>,
    rounds: u64,
    accepted: u64,
    next: Option<u32>,
    finish: Option<&'static str>,
    stop_sequence: Option<String>,
    constraint: Option<Constraint>,
    think_open: bool,
    think_count: usize,
    /// Log probability of every token in out with the byte length of text once it was appended
    lp: Vec<(usize, TokenLogprob)>,
    /// How many lp entries went out with text pieces
    lp_sent: usize,
    /// Log probabilities of the verified tokens still waiting in pending, in their order
    lp_pending: VecDeque<TokenLogprob>,
}

struct Scheduler {
    session: Session,
    tok: Arc<Tok>,
    draft: Option<DraftConn>,
    vision: Option<Arc<Clip>>,
    metrics: Arc<Metrics>,
    waiting: VecDeque<Job>,
    active: Vec<Active>,
    /// The tokens of the think closing tag and a newline, fed when a budget runs out
    think_close: Vec<u32>,
}

fn run(session: Session, tok: Arc<Tok>, draft: Option<DraftConn>, vision: Option<Arc<Clip>>, metrics: Arc<Metrics>, rx: chan::Receiver<Cmd>) {
    let think_close = tok.encode(&format!("{THINK_CLOSE}\n"), false).unwrap_or_default();
    let mut s = Scheduler { session, tok, draft, vision, metrics, waiting: VecDeque::new(), active: Vec::new(), think_close };
    loop {
        if s.active.is_empty() && s.waiting.is_empty() {
            match rx.recv() {
                Ok(cmd) => s.handle(cmd),
                Err(_) => return,
            }
        }
        loop {
            match rx.try_recv() {
                Ok(cmd) => s.handle(cmd),
                Err(chan::TryRecvError::Empty) => break,
                Err(chan::TryRecvError::Disconnected) => return,
            }
        }
        s.admit();
        s.prefill_chunk();
        s.sample_all();
        s.step();
        s.retire();
    }
}

impl Scheduler {
    fn handle(&mut self, cmd: Cmd) {
        match cmd {
            Cmd::Generate(job) => self.waiting.push_back(job),
            Cmd::Draft { tokens, n, reply } => {
                let _ = reply.send(self.draft_tokens(&tokens, n));
            }
            Cmd::Slot { id, action, path, name, reply } => {
                let _ = reply.send(self.slot_action(id, action, &path, &name));
            }
            Cmd::CacheStats { reply } => {
                let _ = reply.send(self.session.backend_mut().expert_cache_stats());
            }
            Cmd::Profile { reply } => {
                let _ = reply.send(RoutingProfile::of(self.session.backend_ref()));
            }
        }
    }

    fn busy(&self, slot: usize) -> bool {
        self.active.iter().any(|a| a.slot == slot)
    }

    /// Proposes up to n greedy tokens after the context in slot zero, reusing the prefix it shares with the last call
    fn draft_tokens(&mut self, tokens: &[u32], n: usize) -> Result<Vec<u32>> {
        if tokens.is_empty() || n == 0 {
            bail!("a draft needs tokens and a count");
        }
        if self.busy(0) {
            bail!("slot 0 is generating");
        }
        let cached = self.session.reuse(0, tokens)?;
        let mut logits = self.session.feed(0, &tokens[cached..])?;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let next = kf_engine::sampler::argmax(&logits);
            if self.tok.is_eos(next) {
                break;
            }
            out.push(next);
            if self.session.n_past(0) + 1 > self.session.n_ctx() {
                break;
            }
            logits = self.session.feed(0, &[next])?;
        }
        self.metrics.completion_tokens.fetch_add(out.len() as u64, Ordering::Relaxed);
        self.metrics.requests.fetch_add(1, Ordering::Relaxed);
        Ok(out)
    }

    /// Saves or restores one slot as a file, refused while the slot is generating
    fn slot_action(&mut self, id: usize, action: SlotAction, path: &PathBuf, name: &str) -> Result<Value> {
        if id >= self.session.n_seq() {
            bail!("this seat keeps {} slots, slot {id} does not exist", self.session.n_seq());
        }
        if self.busy(id) {
            bail!("slot {id} is generating");
        }
        match action {
            SlotAction::Save => {
                let bytes = self.session.save(id)?;
                std::fs::write(path, &bytes)?;
                Ok(json!({ "id_slot": id, "filename": name, "n_saved": self.session.n_past(id), "n_written": bytes.len() }))
            }
            SlotAction::Restore => {
                let bytes = std::fs::read(path)?;
                self.session.load(id, &bytes)?;
                Ok(json!({ "id_slot": id, "filename": name, "n_restored": self.session.n_past(id), "n_read": bytes.len() }))
            }
        }
    }

    /// Gives waiting jobs free slots, the slot sharing the longest prefix with the prompt first
    fn admit(&mut self) {
        let n_seq = self.session.n_seq();
        let mut i = 0;
        while i < self.waiting.len() {
            let job = &self.waiting[i];
            if job.cancel.load(Ordering::Relaxed) {
                let job = self.waiting.remove(i).expect("waiting job");
                let _ = job.tx.blocking_send(Piece::Done(Outcome { text: String::new(), n_prompt: job.ids.len(), cached: 0, n_out: 0, finish: "cancelled", stop_sequence: None, logprobs: Vec::new() }));
                self.metrics.cancelled.fetch_add(1, Ordering::Relaxed);
                continue;
            }
            let slot = match job.slot {
                Some(s) if s >= n_seq => {
                    let job = self.waiting.remove(i).expect("waiting job");
                    let _ = job.tx.blocking_send(Piece::Failed(format!("this seat keeps {n_seq} slots, slot {s} does not exist")));
                    continue;
                }
                Some(s) => (!self.busy(s)).then_some(s),
                None => (0..n_seq).filter(|s| !self.busy(*s)).max_by_key(|s| (self.session.common(*s, &job.ids), usize::MAX - s)),
            };
            let Some(slot) = slot else {
                i += 1;
                continue;
            };
            let mut job = self.waiting.remove(i).expect("waiting job");
            let seed = job.limits.seed;
            let t0 = Instant::now();
            let common = chunk_boundary(&job.chunks, self.session.common(slot, &job.ids));
            let reused = if common == 0 { self.session.reset(slot).map(|_| 0) } else { self.session.truncate(slot, common).map(|_| common).or_else(|_| self.session.reset(slot).map(|_| 0)) };
            match reused {
                Ok(cached) => {
                    let constraint = job.constraint.take();
                    let think_open = job.thinking_open;
                    self.active.push(Active { job, slot, t0, prefill_done: None, cached, fed: cached, logits: None, out: Vec::new(), text: String::new(), sent: 0, rng: Rng::new(seed), pending: VecDeque::new(), forced: VecDeque::new(), rounds: 0, accepted: 0, next: None, finish: None, stop_sequence: None, constraint, think_open, think_count: 0, lp: Vec::new(), lp_sent: 0, lp_pending: VecDeque::new() });
                }
                Err(e) => {
                    let _ = job.tx.blocking_send(Piece::Failed(format!("{e:#}")));
                }
            }
        }
    }

    /// Feeds one batch of prompt rows for the first job still in prefill, a text chunk's next batch or a whole image
    fn prefill_chunk(&mut self) {
        let n_batch = self.session.n_batch();
        let Some(i) = self.active.iter().position(|a| a.fed < a.job.ids.len() && a.finish.is_none()) else { return };
        if self.active[i].job.cancel.load(Ordering::Relaxed) {
            self.active[i].finish = Some("cancelled");
            return;
        }
        let fed = self.active[i].fed;
        let (chunk_at, chunk_start) = {
            let mut start = 0;
            let mut found = None;
            for (k, c) in self.active[i].job.chunks.iter().enumerate() {
                let len = chunk_len(c);
                if fed < start + len {
                    found = Some((k, start));
                    break;
                }
                start += len;
            }
            match found {
                Some(f) => f,
                None => {
                    let a = &mut self.active[i];
                    let _ = a.job.tx.blocking_send(Piece::Failed(format!("prompt rows fed past its {} chunks", a.job.chunks.len())));
                    a.finish = Some("failed");
                    return;
                }
            }
        };
        let slot = self.active[i].slot;
        let total = self.active[i].job.ids.len();
        let result: Result<(usize, Option<Vec<f32>>)> = match &self.active[i].job.chunks[chunk_at] {
            Chunk::Text(ids) => {
                let off = fed - chunk_start;
                let end = (off + n_batch).min(ids.len());
                self.session.feed(slot, &ids[off..end]).map(|l| (chunk_start + end, Some(l)))
            }
            Chunk::Image(img) => {
                let keys = &self.active[i].job.ids[chunk_start..chunk_start + chunk_len(&self.active[i].job.chunks[chunk_at])];
                match &self.vision {
                    Some(clip) => clip.encode(self.session.backend_mut(), img).and_then(|enc| {
                        let n = enc.n_tokens();
                        let rope: Vec<[u32; 4]> = if clip.mrope() { (0..n).map(|t| [0, (t / enc.nx) as u32, (t % enc.nx) as u32, 0]).collect() } else { (0..n).map(|t| [t as u32, t as u32, t as u32, t as u32]).collect() };
                        let advance = if clip.mrope() { enc.nx.max(enc.ny) as u32 } else { n as u32 };
                        self.session.feed_embeds(slot, keys, &enc.embeds, &rope, clip.non_causal(), advance).map(|l| (chunk_start + n, Some(l)))
                    }),
                    None => Err(anyhow!("the prompt carries an image but no projector was loaded")),
                }
            }
        };
        let a = &mut self.active[i];
        match result {
            Ok((end, logits)) => {
                a.fed = end;
                if a.fed == total {
                    a.logits = logits;
                    a.prefill_done = Some(Instant::now());
                }
            }
            Err(e) => {
                let _ = a.job.tx.blocking_send(Piece::Failed(format!("{e:#}")));
                a.finish = Some("failed");
            }
        }
    }

    /// Picks the next token of every decoding job, from the draft seat's accepted tokens or the sampler
    fn sample_all(&mut self) {
        let n_ctx = self.session.n_ctx();
        for i in 0..self.active.len() {
            let ready = self.active[i].logits.is_some() || !self.active[i].pending.is_empty() || !self.active[i].forced.is_empty();
            if self.active[i].finish.is_some() || self.active[i].next.is_some() || !ready {
                continue;
            }
            if self.active[i].job.cancel.load(Ordering::Relaxed) {
                self.active[i].finish = Some("cancelled");
                continue;
            }
            if let Err(e) = self.pick(i) {
                let a = &mut self.active[i];
                let _ = a.job.tx.blocking_send(Piece::Failed(format!("{e:#}")));
                a.finish = Some("failed");
                continue;
            }
            let a = &mut self.active[i];
            if a.finish.is_none() && a.next.is_some() && self.session.n_past(a.slot) + 1 > n_ctx {
                a.finish = Some("length");
                a.next = None;
            }
        }
    }

    /// One token for job i, forced from a spent think budget, verified from a draft round, or sampled
    fn pick(&mut self, i: usize) -> Result<()> {
        let slot = self.active[i].slot;
        let a = &self.active[i];
        if a.forced.is_empty() && a.pending.is_empty() && a.think_open && a.job.think_budget.is_some_and(|b| a.think_count >= b) && !self.think_close.is_empty() {
            let close = self.think_close.clone();
            self.active[i].forced.extend(close);
        }
        if self.active[i].pending.is_empty()
            && self.active[i].forced.is_empty()
            && let Some(d) = &self.draft
        {
            let a = &self.active[i];
            let mut context = text_tail(&a.job.chunks);
            context.extend_from_slice(&a.out);
            let room = a.job.limits.max_tokens - a.out.len();
            let drafted: Vec<u32> = d.draft(&context)?.into_iter().take(room.saturating_sub(1).max(1)).collect();
            let drafted = if self.session.n_past(slot) + drafted.len() + 1 > self.session.n_ctx() { Vec::new() } else { drafted };
            let a = &mut self.active[i];
            a.rounds += 1;
            if !drafted.is_empty() {
                let tok = self.tok.clone();
                let eos = tok.eos.first().copied().unwrap_or(0);
                let mut con = a.constraint.clone();
                let first = a.out.is_empty();
                let mut seen = 0usize;
                let want = a.job.logprobs;
                let mut lps: Vec<TokenLogprob> = Vec::new();
                let (sampling, bias, rng) = (&a.job.sampling, &a.job.limits.bias, &mut a.rng);
                let mut chooser = |l: &[f32], recent: &[u32]| -> Result<u32> {
                    let t = choose(l, sampling, recent, rng, bias, &mut con, &tok, first && seen == 0)?.unwrap_or(eos);
                    if let Some(n) = want {
                        lps.push(logprob_entry(l, t, n));
                    }
                    seen += 1;
                    Ok(t)
                };
                let accepted = kf_engine::verify(&mut self.session, slot, &drafted, &mut chooser)?;
                a.accepted += accepted.len() as u64 - 1;
                if want.is_some() {
                    ensure!(lps.len() == accepted.len(), "{} log probabilities for {} verified tokens", lps.len(), accepted.len());
                    a.lp_pending.extend(lps);
                }
                a.pending.extend(accepted);
                a.constraint = con;
            }
        }
        let a = &mut self.active[i];
        let first = a.out.is_empty();
        let want = a.job.logprobs;
        let mut entry = None;
        let next = if let Some(t) = a.forced.pop_front() {
            if let Some(con) = &mut a.constraint {
                let Some(next) = con.try_advance(self.tok.piece(t, first)) else { bail!("the forced end of reasoning does not fit the grammar") };
                *con = next;
            }
            if let Some(n) = want {
                let l = a.logits.as_deref().ok_or_else(|| anyhow!("no logits for the forced token"))?;
                entry = Some(logprob_entry(l, t, n));
            }
            Some(t)
        } else if let Some(t) = a.pending.pop_front() {
            if want.is_some() {
                entry = Some(a.lp_pending.pop_front().ok_or_else(|| anyhow!("no log probability for the verified token"))?);
            }
            Some(t)
        } else {
            let l = a.logits.take().ok_or_else(|| anyhow!("no logits to sample from"))?;
            let t = choose(&l, &a.job.sampling, &a.out, &mut a.rng, &a.job.limits.bias, &mut a.constraint, &self.tok, first)?;
            if let (Some(n), Some(t)) = (want, t) {
                entry = Some(logprob_entry(&l, t, n));
            }
            t
        };
        a.logits = None;
        let Some(next) = next else {
            a.finish = Some("stop");
            return Ok(());
        };
        if self.tok.is_eos(next) {
            a.finish = Some("stop");
            return Ok(());
        }
        a.out.push(next);
        a.text = self.tok.decode(&a.out)?;
        if let Some(e) = entry {
            a.lp.push((a.text.len(), e));
        }
        let opened = a.text.rfind(THINK_OPEN).map_or(a.job.thinking_open, |o| a.text[o..].rfind(THINK_CLOSE).is_none());
        let closed = a.text.rfind(THINK_CLOSE).is_some_and(|c| a.text[c..].rfind(THINK_OPEN).is_none());
        a.think_open = opened && !closed;
        if a.think_open {
            a.think_count += 1;
        }
        if let Some((at, st)) = a.job.limits.stop.iter().filter_map(|st| a.text.find(st.as_str()).map(|i| (i, st))).min_by_key(|(i, _)| *i) {
            a.stop_sequence = Some(st.clone());
            a.text.truncate(at);
            trim_logprobs(&mut a.lp, a.lp_sent, at);
            a.finish = Some("stop");
            return Ok(());
        }
        let safe = safe_len(&a.text, &a.job.limits.stop);
        if safe > a.sent {
            let lps = take_logprobs(&a.lp, &mut a.lp_sent, safe);
            if a.job.tx.blocking_send(Piece::Text(a.text[a.sent..safe].to_string(), lps)).is_err() {
                a.finish = Some("cancelled");
                return Ok(());
            }
            a.sent = safe;
        }
        if a.out.len() >= a.job.limits.max_tokens {
            a.finish = Some("length");
            return Ok(());
        }
        if a.pending.is_empty() {
            a.next = Some(next);
        }
        Ok(())
    }

    /// Feeds every job's next token in one pass and hands each its logits
    fn step(&mut self) {
        let batch: Vec<(usize, u32)> = self.active.iter().filter(|a| a.finish.is_none()).filter_map(|a| a.next.map(|t| (a.slot, t))).collect();
        if batch.is_empty() {
            return;
        }
        match self.session.step(&batch) {
            Ok(rows) => {
                let vocab = rows.len() / batch.len();
                for (k, (slot, _)) in batch.iter().enumerate() {
                    if let Some(a) = self.active.iter_mut().find(|a| a.slot == *slot) {
                        a.logits = Some(rows[k * vocab..(k + 1) * vocab].to_vec());
                        a.next = None;
                    }
                }
            }
            Err(e) => {
                for a in self.active.iter_mut().filter(|a| a.next.is_some()) {
                    let _ = a.job.tx.blocking_send(Piece::Failed(format!("{e:#}")));
                    a.finish = Some("failed");
                }
            }
        }
    }

    /// Reports every finished job, logs its request event, and frees its slot
    fn retire(&mut self) {
        let mut i = 0;
        while i < self.active.len() {
            let Some(finish) = self.active[i].finish else {
                i += 1;
                continue;
            };
            let a = self.active.remove(i);
            if finish == "failed" {
                continue;
            }
            if !a.pending.is_empty() {
                let keep = a.job.ids.len() + a.out.len();
                if self.session.n_past(a.slot) > keep {
                    let _ = self.session.truncate(a.slot, keep);
                }
            }
            if a.text.len() > a.sent || a.lp.len() > a.lp_sent {
                let rest = a.lp[a.lp_sent..].iter().map(|(_, e)| e.clone()).collect();
                let _ = a.job.tx.blocking_send(Piece::Text(a.text[a.sent..].to_string(), rest));
            }
            let n_prompt = a.job.ids.len();
            let n_out = a.out.len();
            let now = Instant::now();
            let prefill = a.prefill_done.unwrap_or(now).duration_since(a.t0).as_secs_f64();
            let decode = a.prefill_done.map_or(0.0, |p| now.duration_since(p).as_secs_f64());
            let prefill_tps = if prefill > 0.0 { (n_prompt - a.cached) as f64 / prefill } else { 0.0 };
            let decode_tps = if decode > 0.0 { n_out as f64 / decode } else { 0.0 };
            let mut kv = vec![("prompt_tokens", n_prompt.to_string()), ("cached_tokens", a.cached.to_string()), ("completion_tokens", n_out.to_string()), ("prefill_tps", format!("{prefill_tps:.1}")), ("decode_tps", format!("{decode_tps:.1}")), ("finish", finish.to_string()), ("slot", a.slot.to_string())];
            if self.draft.is_some() {
                kv.push(("draft_rounds", a.rounds.to_string()));
                kv.push(("draft_accepted", a.accepted.to_string()));
            }
            event::emit("request", &kv);
            self.metrics.prompt_tokens.fetch_add(n_prompt as u64, Ordering::Relaxed);
            self.metrics.cached_tokens.fetch_add(a.cached as u64, Ordering::Relaxed);
            self.metrics.completion_tokens.fetch_add(n_out as u64, Ordering::Relaxed);
            if finish == "cancelled" {
                self.metrics.cancelled.fetch_add(1, Ordering::Relaxed);
            }
            let logprobs = a.lp.into_iter().map(|(_, e)| e).collect();
            let _ = a.job.tx.blocking_send(Piece::Done(Outcome { text: a.text, n_prompt, cached: a.cached, n_out, finish, stop_sequence: a.stop_sequence, logprobs }));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logprob_entry_is_log_softmax_with_ordered_top() {
        let logits = [1.0f32, 3.0, 2.0, 3.0, f32::NEG_INFINITY];
        let e = logprob_entry(&logits, 2, 3);
        let lse = (1f32.exp() + 3f32.exp() + 2f32.exp() + 3f32.exp()).ln();
        assert!((e.logprob - (2.0 - lse)).abs() < 1e-6);
        assert_eq!(e.top.iter().map(|(i, _)| *i).collect::<Vec<_>>(), vec![1, 3, 2]);
        assert!((e.top[0].1 - (3.0 - lse)).abs() < 1e-6);
        assert_eq!(logprob_entry(&logits, 4, 0).logprob, f32::NEG_INFINITY);
        assert!(logprob_entry(&logits, 4, 0).top.is_empty());
    }

    #[test]
    fn logprobs_follow_text_pieces_and_stop_trimming() {
        let e = |t: u32| TokenLogprob { token: t, logprob: 0.0, top: Vec::new() };
        let mut lp = vec![(2, e(0)), (5, e(1)), (5, e(2)), (9, e(3))];
        let mut sent = 0;
        assert_eq!(take_logprobs(&lp, &mut sent, 4).len(), 1);
        assert_eq!(take_logprobs(&lp, &mut sent, 5).len(), 2);
        assert_eq!(sent, 3);
        trim_logprobs(&mut lp, sent, 7);
        assert_eq!(lp.len(), 4);
        trim_logprobs(&mut lp, sent, 5);
        assert_eq!(lp.len(), 3);
        let mut lp2 = vec![(2, e(0)), (5, e(1))];
        trim_logprobs(&mut lp2, 0, 0);
        assert!(lp2.is_empty());
    }
}

/// Rows one chunk takes in the prompt, an image as many as its tokens
pub fn chunk_len(c: &Chunk) -> usize {
    match c {
        Chunk::Text(ids) => ids.len(),
        Chunk::Image(img) => image_tokens(img),
    }
}

/// Tokens an image stands for after its tower, the merged grid
pub fn image_tokens(img: &ImageInput) -> usize {
    img.n_merged_x() * img.n_merged_y()
}

/// The largest row count at or below n that does not end inside an image chunk
fn chunk_boundary(chunks: &[Chunk], n: usize) -> usize {
    let mut start = 0;
    for c in chunks {
        let len = chunk_len(c);
        if n < start + len {
            return if matches!(c, Chunk::Image(_)) { start } else { n };
        }
        start += len;
    }
    n
}

/// The tokens after the last image, what a draft seat without eyes can continue
fn text_tail(chunks: &[Chunk]) -> Vec<u32> {
    let mut out = Vec::new();
    for c in chunks {
        match c {
            Chunk::Text(ids) => out.extend_from_slice(ids),
            Chunk::Image(_) => out.clear(),
        }
    }
    out
}

//! The scheduler thread that owns the session, admitting requests to slots and decoding every active slot in one pass
use crate::api::Metrics;
use crate::grammar::Constraint;
use crate::head::{Head, Source};
use crate::tokenizer::Tok;
use kf_vision::{Clip, ImageInput};
use anyhow::{Result, anyhow, bail, ensure};
use kf_core::event;
use kf_engine::{Candidates, NGRAM_TOPS, Ngram, Rng, RoutingProfile, Sampling, Session, Tree, sample, sample_sparse};
use serde_json::{Value, json};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, mpsc as chan};
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

/// What the sampler picked from one row, with the whole row when the pick needed it
pub struct Pick {
    pub token: Option<u32>,
    pub dense: Option<Vec<f32>>,
}

/// One token under every rule a request adds, the bias, the grammar, and the end of sequence, from the row's
/// candidates while they settle every pick and from the whole row once they do not, none when the grammar is whole
/// and nothing may follow
#[allow(clippy::too_many_arguments)]
pub fn choose(c: &Candidates, dense: &mut dyn FnMut() -> Result<Vec<f32>>, sampling: &Sampling, recent: &[u32], rng: &mut Rng, bias: &[(u32, f32)], constraint: &mut Option<Constraint>, tok: &Tok, first: bool) -> Result<Pick> {
    let mut cands = c.clone();
    let mut sparse = bias.iter().all(|(id, b)| cands.bias(*id, *b));
    let mut masked: Vec<u32> = Vec::new();
    let mut row: Option<Vec<f32>> = None;
    if let Some(con) = constraint
        && con.complete()
        && !con.can_continue()
    {
        return Ok(Pick { token: None, dense: None });
    }
    loop {
        let t = match if sparse { sample_sparse(&cands, sampling, recent, rng) } else { None } {
            Some(t) => t,
            None => {
                sparse = false;
                let l = match &mut row {
                    Some(l) => l,
                    None => {
                        let mut l = dense()?;
                        for (id, b) in bias {
                            if let Some(v) = l.get_mut(*id as usize) {
                                *v += b;
                            }
                        }
                        for id in &masked {
                            if let Some(v) = l.get_mut(*id as usize) {
                                *v = f32::NEG_INFINITY;
                            }
                        }
                        row.insert(l)
                    }
                };
                if masked.len() >= l.len() {
                    break;
                }
                let t = sample(l, sampling, recent, rng);
                if l[t as usize] == f32::NEG_INFINITY {
                    break;
                }
                t
            }
        };
        let Some(con) = constraint else { return Ok(Pick { token: Some(t), dense: row }) };
        if tok.is_eos(t) {
            if con.complete() {
                return Ok(Pick { token: None, dense: row });
            }
        } else if let Some(next) = con.try_advance(tok.piece(t, first)) {
            *con = next;
            return Ok(Pick { token: Some(t), dense: row });
        }
        cands.mask(t);
        masked.push(t);
        if let Some(l) = &mut row {
            l[t as usize] = f32::NEG_INFINITY;
        }
    }
    if constraint.as_ref().is_some_and(|con| con.complete()) {
        return Ok(Pick { token: None, dense: row });
    }
    bail!("no token of the vocabulary continues the grammar")
}

/// The log probability of one token under the row with the n_top most likely tokens, from the candidates when they
/// hold the token and the top, else from the whole row, fetched when the pick did not already have it
pub fn logprob_entry(c: &Candidates, dense: Option<&[f32]>, token: u32, n_top: usize, fetch: &mut dyn FnMut() -> Result<Vec<f32>>) -> Result<TokenLogprob> {
    if let Some(l) = dense {
        return Ok(logprob_dense(l, token, n_top));
    }
    if let Some(own) = c.logprob(token)
        && n_top <= c.len()
    {
        let top = c.ids[..n_top].iter().zip(&c.logits[..n_top]).map(|(i, v)| (*i, if *v == f32::NEG_INFINITY { *v } else { *v - c.lse })).collect();
        return Ok(TokenLogprob { token, logprob: own, top });
    }
    Ok(logprob_dense(&fetch()?, token, n_top))
}

/// The log probability of one token under the raw logits with the n_top most likely tokens, ties broken by lower id
pub fn logprob_dense(logits: &[f32], token: u32, n_top: usize) -> TokenLogprob {
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
    CacheStats { reply: oneshot::Sender<Vec<(usize, usize, u64, u64)>> },
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
    tx: Arc<Mutex<Option<chan::Sender<Cmd>>>>,
    handle: Arc<Mutex<Option<std::thread::JoinHandle<()>>>>,
}

impl Engine {
    /// Starts the scheduler thread over a session and hands back its handle
    pub fn start(session: Session, tok: Arc<Tok>, draft: Option<DraftConn>, spec: Option<(Ngram, usize)>, head: Option<Head>, vision: Option<Arc<Clip>>, metrics: Arc<Metrics>) -> Self {
        let (tx, rx) = chan::channel();
        let handle = std::thread::Builder::new().name("kungfu-engine".into()).spawn(move || run(session, tok, draft, spec, head, vision, metrics, rx)).expect("engine thread");
        Self { tx: Arc::new(Mutex::new(Some(tx))), handle: Arc::new(Mutex::new(Some(handle))) }
    }

    pub fn send(&self, cmd: Cmd) -> Result<()> {
        let tx = self.tx.lock().map_err(|_| anyhow!("the engine lock is poisoned"))?;
        tx.as_ref().ok_or_else(|| anyhow!("the engine stopped"))?.send(cmd).map_err(|_| anyhow!("the engine stopped"))
    }

    /// Closes the command channel and waits for the scheduler thread, which releases the session and its backend
    pub fn stop(&self) {
        if let Ok(mut tx) = self.tx.lock() {
            tx.take();
        }
        if let Some(h) = self.handle.lock().ok().and_then(|mut h| h.take()) {
            let _ = h.join();
        }
    }

    pub async fn cache_stats(&self) -> Vec<(usize, usize, u64, u64)> {
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
    logits: Option<Candidates>,
    /// What the last row settled before any further pass, see settle_all
    pre: Option<Pre>,
    out: Vec<u32>,
    text: String,
    sent: usize,
    rng: Rng,
    pending: VecDeque<u32>,
    /// Tokens emitted and fed without sampling, the forced close of a think block
    forced: VecDeque<u32>,
    rounds: u64,
    accepted: u64,
    /// Rounds whose first drafted token the model chose, each verified in one pass of the drafted rows
    passes: u64,
    /// Rounds the learned head drafted
    head_rounds: u64,
    /// Tokens the ngram drafter proposes next round, grown while rounds accept everything and shrunk when they accept nothing
    spec_len: usize,
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

/// What a job's last row of logits decided while the row was still on the backend
enum Pre {
    /// The sampler's token for the next position, none when the grammar is whole, with its log probability entry
    Own(Option<u32>, Option<TokenLogprob>),
    /// The log probability entry of the forced token at the front
    Forced(Option<TokenLogprob>),
}

struct Scheduler {
    session: Session,
    tok: Arc<Tok>,
    draft: Option<DraftConn>,
    /// The ngram drafter with its longest draft, when no draft seat proposes
    spec: Option<(Ngram, usize)>,
    /// The learned head's schedule and measures, when the backend trains one
    head: Option<Head>,
    vision: Option<Arc<Clip>>,
    metrics: Arc<Metrics>,
    waiting: VecDeque<Job>,
    active: Vec<Active>,
    /// The tokens of the think closing tag and a newline, fed when a budget runs out
    think_close: Vec<u32>,
    /// Seconds one decode pass takes, a running average
    decode_secs: f32,
}

#[allow(clippy::too_many_arguments)]
fn run(session: Session, tok: Arc<Tok>, draft: Option<DraftConn>, spec: Option<(Ngram, usize)>, head: Option<Head>, vision: Option<Arc<Clip>>, metrics: Arc<Metrics>, rx: chan::Receiver<Cmd>) {
    let think_close = tok.encode(&format!("{THINK_CLOSE}\n"), false).unwrap_or_default();
    let mut s = Scheduler { session, tok, draft, spec, head, vision, metrics, waiting: VecDeque::new(), active: Vec::new(), think_close, decode_secs: 0.0 };
    loop {
        if s.active.is_empty() && s.waiting.is_empty() {
            // Idle, the head trains on the rows it has earned between commands, then the thread waits for one
            let cmd = if s.train(true) { rx.try_recv().map_err(|e| e == chan::TryRecvError::Disconnected) } else { rx.recv().map_err(|_| true) };
            match cmd {
                Ok(cmd) => s.handle(cmd),
                Err(true) => {
                    s.park();
                    return;
                }
                Err(false) => continue,
            }
        }
        loop {
            match rx.try_recv() {
                Ok(cmd) => s.handle(cmd),
                Err(chan::TryRecvError::Empty) => break,
                Err(chan::TryRecvError::Disconnected) => {
                    s.park();
                    return;
                }
            }
        }
        s.admit();
        s.prefill_chunk();
        s.settle_all();
        s.sample_all();
        s.step();
        s.train(false);
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
            let next = logits.argmax();
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
                    let spec_len = match &mut self.spec {
                        Some((ng, max)) => {
                            ng.observe(&text_tail(&job.chunks));
                            *max
                        }
                        None => 0,
                    };
                    self.active.push(Active { job, slot, t0, prefill_done: None, cached, fed: cached, logits: None, pre: None, out: Vec::new(), text: String::new(), sent: 0, rng: Rng::new(seed), pending: VecDeque::new(), forced: VecDeque::new(), rounds: 0, accepted: 0, passes: 0, head_rounds: 0, spec_len, next: None, finish: None, stop_sequence: None, constraint, think_open, think_count: 0, lp: Vec::new(), lp_sent: 0, lp_pending: VecDeque::new() });
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
        let result: Result<(usize, Option<Candidates>)> = match &self.active[i].job.chunks[chunk_at] {
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
                let done = a.fed == total;
                if done {
                    a.logits = logits;
                    a.prefill_done = Some(Instant::now());
                }
                // The prompt's last row is what the head drafts the first round from
                if done {
                    self.hold(slot, 0);
                }
            }
            Err(e) => {
                let _ = a.job.tx.blocking_send(Piece::Failed(format!("{e:#}")));
                a.finish = Some("failed");
            }
        }
    }

    /// Picks the next token of every decoding job, from the draft seat's accepted tokens or the sampler
    /// Decides from every ready job's last row what only that row can decide, the sampler's own token or the forced
    /// token's log probability, while the row is still on the backend, before any job's verify pass replaces it
    fn settle_all(&mut self) {
        for i in 0..self.active.len() {
            let a = &self.active[i];
            if a.finish.is_some() || a.next.is_some() || a.pre.is_some() || a.job.cancel.load(Ordering::Relaxed) {
                continue;
            }
            if a.forced.is_empty() && a.pending.is_empty() && a.think_open && a.job.think_budget.is_some_and(|b| a.think_count >= b) && !self.think_close.is_empty() {
                let close = self.think_close.clone();
                self.active[i].forced.extend(close);
            }
            if !self.active[i].pending.is_empty() {
                continue;
            }
            if let Err(e) = self.settle(i) {
                let a = &mut self.active[i];
                let _ = a.job.tx.blocking_send(Piece::Failed(format!("{e:#}")));
                a.finish = Some("failed");
            }
        }
    }

    /// One job's decision from its last row, the row consumed
    fn settle(&mut self, i: usize) -> Result<()> {
        let a = &mut self.active[i];
        let want = a.job.logprobs;
        let Some(c) = a.logits.take() else {
            if !a.forced.is_empty() {
                ensure!(want.is_none(), "no logits for the forced token");
                a.pre = Some(Pre::Forced(None));
            }
            return Ok(());
        };
        let session = &mut self.session;
        let mut fetch = || session.dense_row(c.row);
        if let Some(&t) = a.forced.front() {
            let entry = want.map(|n| logprob_entry(&c, None, t, n, &mut fetch)).transpose()?;
            a.pre = Some(Pre::Forced(entry));
            return Ok(());
        }
        let first = a.out.is_empty();
        let pick = choose(&c, &mut fetch, &a.job.sampling, &a.out, &mut a.rng, &a.job.limits.bias, &mut a.constraint, &self.tok, first)?;
        let entry = match (want, pick.token) {
            (Some(n), Some(t)) => Some(logprob_entry(&c, pick.dense.as_deref(), t, n, &mut fetch)?),
            _ => None,
        };
        if let Some((ng, _)) = &mut self.spec {
            let last = a.out.last().copied().or_else(|| text_tail(&a.job.chunks).last().copied());
            if let Some(last) = last {
                ng.recycle(last, &c.ids[..NGRAM_TOPS.min(c.len())]);
            }
        }
        a.pre = Some(Pre::Own(pick.token, entry));
        Ok(())
    }

    fn sample_all(&mut self) {
        let n_ctx = self.session.n_ctx();
        for i in 0..self.active.len() {
            let ready = self.active[i].pre.is_some() || !self.active[i].pending.is_empty();
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

    /// The tree drafted after the job's context from its own token and who drafted it: the draft seat as a chain, the
    /// learned head with the context's repeats grafted in, or the ngram drafter, the head's measure choosing between
    /// the last two, within the request's room, the root alone when nothing fits
    fn drafted(&mut self, i: usize, own: u32) -> Result<(Tree, Source)> {
        let (slot, spec_len, room) = {
            let a = &self.active[i];
            (a.slot, a.spec_len, (a.job.limits.max_tokens - a.out.len()).saturating_sub(1).max(1))
        };
        let mut context = text_tail(&self.active[i].job.chunks);
        context.extend_from_slice(&self.active[i].out);
        let mut source = match (&self.draft, &self.spec, &mut self.head) {
            (Some(_), _, _) => Source::Seat,
            (None, spec, Some(h)) => h.source(spec.is_some(), self.decode_secs),
            (None, Some(_), None) => Source::Ngram,
            (None, None, None) => Source::None,
        };
        let mut tree = match source {
            Source::Seat => {
                let d = self.draft.as_ref().ok_or_else(|| anyhow!("no draft seat"))?;
                context.push(own);
                let mut tree = Tree::root(own);
                let mut parent = 0;
                for t in d.draft(&context)? {
                    match tree.push(parent, t) {
                        Some(row) => parent = row,
                        None => break,
                    }
                }
                tree
            }
            Source::Head => {
                let rows = self.head.as_ref().map_or(0, |h| h.rows);
                let backend = self.session.backend_mut();
                backend.drafter_sync(slot)?;
                let mut tree = backend.drafter_draft(slot, own, rows)?;
                match &self.spec {
                    // A slot the head holds no row for yet leaves the round to the ngram drafter
                    Some((ng, _)) if tree.len() == 1 => {
                        source = Source::Ngram;
                        tree = ng.draft(&context, own, spec_len + 1);
                    }
                    Some((ng, _)) => {
                        context.push(own);
                        tree.graft(&ng.chain(&context, spec_len), rows + spec_len);
                    }
                    None => {}
                }
                tree
            }
            Source::Ngram => self.spec.as_ref().map_or_else(|| Tree::root(own), |(ng, _)| ng.draft(&context, own, spec_len + 1)),
            Source::None => Tree::root(own),
        };
        if tree.len() == 1 {
            source = Source::None;
        }
        // Rows past the room, the context, or the batch go, children before parents never since rows are pushed in order
        let limit = room.min(self.session.n_ctx().saturating_sub(self.session.n_past(slot))).min(self.session.n_batch());
        if tree.len() > limit {
            tree.tokens.truncate(limit.max(1));
            tree.parents.truncate(limit.max(1));
        }
        Ok((tree, source))
    }

    /// One token for job i, forced from a spent think budget, verified from a draft round, or the sampler's own
    fn pick(&mut self, i: usize) -> Result<()> {
        let slot = self.active[i].slot;
        let pre = self.active[i].pre.take();
        if let Some(Pre::Own(Some(own), entry)) = &pre
            && self.active[i].pending.is_empty()
            && (self.draft.is_some() || self.spec.is_some() || self.head.is_some())
        {
            let own = *own;
            let t0 = Instant::now();
            let (drafted, source) = self.drafted(i, own)?;
            let n_drafted = drafted.len() - 1;
            let a = &mut self.active[i];
            a.rounds += 1;
            if n_drafted > 0 {
                let tok = self.tok.clone();
                let eos = tok.eos.first().copied().unwrap_or(0);
                let mut con = a.constraint.clone();
                let want = a.job.logprobs;
                let mut lps: Vec<TokenLogprob> = entry.iter().cloned().collect();
                let mut tops: Vec<(u32, Vec<u32>)> = Vec::new();
                let (sampling, bias, rng) = (&a.job.sampling, &a.job.limits.bias, &mut a.rng);
                let mut chooser = |c: &Candidates, recent: &[u32], dense: &mut dyn FnMut() -> Result<Vec<f32>>| -> Result<u32> {
                    let pick = choose(c, dense, sampling, recent, rng, bias, &mut con, &tok, false)?;
                    let t = pick.token.unwrap_or(eos);
                    if let Some(n) = want {
                        lps.push(logprob_entry(c, pick.dense.as_deref(), t, n, dense)?);
                    }
                    if let Some(&last) = recent.last() {
                        tops.push((last, c.ids[..NGRAM_TOPS.min(c.len())].to_vec()));
                    }
                    Ok(t)
                };
                let accepted = kf_engine::verify_tree(&mut self.session, slot, &drafted, &mut chooser)?;
                let taken = accepted.len() - 1;
                a.accepted += taken as u64;
                a.passes += (taken > 0) as u64;
                a.head_rounds += (source == Source::Head) as u64;
                if want.is_some() {
                    ensure!(lps.len() == accepted.len(), "{} log probabilities for {} verified tokens", lps.len(), accepted.len());
                    a.lp_pending.extend(lps);
                }
                if let Some((ng, max)) = &mut self.spec {
                    // The drafter learns the verified tokens in their context and the model's favourites after each, and
                    // its draft length follows the acceptance of the rounds it drafted
                    for (last, top) in &tops {
                        ng.recycle(*last, top);
                    }
                    let mut context = text_tail(&a.job.chunks);
                    context.extend_from_slice(&a.out);
                    context.extend_from_slice(&accepted);
                    let from = context.len().saturating_sub(accepted.len() + 3);
                    ng.observe(&context[from..]);
                    if source == Source::Ngram {
                        a.spec_len = if taken == n_drafted { (a.spec_len + 1).min(*max) } else if taken == 0 { 1 } else { taken.max(1) };
                    }
                }
                a.pending.extend(accepted);
                a.constraint = con;
                // The round's yield over its time, the decode pass that gave own included, measures its drafter
                let secs = self.decode_secs + t0.elapsed().as_secs_f32();
                if let Some(h) = self.head.as_mut() {
                    h.observe(source, taken, secs);
                    let (head, ngram) = h.rates();
                    self.metrics.rate_head.store(head.to_bits(), Ordering::Relaxed);
                    self.metrics.rate_ngram.store(ngram.to_bits(), Ordering::Relaxed);
                }
                if let Some(k) = source.index() {
                    self.metrics.draft_rounds[k].fetch_add(1, Ordering::Relaxed);
                    self.metrics.draft_accepted[k].fetch_add(taken as u64, Ordering::Relaxed);
                }
            } else {
                if let Some(e) = entry {
                    a.lp_pending.push_back(e.clone());
                }
                a.pending.push_back(own);
            }
        }
        let a = &mut self.active[i];
        let first = a.out.is_empty();
        let want = a.job.logprobs;
        let mut entry = None;
        let next = match pre {
            Some(Pre::Forced(e)) => {
                let t = a.forced.pop_front().ok_or_else(|| anyhow!("a forced token settled with none in front"))?;
                if let Some(con) = &mut a.constraint {
                    let Some(next) = con.try_advance(self.tok.piece(t, first)) else { bail!("the forced end of reasoning does not fit the grammar") };
                    *con = next;
                }
                entry = e;
                Some(t)
            }
            _ if !a.pending.is_empty() => {
                let t = a.pending.pop_front().ok_or_else(|| anyhow!("no verified token in front"))?;
                if want.is_some() {
                    entry = Some(a.lp_pending.pop_front().ok_or_else(|| anyhow!("no log probability for the verified token"))?);
                }
                Some(t)
            }
            Some(Pre::Own(t, e)) => {
                entry = e;
                t
            }
            None => bail!("a job picked with nothing settled and nothing pending"),
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
        if let Some((ng, _)) = &mut self.spec
            && a.pending.is_empty()
        {
            let mut tail = text_tail(&a.job.chunks);
            tail.extend_from_slice(&a.out);
            let from = tail.len().saturating_sub(4);
            ng.observe(&tail[from..]);
        }
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

    /// Feeds every job's next token in one pass and hands each its logits, the head holding every row for its draft
    fn step(&mut self) {
        let batch: Vec<(usize, u32)> = self.active.iter().filter(|a| a.finish.is_none()).filter_map(|a| a.next.map(|t| (a.slot, t))).collect();
        if batch.is_empty() {
            return;
        }
        let t0 = Instant::now();
        match self.session.step(&batch) {
            Ok(rows) => {
                let secs = t0.elapsed().as_secs_f32();
                self.decode_secs = if self.decode_secs == 0.0 { secs } else { self.decode_secs + (secs - self.decode_secs) / 16.0 };
                for (i, ((slot, _), row)) in batch.iter().zip(rows).enumerate() {
                    self.hold(*slot, i);
                    if let Some(a) = self.active.iter_mut().find(|a| a.slot == *slot) {
                        a.logits = Some(row);
                        a.next = None;
                    }
                }
                if let Some(h) = self.head.as_mut() {
                    h.passed();
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

    /// Keeps a slot's hidden row of the pass just run for the head's next draft from that slot
    fn hold(&mut self, slot: usize, row: usize) {
        if self.head.is_none() {
            return;
        }
        let pos = self.session.n_past(slot).saturating_sub(1) as u32;
        if let Err(e) = self.session.backend_mut().drafter_hold(slot, row, pos) {
            event::error("drafter", &[("detail", format!("{e:#}"))]);
        }
    }

    /// One training step of the head when one is due, with the vocabulary recount and the save it brings, true when
    /// a step ran
    fn train(&mut self, idle: bool) -> bool {
        let Some(h) = self.head.as_mut() else { return false };
        h.earn(self.session.backend_ref().drafter_rows());
        if !h.due(idle, self.decode_secs) {
            return false;
        }
        let t0 = Instant::now();
        let backend = self.session.backend_mut();
        let step = match backend.drafter_step() {
            Ok(Some(step)) => step,
            Ok(None) => return false,
            Err(e) => {
                event::error("drafter", &[("detail", format!("{e:#}"))]);
                self.head = None;
                return false;
            }
        };
        h.trained(step, t0.elapsed().as_secs_f32());
        self.metrics.head_steps.store(h.steps, Ordering::Relaxed);
        self.metrics.head_loss.store(h.loss.to_bits(), Ordering::Relaxed);
        self.metrics.head_hit.store(h.hit.to_bits(), Ordering::Relaxed);
        if h.vocab_due()
            && let Err(e) = backend.drafter_refresh_vocab()
        {
            event::error("drafter", &[("detail", format!("{e:#}"))]);
        }
        if h.save_due() {
            match h.save(backend) {
                Ok(()) => event::emit("drafter", &[("saved_steps", h.steps.to_string()), ("loss", format!("{:.4}", h.loss)), ("hit", format!("{:.3}", h.hit))]),
                Err(e) => event::error("drafter", &[("detail", format!("{e:#}"))]),
            }
        }
        true
    }

    /// Saves the head as the scheduler ends, so the next run starts where this one left off
    fn park(&mut self) {
        let Some(h) = self.head.as_mut() else { return };
        if !h.unsaved() {
            return;
        }
        match h.save(self.session.backend_mut()) {
            Ok(()) => event::emit("drafter", &[("saved_steps", h.steps.to_string()), ("loss", format!("{:.4}", h.loss)), ("hit", format!("{:.3}", h.hit))]),
            Err(e) => event::error("drafter", &[("detail", format!("{e:#}"))]),
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
            if self.draft.is_some() || self.spec.is_some() || self.head.is_some() {
                kv.push(("draft_rounds", a.rounds.to_string()));
                kv.push(("draft_passes", a.passes.to_string()));
                kv.push(("draft_accepted", a.accepted.to_string()));
            }
            if self.head.is_some() {
                kv.push(("head_rounds", a.head_rounds.to_string()));
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
        let e = logprob_dense(&logits, 2, 3);
        let lse = (1f32.exp() + 3f32.exp() + 2f32.exp() + 3f32.exp()).ln();
        assert!((e.logprob - (2.0 - lse)).abs() < 1e-6);
        assert_eq!(e.top.iter().map(|(i, _)| *i).collect::<Vec<_>>(), vec![1, 3, 2]);
        assert!((e.top[0].1 - (3.0 - lse)).abs() < 1e-6);
        assert_eq!(logprob_dense(&logits, 4, 0).logprob, f32::NEG_INFINITY);
        assert!(logprob_dense(&logits, 4, 0).top.is_empty());
        // The candidates answer for a token they hold and fetch the row for one they do not
        let c = Candidates::select(&logits, 3, 0).expect("candidates");
        let fetched = std::cell::Cell::new(0);
        let mut fetch = || {
            fetched.set(fetched.get() + 1);
            Ok(logits.to_vec())
        };
        let s = logprob_entry(&c, None, 2, 3, &mut fetch).expect("sparse");
        assert_eq!((s.token, s.top.iter().map(|(i, _)| *i).collect::<Vec<_>>()), (e.token, e.top.iter().map(|(i, _)| *i).collect::<Vec<_>>()));
        assert!((s.logprob - e.logprob).abs() < 1e-5);
        assert_eq!(fetched.get(), 0);
        let d = logprob_entry(&c, None, 0, 3, &mut fetch).expect("dense");
        assert_eq!((fetched.get(), d.logprob), (1, logprob_dense(&logits, 0, 3).logprob));
        let d = logprob_entry(&c, None, 2, 4, &mut fetch).expect("dense top");
        assert_eq!((fetched.get(), d.top.len()), (2, 4));
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

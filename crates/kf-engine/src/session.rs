use crate::loader::{Loaded, Store};
use crate::stage::StageConn;
use anyhow::{Result, bail, ensure};
use kf_core::{Backend, Graph, GraphBuilder, Inputs, Plan, RopeKind};
use kf_layers::{Model, Segment};
use std::collections::HashMap;
use std::sync::Arc;

/// One part of a pass, a segment this process runs or a stage that runs its layers
enum Hop {
    Local(Segment),
    Remote(StageConn),
}

/// The tokens one slot has fed, the rotary position after each, and the logits of its last row
#[derive(Clone, Debug, Default)]
struct Slot {
    tokens: Vec<u32>,
    after: Vec<u32>,
    last_logits: Option<Vec<f32>>,
}

impl Slot {
    /// The rotary position the next row takes
    fn next_pos(&self) -> u32 {
        self.after.last().copied().unwrap_or(0)
    }
}

/// Rows fed to one slot in a pass, tokens through the embedding or embeddings standing in for them
pub enum Rows<'a> {
    Tokens(&'a [u32]),
    /// Embeddings with the key each row takes in the slot's history, their rotary streams, whether every row sees the whole run, and the positions the run advances
    Embeds { keys: &'a [u32], embeds: &'a [f32], rope: &'a [[u32; 4]], whole: bool, advance: u32 },
}

impl Rows<'_> {
    fn len(&self) -> usize {
        match self {
            Rows::Tokens(t) => t.len(),
            Rows::Embeds { keys, .. } => keys.len(),
        }
    }
}

/// Sequence slots against one backend, each slot's positions advancing as its rows are fed, slots decoded together in one pass
pub struct Session {
    backend: Box<dyn Backend>,
    store: Arc<Store>,
    model: Arc<Model>,
    n_ctx: usize,
    n_batch: usize,
    width: usize,
    /// Whether text positions pin every rotary stream and images take a grid, the multi stream rope of vision models
    mrope: bool,
    slots: Vec<Slot>,
    graphs: HashMap<(usize, usize, usize, bool), Arc<Graph>>,
    hops: Vec<Hop>,
}

impl Session {
    pub fn new(loaded: &Loaded, backend: Box<dyn Backend>, plan: &Plan) -> Self {
        let n_layers = loaded.model.n_layers();
        Self {
            backend,
            store: loaded.store.clone(),
            model: loaded.model.clone(),
            n_ctx: plan.n_ctx,
            n_batch: plan.n_batch.max(1),
            width: loaded.hparams.n_embd * loaded.model.streams,
            mrope: matches!(loaded.hparams.rope.kind, RopeKind::MRope | RopeKind::IMRope),
            slots: vec![Slot::default(); plan.n_seq.max(1)],
            graphs: HashMap::new(),
            hops: vec![Hop::Local(Segment { from: 0, to: n_layers, embed: true, inject: false, n_out: None })],
        }
    }

    /// A session whose layers in the stages' ranges run there, the rest here with the embedding and the head
    pub fn with_stages(loaded: &Loaded, backend: Box<dyn Backend>, plan: &Plan, mut stages: Vec<StageConn>) -> Result<Self> {
        let n_layers = loaded.model.n_layers();
        let width = loaded.hparams.n_embd * loaded.model.streams;
        stages.sort_by_key(|s| s.rank);
        let mut hops = Vec::new();
        let mut at = 0;
        for (i, s) in stages.into_iter().enumerate() {
            ensure!(s.rank == i, "stage ranks must run from 0 without gaps, rank {} at {} is out of order or missing", s.rank, s.addr);
            s.check(plan.n_ctx, plan.n_batch, plan.n_seq, width)?;
            ensure!(s.from < s.to && s.to <= n_layers, "stage {} holds {}..{} of a {n_layers} layer model", s.addr, s.from, s.to);
            ensure!(s.from >= at, "stage {} of rank {} holds {}..{} which overlaps or precedes the layers of the ranks before it", s.addr, s.rank, s.from, s.to);
            if at == 0 || s.from > at {
                hops.push(Hop::Local(Segment { from: at, to: s.from, embed: at == 0, inject: false, n_out: None }));
            }
            at = s.to;
            hops.push(Hop::Remote(s));
        }
        hops.push(Hop::Local(Segment { from: at, to: n_layers, embed: at == 0, inject: false, n_out: None }));
        if hops.len() > 1 && loaded.model.hands_off() {
            bail!("the model hands tensors between blocks, so no stage can hold a range of its layers");
        }
        let mut s = Self::new(loaded, backend, plan);
        s.hops = hops;
        Ok(s)
    }

    pub fn n_seq(&self) -> usize {
        self.slots.len()
    }

    pub fn n_ctx(&self) -> usize {
        self.n_ctx
    }

    pub fn n_batch(&self) -> usize {
        self.n_batch
    }

    pub fn mrope(&self) -> bool {
        self.mrope
    }

    pub fn n_past(&self, seq: usize) -> usize {
        self.slots.get(seq).map_or(0, |s| s.tokens.len())
    }

    /// Every token or image key a slot fed so far
    pub fn recent(&self, seq: usize) -> &[u32] {
        self.slots.get(seq).map_or(&[], |s| s.tokens.as_slice())
    }

    pub fn last_logits(&self, seq: usize) -> Option<&[f32]> {
        self.slots.get(seq).and_then(|s| s.last_logits.as_deref())
    }

    pub fn backend_mut(&mut self) -> &mut dyn Backend {
        self.backend.as_mut()
    }

    pub fn backend_ref(&self) -> &dyn Backend {
        self.backend.as_ref()
    }

    fn slot(&self, seq: usize) -> Result<&Slot> {
        self.slots.get(seq).ok_or_else(|| anyhow::anyhow!("sequence slot {seq} of {}", self.slots.len()))
    }

    /// The graph of a local segment for a chunk shape, built once per shape
    fn graph(&mut self, hop: usize, seg: Segment, n_tokens: usize, n_out: usize) -> Result<Arc<Graph>> {
        let key = (hop, n_tokens, n_out, seg.inject);
        if !self.graphs.contains_key(&key) {
            let mut g = GraphBuilder::new();
            let out = self.model.segment(&mut g, n_tokens, seg)?;
            self.graphs.insert(key, Arc::new(g.finish(out)));
        }
        Ok(self.graphs[&key].clone())
    }

    /// One pass over runs of rows, one run per slot, through every hop, returning the last n_out rows of logits
    fn pass(&mut self, runs: &[(usize, Rows<'_>)], n_out: usize) -> Result<Vec<f32>> {
        let mut tokens = Vec::new();
        let mut positions = Vec::new();
        let mut seqs = Vec::new();
        let mut rope: Vec<[u32; 4]> = Vec::new();
        let mut attend: Vec<u32> = Vec::new();
        let mut embeds: Vec<f32> = Vec::new();
        let mut afters: Vec<Vec<u32>> = Vec::new();
        let inject = runs.iter().any(|(_, r)| matches!(r, Rows::Embeds { .. }));
        if inject && runs.iter().any(|(_, r)| matches!(r, Rows::Tokens(_))) {
            bail!("a pass feeds tokens or embeddings, not both");
        }
        for (seq, rows) in runs {
            let slot = self.slot(*seq)?;
            let past = slot.tokens.len();
            let start = slot.next_pos();
            let n = rows.len();
            match rows {
                Rows::Tokens(toks) => {
                    tokens.extend_from_slice(toks);
                    rope.extend((0..n).map(|i| {
                        let p = start + i as u32;
                        [p, p, p, p]
                    }));
                    attend.extend((past..past + n).map(|p| p as u32));
                    afters.push((1..=n).map(|i| start + i as u32).collect());
                }
                Rows::Embeds { keys, embeds: e, rope: r, whole, advance } => {
                    ensure!(e.len() == n * self.width, "{} embedding values for {n} rows of {}", e.len(), self.width);
                    ensure!(r.len() == n, "{} rotary rows for {n} embedded rows", r.len());
                    tokens.extend_from_slice(keys);
                    embeds.extend_from_slice(e);
                    rope.extend(r.iter().map(|s| [start + s[0], start + s[1], start + s[2], s[3]]));
                    let last = (past + n - 1) as u32;
                    attend.extend((past..past + n).map(|p| if *whole { last } else { p as u32 }));
                    afters.push((0..n).map(|i| if i + 1 == n { start + advance } else { start }).collect());
                }
            }
            positions.extend((past..past + n).map(|p| p as u32));
            seqs.extend(std::iter::repeat_n(*seq as u32, n));
        }
        let rope_in = if self.mrope || inject { rope.clone() } else { Vec::new() };
        let last = self.hops.len() - 1;
        let mut hidden: Vec<f32> = if inject { embeds } else { Vec::new() };
        let mut out = Vec::new();
        for h in 0..=last {
            match &mut self.hops[h] {
                Hop::Remote(conn) => {
                    hidden = conn.forward(&tokens, &positions, &seqs, &rope_in, &attend, &hidden, self.width)?;
                }
                Hop::Local(seg) => {
                    let mut seg = *seg;
                    if h == last {
                        seg.n_out = Some(n_out);
                    }
                    if h == 0 && inject {
                        seg.embed = false;
                        seg.inject = true;
                    }
                    let graph = self.graph(h, seg, tokens.len(), n_out)?;
                    let inputs = Inputs { tokens: tokens.clone(), positions: positions.clone(), seqs: seqs.clone(), hidden: std::mem::take(&mut hidden), hidden_cols: self.width, rope: rope_in.clone(), attend: attend.clone() };
                    out = self.backend.run(&graph, &inputs, self.store.as_ref())?;
                    if h != last {
                        hidden = std::mem::take(&mut out);
                    }
                }
            }
        }
        let vocab = out.len() / n_out.max(1);
        let mut at = 0;
        for ((seq, rows), after) in runs.iter().zip(afters) {
            let n = rows.len();
            let slot = &mut self.slots[*seq];
            slot.tokens.extend_from_slice(&tokens[at..at + n]);
            slot.after.extend(after);
            at += n;
            let row = if n_out == tokens.len() { at - 1 } else { n_out - 1 };
            slot.last_logits = Some(out[row * vocab..(row + 1) * vocab].to_vec());
        }
        Ok(out)
    }

    fn check(&self, seq: usize, n: usize) -> Result<()> {
        let past = self.slot(seq)?.tokens.len();
        if n == 0 {
            bail!("nothing to feed");
        }
        if past + n > self.n_ctx {
            bail!("{} tokens past a context of {}", past + n, self.n_ctx);
        }
        Ok(())
    }

    /// Feeds one slot's tokens in batch sized chunks and returns the logits of the last one
    pub fn feed(&mut self, seq: usize, tokens: &[u32]) -> Result<Vec<f32>> {
        self.check(seq, tokens.len())?;
        let mut out = Vec::new();
        for chunk in tokens.chunks(self.n_batch) {
            out = self.pass(&[(seq, Rows::Tokens(chunk))], 1)?;
        }
        Ok(out)
    }

    /// Feeds one slot's tokens in batch sized chunks and returns the logits of every one, row by row
    pub fn feed_all(&mut self, seq: usize, tokens: &[u32]) -> Result<Vec<f32>> {
        self.check(seq, tokens.len())?;
        let mut out = Vec::new();
        for chunk in tokens.chunks(self.n_batch) {
            out.extend(self.pass(&[(seq, Rows::Tokens(chunk))], chunk.len())?);
        }
        Ok(out)
    }

    /// Feeds image embeddings to one slot as rows standing in for tokens, every row seeing the whole image when asked, and returns the last logits
    pub fn feed_embeds(&mut self, seq: usize, keys: &[u32], embeds: &[f32], rope: &[[u32; 4]], whole: bool, advance: u32) -> Result<Vec<f32>> {
        let n = keys.len();
        self.check(seq, n)?;
        ensure!(!whole || n <= self.n_batch, "an image of {n} tokens seeing itself whole needs a batch of at least {n}, the batch is {}", self.n_batch);
        let mut out = Vec::new();
        let mut at = 0;
        while at < n {
            let end = (at + self.n_batch).min(n);
            let last = end == n;
            let rows = Rows::Embeds { keys: &keys[at..end], embeds: &embeds[at * self.width..end * self.width], rope: &rope[at..end], whole, advance: if last { advance } else { 0 } };
            out = self.pass(&[(seq, rows)], 1)?;
            at = end;
        }
        Ok(out)
    }

    /// Feeds one token to each of several slots in one pass and returns one row of logits per slot in order
    pub fn step(&mut self, next: &[(usize, u32)]) -> Result<Vec<f32>> {
        if next.is_empty() {
            bail!("nothing to feed");
        }
        if next.len() > self.n_batch {
            bail!("{} slots stepped past a batch of {}", next.len(), self.n_batch);
        }
        for (i, (seq, _)) in next.iter().enumerate() {
            self.check(*seq, 1)?;
            if next[..i].iter().any(|(s, _)| s == seq) {
                bail!("slot {seq} stepped twice in one pass");
            }
        }
        let runs: Vec<(usize, Rows<'_>)> = next.iter().map(|(s, t)| (*s, Rows::Tokens(std::slice::from_ref(t)))).collect();
        self.pass(&runs, next.len())
    }

    /// Keeps the first n positions of one slot's state here and on every stage, for prefix reuse
    pub fn truncate(&mut self, seq: usize, n: usize) -> Result<()> {
        self.slot(seq)?;
        self.backend.truncate(seq, n)?;
        for h in &mut self.hops {
            if let Hop::Remote(conn) = h {
                conn.truncate(seq, n)?;
            }
        }
        let slot = &mut self.slots[seq];
        if slot.tokens.len() > n {
            slot.last_logits = None;
        }
        slot.tokens.truncate(n);
        slot.after.truncate(n);
        if n == 0 {
            slot.last_logits = None;
        }
        Ok(())
    }

    /// Rows of the longest prefix a slot shares with a prompt's keys, one short of the whole so a row remains to feed
    pub fn common(&self, seq: usize, prompt: &[u32]) -> usize {
        let tokens = self.recent(seq);
        let common = tokens.iter().zip(prompt).take_while(|(a, b)| a == b).count();
        common.min(prompt.len().saturating_sub(1))
    }

    /// Keeps the longest prefix of a slot's fed rows that a new prompt shares, one short of the whole so a row remains to feed
    pub fn reuse(&mut self, seq: usize, prompt: &[u32]) -> Result<usize> {
        let keep = self.common(seq, prompt);
        if keep == 0 {
            self.reset(seq)?;
            return Ok(0);
        }
        if self.truncate(seq, keep).is_err() {
            self.reset(seq)?;
            return Ok(0);
        }
        Ok(keep)
    }

    /// One slot as bytes another process of the same plan loads, the keys and rotary positions with every byte of backend state
    pub fn save(&mut self, seq: usize) -> Result<Vec<u8>> {
        ensure!(self.hops.len() == 1, "a session over stages cannot be saved, its state is spread across them");
        let slot = self.slot(seq)?;
        let (tokens, after) = (slot.tokens.clone(), slot.after.clone());
        let state = self.backend.save_state(seq)?;
        let mut out = Vec::with_capacity(8 + tokens.len() * 8 + state.len());
        out.extend_from_slice(&(tokens.len() as u64).to_le_bytes());
        for t in &tokens {
            out.extend_from_slice(&t.to_le_bytes());
        }
        for a in &after {
            out.extend_from_slice(&a.to_le_bytes());
        }
        out.extend_from_slice(&state);
        Ok(out)
    }

    /// Replaces one slot with a sequence saved under the same plan
    pub fn load(&mut self, seq: usize, bytes: &[u8]) -> Result<()> {
        ensure!(self.hops.len() == 1, "a session over stages cannot be loaded, its state is spread across them");
        self.slot(seq)?;
        ensure!(bytes.len() >= 8, "saved sequence of {} bytes", bytes.len());
        let n = u64::from_le_bytes(bytes[..8].try_into()?) as usize;
        ensure!(bytes.len() >= 8 + n * 8, "saved sequence names {n} rows in {} bytes", bytes.len());
        ensure!(n <= self.n_ctx, "saved sequence of {n} rows past a context of {}", self.n_ctx);
        let u32s = |b: &[u8]| -> Vec<u32> { b.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect() };
        let tokens = u32s(&bytes[8..8 + n * 4]);
        let after = u32s(&bytes[8 + n * 4..8 + n * 8]);
        self.backend.load_state(seq, &bytes[8 + n * 8..])?;
        self.slots[seq] = Slot { tokens, after, last_logits: None };
        Ok(())
    }

    /// Drops every position of one slot here and on every stage
    pub fn reset(&mut self, seq: usize) -> Result<()> {
        self.slot(seq)?;
        self.backend.reset_seq(seq);
        for h in &mut self.hops {
            if let Hop::Remote(conn) = h {
                conn.reset(Some(seq))?;
            }
        }
        self.slots[seq] = Slot::default();
        Ok(())
    }

    /// Drops every position of every slot here and on every stage
    pub fn reset_all(&mut self) -> Result<()> {
        self.backend.reset();
        for h in &mut self.hops {
            if let Hop::Remote(conn) = h {
                conn.reset(None)?;
            }
        }
        self.slots.iter_mut().for_each(|s| *s = Slot::default());
        Ok(())
    }
}

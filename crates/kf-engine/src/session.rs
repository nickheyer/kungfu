use crate::loader::{Loaded, Store};
use crate::sampler::Candidates;
use crate::stage::StageConn;
use anyhow::{Result, bail, ensure};
use kf_core::{Backend, Graph, GraphBuilder, Inputs, Op, Plan, RopeKind, TREE_ROWS, TreeRow};
use kf_layers::{Model, Segment};
use std::collections::HashMap;
use std::sync::Arc;

/// One part of a pass, a segment this process runs or a stage that runs its layers
enum Hop {
    Local(Segment),
    Remote(StageConn),
}

/// The tokens one slot has fed and the rotary position after each
#[derive(Clone, Debug, Default)]
struct Slot {
    tokens: Vec<u32>,
    after: Vec<u32>,
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
    /// Tokens forming a tree, each row's parent an earlier row or minus one for the first, every row attending to the
    /// slot's history and its ancestors alone and taking the position its depth gives it
    Tree { tokens: &'a [u32], parents: &'a [i32] },
}

impl Rows<'_> {
    fn len(&self) -> usize {
        match self {
            Rows::Tokens(t) => t.len(),
            Rows::Embeds { keys, .. } => keys.len(),
            Rows::Tree { tokens, .. } => tokens.len(),
        }
    }
}

/// Depth of every node of a tree given by parents, the first node at depth zero
pub fn tree_depths(parents: &[i32]) -> Result<Vec<u32>> {
    ensure!(parents.len() <= TREE_ROWS, "a tree of {} rows past {TREE_ROWS}", parents.len());
    let mut depth = Vec::with_capacity(parents.len());
    for (i, p) in parents.iter().enumerate() {
        match *p {
            -1 => {
                ensure!(i == 0, "row {i} has no parent, only the first row may");
                depth.push(0);
            }
            p if p >= 0 && (p as usize) < i => depth.push(depth[p as usize] + 1),
            p => bail!("row {i} names parent {p}, which is not an earlier row"),
        }
    }
    Ok(depth)
}

/// The ancestor mask of row i of a tree over the tree's rows relative to its last row, see TreeRow
fn tree_mask(parents: &[i32], i: usize) -> u64 {
    let n = parents.len();
    let mut mask = if n < 64 { !0u64 << n } else { 0 };
    let mut at = i as i32;
    while at >= 0 {
        mask |= 1u64 << (n - 1 - at as usize);
        at = parents[at as usize];
    }
    mask
}

/// Candidates a pass hands back per row of logits, the rest of the row staying on the backend
pub const TOP: usize = 64;

/// Sequence slots against one backend, each slot's positions advancing as its rows are fed, slots decoded together in
/// one pass, every row of logits coming back as its top candidates with the row's log sum exp
pub struct Session {
    backend: Box<dyn Backend>,
    store: Arc<Store>,
    model: Arc<Model>,
    n_ctx: usize,
    n_batch: usize,
    width: usize,
    n_vocab: usize,
    /// Candidates per row of logits
    top: usize,
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
            n_vocab: loaded.hparams.n_vocab,
            top: TOP.min(loaded.hparams.n_vocab.max(1)),
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
    /// The graph of a hop, the head's logits reduced to their top candidates per row
    fn graph(&mut self, hop: usize, seg: Segment, n_tokens: usize, n_out: usize) -> Result<Arc<Graph>> {
        let key = (hop, n_tokens, n_out, seg.inject);
        if !self.graphs.contains_key(&key) {
            let mut g = GraphBuilder::new();
            let mut out = self.model.segment(&mut g, n_tokens, seg)?;
            if seg.n_out.is_some() {
                out = g.push(Op::TopK { k: self.top }, &[out]);
            }
            self.graphs.insert(key, Arc::new(g.finish(out)));
        }
        Ok(self.graphs[&key].clone())
    }

    /// One full row of logits of the last pass, by the row index its candidates carry, valid until the next pass
    pub fn dense_row(&mut self, row: usize) -> Result<Vec<f32>> {
        self.backend.dense_row(row)
    }

    /// One pass over runs of rows, one run per slot, through every hop, returning the candidates of the last n_out rows
    fn pass(&mut self, runs: &[(usize, Rows<'_>)], n_out: usize) -> Result<Vec<Candidates>> {
        let mut tokens = Vec::new();
        let mut positions = Vec::new();
        let mut seqs = Vec::new();
        let mut rope: Vec<[u32; 4]> = Vec::new();
        let mut attend: Vec<u32> = Vec::new();
        let mut embeds: Vec<f32> = Vec::new();
        let mut afters: Vec<Vec<u32>> = Vec::new();
        let mut tree: Vec<TreeRow> = Vec::new();
        let inject = runs.iter().any(|(_, r)| matches!(r, Rows::Embeds { .. }));
        if inject && runs.iter().any(|(_, r)| !matches!(r, Rows::Embeds { .. })) {
            bail!("a pass feeds tokens or embeddings, not both");
        }
        let treed = runs.iter().any(|(_, r)| matches!(r, Rows::Tree { .. }));
        if treed && runs.len() > 1 {
            bail!("a tree pass feeds one slot");
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
                Rows::Tree { tokens: toks, parents } => {
                    ensure!(toks.len() == parents.len(), "{} tree tokens for {} parents", toks.len(), parents.len());
                    let depth = tree_depths(parents)?;
                    tokens.extend_from_slice(toks);
                    rope.extend(depth.iter().map(|d| {
                        let p = start + d;
                        [p, p, p, p]
                    }));
                    let last = (past + n - 1) as u32;
                    attend.extend(std::iter::repeat_n(last, n));
                    tree.extend((0..n).map(|i| TreeRow { mask: tree_mask(parents, i), position: start + depth[i] }));
                    afters.push(depth.iter().map(|d| start + d + 1).collect());
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
        let rope_in = if self.mrope || inject || treed { rope.clone() } else { Vec::new() };
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
                    let inputs = Inputs { tokens: tokens.clone(), positions: positions.clone(), seqs: seqs.clone(), hidden: std::mem::take(&mut hidden), hidden_cols: self.width, rope: rope_in.clone(), attend: attend.clone(), tree: tree.clone() };
                    out = self.backend.run(&graph, &inputs, self.store.as_ref())?;
                    if h != last {
                        hidden = std::mem::take(&mut out);
                    }
                }
            }
        }
        let mut at = 0;
        for ((seq, rows), after) in runs.iter().zip(afters) {
            let n = rows.len();
            let slot = &mut self.slots[*seq];
            slot.tokens.extend_from_slice(&tokens[at..at + n]);
            slot.after.extend(after);
            at += n;
        }
        let w = 2 * self.top + 1;
        ensure!(out.len() == n_out * w, "{} values for {n_out} rows of {} candidates", out.len(), self.top);
        out.chunks_exact(w).enumerate().map(|(r, packed)| Candidates::from_packed(packed, self.top, self.n_vocab, r)).collect()
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

    /// Feeds one slot's tokens in batch sized chunks and returns the candidates of the last one
    pub fn feed(&mut self, seq: usize, tokens: &[u32]) -> Result<Candidates> {
        self.check(seq, tokens.len())?;
        let mut out = Vec::new();
        for chunk in tokens.chunks(self.n_batch) {
            out = self.pass(&[(seq, Rows::Tokens(chunk))], 1)?;
        }
        out.pop().ok_or_else(|| anyhow::anyhow!("the pass scored no row"))
    }

    /// Feeds one slot's tokens in batch sized chunks and returns the candidates of every one, row by row
    pub fn feed_all(&mut self, seq: usize, tokens: &[u32]) -> Result<Vec<Candidates>> {
        self.check(seq, tokens.len())?;
        let mut out = Vec::new();
        for chunk in tokens.chunks(self.n_batch) {
            out.extend(self.pass(&[(seq, Rows::Tokens(chunk))], chunk.len())?);
        }
        Ok(out)
    }

    /// Feeds a tree of tokens to one slot in one pass, each row attending to the slot's history and its ancestors, and
    /// returns the candidates of every row, the rows standing in the slot's cache until accept_path keeps one path
    pub fn feed_tree(&mut self, seq: usize, tokens: &[u32], parents: &[i32]) -> Result<Vec<Candidates>> {
        self.check(seq, tokens.len())?;
        ensure!(tokens.len() <= self.n_batch && tokens.len() <= TREE_ROWS, "a tree of {} rows past a batch of {} or the limit of {TREE_ROWS}", tokens.len(), self.n_batch);
        self.pass(&[(seq, Rows::Tree { tokens, parents })], tokens.len())
    }

    /// Keeps one path of the tree the slot's last pass fed, its rows in order from the first, dropping every other row
    pub fn accept_path(&mut self, seq: usize, n_tree: usize, path: &[usize]) -> Result<()> {
        self.slot(seq)?;
        ensure!(!path.is_empty() && path[0] == 0 && path.windows(2).all(|w| w[0] < w[1]) && path.iter().all(|i| *i < n_tree), "a path of tree rows from the first in rising order");
        let base = self.slots[seq].tokens.len() - n_tree;
        self.backend.compact(seq, base, path)?;
        for h in &mut self.hops {
            if let Hop::Remote(conn) = h {
                conn.truncate(seq, base + path.len())?;
            }
        }
        let slot = &mut self.slots[seq];
        let kept: Vec<u32> = path.iter().map(|i| slot.tokens[base + i]).collect();
        slot.tokens.truncate(base);
        slot.tokens.extend(kept);
        let start = slot.after.get(base.wrapping_sub(1)).copied().unwrap_or(0);
        slot.after.truncate(base);
        slot.after.extend((1..=path.len()).map(|i| start + i as u32));
        Ok(())
    }

    /// Feeds image embeddings to one slot as rows standing in for tokens, every row seeing the whole image when asked, and returns the last candidates
    pub fn feed_embeds(&mut self, seq: usize, keys: &[u32], embeds: &[f32], rope: &[[u32; 4]], whole: bool, advance: u32) -> Result<Candidates> {
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
        out.pop().ok_or_else(|| anyhow::anyhow!("the pass scored no row"))
    }

    /// Feeds one token to each of several slots in one pass and returns the candidates of one row per slot in order
    pub fn step(&mut self, next: &[(usize, u32)]) -> Result<Vec<Candidates>> {
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
        slot.tokens.truncate(n);
        slot.after.truncate(n);
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
        self.slots[seq] = Slot { tokens, after };
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

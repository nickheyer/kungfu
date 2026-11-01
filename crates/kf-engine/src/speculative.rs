use crate::sampler::Candidates;
use crate::session::Session;
use anyhow::{Result, bail, ensure};
pub use kf_core::Tree;

/// Anything that proposes tokens ahead of the model, next token heads, draft models, or lookup
pub trait Drafter: Send {
    fn name(&self) -> &'static str;
    /// A tree rooted at the token chosen for the next position, up to n rows in all
    fn draft(&mut self, context: &[u32], own: u32, n: usize) -> Tree;
    fn accept(&mut self, accepted: &[u32]);
}

/// Picks one token from the candidates of a row given the tokens before it and a way to fetch the whole row, the
/// sampler with every rule a request adds
pub type Chooser<'a> = dyn FnMut(&Candidates, &[u32], &mut dyn FnMut() -> Result<Vec<f32>>) -> Result<u32> + 'a;

/// Verifies a chain of drafted tokens of one slot in one pass given the token the sampler picked for the slot's next
/// position, and returns the accepted prefix plus the model's next token
pub fn verify(session: &mut Session, seq: usize, own: u32, drafted: &[u32], choose: &mut Chooser<'_>) -> Result<Vec<u32>> {
    if drafted.is_empty() {
        bail!("nothing drafted");
    }
    if own != drafted[0] {
        return Ok(vec![own]);
    }
    let tree = Tree { tokens: drafted.to_vec(), parents: (0..drafted.len()).map(|i| i as i32 - 1).collect() };
    verify_tree(session, seq, &tree, choose)
}

/// Verifies a tree of drafted tokens of one slot in one pass. From the root, the chooser's own pick at each row stands
/// when a child holds it and the walk moves to that child, else the pick ends the walk as the model's next token. The
/// accepted path stays in the slot's cache and every other row leaves it. Returns the path's tokens plus the next token.
pub fn verify_tree(session: &mut Session, seq: usize, tree: &Tree, choose: &mut Chooser<'_>) -> Result<Vec<u32>> {
    ensure!(!tree.is_empty(), "nothing drafted");
    ensure!(tree.len() <= session.n_batch(), "{} drafted rows past a batch of {}", tree.len(), session.n_batch());
    let mut history = session.recent(seq).to_vec();
    history.push(tree.tokens[0]);
    let rows = session.feed_tree(seq, &tree.tokens, &tree.parents)?;
    let mut path = vec![0usize];
    let mut accepted = vec![tree.tokens[0]];
    let next = loop {
        let at = *path.last().ok_or_else(|| anyhow::anyhow!("an empty path"))?;
        let c = &rows[at];
        let mut dense = || session.dense_row(c.row);
        let choice = choose(c, &history, &mut dense)?;
        match tree.child(at, choice) {
            Some(child) => {
                path.push(child);
                accepted.push(choice);
                history.push(choice);
            }
            None => break choice,
        }
    };
    session.accept_path(seq, tree.len(), &path)?;
    accepted.push(next);
    Ok(accepted)
}

/// Three grams the shared table holds, a power of two
const NGRAM_SLOTS: usize = 1 << 20;

/// Longest tail of the context the context lookup matches
const NGRAM_LONGEST: usize = 5;

/// Context tokens the lookup scans back over
const NGRAM_WINDOW: usize = 8192;

/// Successors the drafter keeps per token from the model's own candidates
pub const NGRAM_TOPS: usize = 3;

/// Drafts the tokens that followed the context's own last n gram, else what followed its last three gram in every
/// sequence this process has seen, else the model's own most likely successors of each token as its logits last
/// said, no model pass involved
pub struct Ngram {
    keys: Vec<u64>,
    next: Vec<u32>,
    /// Most likely next tokens after each token the last time the model scored a row ending in it, best first,
    /// u32::MAX where none
    top: Vec<[u32; NGRAM_TOPS]>,
}

impl Ngram {
    pub fn new(n_vocab: usize) -> Self {
        Self { keys: vec![u64::MAX; NGRAM_SLOTS], next: vec![0; NGRAM_SLOTS], top: vec![[u32::MAX; NGRAM_TOPS]; n_vocab] }
    }

    /// Learns the model's favourite successors of token last from the candidates of a row ending in it
    pub fn recycle(&mut self, last: u32, tops: &[u32]) {
        if let Some(t) = self.top.get_mut(last as usize) {
            for (i, slot) in t.iter_mut().enumerate() {
                *slot = tops.get(i).copied().unwrap_or(u32::MAX);
            }
        }
    }

    /// The model's favourite successors of a token, best first
    fn tops(&self, token: u32) -> impl Iterator<Item = u32> + '_ {
        self.top.get(token as usize).into_iter().flat_map(|t| t.iter().copied()).filter(|t| *t != u32::MAX)
    }

    fn key(a: u32, b: u32, c: u32) -> u64 {
        let k = (a as u64) << 42 ^ (b as u64) << 21 ^ c as u64;
        k.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1
    }

    /// Learns every three gram of the tokens and what followed it
    pub fn observe(&mut self, tokens: &[u32]) {
        for w in tokens.windows(4) {
            let k = Self::key(w[0], w[1], w[2]);
            let i = (k >> 44) as usize & (NGRAM_SLOTS - 1);
            self.keys[i] = k;
            self.next[i] = w[3];
        }
    }

    /// The continuation of the longest tail of between lo and NGRAM_LONGEST tokens that occurred earlier in the
    /// context, the most recent occurrence, up to n tokens
    fn from_context(&self, context: &[u32], lo: usize, n: usize) -> Vec<u32> {
        let len = context.len();
        let start = len.saturating_sub(NGRAM_WINDOW);
        for m in (lo..=NGRAM_LONGEST.min(len - 1)).rev() {
            let tail = &context[len - m..];
            let found = (start..len - m).rev().find(|&p| &context[p..p + m] == tail);
            if let Some(p) = found {
                let from = p + m;
                let out: Vec<u32> = context[from..(from + n).min(len)].to_vec();
                if !out.is_empty() {
                    return out;
                }
            }
        }
        Vec::new()
    }

    /// A chain through the shared three gram table, extended by the recycled top tokens once the table runs out
    fn from_tables(&self, context: &[u32], n: usize) -> Vec<u32> {
        let len = context.len();
        let mut out = Vec::with_capacity(n);
        let mut last = context[len - 1];
        let (mut a, mut b) = if len >= 3 { (context[len - 3], context[len - 2]) } else { (u32::MAX, u32::MAX) };
        let mut grams = len >= 3;
        while out.len() < n {
            let t = if grams {
                let k = Self::key(a, b, last);
                let i = (k >> 44) as usize & (NGRAM_SLOTS - 1);
                if self.keys[i] == k { Some(self.next[i]) } else { None }
            } else {
                None
            };
            let t = match t {
                Some(t) => t,
                None => {
                    grams = false;
                    match self.tops(last).next() {
                        Some(t) => t,
                        None => break,
                    }
                }
            };
            out.push(t);
            (a, b, last) = (b, last, t);
        }
        out
    }

    /// Up to n tokens after the context, first a repeat of at least two tokens inside the context, else the tables, else a
    /// repeat of one token, empty when nothing knows the tail
    pub fn chain(&self, context: &[u32], n: usize) -> Vec<u32> {
        if n == 0 || context.is_empty() {
            return Vec::new();
        }
        let out = self.from_context(context, 2, n);
        if !out.is_empty() {
            return out;
        }
        let out = self.from_tables(context, n);
        if !out.is_empty() {
            return out;
        }
        self.from_context(context, 1, n)
    }

    /// A tree rooted at own of up to n rows: the chain the context and tables give after own, then the model's other
    /// favourite successors of own as its remaining children, each followed by its own favourite, while rows remain
    pub fn draft(&self, context: &[u32], own: u32, n: usize) -> Tree {
        let mut tree = Tree::root(own);
        if n <= 1 {
            return tree;
        }
        let mut ctx = context.to_vec();
        ctx.push(own);
        let mut parent = 0;
        for t in self.chain(&ctx, n - 1) {
            match tree.push(parent, t) {
                Some(row) => parent = row,
                None => break,
            }
        }
        for alt in self.tops(own) {
            if tree.len() >= n {
                break;
            }
            if tree.has_child(0, alt) {
                continue;
            }
            let Some(row) = tree.push(0, alt) else { break };
            if tree.len() < n
                && let Some(follow) = self.tops(alt).next()
            {
                tree.push(row, follow);
            }
        }
        tree
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::tree_depths;

    #[test]
    fn trees_grow_find_children_and_walk_paths() {
        let mut t = Tree::root(7);
        let a = t.push(0, 1).expect("row");
        let b = t.push(0, 2).expect("row");
        let c = t.push(a, 3).expect("row");
        assert_eq!((a, b, c), (1, 2, 3));
        assert_eq!(t.child(0, 2), Some(2));
        assert_eq!(t.child(0, 9), None);
        assert!(t.has_child(1, 3) && !t.has_child(2, 3));
        assert_eq!(t.path_tokens(3), vec![7, 1, 3]);
        assert_eq!(tree_depths(&t.parents).expect("depths"), vec![0, 1, 1, 2]);
        assert!(tree_depths(&[-1, 2]).is_err());
        assert!(tree_depths(&[0]).is_err());
        let mut full = Tree::root(0);
        while full.push(0, 1).is_some() {}
        assert_eq!(full.len(), kf_core::TREE_ROWS);
        // A grafted chain follows the children already there and grows new rows only within the limit
        let mut t = Tree::root(7);
        let a = t.push(0, 1).expect("row");
        t.push(a, 3).expect("row");
        t.graft(&[1, 3, 4, 5], 5);
        assert_eq!((t.len(), t.child(2, 4), t.child(3, 5)), (5, Some(3), Some(4)));
        t.graft(&[1, 3, 4, 5, 6], 5);
        assert_eq!(t.len(), 5, "the limit holds");
        t.graft(&[9], 8);
        assert_eq!(t.child(0, 9), Some(5));
    }

    #[test]
    fn ngram_drafts_a_tree_from_the_chain_and_the_favourites() {
        let mut ng = Ngram::new(100);
        // The context repeats 10 11 12 13, so after 10 11 the chain says 12 13
        let context = [10u32, 11, 12, 13, 20, 10, 11];
        ng.recycle(11, &[12, 30, 31]);
        ng.recycle(30, &[32]);
        // Six rows: the root and the chain 12 13 20 10 11 the context repeats, which leaves no row for an alternative
        let tree = ng.draft(&context[..6], 11, 6);
        assert_eq!(tree.tokens[0], 11);
        assert_eq!(tree.child(0, 12), Some(1));
        assert_eq!(tree.child(1, 13), Some(2));
        assert_eq!((tree.len(), tree.child(0, 30)), (6, None));
        // Nine rows leave room for the alternative with its own favourite and the second alternative
        let tree = ng.draft(&context[..6], 11, 9);
        let alt = tree.child(0, 30).expect("alternative");
        assert_eq!(tree.child(alt, 32), Some(alt + 1));
        assert_eq!(tree.child(0, 31), Some(alt + 2));
        assert_eq!(tree.len(), 9);
        assert_eq!(ng.draft(&context, 11, 1), Tree::root(11));
    }
}

use crate::session::Session;
use anyhow::{Result, bail};

/// Anything that proposes tokens ahead of the model, next token heads, draft models, or lookup
pub trait Drafter: Send {
    fn name(&self) -> &'static str;
    fn draft(&mut self, context: &[u32], n: usize) -> Vec<u32>;
    fn accept(&mut self, accepted: &[u32]);
}

/// Picks one token from a row of logits given the tokens before it, the sampler with every rule a request adds
pub type Chooser<'a> = dyn FnMut(&[f32], &[u32]) -> Result<u32> + 'a;

/// Verifies drafted tokens of one slot in one pass and returns the accepted prefix plus the model's next token
pub fn verify(session: &mut Session, seq: usize, drafted: &[u32], choose: &mut Chooser<'_>) -> Result<Vec<u32>> {
    speculative_verify(session, seq, drafted, choose)
}

/// Each drafted token stands when the chooser's own pick at that position equals it, the first miss replaces it
pub fn speculative_verify(session: &mut Session, seq: usize, drafted: &[u32], choose: &mut Chooser<'_>) -> Result<Vec<u32>> {
    if drafted.is_empty() {
        bail!("nothing drafted");
    }
    let prev = session.last_logits(seq).ok_or_else(|| anyhow::anyhow!("verify before any token was fed"))?.to_vec();
    let recent = session.recent(seq).to_vec();
    let own = choose(&prev, &recent)?;
    if own != drafted[0] {
        return Ok(vec![own]);
    }
    let start = session.n_past(seq);
    let rows = session.feed_all(seq, drafted)?;
    let vocab = rows.len() / drafted.len();
    let mut accepted = vec![drafted[0]];
    let mut next = None;
    let mut history = recent;
    history.push(drafted[0]);
    for i in 0..drafted.len() {
        let choice = choose(&rows[i * vocab..(i + 1) * vocab], &history)?;
        if i + 1 < drafted.len() && choice == drafted[i + 1] {
            accepted.push(choice);
            history.push(choice);
            continue;
        }
        next = Some(choice);
        break;
    }
    session.truncate(seq, start + accepted.len())?;
    accepted.push(next.ok_or_else(|| anyhow::anyhow!("verification ended without a model token"))?);
    Ok(accepted)
}

use anyhow::{Result, ensure};

#[derive(Clone, Debug)]
pub struct Sampling {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub repeat_penalty: f32,
    pub repeat_window: usize,
}

impl Default for Sampling {
    fn default() -> Self {
        Self { temperature: 0.7, top_k: 40, top_p: 0.95, min_p: 0.0, repeat_penalty: 1.0, repeat_window: 64 }
    }
}

/// Xorshift generator, deterministic from its seed
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }
    pub fn next_f32(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// One token from the logits, greedy at zero temperature
pub fn sample(logits: &[f32], s: &Sampling, recent: &[u32], rng: &mut Rng) -> u32 {
    if s.temperature <= 0.0 && s.repeat_penalty == 1.0 {
        return argmax(logits);
    }
    let mut l = logits.to_vec();
    if s.repeat_penalty != 1.0 {
        let window = &recent[recent.len().saturating_sub(s.repeat_window)..];
        for t in window {
            let Some(v) = l.get_mut(*t as usize) else { continue };
            *v = penalized(*v, s.repeat_penalty);
        }
    }
    if s.temperature <= 0.0 {
        return argmax(&l);
    }
    let mut idx: Vec<usize> = (0..l.len()).collect();
    idx.sort_by(|a, b| l[*b].partial_cmp(&l[*a]).unwrap_or(std::cmp::Ordering::Equal));
    if s.top_k > 0 {
        idx.truncate(s.top_k);
    }
    let ids: Vec<u32> = idx.iter().map(|i| *i as u32).collect();
    let vals: Vec<f32> = idx.iter().map(|i| l[*i]).collect();
    draw(&ids, &vals, s, rng)
}

/// A logit after the repeat penalty, divided when positive and multiplied when negative
fn penalized(v: f32, penalty: f32) -> f32 {
    if v > 0.0 { v / penalty } else { v * penalty }
}

/// One token from ids ordered by falling logit, the temperature, min p, and top p rules applied in that order
fn draw(ids: &[u32], l: &[f32], s: &Sampling, rng: &mut Rng) -> u32 {
    let max = l[0];
    let mut probs: Vec<f32> = l.iter().map(|v| ((v - max) / s.temperature).exp()).collect();
    let sum: f32 = probs.iter().sum();
    probs.iter_mut().for_each(|p| *p /= sum);
    let mut keep = probs.len();
    if s.min_p > 0.0 {
        let floor = probs[0] * s.min_p;
        keep = probs.iter().take_while(|p| **p >= floor).count().max(1);
    }
    if s.top_p < 1.0 {
        let mut acc = 0.0;
        let mut n = 0;
        for p in &probs[..keep] {
            acc += p;
            n += 1;
            if acc >= s.top_p {
                break;
            }
        }
        keep = n;
    }
    let probs = &probs[..keep];
    let sum: f32 = probs.iter().sum();
    let mut r = rng.next_f32() * sum;
    for (i, p) in ids[..keep].iter().zip(probs) {
        if r < *p {
            return *i;
        }
        r -= p;
    }
    ids[keep - 1]
}

/// The top candidates of one row of logits, largest first and lower id first among equals, every other token's logit
/// at most floor, with the row's log sum exp and the row's index in the pass that scored it
#[derive(Clone, Debug, PartialEq)]
pub struct Candidates {
    pub ids: Vec<u32>,
    pub logits: Vec<f32>,
    /// Whether a bias, penalty, or mask moved the logit, so the token may rank below tokens outside the candidates
    pub touched: Vec<bool>,
    pub floor: f32,
    pub lse: f32,
    pub vocab: usize,
    pub row: usize,
}

impl Candidates {
    /// From the packed row a top k op writes, k ids as floats, k values, and the log sum exp
    pub fn from_packed(packed: &[f32], k: usize, vocab: usize, row: usize) -> Result<Self> {
        ensure!(k >= 1 && packed.len() == 2 * k + 1, "{} packed values for {k} candidates", packed.len());
        let ids: Vec<u32> = packed[..k].iter().map(|v| *v as u32).collect();
        let logits = packed[k..2 * k].to_vec();
        ensure!(ids.iter().all(|id| (*id as usize) < vocab), "a candidate id past the vocabulary of {vocab}");
        Ok(Self { ids, logits, touched: vec![false; k], floor: packed[2 * k - 1], lse: packed[2 * k], vocab, row })
    }

    /// The top k of a full row, as the top k op selects them
    pub fn select(logits: &[f32], k: usize, row: usize) -> Result<Self> {
        let packed = kf_backend_cpu::ops::topk_rows(&kf_backend_cpu::Buf::from_vec(logits.to_vec(), 1, logits.len()), k)?;
        Self::from_packed(&packed.data, k, logits.len(), row)
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// The most likely token as the pass scored the row, before any bias or mask
    pub fn argmax(&self) -> u32 {
        self.ids[0]
    }

    pub fn logit(&self, id: u32) -> Option<f32> {
        self.ids.iter().position(|i| *i == id).map(|i| self.logits[i])
    }

    /// Log probability of a candidate under the row, none for a token outside the candidates
    pub fn logprob(&self, id: u32) -> Option<f32> {
        self.logit(id).map(|v| if v == f32::NEG_INFINITY { v } else { v - self.lse })
    }

    /// Adds b to a token's logit, false when a positive bias lifts a token outside the candidates, which they cannot rank
    pub fn bias(&mut self, id: u32, b: f32) -> bool {
        match self.ids.iter().position(|i| *i == id) {
            Some(i) => {
                self.logits[i] += b;
                self.touched[i] = true;
                true
            }
            None => b <= 0.0 || (id as usize) >= self.vocab,
        }
    }

    /// Takes a token out of the sampler's reach, a token outside the candidates needing nothing
    pub fn mask(&mut self, id: u32) {
        if let Some(i) = self.ids.iter().position(|i| *i == id) {
            self.logits[i] = f32::NEG_INFINITY;
            self.touched[i] = true;
        }
    }
}

/// The token the dense sampler would pick from the whole row when the candidates settle it, none when the sampler's
/// support may reach a token outside them, the generator advanced only when a token comes back
pub fn sample_sparse(c: &Candidates, s: &Sampling, recent: &[u32], rng: &mut Rng) -> Option<u32> {
    let k = c.ids.len();
    if k == 0 {
        return None;
    }
    let mut l = c.logits.clone();
    let mut touched = c.touched.clone();
    if s.repeat_penalty != 1.0 {
        let window = &recent[recent.len().saturating_sub(s.repeat_window)..];
        for t in window {
            if (*t as usize) >= c.vocab {
                continue;
            }
            match c.ids.iter().position(|id| id == t) {
                Some(i) => {
                    l[i] = penalized(l[i], s.repeat_penalty);
                    touched[i] = true;
                }
                // A penalty below one lifts a recent token the candidates do not hold
                None if s.repeat_penalty < 1.0 => return None,
                None => {}
            }
        }
    }
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|a, b| l[*b].partial_cmp(&l[*a]).unwrap_or(std::cmp::Ordering::Equal).then(c.ids[*a].cmp(&c.ids[*b])));
    // Candidates the row placed above every outside token, or moved and still above the floor, lead the whole row
    let certain = order.iter().take_while(|&&i| !touched[i] || l[i] > c.floor).count();
    if certain == 0 {
        return None;
    }
    if s.temperature <= 0.0 {
        return Some(c.ids[order[0]]);
    }
    if s.top_k == 0 || s.top_k > certain {
        return None;
    }
    let ids: Vec<u32> = order[..s.top_k].iter().map(|i| c.ids[*i]).collect();
    let vals: Vec<f32> = order[..s.top_k].iter().map(|i| l[*i]).collect();
    Some(draw(&ids, &vals, s, rng))
}

pub fn argmax(l: &[f32]) -> u32 {
    let mut best = 0;
    for (i, v) in l.iter().enumerate() {
        if *v > l[best] {
            best = i;
        }
    }
    best as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(rng: &mut Rng, n: usize) -> Vec<f32> {
        (0..n).map(|_| (rng.next_f32() * 12.0 - 6.0) + if rng.next_f32() < 0.02 { 6.0 } else { 0.0 }).collect()
    }

    /// Every token the sparse sampler returns is the dense sampler's token from the same generator state, and the
    /// generator advanced the same way
    #[test]
    fn sparse_sampling_matches_dense_whenever_it_answers() {
        let mut rng = Rng::new(7);
        let vocab = 500;
        let k = 64;
        let mut answered = 0;
        let mut asked = 0;
        for round in 0..400 {
            let mut logits = row(&mut rng, vocab);
            if round % 5 == 0 {
                logits.iter_mut().take(200).for_each(|v| *v = (*v * 2.0).round() / 2.0);
            }
            if round % 7 == 0 {
                logits[10..60].iter_mut().for_each(|v| *v = f32::NEG_INFINITY);
            }
            let mut c = Candidates::select(&logits, k, 3).expect("select");
            assert_eq!(c.row, 3);
            let temps = [0.0, 0.5, 1.0, 1.7];
            let s = Sampling {
                temperature: temps[round % 4],
                top_k: [0, 3, 10, 40, 64, 100][round % 6],
                top_p: [1.0, 0.9, 0.5][round % 3],
                min_p: [0.0, 0.05, 0.3][(round / 3) % 3],
                repeat_penalty: [1.0, 1.3, 0.8][(round / 4) % 3],
                repeat_window: 64,
            };
            let recent: Vec<u32> = (0..(round % 9) * 4).map(|_| if rng.next_f32() < 0.5 { c.ids[(rng.next_f32() * k as f32) as usize % k] } else { (rng.next_f32() * vocab as f32) as u32 % vocab as u32 }).collect();
            let mut dense = logits.clone();
            if round % 3 == 1 {
                let id = (rng.next_f32() * vocab as f32) as u32 % vocab as u32;
                let b = rng.next_f32() * 6.0 - 3.0;
                if c.bias(id, b) {
                    dense[id as usize] += b;
                } else {
                    continue;
                }
            }
            if round % 4 == 2 {
                let id = c.ids[(rng.next_f32() * k as f32) as usize % k];
                c.mask(id);
                dense[id as usize] = f32::NEG_INFINITY;
            }
            let seed = (round as u64 + 1) * 977;
            let (mut r1, mut r2) = (Rng::new(seed), Rng::new(seed));
            asked += 1;
            if let Some(t) = sample_sparse(&c, &s, &recent, &mut r1) {
                answered += 1;
                let want = sample(&dense, &s, &recent, &mut r2);
                assert_eq!(t, want, "round {round} sampler {s:?}");
                assert_eq!(r1.next_f32(), r2.next_f32(), "round {round} generator state");
            }
        }
        assert!(answered * 3 > asked, "the candidates answered {answered} of {asked} rounds");
    }

    /// The default sampler and the greedy sampler never need the whole row on an untouched row
    #[test]
    fn default_and_greedy_sampling_stay_sparse() {
        let mut rng = Rng::new(3);
        for _ in 0..50 {
            let logits = row(&mut rng, 2000);
            let c = Candidates::select(&logits, 64, 0).expect("select");
            assert!(sample_sparse(&c, &Sampling::default(), &[], &mut Rng::new(1)).is_some());
            let greedy = Sampling { temperature: 0.0, ..Default::default() };
            assert_eq!(sample_sparse(&c, &greedy, &[], &mut Rng::new(1)), Some(sample(&logits, &greedy, &[], &mut Rng::new(1))));
            assert_eq!(c.argmax(), argmax(&logits));
        }
    }

    /// A candidate's log probability is its logit less the row's log sum exp
    #[test]
    fn candidate_logprobs_are_log_softmax() {
        let logits = [1.0f32, 3.0, 2.0, 3.0, f32::NEG_INFINITY];
        let c = Candidates::select(&logits, 3, 0).expect("select");
        assert_eq!(c.ids, vec![1, 3, 2]);
        assert_eq!(c.floor, 2.0);
        let lse = (1f32.exp() + 3f32.exp() + 2f32.exp() + 3f32.exp()).ln();
        assert!((c.logprob(2).expect("held") - (2.0 - lse)).abs() < 1e-5);
        assert_eq!(c.logprob(0), None);
        let packed = [1.0, 3.0, 2.0, 3.0, 3.0, 2.0, lse];
        assert_eq!(Candidates::from_packed(&packed, 3, 5, 4).expect("packed"), Candidates { ids: vec![1, 3, 2], logits: vec![3.0, 3.0, 2.0], touched: vec![false; 3], floor: 2.0, lse, vocab: 5, row: 4 });
    }
}

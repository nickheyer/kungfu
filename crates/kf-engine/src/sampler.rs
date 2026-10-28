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
    let mut l = logits.to_vec();
    if s.repeat_penalty != 1.0 {
        let window = &recent[recent.len().saturating_sub(s.repeat_window)..];
        for t in window {
            let Some(v) = l.get_mut(*t as usize) else { continue };
            *v = if *v > 0.0 { *v / s.repeat_penalty } else { *v * s.repeat_penalty };
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
    let max = l[idx[0]];
    let mut probs: Vec<f32> = idx.iter().map(|i| ((l[*i] - max) / s.temperature).exp()).collect();
    let sum: f32 = probs.iter().sum();
    probs.iter_mut().for_each(|p| *p /= sum);
    if s.min_p > 0.0 {
        let floor = probs[0] * s.min_p;
        let keep = probs.iter().take_while(|p| **p >= floor).count().max(1);
        idx.truncate(keep);
        probs.truncate(keep);
    }
    if s.top_p < 1.0 {
        let mut acc = 0.0;
        let mut keep = 0;
        for p in &probs {
            acc += p;
            keep += 1;
            if acc >= s.top_p {
                break;
            }
        }
        idx.truncate(keep);
        probs.truncate(keep);
    }
    let sum: f32 = probs.iter().sum();
    let mut r = rng.next_f32() * sum;
    for (i, p) in idx.iter().zip(&probs) {
        if r < *p {
            return *i as u32;
        }
        r -= p;
    }
    *idx.last().unwrap_or(&0) as u32
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

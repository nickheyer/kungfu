/// Which experts of a layer sit in device memory, warmed from a routing profile and adapted as it runs
pub struct ExpertCache {
    pub layer: usize,
    pub slots: usize,
    pub resident: Vec<Option<usize>>,
    pub hits: u64,
    pub misses: u64,
    stamp: Vec<u64>,
    clock: u64,
}

impl ExpertCache {
    pub fn new(layer: usize, slots: usize) -> Self {
        Self { layer, slots, resident: vec![None; slots], hits: 0, misses: 0, stamp: vec![0; slots], clock: 0 }
    }

    /// Fills the slots from a routing profile, most used experts first
    pub fn warm(&mut self, profile: &[u32]) {
        expert_cache_warm(self, profile)
    }

    /// Slot of an expert, swapping one in when it is absent
    pub fn lookup(&mut self, expert: usize) -> usize {
        expert_cache_lookup(self, expert)
    }

    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 { 0.0 } else { self.hits as f64 / total as f64 }
    }
}

/// Slots taken by the experts with the highest counts, ties resolved by the lower expert id
pub fn expert_cache_warm(cache: &mut ExpertCache, profile: &[u32]) {
    let mut order: Vec<(usize, u32)> = profile.iter().copied().enumerate().collect();
    order.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    cache.clock += 1;
    for (s, (e, _)) in order.into_iter().take(cache.slots).enumerate() {
        cache.resident[s] = Some(e);
        cache.stamp[s] = cache.clock;
    }
}

/// The resident slot of an expert, else the least recently used slot reassigned to it
pub fn expert_cache_lookup(cache: &mut ExpertCache, expert: usize) -> usize {
    cache.clock += 1;
    if let Some(s) = cache.resident.iter().position(|r| *r == Some(expert)) {
        cache.hits += 1;
        cache.stamp[s] = cache.clock;
        return s;
    }
    cache.misses += 1;
    let victim = (0..cache.slots).min_by_key(|s| (cache.resident[*s].is_some(), cache.stamp[*s])).unwrap_or(0);
    cache.resident[victim] = Some(expert);
    cache.stamp[victim] = cache.clock;
    victim
}

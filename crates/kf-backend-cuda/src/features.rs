//! The rows the passes scored, each with its slot, position, and token, kept in a ring on the device for a draft head
//! to learn from. Two rows of one slot at neighbouring positions and the same epoch form a training pair, the epoch
//! of a slot rising whenever its rows roll back, so no pair crosses a rollback.
use crate::CudaBackend;
use crate::train::Tensor;
use anyhow::{Result, ensure};
use cudarc::driver::result;
use std::collections::HashMap;

/// Rows the ring holds at most
pub const RING_ROWS: usize = kf_core::FEATURE_RING_ROWS;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Entry {
    pub seq: u32,
    pub pos: u32,
    pub token: u32,
    pub epoch: u32,
    pub live: bool,
}

pub struct FeatureRing {
    feats: Tensor,
    pub cap: usize,
    pub cols: usize,
    head: usize,
    pub count: usize,
    /// Rows appended since the ring was made, those it has since overwritten included
    pub appended: u64,
    meta: Vec<Entry>,
    epoch: Vec<u32>,
}

impl FeatureRing {
    pub fn new(b: &CudaBackend, cap: usize, cols: usize, n_seq: usize) -> Result<Self> {
        ensure!(cap >= 2 && cols >= 1, "a ring of {cap} rows of {cols}");
        Ok(Self { feats: Tensor::zeros(b, cap, cols, "feature ring")?, cap, cols, head: 0, count: 0, appended: 0, meta: vec![Entry::default(); cap], epoch: vec![0; n_seq.max(1)] })
    }

    pub fn ptr(&self) -> u64 {
        self.feats.ptr()
    }

    /// The epoch a slot's rows join now
    pub fn epoch(&self, seq: usize) -> u32 {
        self.epoch.get(seq).copied().unwrap_or(0)
    }

    /// Ends the pairs of a slot, whose rows roll back
    pub fn roll_back(&mut self, seq: usize) {
        if let Some(e) = self.epoch.get_mut(seq) {
            *e += 1;
        }
    }

    /// Ends the pairs of every slot
    pub fn roll_back_all(&mut self) {
        self.epoch.iter_mut().for_each(|e| *e += 1);
    }

    /// Appends n rows read from src, row i of them standing for the entry rows[i], copied on the stream
    pub fn append(&mut self, stream: &cudarc::driver::CudaStream, src: u64, rows: &[(u32, u32, u32)]) -> Result<()> {
        let n = rows.len();
        ensure!(n <= self.cap, "{n} rows into a ring of {}", self.cap);
        let mut at = 0;
        while at < n {
            let span = (n - at).min(self.cap - self.head);
            let bytes = span * self.cols * 4;
            unsafe { result::memcpy_dtod_async(self.feats.ptr() + (self.head * self.cols * 4) as u64, src + (at * self.cols * 4) as u64, bytes, stream.cu_stream()) }?;
            for i in 0..span {
                let (seq, pos, token) = rows[at + i];
                self.meta[self.head + i] = Entry { seq, pos, token, epoch: self.epoch.get(seq as usize).copied().unwrap_or(0), live: true };
            }
            self.head = (self.head + span) % self.cap;
            at += span;
        }
        self.count = (self.count + n).min(self.cap);
        self.appended += n as u64;
        Ok(())
    }

    /// Every pair of ring rows (i, j) where j holds the row after i in the same slot and epoch
    pub fn pairs(&self) -> Vec<(u32, u32)> {
        let mut by_key: HashMap<(u32, u32, u32), u32> = HashMap::with_capacity(self.count);
        for (i, e) in self.meta.iter().enumerate() {
            if e.live {
                by_key.insert((e.seq, e.epoch, e.pos), i as u32);
            }
        }
        let mut out = Vec::with_capacity(self.count);
        for (i, e) in self.meta.iter().enumerate() {
            if e.live
                && let Some(j) = by_key.get(&(e.seq, e.epoch, e.pos + 1))
            {
                out.push((i as u32, *j));
            }
        }
        out
    }

    pub fn entry(&self, i: usize) -> Entry {
        self.meta[i]
    }

    /// Runs of ring rows of one slot and epoch at consecutive positions, each as the ring indices in position order
    pub fn runs(&self) -> Vec<Vec<u32>> {
        let mut live: Vec<(u32, u32, u32, u32)> = self.meta.iter().enumerate().filter(|(_, e)| e.live).map(|(i, e)| (e.seq, e.epoch, e.pos, i as u32)).collect();
        live.sort_unstable();
        let mut runs: Vec<Vec<u32>> = Vec::new();
        for (k, e) in live.iter().enumerate() {
            let joins = k > 0 && live[k - 1].0 == e.0 && live[k - 1].1 == e.1 && live[k - 1].2 + 1 == e.2;
            if joins {
                runs.last_mut().expect("a run").push(e.3);
            } else {
                runs.push(vec![e.3]);
            }
        }
        runs
    }

}

/// The ring rows at the given indices, gathered on the device and brought to the host
pub fn ring_rows(b: &mut CudaBackend, idx: &[u32]) -> Result<Vec<f32>> {
    let (ptr, cols) = {
        let ring = b.ring.as_ref().ok_or_else(|| anyhow::anyhow!("no feature ring yet"))?;
        (ring.ptr(), ring.cols)
    };
    let ids: Vec<i32> = idx.iter().map(|i| *i as i32).collect();
    let idx_dev = crate::exec::dev_upload(&b.stream, b.pool(), &ids, kf_core::AllocKind::Compute, "ring index")?;
    let out = Tensor::zeros(b, idx.len().max(1), cols, "ring rows")?;
    crate::train::gather_rows(b, ptr, crate::exec::dptr(&idx_dev, &b.stream), out.ptr(), idx.len(), cols)?;
    out.download(b)
}

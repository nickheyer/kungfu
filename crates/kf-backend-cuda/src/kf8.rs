//! The KF8 weight form on the device and the tensor core product on it, see docs/kf8-format.md.
//!
//! Every block quant type unpacks through its own `unpack_i8` kernel into int8 integers with an f32 scale and
//! optional offset per group. The product is one kernel per group size and offset presence. The unpacked copy
//! is kept per weight while the device has room for it and otherwise made into a shared scratch just before
//! each product.

use crate::exec::{Arg, Launch, dev_alloc, dptr};
use crate::registry::KernelRegistry;
use crate::sources;
use anyhow::{Context, Result, ensure};
use cudarc::driver::result;
use cudarc::driver::{CudaSlice, CudaStream};
use kf_core::{AllocKind, Pool, Wid};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

pub const THREADS: u32 = 256;

/// Device shape of one KF8 tensor: integers, scales, offsets, and the block width of the activations it multiplies
#[derive(Clone, Copy, Debug)]
pub struct Form {
    pub g: usize,
    pub has_m: bool,
    /// Activation block of the type's rounding, 32 for Q8_0 and Q8_1 kinds and 256 for Q8K
    pub w: usize,
    /// Scales of more than 22 significant bits, the f32 scale of q8_K, which the fused magic recovery cannot carry exactly
    pub wide: bool,
    pub n_in: usize,
    pub n_out: usize,
}

impl Form {
    pub fn of(ty: u32, n_in: usize, n_out: usize) -> Option<Self> {
        let (g, has_m) = sources::kf8_form(ty)?;
        let q = kf_quant::by_id(ty)?;
        let w = match kf_quant::act_kind(q) {
            kf_quant::ActKind::Q8K => 256,
            _ => 32,
        };
        Some(Self { g: g as usize, has_m, w, wide: sources::kf8_wide(ty), n_in, n_out })
    }

    /// Whether the product kernel can run this shape: the activation block must divide the columns
    pub fn fits(&self) -> bool {
        self.n_in % self.w == 0
    }

    pub fn groups(&self) -> usize {
        self.n_in / self.g
    }

    /// Offsets of the scale and offset arrays from the integer base, each aligned to 256
    fn layout(&self) -> (usize, usize) {
        let d = (self.n_out * self.n_in).next_multiple_of(256);
        let m = (d + self.n_out * self.groups() * 4).next_multiple_of(256);
        (d, m)
    }

    /// Total bytes of one buffer holding all three arrays with alignment
    pub fn buffer_bytes(&self) -> usize {
        let (_, m) = self.layout();
        m + if self.has_m { self.n_out * self.groups() * 4 } else { 0 }
    }

    /// The kernel module of the form, the 128 deep two stage module for 32 wide groups and the 64 deep three stage one for 16
    pub fn module(&self) -> &'static str {
        if self.g == 32 { "kf8w" } else { "kf8" }
    }

    /// The k step and pipeline stages of the form's module
    fn pipeline(&self) -> (usize, usize) {
        if self.g == 32 { (128, 2) } else { (64, 3) }
    }

    /// Bytes of one pipeline stage of a tile, the two int8 tiles and the scale and offset words
    fn stage_bytes(&self, tile: Tile) -> usize {
        let (bk, _) = self.pipeline();
        let ng = bk / self.g;
        let (bm, bn) = tile.dims();
        let tiles = (bm + bn) * bk;
        let scales = (bm + bn) * ng * 4;
        let offsets = if self.has_m { (bm + bn) * ng * 4 } else { 0 };
        tiles + scales + offsets
    }

    /// Dynamic shared memory of the product kernel for this form and tile
    pub fn smem(&self, tile: Tile) -> u32 {
        (self.pipeline().1 * self.stage_bytes(tile)) as u32
    }

    /// The mid tile when its grid fills every SM twice over or the type carries offsets, whose extra scale words
    /// cost more in the small tile, else the small tile
    pub fn tile(&self, rows: usize, sms: usize) -> Tile {
        if let Ok(v) = std::env::var("KF_KF8_TILE") {
            return if v == "mid" { Tile::Mid } else { Tile::Small };
        }
        let (bm, bn) = Tile::Mid.dims();
        if self.has_m || rows.div_ceil(bm) * self.n_out.div_ceil(bn) >= 2 * sms { Tile::Mid } else { Tile::Small }
    }

    pub fn entry(&self, tile: Tile) -> String {
        let w = if self.w == 32 { 32 } else { 256 };
        let suffix = match (self.has_m, self.wide) {
            (true, _) => "_m",
            (false, true) => "_f32",
            (false, false) => "",
        };
        format!("gemm_{}_g{}_w{w}{suffix}", tile.name(), self.g)
    }
}

/// Block tiles of the product, rows by outputs
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tile {
    Mid,
    Small,
    /// 32 rows, for routed experts whose pairs are few, only the expert entries exist
    Tiny,
}

impl Tile {
    pub fn dims(self) -> (usize, usize) {
        match self {
            Tile::Mid => (128, 128),
            Tile::Small => (64, 128),
            Tile::Tiny => (32, 128),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Tile::Mid => "mid",
            Tile::Small => "small",
            Tile::Tiny => "tiny",
        }
    }
}

/// The three device pointers of a KF8 tensor
#[derive(Clone, Copy, Debug)]
pub struct Ptrs {
    pub q: u64,
    pub d: u64,
    pub m: u64,
}

impl Ptrs {
    pub fn at(base: u64, f: &Form) -> Self {
        let (d, m) = f.layout();
        Self { q: base, d: base + d as u64, m: if f.has_m { base + m as u64 } else { 0 } }
    }
}

/// Unpacked copies kept per weight and one scratch for the rest
#[derive(Default)]
pub struct Store {
    resident: HashMap<Wid, (CudaSlice<u8>, u64, Form)>,
    /// Resident copies of groups of weights stacked along their rows, keyed by the members in order
    groups: HashMap<Vec<Wid>, (CudaSlice<u8>, u64)>,
    groups_declined: HashSet<Vec<Wid>>,
    scratch: Option<(CudaSlice<u8>, u64)>,
    /// Weights the device had no room for, not retried
    declined: HashSet<Wid>,
    /// Bytes kept resident so far
    pub resident_bytes: u64,
    /// Device bytes that stay free after a resident copy, so state and compute buffers made later still fit
    pub reserve: u64,
    /// Device bytes the plan places as weights, state, and expert slots, allocated as layers first run
    pub planned: u64,
    /// Whether the scratch moved since the flag was last taken, so launches holding its address are rebuilt
    pub scratch_grew: bool,
}

impl Store {
    /// The reserve from the device size, an eighth of it and at least a gigabyte, fixed at first use
    fn reserve(&mut self) -> u64 {
        if self.reserve == 0 {
            let total = result::mem_get_info().map_or(0, |(_, t)| t as u64);
            self.reserve = (total / 8).max(1 << 30);
        }
        self.reserve
    }

    /// Whether a weight has a resident copy
    pub fn is_resident(&self, w: Wid) -> bool {
        self.resident.contains_key(&w)
    }

    /// Device bytes a resident copy must leave free: the reserve plus what the plan has yet to allocate
    fn headroom(&mut self, pool: Pool) -> u64 {
        let held: u64 = [AllocKind::Weights, AllocKind::State, AllocKind::Experts].iter().map(|k| kf_core::event::total(pool, *k)).sum();
        let pending = self.planned.saturating_sub(held.saturating_sub(self.resident_bytes));
        self.reserve() + pending
    }

    /// Whether the device has room for a resident copy of bytes beyond the headroom
    fn room_for(&mut self, pool: Pool, bytes: usize) -> bool {
        let free = result::mem_get_info().map_or(0, |(f, _)| f as u64);
        free > bytes as u64 + self.headroom(pool)
    }

    /// The shared scratch grown to at least bytes, its address and whether it moved, which stales every launch holding the old one
    pub fn scratch_for(&mut self, stream: &Arc<CudaStream>, pool: Pool, bytes: usize) -> Result<(u64, bool)> {
        let have = self.scratch.as_ref().map_or(0, |s| s.0.len());
        if have >= bytes {
            return Ok((self.scratch.as_ref().map(|s| s.1).context("kf8 scratch")?, false));
        }
        self.scratch = None;
        let buf = dev_alloc::<u8>(stream, pool, bytes, AllocKind::Compute, "kf8 scratch")?;
        let base = dptr(&buf, stream);
        self.scratch = Some((buf, base));
        Ok((base, true))
    }

    /// The unpack launch of a native weight into the KF8 arrays
    pub fn unpack_launch(k: &mut KernelRegistry, ty: u32, src: u64, row_bytes: i64, f: &Form, p: Ptrs) -> Result<Launch> {
        let n16 = f.n_out * f.n_in / 16;
        Ok(Launch {
            func: k.quant_function(ty, "unpack_i8")?,
            grid: (n16.div_ceil(256).clamp(1, 1 << 16) as u32, 1, 1),
            block: 256,
            smem: 0,
            args: vec![Arg::P(src), Arg::P(p.q), Arg::P(p.d), Arg::P(p.m), Arg::I(f.n_in as i32), Arg::L(f.n_out as i64), Arg::L(row_bytes), Arg::I(f.g as i32)],
        })
    }

    /// Pointers of the KF8 copy of a weight, unpacked now into a resident buffer when the device has room beyond the reserve,
    /// else into the shared scratch with the unpack launch pushed before the product
    #[allow(clippy::too_many_arguments)]
    pub fn ensure(&mut self, k: &mut KernelRegistry, stream: &Arc<CudaStream>, pool: Pool, w: Wid, ty: u32, src: u64, row_bytes: i64, f: Form, v: &mut Vec<Launch>) -> Result<Ptrs> {
        if let Some((_, base, _)) = self.resident.get(&w) {
            return Ok(Ptrs::at(*base, &f));
        }
        let bytes = f.buffer_bytes();
        if !self.declined.contains(&w) {
            if self.room_for(pool, bytes) {
                if let Ok(buf) = dev_alloc::<u8>(stream, pool, bytes, AllocKind::Weights, "kf8 copy") {
                    let base = dptr(&buf, stream);
                    let p = Ptrs::at(base, &f);
                    Self::unpack_launch(k, ty, src, row_bytes, &f, p)?.run(stream)?;
                    self.resident_bytes += bytes as u64;
                    self.resident.insert(w, (buf, base, f));
                    return Ok(p);
                }
            }
        }
        self.declined.insert(w);
        let (base, grew) = self.scratch_for(stream, pool, bytes)?;
        self.scratch_grew |= grew;
        let p = Ptrs::at(base, &f);
        v.push(Self::unpack_launch(k, ty, src, row_bytes, &f, p)?);
        Ok(p)
    }

    /// The product launch of routed pairs grouped by slot against the KF8 form of n_slots stacked experts at p
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_moe(&mut self, k: &mut KernelRegistry, f: &Form, p: Ptrs, xq: u64, xd: u64, xs: u64, order: u64, offsets: u64, x_per_slot: bool, y: u64, rows_max: usize, n_used: usize, slot0: usize, n_slots: usize, rows_per_expert: usize) -> Result<Launch> {
        ensure!(f.fits(), "kf8 product over {} columns with activation block {}", f.n_in, f.w);
        // Experts averaging under 48 pairs waste most of a 64 row tile
        let tile = if rows_per_expert < 48 { Tile::Tiny } else { Tile::Small };
        let (bm, bn) = tile.dims();
        let smem = f.smem(tile);
        let func = k.function_smem(f.module(), &format!("{}_moe", f.entry(tile)), smem)?;
        Ok(Launch {
            func,
            grid: (rows_max.div_ceil(bm) as u32, f.n_out.div_ceil(bn) as u32, n_slots as u32),
            block: THREADS,
            smem,
            args: vec![Arg::P(p.q), Arg::P(p.d), Arg::P(p.m), Arg::P(xq), Arg::P(xd), Arg::P(xs), Arg::P(order), Arg::P(offsets), Arg::I(x_per_slot as i32), Arg::P(y), Arg::I(f.n_in as i32), Arg::I(f.n_out as i32), Arg::I(n_used as i32), Arg::I(slot0 as i32)],
        })
    }

    /// Pointers of a resident stacked copy of several weights of one type and width, unpacked now when the device has
    /// room beyond the reserve, none when it has not
    #[allow(clippy::too_many_arguments)]
    pub fn ensure_group(&mut self, k: &mut KernelRegistry, stream: &Arc<CudaStream>, pool: Pool, members: &[(Wid, u64, i64, usize)], ty: u32, f: &Form) -> Result<Option<Ptrs>> {
        let key: Vec<Wid> = members.iter().map(|m| m.0).collect();
        if let Some((_, base)) = self.groups.get(&key) {
            return Ok(Some(Ptrs::at(*base, f)));
        }
        if self.groups_declined.contains(&key) {
            return Ok(None);
        }
        let bytes = f.buffer_bytes();
        if self.room_for(pool, bytes)
            && let Ok(buf) = dev_alloc::<u8>(stream, pool, bytes, AllocKind::Weights, "kf8 group copy")
        {
            let base = dptr(&buf, stream);
            let p = Ptrs::at(base, f);
            let mut start = 0usize;
            for &(_, src, row_bytes, n_out) in members {
                let part = Form { n_out, ..*f };
                let ng = f.groups();
                let at = Ptrs { q: p.q + (start * f.n_in) as u64, d: p.d + (start * ng * 4) as u64, m: if f.has_m { p.m + (start * ng * 4) as u64 } else { 0 } };
                Self::unpack_launch(k, ty, src, row_bytes, &part, at)?.run(stream)?;
                start += n_out;
            }
            self.resident_bytes += bytes as u64;
            self.groups.insert(key, (buf, base));
            return Ok(Some(p));
        }
        self.groups_declined.insert(key);
        Ok(None)
    }

    /// The product launch of rows of int8 activations against the KF8 arrays, the residual rows at res added when nonzero,
    /// or against a stacked group whose column tiles write through the nseg entries of the segment table at segs
    #[allow(clippy::too_many_arguments)]
    pub fn gemm(&mut self, k: &mut KernelRegistry, f: &Form, p: Ptrs, xq: u64, xd: u64, xs: u64, bias: u64, res: u64, y: u64, rows: usize, segs: u64, nseg: usize) -> Result<Launch> {
        ensure!(f.fits(), "kf8 product over {} columns with activation block {}", f.n_in, f.w);
        let tile = f.tile(rows, k.sm_count());
        let (bm, bn) = tile.dims();
        let smem = f.smem(tile);
        let func = k.function_smem(f.module(), &f.entry(tile), smem)?;
        Ok(Launch {
            func,
            grid: (rows.div_ceil(bm) as u32, f.n_out.div_ceil(bn) as u32, 1),
            block: THREADS,
            smem,
            args: vec![Arg::P(p.q), Arg::P(p.d), Arg::P(p.m), Arg::P(xq), Arg::P(xd), Arg::P(xs), Arg::P(bias), Arg::P(res), Arg::P(y), Arg::I(rows as i32), Arg::I(f.n_in as i32), Arg::I(f.n_out as i32), Arg::P(segs), Arg::I(nseg as i32)],
        })
    }

    /// Drops every copy, used when the device is reclaimed
    pub fn clear(&mut self) {
        self.resident.clear();
        self.groups.clear();
        self.groups_declined.clear();
        self.scratch = None;
        self.declined.clear();
        self.resident_bytes = 0;
    }
}

/// The int8 activation scratch laid out in one float buffer of rows times n: the int8 rows, then the scales and integer
/// sums per 16 columns, then per 32, every array sixteen byte aligned
pub fn act_scratch_all(base: u64, rows: usize, n: usize) -> [u64; 5] {
    let up = |x: u64| x.next_multiple_of(16);
    let q = base;
    let s16 = up(q + (rows * n) as u64);
    let m16 = up(s16 + (rows * n / 16 * 4) as u64);
    let s32 = up(m16 + (rows * n / 16 * 4) as u64);
    let m32 = up(s32 + (rows * n / 32 * 4) as u64);
    [q, s16, m16, s32, m32]
}

/// The integers, scales, and sums a product over groups of g columns reads, g being 16 or 32
pub fn act_scratch(base: u64, rows: usize, n: usize, g: usize) -> (u64, u64, u64) {
    let [q, s16, m16, s32, m32] = act_scratch_all(base, rows, n);
    if g == 16 { (q, s16, m16) } else { (q, s32, m32) }
}

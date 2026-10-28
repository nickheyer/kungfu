use crate::{Buf, CpuBackend};
use anyhow::{Result, bail};
use kf_core::{Inputs, Op, RopeKind, RopeParams, WeightSource};
use std::f32::consts::PI;

/// Frequency table state shared by every row of one rope call
pub struct RopeTable<'a> {
    p: &'a RopeParams,
    dims: usize,
    low: f32,
    high: f32,
    mscale: f32,
    theta_scale: f32,
    factors: Option<&'a [f32]>,
}

impl<'a> RopeTable<'a> {
    pub fn new(p: &'a RopeParams, head_dim: usize, factors: Option<&'a [f32]>) -> Result<Self> {
        let dims = p.dims.min(head_dim);
        if let Some(f) = factors
            && f.len() < dims / 2
        {
            bail!("rope factors of {} for {} frequencies", f.len(), dims / 2);
        }
        let (low, high) = match p.yarn {
            Some(y) => {
                let corr = |rot: f32| (dims as f32 * (p.orig_ctx as f32 / (rot * 2.0 * PI)).ln()) / (2.0 * p.base.ln());
                (corr(y.beta_fast).floor().max(0.0), corr(y.beta_slow).ceil().min(dims as f32 - 1.0))
            }
            None => (0.0, 0.0),
        };
        let mscale = match p.yarn {
            Some(y) if y.ext_factor != 0.0 => p.attn_factor * (1.0 + 0.1 * (1.0 / p.scale).ln()),
            _ => p.attn_factor,
        };
        Ok(Self { p, dims, low, high, mscale, theta_scale: p.base.powf(-2.0 / dims as f32), factors })
    }

    pub fn dims(&self) -> usize {
        self.dims
    }

    fn yarn(&self, theta_extrap: f32, i: usize) -> f32 {
        let interp = self.p.scale * theta_extrap;
        match self.p.yarn {
            Some(y) if y.ext_factor != 0.0 => {
                let span = (self.high - self.low).max(0.001);
                let ramp = 1.0 - ((i as f32 - self.low) / span).clamp(0.0, 1.0);
                let mix = ramp * y.ext_factor;
                interp * (1.0 - mix) + theta_extrap * mix
            }
            _ => interp,
        }
    }

    /// Cos and sin per rotated pair for one token, section rules picking the position stream
    pub fn cache(&self, pos: [f32; 4], inverse: bool, out: &mut Vec<f32>) {
        let half = self.dims / 2;
        out.clear();
        let sign = if inverse { -1.0 } else { 1.0 };
        let s = self.p.sections;
        let sect = s.iter().sum::<usize>().max(1);
        let mut theta = pos;
        if self.p.kind == RopeKind::Vision {
            let s0 = s[0].max(1);
            for i in 0..half {
                let (stream, k) = if i < s0 { (pos[0], i) } else { (pos[1], i - s0) };
                let t = stream * self.p.base.powf(-2.0 * k as f32 / half as f32);
                out.push(t.cos() * self.mscale);
                out.push(t.sin() * self.mscale * sign);
            }
            return;
        }
        for i in 0..half {
            let ff = self.factors.map_or(1.0, |f| f[i]);
            let base = match self.p.kind {
                RopeKind::Normal | RopeKind::Neox | RopeKind::Vision => theta[0],
                RopeKind::MRope => {
                    let sector = i % sect;
                    if sector < s[0] {
                        theta[0]
                    } else if sector < s[0] + s[1] {
                        theta[1]
                    } else if sector < s[0] + s[1] + s[2] {
                        theta[2]
                    } else {
                        theta[3]
                    }
                }
                RopeKind::IMRope => {
                    let sector = i % sect;
                    if sector % 3 == 1 && sector < 3 * s[1] {
                        theta[1]
                    } else if sector % 3 == 2 && sector < 3 * s[2] {
                        theta[2]
                    } else if sector % 3 == 0 && sector < 3 * s[0] {
                        theta[0]
                    } else {
                        theta[3]
                    }
                }
            };
            let t = self.yarn(base / ff, i);
            out.push(t.cos() * self.mscale);
            out.push(t.sin() * self.mscale * sign);
            theta.iter_mut().for_each(|v| *v *= self.theta_scale);
        }
    }

    /// Rotates the dims from offset of one head row with a prepared cache
    pub fn rotate(&self, head: &mut [f32], offset: usize, cache: &[f32]) {
        let half = self.dims / 2;
        for i in 0..half {
            let (a, b) = match self.p.kind {
                RopeKind::Normal => (offset + 2 * i, offset + 2 * i + 1),
                _ => (offset + i, offset + i + half),
            };
            let (c, s) = (cache[2 * i], cache[2 * i + 1]);
            let x0 = head[a];
            let x1 = head[b];
            head[a] = x0 * c - x1 * s;
            head[b] = x0 * s + x1 * c;
        }
    }
}

/// Position streams of a text token, the fourth stream zero unless the caller pins it to the position
pub fn text_pos(p: u32, fourth: bool) -> [f32; 4] {
    let f = p as f32;
    [f, f, f, if fourth { f } else { 0.0 }]
}

/// Rotary embedding over every head of each row, positions given per row
#[allow(clippy::too_many_arguments)]
pub fn apply(x: &Buf, positions: &[[f32; 4]], p: &RopeParams, n_head: usize, head_dim: usize, offset: usize, inverse: bool, factors: Option<&[f32]>) -> Result<Buf> {
    if x.cols != n_head * head_dim {
        bail!("rope over {} columns, expected {} heads of {}", x.cols, n_head, head_dim);
    }
    if positions.len() != x.rows {
        bail!("rope with {} positions for {} rows", positions.len(), x.rows);
    }
    let table = RopeTable::new(p, head_dim.saturating_sub(offset), factors)?;
    if offset + table.dims() > head_dim {
        bail!("rope of {} dims at offset {offset} inside heads of {head_dim}", table.dims());
    }
    let mut out = x.clone();
    let mut cache = Vec::with_capacity(table.dims());
    for r in 0..out.rows {
        table.cache(positions[r], inverse, &mut cache);
        let row = out.row_mut(r);
        for h in 0..n_head {
            table.rotate(&mut row[h * head_dim..(h + 1) * head_dim], offset, &cache);
        }
    }
    Ok(out)
}

/// Rope and the positional query temperature, both reading the positions input
pub fn exec(b: &mut CpuBackend, op: &Op, ins: &[Buf], inputs: &Inputs, weights: &dyn WeightSource) -> Result<Buf> {
    if ins.len() < 2 {
        bail!("{} inputs for an op needing x and positions", ins.len());
    }
    let x = &ins[0];
    let pos = &ins[1];
    if pos.data.len() != x.rows {
        bail!("{} positions for {} rows", pos.data.len(), x.rows);
    }
    match op {
        Op::Rope { params, n_head, head_dim, offset, inverse } => {
            let factors = match params.factors {
                Some(f) => Some(b.small(f, weights)?.to_vec()),
                None => None,
            };
            let positions: Vec<[f32; 4]> = if inputs.rope.is_empty() {
                pos.data.iter().map(|p| text_pos(*p as u32, false)).collect()
            } else {
                if inputs.rope.len() != x.rows {
                    bail!("{} rotary rows for {} rows", inputs.rope.len(), x.rows);
                }
                inputs.rope.iter().map(|r| [r[0] as f32, r[1] as f32, r[2] as f32, r[3] as f32]).collect()
            };
            apply(x, &positions, params, *n_head, *head_dim, *offset, *inverse, factors.as_deref())
        }
        Op::AttnTemp { floor_scale, scale, offset, pow } => {
            let mut out = x.clone();
            for r in 0..out.rows {
                let p = pos.data[r] + offset;
                let s = if *pow { (1.0 + p / floor_scale).powf(*scale) } else { ((p / floor_scale).floor() + 1.0).ln() * scale + 1.0 };
                out.row_mut(r).iter_mut().for_each(|v| *v *= s);
            }
            Ok(out)
        }
        other => bail!("{other:?} routed to the rope module"),
    }
}

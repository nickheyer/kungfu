//! Image decoding, the reference's fixed point separable resize, normalization, and patch rows in tower order
use crate::{Hparams, ImageInput, Projector};
use anyhow::{Result, bail};

const PRECISION_BITS: u32 = 22;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Algo {
    Bilinear,
    Bicubic,
}

/// An 8 bit RGB image, rows top down, pixels interleaved
#[derive(Clone, Debug)]
pub struct Rgb {
    pub w: usize,
    pub h: usize,
    pub data: Vec<u8>,
}

/// Decodes PNG, JPEG, WebP, GIF, or BMP bytes to RGB
pub fn decode(bytes: &[u8]) -> Result<Rgb> {
    let img = image::load_from_memory(bytes).map_err(|e| anyhow::anyhow!("decode image: {e}"))?.to_rgb8();
    Ok(Rgb { w: img.width() as usize, h: img.height() as usize, data: img.into_raw() })
}

fn filter(algo: Algo, x: f64) -> f64 {
    let x = x.abs();
    match algo {
        Algo::Bilinear => {
            if x < 1.0 {
                1.0 - x
            } else {
                0.0
            }
        }
        Algo::Bicubic => {
            let a = -0.5;
            if x < 1.0 {
                ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
            } else if x < 2.0 {
                (((x - 5.0) * x + 8.0) * x - 4.0) * a
            } else {
                0.0
            }
        }
    }
}

/// Fixed point filter weights and input ranges for one axis, as the reference computes them
fn weights(algo: Algo, in_size: usize, out_size: usize) -> (usize, Vec<(usize, usize)>, Vec<i32>) {
    let support0 = match algo {
        Algo::Bilinear => 1.0,
        Algo::Bicubic => 2.0,
    };
    let scale = in_size as f64 / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = support0 * filterscale;
    let ksize = (support.ceil() as usize) * 2 + 1;
    let mut pre = vec![0.0f64; out_size * ksize];
    let mut bounds = Vec::with_capacity(out_size);
    for xx in 0..out_size {
        let center = (xx as f64 + 0.5) * scale;
        let ss = 1.0 / filterscale;
        let xmin = ((center - support + 0.5) as i64).max(0) as usize;
        let xmax = ((center + support + 0.5) as i64).min(in_size as i64) as usize;
        let cnt = xmax.saturating_sub(xmin);
        let mut ww = 0.0;
        for x in 0..cnt {
            let w = filter(algo, ((x + xmin) as f64 - center + 0.5) * ss);
            pre[xx * ksize + x] = w;
            ww += w;
        }
        if ww != 0.0 {
            for x in 0..cnt {
                pre[xx * ksize + x] /= ww;
            }
        }
        bounds.push((xmin, cnt));
    }
    let fxp = (1u64 << PRECISION_BITS) as f64;
    let w: Vec<i32> = pre.iter().map(|v| (v * fxp + if *v < 0.0 { -0.5 } else { 0.5 }) as i32).collect();
    (ksize, bounds, w)
}

fn clip8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

fn horizontal(src: &Rgb, out_w: usize, algo: Algo) -> Rgb {
    let (ksize, bounds, k) = weights(algo, src.w, out_w);
    let mut out = vec![0u8; out_w * src.h * 3];
    let bias = 1i32 << (PRECISION_BITS - 1);
    for y in 0..src.h {
        let row = &src.data[y * src.w * 3..(y + 1) * src.w * 3];
        for xx in 0..out_w {
            let (xmin, cnt) = bounds[xx];
            let kk = &k[xx * ksize..xx * ksize + cnt];
            let mut acc = [bias; 3];
            for (x, w) in kk.iter().enumerate() {
                let p = &row[(xmin + x) * 3..(xmin + x) * 3 + 3];
                acc[0] += p[0] as i32 * w;
                acc[1] += p[1] as i32 * w;
                acc[2] += p[2] as i32 * w;
            }
            let o = (y * out_w + xx) * 3;
            out[o] = clip8(acc[0] >> PRECISION_BITS);
            out[o + 1] = clip8(acc[1] >> PRECISION_BITS);
            out[o + 2] = clip8(acc[2] >> PRECISION_BITS);
        }
    }
    Rgb { w: out_w, h: src.h, data: out }
}

fn vertical(src: &Rgb, out_h: usize, algo: Algo) -> Rgb {
    let (ksize, bounds, k) = weights(algo, src.h, out_h);
    let row_elems = src.w * 3;
    let mut out = vec![0u8; row_elems * out_h];
    let bias = 1i32 << (PRECISION_BITS - 1);
    let mut acc = vec![0i32; row_elems];
    for yy in 0..out_h {
        let (ymin, cnt) = bounds[yy];
        acc.iter_mut().for_each(|a| *a = bias);
        for y in 0..cnt {
            let w = k[yy * ksize + y];
            let row = &src.data[(ymin + y) * row_elems..(ymin + y + 1) * row_elems];
            for (a, p) in acc.iter_mut().zip(row) {
                *a += *p as i32 * w;
            }
        }
        for (o, a) in out[yy * row_elems..(yy + 1) * row_elems].iter_mut().zip(&acc) {
            *o = clip8(*a >> PRECISION_BITS);
        }
    }
    Rgb { w: src.w, h: out_h, data: out }
}

/// Separable resize with the reference's fixed point filters, the horizontal pass first
pub fn resize(src: &Rgb, w: usize, h: usize, algo: Algo) -> Result<Rgb> {
    if w == 0 || h == 0 || w > 65536 || h > 65536 {
        bail!("resize target {w} by {h} is out of range");
    }
    let mut cur = src.clone();
    if w != cur.w {
        cur = horizontal(&cur, w, algo);
    }
    if h != cur.h {
        cur = vertical(&cur, h, algo);
    }
    Ok(cur)
}

/// Aspect preserving resize centered on a black canvas of the target size, the reference's f32 ceil padding
pub fn resize_padded(src: &Rgb, w: usize, h: usize, algo: Algo) -> Result<Rgb> {
    if src.w == w && src.h == h {
        return Ok(src.clone());
    }
    let scale = (w as f32 / src.w as f32).min(h as f32 / src.h as f32);
    let nw = ((src.w as f32 * scale).ceil() as usize).min(w);
    let nh = ((src.h as f32 * scale).ceil() as usize).min(h);
    let inner = resize(src, nw, nh, algo)?;
    let mut out = Rgb { w, h, data: vec![0u8; w * h * 3] };
    let (ox, oy) = ((w - nw) / 2, (h - nh) / 2);
    for y in 0..nh {
        let dst = ((oy + y) * w + ox) * 3;
        out.data[dst..dst + nw * 3].copy_from_slice(&inner.data[y * nw * 3..(y + 1) * nw * 3]);
    }
    Ok(out)
}

fn round_half_even(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 && r as i64 % 2 != 0 { r - x.signum() } else { r }
}

/// The reference's size with the aspect ratio kept, aligned to the merged patch and within the pixel bounds
pub fn dynamic_size(w: usize, h: usize, align: usize, min_pixels: usize, max_pixels: usize) -> (usize, usize) {
    let f = align as f64;
    let round_by = |x: f64| (round_half_even(x / f) as usize) * align;
    let ceil_by = |x: f64| ((x / f).ceil() as usize) * align;
    let floor_by = |x: f64| ((x / f).floor() as usize) * align;
    let mut wb = round_by(w as f64).max(align);
    let mut hb = round_by(h as f64).max(align);
    if max_pixels > 0 && hb * wb > max_pixels {
        let beta = ((h * w) as f64 / max_pixels as f64).sqrt();
        hb = floor_by(h as f64 / beta).max(align);
        wb = floor_by(w as f64 / beta).max(align);
    } else if min_pixels > 0 && hb * wb < min_pixels {
        let beta = (min_pixels as f64 / (h * w) as f64).sqrt();
        hb = ceil_by(h as f64 * beta);
        wb = ceil_by(w as f64 * beta);
    }
    (wb, hb)
}

/// One patch's values in channel, row, column order, normalized
fn patch_row(img: &Rgb, px: usize, py: usize, ps: usize, mean: &[f32; 3], std: &[f32; 3], out: &mut Vec<f32>) {
    for c in 0..3 {
        for ky in 0..ps {
            for kx in 0..ps {
                let v = img.data[((py * ps + ky) * img.w + px * ps + kx) * 3 + c] as f32 / 255.0;
                out.push((v - mean[c]) / std[c]);
            }
        }
    }
}

/// Decodes, resizes as the projector family does, and unfolds patches in the order its tower reads them
pub fn prepare(h: &Hparams, bytes: &[u8]) -> Result<ImageInput> {
    let src = decode(bytes)?;
    if src.w == 0 || src.h == 0 {
        bail!("image has no pixels");
    }
    let ps = h.patch_size;
    let (img, order): (Rgb, Vec<(usize, usize)>) = match h.projector {
        Projector::Qwen2Vl => {
            let (w, hgt) = dynamic_size(src.w, src.h, ps * h.n_merge, h.min_pixels, h.max_pixels);
            let img = resize(&src, w, hgt, Algo::Bicubic)?;
            let (nx, ny) = (w / ps, hgt / ps);
            let mut order = Vec::with_capacity(nx * ny);
            for by in (0..ny).step_by(2) {
                for bx in (0..nx).step_by(2) {
                    for dy in 0..2 {
                        for dx in 0..2 {
                            order.push((bx + dx, by + dy));
                        }
                    }
                }
            }
            (img, order)
        }
        Projector::Gemma3 => {
            let img = resize_padded(&src, h.image_size, h.image_size, Algo::Bilinear)?;
            let n = h.image_size / ps;
            let order = (0..n * n).map(|i| (i % n, i / n)).collect();
            (img, order)
        }
    };
    let (nx, ny) = (img.w / ps, img.h / ps);
    let mut patches = Vec::with_capacity(nx * ny * 3 * ps * ps);
    let mut rope = Vec::with_capacity(nx * ny);
    for (px, py) in &order {
        patch_row(&img, *px, *py, ps, &h.mean, &h.std, &mut patches);
        rope.push([*py as u32, *px as u32, *py as u32, *px as u32]);
    }
    if h.projector == Projector::Gemma3 {
        rope.clear();
    }
    Ok(ImageInput { patches, patch_cols: 3 * ps * ps, n_patches_x: nx, n_patches_y: ny, rope, merge: h.n_merge })
}

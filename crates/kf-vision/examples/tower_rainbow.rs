//! Feeds the reference debug tool's synthetic rainbow image to the tower and prints per node sums matching its dump
use anyhow::Result;
use kf_core::Backend;
use kf_vision::{Clip, ImageInput, Projector};

/// The raspberry pi like rainbow of llama-mtmd-debug, interleaved RGB rows in f32 arithmetic
fn rainbow(n: usize) -> Vec<f32> {
    let (cx, cy) = (n as f32 / 2.0, n as f32 / 2.0);
    let max_dist = (cx * cx + cy * cy).sqrt();
    let mut img = vec![0f32; n * n * 3];
    for y in 0..n {
        for x in 0..n {
            let (dx, dy) = (x as f32 - cx, y as f32 - cy);
            let mut hue = dy.atan2(dx) / (2.0f32 * 3.14159265f32);
            if hue < 0.0 {
                hue += 1.0;
            }
            let sat = ((dx * dx + dy * dy).sqrt() / max_dist).min(1.0);
            let h6 = hue * 6.0;
            let i6 = h6 as i32;
            let f = h6 - i6 as f32;
            let (p, q, t) = (1.0 - sat, 1.0 - sat * f, 1.0 - sat * (1.0 - f));
            let (r, g, b) = match i6 % 6 {
                0 => (1.0, t, p),
                1 => (q, 1.0, p),
                2 => (p, 1.0, t),
                3 => (p, q, 1.0),
                4 => (t, p, 1.0),
                _ => (1.0, p, q),
            };
            let o = (y * n + x) * 3;
            img[o] = r;
            img[o + 1] = g;
            img[o + 2] = b;
        }
    }
    img
}

/// Patch rows of a raw float image in the tower's order, values rounded through f16 as the reference's im2col does
fn input(clip: &Clip, img: &[f32], n: usize) -> ImageInput {
    let h = &clip.hparams;
    let ps = h.patch_size;
    let side = n / ps;
    let order: Vec<(usize, usize)> = match h.projector {
        Projector::Qwen2Vl => {
            let mut o = Vec::new();
            for by in (0..side).step_by(2) {
                for bx in (0..side).step_by(2) {
                    for dy in 0..2 {
                        for dx in 0..2 {
                            o.push((bx + dx, by + dy));
                        }
                    }
                }
            }
            o
        }
        Projector::Gemma3 => (0..side * side).map(|i| (i % side, i / side)).collect(),
    };
    let mut patches = Vec::with_capacity(side * side * 3 * ps * ps);
    let mut rope = Vec::new();
    for (px, py) in &order {
        for c in 0..3 {
            for ky in 0..ps {
                for kx in 0..ps {
                    let v = img[((py * ps + ky) * n + px * ps + kx) * 3 + c];
                    patches.push(half::f16::from_f32(v).to_f32());
                }
            }
        }
        rope.push([*py as u32, *px as u32, *py as u32, *px as u32]);
    }
    if h.projector == Projector::Gemma3 {
        rope.clear();
    }
    ImageInput { patches, patch_cols: 3 * ps * ps, n_patches_x: side, n_patches_y: side, rope, merge: h.n_merge }
}

/// Rows of a probe tensor, the patch count or the merged token count, whichever gives a tower width
fn rows_of(clip: &Clip, img: &ImageInput, len: usize) -> usize {
    let h = &clip.hparams;
    let widths = [h.n_embd, h.n_ff, h.projection_dim, 4 * h.n_embd, h.n_embd * h.n_merge * h.n_merge];
    let patches = img.n_patches();
    let merged = patches / (img.merge * img.merge);
    [patches, merged].into_iter().find(|r| *r > 0 && len % r == 0 && widths.contains(&(len / r))).unwrap_or(patches)
}

/// The reference dump's sample, the first and last three rows with their first and last three values
fn show(cut: Option<usize>, name: &str, v: &[f32], rows: usize) {
    let cols = v.len() / rows.max(1);
    let sum: f64 = v.iter().map(|x| *x as f64).sum();
    let amax = v.iter().fold(0f32, |m, x| m.max(x.abs()));
    println!("cut {cut:?} {name}: {rows}x{cols} sum {sum:.4} amax {amax:.4}");
    let ends = |r: &[f32]| format!("[{:11.4}, {:11.4}, {:11.4},    ..., {:11.4}, {:11.4}, {:11.4}]", r[0], r[1], r[2], r[cols - 3], r[cols - 2], r[cols - 1]);
    let picks: Vec<usize> = if rows > 6 { vec![0, 1, 2, rows - 3, rows - 2, rows - 1] } else { (0..rows).collect() };
    for r in picks {
        println!("    {}", ends(&v[r * cols..(r + 1) * cols]));
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let clip = Clip::load(&args[1])?;
    let n: usize = args[3].parse()?;
    let img = input(&clip, &rainbow(n), n);
    println!("patches {}x{} merge {}", img.n_patches_x, img.n_patches_y, img.merge);
    let cuts: Vec<Option<usize>> = args[4].split(',').map(|c| if c == "end" { None } else { Some(c.parse::<usize>().expect("cut index")) }).collect();
    let plan = kf_core::Plan { layers: Vec::new(), pinned: Vec::new(), weights_bytes: 0, state_bytes: 0, n_ctx: 64, n_batch: 64, n_seq: 1 };
    for which in args[2].split(',') {
        let mut backend: Box<dyn Backend> = match which {
            "cpu" => Box::new(kf_backend_cpu::CpuBackend::new()),
            _ => Box::new(kf_backend_cuda::CudaBackend::new(0, None)?),
        };
        backend.apply_plan(&plan)?;
        for cut in &cuts {
            let t0 = std::time::Instant::now();
            let (v, name) = clip.probe(backend.as_mut(), &img, *cut)?;
            let rows = rows_of(&clip, &img, v.len());
            print!("{which} {:.1}s ", t0.elapsed().as_secs_f64());
            show(*cut, &name, &v, rows);
        }
    }
    Ok(())
}

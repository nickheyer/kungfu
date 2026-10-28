//! Encodes an image with the tower on the host and on the first device and prints what comes out
use anyhow::Result;
use kf_core::Backend;
use kf_vision::Clip;

fn stats(name: &str, v: &[f32]) {
    let nan = v.iter().filter(|x| !x.is_finite()).count();
    let amax = v.iter().filter(|x| x.is_finite()).fold(0f32, |m, x| m.max(x.abs()));
    let mean: f64 = v.iter().filter(|x| x.is_finite()).map(|x| *x as f64).sum::<f64>() / v.len().max(1) as f64;
    println!("{name}: {} values, {nan} non finite, amax {amax:.4}, mean {mean:.5}, head {:?}", v.len(), &v[..v.len().min(6)]);
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let clip = Clip::load(&args[1])?;
    let img = clip.preprocess(&std::fs::read(&args[2])?)?;
    println!("patches {}x{} merge {} rope rows {} patch cols {}", img.n_patches_x, img.n_patches_y, img.merge, img.rope.len(), img.patch_cols);
    stats("patches", &img.patches);
    let plan = kf_core::Plan { layers: Vec::new(), pinned: Vec::new(), weights_bytes: 0, state_bytes: 0, n_ctx: 64, n_batch: 64, n_seq: 1 };
    let mut outs = Vec::new();
    for which in args[3].split(',') {
        let mut backend: Box<dyn Backend> = match which {
            "cpu" => Box::new(kf_backend_cpu::CpuBackend::new()),
            _ => Box::new(kf_backend_cuda::CudaBackend::new(0, None)?),
        };
        backend.apply_plan(&plan)?;
        let t0 = std::time::Instant::now();
        let enc = clip.encode(backend.as_mut(), &img)?;
        println!("{which}: {} tokens of {} in {:.2}s", enc.n_tokens(), enc.n_embd, t0.elapsed().as_secs_f64());
        stats(which, &enc.embeds);
        outs.push(enc.embeds);
    }
    if outs.len() == 2 {
        let n_embd = outs[0].len() / img.n_merged_x().max(1) / img.n_merged_y().max(1);
        let mut rows: Vec<(usize, f32, usize)> = outs[0]
            .chunks(n_embd)
            .zip(outs[1].chunks(n_embd))
            .enumerate()
            .map(|(r, (a, b))| {
                let (mut worst, mut col) = (0f32, 0usize);
                for (c, (x, y)) in a.iter().zip(b).enumerate() {
                    let d = (x - y).abs();
                    if d > worst {
                        worst = d;
                        col = c;
                    }
                }
                (r, worst, col)
            })
            .collect();
        rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        println!("rows with the largest differences: {:?}", &rows[..rows.len().min(6)]);
        let bad = rows.iter().filter(|r| r.1 > 0.5).count();
        println!("rows differing by more than 0.5: {bad} of {}", rows.len());
    }
    Ok(())
}

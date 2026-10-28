//! Compares the tower's tensor after given nodes between the host and the first device
use anyhow::Result;
use kf_core::Backend;
use kf_vision::Clip;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let clip = Clip::load(&args[1])?;
    let img = clip.preprocess(&std::fs::read(&args[2])?)?;
    let plan = kf_core::Plan { layers: Vec::new(), pinned: Vec::new(), weights_bytes: 0, state_bytes: 0, n_ctx: 64, n_batch: 64, n_seq: 1 };
    let mut cpu: Box<dyn Backend> = Box::new(kf_backend_cpu::CpuBackend::new());
    let mut gpu: Box<dyn Backend> = Box::new(kf_backend_cuda::CudaBackend::new(0, None)?);
    cpu.apply_plan(&plan)?;
    gpu.apply_plan(&plan)?;
    println!("patches {}x{}", img.n_patches_x, img.n_patches_y);
    for cut in args[3].split(',') {
        let cut = if cut == "end" { None } else { Some(cut.parse::<usize>()?) };
        let (a, name) = clip.probe(cpu.as_mut(), &img, cut)?;
        let (b, _) = clip.probe(gpu.as_mut(), &img, cut)?;
        let rows = img.n_patches();
        let cols = a.len() / rows.max(1);
        let (mut worst, mut at) = (0f32, (0usize, 0usize));
        for (i, (x, y)) in a.iter().zip(&b).enumerate() {
            let d = (x - y).abs();
            if d > worst {
                worst = d;
                at = (i / cols.max(1), i % cols.max(1));
            }
        }
        let amax = a.iter().fold(0f32, |m, v| m.max(v.abs()));
        let nan = b.iter().filter(|v| !v.is_finite()).count();
        println!("cut {cut:?} {name}: {} values, host amax {amax:.4}, worst diff {worst:.5} at row {} col {}, device non finite {nan}", a.len(), at.0, at.1);
    }
    Ok(())
}

//! Feeds a prompt then forced tokens one at a time on both backends and prints the top logits at every step
use anyhow::Result;
use kf_engine::{BackendChoice, CtxChoice, Session};

fn top(logits: &[f32]) -> String {
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|a, b| logits[*b].partial_cmp(&logits[*a]).unwrap_or(std::cmp::Ordering::Equal));
    idx.iter().take(5).map(|i| format!("({i} {:.2})", logits[*i])).collect::<Vec<_>>().join(" ")
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let loaded = kf_engine::load(&args[1])?;
    let prompt: Vec<u32> = serde_json::from_str(&args[2])?;
    let forced: Vec<u32> = serde_json::from_str(&args[3])?;
    for choice in [BackendChoice::Cpu, BackendChoice::Cuda { device: 0 }] {
        let p = kf_engine::prepare(&loaded, choice, CtxChoice::Tokens(4096), 512, 1, None, None, None, &kf_engine::Scope::all(loaded.model.n_layers()))?;
        let mut s = Session::new(&loaded, p.backend, &p.plan);
        let l = s.feed(0, &prompt)?;
        println!("{choice:?} prefill: {}", top(&l));
        for t in &forced {
            let l = s.feed(0, &[*t])?;
            println!("{choice:?} after {t}: {}", top(&l));
        }
    }
    Ok(())
}

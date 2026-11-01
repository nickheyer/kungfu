//! Feeds a prompt then forced tokens one at a time on both backends and prints the top logits at every step
use anyhow::Result;
use kf_engine::{BackendChoice, CtxChoice, Session};

fn top(c: &kf_engine::Candidates) -> String {
    c.ids.iter().zip(&c.logits).take(5).map(|(i, v)| format!("({i} {v:.2})")).collect::<Vec<_>>().join(" ")
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let loaded = kf_engine::load(&args[1])?;
    let prompt: Vec<u32> = serde_json::from_str(&args[2])?;
    let forced: Vec<u32> = serde_json::from_str(&args[3])?;
    for choice in [BackendChoice::Cpu, BackendChoice::Cuda { device: 0 }] {
        let p = kf_engine::prepare(&loaded, choice, CtxChoice::Tokens(4096), 512, 1, None, None, None, &kf_engine::Scope::all(loaded.model.n_layers()), 0)?;
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

//! Tokenizes a text with a file's tokenizer and walks the token pieces through a schema grammar, printing where it stops
use anyhow::Result;
use kf_serve::grammar::{Constraint, GrammarBuilder};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let gguf = kf_gguf::Gguf::open(&args[1])?;
    let tok = kf_serve::Tok::from_gguf(&gguf.meta)?;
    let schema: serde_json::Value = serde_json::from_str(&args[2])?;
    let text = &args[3];
    let mut b = GrammarBuilder::new();
    let node = b.json(&schema)?;
    let mut c = Constraint::new(b.finish(node));
    let ids = tok.encode(text, false)?;
    for (i, id) in ids.iter().enumerate() {
        let piece = tok.piece(*id, i == 0);
        match c.try_advance(piece) {
            Some(n) => {
                println!("ok  {id:>7} {:?}", String::from_utf8_lossy(piece));
                c = n;
            }
            None => {
                println!("REJ {id:>7} {:?}", String::from_utf8_lossy(piece));
                break;
            }
        }
    }
    println!("complete {} can_continue {}", c.complete(), c.can_continue());
    Ok(())
}

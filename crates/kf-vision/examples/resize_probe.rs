//! Resizes an image with the tower's resampler, padded when asked, and writes raw RGB bytes for comparison against Pillow
use anyhow::Result;
use kf_vision::preprocess::{Algo, decode, resize, resize_padded};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let img = decode(&std::fs::read(&args[1])?)?;
    let w: usize = args[2].parse()?;
    let h: usize = args[3].parse()?;
    let algo = if args[4] == "bilinear" { Algo::Bilinear } else { Algo::Bicubic };
    let out = if args.get(6).is_some_and(|a| a == "pad") { resize_padded(&img, w, h, algo)? } else { resize(&img, w, h, algo)? };
    std::fs::write(&args[5], &out.data)?;
    println!("{} {} {}", out.w, out.h, out.data.len());
    Ok(())
}

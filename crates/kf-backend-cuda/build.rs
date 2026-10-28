//! Embeds PTX for every kernel module and supported arch in release builds when nvcc is found, the embed-ptx feature insisting on it
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

const ARCHS: &[u32] = &[75, 80, 86, 89, 90, 100, 120];

fn main() {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    let kernels = Path::new("src/kernels");
    println!("cargo:rerun-if-changed=src/kernels");
    println!("cargo:rerun-if-env-changed=PATH");
    println!("cargo:rerun-if-env-changed=KF_SKIP_PTX");
    let mut table = String::from("pub static EMBEDDED_PTX: &[(&str, u32, &str)] = &[\n");
    let insist = std::env::var_os("CARGO_FEATURE_EMBED_PTX").is_some();
    let release = std::env::var("PROFILE").is_ok_and(|p| p == "release") && std::env::var_os("KF_SKIP_PTX").is_none();
    let found = nvcc_found();
    assert!(!insist || found, "the embed-ptx feature needs nvcc on the path");
    if found && (insist || release) {
        let mut sources: Vec<PathBuf> = std::fs::read_dir(kernels)
            .expect("read src/kernels")
            .map(|e| e.expect("kernel dir entry").path())
            .filter(|p| p.extension().is_some_and(|x| x == "cu"))
            .collect();
        sources.sort();
        for src in &sources {
            let module = src.file_stem().and_then(|s| s.to_str()).expect("kernel file name");
            for arch in ARCHS {
                let ptx = out_dir.join(format!("{module}.sm_{arch}.ptx"));
                let status = Command::new("nvcc")
                    .args(["--ptx", "-std=c++17", &format!("-arch=compute_{arch}"), "-I"])
                    .arg(kernels)
                    .arg("-o")
                    .arg(&ptx)
                    .arg(src)
                    .status()
                    .expect("run nvcc");
                assert!(status.success(), "nvcc failed on {} for sm_{arch}", src.display());
                writeln!(table, "    ({module:?}, {arch}, include_str!({:?})),", ptx.display().to_string()).expect("format table");
            }
        }
    }
    table.push_str("];\n");
    std::fs::write(out_dir.join("embedded_ptx.rs"), table).expect("write embedded_ptx.rs");
}

fn nvcc_found() -> bool {
    Command::new("nvcc").arg("--version").output().is_ok_and(|o| o.status.success())
}

mod common;

use common::*;

/// Device time of the int8 products alone, run with --ignored --nocapture, KF_BENCH_TYPES and KF_BENCH_SHAPES as rows x n_in x n_out
#[test]
#[ignore]
fn product_kernels() {
    let mut rng = Rng(0x1234_5678_9abc_def1);
    let shapes = std::env::var("KF_BENCH_SHAPES").unwrap_or_else(|_| "512x2048x16384,512x2048x2048,512x1024x4096,1x2048x16384,1x2048x2048,1x2048x3072".into());
    let types = std::env::var("KF_BENCH_TYPES").unwrap_or_else(|_| "q8_0,q4_K,q6_K,iq2_xxs".into());
    let mut gpu = cuda();
    for shape in shapes.split(',') {
        let dims: Vec<usize> = shape.split('x').map(|v| v.parse().expect("shape")).collect();
        let (rows, n_in, n_out) = (dims[0], dims[1], dims[2]);
        for name in types.split(',') {
            let q = kf_quant::by_name(name).expect("type");
            let bytes = synth(&mut rng, q.info().id, n_out, n_in);
            let wbytes = bytes.len() as f64;
            let (t, t_unpack) = kf_backend_cuda::bench::product_us(&mut gpu, q.info().id, &bytes, rows, n_in, n_out, 20).expect("bench");
            let flops = 2.0 * rows as f64 * n_in as f64 * n_out as f64;
            if rows <= 8 {
                eprintln!("{name:8} {rows}x{n_in}x{n_out}: matvec {t:8.1} us  {:6.0} GB/s", wbytes / t / 1e3);
            } else {
                eprintln!("{name:8} {rows}x{n_in}x{n_out}: gemm {t:8.1} us  {:6.1} TOPS   unpack {t_unpack:6.1} us", flops / t / 1e6);
            }
        }
    }
}

/// The fragment product against the tile and the matvec on the same bytes, run with --ignored --nocapture, KF_BENCH_TYPES
/// and KF_BENCH_SHAPES as for product_kernels
#[test]
#[ignore]
fn frag_kinds() {
    let mut rng = Rng(0x1234_5678_9abc_def1);
    let shapes = std::env::var("KF_BENCH_SHAPES").unwrap_or_else(|_| "16x2048x16384,16x8192x2048,16x2048x2048,8x2048x16384,24x2048x16384,32x2048x16384,2x2048x16384,16x2048x128256".into());
    let types = std::env::var("KF_BENCH_TYPES").unwrap_or_else(|_| "q4_K".into());
    let mut gpu = cuda();
    for shape in shapes.split(',') {
        let dims: Vec<usize> = shape.split('x').map(|v| v.parse().expect("shape")).collect();
        let (rows, n_in, n_out) = (dims[0], dims[1], dims[2]);
        for name in types.split(',') {
            let q = kf_quant::by_name(name).expect("type");
            let bytes = synth(&mut rng, q.info().id, n_out, n_in);
            let wbytes = bytes.len() as f64;
            for kind in [kf_backend_cuda::Multi::Frag, kf_backend_cuda::Multi::Tile] {
                match kf_backend_cuda::bench::product_kind_us(&mut gpu, q.info().id, &bytes, rows, n_in, n_out, 20, Some(kind)) {
                    Ok((t, _)) => eprintln!("{name:8} {rows}x{n_in}x{n_out} {kind:?}: {t:8.1} us  {:6.0} GB/s", wbytes / t / 1e3),
                    Err(e) => eprintln!("{name:8} {rows}x{n_in}x{n_out} {kind:?}: {e:#}"),
                }
            }
        }
    }
}

/// The fragment product under its bench knobs, run with --ignored --nocapture, KF_BENCH_TYPES and KF_BENCH_SHAPES as for
/// product_kernels and KF_BENCH_VARIANTS a comma list of the variant labels to run
#[test]
#[ignore]
fn frag_variants() {
    let mut rng = Rng(0x1234_5678_9abc_def1);
    let shapes = std::env::var("KF_BENCH_SHAPES").unwrap_or_else(|_| "16x2048x16384,2x2048x16384".into());
    let types = std::env::var("KF_BENCH_TYPES").unwrap_or_else(|_| "q4_K".into());
    let variants: Vec<(&str, &str, Option<usize>, Option<usize>)> = vec![
        ("base", "", None, None),
        ("stages3", "#define KF_FRAG_STAGES 3", Some(3), None),
        ("warps4", "#define KF_FRAG_WARPS 4", None, Some(4)),
        ("warps8", "#define KF_FRAG_WARPS 8", None, Some(8)),
        ("warps4 stages3", "#define KF_FRAG_WARPS 4\n#define KF_FRAG_STAGES 3", Some(3), Some(4)),
        ("no mma", "#define KF_FRAG_NO_MMA 1", None, None),
        ("no b", "#define KF_FRAG_BFAKE 1", None, None),
        ("no xs", "#define KF_FRAG_NO_XS 1", None, None),
        ("no b no xs", "#define KF_FRAG_BFAKE 1\n#define KF_FRAG_NO_XS 1", None, None),
        ("no mma no b no xs", "#define KF_FRAG_NO_MMA 1\n#define KF_FRAG_BFAKE 1\n#define KF_FRAG_NO_XS 1", None, None),
        ("no decode", "#define KF_FRAG_NO_DECODE 1", None, None),
        ("stream only", "#define KF_FRAG_NO_MMA 1\n#define KF_FRAG_BFAKE 1\n#define KF_FRAG_NO_XS 1\n#define KF_FRAG_NO_DECODE 1", None, None),
    ];
    let only = std::env::var("KF_BENCH_VARIANTS").ok();
    let mut gpu = cuda();
    for shape in shapes.split(',') {
        let dims: Vec<usize> = shape.split('x').map(|v| v.parse().expect("shape")).collect();
        let (rows, n_in, n_out) = (dims[0], dims[1], dims[2]);
        for name in types.split(',') {
            let q = kf_quant::by_name(name).expect("type");
            let bytes = synth(&mut rng, q.info().id, n_out, n_in);
            let wbytes = bytes.len() as f64;
            for (label, defines, stages, warps) in &variants {
                if only.as_ref().is_some_and(|o| !o.split(',').any(|v| v == *label)) {
                    continue;
                }
                match kf_backend_cuda::bench::frag_variant_us(&mut gpu, defines, *stages, *warps, q.info().id, &bytes, rows, n_in, n_out, 20) {
                    Ok(t) => eprintln!("{name:8} {rows}x{n_in}x{n_out} {label:18}: {t:8.1} us  {:6.0} GB/s", wbytes / t / 1e3),
                    Err(e) => eprintln!("{name:8} {rows}x{n_in}x{n_out} {label:18}: {e:#}"),
                }
            }
        }
    }
}

/// The KF8 product under source variants, run with --ignored --nocapture, KF_BENCH_TYPES and KF_BENCH_SHAPES as for product_kernels
#[test]
#[ignore]
fn gemm_variants() {
    let mut rng = Rng(0x1234_5678_9abc_def1);
    let shapes = std::env::var("KF_BENCH_SHAPES").unwrap_or_else(|_| "512x2048x16384,512x2048x2048".into());
    let types = std::env::var("KF_BENCH_TYPES").unwrap_or_else(|_| "q8_0,q4_K".into());
    // name, defines, bm, bn, bk, stages
    let variants: Vec<(&str, &str, usize, usize, usize, usize)> = vec![
        ("mid 64x3", "", 128, 128, 64, 3),
        ("small 64x3", "", 64, 128, 64, 3),
        ("mid 128x2", "#define KF8_BK 128\n#define KF8_STAGES 2", 128, 128, 128, 2),
        ("small 128x2", "#define KF8_BK 128\n#define KF8_STAGES 2", 64, 128, 128, 2),
        ("small 128x3", "#define KF8_BK 128\n#define KF8_STAGES 3\n#define KF8_SMALL_BLOCKS 1", 64, 128, 128, 3),
        ("mid 64x2", "#define KF8_STAGES 2", 128, 128, 64, 2),
        ("mid 64x4", "#define KF8_STAGES 4", 128, 128, 64, 4),
        ("mid 64x3 no m", "#define KF8_EXPERIMENT_NO_M", 128, 128, 64, 3),
        ("mid 64x3 direct", "#define KF8_DIRECT", 128, 128, 64, 3),
        ("mid 128x2 direct", "#define KF8_DIRECT\n#define KF8_BK 128\n#define KF8_STAGES 2", 128, 128, 128, 2),
        ("small 64x3 1blk", "#define KF8_SMALL_BLOCKS 1", 64, 128, 64, 3),
        ("small 64x3 direct", "#define KF8_DIRECT", 64, 128, 64, 3),
        ("small 128x2 dir1b", "#define KF8_DIRECT\n#define KF8_BK 128\n#define KF8_STAGES 2\n#define KF8_SMALL_BLOCKS 1", 64, 128, 128, 2),
    ];
    let mut gpu = cuda();
    for shape in shapes.split(',') {
        let dims: Vec<usize> = shape.split('x').map(|v| v.parse().expect("shape")).collect();
        let (rows, n_in, n_out) = (dims[0], dims[1], dims[2]);
        for name in types.split(',') {
            let q = kf_quant::by_name(name).expect("type");
            let bytes = synth(&mut rng, q.info().id, n_out, n_in);
            let flops = 2.0 * rows as f64 * n_in as f64 * n_out as f64;
            for (label, defines, bm, bn, bk, stages) in &variants {
                match kf_backend_cuda::bench::gemm_variant_us(&mut gpu, defines, q.info().id, &bytes, rows, n_in, n_out, *bm, *bn, *bk, *stages, 20) {
                    Ok(t) => eprintln!("{name:8} {rows}x{n_in}x{n_out} {label:16}: {t:8.1} us  {:6.1} TOPS", flops / t / 1e6),
                    Err(e) => eprintln!("{name:8} {rows}x{n_in}x{n_out} {label:16}: {e:#}"),
                }
            }
        }
    }
}

/// The integer sub scale product on synthetic data from a kernel source named by KF_BENCH_K8_SRC, run with --ignored --nocapture
#[test]
#[ignore]
fn gemmk_variants() {
    let src = std::fs::read_to_string(std::env::var("KF_BENCH_K8_SRC").expect("KF_BENCH_K8_SRC names the kernel source")).expect("read kernel source");
    let shapes = std::env::var("KF_BENCH_SHAPES").unwrap_or_else(|_| "512x2048x16384,512x2048x2048".into());
    // label, defines, g, has_m, bm, bn, threads
    let variants: Vec<(&str, &str, usize, bool, usize, usize, u32)> = vec![
        ("g32 mid", "", 32, false, 128, 64, 512),
        ("g32m mid", "", 32, true, 128, 64, 512),
        ("g16 mid", "", 16, false, 128, 64, 512),
        ("g16m mid", "", 16, true, 128, 64, 512),
        ("g32 small", "", 32, false, 64, 64, 256),
        ("g32m small", "", 32, true, 64, 64, 256),
        ("g16 small", "", 16, false, 64, 64, 256),
        ("g32 small1b", "#define K8_SMALL_BLOCKS 1", 32, false, 64, 64, 256),
    ];
    let mut gpu = cuda();
    for shape in shapes.split(',') {
        let dims: Vec<usize> = shape.split('x').map(|v| v.parse().expect("shape")).collect();
        let (rows, n_in, n_out) = (dims[0], dims[1], dims[2]);
        let flops = 2.0 * rows as f64 * n_in as f64 * n_out as f64;
        for (label, defines, g, has_m, bm, bn, threads) in &variants {
            match kf_backend_cuda::bench::gemmk_us(&mut gpu, &src, defines, *g, *has_m, rows, n_in, n_out, *bm, *bn, *threads, 20) {
                Ok(t) => eprintln!("k8 {rows}x{n_in}x{n_out} {label:14}: {t:8.1} us  {:6.1} TOPS", flops / t / 1e6),
                Err(e) => eprintln!("k8 {rows}x{n_in}x{n_out} {label:14}: {e:#}"),
            }
        }
    }
}

/// The decode attention over head layouts, key counts, part counts, and rope folding, run with --ignored --nocapture
#[test]
#[ignore]
fn attention_variants() {
    let layouts: Vec<(&str, usize, usize, usize)> = vec![("llama 32/8 hd64", 64, 32, 8), ("qwen3 16/8 hd128", 128, 16, 8), ("qwen2vl 12/2 hd128", 128, 12, 2), ("gemma 8/4 hd256", 256, 8, 4)];
    let mut gpu = cuda();
    for (name, hd, n_head, n_head_kv) in &layouts {
        for nkeys in [128usize, 1024, 4096] {
            for parts in [1usize, 2, 4, 20] {
                for fused in [false, true] {
                    match kf_backend_cuda::bench::attention_us(&mut gpu, *hd, *n_head, *n_head_kv, nkeys, parts, fused, 50) {
                        Ok(t) => eprintln!("attn {name:20} keys {nkeys:5} parts {parts:2} fused {}: {t:7.1} us", fused as u8),
                        Err(e) => eprintln!("attn {name:20} keys {nkeys:5} parts {parts:2} fused {}: {e:#}", fused as u8),
                    }
                }
            }
        }
    }
}

/// The int8 matvec under source variants and block shapes, run with --ignored --nocapture, KF_BENCH_TYPES and
/// KF_BENCH_SHAPES as rows x n_in x n_out, KF_BENCH_VARIANTS as label=defines;... with defines separated by |
#[test]
#[ignore]
fn matvec_variants() {
    let mut rng = Rng(0x1234_5678_9abc_def1);
    let shapes = std::env::var("KF_BENCH_SHAPES").unwrap_or_else(|_| "1x8192x2048,1x2048x2048".into());
    let types = std::env::var("KF_BENCH_TYPES").unwrap_or_else(|_| "q4_K,q6_K,q8_0".into());
    let variants = std::env::var("KF_BENCH_VARIANTS").unwrap_or_else(|_| "base=".into());
    let wprs = std::env::var("KF_BENCH_WPR").unwrap_or_else(|_| "1,2,4,8".into());
    let mut gpu = cuda();
    for shape in shapes.split(',') {
        let dims: Vec<usize> = shape.split('x').map(|v| v.parse().expect("shape")).collect();
        let (rows, n_in, n_out) = (dims[0], dims[1], dims[2]);
        for name in types.split(',') {
            let q = kf_quant::by_name(name).expect("type");
            let bytes = synth(&mut rng, q.info().id, n_out, n_in);
            let wbytes = bytes.len() as f64;
            for v in variants.split(';') {
                let (label, defines) = v.split_once('=').expect("label=defines");
                let defines = defines.replace('|', "\n");
                for wpr in wprs.split(',') {
                    let wpr: usize = wpr.parse().expect("warps per row");
                    match kf_backend_cuda::bench::matvec_variant_us(&mut gpu, &defines, q.info().id, &bytes, rows, n_in, n_out, wpr, 50) {
                        Ok(t) => eprintln!("{name:6} {rows}x{n_in}x{n_out} {label:12} wpr{wpr}: {t:7.1} us {:6.0} GB/s", wbytes / t / 1e3),
                        Err(e) => eprintln!("{name:6} {rows}x{n_in}x{n_out} {label:12} wpr{wpr}: {e:#}"),
                    }
                }
            }
        }
    }
}

/// Coalesced read bandwidth over sizes of the decode weights, run with --ignored --nocapture
#[test]
#[ignore]
fn read_bandwidth() {
    let mut gpu = cuda();
    for mb in [2.36f64, 9.4, 13.8, 29.2, 215.0] {
        let bytes = (mb * 1e6) as usize / 16 * 16;
        for (bps, threads, unroll) in [(4usize, 256u32, 1usize), (4, 256, 4), (8, 256, 4), (16, 128, 4), (8, 256, 8), (32, 256, 1)] {
            match kf_backend_cuda::bench::read_us(&mut gpu, bytes, bps, threads, unroll, 50) {
                Ok(t) => eprintln!("read {mb:6.1} MB  blocks/sm {bps:2} threads {threads} unroll {unroll}: {t:7.1} us {:6.0} GB/s", bytes as f64 / t / 1e3),
                Err(e) => eprintln!("read {mb:6.1} MB: {e:#}"),
            }
        }
    }
}

/// Row streaming bandwidth of the matvec access pattern over chunk sizes and prefetch depths, run with --ignored --nocapture
#[test]
#[ignore]
fn read_rows_bandwidth() {
    let mut gpu = cuda();
    // rows, row bytes: q4_K 8192 deep, q4_K 2048 deep wide, q8_0 8960 deep
    for (nrows, row_bytes) in [(2048usize, 4608usize), (16384, 1152), (1536, 9520), (2048, 1152)] {
        for chunk in [288usize, 576, 1152] {
            if chunk > row_bytes {
                continue;
            }
            for pf in [1usize, 2, 4] {
                for warps in [8u32] {
                    match kf_backend_cuda::bench::read_rows_us(&mut gpu, nrows, row_bytes, chunk, pf, warps, 50) {
                        Ok(t) => eprintln!("rows {nrows:6} x {row_bytes:5} chunk {chunk:5} pf {pf} warps {warps}: {t:7.1} us {:6.0} GB/s", (nrows * row_bytes) as f64 / t / 1e3),
                        Err(e) => eprintln!("rows {nrows} x {row_bytes}: {e:#}"),
                    }
                }
            }
        }
    }
}

/// The decode attention under source variants over key counts, run with --ignored --nocapture, KF_BENCH_VARIANTS as
/// label=defines;... with defines separated by |
#[test]
#[ignore]
fn attention_breakdown() {
    let variants = std::env::var("KF_BENCH_VARIANTS").unwrap_or_else(|_| "base=".into());
    let mut gpu = cuda();
    for (name, hd, n_head, n_head_kv) in [("llama 32/8 hd64", 64usize, 32usize, 8usize), ("qwen2vl 12/2 hd128", 128, 12, 2)] {
        for nkeys in [1usize, 32, 128, 1024] {
            for v in variants.split(';') {
                let (label, defines) = v.split_once('=').expect("label=defines");
                let defines = defines.replace('|', "\n");
                for (fused, kind) in [(false, 0), (false, 4), (true, 4)] {
                    match kf_backend_cuda::bench::attention_variant_us(&mut gpu, &defines, hd, n_head, n_head_kv, nkeys, 20, fused, kind, 50) {
                        Ok(t) => eprintln!("attn {name:20} keys {nkeys:5} {label:12} fused {} kind {kind}: {t:7.1} us", fused as u8),
                        Err(e) => eprintln!("attn {name:20} keys {nkeys:5} {label:12}: {e:#}"),
                    }
                }
            }
        }
    }
}

/// Launch gap inside a captured graph and the cost of a grid barrier in a persistent kernel, run with --ignored --nocapture
#[test]
#[ignore]
fn persistent_probe() {
    let mut gpu = cuda();
    for n in [16usize, 64, 256] {
        match kf_backend_cuda::bench::graph_gap_us(&mut gpu, n, 20) {
            Ok(t) => eprintln!("graph of {n:3} one block kernels: {t:6.2} us per kernel"),
            Err(e) => eprintln!("graph of {n}: {e:#}"),
        }
    }
    for variant in [3usize, 4, 5, 6, 7] {
        for bps in [1usize, 2, 4] {
            for kb in [0isize, 12, -16, -64] {
                match kf_backend_cuda::bench::barrier_us(&mut gpu, variant, bps, 64, kb, true, 20) {
                    Ok(t) => eprintln!("barrier v{variant} blocks/sm {bps} read {kb:3} KB/block/stage: {t:6.2} us per stage"),
                    Err(e) => eprintln!("barrier v{variant} blocks/sm {bps} read {kb} KB: {e:#}"),
                }
            }
        }
    }
}

/// Device time of the top k kernels on rows shaped like logits, run with --ignored --nocapture
#[test]
#[ignore]
fn topk_kernels() {
    let mut rng = Rng(0x5151);
    let mut gpu = cuda();
    let cols = 128256;
    for rows in [1usize, 8, 16] {
        // Roughly normal logits around zero with a heavier right tail, as a head produces
        let logits: Vec<f32> = (0..rows * cols).map(|_| {
            let g: f32 = (0..4).map(|_| rng.unit()).sum::<f32>() - 2.0;
            g * 4.0 + if rng.unit() < 0.01 { 8.0 } else { 0.0 }
        }).collect();
        for k in [64usize, 128] {
            let t = kf_backend_cuda::bench::topk_us(&mut gpu, "", &logits, rows, cols, k, 50).expect("bench");
            eprintln!("rows {rows:3} top {k:3}: {t:7.1} us");
        }
    }
}

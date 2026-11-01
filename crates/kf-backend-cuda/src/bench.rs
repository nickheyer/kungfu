//! Device time of single kernels, the measurement the tuning loop runs against
use crate::CudaBackend;
use crate::exec::{Arg, Launch, MATVEC_ROWS, MATVEC_THREADS, act_args, dev_alloc, dev_upload, dptr, matvec_i8_launch, matvec_rows, mm_i8_fits, quant_launch, round_code, row_bytes, topk_launches, topk_slices};
use crate::kf8::{Form, Ptrs, Store, act_scratch};
use crate::kf8::Tile;
use anyhow::{Context, Result, ensure};
use cudarc::driver::{CudaSlice, sys};
use kf_core::AllocKind;
use sha2::{Digest, Sha256};

/// Microseconds per launch over iters launches after one warm launch, measured between device events
fn time_us(b: &CudaBackend, l: &Launch, iters: usize) -> Result<f64> {
    l.run(&b.stream)?;
    let e0 = b.stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
    for _ in 0..iters {
        l.run(&b.stream)?;
    }
    let e1 = b.stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
    b.stream.synchronize()?;
    Ok(e0.elapsed_ms(&e1)? as f64 * 1000.0 / iters as f64)
}

/// Device microseconds of the int8 product of rows of activations against a weight given as native bytes of a block
/// type, the tensor core tile past eight rows and the matvec up to it, then the microseconds of the unpack into KF8
pub fn product_us(b: &mut CudaBackend, ty: u32, bytes: &[u8], rows: usize, n_in: usize, n_out: usize, iters: usize) -> Result<(f64, f64)> {
    b.ctx.bind_to_thread()?;
    let pool = b.pool();
    let rb = row_bytes(ty, n_in)? as i64;
    ensure!(bytes.len() == rb as usize * n_out, "{} bytes for {n_out} rows of {rb}", bytes.len());
    let w = dev_upload(&b.stream, pool, bytes, AllocKind::Weights, "bench weight")?;
    let w_ptr = dptr(&w, &b.stream);
    let f = Form::of(ty, n_in, n_out).context("kf8 form")?;
    ensure!(f.fits(), "{n_in} columns against an activation block of {}", f.w);
    let code = round_code(ty)?;
    let x: Vec<f32> = (0..rows * n_in).map(|i| ((i as u32).wrapping_mul(2654435761) >> 8) as f32 / (1u32 << 24) as f32 - 0.5).collect();
    let xb = dev_upload(&b.stream, pool, &x, AllocKind::Compute, "bench x")?;
    let xq = dev_alloc::<f32>(&b.stream, pool, rows * n_in, AllocKind::Compute, "bench xq")?;
    let y = dev_alloc::<f32>(&b.stream, pool, rows * n_out, AllocKind::Compute, "bench y")?;
    let (xq_ptr, y_ptr) = (dptr(&xq, &b.stream), dptr(&y, &b.stream));
    quant_launch(&mut b.kernels, dptr(&xb, &b.stream), xq_ptr, rows, n_in, code)?.run(&b.stream)?;
    if rows <= matvec_rows(ty, n_in, &[]) {
        let l = matvec_i8_launch(&mut b.kernels, ty, w_ptr, xq_ptr, 0, 0, y_ptr, rows, n_in, n_out, rb, 0, 0, mm_i8_fits(ty, n_in, &[]))?;
        return Ok((time_us(b, &l, iters)?, 0.0));
    }
    let (q, scales, sums) = act_scratch(xq_ptr, rows, n_in, f.g);
    let kf = dev_alloc::<u8>(&b.stream, pool, f.buffer_bytes(), AllocKind::Weights, "bench kf8")?;
    let p = Ptrs::at(dptr(&kf, &b.stream), &f);
    let unpack = Store::unpack_launch(&mut b.kernels, ty, w_ptr, rb, &f, p)?;
    let t_unpack = time_us(b, &unpack, iters)?;
    let gemm = b.res.kf8.gemm(&mut b.kernels, &f, p, q, scales, sums, 0, 0, y_ptr, rows, 0, 0)?;
    let t_gemm = time_us(b, &gemm, iters)?;
    Ok((t_gemm, t_unpack))
}

/// Device microseconds of the top k kernel compiled from the sample module source with defines prepended, over rows of
/// the given logits, every row cols wide
pub fn topk_us(b: &mut CudaBackend, defines: &str, logits: &[f32], rows: usize, cols: usize, k: usize, iters: usize) -> Result<f64> {
    b.ctx.bind_to_thread()?;
    ensure!(logits.len() == rows * cols, "{} logits for {rows} rows of {cols}", logits.len());
    let pool = b.pool();
    let x = dev_upload(&b.stream, pool, logits, AllocKind::Compute, "bench logits")?;
    let y = dev_alloc::<f32>(&b.stream, pool, rows * (2 * k + 1), AllocKind::Compute, "bench topk")?;
    let mut h = Sha256::new();
    h.update(defines.as_bytes());
    let name = format!("samplev_{:x}", u64::from_le_bytes(h.finalize()[..8].try_into().expect("hash bytes")));
    if !b.kernels.has_module(&name) {
        let src = format!("{defines}\n{}", crate::sources::module_source("sample").context("sample source")?);
        b.kernels.load(&name, &src)?;
    }
    let s = topk_slices(cols);
    let keys = dev_alloc::<f32>(&b.stream, pool, rows * s * k * 2, AllocKind::Compute, "bench topk keys")?;
    let stats = dev_alloc::<f32>(&b.stream, pool, rows * s * 2, AllocKind::Compute, "bench topk stats")?;
    let launches = topk_launches(&mut b.kernels, &name, dptr(&x, &b.stream), dptr(&y, &b.stream), dptr(&keys, &b.stream), dptr(&stats, &b.stream), rows, cols, k)?;
    let run = |b: &CudaBackend| launches.iter().try_for_each(|l| l.run(&b.stream));
    run(b)?;
    let e0 = b.stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
    for _ in 0..iters {
        run(b)?;
    }
    let e1 = b.stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
    b.stream.synchronize()?;
    Ok(e0.elapsed_ms(&e1)? as f64 * 1000.0 / iters as f64)
}

/// Device microseconds of the KF8 product compiled from the kf8 module source with defines prepended, at a tile of
/// bm by bn with a k step of bk over stages pipeline stages, the shared memory sized to match
#[allow(clippy::too_many_arguments)]
pub fn gemm_variant_us(b: &mut CudaBackend, defines: &str, ty: u32, bytes: &[u8], rows: usize, n_in: usize, n_out: usize, bm: usize, bn: usize, bk: usize, stages: usize, iters: usize) -> Result<f64> {
    b.ctx.bind_to_thread()?;
    let pool = b.pool();
    let rb = row_bytes(ty, n_in)? as i64;
    ensure!(bytes.len() == rb as usize * n_out, "{} bytes for {n_out} rows of {rb}", bytes.len());
    let w = dev_upload(&b.stream, pool, bytes, AllocKind::Weights, "bench weight")?;
    let f = Form::of(ty, n_in, n_out).context("kf8 form")?;
    ensure!(f.fits() && rows > MATVEC_ROWS, "variant product needs the tensor core shape");
    let code = round_code(ty)?;
    let x: Vec<f32> = (0..rows * n_in).map(|i| ((i as u32).wrapping_mul(2654435761) >> 8) as f32 / (1u32 << 24) as f32 - 0.5).collect();
    let xb = dev_upload(&b.stream, pool, &x, AllocKind::Compute, "bench x")?;
    let xq = dev_alloc::<f32>(&b.stream, pool, rows * n_in, AllocKind::Compute, "bench xq")?;
    let y = dev_alloc::<f32>(&b.stream, pool, rows * n_out, AllocKind::Compute, "bench y")?;
    let (xq_ptr, y_ptr) = (dptr(&xq, &b.stream), dptr(&y, &b.stream));
    quant_launch(&mut b.kernels, dptr(&xb, &b.stream), xq_ptr, rows, n_in, code)?.run(&b.stream)?;
    let (q, scales, sums) = act_scratch(xq_ptr, rows, n_in, f.g);
    let kf = dev_alloc::<u8>(&b.stream, pool, f.buffer_bytes(), AllocKind::Weights, "bench kf8")?;
    let p = Ptrs::at(dptr(&kf, &b.stream), &f);
    Store::unpack_launch(&mut b.kernels, ty, dptr(&w, &b.stream), rb, &f, p)?.run(&b.stream)?;
    let mut h = Sha256::new();
    h.update(defines.as_bytes());
    let name = format!("kf8v_{:x}", u64::from_le_bytes(h.finalize()[..8].try_into().expect("hash bytes")));
    if !b.kernels.has_module(&name) {
        let src = format!("{defines}\n{}", crate::sources::module_source("kf8").context("kf8 source")?);
        b.kernels.load(&name, &src)?;
    }
    let tile = if bm == 128 { Tile::Mid } else { Tile::Small };
    let ng = bk / f.g;
    let words = (bm + bn) * ng * 4 * if f.has_m { 2 } else { 1 };
    let smem = (stages * ((bm + bn) * bk + words)) as u32;
    let func = b.kernels.function_smem(&name, &f.entry(tile), smem)?;
    let threads: u32 = defines.lines().find_map(|l| l.strip_prefix("#define KF8_MID_THREADS ")).and_then(|v| v.trim().parse().ok()).filter(|_| tile == Tile::Mid).unwrap_or(256);
    let l = Launch {
        func,
        grid: (rows.div_ceil(bm) as u32, n_out.div_ceil(bn) as u32, 1),
        block: threads,
        smem,
        args: vec![Arg::P(p.q), Arg::P(p.d), Arg::P(p.m), Arg::P(q), Arg::P(scales), Arg::P(sums), Arg::P(0), Arg::P(0), Arg::P(y_ptr), Arg::I(rows as i32), Arg::I(n_in as i32), Arg::I(n_out as i32), Arg::P(0), Arg::I(0)],
    };
    time_us(b, &l, iters)
}

/// Device microseconds of the KF8K product compiled from a kernel source with defines prepended, on synthetic
/// integers, sub scales, and super scales of the shape, at a tile of bm by bn run by threads per block
#[allow(clippy::too_many_arguments)]
pub fn gemmk_us(b: &mut CudaBackend, src: &str, defines: &str, g: usize, has_m: bool, rows: usize, n_in: usize, n_out: usize, bm: usize, bn: usize, threads: u32, iters: usize) -> Result<f64> {
    b.ctx.bind_to_thread()?;
    let pool = b.pool();
    ensure!(n_in % 256 == 0 && rows > MATVEC_ROWS, "KF8K product needs columns in super blocks of 256 and more than eight rows");
    let ng = n_in / g;
    let nsb = n_in / 256;
    let noise = |i: usize| ((i as u32).wrapping_mul(2654435761) >> 8) as f32 / (1u32 << 24) as f32 - 0.5;
    let q: Vec<i8> = (0..n_out * n_in).map(|i| (noise(i) * 200.0) as i8).collect();
    let sc: Vec<i8> = (0..n_out * ng).map(|i| (noise(i + 7) * 60.0) as i8).collect();
    let d: Vec<f32> = (0..n_out * nsb).map(|i| noise(i + 11) * 0.01).collect();
    let x: Vec<f32> = (0..rows * n_in).map(noise).collect();
    let qb = dev_upload(&b.stream, pool, &q, AllocKind::Weights, "bench k8 q")?;
    let scb = dev_upload(&b.stream, pool, &sc, AllocKind::Weights, "bench k8 sc")?;
    let mb = dev_upload(&b.stream, pool, &sc, AllocKind::Weights, "bench k8 m")?;
    let db = dev_upload(&b.stream, pool, &d, AllocKind::Weights, "bench k8 d")?;
    let dmb = dev_upload(&b.stream, pool, &d, AllocKind::Weights, "bench k8 dmin")?;
    let xb = dev_upload(&b.stream, pool, &x, AllocKind::Compute, "bench x")?;
    let xq = dev_alloc::<f32>(&b.stream, pool, rows * n_in, AllocKind::Compute, "bench xq")?;
    let y = dev_alloc::<f32>(&b.stream, pool, rows * n_out, AllocKind::Compute, "bench y")?;
    let (xq_ptr, y_ptr) = (dptr(&xq, &b.stream), dptr(&y, &b.stream));
    quant_launch(&mut b.kernels, dptr(&xb, &b.stream), xq_ptr, rows, n_in, 4)?.run(&b.stream)?;
    let (xqp, _, xs) = act_scratch(xq_ptr, rows, n_in, g);
    let (_, xd32, _) = act_scratch(xq_ptr, rows, n_in, 32);
    let mut h = Sha256::new();
    h.update(src.as_bytes());
    h.update(defines.as_bytes());
    let name = format!("kf8k_{:x}", u64::from_le_bytes(h.finalize()[..8].try_into().expect("hash bytes")));
    if !b.kernels.has_module(&name) {
        b.kernels.load(&name, &format!("{defines}\n{src}"))?;
    }
    let tile = if bm == 128 { "mid" } else if bm == 64 { "small" } else { "tiny" };
    let entry = format!("gemmk_{tile}_g{g}{}", if has_m { "_m" } else { "" });
    let ng_tile = 128 / g;
    let stage = (bm + bn) * 128 + bn * ng_tile + if has_m { bn * ng_tile + bm * ng_tile * 4 } else { 0 };
    let smem = (2 * stage) as u32;
    let func = b.kernels.function_smem(&name, &entry, smem)?;
    let l = Launch {
        func,
        grid: (rows.div_ceil(bm) as u32, n_out.div_ceil(bn) as u32, 1),
        block: threads,
        smem,
        args: vec![Arg::P(dptr(&qb, &b.stream)), Arg::P(dptr(&scb, &b.stream)), Arg::P(dptr(&mb, &b.stream)), Arg::P(dptr(&db, &b.stream)), Arg::P(dptr(&dmb, &b.stream)), Arg::P(xqp), Arg::P(xd32), Arg::P(xs), Arg::P(0), Arg::P(0), Arg::P(y_ptr), Arg::I(rows as i32), Arg::I(n_in as i32), Arg::I(n_out as i32), Arg::P(0), Arg::I(0)],
    };
    time_us(b, &l, iters)
}

/// Device microseconds of the decode attention of one row over nkeys cached keys, parts blocks per kv head, the ropes
/// and norms folded in when fused and the output quantized for a Q8K product
#[allow(clippy::too_many_arguments)]
pub fn attention_us(b: &mut CudaBackend, hd: usize, n_head: usize, n_head_kv: usize, nkeys: usize, parts: usize, fused: bool, iters: usize) -> Result<f64> {
    attention_variant_us(b, "", hd, n_head, n_head_kv, nkeys, parts, fused, 4, iters)
}

/// The decode attention compiled from the attention module with defines prepended, the output quantized as kind
/// or left as f32 at kind zero
#[allow(clippy::too_many_arguments)]
pub fn attention_variant_us(b: &mut CudaBackend, defines: &str, hd: usize, n_head: usize, n_head_kv: usize, nkeys: usize, parts: usize, fused: bool, kind: i32, iters: usize) -> Result<f64> {
    b.ctx.bind_to_thread()?;
    let pool = b.pool();
    let cap = nkeys.next_multiple_of(256);
    let noise = |i: usize| ((i as u32).wrapping_mul(2654435761) >> 8) as f32 / (1u32 << 24) as f32 - 0.5;
    let kv: Vec<u16> = (0..n_head_kv * hd * cap).map(|i| half::f16::from_f32(noise(i)).to_bits()).collect();
    let kc = dev_upload(&b.stream, pool, &kv, AllocKind::State, "bench keys")?;
    let vc = dev_upload(&b.stream, pool, &kv, AllocKind::State, "bench values")?;
    let q: Vec<f32> = (0..n_head * hd).map(noise).collect();
    let k: Vec<f32> = (0..n_head_kv * hd).map(|i| noise(i + 3)).collect();
    let qb = dev_upload(&b.stream, pool, &q, AllocKind::Compute, "bench q")?;
    let kb = dev_upload(&b.stream, pool, &k, AllocKind::Compute, "bench k")?;
    let vb = dev_upload(&b.stream, pool, &k, AllocKind::Compute, "bench v")?;
    let p = (nkeys - 1) as i32;
    let pos = dev_upload(&b.stream, pool, &[p], AllocKind::Compute, "bench pos")?;
    let seq = dev_upload(&b.stream, pool, &[0i32], AllocKind::Compute, "bench seq")?;
    let att = dev_upload(&b.stream, pool, &[p], AllocKind::Compute, "bench attend")?;
    let rope4 = dev_upload(&b.stream, pool, &[p, p, p, 0], AllocKind::Compute, "bench rope")?;
    let freq: Vec<f32> = (0..hd / 2).map(|i| 10000f32.powf(-2.0 * i as f32 / hd as f32)).collect();
    let cs: Vec<f32> = (0..=nkeys).flat_map(|p| freq.iter().map(move |f| f * p as f32)).flat_map(|a| [a.cos(), a.sin()]).collect();
    let fb = dev_upload(&b.stream, pool, &cs, AllocKind::Compute, "bench angles")?;
    let nsplit = (n_head / n_head_kv).div_ceil(4).max(1);
    let counters = dev_alloc::<i32>(&b.stream, pool, n_head_kv * nsplit + 1, AllocKind::Compute, "bench counters")?;
    let partials = dev_alloc::<f32>(&b.stream, pool, parts * n_head * (2 + hd), AllocKind::Compute, "bench partials")?;
    let n = n_head * hd;
    let out = dev_alloc::<f32>(&b.stream, pool, n, AllocKind::Compute, "bench out")?;
    let act = dev_alloc::<f32>(&b.stream, pool, n, AllocKind::Compute, "bench act")?;
    let d = |s: &CudaSlice<f32>| dptr(s, &b.stream);
    let rope: Vec<Arg> = if fused {
        [vec![Arg::P(d(&kb)), Arg::P(d(&vb)), Arg::P(dptr(&rope4, &b.stream)), Arg::P(d(&fb)), Arg::I((hd / 2) as i32), Arg::I(1), Arg::F(1.0)], vec![Arg::I(0); 5], vec![Arg::I(0), Arg::P(0), Arg::P(0), Arg::F(0.0)], vec![Arg::I(0), Arg::P(0), Arg::P(0), Arg::F(0.0)]].concat()
    } else {
        [vec![Arg::P(0), Arg::P(0), Arg::P(0), Arg::P(0), Arg::I(0), Arg::I(0), Arg::F(0.0)], vec![Arg::I(0); 5], vec![Arg::I(0), Arg::P(0), Arg::P(0), Arg::F(0.0)], vec![Arg::I(0), Arg::P(0), Arg::P(0), Arg::F(0.0)]].concat()
    };
    let mut h = Sha256::new();
    h.update(defines.as_bytes());
    let name = format!("attv_{:x}", u64::from_le_bytes(h.finalize()[..8].try_into().expect("hash bytes")));
    if !b.kernels.has_module(&name) {
        let src = format!("{defines}\n{}", crate::sources::module_source("attention").context("attention source")?);
        b.kernels.load(&name, &src)?;
    }
    let l = Launch {
        func: b.kernels.function(&name, &format!("attention_decode_hd{hd}"))?,
        grid: (n_head_kv as u32, parts as u32, nsplit as u32),
        block: 256,
        smem: (n * 4) as u32,
        args: [
            vec![Arg::P(d(&qb)), Arg::P(dptr(&kc, &b.stream)), Arg::P(dptr(&vc, &b.stream)), Arg::P(dptr(&pos, &b.stream)), Arg::P(dptr(&seq, &b.stream)), Arg::P(dptr(&att, &b.stream)), Arg::P(0), Arg::P(d(&out)), Arg::P(d(&partials)), Arg::I(n_head as i32), Arg::I(n_head_kv as i32), Arg::F(0.125), Arg::I(0), Arg::F(0.0), Arg::I(cap as i32), Arg::I(1), Arg::P(dptr(&counters, &b.stream)), Arg::I(kind)],
            act_args(d(&act), 1, n),
            rope,
        ]
        .concat(),
    };
    time_us(b, &l, iters)
}

/// Device microseconds of the int8 matvec compiled from a quant module with defines prepended, rows of
/// activations against native weight bytes, wpr warps of each block on one output row
#[allow(clippy::too_many_arguments)]
pub fn matvec_variant_us(b: &mut CudaBackend, defines: &str, ty: u32, bytes: &[u8], rows: usize, n_in: usize, n_out: usize, wpr: usize, iters: usize) -> Result<f64> {
    b.ctx.bind_to_thread()?;
    let pool = b.pool();
    let rb = row_bytes(ty, n_in)? as i64;
    ensure!(bytes.len() == rb as usize * n_out, "{} bytes for {n_out} rows of {rb}", bytes.len());
    let w = dev_upload(&b.stream, pool, bytes, AllocKind::Weights, "bench weight")?;
    let f = Form::of(ty, n_in, n_out).context("kf8 form")?;
    ensure!(f.fits() && rows <= MATVEC_ROWS, "variant matvec needs the matvec shape");
    let code = round_code(ty)?;
    let x: Vec<f32> = (0..rows * n_in).map(|i| ((i as u32).wrapping_mul(2654435761) >> 8) as f32 / (1u32 << 24) as f32 - 0.5).collect();
    let xb = dev_upload(&b.stream, pool, &x, AllocKind::Compute, "bench x")?;
    let xq = dev_alloc::<f32>(&b.stream, pool, rows * n_in, AllocKind::Compute, "bench xq")?;
    let y = dev_alloc::<f32>(&b.stream, pool, rows * n_out, AllocKind::Compute, "bench y")?;
    let (xq_ptr, y_ptr) = (dptr(&xq, &b.stream), dptr(&y, &b.stream));
    quant_launch(&mut b.kernels, dptr(&xb, &b.stream), xq_ptr, rows, n_in, code)?.run(&b.stream)?;
    let [q, s16, m16, s32, m32] = crate::kf8::act_scratch_all(xq_ptr, rows, n_in);
    let module = crate::sources::quant_module(ty).context("quant module")?;
    let mut h = Sha256::new();
    h.update(module.as_bytes());
    h.update(defines.as_bytes());
    let name = format!("mvv_{:x}", u64::from_le_bytes(h.finalize()[..8].try_into().expect("hash bytes")));
    if !b.kernels.has_module(&name) {
        let src = format!("{defines}\n{}", crate::sources::module_source(&module).context("quant source")?);
        b.kernels.load(&name, &src)?;
    }
    let smem = crate::exec::matvec_smem(rows, n_in);
    let steps = n_in.div_ceil(32 * crate::sources::matvec_cols(ty)).max(1);
    let blocks = n_out.div_ceil(8 / wpr);
    let entry = match (rows, steps, blocks > 3 * b.kernels.sm_count() && blocks <= 4 * b.kernels.sm_count()) {
        (1, 1, true) => "matvec_i8_1s",
        (1, _, _) => "matvec_i8_1",
        _ => "matvec_i8",
    };
    let func = b.kernels.function_smem(&name, entry, smem)?;
    let l = Launch {
        func,
        grid: (n_out.div_ceil(8 / wpr) as u32, 1, 1),
        block: MATVEC_THREADS,
        smem,
        args: vec![Arg::P(dptr(&w, &b.stream)), Arg::P(q), Arg::P(s16), Arg::P(m16), Arg::P(s32), Arg::P(m32), Arg::P(0), Arg::P(0), Arg::P(y_ptr), Arg::I(rows as i32), Arg::I(n_in as i32), Arg::I(n_out as i32), Arg::L(rb), Arg::P(0), Arg::I(0), Arg::I(wpr as i32)],
    };
    time_us(b, &l, iters)
}

/// Device microseconds of one coalesced read of bytes by a grid of blocks per SM, each thread unroll sixteen byte
/// loads in flight, the ceiling the matvec is measured against
pub fn read_us(b: &mut CudaBackend, bytes: usize, blocks_per_sm: usize, threads: u32, unroll: usize, iters: usize) -> Result<f64> {
    b.ctx.bind_to_thread()?;
    let pool = b.pool();
    let buf = dev_alloc::<u8>(&b.stream, pool, bytes, AllocKind::Compute, "bench read")?;
    let out = dev_alloc::<i32>(&b.stream, pool, 1, AllocKind::Compute, "bench read out")?;
    let name = format!("readbw_u{unroll}");
    if !b.kernels.has_module(&name) {
        let src = format!(
            "typedef long long i64;\nextern \"C\" __global__ void read_bytes(const int4* p, i64 n16, int* out) {{\n  i64 stride = (i64)gridDim.x * blockDim.x;\n  i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x;\n  int acc = 0;\n  for (; i + {u1} * stride < n16; i += {unroll} * stride) {{\n    int4 v[{unroll}];\n#pragma unroll\n    for (int u = 0; u < {unroll}; u++) v[u] = p[i + u * stride];\n#pragma unroll\n    for (int u = 0; u < {unroll}; u++) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;\n  }}\n  for (; i < n16; i += stride) {{ int4 v = p[i]; acc ^= v.x ^ v.y ^ v.z ^ v.w; }}\n  if (acc == 0x7fffffff) out[0] = acc;\n}}\n",
            u1 = unroll - 1
        );
        b.kernels.load(&name, &src)?;
    }
    let func = b.kernels.function(&name, "read_bytes")?;
    let l = Launch { func, grid: ((blocks_per_sm * b.kernels.sm_count()) as u32, 1, 1), block: threads, smem: 0, args: vec![Arg::P(dptr(&buf, &b.stream)), Arg::L((bytes / 16) as i64), Arg::P(dptr(&out, &b.stream))] };
    time_us(b, &l, iters)
}

/// Device microseconds of reading nrows rows of row_bytes each, a warp per row taking chunk contiguous bytes per
/// step as sixteen byte lane loads with pf steps in flight, the access pattern of the matvec without its arithmetic
pub fn read_rows_us(b: &mut CudaBackend, nrows: usize, row_bytes: usize, chunk: usize, pf: usize, warps: u32, iters: usize) -> Result<f64> {
    b.ctx.bind_to_thread()?;
    let pool = b.pool();
    let buf = dev_alloc::<u8>(&b.stream, pool, nrows * row_bytes + 4096, AllocKind::Compute, "bench rows")?;
    let out = dev_alloc::<i32>(&b.stream, pool, 1, AllocKind::Compute, "bench rows out")?;
    let name = format!("readrows_c{chunk}_p{pf}");
    if !b.kernels.has_module(&name) {
        let lanes = chunk / 16;
        let src = format!(
            "typedef long long i64; typedef unsigned char u8;\nextern \"C\" __global__ void read_rows(const u8* p, int nrows, i64 row_bytes, int* out) {{\n  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;\n  if (warp >= nrows) return;\n  const u8* row = p + (i64)warp * row_bytes;\n  const int nsteps = (int)(row_bytes / {chunk});\n  int acc = 0;\n  int4 ring[{pf}];\n#pragma unroll\n  for (int i = 0; i < {pf}; i++) ring[i] = (lane < {lanes} && i < nsteps) ? *reinterpret_cast<const int4*>(row + i * {chunk} + lane * 16) : make_int4(0,0,0,0);\n  for (int s0 = 0; s0 < nsteps; s0 += {pf}) {{\n#pragma unroll\n    for (int i = 0; i < {pf}; i++) {{\n      int s = s0 + i;\n      if (s >= nsteps) break;\n      int4 v = ring[i];\n      int sn = s + {pf};\n      ring[i] = (lane < {lanes} && sn < nsteps) ? *reinterpret_cast<const int4*>(row + (i64)sn * {chunk} + lane * 16) : make_int4(0,0,0,0);\n      acc ^= v.x ^ v.y ^ v.z ^ v.w;\n    }}\n  }}\n  if (acc == 0x7fffffff) out[0] = acc;\n}}\n"
        );
        b.kernels.load(&name, &src)?;
    }
    let func = b.kernels.function(&name, "read_rows")?;
    let l = Launch { func, grid: ((nrows as u32).div_ceil(warps), 1, 1), block: 32 * warps, smem: 0, args: vec![Arg::P(dptr(&buf, &b.stream)), Arg::I(nrows as i32), Arg::L(row_bytes as i64), Arg::P(dptr(&out, &b.stream))] };
    time_us(b, &l, iters)
}

/// Device microseconds per kernel of a captured graph of n back to back one block launches, the floor of a launch
/// sequence replayed from a graph
pub fn graph_gap_us(b: &mut CudaBackend, n: usize, iters: usize) -> Result<f64> {
    b.ctx.bind_to_thread()?;
    let pool = b.pool();
    let out = dev_alloc::<i32>(&b.stream, pool, 1, AllocKind::Compute, "bench gap out")?;
    let name = "gap_probe";
    if !b.kernels.has_module(name) {
        b.kernels.load(name, "extern \"C\" __global__ void touch(int* out) { if (threadIdx.x == 0) atomicAdd(out, 1); }\n")?;
    }
    let l = Launch { func: b.kernels.function(name, "touch")?, grid: (1, 1, 1), block: 32, smem: 0, args: vec![Arg::P(dptr(&out, &b.stream))] };
    b.stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
    for _ in 0..n {
        l.run(&b.stream)?;
    }
    let graph = b.stream.end_capture(sys::CUgraphInstantiate_flags(0))?.context("empty capture")?;
    graph.launch()?;
    let e0 = b.stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
    for _ in 0..iters {
        graph.launch()?;
    }
    let e1 = b.stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
    b.stream.synchronize()?;
    Ok(e0.elapsed_ms(&e1)? as f64 * 1000.0 / (iters * n) as f64)
}

/// Device microseconds per stage of a persistent kernel of blocks_per_sm blocks per SM running stages grid barriers,
/// each block reading kb kilobytes of its own weights per stage before arriving, block zero alone reading that much
/// when kb is negative, launched cooperatively inside a captured graph when captured is set
pub fn barrier_us(b: &mut CudaBackend, variant: usize, blocks_per_sm: usize, stages: usize, kb: isize, captured: bool, iters: usize) -> Result<f64> {
    b.ctx.bind_to_thread()?;
    let pool = b.pool();
    let g = blocks_per_sm * b.kernels.sm_count();
    let bytes = (g * stages * kb.unsigned_abs() * 1024).max(16);
    let w = dev_alloc::<u8>(&b.stream, pool, bytes, AllocKind::Compute, "bench barrier weights")?;
    let sync = dev_alloc::<i32>(&b.stream, pool, 2 * stages + 1, AllocKind::Compute, "bench barrier counters")?;
    let out = dev_alloc::<i32>(&b.stream, pool, 1, AllocKind::Compute, "bench barrier out")?;
    let name = format!("barrier_probe_v{variant}");
    if !b.kernels.has_module(&name) {
        // Variant 0 fences around a returning atomic and a volatile poll, 1 a release reduction with an acquire poll on
        // monotonic counters, 2 the same polled without sleeping, 3 a relaxed poll with one acquire fence after it, 4
        // the relaxed poll sleeping between turns
        let (arrive, poll) = match variant {
            0 => (
                "__threadfence(); if (atomicAdd(&cnt[s], 1) == g - 1) { cnt[s] = 0; __threadfence(); flag[s] = gen + 1; if (s == nstages - 1) *gen_ptr = gen + 1; }",
                "while (ld_vol(&flag[s]) != gen + 1) __nanosleep(32); __threadfence();",
            ),
            1 => ("red_release(&cnt[s], 1);", "while ((int)(ld_acquire(&cnt[s]) - target) < 0) __nanosleep(32);"),
            2 => ("red_release(&cnt[s], 1);", "while ((int)(ld_acquire(&cnt[s]) - target) < 0) {}"),
            3 => ("red_release(&cnt[s], 1);", "while ((int)(ld_relaxed(&cnt[s]) - target) < 0) {} fence_acquire();"),
            4 => ("red_release(&cnt[s], 1);", "while ((int)(ld_relaxed(&cnt[s]) - target) < 0) __nanosleep(32); fence_acquire();"),
            5 => ("red_release(&cnt[s], 1);", "while ((int)(ld_relaxed(&cnt[s]) - target) < 0) __nanosleep(200); fence_acquire();"),
            6 => ("red_release(&cnt[s], 1);", "while ((int)(ld_vol(&cnt[s]) - target) < 0) {} fence_acquire();"),
            _ => ("red_release(&cnt[s], 1);", "for (int t = 0; (int)(ld_relaxed(&cnt[s]) - target) < 0; t++) { if (t > 8) __nanosleep(100); } fence_acquire();"),
        };
        let tail = if variant == 0 { "" } else { "if (blockIdx.x == 0 && threadIdx.x == 0) *gen_ptr = gen + 1;" };
        let src = format!(
            r#"
typedef long long i64;
__device__ __forceinline__ int ld_vol(const int* p) {{ int v; asm volatile("ld.volatile.global.s32 %0, [%1];" : "=r"(v) : "l"(p)); return v; }}
__device__ __forceinline__ int ld_acquire(const int* p) {{ int v; asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }}
__device__ __forceinline__ void red_release(int* p, int v) {{ asm volatile("red.release.gpu.global.add.s32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }}
__device__ __forceinline__ int ld_relaxed(const int* p) {{ int v; asm volatile("ld.relaxed.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }}
__device__ __forceinline__ void fence_acquire() {{ asm volatile("fence.acq_rel.gpu;" ::: "memory"); }}
extern "C" __global__ void __launch_bounds__(256) stages(const int4* w, i64 words_per_block, int nstages, int* cnt, int* flag, int* gen_ptr, int* out) {{
    const int g = gridDim.x;
    const int gen = ld_vol(gen_ptr);
    int acc = 0;
    for (int s = 0; s < nstages; s++) {{
        const int target = (gen + 1) * g;
        const i64 wpb = words_per_block < 0 ? (blockIdx.x == 0 ? -words_per_block : 0) : words_per_block;
        const int4* p = w + ((i64)s * g + blockIdx.x) * (words_per_block < 0 ? -words_per_block : words_per_block);
        for (i64 i = threadIdx.x; i < wpb; i += 256) {{ int4 v = p[i]; acc ^= v.x ^ v.y ^ v.z ^ v.w; }}
        __syncthreads();
        if (threadIdx.x == 0) {{
            {arrive}
            {poll}
        }}
        __syncthreads();
    }}
    {tail}
    if (acc == 0x7fffffff) out[0] = acc;
}}
"#
        );
        b.kernels.load(&name, &src)?;
    }
    let sp = dptr(&sync, &b.stream);
    let l = Launch {
        func: b.kernels.function(&name, "stages")?,
        grid: (g as u32, 1, 1),
        block: 256,
        smem: 0,
        args: vec![Arg::P(dptr(&w, &b.stream)), Arg::L((kb * 1024 / 16) as i64), Arg::I(stages as i32), Arg::P(sp), Arg::P(sp + 4 * stages as u64), Arg::P(sp + 8 * stages as u64), Arg::P(dptr(&out, &b.stream))],
    };
    let time = |b: &CudaBackend, run: &dyn Fn() -> Result<()>| -> Result<f64> {
        run()?;
        let e0 = b.stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        for _ in 0..iters {
            run()?;
        }
        let e1 = b.stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        b.stream.synchronize()?;
        Ok(e0.elapsed_ms(&e1)? as f64 * 1000.0 / (iters * stages) as f64)
    };
    if captured {
        b.stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
        let r = l.run_cooperative(&b.stream);
        let graph = b.stream.end_capture(sys::CUgraphInstantiate_flags(0));
        r?;
        let graph = graph?.context("empty capture")?;
        return time(b, &|| graph.launch().context("graph launch"));
    }
    time(b, &|| l.run_cooperative(&b.stream))
}

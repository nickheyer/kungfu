use anyhow::{Result, bail};

/// Headers the kernel sources include, spliced in place before compiling
pub const HEADERS: &[(&str, &str)] = &[
    ("common.cuh", include_str!("kernels/common.cuh")),
    ("quant.cuh", include_str!("kernels/quant.cuh")),
    ("mm_i8.cuh", include_str!("kernels/mm_i8.cuh")),
    ("mm_frag.cuh", include_str!("kernels/mm_frag.cuh")),
    ("grids.cuh", include_str!("kernels/grids.cuh")),
    ("kf8.cu", include_str!("kernels/kf8.cu")),
    ("attention_decode.cuh", include_str!("kernels/attention_decode.cuh")),
    ("persist.cuh", include_str!("kernels/persist.cuh")),
];

/// Op kernel modules by name
pub const MODULES: &[(&str, &str)] = &[
    ("elementwise", include_str!("kernels/elementwise.cu")),
    ("norm", include_str!("kernels/norm.cu")),
    ("rope", include_str!("kernels/rope.cu")),
    ("attention", include_str!("kernels/attention.cu")),
    ("moe", include_str!("kernels/moe.cu")),
    ("sample", include_str!("kernels/sample.cu")),
    ("kf8", include_str!("kernels/kf8.cu")),
    ("kf8w", include_str!("kernels/kf8w.cu")),
    ("flash", include_str!("kernels/flash.cu")),
    ("fused", include_str!("kernels/fused.cu")),
    ("train", include_str!("kernels/train.cu")),
];

/// Quant kernel modules keyed by the kf-quant type name
pub const QUANT_MODULES: &[(&str, &str)] = &[
    ("f32", include_str!("kernels/quant_f32.cu")),
    ("f16", include_str!("kernels/quant_f16.cu")),
    ("bf16", include_str!("kernels/quant_bf16.cu")),
    ("q4_0", include_str!("kernels/quant_q4_0.cu")),
    ("q4_1", include_str!("kernels/quant_q4_1.cu")),
    ("q5_0", include_str!("kernels/quant_q5_0.cu")),
    ("q5_1", include_str!("kernels/quant_q5_1.cu")),
    ("q8_0", include_str!("kernels/quant_q8_0.cu")),
    ("q8_1", include_str!("kernels/quant_q8_1.cu")),
    ("q2_K", include_str!("kernels/quant_q2_K.cu")),
    ("q3_K", include_str!("kernels/quant_q3_K.cu")),
    ("q4_K", include_str!("kernels/quant_q4_K.cu")),
    ("q5_K", include_str!("kernels/quant_q5_K.cu")),
    ("q6_K", include_str!("kernels/quant_q6_K.cu")),
    ("q8_K", include_str!("kernels/quant_q8_K.cu")),
    ("iq2_xxs", include_str!("kernels/quant_iq2_xxs.cu")),
    ("iq2_xs", include_str!("kernels/quant_iq2_xs.cu")),
    ("iq3_xxs", include_str!("kernels/quant_iq3_xxs.cu")),
    ("iq1_s", include_str!("kernels/quant_iq1_s.cu")),
    ("iq4_nl", include_str!("kernels/quant_iq4_nl.cu")),
    ("iq3_s", include_str!("kernels/quant_iq3_s.cu")),
    ("iq2_s", include_str!("kernels/quant_iq2_s.cu")),
    ("iq4_xs", include_str!("kernels/quant_iq4_xs.cu")),
    ("iq1_m", include_str!("kernels/quant_iq1_m.cu")),
    ("tq1_0", include_str!("kernels/quant_tq1_0.cu")),
    ("tq2_0", include_str!("kernels/quant_tq2_0.cu")),
    ("mxfp4", include_str!("kernels/quant_mxfp4.cu")),
    ("nvfp4", include_str!("kernels/quant_nvfp4.cu")),
    ("q1_0", include_str!("kernels/quant_q1_0.cu")),
    ("q2_0", include_str!("kernels/quant_q2_0.cu")),
];

/// Quant types whose kernel file gives q16 and so the KF8 unpack, every block type but the floats, with the KF8 group size
/// and whether the type has a nonzero offset, see docs/kf8-format.md. The group is 32 where the native scale is constant over
/// every aligned 32 and 16 where it changes every 16.
pub const INT8_TYPES: &[(&str, u32, bool)] = &[
    ("q4_0", 32, false),
    ("q4_1", 32, true),
    ("q5_0", 32, false),
    ("q5_1", 32, true),
    ("q8_0", 32, false),
    ("q8_1", 32, false),
    ("q2_K", 16, true),
    ("q3_K", 16, false),
    ("q4_K", 32, true),
    ("q5_K", 32, true),
    ("q6_K", 16, false),
    ("q8_K", 32, false),
    ("iq2_xxs", 32, false),
    ("iq2_xs", 16, false),
    ("iq3_xxs", 32, false),
    ("iq1_s", 32, false),
    ("iq4_nl", 32, false),
    ("iq3_s", 32, false),
    ("iq2_s", 16, false),
    ("iq4_xs", 32, false),
    ("iq1_m", 16, false),
    ("tq1_0", 32, false),
    ("tq2_0", 32, false),
    ("mxfp4", 32, false),
    ("nvfp4", 16, false),
    ("q1_0", 32, false),
    ("q2_0", 32, false),
];

/// Columns one lane of the int8 matvec takes per block for a ggml type, the KfLane width of its kernel file
pub fn matvec_cols(ggml_type: u32) -> usize {
    match kf_quant::info(ggml_type).map(|i| i.name) {
        Some("q4_K" | "q5_K" | "q6_K") => 64,
        Some("q8_0") => 32,
        _ => 16,
    }
}

/// Whether a ggml type has the fragment product plan, see mm_frag.cuh
pub fn has_frag(ggml_type: u32) -> bool {
    frag_plan(ggml_type).is_some()
}

/// Shared memory pitch of one weight row of a super block, the ring stages per warp, and the warps per block of the
/// fragment product, the PITCH, STAGES and WARPS of the type's KfFrag
pub fn frag_plan(ggml_type: u32) -> Option<(usize, usize, usize)> {
    match kf_quant::info(ggml_type).map(|i| i.name) {
        Some("q4_K") => Some((160, 2, 8)),
        Some("q6_K") => Some((240, 2, 4)),
        _ => None,
    }
}

/// Columns of one run of the int8 matvec lane for a ggml type, the columns under one weight scale, KfLaneWords RUN
pub fn matvec_run(ggml_type: u32) -> usize {
    match kf_quant::info(ggml_type).map(|i| i.name) {
        Some("q4_K" | "q5_K" | "q8_0") => 32,
        _ => 16,
    }
}

/// The KF8 form of a ggml type: group size and whether an offset array exists, none for the float types
pub fn kf8_form(ggml_type: u32) -> Option<(u32, bool)> {
    let name = kf_quant::info(ggml_type)?.name;
    INT8_TYPES.iter().find(|(n, _, _)| *n == name).map(|&(_, g, m)| (g, m))
}

/// Whether a type's group scale carries more than 22 significant bits, so the product recovers each group sum by
/// subtraction instead of the fused magic multiply, true only for the f32 scale of q8_K
pub fn kf8_wide(ggml_type: u32) -> bool {
    kf_quant::info(ggml_type).is_some_and(|i| i.name == "q8_K")
}

/// Whether the quant module of a ggml type has the KF8 unpack and so the int8 tensor core product
pub fn has_int8(ggml_type: u32) -> bool {
    kf8_form(ggml_type).is_some()
}

/// Module name of the quant kernels for a ggml type id, none when no kernel file exists
pub fn quant_module(ggml_type: u32) -> Option<String> {
    let name = kf_quant::info(ggml_type)?.name;
    QUANT_MODULES.iter().any(|(n, _)| *n == name).then(|| format!("quant_{name}"))
}

/// Embedded source of a module by name
pub fn module_source(name: &str) -> Option<&'static str> {
    if let Some(q) = name.strip_prefix("quant_") {
        return QUANT_MODULES.iter().find(|(n, _)| *n == q).map(|(_, s)| *s);
    }
    MODULES.iter().find(|(n, _)| *n == name).map(|(_, s)| *s)
}

/// One translation unit with every quoted include replaced by its header once
pub fn expand(src: &str) -> Result<String> {
    let mut out = String::with_capacity(src.len() * 2);
    let mut seen = Vec::new();
    splice(src, &mut out, &mut seen)?;
    Ok(out)
}

fn splice(src: &str, out: &mut String, seen: &mut Vec<&'static str>) -> Result<()> {
    for line in src.lines() {
        let t = line.trim();
        if t == "#pragma once" {
            continue;
        }
        if let Some(rest) = t.strip_prefix("#include") {
            let name = rest.trim().trim_matches('"');
            let Some((hname, body)) = HEADERS.iter().find(|(n, _)| *n == name) else { bail!("kernel include {name} is not an embedded header") };
            if !seen.contains(hname) {
                seen.push(hname);
                splice(body, out, seen)?;
            }
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    Ok(())
}

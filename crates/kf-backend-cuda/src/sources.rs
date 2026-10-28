use anyhow::{Result, bail};

/// Headers the kernel sources include, spliced in place before compiling
pub const HEADERS: &[(&str, &str)] = &[
    ("common.cuh", include_str!("kernels/common.cuh")),
    ("quant.cuh", include_str!("kernels/quant.cuh")),
    ("grids.cuh", include_str!("kernels/grids.cuh")),
];

/// Op kernel modules by name
pub const MODULES: &[(&str, &str)] = &[
    ("elementwise", include_str!("kernels/elementwise.cu")),
    ("norm", include_str!("kernels/norm.cu")),
    ("rope", include_str!("kernels/rope.cu")),
    ("attention", include_str!("kernels/attention.cu")),
    ("moe", include_str!("kernels/moe.cu")),
    ("sample", include_str!("kernels/sample.cu")),
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

/// Quant types whose kernel file gives q16 and so the matmul_i8 tensor core entry, every block type but the floats
pub const INT8_TYPES: &[&str] = &[
    "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "q8_1", "q2_K", "q3_K", "q4_K", "q5_K", "q6_K", "q8_K", "iq2_xxs", "iq2_xs", "iq3_xxs", "iq1_s", "iq4_nl", "iq3_s", "iq2_s", "iq4_xs", "iq1_m", "tq1_0", "tq2_0", "mxfp4", "nvfp4", "q1_0", "q2_0",
];

/// Whether the quant module of a ggml type has the int8 tensor core product
pub fn has_int8(ggml_type: u32) -> bool {
    kf_quant::info(ggml_type).is_some_and(|i| INT8_TYPES.contains(&i.name))
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

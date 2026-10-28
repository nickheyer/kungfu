//! The only list of architectures, one line per module
use crate::Architecture;

pub mod chatglm;
pub mod deepseek2;
pub mod deepseek41;
pub mod gemma;
pub mod gemma2;
pub mod gemma3;
pub mod glm4;
pub mod glm4moe;
pub mod glm5_next;
pub mod glm_dsa;
pub mod kimi_linear;
pub mod llama;
pub mod muse_glimmer;
pub mod muse_spark;
pub mod qwen;
pub mod qwen2;
pub mod qwen2moe;
pub mod qwen2vl;
pub mod qwen3;
pub mod qwen35;
pub mod qwen35moe;
pub mod qwen3moe;
pub mod qwen3next;
pub mod qwen4exp;

pub fn all() -> &'static [&'static dyn Architecture] {
    &[
        &llama::Llama,
        &qwen::Qwen,
        &qwen2::Qwen2,
        &qwen2moe::Qwen2Moe,
        &qwen2vl::Qwen2Vl,
        &qwen3::Qwen3,
        &qwen3moe::Qwen3Moe,
        &qwen3next::Qwen3Next,
        &qwen35::Qwen35,
        &qwen35moe::Qwen35Moe,
        &qwen4exp::Qwen4Exp,
        &deepseek2::DeepSeek2,
        &deepseek41::DeepSeek41,
        &chatglm::ChatGlm,
        &glm4::Glm4,
        &glm4moe::Glm4Moe,
        &glm_dsa::GlmDsa,
        &glm5_next::Glm5Next,
        &kimi_linear::KimiLinear,
        &muse_glimmer::MuseGlimmer,
        &muse_spark::MuseSpark,
        &gemma::Gemma,
        &gemma2::Gemma2,
        &gemma3::Gemma3,
    ]
}

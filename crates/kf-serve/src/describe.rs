//! The self description a manager reads instead of knowing the runtime
use crate::params::PARAMS;
use serde_json::{Value, json};

pub const PROTOCOL_VERSION: u32 = 1;

/// Seconds the server drains in flight requests after SIGTERM before it exits
pub const STOP_GRACE_SECONDS: u64 = 15;

pub const HEALTH_INTERVAL_SECONDS: u64 = 1;
pub const HEALTH_TIMEOUT_SECONDS: u64 = 600;

/// A log key the binary emits and what it measures
pub struct Measurement {
    pub key: &'static str,
    pub meaning: &'static str,
}

pub const MEASUREMENTS: &[Measurement] = &[
    Measurement { key: "device:N.weights", meaning: "bytes of weights and expanded norms resident on device N, summed from alloc events" },
    Measurement { key: "device:N.state", meaning: "bytes of cache rings and lengths on device N at the planned context" },
    Measurement { key: "device:N.experts", meaning: "bytes of expert cache slots on device N" },
    Measurement { key: "device:N.compute", meaning: "bytes of activation, token, and scratch buffers on device N, under the plan's compute_reserve" },
    Measurement { key: "host.weights", meaning: "mapped file bytes of every weight the plan leaves on the host" },
    Measurement { key: "host.state", meaning: "bytes of host cache rings and recurrent state with their roll back snapshots" },
    Measurement { key: "host.experts", meaning: "pinned host bytes of the expert tensors the device caches" },
    Measurement { key: "pool.failed", meaning: "bytes of the allocation that failed on a pool, from alloc_failed events" },
];

/// An error code the binary emits with its keys, summary, hint, and the parameter change that fixes it
pub struct ErrorCode {
    pub code: &'static str,
    pub keys: &'static [&'static str],
    pub summary: &'static str,
    pub hint: &'static str,
    pub fix: &'static [(&'static str, &'static str)],
}

pub const ERRORS: &[ErrorCode] = &[
    ErrorCode { code: "bad_flag", keys: &["detail"], summary: "a flag was refused: ${detail}", hint: "kungfu describe lists every flag with its type, choices, and bounds", fix: &[] },
    ErrorCode { code: "model_open", keys: &["path", "detail"], summary: "the model file could not be read: ${detail}", hint: "check the path and that every shard of a split file is beside it", fix: &[] },
    ErrorCode { code: "unknown_arch", keys: &["arch"], summary: "this build does not know architecture ${arch}", hint: "a newer kungfu adds an architecture as one file in its registry", fix: &[] },
    ErrorCode { code: "unclaimed_tensors", keys: &["arch", "count", "names"], summary: "the ${arch} module left ${count} tensors unclaimed", hint: "the file carries tensors this build's module does not compose, a newer kungfu claims them", fix: &[] },
    ErrorCode { code: "compose", keys: &["arch", "detail"], summary: "the ${arch} model could not be composed: ${detail}", hint: "the file's hyperparameters or tensor shapes do not match what the module expects", fix: &[] },
    ErrorCode { code: "no_device", keys: &["device", "visible"], summary: "cuda device ${device} is not among the ${visible} visible", hint: "set CUDA_VISIBLE_DEVICES or run on the host", fix: &[("backend", "cpu")] },
    ErrorCode { code: "no_fit", keys: &["n_ctx", "detail"], summary: "the model does not fit at ${n_ctx} tokens: ${detail}", hint: "let the planner pick the context", fix: &[("n_ctx", "auto")] },
    ErrorCode { code: "device_oom", keys: &["detail"], summary: "the device ran out of memory: ${detail}", hint: "lower the context or place layers on the host in the plan", fix: &[("n_ctx", "auto")] },
    ErrorCode { code: "plan", keys: &["path", "detail"], summary: "the plan document was refused: ${detail}", hint: "pass a document shaped like the plan event prints", fix: &[] },
    ErrorCode { code: "bind", keys: &["host", "port", "detail"], summary: "could not bind ${host}:${port}: ${detail}", hint: "free the port or pick another", fix: &[] },
    ErrorCode { code: "no_chat_template", keys: &["path"], summary: "the file carries no chat template", hint: "serve needs tokenizer.chat_template in the file's metadata", fix: &[] },
    ErrorCode { code: "tokenizer", keys: &["detail"], summary: "the tokenizer could not be built: ${detail}", hint: "the file's tokenizer.ggml.pre names a pretokenizer this build lacks", fix: &[] },
    ErrorCode { code: "backend", keys: &["detail"], summary: "a pass failed: ${detail}", hint: "a kernel or driver error, the preceding events say which allocation or op", fix: &[] },
];

/// The roles this process plays, each with the facts a manager launches and watches it by
fn roles() -> Value {
    let http = json!({ "kind": "http", "path": "/health", "interval_seconds": HEALTH_INTERVAL_SECONDS, "timeout_seconds": HEALTH_TIMEOUT_SECONDS });
    let canary = json!({ "kind": "canary", "path": "/health", "interval_seconds": HEALTH_INTERVAL_SECONDS, "timeout_seconds": HEALTH_TIMEOUT_SECONDS });
    json!({
        "head": { "phase": 1, "files": "weights", "health": http, "listens": true, "single": false, "rendezvous": false, "head": true, "connects_from": ["conductor"] },
        "replica": { "phase": 0, "files": "weights", "health": http, "listens": true, "single": false, "rendezvous": false, "head": true, "connects_from": ["conductor"] },
        "stage": { "phase": 0, "files": "weights", "health": { "kind": "hello", "interval_seconds": HEALTH_INTERVAL_SECONDS, "timeout_seconds": HEALTH_TIMEOUT_SECONDS }, "listens": true, "single": true, "rendezvous": false, "head": false, "connects_from": ["head"] },
        "prefill": { "phase": 0, "files": "weights", "health": canary, "listens": true, "single": false, "rendezvous": false, "head": false, "connects_from": ["conductor"] },
        "decode": { "phase": 0, "files": "weights", "health": canary, "listens": true, "single": false, "rendezvous": false, "head": true, "connects_from": ["conductor"] },
        "draft": { "phase": 0, "files": "weights", "health": http, "listens": true, "single": true, "rendezvous": false, "head": false, "connects_from": ["head"] }
    })
}

/// The devices each backend of this build can run on, probed now
fn backends() -> Value {
    let (free, total) = kf_engine::host_memory();
    let mut out = vec![json!({ "name": "cpu", "devices": [{ "ordinal": 0, "name": "host", "total_bytes": total, "free_bytes": free }] })];
    if let Some(name) = kf_engine::device::NAME {
        let devices: Vec<Value> = kf_engine::device::devices().into_iter().map(|d| json!({ "ordinal": d.ordinal, "name": d.name, "total_bytes": d.total_bytes, "free_bytes": d.free_bytes })).collect();
        out.push(json!({ "name": name, "devices": devices }));
    }
    Value::Array(out)
}

/// Everything a manager needs to install, plan, launch, watch, and triage this binary
pub fn describe() -> Value {
    let archs: Vec<&str> = kf_arch::registry().iter().map(|a| a.id()).collect();
    let quants: Vec<Value> = kf_quant::types::all()
        .iter()
        .map(|q| {
            let i = q.info();
            let mut backends = vec!["cpu"];
            if let Some(name) = kf_engine::device::NAME
                && kf_engine::device::has_quant(i.id)
            {
                backends.push(name);
            }
            json!({ "id": i.id, "name": i.name, "block": i.block, "size": i.size, "backends": backends })
        })
        .collect();
    let measurements: Vec<Value> = MEASUREMENTS.iter().map(|m| json!({ "key": m.key, "meaning": m.meaning })).collect();
    let errors: Vec<Value> = ERRORS
        .iter()
        .map(|e| {
            let fix: serde_json::Map<String, Value> = e.fix.iter().map(|(k, v)| (k.to_string(), Value::String(v.to_string()))).collect();
            json!({ "code": e.code, "keys": e.keys, "summary": e.summary, "hint": e.hint, "fix": fix })
        })
        .collect();
    json!({
        "name": "kungfu",
        "version": env!("CARGO_PKG_VERSION"),
        "protocol": PROTOCOL_VERSION,
        "architectures": archs,
        "formats": ["gguf"],
        "quant_types": quants,
        "backends": backends(),
        "params": PARAMS,
        "health": { "path": "/health", "interval_seconds": HEALTH_INTERVAL_SECONDS, "timeout_seconds": HEALTH_TIMEOUT_SECONDS },
        "api": {
            "openai": ["/v1/chat/completions", "/v1/models"],
            "responses": ["/v1/responses"],
            "anthropic": ["/v1/messages"],
            "slots": "/slots/{id}",
            "health": "/health",
            "describe": "/describe",
            "metrics": "/metrics",
            "chat_fields": ["tools", "tool_choice", "reasoning_effort", "chat_template_kwargs", "seed", "stop", "max_tokens", "temperature", "top_p", "top_k", "min_p", "repeat_penalty", "logit_bias", "response_format", "logprobs", "top_logprobs", "stream", "id_slot"],
            "response_format": ["text", "json_object", "json_schema"],
            "images": { "parts": ["image_url"], "urls": ["data", "http", "https"], "needs": "--mmproj", "projectors": ["qwen2vl_merger", "gemma3"] },
            "tool_choice": ["auto", "none", "required", "function"],
            "reasoning_effort": crate::chat::REASONING_EFFORTS.iter().map(|(n, b)| (n.to_string(), json!(b))).collect::<serde_json::Map<String, Value>>(),
            "schema_keywords_refused": crate::grammar::KEYWORDS_REFUSED
        },
        "roles": roles(),
        "shapes": { "solo": ["head"], "replicas": ["replica"], "chain": ["stage", "head"], "relay": ["prefill", "decode"], "draft": ["draft", "head"] },
        "shape_facts": {
            "ring": false,
            "activation_bytes": 4,
            "chunks_in_flight": 1,
            "reductions_per_layer": 0,
            "relay_overlap": false,
            "relay_rdma": false,
            "relay_disk": true,
            "chain_aggregate": false,
            "chain_speculative": true,
            "handoff": "llamacpp-slot",
            "draft_tokens": 3
        },
        "draft": { "path": "/draft" },
        "relay": { "slots": 1, "path": "/slots/{id}", "actions": ["save", "restore"] },
        "seat_flags": ["--role", "--rank", "--count", "--peer", "--layer-from", "--layer-to", "--admit", "--slot-dir", "--draft-model", "--draft-max"],
        "events": ["plan", "alloc", "alloc_failed", "ready", "cache", "transport", "refused", "error", "request"],
        "measurements": measurements,
        "errors": errors,
        "stop_grace_seconds": STOP_GRACE_SECONDS,
    })
}

//! Structured log events on stderr, one line each, the only output a manager parses
use crate::plan::Pool;
use std::collections::BTreeMap;
use std::io::Write;
use std::sync::Mutex;

/// What an allocation holds, the four kinds a plan accounts for
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AllocKind {
    Weights,
    State,
    Compute,
    Experts,
}

impl AllocKind {
    pub fn name(self) -> &'static str {
        match self {
            AllocKind::Weights => "weights",
            AllocKind::State => "state",
            AllocKind::Compute => "compute",
            AllocKind::Experts => "experts",
        }
    }
}

static TOTALS: Mutex<BTreeMap<String, u64>> = Mutex::new(BTreeMap::new());
static OUT: Mutex<()> = Mutex::new(());

/// Writes one event line, a value with a space, quote, or equals sign travels as a JSON string
pub fn emit(name: &str, kv: &[(&str, String)]) {
    let mut line = String::with_capacity(64);
    line.push_str("kungfu event=");
    line.push_str(name);
    for (k, v) in kv {
        line.push(' ');
        line.push_str(k);
        line.push('=');
        line.push_str(&quote(v));
    }
    line.push('\n');
    let _guard = OUT.lock().unwrap_or_else(|e| e.into_inner());
    let _ = std::io::stderr().write_all(line.as_bytes());
}

/// The value as it travels, raw when it has no space, quote, or equals sign or is compact JSON, else a JSON string
pub fn quote(v: &str) -> String {
    let compact = !v.chars().any(|c| c.is_whitespace() || c.is_control());
    let json = compact && (v.starts_with('{') && v.ends_with('}') || v.starts_with('[') && v.ends_with(']'));
    let plain = !v.is_empty() && compact && !v.contains('"') && !v.contains('=');
    if plain || json {
        return v.to_string();
    }
    let mut out = String::with_capacity(v.len() + 2);
    out.push('"');
    for c in v.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Records bytes taken from a pool for a kind and logs the alloc event
pub fn alloc(pool: Pool, kind: AllocKind, bytes: u64) {
    let key = format!("{}.{}", pool.id(), kind.name());
    *TOTALS.lock().unwrap_or_else(|e| e.into_inner()).entry(key).or_insert(0) += bytes;
    emit("alloc", &[("pool", pool.id()), ("kind", kind.name().to_string()), ("bytes", bytes.to_string())]);
}

/// Records bytes given back to a pool for a kind
pub fn release(pool: Pool, kind: AllocKind, bytes: u64) {
    let key = format!("{}.{}", pool.id(), kind.name());
    let mut t = TOTALS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(v) = t.get_mut(&key) {
        *v = v.saturating_sub(bytes);
    }
}

pub fn alloc_failed(pool: Pool, kind: AllocKind, bytes: u64, free: u64) {
    emit("alloc_failed", &[("pool", pool.id()), ("kind", kind.name().to_string()), ("bytes", bytes.to_string()), ("free", free.to_string())]);
}

/// Bytes held per pool and kind, keyed pool.kind as a manager sums them
pub fn totals() -> BTreeMap<String, u64> {
    TOTALS.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

pub fn total(pool: Pool, kind: AllocKind) -> u64 {
    TOTALS.lock().unwrap_or_else(|e| e.into_inner()).get(&format!("{}.{}", pool.id(), kind.name())).copied().unwrap_or(0)
}

/// Logs an error event under a code the self description lists
pub fn error(code: &str, kv: &[(&str, String)]) {
    let mut all = vec![("code", code.to_string())];
    all.extend(kv.iter().map(|(k, v)| (*k, v.clone())));
    emit("error", &all);
}

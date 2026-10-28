//! The launch parameter table, printed by describe and parsed by the command line so neither can drift
use anyhow::{Result, bail};
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::PathBuf;

/// One launch parameter as a manager's form and the command line both see it
#[derive(Clone, Debug, Serialize)]
pub struct Param {
    pub name: &'static str,
    #[serde(rename = "type")]
    pub kind: Kind,
    pub default: &'static str,
    /// The manager chooses the value at launch when it is auto
    pub solved: bool,
    pub label: &'static str,
    pub unit: &'static str,
    pub min: f64,
    pub max: f64,
    pub step: f64,
    pub group: &'static str,
    pub flag: &'static str,
    pub env: &'static str,
    /// What auto resolves to
    pub rule: &'static str,
    pub choices: &'static [&'static str],
    pub advanced: bool,
    /// Stored component architecture the path picks, model for any weights
    pub picks: &'static str,
    pub description: &'static str,
    /// The flag may repeat, every value kept in order
    pub repeated: bool,
}


#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    String,
    Int,
    Float,
    Path,
}

const fn p(name: &'static str, kind: Kind, default: &'static str, group: &'static str, flag: &'static str, label: &'static str, description: &'static str) -> Param {
    Param { name, kind, default, solved: false, label, unit: "", min: 0.0, max: 0.0, step: 0.0, group, flag, env: "", rule: "", choices: &[], advanced: false, picks: "", description, repeated: false }
}

/// Every launch parameter in the order a manager shows them
pub static PARAMS: &[Param] = &[
    Param { picks: "model", ..p("model", Kind::Path, "", "Model", "--model", "Weights", "the first shard of a GGUF") },
    Param { picks: "projector", ..p("mmproj", Kind::Path, "", "Model", "--mmproj", "Vision projector", "the projector file that turns images into the model's embeddings, images refused without it") },
    p("alias", Kind::String, "", "Model", "--alias", "Served name", "the id models lists, the file's name when empty"),
    p("host", Kind::String, "127.0.0.1", "Serving", "--host", "Host", "address the API binds"),
    Param { min: 1.0, max: 65535.0, step: 1.0, ..p("port", Kind::Int, "8080", "Serving", "--port", "Port", "port the API binds") },
    Param { solved: true, unit: "tokens", min: 256.0, step: 256.0, rule: "the largest context that fits the plan, up to the trained length", ..p("n_ctx", Kind::Int, "auto", "Context", "--ctx", "Context length", "tokens of context per sequence") },
    Param { unit: "tokens", min: 32.0, max: 8192.0, step: 32.0, advanced: true, ..p("n_batch", Kind::Int, "512", "Context", "--batch", "Batch size", "most tokens one prefill pass feeds") },
    Param { unit: "sequences", min: 1.0, max: 64.0, step: 1.0, ..p("n_parallel", Kind::Int, "1", "Context", "--parallel", "Parallel sequences", "sequence slots decoded together, one request each, every slot with its own state at the context length") },
    Param { choices: &["auto", "cpu", "cuda"], advanced: true, ..p("backend", Kind::String, "auto", "Placement", "--backend", "Backend", "cuda when a device is visible, else cpu") },
    Param { min: 0.0, step: 1.0, advanced: true, ..p("device", Kind::Int, "0", "Placement", "--device", "Device", "ordinal among the visible devices") },
    Param { advanced: true, ..p("plan", Kind::Path, "", "Placement", "--plan", "Memory plan", "a plan document, per group kind and pool with bytes, honored per layer") },
    Param { advanced: true, ..p("cache_dir", Kind::Path, "", "Placement", "--cache-dir", "Cache directory", "where the routing profile and the PTX cache go") },
    Param { advanced: true, ..p("profile", Kind::Path, "", "Placement", "--profile", "Routing profile", "expert counts to warm the cache from") },
    Param { min: 0.0, max: 2.0, step: 0.05, ..p("temperature", Kind::Float, "0.7", "Sampling", "--temperature", "Temperature", "default when a request sets none, zero for greedy") },
    Param { min: 0.0, step: 1.0, ..p("top_k", Kind::Int, "40", "Sampling", "--top-k", "Top k", "default when a request sets none, zero disables") },
    Param { min: 0.0, max: 1.0, step: 0.01, ..p("top_p", Kind::Float, "0.95", "Sampling", "--top-p", "Top p", "default when a request sets none") },
    Param { min: 0.0, max: 1.0, step: 0.01, advanced: true, ..p("min_p", Kind::Float, "0", "Sampling", "--min-p", "Min p", "default when a request sets none") },
    Param { min: 1.0, max: 2.0, step: 0.01, advanced: true, ..p("repeat_penalty", Kind::Float, "1", "Sampling", "--repeat-penalty", "Repeat penalty", "default when a request sets none") },
    Param { unit: "tokens", min: 1.0, step: 1.0, advanced: true, ..p("max_tokens", Kind::Int, "1024", "Sampling", "--max-tokens", "Max tokens", "default when a request sets none") },
    Param { choices: ROLES, advanced: true, ..p("role", Kind::String, "head", "Formation", "--role", "Role", "head serves the API over its own layers and any stages, replica serves the whole model, stage serves a layer range to a head") },
    Param { min: 0.0, step: 1.0, advanced: true, ..p("rank", Kind::Int, "0", "Formation", "--rank", "Rank", "this seat among those of its role") },
    Param { min: 1.0, step: 1.0, advanced: true, ..p("count", Kind::Int, "1", "Formation", "--count", "Count", "seats in the formation") },
    Param { advanced: true, repeated: true, ..p("peer", Kind::String, "", "Formation", "--peer", "Peer", "role=rank@addr:port of another seat, once per peer") },
    Param { min: 0.0, step: 1.0, advanced: true, ..p("layer_from", Kind::Int, "0", "Formation", "--layer-from", "First layer", "first layer a stage holds") },
    Param { min: 0.0, step: 1.0, advanced: true, ..p("layer_to", Kind::Int, "0", "Formation", "--layer-to", "Layer end", "one past the last layer a stage holds, zero for every layer") },
    Param { advanced: true, repeated: true, ..p("admit", Kind::String, "", "Formation", "--admit", "Admit", "address a stage admits on its port, once per address, every other address refused") },
    Param { advanced: true, ..p("slot_dir", Kind::Path, "", "Formation", "--slot-dir", "Slot directory", "where a prefill or decode seat saves and restores the sequence files the slots API names") },
    Param { picks: "draft", advanced: true, ..p("draft_model", Kind::Path, "", "Speculative decoding", "--draft-model", "Draft model", "a smaller model of the same vocabulary the draft seat proposes tokens from") },
    Param { unit: "tokens", min: 1.0, max: 64.0, step: 1.0, advanced: true, ..p("draft_max", Kind::Int, "3", "Speculative decoding", "--draft-max", "Draft tokens per round", "tokens a draft seat proposes per round") },
];

pub const ROLES: &[&str] = &["head", "replica", "stage", "prefill", "decode", "draft"];

/// Raw values by parameter name after the flags were checked against the table
#[derive(Clone, Debug, Default)]
pub struct Values {
    map: BTreeMap<&'static str, Vec<String>>,
}

impl Values {
    /// Values from the table defaults with every flag of the command line applied, an error naming any flag the table lacks
    pub fn parse(args: &[String]) -> Result<Self> {
        let mut map: BTreeMap<&'static str, Vec<String>> = PARAMS.iter().filter(|p| !p.default.is_empty()).map(|p| (p.name, vec![p.default.to_string()])).collect();
        let mut i = 0;
        while i < args.len() {
            let arg = &args[i];
            let (flag, inline) = match arg.split_once('=') {
                Some((f, v)) if f.starts_with("--") => (f, Some(v.to_string())),
                _ => (arg.as_str(), None),
            };
            let Some(param) = PARAMS.iter().find(|p| p.flag == flag) else { bail!("unknown flag {arg}, kungfu describe lists every flag") };
            i += 1;
            let value = match inline {
                Some(v) => v,
                None => {
                    let Some(v) = args.get(i) else { bail!("flag {flag} needs a value") };
                    i += 1;
                    v.clone()
                }
            };
            check(param, &value)?;
            if param.repeated {
                map.entry(param.name).or_default().push(value);
            } else {
                map.insert(param.name, vec![value]);
            }
        }
        Ok(Self { map })
    }

    pub fn str(&self, name: &str) -> &str {
        self.map.get(name).and_then(|v| v.last()).map_or("", String::as_str)
    }

    pub fn list(&self, name: &str) -> Vec<String> {
        self.map.get(name).cloned().unwrap_or_default()
    }

    pub fn int(&self, name: &str) -> Result<i64> {
        let s = self.str(name);
        s.parse().map_err(|_| anyhow::anyhow!("parameter {name} is {s:?}, not a whole number"))
    }

    pub fn float(&self, name: &str) -> Result<f64> {
        let s = self.str(name);
        s.parse().map_err(|_| anyhow::anyhow!("parameter {name} is {s:?}, not a number"))
    }

    pub fn path(&self, name: &str) -> Option<PathBuf> {
        let s = self.str(name);
        (!s.is_empty() && !s.eq_ignore_ascii_case("none")).then(|| PathBuf::from(s))
    }

    /// Whether a solved parameter was left for the binary to choose
    pub fn is_auto(&self, name: &str) -> bool {
        self.str(name).eq_ignore_ascii_case("auto")
    }
}

/// Rejects a value the parameter's type, choices, or bounds exclude
fn check(param: &Param, value: &str) -> Result<()> {
    if param.solved && value.eq_ignore_ascii_case("auto") {
        return Ok(());
    }
    if !param.choices.is_empty() && !param.choices.contains(&value) {
        bail!("{} takes one of {}, not {value:?}", param.flag, param.choices.join(", "));
    }
    let num = match param.kind {
        Kind::Int => Some(value.parse::<i64>().map_err(|_| anyhow::anyhow!("{} takes a whole number, not {value:?}", param.flag))? as f64),
        Kind::Float => Some(value.parse::<f64>().map_err(|_| anyhow::anyhow!("{} takes a number, not {value:?}", param.flag))?),
        Kind::String | Kind::Path => None,
    };
    if let Some(n) = num {
        if n < param.min {
            bail!("{} is at least {}, not {value}", param.flag, param.min);
        }
        if param.max != 0.0 && n > param.max {
            bail!("{} is at most {}, not {value}", param.flag, param.max);
        }
    }
    Ok(())
}

/// Context length a launch asks for, a number or the planner's choice
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ctx {
    Auto,
    Tokens(usize),
}

/// A peer seat of the formation as the peer flag names it
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Peer {
    pub role: String,
    pub rank: usize,
    pub addr: String,
}

impl Peer {
    /// role=rank@addr:port
    pub fn parse(s: &str) -> Result<Self> {
        let Some((role, rest)) = s.split_once('=') else { bail!("peer {s:?} is not role=rank@addr:port") };
        let Some((rank, addr)) = rest.split_once('@') else { bail!("peer {s:?} is not role=rank@addr:port") };
        if !ROLES.contains(&role) {
            bail!("peer {s:?} names role {role:?}, one of {} expected", ROLES.join(", "));
        }
        let rank = rank.parse().map_err(|_| anyhow::anyhow!("peer {s:?} has rank {rank:?}, a number expected"))?;
        if addr.rsplit_once(':').and_then(|(_, p)| p.parse::<u16>().ok()).is_none() {
            bail!("peer {s:?} has address {addr:?}, addr:port expected");
        }
        Ok(Self { role: role.to_string(), rank, addr: addr.to_string() })
    }
}

/// The seat flags of a formation launch
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Seat {
    pub role: String,
    pub rank: usize,
    pub count: usize,
    pub peers: Vec<Peer>,
    pub layer_from: usize,
    pub layer_to: usize,
    pub admit: Vec<String>,
    pub slot_dir: Option<PathBuf>,
    pub draft_model: Option<PathBuf>,
    pub draft_max: usize,
}

/// Everything a launch needs, typed from the table's values
#[derive(Clone, Debug)]
pub struct Launch {
    pub model: PathBuf,
    pub mmproj: Option<PathBuf>,
    pub alias: String,
    pub host: String,
    pub port: u16,
    pub ctx: Ctx,
    pub n_batch: usize,
    pub n_parallel: usize,
    pub backend: String,
    pub device: usize,
    pub plan: Option<PathBuf>,
    pub cache_dir: Option<PathBuf>,
    pub profile: Option<PathBuf>,
    pub sampling: kf_engine::Sampling,
    pub max_tokens: usize,
    pub seat: Seat,
}

impl Launch {
    pub fn parse(args: &[String]) -> Result<Self> {
        Self::from_values(Values::parse(args)?)
    }

    /// Reads every parameter of the table, so a parameter listed and never read cannot exist
    pub fn from_values(values: Values) -> Result<Self> {
        let mut read = std::collections::HashSet::new();
        let mut v = |name: &'static str| {
            read.insert(name);
            &values
        };
        let model = v("model").path("model").ok_or_else(|| anyhow::anyhow!("--model names the first shard of a GGUF"))?;
        let ctx = if v("n_ctx").is_auto("n_ctx") { Ctx::Auto } else { Ctx::Tokens(v("n_ctx").int("n_ctx")? as usize) };
        let sampling = kf_engine::Sampling {
            temperature: v("temperature").float("temperature")? as f32,
            top_k: v("top_k").int("top_k")? as usize,
            top_p: v("top_p").float("top_p")? as f32,
            min_p: v("min_p").float("min_p")? as f32,
            repeat_penalty: v("repeat_penalty").float("repeat_penalty")? as f32,
            ..Default::default()
        };
        let peers = v("peer").list("peer").iter().map(|s| Peer::parse(s)).collect::<Result<Vec<_>>>()?;
        let seat = Seat {
            role: v("role").str("role").to_string(),
            rank: v("rank").int("rank")? as usize,
            count: v("count").int("count")? as usize,
            peers,
            layer_from: v("layer_from").int("layer_from")? as usize,
            layer_to: v("layer_to").int("layer_to")? as usize,
            admit: v("admit").list("admit"),
            slot_dir: v("slot_dir").path("slot_dir"),
            draft_model: v("draft_model").path("draft_model"),
            draft_max: v("draft_max").int("draft_max")? as usize,
        };
        let launch = Self {
            model,
            mmproj: v("mmproj").path("mmproj"),
            alias: v("alias").str("alias").to_string(),
            host: v("host").str("host").to_string(),
            port: v("port").int("port")? as u16,
            ctx,
            n_batch: v("n_batch").int("n_batch")? as usize,
            n_parallel: v("n_parallel").int("n_parallel")? as usize,
            backend: v("backend").str("backend").to_string(),
            device: v("device").int("device")? as usize,
            plan: v("plan").path("plan"),
            cache_dir: v("cache_dir").path("cache_dir"),
            profile: v("profile").path("profile"),
            sampling,
            max_tokens: v("max_tokens").int("max_tokens")? as usize,
            seat,
        };
        if let Some(p) = PARAMS.iter().find(|p| !read.contains(p.name)) {
            bail!("parameter {} is listed but never read", p.name);
        }
        Ok(launch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_flag_parses_and_is_read() {
        let mut args: Vec<String> = Vec::new();
        for p in PARAMS {
            args.push(p.flag.to_string());
            args.push(match p.name {
                "model" => "model.gguf".into(),
                "n_ctx" => "auto".into(),
                "peer" => "stage=0@10.0.0.2:5000".into(),
                "admit" => "10.0.0.1".into(),
                "slot_dir" => "slots".into(),
                _ if !p.default.is_empty() => p.default.into(),
                _ if p.kind == Kind::Path => "some/path".into(),
                _ => "x".into(),
            });
        }
        let l = Launch::parse(&args).expect("every listed flag parses");
        assert_eq!(l.ctx, Ctx::Auto);
        assert_eq!(l.n_batch, 512);
        assert_eq!(l.seat.peers, vec![Peer { role: "stage".into(), rank: 0, addr: "10.0.0.2:5000".into() }]);
        assert_eq!(l.seat.admit, vec!["10.0.0.1".to_string()]);
    }

    #[test]
    fn unknown_flag_is_refused() {
        let err = Launch::parse(&["--model".into(), "m.gguf".into(), "--nope".into(), "1".into()]).unwrap_err();
        assert!(err.to_string().contains("--nope"));
    }

    #[test]
    fn bounds_and_choices_are_checked() {
        assert!(Launch::parse(&["--model".into(), "m".into(), "--port".into(), "70000".into()]).is_err());
        assert!(Launch::parse(&["--model".into(), "m".into(), "--backend".into(), "metal".into()]).is_err());
        assert!(Launch::parse(&["--model".into(), "m".into(), "--role".into(), "king".into()]).is_err());
        assert!(Launch::parse(&["--model".into(), "m".into(), "--peer".into(), "stage=x@h:1".into()]).is_err());
        let two = Launch::parse(&["--model".into(), "m".into(), "--peer".into(), "stage=0@h:1".into(), "--peer".into(), "stage=1@h:2".into()]).expect("two peers");
        assert_eq!(two.seat.peers.len(), 2);
        assert!(Launch::parse(&["--model".into(), "m".into(), "--ctx".into(), "100".into()]).is_err());
        assert!(Launch::parse(&["--model".into(), "m".into(), "--ctx=4096".into()]).is_ok());
    }
}

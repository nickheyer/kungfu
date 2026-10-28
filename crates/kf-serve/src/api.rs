use crate::chat::{self, ChatSpec, ResponseFormat, ToolChoice};
use crate::describe::{STOP_GRACE_SECONDS, describe};
use crate::engine::{Cmd, DraftConn, Engine, Outcome, Piece, SlotAction};
use crate::params::{Ctx, Launch};
use crate::parse::{Delta, Format, StreamParser, parse_output};
use crate::template::{ChatTemplate, Message};
use crate::tokenizer::Tok;
use anyhow::{Result, anyhow};
use axum::extract::{Path as AxPath, Query, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_util::stream::Stream;
use kf_core::{AllocKind, Pool, event};
use kf_engine::{BackendChoice, CtxChoice, Sampling, Scope, Session, StageConn, StageRunner};
use serde::Deserialize;
use serde_json::{Value, json};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot};

/// The model behind the API once it is warm
pub struct Served {
    pub tok: Arc<Tok>,
    /// The chat template of the file, none on a draft seat which renders no chats
    pub template: Option<ChatTemplate>,
    pub format: Format,
    /// Whether the template speaks think tags, so a grammar admits a think block before the answer
    pub thinks: bool,
    pub model_name: String,
    pub engine: Engine,
    pub defaults: Sampling,
    pub max_tokens: usize,
    pub n_ctx: usize,
    pub n_seq: usize,
    pub slot_dir: Option<std::path::PathBuf>,
    /// The vision tower that turns images into prompt rows, none without a projector file
    pub vision: Option<Arc<kf_vision::Clip>>,
    pub metrics: Arc<Metrics>,
}

#[derive(Deserialize)]
pub struct DraftRequest {
    pub tokens: Vec<u32>,
    pub n: usize,
}

/// Counters the metrics endpoint prints
#[derive(Default)]
pub struct Metrics {
    pub requests: AtomicU64,
    pub prompt_tokens: AtomicU64,
    pub cached_tokens: AtomicU64,
    pub completion_tokens: AtomicU64,
    pub cancelled: AtomicU64,
    pub cache_stats: Mutex<Vec<(usize, usize, u64, u64, u64, u64)>>,
}

const LOADING: u8 = 0;
const WARMING: u8 = 1;
const READY: u8 = 2;

/// Where startup stands, read by the health check
pub struct Phase {
    kind: AtomicU8,
    planned: AtomicU64,
    pool: Mutex<Option<Pool>>,
    warmed: AtomicBool,
}

impl Phase {
    fn progress(&self) -> f64 {
        match self.kind.load(Ordering::Relaxed) {
            LOADING => {
                let planned = self.planned.load(Ordering::Relaxed);
                let pool = *self.pool.lock().unwrap_or_else(|e| e.into_inner());
                match (planned, pool) {
                    (p, Some(pool)) if p > 0 => {
                        let have: u64 = [AllocKind::Weights, AllocKind::State, AllocKind::Experts].into_iter().map(|k| event::total(pool, k)).sum();
                        (have as f64 / p as f64).min(1.0)
                    }
                    _ => 0.0,
                }
            }
            WARMING => {
                if self.warmed.load(Ordering::Relaxed) {
                    1.0
                } else {
                    0.5
                }
            }
            _ => 1.0,
        }
    }

    fn name(&self) -> &'static str {
        match self.kind.load(Ordering::Relaxed) {
            LOADING => "loading",
            WARMING => "warming",
            _ => "ready",
        }
    }
}

/// The listener's state, the phase from the first byte and the model once it is warm
pub struct App {
    phase: Phase,
    served: Arc<RwLock<Option<Arc<Served>>>>,
}

#[derive(Deserialize)]
pub struct ChatRequest {
    pub messages: Vec<Message>,
    #[serde(default)]
    pub tools: Option<Value>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
    #[serde(default)]
    pub response_format: Option<Value>,
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub min_p: Option<f32>,
    #[serde(default)]
    pub repeat_penalty: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub stop: Option<Value>,
    #[serde(default)]
    pub logit_bias: Option<serde_json::Map<String, Value>>,
    #[serde(default)]
    pub chat_template_kwargs: serde_json::Map<String, Value>,
    /// The one sequence slot this seat keeps, as the relay gateway names it
    #[serde(default)]
    pub id_slot: Option<usize>,
}

/// An error already logged as an error event
#[derive(Debug)]
struct Logged(anyhow::Error);

impl std::fmt::Display for Logged {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:#}", self.0)
    }
}

impl std::error::Error for Logged {}

/// Serves until the process is told to stop, every failure logged as an error event before it returns
pub async fn serve(launch: Launch) -> Result<()> {
    let cache_dir = launch.cache_dir.clone();
    let served = Arc::new(RwLock::new(None));
    let r = match launch.seat.role.as_str() {
        "stage" => serve_stage_role(launch).await,
        _ => serve_inner(launch, served.clone()).await,
    };
    if let Err(e) = &r
        && e.downcast_ref::<Logged>().is_none()
    {
        event::error("backend", &[("detail", format!("{e:#}"))]);
    }
    if let (Some(dir), Some(s)) = (cache_dir, served.read().ok().and_then(|g| g.clone())) {
        let s: Arc<Served> = s;
        let engine = s.engine.clone();
        let profile = tokio::task::spawn_blocking(move || engine.profile()).await.ok().flatten();
        if let Some(p) = profile {
            std::fs::create_dir_all(&dir)?;
            p.write(&dir.join("routing_profile.json"))?;
        }
    }
    r
}

async fn serve_inner(launch: Launch, served: Arc<RwLock<Option<Arc<Served>>>>) -> Result<()> {
    let listener = match tokio::net::TcpListener::bind((launch.host.as_str(), launch.port)).await {
        Ok(l) => l,
        Err(e) => {
            return Err(fail("bind", &[("host", launch.host.clone()), ("port", launch.port.to_string()), ("detail", e.to_string())]));
        }
    };
    let app = Arc::new(App { phase: Phase { kind: AtomicU8::new(LOADING), planned: AtomicU64::new(0), pool: Mutex::new(None), warmed: AtomicBool::new(false) }, served });
    let router = Router::new().route("/health", get(health)).route("/describe", get(describe_route)).route("/metrics", get(metrics));
    let router = if launch.seat.role == "draft" {
        router.route("/draft", post(draft))
    } else {
        router
            .route("/v1/models", get(models))
            .route("/v1/chat/completions", post(chat))
            .route("/v1/messages", post(crate::anthropic::messages))
            .route("/v1/responses", post(crate::responses::responses))
            .route("/slots/{id}", post(slot))
    };
    let router = router.with_state(app.clone());
    let (stop_tx, stop_rx) = tokio::sync::oneshot::channel::<()>();
    let server = tokio::spawn(async move {
        axum::serve(listener, router)
            .with_graceful_shutdown(async move {
                let _ = stop_rx.await;
            })
            .await
    });
    let loading = app.clone();
    let loader = tokio::task::spawn_blocking(move || load_and_warm(launch, loading));
    let mut server = server;
    tokio::select! {
        r = &mut server => {
            r.map_err(|e| anyhow!("server task: {e}"))??;
            return Ok(());
        }
        l = loader => {
            l.map_err(|e| anyhow!("load task: {e}"))??;
        }
    }
    let cache_app = app.clone();
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(Duration::from_secs(10));
        tick.tick().await;
        loop {
            tick.tick().await;
            let Some(s) = cache_app.served.read().ok().and_then(|g| g.clone()) else { continue };
            let stats = s.engine.cache_stats().await;
            for (layer, slots, hits, misses, pre, pre_hits) in &stats {
                let total = hits + misses;
                let rate = if total == 0 { 0.0 } else { *hits as f64 / total as f64 };
                let pre_rate = if *pre == 0 { 0.0 } else { *pre_hits as f64 / *pre as f64 };
                event::emit("cache", &[("layer", layer.to_string()), ("slots", slots.to_string()), ("hit_rate", format!("{rate:.4}")), ("prefetched", pre.to_string()), ("prefetch_hit_rate", format!("{pre_rate:.4}"))]);
            }
            if let Ok(mut g) = s.metrics.cache_stats.lock() {
                *g = stats;
            }
        }
    });
    tokio::select! {
        r = &mut server => {
            r.map_err(|e| anyhow!("server task: {e}"))??;
        }
        _ = terminate() => {
            let _ = stop_tx.send(());
            let _ = tokio::time::timeout(Duration::from_secs(STOP_GRACE_SECONDS), server).await;
        }
    }
    Ok(())
}

/// SIGTERM or an interrupt
async fn terminate() {
    #[cfg(unix)]
    {
        let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).expect("SIGTERM handler");
        tokio::select! {
            _ = term.recv() => {}
            _ = tokio::signal::ctrl_c() => {}
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}

/// Logs an error event under a code and returns the error marked as logged
fn fail(code: &str, kv: &[(&str, String)]) -> anyhow::Error {
    event::error(code, kv);
    let detail = kv.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join(" ");
    anyhow::Error::new(Logged(anyhow!("{code}: {detail}")))
}

/// The backend a launch names, cuda when a device is visible under auto
fn choose_backend(launch: &Launch) -> Result<BackendChoice> {
    Ok(match launch.backend.as_str() {
        "cuda" => {
            let visible = kf_engine::device::devices().len();
            if launch.device >= visible {
                return Err(fail("no_device", &[("device", launch.device.to_string()), ("visible", visible.to_string())]));
            }
            BackendChoice::Cuda { device: launch.device }
        }
        "cpu" => BackendChoice::Cpu,
        _ => BackendChoice::auto(launch.device)?,
    })
}

/// Plans and opens the backend for a scope, each failure under its error code
fn prepare_backend(launch: &Launch, loaded: &kf_engine::Loaded, choice: BackendChoice, scope: &Scope) -> Result<kf_engine::Prepared> {
    let ctx = match launch.ctx {
        Ctx::Auto => CtxChoice::Auto,
        Ctx::Tokens(n) => CtxChoice::Tokens(n),
    };
    kf_engine::prepare(loaded, choice, ctx, launch.n_batch, launch.n_parallel, launch.plan.as_deref(), launch.cache_dir.as_deref(), launch.profile.as_deref(), scope).map_err(|e| {
        let detail = format!("{e:#}");
        match (launch.plan.as_ref(), detail.contains("free:")) {
            (_, true) => fail("device_oom", &[("detail", detail)]),
            (Some(p), _) => fail("plan", &[("path", p.display().to_string()), ("detail", detail)]),
            (None, _) => fail("no_fit", &[("n_ctx", match launch.ctx { Ctx::Auto => "auto".to_string(), Ctx::Tokens(n) => n.to_string() }), ("detail", detail)]),
        }
    })
}

/// Loads the file and refuses it when the module leaves tensors unclaimed
fn load_claimed(launch: &Launch) -> Result<kf_engine::Loaded> {
    let loaded = kf_engine::load(&launch.model).map_err(|e| fail(e.code(), &e.keys()))?;
    let unclaimed = loaded.unclaimed();
    if !unclaimed.is_empty() {
        return Err(fail("unclaimed_tensors", &[("arch", loaded.hparams.arch.clone()), ("count", unclaimed.len().to_string()), ("names", unclaimed.join(","))]));
    }
    Ok(loaded)
}

/// Refuses a seat whose rank is not below its count
fn check_seat(launch: &Launch) -> Result<()> {
    if launch.seat.rank >= launch.seat.count {
        return Err(fail("bad_flag", &[("detail", format!("rank {} of {} seats", launch.seat.rank, launch.seat.count))]));
    }
    Ok(())
}

/// Serves a layer range to a head over the chain protocol on the seat's port, admitting the addresses named
async fn serve_stage_role(launch: Launch) -> Result<()> {
    let t0 = Instant::now();
    check_seat(&launch)?;
    let admit = kf_engine::admitted(&launch.seat.admit).map_err(|e| fail("bad_flag", &[("detail", format!("{e:#}"))]))?;
    if admit.is_empty() {
        return Err(fail("bad_flag", &[("detail", "a stage admits nothing without --admit, name the head's address".to_string())]));
    }
    if !launch.seat.peers.is_empty() {
        return Err(fail("bad_flag", &[("detail", "a stage takes no peers, the head connects to it".to_string())]));
    }
    let loaded = load_claimed(&launch)?;
    let n_layers = loaded.model.n_layers();
    let (from, to) = match (launch.seat.layer_from, launch.seat.layer_to) {
        (0, 0) => (0, n_layers),
        (a, b) => (a, b),
    };
    if from >= to || to > n_layers {
        return Err(fail("bad_flag", &[("detail", format!("stage layers {from}..{to} of a {n_layers} layer model"))]));
    }
    let choice = choose_backend(&launch)?;
    let scope = Scope::range(from, to);
    let prepared = prepare_backend(&launch, &loaded, choice, &scope)?;
    let n_ctx = prepared.plan.n_ctx;
    let mut runner = StageRunner::new(&loaded, prepared.backend, &prepared.plan, from, to).map_err(|e| fail("compose", &[("arch", loaded.hparams.arch.clone()), ("detail", format!("{e:#}"))]))?;
    let width = loaded.hparams.n_embd * loaded.model.streams;
    let bos = bos_token(&loaded);
    runner.forward(&[bos], &[0], &[0], Vec::new(), vec![0], vec![0.0; width], width).map_err(|e| fail("backend", &[("detail", format!("{e:#}"))]))?;
    runner.forward(&[bos], &[1], &[0], Vec::new(), vec![1], vec![0.0; width], width).map_err(|e| fail("backend", &[("detail", format!("{e:#}"))]))?;
    runner.reset(None);
    let listener = std::net::TcpListener::bind((launch.host.as_str(), launch.port)).map_err(|e| fail("bind", &[("host", launch.host.clone()), ("port", launch.port.to_string()), ("detail", e.to_string())]))?;
    let measured = serde_json::to_string(&event::totals())?;
    event::emit("ready", &[("n_ctx", n_ctx.to_string()), ("warm_ms", t0.elapsed().as_millis().to_string()), ("measured", measured), ("role", "stage".to_string()), ("rank", launch.seat.rank.to_string()), ("count", launch.seat.count.to_string()), ("layer_from", from.to_string()), ("layer_to", to.to_string())]);
    let stop = Arc::new(AtomicBool::new(false));
    let flag = stop.clone();
    let mut server = tokio::task::spawn_blocking(move || kf_engine::serve_stage(listener, &admit, runner, flag));
    let runner = tokio::select! {
        r = &mut server => r.map_err(|e| anyhow!("stage task: {e}"))??,
        _ = terminate() => {
            stop.store(true, Ordering::Relaxed);
            server.await.map_err(|e| anyhow!("stage task: {e}"))??
        }
    };
    if let (Some(dir), Some(p)) = (&launch.cache_dir, kf_engine::RoutingProfile::of(runner.backend_ref())) {
        std::fs::create_dir_all(dir)?;
        p.write(&dir.join("routing_profile.json"))?;
    }
    Ok(())
}

/// The token a warm pass feeds, the beginning of sequence token or zero
fn bos_token(loaded: &kf_engine::Loaded) -> u32 {
    loaded.store.gguf.meta.u32("tokenizer.ggml.bos_token_id").unwrap_or(0)
}

/// Loads the file, plans, connects any stages, warms one prefill and one decode, and hands the model to the listener
fn load_and_warm(launch: Launch, app: Arc<App>) -> Result<()> {
    let t0 = Instant::now();
    check_seat(&launch)?;
    if launch.seat.layer_from != 0 || launch.seat.layer_to != 0 {
        return Err(fail("bad_flag", &[("detail", format!("a {} holds the layers its stages do not, --layer-from and --layer-to belong to a stage", launch.seat.role))]));
    }
    let relay = matches!(launch.seat.role.as_str(), "prefill" | "decode");
    if relay && launch.seat.slot_dir.is_none() {
        return Err(fail("bad_flag", &[("detail", format!("a {} seat hands sequences over through --slot-dir", launch.seat.role))]));
    }
    if relay && !launch.seat.peers.is_empty() {
        return Err(fail("bad_flag", &[("detail", format!("a {} seat serves the whole model and takes no peers, the gateway joins the pair", launch.seat.role))]));
    }
    if let Some(dir) = &launch.seat.slot_dir {
        std::fs::create_dir_all(dir).map_err(|e| fail("bad_flag", &[("detail", format!("slot directory {}: {e}", dir.display()))]))?;
    }
    let is_draft = launch.seat.role == "draft";
    if is_draft && !launch.seat.peers.is_empty() {
        return Err(fail("bad_flag", &[("detail", "a draft seat takes no peers, the head connects to it".to_string())]));
    }
    if !is_draft && launch.seat.draft_model.is_some() {
        return Err(fail("bad_flag", &[("detail", format!("--draft-model belongs to the draft seat, a {} proposes nothing", launch.seat.role))]));
    }
    let loaded = if is_draft {
        let Some(draft_path) = &launch.seat.draft_model else { return Err(fail("bad_flag", &[("detail", "a draft seat proposes from --draft-model".to_string())])) };
        let main = kf_gguf::Gguf::open(&launch.model).map_err(|e| fail("model_open", &[("path", launch.model.display().to_string()), ("detail", format!("{e:#}"))]))?;
        let main_vocab = main.meta.arr_len("tokenizer.ggml.tokens").unwrap_or(0);
        drop(main);
        let draft = kf_engine::load(draft_path).map_err(|e| fail(e.code(), &e.keys()))?;
        let draft_vocab = draft.store.gguf.meta.arr_len("tokenizer.ggml.tokens").unwrap_or(0);
        if draft_vocab != main_vocab {
            return Err(fail("bad_flag", &[("detail", format!("draft model has {draft_vocab} tokens, the model {main_vocab}, a draft needs the same vocabulary"))]));
        }
        let unclaimed = draft.unclaimed();
        if !unclaimed.is_empty() {
            return Err(fail("unclaimed_tensors", &[("arch", draft.hparams.arch.clone()), ("count", unclaimed.len().to_string()), ("names", unclaimed.join(","))]));
        }
        draft
    } else {
        load_claimed(&launch)?
    };
    let tok = Arc::new(Tok::from_gguf(&loaded.store.gguf.meta).map_err(|e| fail("tokenizer", &[("detail", format!("{e:#}"))]))?);
    let (template, format, thinks) = if is_draft {
        (None, Format::None, false)
    } else {
        let source = loaded.store.gguf.meta.str("tokenizer.chat_template").ok_or_else(|| fail("no_chat_template", &[("path", launch.model.display().to_string())]))?.to_string();
        let template = ChatTemplate::new(&source).map_err(|e| fail("no_chat_template", &[("path", launch.model.display().to_string()), ("detail", format!("{e:#}"))]))?;
        let thinks = source.contains(crate::engine::THINK_OPEN) || source.contains("enable_thinking");
        (Some(template), Format::detect(&source), thinks)
    };
    let model_name = if launch.alias.is_empty() { loaded.store.gguf.meta.str("general.name").unwrap_or("kungfu").to_string() } else { launch.alias.clone() };
    let choice = choose_backend(&launch)?;
    let stage_peers: Vec<&crate::params::Peer> = launch.seat.peers.iter().filter(|p| p.role == "stage").collect();
    let draft_peers: Vec<&crate::params::Peer> = launch.seat.peers.iter().filter(|p| p.role == "draft").collect();
    if launch.seat.role == "replica" && !launch.seat.peers.is_empty() {
        return Err(fail("bad_flag", &[("detail", "a replica serves the whole model and takes no peers".to_string())]));
    }
    if let Some(p) = launch.seat.peers.iter().find(|p| p.role != "stage" && p.role != "draft") {
        return Err(fail("bad_flag", &[("detail", format!("a head takes stage and draft peers only, not {}", p.role))]));
    }
    if draft_peers.len() > 1 {
        return Err(fail("bad_flag", &[("detail", format!("a head verifies one draft seat, {} named", draft_peers.len()))]));
    }
    let draft = match draft_peers.first() {
        Some(p) => {
            let url = format!("http://{}/draft", p.addr);
            let conn = DraftConn { url, n: launch.seat.draft_max };
            if launch.n_parallel != 1 {
                return Err(fail("bad_flag", &[("detail", format!("a head with a draft seat decodes one sequence at a time, --parallel {} was given", launch.n_parallel))]));
            }
            conn.draft(&[bos_token(&loaded)]).map_err(|e| fail("backend", &[("detail", format!("{e:#}"))]))?;
            event::emit("transport", &[("kind", "tcp".to_string()), ("peer", p.addr.clone())]);
            Some(conn)
        }
        None => None,
    };
    let mut stages = Vec::new();
    for p in &stage_peers {
        stages.push(StageConn::connect(&p.addr, p.rank).map_err(|e| fail("backend", &[("detail", format!("{e:#}"))]))?);
    }
    let ranges: Vec<(usize, usize)> = stages.iter().map(|s| (s.from, s.to)).collect();
    let scope = Scope::outside(loaded.model.n_layers(), &ranges);
    let prepared = prepare_backend(&launch, &loaded, choice, &scope)?;
    let main = match choice {
        BackendChoice::Cpu => Pool::Host,
        BackendChoice::Cuda { device } => Pool::Device(device as u32),
    };
    let planned: u64 = prepared.doc.pools.iter().filter(|p| p.id == main.id()).map(|p| p.weights + p.state + p.experts).sum();
    app.phase.planned.store(planned, Ordering::Relaxed);
    *app.phase.pool.lock().unwrap_or_else(|e| e.into_inner()) = Some(main);
    let n_ctx = prepared.plan.n_ctx;
    let mut session = if stages.is_empty() {
        Session::new(&loaded, prepared.backend, &prepared.plan)
    } else {
        Session::with_stages(&loaded, prepared.backend, &prepared.plan, stages).map_err(|e| fail("backend", &[("detail", format!("{e:#}"))]))?
    };
    let first = tok.bos.unwrap_or(0);
    let second = tok.eos.first().copied().unwrap_or(first);
    session.feed(0, &[first]).map_err(|e| fail("backend", &[("detail", format!("{e:#}"))]))?;
    app.phase.kind.store(WARMING, Ordering::Relaxed);
    session.feed(0, &[second]).map_err(|e| fail("backend", &[("detail", format!("{e:#}"))]))?;
    if session.n_seq() > 1 {
        let all: Vec<(usize, u32)> = (1..session.n_seq()).map(|s| (s, first)).collect();
        session.step(&all).map_err(|e| fail("backend", &[("detail", format!("{e:#}"))]))?;
    }
    app.phase.warmed.store(true, Ordering::Relaxed);
    session.reset_all().map_err(|e| fail("backend", &[("detail", format!("{e:#}"))]))?;
    let measured = event::totals();
    let measured_json = serde_json::to_string(&measured)?;
    event::emit("ready", &[("n_ctx", n_ctx.to_string()), ("warm_ms", t0.elapsed().as_millis().to_string()), ("measured", measured_json), ("role", launch.seat.role.clone()), ("rank", launch.seat.rank.to_string()), ("count", launch.seat.count.to_string()), ("stages", stage_peers.len().to_string()), ("draft", draft_peers.len().to_string())]);
    let vision = match &launch.mmproj {
        Some(path) => {
            let clip = kf_vision::Clip::load(path).map_err(|e| fail("model_open", &[("path", path.display().to_string()), ("detail", format!("{e:#}"))]))?;
            if clip.mrope() != session.mrope() {
                return Err(fail("compose", &[("arch", loaded.hparams.arch.clone()), ("detail", format!("projector {} places image tokens {} but the model's rope is {}", path.display(), if clip.mrope() { "on a grid" } else { "in sequence" }, if session.mrope() { "multi stream" } else { "single stream" }))]));
            }
            Some(Arc::new(clip))
        }
        None => None,
    };
    let metrics = Arc::new(Metrics::default());
    let n_seq = session.n_seq();
    let engine = Engine::start(session, tok.clone(), draft, vision.clone(), metrics.clone());
    let served = Arc::new(Served {
        tok,
        template,
        format,
        thinks,
        model_name,
        engine,
        defaults: launch.sampling.clone(),
        max_tokens: launch.max_tokens,
        n_ctx,
        n_seq,
        slot_dir: launch.seat.slot_dir.clone(),
        vision,
        metrics,
    });
    *app.served.write().map_err(|_| anyhow!("app state poisoned"))? = Some(served);
    app.phase.kind.store(READY, Ordering::Relaxed);
    Ok(())
}

pub fn ready(app: &App) -> Option<Arc<Served>> {
    app.served.read().ok().and_then(|g| g.clone())
}

async fn health(State(app): State<Arc<App>>) -> Response {
    match ready(&app) {
        Some(_) => Json(json!({ "status": "ok" })).into_response(),
        None => (StatusCode::SERVICE_UNAVAILABLE, Json(json!({ "phase": app.phase.name(), "progress": app.phase.progress() }))).into_response(),
    }
}

async fn describe_route() -> Json<Value> {
    Json(describe())
}

async fn models(State(app): State<Arc<App>>) -> Response {
    match ready(&app) {
        Some(s) => Json(json!({ "object": "list", "data": [{ "id": s.model_name, "object": "model", "owned_by": "kungfu" }] })).into_response(),
        None => not_ready(&app),
    }
}

pub fn not_ready(app: &App) -> Response {
    (StatusCode::SERVICE_UNAVAILABLE, Json(json!({ "error": { "message": format!("model is {}", app.phase.name()), "phase": app.phase.name(), "progress": app.phase.progress() } }))).into_response()
}

/// Prometheus text of the request counters, the expert caches, and the memory totals
async fn metrics(State(app): State<Arc<App>>) -> Response {
    let mut out = String::new();
    let ready_flag = ready(&app).is_some() as u8;
    out.push_str("# TYPE kungfu_ready gauge\n");
    out.push_str(&format!("kungfu_ready {ready_flag}\n"));
    if let Some(s) = ready(&app) {
        let m = &s.metrics;
        for (name, v) in [
            ("kungfu_requests_total", m.requests.load(Ordering::Relaxed)),
            ("kungfu_prompt_tokens_total", m.prompt_tokens.load(Ordering::Relaxed)),
            ("kungfu_cached_tokens_total", m.cached_tokens.load(Ordering::Relaxed)),
            ("kungfu_completion_tokens_total", m.completion_tokens.load(Ordering::Relaxed)),
            ("kungfu_cancelled_total", m.cancelled.load(Ordering::Relaxed)),
        ] {
            out.push_str(&format!("# TYPE {name} counter\n{name} {v}\n"));
        }
        out.push_str(&format!("# TYPE kungfu_context_tokens gauge\nkungfu_context_tokens {}\n", s.n_ctx));
        out.push_str(&format!("# TYPE kungfu_sequence_slots gauge\nkungfu_sequence_slots {}\n", s.n_seq));
        if let Ok(stats) = m.cache_stats.lock() {
            out.push_str("# TYPE kungfu_expert_cache_hits_total counter\n# TYPE kungfu_expert_cache_misses_total counter\n# TYPE kungfu_expert_cache_slots gauge\n# TYPE kungfu_expert_prefetch_total counter\n# TYPE kungfu_expert_prefetch_hits_total counter\n");
            for (layer, slots, hits, misses, pre, pre_hits) in stats.iter() {
                out.push_str(&format!("kungfu_expert_cache_slots{{layer=\"{layer}\"}} {slots}\nkungfu_expert_cache_hits_total{{layer=\"{layer}\"}} {hits}\nkungfu_expert_cache_misses_total{{layer=\"{layer}\"}} {misses}\nkungfu_expert_prefetch_total{{layer=\"{layer}\"}} {pre}\nkungfu_expert_prefetch_hits_total{{layer=\"{layer}\"}} {pre_hits}\n"));
            }
        }
    }
    out.push_str("# TYPE kungfu_memory_bytes gauge\n");
    for (key, bytes) in event::totals() {
        let (pool, kind) = key.rsplit_once('.').unwrap_or((key.as_str(), ""));
        out.push_str(&format!("kungfu_memory_bytes{{pool=\"{pool}\",kind=\"{kind}\"}} {bytes}\n"));
    }
    ([("content-type", "text/plain; version=0.0.4")], out).into_response()
}

pub struct Failure(pub anyhow::Error);

impl IntoResponse for Failure {
    fn into_response(self) -> Response {
        (StatusCode::BAD_REQUEST, Json(json!({ "error": { "message": self.0.to_string() } }))).into_response()
    }
}

impl From<anyhow::Error> for Failure {
    fn from(e: anyhow::Error) -> Self {
        Self(e)
    }
}

/// Sets the flag when dropped, which happens when the client goes away before the answer
pub struct CancelOnDrop(pub Arc<AtomicBool>);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

/// Proposes up to n greedy tokens after the context on the draft model, reusing the prefix it shares with the last call
async fn draft(State(app): State<Arc<App>>, Json(req): Json<DraftRequest>) -> Result<Response, Failure> {
    let Some(s) = ready(&app) else { return Ok(not_ready(&app)) };
    let (reply, rx) = oneshot::channel();
    s.engine.send(Cmd::Draft { tokens: req.tokens, n: req.n, reply })?;
    let out = rx.await.map_err(|_| anyhow!("the engine stopped"))??;
    Ok(Json(json!({ "tokens": out })).into_response())
}

#[derive(Deserialize)]
struct SlotQuery {
    action: String,
}

#[derive(Deserialize)]
struct SlotFile {
    filename: String,
}

/// Saves or restores one sequence slot as a file under the slot directory, the handoff a relay gateway drives
async fn slot(State(app): State<Arc<App>>, AxPath(id): AxPath<usize>, Query(q): Query<SlotQuery>, Json(body): Json<SlotFile>) -> Result<Response, Failure> {
    let Some(s) = ready(&app) else { return Ok(not_ready(&app)) };
    if id >= s.n_seq {
        return Err(anyhow!("this seat keeps {} slots, slot {id} does not exist", s.n_seq).into());
    }
    let Some(dir) = s.slot_dir.clone() else { return Err(anyhow!("this seat was launched without --slot-dir, so it keeps no slot files").into()) };
    if body.filename.is_empty() || body.filename.contains(['/', '\\']) || body.filename == "." || body.filename == ".." {
        return Err(anyhow!("filename {:?} must be a plain name inside the slot directory", body.filename).into());
    }
    let action = match q.action.as_str() {
        "save" => SlotAction::Save,
        "restore" => SlotAction::Restore,
        other => return Err(anyhow!("action {other:?}, save or restore expected").into()),
    };
    let (reply, rx) = oneshot::channel();
    s.engine.send(Cmd::Slot { id, action, path: dir.join(&body.filename), name: body.filename, reply })?;
    let out = rx.await.map_err(|_| anyhow!("the engine stopped"))??;
    Ok(Json(out).into_response())
}

/// The OpenAI chat request as a spec
fn chat_spec(req: &ChatRequest) -> Result<ChatSpec> {
    let mut spec = ChatSpec::new(req.messages.clone());
    spec.tools = match &req.tools {
        Some(Value::Array(a)) => Some(a.clone()),
        Some(Value::Null) | None => None,
        Some(other) => return Err(anyhow!("tools {other} is not an array")),
    };
    spec.tool_choice = ToolChoice::parse(req.tool_choice.as_ref())?;
    spec.format = ResponseFormat::parse(req.response_format.as_ref())?;
    spec.reasoning_effort = req.reasoning_effort.clone();
    spec.temperature = req.temperature;
    spec.top_p = req.top_p;
    spec.top_k = req.top_k;
    spec.min_p = req.min_p;
    spec.repeat_penalty = req.repeat_penalty;
    spec.max_tokens = req.max_tokens;
    spec.seed = req.seed;
    spec.stop = chat::stop_strings(req.stop.as_ref());
    spec.logit_bias = chat::logit_bias(req.logit_bias.as_ref())?;
    spec.chat_template_kwargs = req.chat_template_kwargs.clone();
    spec.id_slot = req.id_slot;
    Ok(spec)
}

/// Starts a generation for a spec and hands back its piece channel with the prepared prompt facts
pub fn start(s: &Served, spec: &ChatSpec) -> Result<(mpsc::Receiver<Piece>, CancelOnDrop, bool, usize)> {
    let prepared = chat::prepare(s, spec)?;
    let thinking_open = prepared.thinking_open;
    let n_prompt = prepared.n_prompt;
    let cancel = Arc::new(AtomicBool::new(false));
    let guard = CancelOnDrop(cancel.clone());
    let (tx, rx) = mpsc::channel::<Piece>(64);
    s.engine.send(Cmd::Generate(prepared.job(spec.id_slot, cancel, tx)))?;
    Ok((rx, guard, thinking_open, n_prompt))
}

async fn chat(State(app): State<Arc<App>>, Json(req): Json<ChatRequest>) -> Result<Response, Failure> {
    let Some(s) = ready(&app) else { return Ok(not_ready(&app)) };
    let spec = chat_spec(&req)?;
    let id = format!("chatcmpl-{}", s.metrics.requests.fetch_add(1, Ordering::Relaxed) + 1);
    let (rx, guard, thinking_open, _) = start(&s, &spec)?;
    if req.stream {
        let parser = StreamParser::new(s.format, thinking_open);
        let stream = stream_events(rx, parser, id, s.model_name.clone(), guard);
        return Ok(Sse::new(stream).keep_alive(KeepAlive::default()).into_response());
    }
    let out = collect(rx).await?;
    drop(guard);
    let parsed = parse_output(&out.text, s.format, thinking_open)?;
    let finish = if parsed.tool_calls.is_empty() { out.finish } else { "tool_calls" };
    let tool_calls: Vec<Value> = parsed.tool_calls.iter().map(|c| json!({ "id": c.id, "type": "function", "function": { "name": c.name, "arguments": c.arguments } })).collect();
    Ok(Json(json!({
        "id": id,
        "object": "chat.completion",
        "created": now_secs(),
        "model": s.model_name,
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": parsed.content, "reasoning_content": parsed.reasoning, "tool_calls": if tool_calls.is_empty() { Value::Null } else { Value::Array(tool_calls) } },
            "finish_reason": finish
        }],
        "usage": usage(&out)
    }))
    .into_response())
}

pub fn now_secs() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

/// Every piece of a generation gathered into its outcome
pub async fn collect(mut rx: mpsc::Receiver<Piece>) -> Result<Outcome> {
    while let Some(p) = rx.recv().await {
        match p {
            Piece::Text(_) => {}
            Piece::Done(o) => return Ok(o),
            Piece::Failed(m) => return Err(anyhow!("{m}")),
        }
    }
    Err(anyhow!("the generation ended without an outcome"))
}

pub fn usage(o: &Outcome) -> Value {
    json!({ "prompt_tokens": o.n_prompt, "completion_tokens": o.n_out, "total_tokens": o.n_prompt + o.n_out, "prompt_tokens_details": { "cached_tokens": o.cached } })
}

struct StreamState {
    rx: mpsc::Receiver<Piece>,
    parser: StreamParser,
    id: String,
    model: String,
    saw_tool: bool,
    done: bool,
    pending: std::collections::VecDeque<Event>,
    _guard: CancelOnDrop,
}

fn chunk(id: &str, model: &str, delta: Value, finish: Option<&str>, usage: Option<Value>) -> Event {
    let mut body = json!({ "id": id, "object": "chat.completion.chunk", "model": model, "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }] });
    if let Some(u) = usage {
        body["usage"] = u;
    }
    Event::default().data(body.to_string())
}

fn delta_event(st: &mut StreamState, d: Delta) -> Event {
    match d {
        Delta::Reasoning(r) => chunk(&st.id, &st.model, json!({ "reasoning_content": r }), None, None),
        Delta::Content(c) => chunk(&st.id, &st.model, json!({ "content": c }), None, None),
        Delta::ToolCall(c) => {
            st.saw_tool = true;
            let idx = c.id.trim_start_matches("call_").parse::<usize>().unwrap_or(0);
            chunk(&st.id, &st.model, json!({ "tool_calls": [{ "index": idx, "id": c.id, "type": "function", "function": { "name": c.name, "arguments": c.arguments } }] }), None, None)
        }
    }
}

/// Server sent events of parsed deltas, a finishing chunk with the usage, then the done marker
fn stream_events(rx: mpsc::Receiver<Piece>, parser: StreamParser, id: String, model: String, guard: CancelOnDrop) -> impl Stream<Item = Result<Event, std::convert::Infallible>> {
    let st = StreamState { rx, parser, id, model, saw_tool: false, done: false, pending: Default::default(), _guard: guard };
    futures_util::stream::unfold(st, |mut st| async move {
        loop {
            if let Some(e) = st.pending.pop_front() {
                return Some((Ok(e), st));
            }
            if st.done {
                return None;
            }
            match st.rx.recv().await {
                Some(Piece::Text(piece)) => match st.parser.feed(&piece) {
                    Ok(deltas) => {
                        for d in deltas {
                            let e = delta_event(&mut st, d);
                            st.pending.push_back(e);
                        }
                    }
                    Err(err) => {
                        st.pending.push_back(Event::default().data(json!({ "error": { "message": err.to_string() } }).to_string()));
                        st.done = true;
                    }
                },
                Some(Piece::Failed(msg)) => {
                    st.pending.push_back(Event::default().data(json!({ "error": { "message": msg } }).to_string()));
                    st.done = true;
                }
                last => {
                    let out = match last {
                        Some(Piece::Done(o)) => Some(o),
                        _ => None,
                    };
                    if let Ok(deltas) = st.parser.finish() {
                        for d in deltas {
                            let e = delta_event(&mut st, d);
                            st.pending.push_back(e);
                        }
                    }
                    let finish = if st.saw_tool { "tool_calls" } else { out.as_ref().map_or("stop", |o| o.finish) };
                    st.pending.push_back(chunk(&st.id, &st.model, json!({}), Some(finish), out.as_ref().map(usage)));
                    st.pending.push_back(Event::default().data("[DONE]"));
                    st.done = true;
                }
            }
        }
    })
}

//! The Anthropic messages API over the same engine, content blocks in and content blocks out
use crate::api::{App, CancelOnDrop, Failure, collect, not_ready, now_secs, ready, start};
use crate::chat::{ChatSpec, ToolChoice};
use crate::engine::{Outcome, Piece};
use crate::parse::{Delta, StreamParser, parse_output};
use crate::template::Message;
use anyhow::{Result, anyhow, bail};
use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Deserialize;
use serde_json::{Value, json};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use tokio::sync::mpsc;

#[derive(Deserialize)]
pub struct MessagesRequest {
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub system: Option<Value>,
    pub messages: Vec<Value>,
    #[serde(default)]
    pub tools: Option<Vec<Value>>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub stop_sequences: Vec<String>,
    #[serde(default)]
    pub thinking: Option<Value>,
}

/// Text of a string or of the text blocks of an array
fn block_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Array(a) => a.iter().filter(|b| b.get("type").and_then(Value::as_str) == Some("text")).filter_map(|b| b.get("text").and_then(Value::as_str)).collect::<Vec<_>>().join("\n"),
        _ => String::new(),
    }
}

/// An Anthropic image block as the OpenAI image part the prompt builder reads
fn image_part(b: &Value) -> Result<Value> {
    let src = b.get("source").ok_or_else(|| anyhow!("image block without a source"))?;
    let url = match src.get("type").and_then(Value::as_str) {
        Some("base64") => {
            let media = src.get("media_type").and_then(Value::as_str).unwrap_or("image/png");
            let data = src.get("data").and_then(Value::as_str).ok_or_else(|| anyhow!("base64 image without data"))?;
            format!("data:{media};base64,{data}")
        }
        Some("url") => src.get("url").and_then(Value::as_str).ok_or_else(|| anyhow!("url image without a url"))?.to_string(),
        other => bail!("image source type {other:?}, base64 or url expected"),
    };
    Ok(json!({ "type": "image_url", "image_url": { "url": url } }))
}

/// Anthropic messages as the chat messages the template renders, tool results as tool messages and tool uses as tool calls
fn convert_messages(system: Option<&Value>, messages: &[Value]) -> Result<Vec<Message>> {
    let mut out = Vec::new();
    if let Some(sys) = system {
        let text = block_text(sys);
        if !text.is_empty() {
            out.push(Message { role: "system".into(), content: Value::String(text), reasoning_content: None, tool_calls: None, tool_call_id: None, name: None });
        }
    }
    for m in messages {
        let role = m.get("role").and_then(Value::as_str).ok_or_else(|| anyhow!("message without a role"))?;
        let content = m.get("content").cloned().unwrap_or(Value::Null);
        match role {
            "user" => {
                let mut parts: Vec<Value> = Vec::new();
                let mut text = String::new();
                let mut tool_results = Vec::new();
                match &content {
                    Value::String(s) => text = s.clone(),
                    Value::Array(blocks) => {
                        for b in blocks {
                            match b.get("type").and_then(Value::as_str) {
                                Some("text") => {
                                    let t = b.get("text").and_then(Value::as_str).unwrap_or("");
                                    if !text.is_empty() {
                                        text.push('\n');
                                    }
                                    text.push_str(t);
                                    parts.push(json!({ "type": "text", "text": t }));
                                }
                                Some("image") => parts.push(image_part(b)?),
                                Some("tool_result") => {
                                    let id = b.get("tool_use_id").and_then(Value::as_str).unwrap_or("").to_string();
                                    let body = b.get("content").map(block_text).unwrap_or_default();
                                    tool_results.push(Message { role: "tool".into(), content: Value::String(body), reasoning_content: None, tool_calls: None, tool_call_id: Some(id), name: None });
                                }
                                other => bail!("user content block type {other:?} is not text, image, or tool_result"),
                            }
                        }
                    }
                    _ => {}
                }
                out.extend(tool_results);
                let has_image = parts.iter().any(|p| p.get("type").and_then(Value::as_str) == Some("image_url"));
                if has_image {
                    out.push(Message { role: "user".into(), content: Value::Array(parts), reasoning_content: None, tool_calls: None, tool_call_id: None, name: None });
                } else if !text.is_empty() || content.is_string() {
                    out.push(Message { role: "user".into(), content: Value::String(text), reasoning_content: None, tool_calls: None, tool_call_id: None, name: None });
                }
            }
            "assistant" => {
                let mut text = String::new();
                let mut reasoning = String::new();
                let mut calls = Vec::new();
                match &content {
                    Value::String(s) => text = s.clone(),
                    Value::Array(blocks) => {
                        for b in blocks {
                            match b.get("type").and_then(Value::as_str) {
                                Some("text") => text.push_str(b.get("text").and_then(Value::as_str).unwrap_or("")),
                                Some("thinking") => reasoning.push_str(b.get("thinking").and_then(Value::as_str).unwrap_or("")),
                                Some("redacted_thinking") => {}
                                Some("tool_use") => {
                                    let id = b.get("id").and_then(Value::as_str).unwrap_or("").to_string();
                                    let name = b.get("name").and_then(Value::as_str).unwrap_or("").to_string();
                                    let input = b.get("input").cloned().unwrap_or(json!({}));
                                    calls.push(json!({ "id": id, "type": "function", "function": { "name": name, "arguments": input.to_string() } }));
                                }
                                other => bail!("assistant content block type {other:?} is not text, thinking, or tool_use"),
                            }
                        }
                    }
                    _ => {}
                }
                out.push(Message {
                    role: "assistant".into(),
                    content: Value::String(text),
                    reasoning_content: (!reasoning.is_empty()).then_some(reasoning),
                    tool_calls: (!calls.is_empty()).then_some(Value::Array(calls)),
                    tool_call_id: None,
                    name: None,
                });
            }
            other => bail!("message role {other:?}, user or assistant expected"),
        }
    }
    Ok(out)
}

/// Anthropic tools as the OpenAI function tools the template renders
fn convert_tools(tools: &[Value]) -> Result<Vec<Value>> {
    tools
        .iter()
        .map(|t| {
            let name = t.get("name").and_then(Value::as_str).ok_or_else(|| anyhow!("a tool without a name"))?;
            let params = t.get("input_schema").cloned().unwrap_or_else(|| json!({ "type": "object" }));
            let mut f = json!({ "name": name, "parameters": params });
            if let Some(d) = t.get("description") {
                f["description"] = d.clone();
            }
            Ok(json!({ "type": "function", "function": f }))
        })
        .collect()
}

fn convert_tool_choice(v: Option<&Value>) -> Result<ToolChoice> {
    let Some(v) = v else { return Ok(ToolChoice::Auto) };
    Ok(match v.get("type").and_then(Value::as_str) {
        None | Some("auto") => ToolChoice::Auto,
        Some("any") => ToolChoice::Required,
        Some("none") => ToolChoice::None,
        Some("tool") => ToolChoice::Named(v.get("name").and_then(Value::as_str).ok_or_else(|| anyhow!("tool_choice of type tool without a name"))?.to_string()),
        Some(other) => bail!("tool_choice type {other:?}, auto, any, tool, or none expected"),
    })
}

fn spec_of(req: &MessagesRequest) -> Result<ChatSpec> {
    let mut spec = ChatSpec::new(convert_messages(req.system.as_ref(), &req.messages)?);
    spec.tools = req.tools.as_ref().map(|t| convert_tools(t)).transpose()?;
    spec.tool_choice = convert_tool_choice(req.tool_choice.as_ref())?;
    spec.temperature = req.temperature;
    spec.top_p = req.top_p;
    spec.top_k = req.top_k;
    spec.max_tokens = req.max_tokens;
    spec.stop = req.stop_sequences.clone();
    if let Some(t) = &req.thinking {
        spec.think_budget = Some(match t.get("type").and_then(Value::as_str) {
            Some("enabled") => t.get("budget_tokens").and_then(Value::as_u64).map(|b| b as usize),
            Some("disabled") => Some(0),
            other => bail!("thinking type {other:?}, enabled or disabled expected"),
        });
    }
    Ok(spec)
}

fn stop_reason(o: &Outcome, tool_calls: bool) -> (&'static str, Option<String>) {
    if tool_calls {
        return ("tool_use", None);
    }
    match o.finish {
        "length" => ("max_tokens", None),
        "stop" if o.stop_sequence.is_some() => ("stop_sequence", o.stop_sequence.clone()),
        _ => ("end_turn", None),
    }
}

fn usage(o: &Outcome) -> Value {
    json!({ "input_tokens": o.n_prompt - o.cached, "cache_read_input_tokens": o.cached, "cache_creation_input_tokens": 0, "output_tokens": o.n_out })
}

fn tool_input(arguments: &str) -> Value {
    serde_json::from_str(arguments).unwrap_or_else(|_| Value::String(arguments.to_string()))
}

pub async fn messages(State(app): State<Arc<App>>, Json(req): Json<MessagesRequest>) -> Result<Response, Failure> {
    let Some(s) = ready(&app) else { return Ok(not_ready(&app)) };
    let spec = spec_of(&req)?;
    let id = format!("msg_{}", s.metrics.requests.fetch_add(1, Ordering::Relaxed) + 1);
    let (rx, guard, thinking_open, n_prompt) = start(&s, &spec)?;
    if req.stream {
        let parser = StreamParser::new(s.format, thinking_open);
        let stream = stream_events(rx, parser, id, s.model_name.clone(), n_prompt, guard);
        return Ok(Sse::new(stream).keep_alive(KeepAlive::default()).into_response());
    }
    let out = collect(rx).await?;
    drop(guard);
    let parsed = parse_output(&out.text, s.format, thinking_open)?;
    let mut content = Vec::new();
    if let Some(r) = &parsed.reasoning {
        content.push(json!({ "type": "thinking", "thinking": r, "signature": "" }));
    }
    if !parsed.content.is_empty() || (parsed.tool_calls.is_empty() && parsed.reasoning.is_none()) {
        content.push(json!({ "type": "text", "text": parsed.content }));
    }
    for c in &parsed.tool_calls {
        content.push(json!({ "type": "tool_use", "id": format!("toolu_{}_{}", id, c.id), "name": c.name, "input": tool_input(&c.arguments) }));
    }
    let (reason, seq) = stop_reason(&out, !parsed.tool_calls.is_empty());
    Ok(Json(json!({
        "id": id,
        "type": "message",
        "role": "assistant",
        "model": s.model_name,
        "content": content,
        "stop_reason": reason,
        "stop_sequence": seq,
        "usage": usage(&out)
    }))
    .into_response())
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Block {
    None,
    Thinking,
    Text,
}

struct StreamState {
    rx: mpsc::Receiver<Piece>,
    parser: StreamParser,
    id: String,
    model: String,
    n_prompt: usize,
    index: usize,
    block: Block,
    saw_tool: bool,
    started: bool,
    done: bool,
    pending: std::collections::VecDeque<Event>,
    _guard: CancelOnDrop,
}

fn ev(name: &str, body: Value) -> Event {
    let mut body = body;
    body["type"] = Value::String(name.to_string());
    Event::default().event(name).data(body.to_string())
}

impl StreamState {
    fn close_block(&mut self) {
        if self.block != Block::None {
            self.pending.push_back(ev("content_block_stop", json!({ "index": self.index })));
            self.index += 1;
            self.block = Block::None;
        }
    }

    fn open(&mut self, kind: Block) {
        if self.block == kind {
            return;
        }
        self.close_block();
        let block = match kind {
            Block::Thinking => json!({ "type": "thinking", "thinking": "", "signature": "" }),
            _ => json!({ "type": "text", "text": "" }),
        };
        self.pending.push_back(ev("content_block_start", json!({ "index": self.index, "content_block": block })));
        self.block = kind;
    }

    fn delta(&mut self, d: Delta) {
        match d {
            Delta::Reasoning(r) => {
                self.open(Block::Thinking);
                self.pending.push_back(ev("content_block_delta", json!({ "index": self.index, "delta": { "type": "thinking_delta", "thinking": r } })));
            }
            Delta::Content(c) => {
                self.open(Block::Text);
                self.pending.push_back(ev("content_block_delta", json!({ "index": self.index, "delta": { "type": "text_delta", "text": c } })));
            }
            Delta::ToolCall(c) => {
                self.close_block();
                self.saw_tool = true;
                let tid = format!("toolu_{}_{}", self.id, c.id);
                self.pending.push_back(ev("content_block_start", json!({ "index": self.index, "content_block": { "type": "tool_use", "id": tid, "name": c.name, "input": {} } })));
                self.pending.push_back(ev("content_block_delta", json!({ "index": self.index, "delta": { "type": "input_json_delta", "partial_json": c.arguments } })));
                self.pending.push_back(ev("content_block_stop", json!({ "index": self.index })));
                self.index += 1;
            }
        }
    }
}

/// Server sent events of the Anthropic stream, message start, one block per piece kind, the delta with the stop reason, then message stop
fn stream_events(rx: mpsc::Receiver<Piece>, parser: StreamParser, id: String, model: String, n_prompt: usize, guard: CancelOnDrop) -> impl futures_util::stream::Stream<Item = Result<Event, std::convert::Infallible>> {
    let st = StreamState { rx, parser, id, model, n_prompt, index: 0, block: Block::None, saw_tool: false, started: false, done: false, pending: Default::default(), _guard: guard };
    futures_util::stream::unfold(st, |mut st| async move {
        loop {
            if !st.started {
                st.started = true;
                let message = json!({ "id": st.id, "type": "message", "role": "assistant", "model": st.model, "content": [], "stop_reason": Value::Null, "stop_sequence": Value::Null, "usage": { "input_tokens": st.n_prompt, "output_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0 } });
                st.pending.push_back(ev("message_start", json!({ "message": message })));
            }
            if let Some(e) = st.pending.pop_front() {
                return Some((Ok(e), st));
            }
            if st.done {
                return None;
            }
            match st.rx.recv().await {
                Some(Piece::Text(piece, _)) => match st.parser.feed(&piece) {
                    Ok(deltas) => {
                        for d in deltas {
                            st.delta(d);
                        }
                    }
                    Err(err) => {
                        st.pending.push_back(ev("error", json!({ "error": { "type": "api_error", "message": err.to_string() } })));
                        st.done = true;
                    }
                },
                Some(Piece::Failed(msg)) => {
                    st.pending.push_back(ev("error", json!({ "error": { "type": "api_error", "message": msg } })));
                    st.done = true;
                }
                last => {
                    let out = match last {
                        Some(Piece::Done(o)) => Some(o),
                        _ => None,
                    };
                    if let Ok(deltas) = st.parser.finish() {
                        for d in deltas {
                            st.delta(d);
                        }
                    }
                    if st.index == 0 && st.block == Block::None {
                        st.open(Block::Text);
                    }
                    st.close_block();
                    let (reason, seq) = out.as_ref().map_or(("end_turn", None), |o| stop_reason(o, st.saw_tool));
                    let usage = out.as_ref().map(usage).unwrap_or_else(|| json!({ "output_tokens": 0 }));
                    st.pending.push_back(ev("message_delta", json!({ "delta": { "stop_reason": reason, "stop_sequence": seq }, "usage": usage })));
                    st.pending.push_back(ev("message_stop", json!({})));
                    st.done = true;
                }
            }
        }
    })
}

/// Seconds since the epoch, the created stamp of every object this API returns
pub fn created() -> u64 {
    now_secs()
}

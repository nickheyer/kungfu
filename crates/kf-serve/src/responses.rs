//! The OpenAI responses API over the same engine, input items in and output items out
use crate::api::{App, CancelOnDrop, Failure, collect, not_ready, now_secs, ready, start};
use crate::chat::{ChatSpec, ResponseFormat, ToolChoice};
use crate::engine::{Outcome, Piece};
use crate::parse::{Delta, StreamParser, parse_output};
use crate::template::Message;
use crate::tokenizer::Tok;
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
pub struct ResponsesRequest {
    pub input: Value,
    #[serde(default)]
    pub instructions: Option<String>,
    #[serde(default)]
    pub tools: Option<Vec<Value>>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
    #[serde(default)]
    pub max_output_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub reasoning: Option<Value>,
    #[serde(default)]
    pub text: Option<Value>,
    #[serde(default)]
    pub previous_response_id: Option<String>,
}

/// Content of an input message as the template takes it, text joined and images kept as parts
fn message_content(v: &Value) -> Result<Value> {
    match v {
        Value::String(s) => Ok(Value::String(s.clone())),
        Value::Array(parts) => {
            let mut text = String::new();
            let mut out = Vec::new();
            let mut images = false;
            for p in parts {
                match p.get("type").and_then(Value::as_str) {
                    Some("input_text") | Some("output_text") | Some("text") => {
                        let t = p.get("text").and_then(Value::as_str).unwrap_or("");
                        if !text.is_empty() {
                            text.push('\n');
                        }
                        text.push_str(t);
                        out.push(json!({ "type": "text", "text": t }));
                    }
                    Some("input_image") => {
                        images = true;
                        let url = p.get("image_url").and_then(Value::as_str).ok_or_else(|| anyhow!("input_image without an image_url"))?;
                        out.push(json!({ "type": "image_url", "image_url": { "url": url } }));
                    }
                    other => bail!("input content type {other:?} is not input_text, output_text, or input_image"),
                }
            }
            Ok(if images { Value::Array(out) } else { Value::String(text) })
        }
        Value::Null => Ok(Value::String(String::new())),
        other => bail!("message content {other} is not text or parts"),
    }
}

fn text_message(role: &str, content: Value) -> Message {
    Message { role: role.to_string(), content, reasoning_content: None, tool_calls: None, tool_call_id: None, name: None }
}

/// Input items as chat messages, consecutive function calls gathered into one assistant turn
fn convert_input(instructions: Option<&str>, input: &Value) -> Result<Vec<Message>> {
    let mut out = Vec::new();
    if let Some(i) = instructions {
        out.push(text_message("system", Value::String(i.to_string())));
    }
    let items: Vec<Value> = match input {
        Value::String(s) => vec![json!({ "role": "user", "content": s })],
        Value::Array(a) => a.clone(),
        other => bail!("input {other} is not a string or an array of items"),
    };
    let mut calls: Vec<Value> = Vec::new();
    let flush = |calls: &mut Vec<Value>, out: &mut Vec<Message>| {
        if !calls.is_empty() {
            out.push(Message { role: "assistant".into(), content: Value::String(String::new()), reasoning_content: None, tool_calls: Some(Value::Array(std::mem::take(calls))), tool_call_id: None, name: None });
        }
    };
    for item in &items {
        let ty = item.get("type").and_then(Value::as_str).unwrap_or("message");
        match ty {
            "message" => {
                flush(&mut calls, &mut out);
                let role = item.get("role").and_then(Value::as_str).ok_or_else(|| anyhow!("message item without a role"))?;
                let role = if role == "developer" { "system" } else { role };
                out.push(text_message(role, message_content(item.get("content").unwrap_or(&Value::Null))?));
            }
            "function_call" => {
                let id = item.get("call_id").and_then(Value::as_str).unwrap_or("").to_string();
                let name = item.get("name").and_then(Value::as_str).unwrap_or("").to_string();
                let args = match item.get("arguments") {
                    Some(Value::String(s)) => s.clone(),
                    Some(v) => v.to_string(),
                    None => "{}".to_string(),
                };
                calls.push(json!({ "id": id, "type": "function", "function": { "name": name, "arguments": args } }));
            }
            "function_call_output" => {
                flush(&mut calls, &mut out);
                let id = item.get("call_id").and_then(Value::as_str).unwrap_or("").to_string();
                let body = match item.get("output") {
                    Some(Value::String(s)) => s.clone(),
                    Some(v) => v.to_string(),
                    None => String::new(),
                };
                out.push(Message { role: "tool".into(), content: Value::String(body), reasoning_content: None, tool_calls: None, tool_call_id: Some(id), name: None });
            }
            "reasoning" => {}
            other => bail!("input item type {other:?} is not message, function_call, function_call_output, or reasoning"),
        }
    }
    flush(&mut calls, &mut out);
    Ok(out)
}

/// Responses tools, flat function objects, as the nested OpenAI chat tools
fn convert_tools(tools: &[Value]) -> Result<Vec<Value>> {
    tools
        .iter()
        .map(|t| {
            if t.get("type").and_then(Value::as_str) != Some("function") {
                bail!("tool type {} is not function", t.get("type").cloned().unwrap_or(Value::Null));
            }
            if t.get("function").is_some() {
                return Ok(t.clone());
            }
            let name = t.get("name").and_then(Value::as_str).ok_or_else(|| anyhow!("a tool without a name"))?;
            let mut f = json!({ "name": name, "parameters": t.get("parameters").cloned().unwrap_or_else(|| json!({ "type": "object" })) });
            if let Some(d) = t.get("description") {
                f["description"] = d.clone();
            }
            Ok(json!({ "type": "function", "function": f }))
        })
        .collect()
}

fn spec_of(req: &ResponsesRequest) -> Result<ChatSpec> {
    if req.previous_response_id.is_some() {
        bail!("previous_response_id is not kept by this server, send the whole conversation as input");
    }
    let mut spec = ChatSpec::new(convert_input(req.instructions.as_deref(), &req.input)?);
    spec.tools = req.tools.as_ref().map(|t| convert_tools(t)).transpose()?;
    spec.tool_choice = ToolChoice::parse(req.tool_choice.as_ref())?;
    spec.temperature = req.temperature;
    spec.top_p = req.top_p;
    spec.max_tokens = req.max_output_tokens;
    spec.reasoning_effort = req.reasoning.as_ref().and_then(|r| r.get("effort")).and_then(Value::as_str).map(str::to_string);
    if let Some(f) = req.text.as_ref().and_then(|t| t.get("format")) {
        spec.format = match f.get("type").and_then(Value::as_str) {
            None | Some("text") => ResponseFormat::Text,
            Some("json_object") => ResponseFormat::JsonObject,
            Some("json_schema") => ResponseFormat::JsonSchema(f.get("schema").cloned().ok_or_else(|| anyhow!("text format json_schema without a schema"))?),
            Some(other) => bail!("text format type {other:?}, text, json_object, or json_schema expected"),
        };
    }
    Ok(spec)
}

fn usage(o: &Outcome, reasoning_tokens: usize) -> Value {
    json!({ "input_tokens": o.n_prompt, "input_tokens_details": { "cached_tokens": o.cached }, "output_tokens": o.n_out, "output_tokens_details": { "reasoning_tokens": reasoning_tokens }, "total_tokens": o.n_prompt + o.n_out })
}

/// The response envelope around output items
fn envelope(id: &str, model: &str, req: &ResponsesRequest, status: &str, output: Vec<Value>, usage: Option<Value>, incomplete: Option<&str>) -> Value {
    json!({
        "id": id,
        "object": "response",
        "created_at": now_secs(),
        "status": status,
        "error": Value::Null,
        "incomplete_details": incomplete.map(|r| json!({ "reason": r })),
        "instructions": req.instructions,
        "model": model,
        "output": output,
        "parallel_tool_calls": false,
        "previous_response_id": Value::Null,
        "reasoning": req.reasoning.clone().unwrap_or(json!({ "effort": Value::Null })),
        "store": false,
        "temperature": req.temperature,
        "text": req.text.clone().unwrap_or(json!({ "format": { "type": "text" } })),
        "tool_choice": req.tool_choice.clone().unwrap_or(json!("auto")),
        "tools": req.tools.clone().unwrap_or_default(),
        "top_p": req.top_p,
        "truncation": "disabled",
        "usage": usage,
        "metadata": {}
    })
}

fn reasoning_tokens(tok: &Tok, reasoning: Option<&str>) -> usize {
    reasoning.map_or(0, |r| tok.encode(r, false).map_or(0, |v| v.len()))
}

pub async fn responses(State(app): State<Arc<App>>, Json(req): Json<ResponsesRequest>) -> Result<Response, Failure> {
    let Some(s) = ready(&app) else { return Ok(not_ready(&app)) };
    let spec = spec_of(&req)?;
    let id = format!("resp_{}", s.metrics.requests.fetch_add(1, Ordering::Relaxed) + 1);
    let (rx, guard, thinking_open, _) = start(&s, &spec)?;
    if req.stream {
        let parser = StreamParser::new(s.format, thinking_open);
        let stream = stream_events(rx, parser, id, s.model_name.clone(), req, s.tok.clone(), guard);
        return Ok(Sse::new(stream).keep_alive(KeepAlive::default()).into_response());
    }
    let out = collect(rx).await?;
    drop(guard);
    let parsed = parse_output(&out.text, s.format, thinking_open)?;
    let mut output = Vec::new();
    if let Some(r) = &parsed.reasoning {
        output.push(json!({ "type": "reasoning", "id": format!("rs_{id}"), "summary": [{ "type": "summary_text", "text": r }] }));
    }
    if !parsed.content.is_empty() || parsed.tool_calls.is_empty() {
        output.push(json!({ "type": "message", "id": format!("msg_{id}"), "status": "completed", "role": "assistant", "content": [{ "type": "output_text", "text": parsed.content, "annotations": [] }] }));
    }
    for c in &parsed.tool_calls {
        output.push(json!({ "type": "function_call", "id": format!("fc_{id}_{}", c.id), "call_id": format!("{}_{}", id, c.id), "name": c.name, "arguments": c.arguments, "status": "completed" }));
    }
    let incomplete = (out.finish == "length").then_some("max_output_tokens");
    let status = if incomplete.is_some() { "incomplete" } else { "completed" };
    let u = usage(&out, reasoning_tokens(&s.tok, parsed.reasoning.as_deref()));
    Ok(Json(envelope(&id, &s.model_name, &req, status, output, Some(u), incomplete)).into_response())
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Item {
    None,
    Reasoning,
    Message,
}

struct StreamState {
    rx: mpsc::Receiver<Piece>,
    parser: StreamParser,
    id: String,
    model: String,
    req: ResponsesRequest,
    tok: Arc<Tok>,
    seq: u64,
    index: usize,
    item: Item,
    text: String,
    reasoning: String,
    output: Vec<Value>,
    started: bool,
    done: bool,
    pending: std::collections::VecDeque<Event>,
    _guard: CancelOnDrop,
}

impl StreamState {
    fn ev(&mut self, name: &str, mut body: Value) {
        body["type"] = Value::String(name.to_string());
        body["sequence_number"] = Value::from(self.seq);
        self.seq += 1;
        self.pending.push_back(Event::default().event(name).data(body.to_string()));
    }

    fn close_item(&mut self) {
        match self.item {
            Item::None => return,
            Item::Reasoning => {
                let item_id = format!("rs_{}", self.id);
                let text = std::mem::take(&mut self.reasoning);
                self.ev("response.reasoning_summary_text.done", json!({ "item_id": item_id, "output_index": self.index, "summary_index": 0, "text": text }));
                self.ev("response.reasoning_summary_part.done", json!({ "item_id": item_id, "output_index": self.index, "summary_index": 0, "part": { "type": "summary_text", "text": text } }));
                let item = json!({ "type": "reasoning", "id": item_id, "summary": [{ "type": "summary_text", "text": text }] });
                self.ev("response.output_item.done", json!({ "output_index": self.index, "item": item }));
                self.output.push(item);
            }
            Item::Message => {
                let item_id = format!("msg_{}", self.id);
                let text = std::mem::take(&mut self.text);
                self.ev("response.output_text.done", json!({ "item_id": item_id, "output_index": self.index, "content_index": 0, "text": text }));
                self.ev("response.content_part.done", json!({ "item_id": item_id, "output_index": self.index, "content_index": 0, "part": { "type": "output_text", "text": text, "annotations": [] } }));
                let item = json!({ "type": "message", "id": item_id, "status": "completed", "role": "assistant", "content": [{ "type": "output_text", "text": text, "annotations": [] }] });
                self.ev("response.output_item.done", json!({ "output_index": self.index, "item": item }));
                self.output.push(item);
            }
        }
        self.index += 1;
        self.item = Item::None;
    }

    fn open(&mut self, kind: Item) {
        if self.item == kind {
            return;
        }
        self.close_item();
        match kind {
            Item::Reasoning => {
                let item_id = format!("rs_{}", self.id);
                self.ev("response.output_item.added", json!({ "output_index": self.index, "item": { "type": "reasoning", "id": item_id, "summary": [] } }));
                self.ev("response.reasoning_summary_part.added", json!({ "item_id": item_id, "output_index": self.index, "summary_index": 0, "part": { "type": "summary_text", "text": "" } }));
            }
            Item::Message => {
                let item_id = format!("msg_{}", self.id);
                self.ev("response.output_item.added", json!({ "output_index": self.index, "item": { "type": "message", "id": item_id, "status": "in_progress", "role": "assistant", "content": [] } }));
                self.ev("response.content_part.added", json!({ "item_id": item_id, "output_index": self.index, "content_index": 0, "part": { "type": "output_text", "text": "", "annotations": [] } }));
            }
            Item::None => {}
        }
        self.item = kind;
    }

    fn delta(&mut self, d: Delta) {
        match d {
            Delta::Reasoning(r) => {
                self.open(Item::Reasoning);
                self.reasoning.push_str(&r);
                self.ev("response.reasoning_summary_text.delta", json!({ "item_id": format!("rs_{}", self.id), "output_index": self.index, "summary_index": 0, "delta": r }));
            }
            Delta::Content(c) => {
                self.open(Item::Message);
                self.text.push_str(&c);
                self.ev("response.output_text.delta", json!({ "item_id": format!("msg_{}", self.id), "output_index": self.index, "content_index": 0, "delta": c }));
            }
            Delta::ToolCall(c) => {
                self.close_item();
                let item_id = format!("fc_{}_{}", self.id, c.id);
                let call_id = format!("{}_{}", self.id, c.id);
                self.ev("response.output_item.added", json!({ "output_index": self.index, "item": { "type": "function_call", "id": item_id, "call_id": call_id, "name": c.name, "arguments": "", "status": "in_progress" } }));
                self.ev("response.function_call_arguments.delta", json!({ "item_id": item_id, "output_index": self.index, "delta": c.arguments }));
                self.ev("response.function_call_arguments.done", json!({ "item_id": item_id, "output_index": self.index, "arguments": c.arguments }));
                let item = json!({ "type": "function_call", "id": item_id, "call_id": call_id, "name": c.name, "arguments": c.arguments, "status": "completed" });
                self.ev("response.output_item.done", json!({ "output_index": self.index, "item": item }));
                self.output.push(item);
                self.index += 1;
            }
        }
    }
}

/// Server sent events of the responses stream, created and in progress, one item per piece kind, then completed with the whole response
fn stream_events(rx: mpsc::Receiver<Piece>, parser: StreamParser, id: String, model: String, req: ResponsesRequest, tok: Arc<Tok>, guard: CancelOnDrop) -> impl futures_util::stream::Stream<Item = Result<Event, std::convert::Infallible>> {
    let st = StreamState { rx, parser, id, model, req, tok, seq: 0, index: 0, item: Item::None, text: String::new(), reasoning: String::new(), output: Vec::new(), started: false, done: false, pending: Default::default(), _guard: guard };
    futures_util::stream::unfold(st, |mut st| async move {
        loop {
            if !st.started {
                st.started = true;
                let env = envelope(&st.id, &st.model, &st.req, "in_progress", Vec::new(), None, None);
                st.ev("response.created", json!({ "response": env }));
                st.ev("response.in_progress", json!({ "response": env }));
            }
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
                            st.delta(d);
                        }
                    }
                    Err(err) => {
                        st.ev("error", json!({ "code": "server_error", "message": err.to_string(), "param": Value::Null }));
                        st.done = true;
                    }
                },
                Some(Piece::Failed(msg)) => {
                    st.ev("error", json!({ "code": "server_error", "message": msg, "param": Value::Null }));
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
                    if st.output.is_empty() && st.item == Item::None {
                        st.open(Item::Message);
                    }
                    st.close_item();
                    let reasoning: String = st.output.iter().filter(|o| o["type"] == "reasoning").filter_map(|o| o["summary"][0]["text"].as_str()).collect();
                    let rt = reasoning_tokens(&st.tok, (!reasoning.is_empty()).then_some(reasoning.as_str()));
                    let (status, incomplete, usage) = match &out {
                        Some(o) if o.finish == "length" => ("incomplete", Some("max_output_tokens"), Some(usage(o, rt))),
                        Some(o) => ("completed", None, Some(usage(o, rt))),
                        None => ("incomplete", Some("error"), None),
                    };
                    let env = envelope(&st.id, &st.model, &st.req, status, st.output.clone(), usage, incomplete);
                    let name = if status == "completed" { "response.completed" } else { "response.incomplete" };
                    st.ev(name, json!({ "response": env }));
                    st.done = true;
                }
            }
        }
    })
}

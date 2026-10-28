//! One request shape every API flavor maps onto, and the job the scheduler runs for it
use crate::api::Served;
use crate::engine::{Chunk, Job, Limits, Piece, THINK_CLOSE, THINK_OPEN, image_tokens};
use crate::grammar::{Constraint, GrammarBuilder, Node};
use crate::parse::Format;
use crate::template::Message;
use anyhow::{Result, anyhow, bail};
use kf_engine::Sampling;
use serde_json::{Map, Value};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use tokio::sync::mpsc;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ToolChoice {
    Auto,
    None,
    Required,
    Named(String),
}

impl ToolChoice {
    /// The OpenAI shape, a mode name or an object naming one function
    pub fn parse(v: Option<&Value>) -> Result<Self> {
        Ok(match v {
            None | Some(Value::Null) => Self::Auto,
            Some(Value::String(s)) => match s.as_str() {
                "auto" => Self::Auto,
                "none" => Self::None,
                "required" | "any" => Self::Required,
                other => bail!("tool_choice {other:?}, auto, none, required, or a function expected"),
            },
            Some(Value::Object(o)) => {
                let name = o.get("function").and_then(|f| f.get("name")).or_else(|| o.get("name")).and_then(Value::as_str).ok_or_else(|| anyhow!("tool_choice object without a function name"))?;
                Self::Named(name.to_string())
            }
            Some(other) => bail!("tool_choice {other} is not a mode or a function"),
        })
    }
}

/// The output format a request constrains the answer to
#[derive(Clone, Debug, PartialEq)]
pub enum ResponseFormat {
    Text,
    JsonObject,
    JsonSchema(Value),
}

impl ResponseFormat {
    /// The OpenAI response_format object
    pub fn parse(v: Option<&Value>) -> Result<Self> {
        let Some(v) = v else { return Ok(Self::Text) };
        if v.is_null() {
            return Ok(Self::Text);
        }
        match v.get("type").and_then(Value::as_str) {
            None | Some("text") => Ok(Self::Text),
            Some("json_object") => Ok(Self::JsonObject),
            Some("json_schema") => {
                let schema = v.get("json_schema").and_then(|j| j.get("schema")).or_else(|| v.get("schema")).ok_or_else(|| anyhow!("response_format json_schema without a schema"))?;
                Ok(Self::JsonSchema(schema.clone()))
            }
            Some(other) => bail!("response_format type {other:?}, text, json_object, or json_schema expected"),
        }
    }
}

/// Reasoning tokens an effort level allows before the think block is closed, none for no limit
pub fn reasoning_budget(effort: &str) -> Result<Option<usize>> {
    Ok(match effort {
        "none" => Some(0),
        "minimal" => Some(512),
        "low" => Some(1024),
        "medium" => Some(4096),
        "high" => None,
        other => bail!("reasoning_effort {other:?}, none, minimal, low, medium, or high expected"),
    })
}

/// The effort levels with their budgets as the self description lists them
pub const REASONING_EFFORTS: &[(&str, Option<usize>)] = &[("none", Some(0)), ("minimal", Some(512)), ("low", Some(1024)), ("medium", Some(4096)), ("high", None)];

/// Everything a generation asks for, whichever API shape it arrived in
#[derive(Clone, Debug)]
pub struct ChatSpec {
    pub messages: Vec<Message>,
    /// Tools in the OpenAI shape, type function with a function object
    pub tools: Option<Vec<Value>>,
    pub tool_choice: ToolChoice,
    pub format: ResponseFormat,
    pub reasoning_effort: Option<String>,
    /// A reasoning budget set directly, overriding the effort's
    pub think_budget: Option<Option<usize>>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub min_p: Option<f32>,
    pub repeat_penalty: Option<f32>,
    pub max_tokens: Option<usize>,
    pub seed: Option<u64>,
    pub stop: Vec<String>,
    pub logit_bias: Vec<(u32, f32)>,
    pub chat_template_kwargs: Map<String, Value>,
    pub id_slot: Option<usize>,
}

impl ChatSpec {
    pub fn new(messages: Vec<Message>) -> Self {
        Self {
            messages,
            tools: None,
            tool_choice: ToolChoice::Auto,
            format: ResponseFormat::Text,
            reasoning_effort: None,
            think_budget: None,
            temperature: None,
            top_p: None,
            top_k: None,
            min_p: None,
            repeat_penalty: None,
            max_tokens: None,
            seed: None,
            stop: Vec::new(),
            logit_bias: Vec::new(),
            chat_template_kwargs: Map::new(),
            id_slot: None,
        }
    }
}

/// The text an image part leaves in the prompt for the template, replaced by the image's rows after rendering
pub const MEDIA_MARKER: &str = "<__media__>";

/// Most bytes one image may carry
const IMAGE_LIMIT: usize = 32 << 20;

/// A rendered prompt with the rules of its generation, ready to become a job
pub struct Prepared {
    pub ids: Vec<u32>,
    pub chunks: Vec<Chunk>,
    pub prompt: String,
    pub thinking_open: bool,
    pub sampling: Sampling,
    pub limits: Limits,
    pub constraint: Option<Constraint>,
    pub think_budget: Option<usize>,
    pub n_prompt: usize,
}

impl Prepared {
    /// The job of this prepared request with the channel its pieces travel on
    pub fn job(self, slot: Option<usize>, cancel: Arc<AtomicBool>, tx: mpsc::Sender<Piece>) -> Job {
        Job { ids: self.ids, chunks: self.chunks, sampling: self.sampling, limits: self.limits, slot, constraint: self.constraint, think_budget: self.think_budget, thinking_open: self.thinking_open, cancel, tx }
    }
}

/// Renders the template, tokenizes, and builds the grammar and budget a spec asks for
pub fn prepare(s: &Served, spec: &ChatSpec) -> Result<Prepared> {
    let Some(template) = &s.template else { bail!("this seat proposes drafts and renders no chats") };
    if let Some(id) = spec.id_slot
        && id >= s.n_seq
    {
        bail!("this seat keeps {} slots, slot {id} does not exist", s.n_seq);
    }
    let mut kwargs = spec.chat_template_kwargs.clone();
    let mut think_budget = match &spec.reasoning_effort {
        Some(e) => {
            let budget = reasoning_budget(e)?;
            kwargs.entry("reasoning_effort").or_insert_with(|| Value::String(e.clone()));
            if budget == Some(0) {
                kwargs.entry("enable_thinking").or_insert(Value::Bool(false));
            }
            budget
        }
        None => None,
    };
    if let Some(b) = spec.think_budget {
        think_budget = b;
        if b == Some(0) {
            kwargs.entry("enable_thinking").or_insert(Value::Bool(false));
        } else {
            kwargs.entry("enable_thinking").or_insert(Value::Bool(true));
        }
    }
    let tools: Option<Value> = match (&spec.tools, &spec.tool_choice) {
        (_, ToolChoice::None) | (None, _) => None,
        (Some(t), _) => Some(Value::Array(t.clone())),
    };
    if matches!(spec.tool_choice, ToolChoice::Required | ToolChoice::Named(_)) && spec.tools.as_ref().is_none_or(Vec::is_empty) {
        bail!("tool_choice forces a tool call but the request lists no tools");
    }
    let (messages, images) = lift_images(&spec.messages)?;
    if !images.is_empty() && s.vision.is_none() {
        bail!("the request carries an image but this seat was launched without --mmproj");
    }
    let prompt = template.render(&messages, tools.as_ref(), true, &kwargs)?;
    let (ids, chunks) = chunk_prompt(s, &prompt, images)?;
    if ids.len() >= s.n_ctx {
        bail!("the prompt is {} tokens, the context {}", ids.len(), s.n_ctx);
    }
    let thinking_open = prompt.trim_end().ends_with(THINK_OPEN);
    let sampling = Sampling {
        temperature: spec.temperature.unwrap_or(s.defaults.temperature),
        top_p: spec.top_p.unwrap_or(s.defaults.top_p),
        top_k: spec.top_k.unwrap_or(s.defaults.top_k),
        min_p: spec.min_p.unwrap_or(s.defaults.min_p),
        repeat_penalty: spec.repeat_penalty.unwrap_or(s.defaults.repeat_penalty),
        ..Default::default()
    };
    let limits = Limits { max_tokens: spec.max_tokens.unwrap_or(s.max_tokens).max(1), seed: spec.seed.unwrap_or(42), stop: spec.stop.clone(), bias: spec.logit_bias.clone() };
    let body = match (&spec.tool_choice, &spec.format) {
        (ToolChoice::Required | ToolChoice::Named(_), ResponseFormat::JsonObject | ResponseFormat::JsonSchema(_)) => bail!("a forced tool call and a response format cannot both constrain the answer"),
        (ToolChoice::Required, _) => Some(tool_call_grammar(s.format, spec.tools.as_deref().unwrap_or(&[]), None)?),
        (ToolChoice::Named(n), _) => Some(tool_call_grammar(s.format, spec.tools.as_deref().unwrap_or(&[]), Some(n))?),
        (_, ResponseFormat::JsonObject) => {
            let mut b = GrammarBuilder::new();
            let n = b.json_object();
            Some((b, n))
        }
        (_, ResponseFormat::JsonSchema(schema)) => {
            let mut b = GrammarBuilder::new();
            let n = b.json(schema)?;
            Some((b, n))
        }
        (_, ResponseFormat::Text) => None,
    };
    let constraint = body.map(|(b, node)| {
        let think = Node::Seq(vec![Node::Literal(THINK_OPEN.as_bytes().to_vec()), Node::FreeUntil(THINK_CLOSE.as_bytes().to_vec())]);
        let prelude = if thinking_open { Node::FreeUntil(THINK_CLOSE.as_bytes().to_vec()) } else { Node::Opt(Box::new(think)) };
        let bare = !s.thinks || (think_budget == Some(0) && !thinking_open);
        let root = if bare { Node::Seq(vec![Node::Space, node]) } else { Node::Seq(vec![prelude, Node::Space, node]) };
        Constraint::new(b.finish(root))
    });
    Ok(Prepared { n_prompt: ids.len(), ids, chunks, prompt, thinking_open, sampling, limits, constraint, think_budget })
}

/// The name and parameter schema of one OpenAI shaped tool
fn tool_parts(t: &Value) -> Result<(String, Value)> {
    let f = t.get("function").unwrap_or(t);
    let name = f.get("name").and_then(Value::as_str).ok_or_else(|| anyhow!("a tool without a name"))?;
    let params = f.get("parameters").cloned().unwrap_or_else(|| serde_json::json!({ "type": "object" }));
    Ok((name.to_string(), params))
}

const DS_BEGIN: &str = "<｜tool▁calls▁begin｜>";
const DS_END: &str = "<｜tool▁calls▁end｜>";
const DS_CALL_BEGIN: &str = "<｜tool▁call▁begin｜>";
const DS_CALL_END: &str = "<｜tool▁call▁end｜>";
const DS_SEP: &str = "<｜tool▁sep｜>";

/// A grammar producing exactly one call of one of the tools in the template's own format, or of the named tool
pub fn tool_call_grammar(format: Format, tools: &[Value], only: Option<&str>) -> Result<(GrammarBuilder, Node)> {
    let mut b = GrammarBuilder::new();
    let mut alts = Vec::new();
    for t in tools {
        let (name, params) = tool_parts(t)?;
        if only.is_some_and(|o| o != name) {
            continue;
        }
        let lit = |s: &str| Node::Literal(s.as_bytes().to_vec());
        let call = match format {
            Format::HermesJson => {
                let args = b.json(&params)?;
                Node::Seq(vec![
                    lit("<tool_call>"),
                    Node::Space,
                    lit("{"),
                    Node::Space,
                    lit("\"name\""),
                    Node::Space,
                    lit(":"),
                    Node::Space,
                    Node::Literal(serde_json::to_vec(&Value::String(name.clone()))?),
                    Node::Space,
                    lit(","),
                    Node::Space,
                    lit("\"arguments\""),
                    Node::Space,
                    lit(":"),
                    Node::Space,
                    args,
                    Node::Space,
                    lit("}"),
                    Node::Space,
                    lit("</tool_call>"),
                ])
            }
            Format::GlmXml => {
                let mut parts = vec![lit("<tool_call>"), lit(&name), Node::Space];
                let props = params.get("properties").and_then(Value::as_object).cloned().unwrap_or_default();
                let required: Vec<&str> = params.get("required").and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_str).collect()).unwrap_or_default();
                for (k, sub) in &props {
                    let is_string = sub.get("type").and_then(Value::as_str) == Some("string") && sub.get("enum").is_none() && sub.get("const").is_none();
                    let mut one = vec![lit("<arg_key>"), lit(k), lit("</arg_key>"), Node::Space, lit("<arg_value>")];
                    if is_string {
                        one.push(Node::FreeUntil(b"</arg_value>".to_vec()));
                    } else {
                        one.push(b.json(sub)?);
                        one.push(Node::Space);
                        one.push(lit("</arg_value>"));
                    }
                    one.push(Node::Space);
                    let seq = Node::Seq(one);
                    parts.push(if required.contains(&k.as_str()) { seq } else { Node::Opt(Box::new(seq)) });
                }
                parts.push(lit("</tool_call>"));
                Node::Seq(parts)
            }
            Format::DeepSeek => {
                let args = b.json(&params)?;
                Node::Seq(vec![lit(DS_BEGIN), lit(DS_CALL_BEGIN), lit("function"), lit(DS_SEP), lit(&name), lit("\n```json\n"), args, Node::Space, lit("```"), lit(DS_CALL_END), lit(DS_END)])
            }
            Format::None => bail!("this model's chat template has no tool call format, so a tool call cannot be forced"),
        };
        alts.push(call);
    }
    if alts.is_empty() {
        bail!("tool_choice names {}, which the request's tools do not list", only.unwrap_or("a tool"));
    }
    let node = if alts.len() == 1 { alts.remove(0) } else { Node::Alt(alts) };
    Ok((b, node))
}

/// Messages of the OpenAI chat shape with every content form the gateway sends
pub fn messages_from_openai(v: &Value) -> Result<Vec<Message>> {
    let arr = v.as_array().ok_or_else(|| anyhow!("messages must be an array"))?;
    arr.iter().map(|m| serde_json::from_value::<Message>(m.clone()).map_err(|e| anyhow!("message: {e}"))).collect()
}

/// Stop strings from a string or an array of strings
pub fn stop_strings(v: Option<&Value>) -> Vec<String> {
    match v {
        Some(Value::String(x)) => vec![x.clone()],
        Some(Value::Array(a)) => a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect(),
        _ => Vec::new(),
    }
}

/// Logit bias from a map of token id to value
pub fn logit_bias(v: Option<&Map<String, Value>>) -> Result<Vec<(u32, f32)>> {
    let mut bias = Vec::new();
    for (k, v) in v.into_iter().flatten() {
        let id: u32 = k.parse().map_err(|_| anyhow!("logit_bias key {k:?} is not a token id"))?;
        let b = v.as_f64().ok_or_else(|| anyhow!("logit_bias value {v} is not a number"))? as f32;
        bias.push((id, b));
    }
    Ok(bias)
}

/// Messages with every image part replaced by the media marker, the images' bytes in prompt order
fn lift_images(messages: &[Message]) -> Result<(Vec<Message>, Vec<Vec<u8>>)> {
    let mut out = Vec::with_capacity(messages.len());
    let mut images = Vec::new();
    for m in messages {
        let mut m = m.clone();
        if let Value::Array(parts) = &m.content {
            let mut lifted = Vec::with_capacity(parts.len());
            for p in parts {
                match p.get("type").and_then(Value::as_str) {
                    Some("image_url") => {
                        let url = p.get("image_url").and_then(|u| u.get("url").and_then(Value::as_str).or_else(|| u.as_str())).ok_or_else(|| anyhow!("image_url part without a url"))?;
                        images.push(fetch_image(url)?);
                        lifted.push(serde_json::json!({ "type": "text", "text": MEDIA_MARKER }));
                    }
                    _ => lifted.push(p.clone()),
                }
            }
            m.content = Value::Array(lifted);
        }
        out.push(m);
    }
    Ok((out, images))
}

/// The bytes of a data url or of an http url fetched now, within the size limit
fn fetch_image(url: &str) -> Result<Vec<u8>> {
    if let Some(rest) = url.strip_prefix("data:") {
        let (_, payload) = rest.split_once(',').ok_or_else(|| anyhow!("data url without a comma"))?;
        let mime = rest.split_once(',').map(|(m, _)| m).unwrap_or("");
        if !mime.contains(";base64") {
            bail!("data url is not base64");
        }
        use base64::Engine as _;
        let bytes = base64::engine::general_purpose::STANDARD.decode(payload.trim()).map_err(|e| anyhow!("data url base64: {e}"))?;
        if bytes.len() > IMAGE_LIMIT {
            bail!("image of {} bytes is over the {IMAGE_LIMIT} byte limit", bytes.len());
        }
        return Ok(bytes);
    }
    if url.starts_with("http://") || url.starts_with("https://") {
        let resp = ureq::get(url).timeout(std::time::Duration::from_secs(30)).call().map_err(|e| anyhow!("fetch image {url}: {e}"))?;
        let mut bytes = Vec::new();
        use std::io::Read as _;
        resp.into_reader().take(IMAGE_LIMIT as u64 + 1).read_to_end(&mut bytes)?;
        if bytes.len() > IMAGE_LIMIT {
            bail!("image at {url} is over the {IMAGE_LIMIT} byte limit");
        }
        return Ok(bytes);
    }
    bail!("image url {url:?} is not a data url or an http url")
}

/// The rendered prompt as token and image chunks, each image wrapped in the tower's marker tokens, with the key of every row
fn chunk_prompt(s: &Served, prompt: &str, images: Vec<Vec<u8>>) -> Result<(Vec<u32>, Vec<Chunk>)> {
    let pieces: Vec<&str> = prompt.split(MEDIA_MARKER).collect();
    if pieces.len() != images.len() + 1 {
        bail!("the template left {} image markers for {} images", pieces.len() - 1, images.len());
    }
    let mut ids = Vec::new();
    let mut chunks: Vec<Chunk> = Vec::new();
    let mut text = Vec::new();
    let push_text = |text: &mut Vec<u32>, chunks: &mut Vec<Chunk>, ids: &mut Vec<u32>| {
        if !text.is_empty() {
            ids.extend_from_slice(text);
            chunks.push(Chunk::Text(std::mem::take(text)));
        }
    };
    let mut images = images.into_iter();
    for (k, piece) in pieces.iter().enumerate() {
        text.extend(s.tok.encode(piece, k == 0)?);
        let Some(bytes) = (k + 1 < pieces.len()).then(|| images.next()).flatten() else { continue };
        let clip = s.vision.as_ref().ok_or_else(|| anyhow!("image without a projector"))?;
        let (beg, end) = clip.markers();
        text.extend(s.tok.encode(beg, false)?);
        push_text(&mut text, &mut chunks, &mut ids);
        let img = clip.preprocess(&bytes)?;
        let n = image_tokens(&img);
        if n == 0 {
            bail!("image leaves no tokens after preprocessing");
        }
        let key = img.key();
        ids.extend(std::iter::repeat_n(key, n));
        chunks.push(Chunk::Image(img));
        text.extend(s.tok.encode(end, false)?);
    }
    push_text(&mut text, &mut chunks, &mut ids);
    Ok((ids, chunks))
}

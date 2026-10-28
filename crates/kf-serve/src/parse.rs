use anyhow::Result;
use serde::Serialize;

/// The tool call format a template speaks, read from the template's own markers
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Format {
    None,
    HermesJson,
    GlmXml,
    DeepSeek,
}

impl Format {
    pub fn detect(template: &str) -> Self {
        if template.contains("<arg_key>") {
            Self::GlmXml
        } else if template.contains("tool▁calls▁begin") {
            Self::DeepSeek
        } else if template.contains("<tool_call>") {
            Self::HermesJson
        } else {
            Self::None
        }
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Parsed {
    pub content: String,
    pub reasoning: Option<String>,
    pub tool_calls: Vec<ToolCall>,
}

/// Splits model output into reasoning, content, and tool calls by the template's format
pub fn parse_output(text: &str, format: Format, thinking_open: bool) -> Result<Parsed> {
    let (reasoning, rest) = split_think(text, thinking_open);
    let mut parsed = Parsed { reasoning, ..Default::default() };
    match format {
        Format::None => parsed.content = rest.trim().to_string(),
        Format::HermesJson => {
            let (content, calls) = split_hermes(&rest);
            parsed.content = content;
            parsed.tool_calls = calls;
        }
        Format::GlmXml => return parse_glm_xml_tool_calls(&rest, parsed),
        Format::DeepSeek => return parse_deepseek_tool_calls(&rest, parsed),
    }
    Ok(parsed)
}

/// Reasoning and the rest, the prompt having opened a think block or the output opening one itself
fn split_think(text: &str, thinking_open: bool) -> (Option<String>, String) {
    let open = "<think>";
    let close = "</think>";
    let body = text.trim_start();
    let after = match (thinking_open, body.strip_prefix(open)) {
        (_, Some(after)) => after,
        (true, None) => body,
        (false, None) => return (None, text.to_string()),
    };
    match after.find(close) {
        Some(end) => (Some(after[..end].trim().to_string()), after[end + close.len()..].to_string()),
        None => (Some(after.trim().to_string()), String::new()),
    }
}

/// One increment of parsed output
#[derive(Clone, Debug, PartialEq)]
pub enum Delta {
    Reasoning(String),
    Content(String),
    ToolCall(ToolCall),
}

impl PartialEq for ToolCall {
    fn eq(&self, o: &Self) -> bool {
        self.id == o.id && self.name == o.name && self.arguments == o.arguments
    }
}

const TAGS: [&str; 4] = ["<think>", "</think>", "<tool_call>", "</tool_call>"];

/// Parses output as it streams, holding back only what might be the start of a tag
pub struct StreamParser {
    format: Format,
    buf: String,
    in_think: bool,
    in_tool: bool,
    n_calls: usize,
}

impl StreamParser {
    pub fn new(format: Format, thinking_open: bool) -> Self {
        Self { format, buf: String::new(), in_think: thinking_open, in_tool: false, n_calls: 0 }
    }

    pub fn feed(&mut self, piece: &str) -> Result<Vec<Delta>> {
        self.buf.push_str(piece);
        match self.format {
            Format::None | Format::HermesJson => Ok(self.drain(false)),
            Format::GlmXml => parse_glm_xml_stream(self, piece),
            Format::DeepSeek => parse_deepseek_stream(self, piece),
        }
    }

    pub fn finish(&mut self) -> Result<Vec<Delta>> {
        match self.format {
            Format::None | Format::HermesJson => Ok(self.drain(true)),
            Format::GlmXml => parse_glm_xml_stream(self, ""),
            Format::DeepSeek => parse_deepseek_stream(self, ""),
        }
    }

    fn drain(&mut self, at_end: bool) -> Vec<Delta> {
        let mut out = Vec::new();
        loop {
            if self.in_tool {
                let Some(end) = self.buf.find("</tool_call>") else {
                    if at_end {
                        out.push(Delta::Content(std::mem::take(&mut self.buf)));
                    }
                    return out;
                };
                let body = self.buf[..end].trim().to_string();
                self.buf.drain(..end + "</tool_call>".len());
                self.in_tool = false;
                match serde_json::from_str::<serde_json::Value>(&body) {
                    Ok(v) => {
                        let name = v.get("name").and_then(|n| n.as_str()).unwrap_or("").to_string();
                        let arguments = v.get("arguments").map(|a| a.to_string()).unwrap_or_else(|| "{}".to_string());
                        out.push(Delta::ToolCall(ToolCall { id: format!("call_{}", self.n_calls), name, arguments }));
                        self.n_calls += 1;
                    }
                    Err(_) => out.push(Delta::Content(format!("<tool_call>{body}</tool_call>"))),
                }
                continue;
            }
            if self.in_think {
                match self.buf.find("</think>") {
                    Some(end) => {
                        let r = self.buf[..end].to_string();
                        if !r.is_empty() {
                            out.push(Delta::Reasoning(r));
                        }
                        self.buf.drain(..end + "</think>".len());
                        self.in_think = false;
                        continue;
                    }
                    None => {
                        let keep = if at_end { 0 } else { held_back(&self.buf) };
                        let emit: String = self.buf.drain(..self.buf.len() - keep).collect();
                        if !emit.is_empty() {
                            out.push(Delta::Reasoning(emit));
                        }
                        return out;
                    }
                }
            }
            let think = self.buf.find("<think>");
            let tool = if self.format == Format::HermesJson { self.buf.find("<tool_call>") } else { None };
            let next = [think.map(|i| (i, "<think>")), tool.map(|i| (i, "<tool_call>"))].into_iter().flatten().min_by_key(|(i, _)| *i);
            match next {
                Some((i, tag)) => {
                    let before = self.buf[..i].to_string();
                    if !before.is_empty() {
                        out.push(Delta::Content(before));
                    }
                    self.buf.drain(..i + tag.len());
                    if tag == "<think>" {
                        self.in_think = true;
                    } else {
                        self.in_tool = true;
                    }
                }
                None => {
                    let keep = if at_end { 0 } else { held_back(&self.buf) };
                    let emit: String = self.buf.drain(..self.buf.len() - keep).collect();
                    if !emit.is_empty() {
                        out.push(Delta::Content(emit));
                    }
                    return out;
                }
            }
        }
    }
}

/// Length of a trailing fragment that could still grow into a tag
fn held_back(buf: &str) -> usize {
    let bytes = buf.as_bytes();
    (1..=bytes.len().min(12))
        .rev()
        .find(|n| buf.is_char_boundary(buf.len() - n) && TAGS.iter().any(|t| t.starts_with(&buf[buf.len() - n..])))
        .unwrap_or(0)
}

fn split_hermes(text: &str) -> (String, Vec<ToolCall>) {
    let mut content = String::new();
    let mut calls = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find("<tool_call>") {
        content.push_str(&rest[..start]);
        let after = &rest[start + "<tool_call>".len()..];
        let end = after.find("</tool_call>").unwrap_or(after.len());
        let body = after[..end].trim();
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(body) {
            let name = v.get("name").and_then(|n| n.as_str()).unwrap_or("").to_string();
            let arguments = v.get("arguments").map(|a| a.to_string()).unwrap_or_else(|| "{}".to_string());
            calls.push(ToolCall { id: format!("call_{}", calls.len()), name, arguments });
        } else {
            content.push_str("<tool_call>");
            content.push_str(body);
        }
        rest = if end < after.len() { &after[end + "</tool_call>".len()..] } else { "" };
    }
    content.push_str(rest);
    (content.trim().to_string(), calls)
}

const GLM_OPEN: &str = "<tool_call>";
const GLM_CLOSE: &str = "</tool_call>";
const DS_BEGIN: &str = "<｜tool▁calls▁begin｜>";
const DS_END: &str = "<｜tool▁calls▁end｜>";
const DS_CALL_BEGIN: &str = "<｜tool▁call▁begin｜>";
const DS_CALL_END: &str = "<｜tool▁call▁end｜>";
const DS_SEP: &str = "<｜tool▁sep｜>";

/// One argument value, kept as JSON when it parses and quoted otherwise
fn json_value(raw: &str) -> serde_json::Value {
    serde_json::from_str(raw.trim()).unwrap_or_else(|_| serde_json::Value::String(raw.trim().to_string()))
}

/// The call inside one GLM block, the name then arg_key and arg_value pairs
fn glm_call(body: &str, index: usize) -> ToolCall {
    let name_end = body.find("<arg_key>").unwrap_or(body.len());
    let name = body[..name_end].trim().to_string();
    let mut args = serde_json::Map::new();
    let mut rest = &body[name_end..];
    while let Some(k0) = rest.find("<arg_key>") {
        let after = &rest[k0 + "<arg_key>".len()..];
        let Some(k1) = after.find("</arg_key>") else { break };
        let key = after[..k1].trim().to_string();
        let after = &after[k1 + "</arg_key>".len()..];
        let Some(v0) = after.find("<arg_value>") else { break };
        let after = &after[v0 + "<arg_value>".len()..];
        let v1 = after.find("</arg_value>").unwrap_or(after.len());
        args.insert(key, json_value(&after[..v1]));
        rest = if v1 < after.len() { &after[v1 + "</arg_value>".len()..] } else { "" };
    }
    ToolCall { id: format!("call_{index}"), name, arguments: serde_json::Value::Object(args).to_string() }
}

/// GLM style blocks, a name then tagged key value pairs inside tool_call tags
pub fn parse_glm_xml_tool_calls(text: &str, mut parsed: Parsed) -> Result<Parsed> {
    let mut content = String::new();
    let mut rest = text;
    while let Some(start) = rest.find(GLM_OPEN) {
        content.push_str(&rest[..start]);
        let after = &rest[start + GLM_OPEN.len()..];
        let end = after.find(GLM_CLOSE).unwrap_or(after.len());
        parsed.tool_calls.push(glm_call(&after[..end], parsed.tool_calls.len()));
        rest = if end < after.len() { &after[end + GLM_CLOSE.len()..] } else { "" };
    }
    content.push_str(rest);
    parsed.content = content.trim().to_string();
    Ok(parsed)
}

/// Streaming GLM blocks, content emitted as it arrives and each block emitted once closed
pub fn parse_glm_xml_stream(p: &mut StreamParser, piece: &str) -> Result<Vec<Delta>> {
    let at_end = piece.is_empty();
    let mut out = Vec::new();
    loop {
        if p.in_think {
            match p.buf.find("</think>") {
                Some(end) => {
                    if end > 0 {
                        out.push(Delta::Reasoning(p.buf[..end].to_string()));
                    }
                    p.buf.drain(..end + "</think>".len());
                    p.in_think = false;
                    continue;
                }
                None => {
                    let keep = if at_end { 0 } else { held_back(&p.buf) };
                    let emit: String = p.buf.drain(..p.buf.len() - keep).collect();
                    if !emit.is_empty() {
                        out.push(Delta::Reasoning(emit));
                    }
                    return Ok(out);
                }
            }
        }
        if p.in_tool {
            let Some(end) = p.buf.find(GLM_CLOSE) else {
                if at_end {
                    let body: String = p.buf.drain(..).collect();
                    out.push(Delta::ToolCall(glm_call(&body, p.n_calls)));
                    p.n_calls += 1;
                    p.in_tool = false;
                }
                return Ok(out);
            };
            let body = p.buf[..end].to_string();
            p.buf.drain(..end + GLM_CLOSE.len());
            p.in_tool = false;
            out.push(Delta::ToolCall(glm_call(&body, p.n_calls)));
            p.n_calls += 1;
            continue;
        }
        let think = p.buf.find("<think>");
        let tool = p.buf.find(GLM_OPEN);
        match [think.map(|i| (i, "<think>")), tool.map(|i| (i, GLM_OPEN))].into_iter().flatten().min_by_key(|(i, _)| *i) {
            Some((i, tag)) => {
                if i > 0 {
                    out.push(Delta::Content(p.buf[..i].to_string()));
                }
                p.buf.drain(..i + tag.len());
                if tag == "<think>" {
                    p.in_think = true;
                } else {
                    p.in_tool = true;
                }
            }
            None => {
                let keep = if at_end { 0 } else { held_back(&p.buf) };
                let emit: String = p.buf.drain(..p.buf.len() - keep).collect();
                if !emit.is_empty() {
                    out.push(Delta::Content(emit));
                }
                return Ok(out);
            }
        }
    }
}

/// The calls inside one DeepSeek section, each a function name then a fenced json argument block
fn deepseek_calls(section: &str, first_index: usize) -> Vec<ToolCall> {
    let mut calls = Vec::new();
    let mut rest = section;
    while let Some(c0) = rest.find(DS_CALL_BEGIN) {
        let after = &rest[c0 + DS_CALL_BEGIN.len()..];
        let end = after.find(DS_CALL_END).unwrap_or(after.len());
        let body = &after[..end];
        let name_start = body.find(DS_SEP).map_or(0, |i| i + DS_SEP.len());
        let name_end = body[name_start..].find('\n').map_or(body.len(), |i| name_start + i);
        let name = body[name_start..name_end].trim().to_string();
        let args = match body[name_end..].find("```") {
            Some(f0) => {
                let fenced = &body[name_end + f0 + 3..];
                let fenced = fenced.strip_prefix("json").unwrap_or(fenced);
                let f1 = fenced.find("```").unwrap_or(fenced.len());
                json_value(&fenced[..f1]).to_string()
            }
            None => json_value(&body[name_end..]).to_string(),
        };
        calls.push(ToolCall { id: format!("call_{}", first_index + calls.len()), name, arguments: args });
        rest = if end < after.len() { &after[end + DS_CALL_END.len()..] } else { "" };
    }
    calls
}

/// DeepSeek V3 sections with their unicode markers, every call a name and a fenced json block
pub fn parse_deepseek_tool_calls(text: &str, mut parsed: Parsed) -> Result<Parsed> {
    let mut content = String::new();
    let mut rest = text;
    while let Some(start) = rest.find(DS_BEGIN) {
        content.push_str(&rest[..start]);
        let after = &rest[start + DS_BEGIN.len()..];
        let end = after.find(DS_END).unwrap_or(after.len());
        let n = parsed.tool_calls.len();
        parsed.tool_calls.extend(deepseek_calls(&after[..end], n));
        rest = if end < after.len() { &after[end + DS_END.len()..] } else { "" };
    }
    content.push_str(rest);
    parsed.content = content.trim().to_string();
    Ok(parsed)
}

/// Streaming DeepSeek sections, content emitted as it arrives and each call emitted once its end marker lands
pub fn parse_deepseek_stream(p: &mut StreamParser, piece: &str) -> Result<Vec<Delta>> {
    let at_end = piece.is_empty();
    let mut out = Vec::new();
    loop {
        if p.in_think {
            match p.buf.find("</think>") {
                Some(end) => {
                    if end > 0 {
                        out.push(Delta::Reasoning(p.buf[..end].to_string()));
                    }
                    p.buf.drain(..end + "</think>".len());
                    p.in_think = false;
                    continue;
                }
                None => {
                    let keep = if at_end { 0 } else { held_back_ds(&p.buf) };
                    let emit: String = p.buf.drain(..p.buf.len() - keep).collect();
                    if !emit.is_empty() {
                        out.push(Delta::Reasoning(emit));
                    }
                    return Ok(out);
                }
            }
        }
        if p.in_tool {
            if let Some(end) = p.buf.find(DS_CALL_END) {
                let body = p.buf[..end + DS_CALL_END.len()].to_string();
                p.buf.drain(..end + DS_CALL_END.len());
                for c in deepseek_calls(&body, p.n_calls) {
                    out.push(Delta::ToolCall(c));
                    p.n_calls += 1;
                }
                continue;
            }
            if let Some(end) = p.buf.find(DS_END) {
                let body = p.buf[..end].to_string();
                p.buf.drain(..end + DS_END.len());
                p.in_tool = false;
                for c in deepseek_calls(&body, p.n_calls) {
                    out.push(Delta::ToolCall(c));
                    p.n_calls += 1;
                }
                continue;
            }
            if at_end {
                let body: String = p.buf.drain(..).collect();
                for c in deepseek_calls(&body, p.n_calls) {
                    out.push(Delta::ToolCall(c));
                    p.n_calls += 1;
                }
                p.in_tool = false;
            }
            return Ok(out);
        }
        let think = p.buf.find("<think>");
        let tool = p.buf.find(DS_BEGIN);
        match [think.map(|i| (i, "<think>")), tool.map(|i| (i, DS_BEGIN))].into_iter().flatten().min_by_key(|(i, _)| *i) {
            Some((i, tag)) => {
                if i > 0 {
                    out.push(Delta::Content(p.buf[..i].to_string()));
                }
                p.buf.drain(..i + tag.len());
                if tag == "<think>" {
                    p.in_think = true;
                } else {
                    p.in_tool = true;
                }
            }
            None => {
                let keep = if at_end { 0 } else { held_back_ds(&p.buf) };
                let emit: String = p.buf.drain(..p.buf.len() - keep).collect();
                if !emit.is_empty() {
                    out.push(Delta::Content(emit));
                }
                return Ok(out);
            }
        }
    }
}

/// Length of a trailing fragment that could still grow into a think tag or a DeepSeek marker
fn held_back_ds(buf: &str) -> usize {
    let tags = ["<think>", "</think>", DS_BEGIN, DS_END];
    let longest = tags.iter().map(|t| t.len()).max().unwrap_or(0);
    (1..=buf.len().min(longest))
        .rev()
        .find(|n| buf.is_char_boundary(buf.len() - n) && tags.iter().any(|t| t.starts_with(&buf[buf.len() - n..])))
        .unwrap_or(0)
}

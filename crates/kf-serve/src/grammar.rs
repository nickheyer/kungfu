//! Grammar constrained sampling, a JSON schema or a wrapped tool call as an acceptor over the bytes the model emits
use anyhow::{Result, bail};
use serde_json::{Map, Value};
use std::sync::Arc;

/// One compiled schema node, every reference resolved to an index
#[derive(Clone, Debug, PartialEq)]
pub struct Schema {
    pub types: Vec<Type>,
    /// Alternatives, one thread per entry, the node's own keywords ignored when present
    pub any_of: Vec<usize>,
    /// Exact JSON texts the value may be, from enum or const
    pub consts: Vec<Vec<u8>>,
    /// Property names as the escaped bytes between their quotes, with their schemas
    pub properties: Vec<(Vec<u8>, usize)>,
    pub required: Vec<bool>,
    /// Schema of keys outside the listed properties, none when they are refused
    pub additional: Option<usize>,
    pub prefix_items: Vec<usize>,
    pub items: Option<usize>,
    pub min_items: usize,
    pub max_items: Option<usize>,
    pub min_length: usize,
    pub max_length: Option<usize>,
    pub minimum: Option<i128>,
    pub maximum: Option<i128>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Type {
    Object,
    Array,
    String,
    Number,
    Integer,
    Boolean,
    Null,
}

/// A grammar node, literals and free text around JSON values
#[derive(Clone, Debug, PartialEq)]
pub enum Node {
    Literal(Vec<u8>),
    /// Any bytes through the first occurrence of the terminator
    FreeUntil(Vec<u8>),
    /// Zero or more JSON whitespace bytes
    Space,
    Json(usize),
    Seq(Vec<Node>),
    Alt(Vec<Node>),
    Opt(Box<Node>),
}

/// A compiled grammar, the node tree with its schema table
#[derive(Clone, Debug)]
pub struct Grammar {
    root: Node,
    schemas: Vec<Schema>,
}

/// Schema keywords the acceptor does not enforce, a schema naming one is refused
pub const KEYWORDS_REFUSED: &[&str] = &["pattern", "allOf", "not", "if", "then", "else", "patternProperties", "propertyNames", "uniqueItems", "multipleOf", "contains", "dependentRequired", "dependentSchemas", "unevaluatedProperties", "unevaluatedItems"];

/// Compiles a JSON schema document into the table, references resolved against the root's definitions
struct Compiler<'a> {
    root: &'a Value,
    schemas: Vec<Schema>,
    /// Reference paths already compiled with their index, recursion lands on the same node
    refs: Vec<(String, usize)>,
}

impl<'a> Compiler<'a> {
    fn resolve(&self, path: &str) -> Result<&'a Value> {
        let Some(rest) = path.strip_prefix("#/") else { bail!("schema reference {path:?} is not a local pointer") };
        let mut cur = self.root;
        for part in rest.split('/') {
            let part = part.replace("~1", "/").replace("~0", "~");
            cur = match cur {
                Value::Object(m) => m.get(&part).ok_or_else(|| anyhow::anyhow!("schema reference {path:?} names nothing"))?,
                Value::Array(a) => a.get(part.parse::<usize>().unwrap_or(usize::MAX)).ok_or_else(|| anyhow::anyhow!("schema reference {path:?} names nothing"))?,
                _ => bail!("schema reference {path:?} names nothing"),
            };
        }
        Ok(cur)
    }

    fn compile(&mut self, v: &'a Value) -> Result<usize> {
        static EMPTY: std::sync::LazyLock<Map<String, Value>> = std::sync::LazyLock::new(Map::new);
        let m: &Map<String, Value> = match v {
            Value::Bool(true) => &EMPTY,
            Value::Bool(false) => bail!("a schema of false admits no value"),
            Value::Object(m) => m,
            other => bail!("schema {other} is not an object"),
        };
        if let Some(r) = m.get("$ref").and_then(Value::as_str) {
            if let Some((_, i)) = self.refs.iter().find(|(p, _)| p == r) {
                return Ok(*i);
            }
            let target = self.resolve(r)?;
            let i = self.schemas.len();
            self.schemas.push(Schema::any());
            self.refs.push((r.to_string(), i));
            let built = self.compile(target)?;
            let node = self.schemas[built].clone();
            self.schemas[i] = node;
            return Ok(i);
        }
        if let Some(k) = KEYWORDS_REFUSED.iter().find(|k| m.contains_key(**k)) {
            bail!("schema keyword {k} is not enforced by this server");
        }
        let i = self.schemas.len();
        self.schemas.push(Schema::any());
        let mut s = Schema::any();
        let alts = m.get("anyOf").or_else(|| m.get("oneOf")).and_then(Value::as_array);
        if let Some(alts) = alts {
            if alts.is_empty() {
                bail!("an empty anyOf admits no value");
            }
            for a in alts {
                let j = self.compile(a)?;
                s.any_of.push(j);
            }
            self.schemas[i] = s;
            return Ok(i);
        }
        s.types = match m.get("type") {
            None => Vec::new(),
            Some(Value::String(t)) => vec![parse_type(t)?],
            Some(Value::Array(a)) => a.iter().map(|t| t.as_str().ok_or_else(|| anyhow::anyhow!("type {t} is not a name")).and_then(parse_type)).collect::<Result<_>>()?,
            Some(other) => bail!("type {other} is not a name"),
        };
        if let Some(c) = m.get("const") {
            s.consts = vec![serde_json::to_vec(c)?];
        } else if let Some(e) = m.get("enum").and_then(Value::as_array) {
            if e.is_empty() {
                bail!("an empty enum admits no value");
            }
            s.consts = e.iter().map(serde_json::to_vec).collect::<Result<_, _>>()?;
        }
        if let Some(props) = m.get("properties").and_then(Value::as_object) {
            let req: Vec<&str> = m.get("required").and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_str).collect()).unwrap_or_default();
            for (k, sub) in props {
                let j = self.compile(sub)?;
                let text = serde_json::to_vec(&Value::String(k.clone()))?;
                s.properties.push((text[1..text.len() - 1].to_vec(), j));
                s.required.push(req.contains(&k.as_str()));
            }
            if let Some(r) = req.iter().find(|r| !props.contains_key(**r)) {
                bail!("required property {r} is not among the properties");
            }
        }
        s.additional = match m.get("additionalProperties") {
            None | Some(Value::Bool(true)) => Some(self.any_schema()),
            Some(Value::Bool(false)) => None,
            Some(sub) => Some(self.compile(sub)?),
        };
        if let Some(pre) = m.get("prefixItems").and_then(Value::as_array) {
            for sub in pre {
                let j = self.compile(sub)?;
                s.prefix_items.push(j);
            }
        }
        s.items = match m.get("items") {
            None | Some(Value::Bool(true)) => Some(self.any_schema()),
            Some(Value::Bool(false)) => None,
            Some(sub) => Some(self.compile(sub)?),
        };
        s.min_items = m.get("minItems").and_then(Value::as_u64).unwrap_or(0) as usize;
        s.max_items = m.get("maxItems").and_then(Value::as_u64).map(|n| n as usize);
        s.min_length = m.get("minLength").and_then(Value::as_u64).unwrap_or(0) as usize;
        s.max_length = m.get("maxLength").and_then(Value::as_u64).map(|n| n as usize);
        let bound = |key: &str| -> Result<Option<i128>> {
            match m.get(key) {
                None => Ok(None),
                Some(v) => match v.as_i64() {
                    Some(n) => Ok(Some(n as i128)),
                    None => bail!("{key} {v} is not a whole number, only integer bounds are enforced"),
                },
            }
        };
        s.minimum = bound("minimum")?;
        s.maximum = bound("maximum")?;
        if let Some(x) = bound("exclusiveMinimum")? {
            s.minimum = Some(s.minimum.map_or(x + 1, |m| m.max(x + 1)));
        }
        if let Some(x) = bound("exclusiveMaximum")? {
            s.maximum = Some(s.maximum.map_or(x - 1, |m| m.min(x - 1)));
        }
        if (s.minimum.is_some() || s.maximum.is_some()) && !s.types.is_empty() && !s.types.contains(&Type::Integer) {
            bail!("minimum and maximum are enforced on integers only");
        }
        if let (Some(a), Some(b)) = (s.minimum, s.maximum)
            && a > b
        {
            bail!("minimum {a} is above maximum {b}");
        }
        self.schemas[i] = s;
        Ok(i)
    }

    /// The one schema admitting any value, shared
    fn any_schema(&mut self) -> usize {
        if let Some(i) = self.schemas.iter().position(|s| *s == Schema::any()) {
            return i;
        }
        self.schemas.push(Schema::any());
        self.schemas.len() - 1
    }
}

fn parse_type(t: &str) -> Result<Type> {
    Ok(match t {
        "object" => Type::Object,
        "array" => Type::Array,
        "string" => Type::String,
        "number" => Type::Number,
        "integer" => Type::Integer,
        "boolean" => Type::Boolean,
        "null" => Type::Null,
        other => bail!("unknown schema type {other}"),
    })
}

impl Schema {
    fn any() -> Self {
        Self { types: Vec::new(), any_of: Vec::new(), consts: Vec::new(), properties: Vec::new(), required: Vec::new(), additional: None, prefix_items: Vec::new(), items: None, min_items: 0, max_items: None, min_length: 0, max_length: None, minimum: None, maximum: None }
    }

    fn allows(&self, t: Type) -> bool {
        self.types.is_empty() || self.types.contains(&t) || (t == Type::Integer && self.types.contains(&Type::Number))
    }

    fn integer_only(&self) -> bool {
        !self.types.is_empty() && !self.types.contains(&Type::Number)
    }
}

/// The grammar as a builder of trees around compiled schemas
pub struct GrammarBuilder {
    schemas: Vec<Schema>,
}

impl Default for GrammarBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl GrammarBuilder {
    pub fn new() -> Self {
        Self { schemas: Vec::new() }
    }

    /// A JSON node for a schema document, failing on keywords the acceptor does not enforce
    pub fn json(&mut self, schema: &Value) -> Result<Node> {
        let mut c = Compiler { root: schema, schemas: std::mem::take(&mut self.schemas), refs: Vec::new() };
        let i = c.compile(schema)?;
        self.schemas = c.schemas;
        Ok(Node::Json(i))
    }

    /// Any JSON object
    pub fn json_object(&mut self) -> Node {
        self.schemas.push(Schema { types: vec![Type::Object], ..Schema::any() });
        let any = self.schemas.len() - 1;
        self.schemas.push(Schema::any());
        let any_v = self.schemas.len() - 1;
        self.schemas[any].additional = Some(any_v);
        Node::Json(any)
    }

    pub fn finish(self, root: Node) -> Grammar {
        Grammar { root, schemas: self.schemas }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Esc {
    None,
    Backslash,
    Hex(u8),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NumState {
    AfterMinus,
    Zero,
    Int,
    Dot,
    Frac,
    Exp,
    ExpSign,
    ExpDigits,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ObjPhase {
    Open,
    AfterKey,
    AfterColon,
    AfterValue,
    AfterComma,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ArrPhase {
    Open,
    AfterValue,
    AfterComma,
}

/// One frame of a thread's stack, what the thread is in the middle of matching
#[derive(Clone, Debug, PartialEq)]
enum Frame {
    Lit { node: Arc<Vec<u8>>, pos: usize },
    Free { term: Arc<Vec<u8>>, matched: usize },
    Space,
    Seq { nodes: Arc<Vec<Node>>, idx: usize },
    Value { schema: usize },
    /// A JSON string after its opening quote, candidates narrow it to listed texts when given
    Str { schema: usize, chars: usize, esc: Esc, utf8: u8, cands: Option<Vec<(usize, usize)>>, raw: Vec<u8>, key: bool },
    Num { schema: usize, state: NumState, neg: bool, digits: i128, overflow: bool },
    Obj { schema: usize, phase: ObjPhase, seen: Vec<bool>, count: usize, prop: Option<usize> },
    Arr { schema: usize, phase: ArrPhase, count: usize },
    Word { text: &'static [u8], pos: usize },
}

/// Whitespace bytes in a row the structure admits between tokens, so a masked model cannot idle on blanks
const MAX_WS_RUN: u8 = 24;

/// One nondeterministic parse of the output so far
#[derive(Clone, Debug, PartialEq)]
struct Thread {
    stack: Vec<Frame>,
    /// Whitespace bytes in a row outside strings and free text
    ws: u8,
}

/// Where constrained sampling stands, the alive parses of everything accepted so far
#[derive(Clone, Debug)]
pub struct Constraint {
    grammar: Arc<Grammar>,
    threads: Vec<Thread>,
}

fn is_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\n' | b'\r' | b'\t')
}

impl Constraint {
    pub fn new(grammar: Grammar) -> Self {
        let grammar = Arc::new(grammar);
        let root = grammar.root.clone();
        let mut threads = Vec::new();
        push_node(&grammar, Thread { stack: Vec::new(), ws: 0 }, &root, &mut threads);
        Self { grammar, threads }
    }

    /// The state after every byte was accepted, none when some byte fits no parse
    pub fn try_advance(&self, bytes: &[u8]) -> Option<Constraint> {
        let mut threads = self.threads.clone();
        for &b in bytes {
            let mut next = Vec::new();
            for t in threads {
                step(&self.grammar, t, b, &mut next);
            }
            if next.is_empty() {
                return None;
            }
            next.dedup();
            if next.len() > 1 {
                let mut uniq: Vec<Thread> = Vec::with_capacity(next.len());
                for t in next {
                    if !uniq.contains(&t) {
                        uniq.push(t);
                    }
                }
                next = uniq;
            }
            threads = next;
        }
        Some(Constraint { grammar: self.grammar.clone(), threads })
    }

    /// Whether the output so far is a whole match, so the sequence may end here
    pub fn complete(&self) -> bool {
        self.threads.iter().any(|t| endable(&self.grammar, t))
    }

    /// Whether some byte could still extend a parse
    pub fn can_continue(&self) -> bool {
        self.threads.iter().any(|t| !t.stack.is_empty())
    }
}

/// Enters a node on top of a thread's stack, forking alternatives and options, settling epsilon frames
fn push_node(g: &Grammar, mut t: Thread, node: &Node, out: &mut Vec<Thread>) {
    match node {
        Node::Literal(l) => {
            if l.is_empty() {
                settle(g, t, out);
            } else {
                t.stack.push(Frame::Lit { node: Arc::new(l.clone()), pos: 0 });
                out.push(t);
            }
        }
        Node::FreeUntil(term) => {
            t.stack.push(Frame::Free { term: Arc::new(term.clone()), matched: 0 });
            out.push(t);
        }
        Node::Space => {
            t.stack.push(Frame::Space);
            out.push(t);
        }
        Node::Json(i) => push_value(g, t, *i, out),
        Node::Seq(nodes) => {
            if nodes.is_empty() {
                settle(g, t, out);
            } else {
                t.stack.push(Frame::Seq { nodes: Arc::new(nodes.clone()), idx: 0 });
                push_node(g, t, &nodes[0], out);
            }
        }
        Node::Alt(alts) => {
            for a in alts {
                push_node(g, t.clone(), a, out);
            }
        }
        Node::Opt(inner) => {
            settle(g, t.clone(), out);
            push_node(g, t, inner, out);
        }
    }
}

/// Enters a JSON value of a schema, alternatives forked
fn push_value(g: &Grammar, mut t: Thread, schema: usize, out: &mut Vec<Thread>) {
    let s = &g.schemas[schema];
    if !s.any_of.is_empty() {
        for &a in &s.any_of {
            push_value(g, t.clone(), a, out);
        }
        return;
    }
    t.stack.push(Frame::Value { schema });
    out.push(t);
}

/// Pops the finished top frame and advances the frame under it, until a frame wants a byte or the stack is empty
fn settle(g: &Grammar, mut t: Thread, out: &mut Vec<Thread>) {
    match t.stack.pop() {
        None => out.push(t),
        Some(Frame::Seq { nodes, idx }) => {
            let next = idx + 1;
            if next < nodes.len() {
                t.stack.push(Frame::Seq { nodes: nodes.clone(), idx: next });
                push_node(g, t, &nodes[next], out);
            } else {
                settle(g, t, out);
            }
        }
        Some(Frame::Obj { schema, phase: ObjPhase::AfterColon, seen, count, prop }) => {
            t.stack.push(Frame::Obj { schema, phase: ObjPhase::AfterValue, seen, count: count + 1, prop });
            out.push(t);
        }
        Some(Frame::Arr { schema, phase: ArrPhase::Open | ArrPhase::AfterComma, count }) => {
            t.stack.push(Frame::Arr { schema, phase: ArrPhase::AfterValue, count: count + 1 });
            out.push(t);
        }
        Some(other) => {
            t.stack.push(other);
            out.push(t);
        }
    }
}

/// Whether a thread could end right now, numbers and spaces closing without another byte
fn endable(g: &Grammar, t: &Thread) -> bool {
    let mut t = t.clone();
    loop {
        match t.stack.last() {
            None => return true,
            Some(Frame::Space) => {
                t.stack.pop();
            }
            Some(Frame::Num { .. }) => {
                let f = t.stack.pop().expect("number frame");
                if !num_closed(g, &f) {
                    return false;
                }
            }
            Some(_) => return false,
        }
        let mut next = Vec::new();
        settle(g, t, &mut next);
        if next.len() != 1 {
            return false;
        }
        t = next.remove(0);
    }
}

fn num_closed(g: &Grammar, f: &Frame) -> bool {
    let Frame::Num { schema, state, neg, digits, overflow } = f else { return false };
    if !matches!(state, NumState::Zero | NumState::Int | NumState::Frac | NumState::ExpDigits) {
        return false;
    }
    let s = &g.schemas[*schema];
    if matches!(state, NumState::Zero | NumState::Int) && (s.minimum.is_some() || s.maximum.is_some()) {
        if *overflow {
            return false;
        }
        let v = if *neg { -digits } else { *digits };
        return s.minimum.is_none_or(|m| v >= m) && s.maximum.is_none_or(|m| v <= m);
    }
    true
}

/// Whether some digits after the ones seen can land in the integer bounds
fn digits_viable(s: &Schema, neg: bool, digits: i128, overflow: bool) -> bool {
    if s.minimum.is_none() && s.maximum.is_none() {
        return true;
    }
    if overflow {
        return false;
    }
    let (lo, hi) = (s.minimum.unwrap_or(i128::MIN / 4), s.maximum.unwrap_or(i128::MAX / 4));
    let mut scale: i128 = 1;
    for _ in 0..=20 {
        let base = digits.saturating_mul(scale);
        let (a, b) = if neg { (-(base.saturating_add(scale - 1)), -base) } else { (base, base.saturating_add(scale - 1)) };
        if a <= hi && b >= lo {
            return true;
        }
        if base > i128::MAX / 20 {
            break;
        }
        scale = scale.saturating_mul(10);
    }
    false
}

/// Consumes one byte on a thread, the successors pushed to out
fn step(g: &Grammar, mut t: Thread, b: u8, out: &mut Vec<Thread>) {
    let Some(top) = t.stack.pop() else { return };
    let structural = !matches!(top, Frame::Free { .. } | Frame::Str { .. });
    if structural && is_ws(b) {
        if t.ws >= MAX_WS_RUN {
            return;
        }
        t.ws += 1;
    } else if structural {
        t.ws = 0;
    }
    match top {
        Frame::Lit { node, pos } => {
            if node[pos] != b {
                return;
            }
            if pos + 1 == node.len() {
                settle(g, t, out);
            } else {
                t.stack.push(Frame::Lit { node, pos: pos + 1 });
                out.push(t);
            }
        }
        Frame::Free { term, matched } => {
            let mut m = matched;
            loop {
                if term[m] == b {
                    m += 1;
                    break;
                }
                if m == 0 {
                    break;
                }
                m = fallback(&term, m);
            }
            if m == term.len() {
                settle(g, t, out);
            } else {
                t.stack.push(Frame::Free { term, matched: m });
                out.push(t);
            }
        }
        Frame::Space => {
            if is_ws(b) {
                t.stack.push(Frame::Space);
                out.push(t);
            } else {
                let mut settled = Vec::new();
                settle(g, t, &mut settled);
                for s in settled {
                    step(g, s, b, out);
                }
            }
        }
        Frame::Seq { .. } => {
            t.stack.push(top);
            let mut settled = Vec::new();
            settle(g, t, &mut settled);
            for s in settled {
                step(g, s, b, out);
            }
        }
        Frame::Value { schema } => step_value(g, t, schema, b, out),
        Frame::Str { .. } => step_str(g, t, top, b, out),
        Frame::Num { .. } => step_num(g, t, top, b, out),
        Frame::Obj { .. } => step_obj(g, t, top, b, out),
        Frame::Arr { .. } => step_arr(g, t, top, b, out),
        Frame::Word { text, pos } => {
            if text[pos] != b {
                return;
            }
            if pos + 1 == text.len() {
                settle(g, t, out);
            } else {
                t.stack.push(Frame::Word { text, pos: pos + 1 });
                out.push(t);
            }
        }
    }
}

/// The longest proper prefix of the terminator that is also a suffix of its first m bytes
fn fallback(term: &[u8], m: usize) -> usize {
    (1..m).rev().find(|k| term[..*k] == term[m - k..m]).unwrap_or(0)
}

fn step_value(g: &Grammar, mut t: Thread, schema: usize, b: u8, out: &mut Vec<Thread>) {
    let s = &g.schemas[schema];
    if is_ws(b) {
        t.stack.push(Frame::Value { schema });
        out.push(t);
        return;
    }
    if !s.consts.is_empty() {
        for c in &s.consts {
            if c.first() == Some(&b) {
                let mut t2 = t.clone();
                if c.len() == 1 {
                    settle(g, t2, out);
                } else {
                    t2.stack.push(Frame::Lit { node: Arc::new(c.clone()), pos: 1 });
                    out.push(t2);
                }
            }
        }
        return;
    }
    match b {
        b'{' if s.allows(Type::Object) => {
            t.stack.push(Frame::Obj { schema, phase: ObjPhase::Open, seen: vec![false; s.properties.len()], count: 0, prop: None });
            out.push(t);
        }
        b'[' if s.allows(Type::Array) => {
            t.stack.push(Frame::Arr { schema, phase: ArrPhase::Open, count: 0 });
            out.push(t);
        }
        b'"' if s.allows(Type::String) => {
            t.stack.push(Frame::Str { schema, chars: 0, esc: Esc::None, utf8: 0, cands: None, raw: Vec::new(), key: false });
            out.push(t);
        }
        b'-' if s.allows(Type::Number) || s.allows(Type::Integer) => {
            if digits_viable(s, true, 0, false) {
                t.stack.push(Frame::Num { schema, state: NumState::AfterMinus, neg: true, digits: 0, overflow: false });
                out.push(t);
            }
        }
        b'0'..=b'9' if s.allows(Type::Number) || s.allows(Type::Integer) => {
            let d = (b - b'0') as i128;
            if digits_viable(s, false, d, false) {
                t.stack.push(Frame::Num { schema, state: if b == b'0' { NumState::Zero } else { NumState::Int }, neg: false, digits: d, overflow: false });
                out.push(t);
            }
        }
        b't' if s.allows(Type::Boolean) => {
            t.stack.push(Frame::Word { text: b"true", pos: 1 });
            out.push(t);
        }
        b'f' if s.allows(Type::Boolean) => {
            t.stack.push(Frame::Word { text: b"false", pos: 1 });
            out.push(t);
        }
        b'n' if s.allows(Type::Null) => {
            t.stack.push(Frame::Word { text: b"null", pos: 1 });
            out.push(t);
        }
        _ => {}
    }
}

fn step_str(g: &Grammar, mut t: Thread, top: Frame, b: u8, out: &mut Vec<Thread>) {
    let Frame::Str { schema, mut chars, esc, utf8, mut cands, mut raw, key } = top else { return };
    let s = &g.schemas[schema];
    let max = if key { None } else { s.max_length };
    let min = if key { 0 } else { s.min_length };
    let body = |i: usize| -> &[u8] {
        if key {
            &s.properties[i].0
        } else {
            let c = &s.consts[i];
            &c[1..c.len() - 1]
        }
    };
    let closing = esc == Esc::None && utf8 == 0 && b == b'"';
    if let Some(c) = &mut cands
        && !closing
    {
        c.retain(|(i, pos)| body(*i).get(*pos) == Some(&b));
        if c.is_empty() {
            return;
        }
        c.iter_mut().for_each(|(_, pos)| *pos += 1);
    }
    let push = |t: &mut Thread, chars, esc, utf8, cands, raw: Vec<u8>| t.stack.push(Frame::Str { schema, chars, esc, utf8, cands, raw, key });
    if utf8 > 0 {
        if b & 0xC0 != 0x80 {
            return;
        }
        raw.push(b);
        push(&mut t, chars, esc, utf8 - 1, cands, raw);
        out.push(t);
        return;
    }
    match esc {
        Esc::Backslash => {
            let ok = matches!(b, b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' | b'u');
            if !ok {
                return;
            }
            raw.push(b);
            if b == b'u' {
                push(&mut t, chars, Esc::Hex(4), 0, cands, raw);
            } else {
                push(&mut t, chars + 1, Esc::None, 0, cands, raw);
            }
            out.push(t);
        }
        Esc::Hex(n) => {
            if !b.is_ascii_hexdigit() {
                return;
            }
            raw.push(b);
            if n == 1 {
                push(&mut t, chars + 1, Esc::None, 0, cands, raw);
            } else {
                push(&mut t, chars, Esc::Hex(n - 1), 0, cands, raw);
            }
            out.push(t);
        }
        Esc::None => match b {
            b'"' => {
                if chars < min {
                    return;
                }
                if key {
                    let Some(Frame::Obj { schema: os, phase: _, seen, count, .. }) = t.stack.pop() else { return };
                    let obj = &g.schemas[os];
                    let prop = obj.properties.iter().position(|(name, _)| *name == raw);
                    let mut seen = seen;
                    match prop {
                        Some(p) => {
                            if seen[p] {
                                return;
                            }
                            seen[p] = true;
                        }
                        None => {
                            if obj.additional.is_none() {
                                return;
                            }
                        }
                    }
                    t.stack.push(Frame::Obj { schema: os, phase: ObjPhase::AfterKey, seen, count, prop });
                    out.push(t);
                } else {
                    if let Some(c) = &cands
                        && !c.iter().any(|(i, pos)| *pos == body(*i).len())
                    {
                        return;
                    }
                    settle(g, t, out);
                }
            }
            b'\\' => {
                if max.is_some_and(|m| chars >= m) {
                    return;
                }
                raw.push(b);
                push(&mut t, chars, Esc::Backslash, 0, cands, raw);
                out.push(t);
            }
            0x00..=0x1F => {}
            _ => {
                if max.is_some_and(|m| chars >= m) {
                    return;
                }
                chars += 1;
                let cont = if b < 0x80 {
                    0
                } else if b & 0xE0 == 0xC0 {
                    1
                } else if b & 0xF0 == 0xE0 {
                    2
                } else if b & 0xF8 == 0xF0 {
                    3
                } else {
                    return;
                };
                raw.push(b);
                push(&mut t, chars, Esc::None, cont, cands, raw);
                out.push(t);
            }
        },
    }
}

fn step_num(g: &Grammar, mut t: Thread, top: Frame, b: u8, out: &mut Vec<Thread>) {
    let Frame::Num { schema, state, neg, digits, overflow } = top else { return };
    let s = &g.schemas[schema];
    let int_only = s.integer_only();
    let digit = b.is_ascii_digit();
    let push = |t: &mut Thread, state, digits, overflow| t.stack.push(Frame::Num { schema, state, neg, digits, overflow });
    let more_digits = |digits: i128, overflow: bool| -> (i128, bool) {
        let d = (b - b'0') as i128;
        match digits.checked_mul(10).and_then(|v| v.checked_add(d)) {
            Some(v) if v < i128::MAX / 100 => (v, overflow),
            _ => (digits, true),
        }
    };
    match state {
        NumState::AfterMinus => {
            if !digit {
                return;
            }
            let d = (b - b'0') as i128;
            if !digits_viable(s, neg, d, false) {
                return;
            }
            push(&mut t, if b == b'0' { NumState::Zero } else { NumState::Int }, d, false);
            out.push(t);
        }
        NumState::Zero | NumState::Int | NumState::Frac | NumState::ExpDigits => {
            if digit && state == NumState::Int {
                let (d, o) = more_digits(digits, overflow);
                if !digits_viable(s, neg, d, o) {
                    return;
                }
                push(&mut t, NumState::Int, d, o);
                out.push(t);
            } else if digit && state == NumState::Frac {
                push(&mut t, NumState::Frac, digits, overflow);
                out.push(t);
            } else if digit && state == NumState::ExpDigits {
                push(&mut t, NumState::ExpDigits, digits, overflow);
                out.push(t);
            } else if b == b'.' && !int_only && matches!(state, NumState::Zero | NumState::Int) {
                push(&mut t, NumState::Dot, digits, overflow);
                out.push(t);
            } else if (b == b'e' || b == b'E') && !int_only && matches!(state, NumState::Zero | NumState::Int | NumState::Frac) {
                push(&mut t, NumState::Exp, digits, overflow);
                out.push(t);
            } else {
                let f = Frame::Num { schema, state, neg, digits, overflow };
                if !num_closed(g, &f) {
                    return;
                }
                let mut settled = Vec::new();
                settle(g, t, &mut settled);
                for st in settled {
                    step(g, st, b, out);
                }
            }
        }
        NumState::Dot => {
            if digit {
                push(&mut t, NumState::Frac, digits, overflow);
                out.push(t);
            }
        }
        NumState::Exp => {
            if b == b'+' || b == b'-' {
                push(&mut t, NumState::ExpSign, digits, overflow);
                out.push(t);
            } else if digit {
                push(&mut t, NumState::ExpDigits, digits, overflow);
                out.push(t);
            }
        }
        NumState::ExpSign => {
            if digit {
                push(&mut t, NumState::ExpDigits, digits, overflow);
                out.push(t);
            }
        }
    }
}

fn step_obj(g: &Grammar, mut t: Thread, top: Frame, b: u8, out: &mut Vec<Thread>) {
    let Frame::Obj { schema, phase, seen, count, prop } = top else { return };
    let s = &g.schemas[schema];
    if is_ws(b) {
        t.stack.push(Frame::Obj { schema, phase, seen, count, prop });
        out.push(t);
        return;
    }
    let all_required = |seen: &[bool]| s.required.iter().zip(seen).all(|(r, s)| !r || *s);
    let open_key = |t: &mut Thread, seen: Vec<bool>, count: usize| {
        let cands: Option<Vec<(usize, usize)>> = if s.additional.is_none() { Some((0..s.properties.len()).filter(|i| !seen[*i]).map(|i| (i, 0)).collect()) } else { None };
        if cands.as_ref().is_some_and(|c| c.is_empty()) {
            return false;
        }
        t.stack.push(Frame::Obj { schema, phase: ObjPhase::Open, seen, count, prop: None });
        t.stack.push(Frame::Str { schema, chars: 0, esc: Esc::None, utf8: 0, cands, raw: Vec::new(), key: true });
        true
    };
    match (phase, b) {
        (ObjPhase::Open, b'}') if count == 0 && all_required(&seen) => settle(g, t, out),
        (ObjPhase::Open | ObjPhase::AfterComma, b'"') => {
            if open_key(&mut t, seen, count) {
                out.push(t);
            }
        }
        (ObjPhase::AfterKey, b':') => {
            let sub = match prop {
                Some(p) => s.properties[p].1,
                None => s.additional.expect("additional keys are admitted"),
            };
            t.stack.push(Frame::Obj { schema, phase: ObjPhase::AfterColon, seen, count, prop });
            push_value(g, t, sub, out);
        }
        (ObjPhase::AfterValue, b',') => {
            let more = s.additional.is_some() || seen.iter().any(|s| !*s);
            if more {
                t.stack.push(Frame::Obj { schema, phase: ObjPhase::AfterComma, seen, count, prop: None });
                out.push(t);
            }
        }
        (ObjPhase::AfterValue, b'}') if all_required(&seen) => settle(g, t, out),
        _ => {}
    }
}

fn step_arr(g: &Grammar, mut t: Thread, top: Frame, b: u8, out: &mut Vec<Thread>) {
    let Frame::Arr { schema, phase, count } = top else { return };
    let s = &g.schemas[schema];
    if is_ws(b) {
        t.stack.push(Frame::Arr { schema, phase, count });
        out.push(t);
        return;
    }
    let item_schema = |count: usize| -> Option<usize> {
        if count < s.prefix_items.len() {
            Some(s.prefix_items[count])
        } else {
            s.items
        }
    };
    let room = s.max_items.is_none_or(|m| count < m);
    match (phase, b) {
        (ArrPhase::Open, b']') if s.min_items == 0 => settle(g, t, out),
        (ArrPhase::AfterValue, b']') if count >= s.min_items => settle(g, t, out),
        (ArrPhase::AfterValue, b',') if room && item_schema(count).is_some() => {
            t.stack.push(Frame::Arr { schema, phase: ArrPhase::AfterComma, count });
            out.push(t);
        }
        (ArrPhase::Open | ArrPhase::AfterComma, _) if room => {
            let Some(sub) = item_schema(count) else { return };
            t.stack.push(Frame::Arr { schema, phase, count });
            let mut entered = Vec::new();
            push_value(g, t, sub, &mut entered);
            for e in entered {
                step(g, e, b, out);
            }
        }
        _ => {}
    }
}

/// The bytes of every token id, the vocabulary as the acceptor sees it
pub type Pieces = Arc<Vec<Vec<u8>>>;

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn accepts(c: &Constraint, text: &str) -> Option<Constraint> {
        c.try_advance(text.as_bytes())
    }

    fn schema(v: Value) -> Constraint {
        let mut b = GrammarBuilder::new();
        let n = b.json(&v).expect("schema");
        Constraint::new(b.finish(n))
    }

    #[test]
    fn objects_follow_properties_and_required() {
        let c = schema(json!({"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer", "minimum": 0, "maximum": 150}}, "required": ["name"], "additionalProperties": false}));
        assert!(accepts(&c, r#"{"name": "Ann", "age": 42}"#).is_some_and(|c| c.complete()));
        assert!(accepts(&c, r#"{"age": 42}"#).is_none());
        assert!(accepts(&c, r#"{"name": "Ann", "pet": "cat"}"#).is_none());
        assert!(accepts(&c, r#"{"name": "Ann", "age": 151}"#).is_none());
        assert!(accepts(&c, r#"{"name": "Ann", "age": 15"#).is_some_and(|c| !c.complete()));
        assert!(accepts(&c, r#"{"name": "Ann", "age": 1500"#).is_none());
        assert!(accepts(&c, r#"{"name": "An"#).is_some_and(|c| !c.complete()));
        assert!(accepts(&c, r#"{"name": "Ann"}"#).is_some_and(|c| c.complete() && !c.can_continue()));
        assert!(accepts(&c, r#"{"nam"#).is_some());
        assert!(accepts(&c, r#"{"nap"#).is_none());
    }

    #[test]
    fn enums_arrays_and_alternatives() {
        let c = schema(json!({"type": "array", "items": {"enum": ["red", "green", 3, true, null]}, "minItems": 1, "maxItems": 2}));
        assert!(accepts(&c, "[]").is_none());
        assert!(accepts(&c, r#"["red", 3]"#).is_some_and(|c| c.complete()));
        assert!(accepts(&c, r#"["red", 3, true]"#).is_none());
        assert!(accepts(&c, r#"["blue"#).is_none());
        let c = schema(json!({"anyOf": [{"type": "integer"}, {"type": "object", "properties": {"x": {"type": "number"}}, "required": ["x"]}]}));
        assert!(accepts(&c, "-12").is_some_and(|c| c.complete()));
        assert!(accepts(&c, "-12.5").is_none());
        assert!(accepts(&c, r#"{"x": 1.5e3}"#).is_some_and(|c| c.complete()));
        let c = schema(json!({"$defs": {"n": {"type": "object", "properties": {"v": {"type": "integer"}, "next": {"anyOf": [{"$ref": "#/$defs/n"}, {"type": "null"}]}}, "required": ["v"], "additionalProperties": false}}, "$ref": "#/$defs/n"}));
        assert!(accepts(&c, r#"{"v": 1, "next": {"v": 2, "next": null}}"#).is_some_and(|c| c.complete()));
    }

    #[test]
    fn strings_escape_and_utf8() {
        let c = schema(json!({"type": "string", "minLength": 2, "maxLength": 3}));
        assert!(accepts(&c, "\"a\"").is_none());
        assert!(accepts(&c, "\"ab\"").is_some_and(|c| c.complete()));
        assert!(accepts(&c, "\"abcd").is_none());
        assert!(accepts(&c, "\"é\\n\"").is_some_and(|c| c.complete()));
        assert!(accepts(&c, "\"a\\u00e9\"").is_some_and(|c| c.complete()));
        assert!(accepts(&c, "\"a\nb\"").is_none());
        let bad = [b'"', 0xC3, b'"'];
        assert!(c.try_advance(&bad).is_none());
    }

    #[test]
    fn wrapped_tool_call_with_free_thinking() {
        let mut b = GrammarBuilder::new();
        let args = b.json(&json!({"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"], "additionalProperties": false})).expect("args");
        let call = Node::Seq(vec![Node::Literal(b"<tool_call>".to_vec()), Node::Space, Node::Literal(b"{\"name\": \"weather\", \"arguments\": ".to_vec()), args, Node::Space, Node::Literal(b"}".to_vec()), Node::Space, Node::Literal(b"</tool_call>".to_vec())]);
        let root = Node::Seq(vec![Node::Opt(Box::new(Node::Seq(vec![Node::Literal(b"<think>".to_vec()), Node::FreeUntil(b"</think>".to_vec())]))), Node::Space, call]);
        let c = Constraint::new(b.finish(root));
        let out = "<think>Let me see </thin... </think>\n<tool_call>\n{\"name\": \"weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>";
        assert!(accepts(&c, out).is_some_and(|c| c.complete()));
        assert!(accepts(&c, "<tool_call>\n{\"name\": \"weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>").is_some_and(|c| c.complete()));
        assert!(accepts(&c, "<tool_call>\n{\"name\": \"other\"").is_none());
        assert!(accepts(&c, "Sure, ").is_none());
    }

    #[test]
    fn refused_keywords_fail_at_compile() {
        let mut b = GrammarBuilder::new();
        assert!(b.json(&json!({"type": "string", "pattern": "^a"})).is_err());
        assert!(b.json(&json!({"type": "number", "minimum": 1.5})).is_err());
        assert!(b.json(&json!({"allOf": [{"type": "object"}]})).is_err());
    }
}

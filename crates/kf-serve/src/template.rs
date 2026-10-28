use anyhow::{Result, anyhow};
use minijinja::{Environment, Error, ErrorKind, Value, context};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    #[serde(default)]
    pub content: serde_json::Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// The chat template the file carries, rendered the way Hugging Face does
pub struct ChatTemplate {
    env: Environment<'static>,
    pub source: String,
}

impl ChatTemplate {
    pub fn new(source: &str) -> Result<Self> {
        let mut env = Environment::new();
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        minijinja_contrib::add_to_environment(&mut env);
        env.add_function("raise_exception", |msg: String| -> Result<Value, Error> { Err(Error::new(ErrorKind::InvalidOperation, msg)) });
        env.add_function("strftime_now", |fmt: String| chrono::Local::now().format(&fmt).to_string());
        env.add_template_owned("chat", source.to_string()).map_err(|e| anyhow!("template: {e}"))?;
        Ok(Self { env, source: source.to_string() })
    }

    pub fn render(&self, messages: &[Message], tools: Option<&serde_json::Value>, add_generation_prompt: bool, extra: &serde_json::Map<String, serde_json::Value>) -> Result<String> {
        let t = self.env.get_template("chat").map_err(|e| anyhow!("template: {e}"))?;
        let mut ctx = context! {
            messages => Value::from_serialize(messages),
            tools => tools.map(Value::from_serialize).unwrap_or(Value::from(())),
            add_generation_prompt => add_generation_prompt,
            bos_token => "",
            eos_token => "",
        };
        if !extra.is_empty() {
            ctx = context! { ..ctx, ..Value::from_serialize(extra) };
        }
        t.render(ctx).map_err(|e| anyhow!("render: {e}"))
    }
}

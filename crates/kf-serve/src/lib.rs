//! The head, tokenizer and template from the file, OpenAI on the wire, and a self description
pub mod anthropic;
pub mod api;
pub mod chat;
pub mod describe;
pub mod engine;
pub mod grammar;
pub mod params;
pub mod parse;
pub mod responses;
pub mod template;
pub mod tokenizer;

pub use api::serve;
pub use describe::{PROTOCOL_VERSION, STOP_GRACE_SECONDS, describe};
pub use params::{Ctx, Launch, PARAMS, Peer, ROLES, Seat, Values};
pub use parse::{Delta, Format, Parsed, StreamParser, parse_output};
pub use template::{ChatTemplate, Message};
pub use tokenizer::Tok;

use anyhow::{Context, Result, anyhow, bail};
use kf_gguf::Metadata;
use std::collections::{BinaryHeap, HashMap};
use tokenizers::decoders::byte_fallback::ByteFallback;
use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
use tokenizers::decoders::fuse::Fuse;
use tokenizers::decoders::sequence::Sequence as DecoderSequence;
use tokenizers::models::bpe::BPE;
use tokenizers::normalizers::{Prepend, Replace};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::sequence::Sequence;
use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::{AddedToken, DecoderWrapper, NormalizerWrapper, PreTokenizerWrapper, SplitDelimiterBehavior, Tokenizer};

const NORMAL: u32 = 1;
const CONTROL: u32 = 3;
const USER_DEFINED: u32 = 4;
const BYTE: u32 = 6;
const SPACE: &str = "\u{2581}";

/// Token texts the reference treats as end of generation whatever ids the metadata names
const EOG_TEXT: &[&str] = &[
    "<|eot_id|>",
    "<|eot|>",
    "<|im_end|>",
    "<|end|>",
    "<|return|>",
    "<|call|>",
    "<|flush|>",
    "<|calls|>",
    "<end_of_turn>",
    "<|endoftext|>",
    "</s>",
    "<|eom_id|>",
    "<EOT>",
    "_<EOT>",
    "[EOT]",
    "[EOS]",
    "<|end_of_text|>",
    "<end_of_utterance>",
    "<eos>",
    "<turn|>",
    "<|tool_response>",
    "<\u{ff5c}end\u{2581}of\u{2581}sentence\u{ff5c}>",
    "[e~[",
    "<|ifm|im_end|>",
];

/// Token texts the reference detects as fill in the middle pad, repo and file separator markers
const FIM_TEXT: &[&str] = &["<|fim_pad|>", "<fim-pad>", "<fim_pad>", "<PAD>", "[PAD]", "<|fim_repo|>", "<|repo_name|>", "<fim-repo>", "<REPO>", "<reponame>", "<|file_sep|>", "<|plamo:file_separator|>"];

/// Every id the reference stops on, metadata ids plus tokens found by text, minus the harmony end token
fn eog_ids(m: &Metadata, tokens: &[&str], pre: &str) -> Vec<u32> {
    let mut by_text: HashMap<&str, u32> = HashMap::with_capacity(tokens.len());
    for (i, t) in tokens.iter().enumerate() {
        by_text.entry(*t).or_insert(i as u32);
    }
    let keys = ["tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id", "tokenizer.ggml.eom_token_id", "tokenizer.ggml.fim_pad_token_id", "tokenizer.ggml.fim_rep_token_id", "tokenizer.ggml.fim_sep_token_id"];
    let mut eog: Vec<u32> = keys.iter().filter_map(|k| m.u32(k)).collect();
    let mut texts: Vec<&str> = EOG_TEXT.iter().chain(FIM_TEXT).copied().collect();
    if pre == "flint" {
        texts.extend(["<|user|>", "<|observation|>"]);
    }
    eog.extend(texts.iter().filter_map(|t| by_text.get(t).copied()));
    let has = |t: &str| by_text.get(t).is_some_and(|id| eog.contains(id));
    let call = has("<|call|>") || has("<|calls|>");
    let drop_end = has("<|end|>") && call && (has("<|return|>") || has("<|flush|>"));
    if drop_end {
        let end = by_text["<|end|>"];
        eog.retain(|id| *id != end);
    }
    eog.sort_unstable();
    eog.dedup();
    eog
}

enum Inner {
    Bpe(Tokenizer),
    Spm(Spm),
}

/// Tokenizer rebuilt from the vocabulary the GGUF carries
pub struct Tok {
    inner: Inner,
    pub bos: Option<u32>,
    pub eos: Vec<u32>,
    pub add_bos: bool,
    pub model: String,
    pub pre: String,
    /// The bytes each token id adds to the output text
    pieces: Vec<Vec<u8>>,
    /// The leading space of the first decoded token is dropped, the sentencepiece prefix rule
    strip_first_space: bool,
}

/// The byte each byte level BPE character stands for, the inverse of the GPT-2 byte to unicode table
fn byte_level_inverse() -> HashMap<char, u8> {
    let mut map = HashMap::with_capacity(256);
    let keep = |b: u32| (0x21..=0x7E).contains(&b) || (0xA1..=0xAC).contains(&b) || (0xAE..=0xFF).contains(&b);
    let mut n = 0u32;
    for b in 0..256u32 {
        let c = if keep(b) {
            char::from_u32(b).expect("latin char")
        } else {
            n += 1;
            char::from_u32(256 + n - 1).expect("mapped char")
        };
        map.insert(c, b as u8);
    }
    map
}

/// Output bytes per token, byte level characters unmapped, sentencepiece spaces and byte tokens restored, special tokens as their text
fn token_pieces(tokens: &[&str], types: &[u32], byte_level: bool) -> Vec<Vec<u8>> {
    let inverse = byte_level.then(byte_level_inverse);
    tokens
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let ty = types.get(i).copied().unwrap_or(NORMAL);
            if matches!(ty, CONTROL | USER_DEFINED) {
                return t.as_bytes().to_vec();
            }
            if ty == BYTE
                && let Some(b) = t.strip_prefix("<0x").and_then(|s| s.strip_suffix('>')).and_then(|h| u8::from_str_radix(h, 16).ok())
            {
                return vec![b];
            }
            match &inverse {
                Some(inv) => {
                    let mut out = Vec::with_capacity(t.len());
                    for c in t.chars() {
                        match inv.get(&c) {
                            Some(b) => out.push(*b),
                            None => return t.as_bytes().to_vec(),
                        }
                    }
                    out
                }
                None => t.replace(SPACE, " ").into_bytes(),
            }
        })
        .collect()
}

impl Tok {
    pub fn from_gguf(m: &Metadata) -> Result<Self> {
        let model = m.str("tokenizer.ggml.model").unwrap_or("gpt2").to_string();
        let pre = m.str("tokenizer.ggml.pre").unwrap_or("default").to_string();
        let tokens = m.arr_str("tokenizer.ggml.tokens").ok_or_else(|| anyhow!("tokenizer.ggml.tokens missing"))?;
        let types = m.arr_u32("tokenizer.ggml.token_type").unwrap_or_default();
        let (inner, byte_level, strip_first_space) = match model.as_str() {
            "gpt2" => {
                let byte_level = pretokenizer_regexes_for(&pre).with_context(|| format!("pre tokenizer {pre}"))?.byte_encode;
                (Inner::Bpe(byte_level_bpe(m, &tokens, &types, &pre)?), byte_level, false)
            }
            "llama" => {
                let spm = sentencepiece_from_gguf(m, &tokens)?;
                let strip = spm.add_space_prefix;
                (Inner::Spm(spm), false, strip)
            }
            other => bail!("tokenizer model {other} is not supported"),
        };
        let eos = eog_ids(m, &tokens, &pre);
        let add_bos = m.bool("tokenizer.ggml.add_bos_token").unwrap_or(model == "llama");
        let pieces = token_pieces(&tokens, &types, byte_level);
        Ok(Self { inner, bos: m.u32("tokenizer.ggml.bos_token_id"), eos, add_bos, model, pre, pieces, strip_first_space })
    }

    /// The bytes a token adds to the output, the leading space dropped when it is the first token of a sentencepiece output
    pub fn piece(&self, id: u32, first: bool) -> &[u8] {
        let p = self.pieces.get(id as usize).map_or(&[][..], Vec::as_slice);
        if first && self.strip_first_space { p.strip_prefix(b" ").unwrap_or(p) } else { p }
    }

    pub fn encode(&self, text: &str, with_bos: bool) -> Result<Vec<u32>> {
        let mut ids = Vec::new();
        if with_bos && self.add_bos {
            if let Some(b) = self.bos {
                ids.push(b);
            }
        }
        match &self.inner {
            Inner::Bpe(t) => {
                let enc = t.encode(text, false).map_err(|e| anyhow!("encode: {e}"))?;
                ids.extend_from_slice(enc.get_ids());
            }
            Inner::Spm(s) => s.encode(text, &mut ids),
        }
        Ok(ids)
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        match &self.inner {
            Inner::Bpe(t) => t.decode(ids, false).map_err(|e| anyhow!("decode: {e}")),
            Inner::Spm(s) => Ok(s.decode(ids)),
        }
    }

    pub fn is_eos(&self, id: u32) -> bool {
        self.eos.contains(&id)
    }

    pub fn vocab_size(&self) -> usize {
        match &self.inner {
            Inner::Bpe(t) => t.get_vocab_size(true),
            Inner::Spm(s) => s.tokens.len(),
        }
    }
}

/// Split regexes applied in order and whether pieces are byte encoded before the merges
pub struct PreTok {
    pub regexes: Vec<&'static str>,
    pub byte_encode: bool,
}

fn byte_level_bpe(m: &Metadata, tokens: &[&str], types: &[u32], pre: &str) -> Result<Tokenizer> {
    let merges_raw = m.arr_str("tokenizer.ggml.merges").ok_or_else(|| anyhow!("tokenizer.ggml.merges missing"))?;
    let vocab: tokenizers::models::bpe::Vocab = tokens.iter().enumerate().map(|(i, t)| (t.to_string(), i as u32)).collect();
    let merges: Vec<(String, String)> = merges_raw
        .iter()
        .map(|mline| {
            let (a, b) = mline.split_once(' ').ok_or_else(|| anyhow!("merge without a space: {mline}"))?;
            Ok((a.to_string(), b.to_string()))
        })
        .collect::<Result<_>>()?;
    let unk = tokens.iter().position(|t| *t == "<unk>").map(|i| tokens[i].to_string());
    let mut builder = BPE::builder().vocab_and_merges(vocab, merges).ignore_merges(true);
    if let Some(u) = unk {
        builder = builder.unk_token(u);
    }
    let bpe = builder.build().map_err(|e| anyhow!("bpe: {e}"))?;
    let mut tok = Tokenizer::new(bpe);
    let pt = pretokenizer_regexes_for(pre).with_context(|| format!("pre tokenizer {pre}"))?;
    let mut stages: Vec<PreTokenizerWrapper> = Vec::new();
    for pattern in pt.regexes {
        let split = Split::new(SplitPattern::Regex(onig_syntax(pattern)), SplitDelimiterBehavior::Isolated, false).map_err(|e| anyhow!("split: {e}"))?;
        stages.push(PreTokenizerWrapper::Split(split));
    }
    if pt.byte_encode {
        stages.push(PreTokenizerWrapper::ByteLevel(ByteLevel::new(false, false, false)));
        tok.with_decoder(Some(ByteLevelDecoder::default()));
    } else {
        let mut norms: Vec<NormalizerWrapper> = Vec::new();
        if m.bool("tokenizer.ggml.add_space_prefix").unwrap_or(false) {
            norms.push(NormalizerWrapper::Prepend(Prepend::new(SPACE.to_string())));
        }
        norms.push(NormalizerWrapper::Replace(Replace::new(" ", SPACE).map_err(|e| anyhow!("replace: {e}"))?));
        tok.with_normalizer(Some(tokenizers::normalizers::Sequence::new(norms)));
        let decoders: Vec<DecoderWrapper> = vec![
            DecoderWrapper::Replace(Replace::new(SPACE, " ").map_err(|e| anyhow!("replace: {e}"))?),
            DecoderWrapper::ByteFallback(ByteFallback::new()),
            DecoderWrapper::Fuse(Fuse::new()),
        ];
        tok.with_decoder(Some(DecoderSequence::new(decoders)));
    }
    tok.with_pre_tokenizer(Some(Sequence::new(stages)));
    let specials: Vec<AddedToken> = tokens.iter().enumerate().filter(|(i, _)| matches!(types.get(*i), Some(&CONTROL) | Some(&USER_DEFINED))).map(|(_, t)| AddedToken::from(t.to_string(), true)).collect();
    tok.add_special_tokens(&specials);
    Ok(tok)
}

/// Rewrites the escapes of the reference regex dialect that the oniguruma engine spells differently
fn onig_syntax(pattern: &str) -> String {
    pattern.replace("\\u200C", "\\x{200C}").replace("\\u200D", "\\x{200D}")
}

const LLAMA3: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const GPT2: &str = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)";
const QWEN2: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const QWEN35: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const DIGITS3: &str = r"\p{N}{1,3}";
const CJK: &str = r"[一-龥぀-ゟ゠-ヿ]+";
const DEEPSEEK3: &str = "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+";
const SPARK: &str = "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+|[\r\n]|\\s+(?!\\S)|\\s+";
const TEKKEN: &str = r"[^\r\n\p{L}\p{N}]?((?=[\p{L}\p{M}])([^a-z]))*((?=[\p{L}\p{M}])([^A-Z]))+|[^\r\n\p{L}\p{N}]?((?=[\p{L}\p{M}])([^a-z]))+((?=[\p{L}\p{M}])([^A-Z]))*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const GPT4O: &str = r"[^\r\n\p{L}\p{N}]?((?=[\p{L}\p{M}])([^a-z]))*((?=[\p{L}\p{M}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\r\n\p{L}\p{N}]?((?=[\p{L}\p{M}])([^a-z]))+((?=[\p{L}\p{M}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const TINY_AYA_WORDS: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const NEWLINES: &str = r"[^\n]+|[\n]+";
const DEEPSEEK_LLM_LETTERS: &str = "\\s?[A-Za-zµÀ-ÖØ-öø-ƺƼ-ƿǄ-ʓʕ-ʯͰ-ͳͶͷͻ-ͽͿΆΈ-ΊΌΎ-ΡΣ-ϵϷ-ҁҊ-ԯԱ-ՖႠ-ჅᎠ-Ᏽᏸ-ᏽᲐ-ᲺᲽ-Ჿᴀ-ᴫᵫ-ᵷᵹ-ᶚḀ-ἕἘ-Ἕἠ-ὅὈ-Ὅὐ-ὗὙὛὝὟ-ώᾀ-ᾴᾶ-ᾼιῂ-ῄῆ-ῌῐ-ΐῖ-Ίῠ-Ῥῲ-ῴῶ-ῼℂℇℊ-ℓℕℙ-ℝℤΩℨK-ℭℯ-ℴℹℼ-ℿⅅ-ⅉⅎↃↄⰀ-ⱻⱾ-ⳤⳫ-ⳮⳲⳳꙀ-ꙭꚀ-ꚛꜢ-ꝯꝱ-ꞇꞋ-ꞎꭰ-ꮿﬀ-ﬆﬓ-ﬗＡ-Ｚａ-ｚ𐐀-𐑏𐒰-𐓓𐓘-𐓻𐲀-𐲲𐳀-𐳲𑢠-𑣟𞤀-𞥃]+";

/// The split patterns the reference applies in order for every pre tokenizer name it knows
pub fn pretokenizer_regexes_for(pre: &str) -> Result<PreTok> {
    let plain = |v: Vec<&'static str>| Ok(PreTok { regexes: v, byte_encode: true });
    let raw = |v: Vec<&'static str>| Ok(PreTok { regexes: v, byte_encode: false });
    match pre {
        "" | "default" => plain(vec![r"[\p{P}\$\+<=>\^~\|]+", GPT2, r"\p{N}+", "[0-9][0-9][0-9]"]),
        "flint" => plain(vec![
            DIGITS3,
            CJK,
            r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        ]),
        "minicpm5" => plain(vec![DIGITS3, r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}+| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"]),
        "llama3" | "llama-v3" | "llama-bpe" | "falcon3" | "falcon-h1" | "midm-2.0" | "jina-v5-nano" | "lfm2" | "dbrx" | "smaug-bpe" | "chatglm-bpe" | "glm4" | "glm5" => plain(vec![LLAMA3]),
        "jais-2" => plain(vec![
            r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s{512}(?!\S)|\s{256}(?!\S)|\s{128}(?!\S)|\s{64}(?!\S)|\s{32}(?!\S)|\s{16}(?!\S)|\s{8}(?!\S)|\s{4}(?!\S)|\s{1,2}(?!\S)|\s{1}",
        ]),
        "deepseek-llm" => plain(vec!["[\r\n]", DEEPSEEK_LLM_LETTERS, "\\s?[!-/:-~！-／：-～‘-‟　-。]+", r"\s+$", "[一-龥ࠀ-一가-퟿]+", r"\p{N}+"]),
        "deepseek-v3" | "hunyuan-dense" | "joyai-llm" | "hy_v4" => plain(vec![DIGITS3, CJK, DEEPSEEK3]),
        "spark2_5" => plain(vec![DIGITS3, CJK, SPARK, r"\p{N}"]),
        "youtu" => plain(vec!["[가-힣ㄱ-ㆎ]+|[！…“”‘’—：；，、-〿︰-﹏]+|[ㄅ-ㄯ]+|[一-龥぀-ゟ゠-ヿ]+", r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"]),
        "deepseek-coder" => plain(vec!["[\r\n]", r"\s?\p{L}+", r"\s?\p{P}+", "[一-龥ࠀ-一가-퟿]+", r"\p{N}"]),
        "falcon" => plain(vec![r"[\p{P}\$\+<=>\^~\|`]+", GPT2, "[0-9][0-9][0-9]"]),
        "starcoder" | "refact" | "command-r" | "smollm" | "codeshell" | "exaone" | "minerva-7b" | "mellum2" => plain(vec![r"\p{N}", GPT2]),
        "gpt-2" | "phi-2" | "jina-es" | "jina-de" | "gigachat" | "jina-v2-es" | "jina-v2-de" | "a.x-4.0" | "mellum" | "modern-bert" | "jina-v1-en" | "jina-v2-code" | "roberta-bpe" | "mpt" | "olmo" | "jais" | "trillion" | "granite-docling" | "exaone4" => plain(vec![GPT2]),
        "qwen2" | "deepseek-r1-qwen" | "kormo" | "f2llmv2" | "stablelm2" | "hunyuan" | "solar-open" | "megrez" => plain(vec![QWEN2]),
        "qwen35" => plain(vec![QWEN35]),
        "poro-chat" | "bloom" | "gpt3-finnish" => plain(vec![" ?[^(\\s|.,!?…。，、।۔،)]+"]),
        "viking" => plain(vec![" ?[^(\\s|.,!?…。，、।۔،)]+", r"\p{N}"]),
        "tekken" | "pixtral" | "nemotron-3.5-embed" => plain(vec![TEKKEN]),
        "chameleon" => plain(vec!["<sentinel:[0-9]+>", "(IMGIMG)((A|B|C|D|E|F|G|H|I){1,4})Z", "([\\t\\n]|    |  )", r"\p{N}", r"[\p{P}!-/:-@\[-`{-~]", GPT2]),
        "gpt-4o" | "llama4" | "kanana2" | "talkie" | "minimax-m2" | "granite-embed-multi-97m" => plain(vec![GPT4O]),
        "tiny_aya" | "cohere2moe" => plain(vec![r"\d{1,3}(?=(?:\d{3})*\b)", TINY_AYA_WORDS]),
        "kimi-k2" => plain(vec![
            r"[\p{Han}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        ]),
        "superbpe" => plain(vec![r"\p{N}+", r"(?=(\d{3})+(?!\d))"]),
        "bailingmoe" | "bailingmoe2" | "llada-moe" => plain(vec![r"'(?:[sSdDmMtT]|[lL][lL]|[vV][eE]|[rR][eE])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"]),
        "seed-coder" => plain(vec![r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1}| ?[^\s\p{L}\p{N}\r\n]+|\s*[\r\n]+|\s+(?!\S)|\s+"]),
        "ufakzeka" => plain(vec![r"[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"]),
        "grok-2" => plain(vec![QWEN2]),
        "afmoe" => plain(vec![
            r"\p{N}{1,2}(?=(?:\p{N}{3})+(?!\p{N}))|\p{N}{3}",
            "[一-鿿㐀-䶿豈-﫿぀-ゟ゠-ヿ･-ﾟ⼀-⿟เ-๿຀-໿ក-៿က-႟ꩠ-ꩿꧠ-꧿가-힯ᄀ-ᇿ]+",
            "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\\r\\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        ]),
        "laguna" => plain(vec![NEWLINES, QWEN2]),
        "exaone-moe" => plain(vec![r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?(?:\p{L}\p{M}*(?: \p{L}\p{M}*)*)+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]?|\s*[\r\n]|\s+(?!\S)|\s+"]),
        "k2-horizon" => plain(vec![r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?(?:\p{L}|\p{M}|\u200C|\u200D)+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"]),
        "gemma4" | "granite-embed-multi-311m" | "sarvam-moe" | "domyn-small" | "mmbert" => raw(vec![NEWLINES]),
        "soprano" => raw(vec![r"\p{N}", r"\s+|[\p{L}\p{N}_]+|[^\p{L}\p{N}_\s]+"]),
        "whitespace" => raw(vec![r"\S+"]),
        other => bail!("unknown pre tokenizer type {other}"),
    }
}

/// The SentencePiece vocabulary with the reference's greedy merge by score and byte fallback
pub struct Spm {
    tokens: Vec<String>,
    scores: Vec<f32>,
    types: Vec<u32>,
    ids: HashMap<String, u32>,
    bytes: [Option<u32>; 256],
    specials: Vec<(String, u32)>,
    add_space_prefix: bool,
}

#[derive(PartialEq)]
struct Bigram {
    score: f32,
    left: usize,
    right: usize,
    size: usize,
}

impl Eq for Bigram {}

impl Ord for Bigram {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.score.partial_cmp(&o.score).unwrap_or(std::cmp::Ordering::Equal).then(o.left.cmp(&self.left))
    }
}

impl PartialOrd for Bigram {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}

/// The llama tokenizer model from tokens, scores and types, with byte fallback and the space prefix rule
pub fn sentencepiece_from_gguf(m: &Metadata, tokens: &[&str]) -> Result<Spm> {
    let scores = m.arr_f32("tokenizer.ggml.scores").ok_or_else(|| anyhow!("tokenizer.ggml.scores missing"))?;
    let types = m.arr_u32("tokenizer.ggml.token_type").ok_or_else(|| anyhow!("tokenizer.ggml.token_type missing"))?;
    if scores.len() != tokens.len() || types.len() != tokens.len() {
        bail!("tokenizer arrays disagree on the vocabulary size");
    }
    let ids: HashMap<String, u32> = tokens.iter().enumerate().map(|(i, t)| (t.to_string(), i as u32)).collect();
    let mut bytes = [None; 256];
    for b in 0..256usize {
        bytes[b] = ids.get(&format!("<0x{b:02X}>")).copied();
    }
    let mut specials: Vec<(String, u32)> = tokens.iter().enumerate().filter(|(i, _)| matches!(types[*i], CONTROL | USER_DEFINED)).map(|(i, t)| (t.to_string(), i as u32)).collect();
    specials.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then(a.1.cmp(&b.1)));
    Ok(Spm { tokens: tokens.iter().map(|t| t.to_string()).collect(), scores, types, ids, bytes, specials, add_space_prefix: m.bool("tokenizer.ggml.add_space_prefix").unwrap_or(true) })
}

impl Spm {
    /// Longest special tokens first, the text between them handed to the merge tokenizer
    fn encode(&self, text: &str, out: &mut Vec<u32>) {
        let mut pieces: Vec<Result<&str, u32>> = vec![Ok(text)];
        for (special, id) in &self.specials {
            let mut next = Vec::with_capacity(pieces.len());
            for p in pieces {
                match p {
                    Err(t) => next.push(Err(t)),
                    Ok(s) => {
                        let mut rest = s;
                        while let Some(at) = rest.find(special.as_str()) {
                            if at > 0 {
                                next.push(Ok(&rest[..at]));
                            }
                            next.push(Err(*id));
                            rest = &rest[at + special.len()..];
                        }
                        if !rest.is_empty() {
                            next.push(Ok(rest));
                        }
                    }
                }
            }
            pieces = next;
        }
        let mut prev_special = true;
        for p in pieces {
            match p {
                Err(t) => {
                    out.push(t);
                    prev_special = true;
                }
                Ok(s) => {
                    let mut t = String::with_capacity(s.len() + 3);
                    if self.add_space_prefix && prev_special {
                        t.push(' ');
                    }
                    t.push_str(s);
                    let escaped = t.replace(' ', SPACE);
                    self.merge(&escaped, out);
                    prev_special = false;
                }
            }
        }
    }

    /// Greedy bigram merges by score over utf8 characters, unmerged leftovers emitted as byte tokens
    fn merge(&self, text: &str, out: &mut Vec<u32>) {
        let mut syms: Vec<(usize, usize, isize, isize)> = Vec::new();
        let mut i = 0;
        let mut index = 0isize;
        while i < text.len() {
            let n = text[i..].chars().next().map_or(1, char::len_utf8);
            let next = if i + n >= text.len() { -1 } else { index + 1 };
            syms.push((i, n, index - 1, next));
            i += n;
            index += 1;
        }
        let mut queue = BinaryHeap::new();
        let mut rev: HashMap<String, (usize, usize)> = HashMap::new();
        let add = |queue: &mut BinaryHeap<Bigram>, rev: &mut HashMap<String, (usize, usize)>, syms: &[(usize, usize, isize, isize)], l: isize, r: isize| {
            if l < 0 || r < 0 {
                return;
            }
            let (ls, rs) = (syms[l as usize], syms[r as usize]);
            let piece = &text[ls.0..ls.0 + ls.1 + rs.1];
            if let Some(&id) = self.ids.get(piece) {
                queue.push(Bigram { score: self.scores[id as usize], left: l as usize, right: r as usize, size: piece.len() });
                rev.insert(piece.to_string(), (l as usize, r as usize));
            }
        };
        for k in 1..syms.len() {
            add(&mut queue, &mut rev, &syms, k as isize - 1, k as isize);
        }
        while let Some(b) = queue.pop() {
            let (l, r) = (b.left, b.right);
            if syms[l].1 == 0 || syms[r].1 == 0 || syms[l].1 + syms[r].1 != b.size {
                continue;
            }
            syms[l].1 += syms[r].1;
            syms[r].1 = 0;
            syms[l].3 = syms[r].3;
            let rn = syms[r].3;
            if rn >= 0 {
                syms[rn as usize].2 = l as isize;
            }
            add(&mut queue, &mut rev, &syms, syms[l].2, l as isize);
            add(&mut queue, &mut rev, &syms, l as isize, syms[l].3);
        }
        let mut k = 0isize;
        while k >= 0 {
            let s = syms[k as usize];
            self.resegment(text, &syms, s, &rev, out);
            k = s.3;
        }
    }

    fn resegment(&self, text: &str, syms: &[(usize, usize, isize, isize)], sym: (usize, usize, isize, isize), rev: &HashMap<String, (usize, usize)>, out: &mut Vec<u32>) {
        let mut todo = vec![sym];
        while let Some(cur) = todo.pop() {
            let piece = &text[cur.0..cur.0 + cur.1];
            if let Some(&id) = self.ids.get(piece) {
                out.push(id);
                continue;
            }
            match rev.get(piece) {
                Some(&(l, r)) => {
                    todo.push(syms[r]);
                    todo.push(syms[l]);
                }
                None => {
                    for b in piece.bytes() {
                        if let Some(id) = self.bytes[b as usize] {
                            out.push(id);
                        }
                    }
                }
            }
        }
    }

    /// Pieces joined with escaped spaces restored and byte tokens merged, the leading space dropped
    fn decode(&self, ids: &[u32]) -> String {
        let mut bytes: Vec<u8> = Vec::new();
        for (k, id) in ids.iter().enumerate() {
            let Some(t) = self.tokens.get(*id as usize) else { continue };
            match self.types.get(*id as usize).copied().unwrap_or(NORMAL) {
                BYTE => {
                    if let Some(b) = t.strip_prefix("<0x").and_then(|s| s.strip_suffix('>')).and_then(|h| u8::from_str_radix(h, 16).ok()) {
                        bytes.push(b);
                    }
                }
                CONTROL | USER_DEFINED => bytes.extend_from_slice(t.as_bytes()),
                _ => {
                    let mut piece = t.replace(SPACE, " ");
                    if k == 0 && self.add_space_prefix {
                        piece = piece.strip_prefix(' ').map(str::to_string).unwrap_or(piece);
                    }
                    bytes.extend_from_slice(piece.as_bytes());
                }
            }
        }
        match std::str::from_utf8(&bytes) {
            Ok(s) => s.to_string(),
            Err(e) => String::from_utf8_lossy(&bytes[..e.valid_up_to()]).into_owned(),
        }
    }
}

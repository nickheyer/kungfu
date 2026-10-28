//! GGUF v3 reader over memory mapped files, split shards included
use anyhow::{Context, Result, anyhow, bail};
use memmap2::Mmap;
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 4] = b"GGUF";
const DEFAULT_ALIGNMENT: usize = 32;

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    Arr(Vec<Value>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::U8(v) => Some(*v as u64),
            Value::U16(v) => Some(*v as u64),
            Value::U32(v) => Some(*v as u64),
            Value::U64(v) => Some(*v),
            Value::I8(v) if *v >= 0 => Some(*v as u64),
            Value::I16(v) if *v >= 0 => Some(*v as u64),
            Value::I32(v) if *v >= 0 => Some(*v as u64),
            Value::I64(v) if *v >= 0 => Some(*v as u64),
            Value::Bool(v) => Some(*v as u64),
            _ => None,
        }
    }
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::I8(v) => Some(*v as i64),
            Value::I16(v) => Some(*v as i64),
            Value::I32(v) => Some(*v as i64),
            Value::I64(v) => Some(*v),
            other => other.as_u64().map(|v| v as i64),
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::F32(v) => Some(*v as f64),
            Value::F64(v) => Some(*v),
            other => other.as_i64().map(|v| v as f64),
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_arr(&self) -> Option<&[Value]> {
        match self {
            Value::Arr(a) => Some(a),
            _ => None,
        }
    }
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            other => other.as_u64().map(|v| v != 0),
        }
    }
}

/// Typed access to the key value header
#[derive(Clone, Debug, Default)]
pub struct Metadata {
    map: BTreeMap<String, Value>,
}

impl Metadata {
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.map.get(key)
    }
    pub fn has(&self, key: &str) -> bool {
        self.map.contains_key(key)
    }
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.map.keys().map(String::as_str)
    }
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Value)> {
        self.map.iter().map(|(k, v)| (k.as_str(), v))
    }
    pub fn str(&self, key: &str) -> Option<&str> {
        self.get(key).and_then(Value::as_str)
    }
    pub fn u64(&self, key: &str) -> Option<u64> {
        self.get(key).and_then(Value::as_u64)
    }
    pub fn usize(&self, key: &str) -> Option<usize> {
        self.u64(key).map(|v| v as usize)
    }
    pub fn u32(&self, key: &str) -> Option<u32> {
        self.u64(key).map(|v| v as u32)
    }
    pub fn i64(&self, key: &str) -> Option<i64> {
        self.get(key).and_then(Value::as_i64)
    }
    pub fn f32(&self, key: &str) -> Option<f32> {
        self.get(key).and_then(Value::as_f64).map(|v| v as f32)
    }
    pub fn bool(&self, key: &str) -> Option<bool> {
        self.get(key).and_then(Value::as_bool)
    }
    pub fn arr(&self, key: &str) -> Option<&[Value]> {
        self.get(key).and_then(Value::as_arr)
    }
    pub fn arr_u32(&self, key: &str) -> Option<Vec<u32>> {
        self.arr(key)?.iter().map(|v| v.as_u64().map(|x| x as u32)).collect()
    }
    pub fn arr_usize(&self, key: &str) -> Option<Vec<usize>> {
        self.arr(key)?.iter().map(|v| v.as_u64().map(|x| x as usize)).collect()
    }
    pub fn arr_f32(&self, key: &str) -> Option<Vec<f32>> {
        self.arr(key)?.iter().map(|v| v.as_f64().map(|x| x as f32)).collect()
    }
    pub fn arr_str(&self, key: &str) -> Option<Vec<&str>> {
        self.arr(key)?.iter().map(Value::as_str).collect()
    }
    pub fn arr_len(&self, key: &str) -> Option<usize> {
        self.arr(key).map(<[Value]>::len)
    }
    pub fn arch(&self) -> Option<&str> {
        self.str("general.architecture")
    }
    pub fn insert(&mut self, key: String, value: Value) {
        self.map.insert(key, value);
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorInfo {
    pub name: String,
    /// Innermost dimension first, as ggml stores ne
    pub dims: Vec<usize>,
    pub ggml_type: u32,
    pub offset: u64,
    pub shard: usize,
}

impl TensorInfo {
    pub fn numel(&self) -> usize {
        self.dims.iter().product()
    }
    pub fn nbytes(&self) -> Result<usize> {
        kf_quant::nbytes(self.ggml_type, &self.dims)
            .ok_or_else(|| anyhow!("{}: ggml type {} does not fit dims {:?}", self.name, self.ggml_type, self.dims))
    }
}

struct Shard {
    path: PathBuf,
    mmap: Mmap,
    data_start: usize,
}

pub struct Gguf {
    pub meta: Metadata,
    pub tensors: Vec<TensorInfo>,
    by_name: HashMap<String, usize>,
    shards: Vec<Shard>,
}

impl Gguf {
    /// Opens a file and every sibling shard its name implies
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let paths = shard_paths(path)?;
        let mut meta = Metadata::default();
        let mut tensors = Vec::new();
        let mut shards = Vec::new();
        for (i, p) in paths.iter().enumerate() {
            let file = File::open(p).with_context(|| format!("open {}", p.display()))?;
            let mmap = unsafe { Mmap::map(&file) }.with_context(|| format!("map {}", p.display()))?;
            let parsed = parse_header(&mmap).with_context(|| format!("parse {}", p.display()))?;
            if i == 0 {
                meta = parsed.meta;
            }
            for mut t in parsed.tensors {
                t.shard = i;
                tensors.push(t);
            }
            shards.push(Shard { path: p.clone(), mmap, data_start: parsed.data_start });
        }
        let by_name = tensors.iter().enumerate().map(|(i, t)| (t.name.clone(), i)).collect();
        Ok(Self { meta, tensors, by_name, shards })
    }

    pub fn paths(&self) -> Vec<&Path> {
        self.shards.iter().map(|s| s.path.as_path()).collect()
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.by_name.get(name).map(|&i| &self.tensors[i])
    }

    /// Raw bytes of one tensor inside its shard
    pub fn bytes(&self, t: &TensorInfo) -> Result<&[u8]> {
        let shard = &self.shards[t.shard];
        let start = shard.data_start + t.offset as usize;
        let end = start + t.nbytes()?;
        shard.mmap.get(start..end).ok_or_else(|| anyhow!("{}: data past end of {}", t.name, shard.path.display()))
    }

    pub fn total_bytes(&self) -> u64 {
        self.tensors.iter().filter_map(|t| t.nbytes().ok()).map(|n| n as u64).sum()
    }
}

/// Sibling shards of a split file, or the file alone
fn shard_paths(path: &Path) -> Result<Vec<PathBuf>> {
    let name = path.file_name().and_then(|n| n.to_str()).ok_or_else(|| anyhow!("bad path {}", path.display()))?;
    let Some((stem, rest)) = name.rsplit_once("-of-") else { return Ok(vec![path.to_path_buf()]) };
    let Some(total) = rest.strip_suffix(".gguf").and_then(|s| s.parse::<usize>().ok()) else {
        return Ok(vec![path.to_path_buf()]);
    };
    let Some((base, idx)) = stem.rsplit_once('-') else { return Ok(vec![path.to_path_buf()]) };
    if idx.len() != 5 || idx.parse::<usize>().is_err() {
        return Ok(vec![path.to_path_buf()]);
    }
    let dir = path.parent().unwrap_or(Path::new("."));
    Ok((1..=total).map(|i| dir.join(format!("{base}-{i:05}-of-{total:05}.gguf"))).collect())
}

struct Parsed {
    meta: Metadata,
    tensors: Vec<TensorInfo>,
    data_start: usize,
}

struct Cursor<'a> {
    buf: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let s = self.buf.get(self.at..self.at + n).ok_or_else(|| anyhow!("truncated header at {}", self.at))?;
        self.at += n;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into()?))
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into()?))
    }
    fn string(&mut self) -> Result<String> {
        let n = self.u64()? as usize;
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
    fn value(&mut self, ty: u32) -> Result<Value> {
        Ok(match ty {
            0 => Value::U8(self.u8()?),
            1 => Value::I8(self.u8()? as i8),
            2 => Value::U16(self.u16()?),
            3 => Value::I16(self.u16()? as i16),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(self.u32()? as i32),
            6 => Value::F32(f32::from_bits(self.u32()?)),
            7 => Value::Bool(self.u8()? != 0),
            8 => Value::Str(self.string()?),
            9 => {
                let ety = self.u32()?;
                let n = self.u64()? as usize;
                let mut items = Vec::with_capacity(n.min(1 << 20));
                for _ in 0..n {
                    items.push(self.value(ety)?);
                }
                Value::Arr(items)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(self.u64()? as i64),
            12 => Value::F64(f64::from_bits(self.u64()?)),
            other => bail!("unknown metadata value type {other}"),
        })
    }
}

fn parse_header(buf: &[u8]) -> Result<Parsed> {
    let mut c = Cursor { buf, at: 0 };
    if c.take(4)? != MAGIC {
        bail!("not a GGUF file");
    }
    let version = c.u32()?;
    if version != 3 && version != 2 {
        bail!("unsupported GGUF version {version}");
    }
    let n_tensors = c.u64()? as usize;
    let n_kv = c.u64()? as usize;
    let mut meta = Metadata::default();
    for _ in 0..n_kv {
        let key = c.string()?;
        let ty = c.u32()?;
        let value = c.value(ty)?;
        meta.insert(key, value);
    }
    let mut tensors = Vec::with_capacity(n_tensors);
    for _ in 0..n_tensors {
        let name = c.string()?;
        let n_dims = c.u32()? as usize;
        let mut dims = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            dims.push(c.u64()? as usize);
        }
        let ggml_type = c.u32()?;
        let offset = c.u64()?;
        tensors.push(TensorInfo { name, dims, ggml_type, offset, shard: 0 });
    }
    let alignment = meta.usize("general.alignment").unwrap_or(DEFAULT_ALIGNMENT);
    let data_start = c.at.div_ceil(alignment) * alignment;
    Ok(Parsed { meta, tensors, data_start })
}

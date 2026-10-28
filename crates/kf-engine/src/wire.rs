//! The chain protocol between a head and its stages, the ggml rpc hello first and kungfu frames after
use anyhow::{Result, bail, ensure};
use std::io::{Read, Write};

/// The ggml rpc hello command byte a manager sends to check a stage
pub const HELLO: u8 = 14;
/// Bytes of capability block the hello carries
pub const HELLO_CAPS: u64 = 24;
/// Protocol version the hello answers, major minor patch
pub const PROTOCOL: [u8; 3] = [1, 0, 0];

pub const CMD_RANGE: u8 = 0x20;
pub const CMD_FORWARD: u8 = 0x21;
pub const CMD_TRUNCATE: u8 = 0x22;
pub const CMD_RESET: u8 = 0x23;
/// The head's rdma parameters, answered with the stage's or with nothing when the stage stays on tcp
pub const CMD_RDMA: u8 = 0x24;
/// The first frame over the rdma link, answered with nothing
pub const CMD_PING: u8 = 0x25;
/// The head's word that the link stays on tcp
pub const CMD_TCP: u8 = 0x26;
/// The slot value a reset names to drop every slot
pub const ALL_SEQS: u32 = u32::MAX;

/// Most bytes one frame may carry, activations of a batch of the widest model
pub const MAX_FRAME: u64 = 1 << 32;

fn read_u64(r: &mut impl Read) -> Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_payload(r: &mut impl Read) -> Result<Vec<u8>> {
    let n = read_u64(r)?;
    ensure!(n <= MAX_FRAME, "frame of {n} bytes");
    let mut v = vec![0u8; n as usize];
    r.read_exact(&mut v)?;
    Ok(v)
}

/// Sends a command frame, the command byte, the payload length, and the payload
pub fn send(w: &mut impl Write, cmd: u8, payload: &[u8]) -> Result<()> {
    w.write_all(&[cmd])?;
    w.write_all(&(payload.len() as u64).to_le_bytes())?;
    w.write_all(payload)?;
    w.flush()?;
    Ok(())
}

/// Reads the next command byte, none at a clean end of stream
pub fn recv_cmd(r: &mut impl Read) -> Result<Option<u8>> {
    let mut b = [0u8; 1];
    match r.read(&mut b)? {
        0 => Ok(None),
        _ => Ok(Some(b[0])),
    }
}

/// Reads the rest of a command frame after its byte
pub fn recv_payload(r: &mut impl Read) -> Result<Vec<u8>> {
    read_payload(r)
}

/// Answers the hello after its command byte was read, the capability block consumed and the version returned
pub fn answer_hello(s: &mut (impl Read + Write)) -> Result<()> {
    let caps = read_payload(s)?;
    ensure!(caps.len() as u64 == HELLO_CAPS, "hello carried {} bytes of capabilities", caps.len());
    s.write_all(&(PROTOCOL.len() as u64).to_le_bytes())?;
    s.write_all(&PROTOCOL)?;
    s.flush()?;
    Ok(())
}

/// Makes the hello exchange from the head's side and returns the version the stage speaks
pub fn hello(s: &mut (impl Read + Write)) -> Result<[u8; 3]> {
    s.write_all(&[HELLO])?;
    s.write_all(&HELLO_CAPS.to_le_bytes())?;
    s.write_all(&[0u8; HELLO_CAPS as usize])?;
    s.flush()?;
    let v = read_payload(s)?;
    ensure!(v.len() >= 3, "hello answered {} bytes, not a version", v.len());
    Ok([v[0], v[1], v[2]])
}

/// Sends a reply frame, a status byte then the payload or the error text
pub fn reply(w: &mut impl Write, result: Result<Vec<u8>>) -> Result<()> {
    let (status, body) = match result {
        Ok(v) => (0u8, v),
        Err(e) => (1u8, format!("{e:#}").into_bytes()),
    };
    w.write_all(&[status])?;
    w.write_all(&(body.len() as u64).to_le_bytes())?;
    w.write_all(&body)?;
    w.flush()?;
    Ok(())
}

/// Reads a reply frame, the payload on success and the stage's error text as an error
pub fn recv_reply(r: &mut impl Read) -> Result<Vec<u8>> {
    let mut status = [0u8; 1];
    r.read_exact(&mut status)?;
    let body = read_payload(r)?;
    if status[0] != 0 {
        bail!("stage: {}", String::from_utf8_lossy(&body));
    }
    Ok(body)
}

/// Little endian u32 values appended to a buffer
pub fn put_u32s(out: &mut Vec<u8>, v: &[u32]) {
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
}

pub fn put_f32s(out: &mut Vec<u8>, v: &[f32]) {
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
}

/// A cursor over a payload reading little endian scalars and arrays
pub struct Cursor<'a> {
    data: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, at: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.at.checked_add(n).filter(|e| *e <= self.data.len());
        let Some(end) = end else { bail!("frame ends {} bytes short", n.saturating_sub(self.data.len() - self.at)) };
        let s = &self.data[self.at..end];
        self.at = end;
        Ok(s)
    }

    pub fn u32(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    pub fn u32s(&mut self, n: usize) -> Result<Vec<u32>> {
        let b = self.take(n * 4)?;
        Ok(b.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }

    pub fn f32s(&mut self, n: usize) -> Result<Vec<f32>> {
        let b = self.take(n * 4)?;
        Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }

    pub fn done(&self) -> Result<()> {
        ensure!(self.at == self.data.len(), "frame carries {} bytes past its fields", self.data.len() - self.at);
        Ok(())
    }
}

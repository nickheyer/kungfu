//! A stage holds a range of layers for a head, the head's side connects and drives it over tcp or rdma
use crate::loader::{Loaded, Store};
use crate::rdma::{Endpoint, Params};
use crate::wire;
use anyhow::{Context, Result, bail, ensure};
use kf_core::{Backend, Graph, GraphBuilder, Inputs, Plan};
use kf_layers::{Model, Segment};
use std::collections::{HashMap, VecDeque};
use std::io::{Read, Write};
use std::net::{IpAddr, SocketAddr, TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// How long accepts and reads wait before the stop flag is checked again
const POLL: Duration = Duration::from_millis(200);

/// How long the head waits for the ping answer over rdma before it stays on tcp, and the stage for the ping
const PING_WAIT: Duration = Duration::from_secs(3);
const PING_HOLD: Duration = Duration::from_secs(6);

/// The byte stream between a head and a stage, tcp or chunks over a queue pair with tcp kept for control
pub enum Link {
    Tcp(TcpStream),
    Rdma { ctl: TcpStream, ep: Endpoint, inbox: VecDeque<u8>, outbox: Vec<u8>, timeout: Option<Duration> },
}

impl Link {
    pub fn kind(&self) -> &'static str {
        match self {
            Link::Tcp(_) => "tcp",
            Link::Rdma { .. } => "rdma",
        }
    }

    pub fn set_read_timeout(&mut self, t: Option<Duration>) -> Result<()> {
        match self {
            Link::Tcp(s) => Ok(s.set_read_timeout(t)?),
            Link::Rdma { timeout, .. } => {
                *timeout = t;
                Ok(())
            }
        }
    }

    /// The control connection back, the queue pair dropped
    fn into_tcp(self) -> TcpStream {
        match self {
            Link::Tcp(s) => s,
            Link::Rdma { ctl, .. } => ctl,
        }
    }
}

impl Read for Link {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self {
            Link::Tcp(s) => s.read(buf),
            Link::Rdma { ep, inbox, timeout, .. } => {
                if inbox.is_empty() {
                    let deadline = Instant::now() + timeout.unwrap_or(Duration::from_secs(3600));
                    match ep.recv_chunk(deadline) {
                        Ok(Some(chunk)) => inbox.extend(chunk),
                        Ok(None) => return Err(std::io::Error::new(std::io::ErrorKind::TimedOut, "rdma receive timed out")),
                        Err(e) => return Err(std::io::Error::other(format!("{e:#}"))),
                    }
                }
                let n = buf.len().min(inbox.len());
                for (i, b) in inbox.drain(..n).enumerate() {
                    buf[i] = b;
                }
                Ok(n)
            }
        }
    }
}

impl Write for Link {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        match self {
            Link::Tcp(s) => s.write(buf),
            Link::Rdma { outbox, .. } => {
                outbox.extend_from_slice(buf);
                Ok(buf.len())
            }
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        match self {
            Link::Tcp(s) => s.flush(),
            Link::Rdma { ep, outbox, .. } => {
                let r = ep.send_all(outbox).map_err(|e| std::io::Error::other(format!("{e:#}")));
                outbox.clear();
                r
            }
        }
    }
}

/// The head's connection to one stage with the range it holds and the numbers it planned with
pub struct StageConn {
    link: Link,
    pub addr: String,
    pub rank: usize,
    pub from: usize,
    pub to: usize,
    pub n_ctx: usize,
    pub n_batch: usize,
    pub n_seq: usize,
    pub width: usize,
}

impl StageConn {
    /// Connects, makes the hello exchange, and reads the stage's range with its context, batch, and width
    pub fn connect(addr: &str, rank: usize) -> Result<Self> {
        let mut stream = TcpStream::connect(addr).with_context(|| format!("connect stage {addr}"))?;
        stream.set_nodelay(true)?;
        let version = wire::hello(&mut stream)?;
        ensure!(version[0] == wire::PROTOCOL[0], "stage {addr} speaks protocol {}, this head speaks {}", version[0], wire::PROTOCOL[0]);
        wire::send(&mut stream, wire::CMD_RANGE, &[])?;
        let body = wire::recv_reply(&mut stream)?;
        let mut c = wire::Cursor::new(&body);
        let (from, to, n_ctx, n_batch, n_seq, width) = (c.u32()? as usize, c.u32()? as usize, c.u32()? as usize, c.u32()? as usize, c.u32()? as usize, c.u32()? as usize);
        c.done()?;
        let link = negotiate(stream)?;
        kf_core::event::emit("transport", &[("kind", link.kind().to_string()), ("peer", addr.to_string())]);
        Ok(Self { link, addr: addr.to_string(), rank, from, to, n_ctx, n_batch, n_seq, width })
    }

    /// Fails unless the stage planned the same context, batch, slots, and row width as the head
    pub fn check(&self, n_ctx: usize, n_batch: usize, n_seq: usize, width: usize) -> Result<()> {
        ensure!(self.n_ctx == n_ctx, "stage {} plans {} tokens of context, the head {n_ctx}", self.addr, self.n_ctx);
        ensure!(self.n_batch == n_batch, "stage {} feeds batches of {}, the head {n_batch}", self.addr, self.n_batch);
        ensure!(self.n_seq == n_seq, "stage {} keeps {} sequence slots, the head {n_seq}", self.addr, self.n_seq);
        ensure!(self.width == width, "stage {} carries rows of {}, the head {width}", self.addr, self.width);
        Ok(())
    }

    /// Sends the tokens with their positions, slots, rotary streams, attention bounds, and residual rows, returns the rows after the stage's layers
    #[allow(clippy::too_many_arguments)]
    pub fn forward(&mut self, tokens: &[u32], positions: &[u32], seqs: &[u32], rope: &[[u32; 4]], attend: &[u32], hidden: &[f32], cols: usize) -> Result<Vec<f32>> {
        let n = tokens.len();
        ensure!(positions.len() == n && seqs.len() == n && attend.len() == n, "{} positions, {} slots, and {} bounds for {n} tokens", positions.len(), seqs.len(), attend.len());
        ensure!(rope.is_empty() || rope.len() == n, "{} rotary rows for {n} tokens", rope.len());
        let mut p = Vec::with_capacity(12 + n * 36 + hidden.len() * 4);
        wire::put_u32s(&mut p, &[n as u32, cols as u32, rope.len() as u32]);
        wire::put_u32s(&mut p, tokens);
        wire::put_u32s(&mut p, positions);
        wire::put_u32s(&mut p, seqs);
        wire::put_u32s(&mut p, attend);
        for r in rope {
            wire::put_u32s(&mut p, r);
        }
        wire::put_f32s(&mut p, hidden);
        wire::send(&mut self.link, wire::CMD_FORWARD, &p)?;
        let body = wire::recv_reply(&mut self.link)?;
        let mut c = wire::Cursor::new(&body);
        let out = c.f32s(tokens.len() * cols)?;
        c.done()?;
        Ok(out)
    }

    pub fn truncate(&mut self, seq: usize, n: usize) -> Result<()> {
        let mut p = Vec::new();
        wire::put_u32s(&mut p, &[seq as u32, n as u32]);
        wire::send(&mut self.link, wire::CMD_TRUNCATE, &p)?;
        wire::recv_reply(&mut self.link)?;
        Ok(())
    }

    /// Drops one slot, or every slot when none is named
    pub fn reset(&mut self, seq: Option<usize>) -> Result<()> {
        let mut p = Vec::new();
        wire::put_u32s(&mut p, &[seq.map_or(wire::ALL_SEQS, |s| s as u32)]);
        wire::send(&mut self.link, wire::CMD_RESET, &p)?;
        wire::recv_reply(&mut self.link)?;
        Ok(())
    }
}

/// The head's side of the transport choice, rdma when both ends have a device and the first frame crosses it, tcp otherwise
fn negotiate(mut stream: TcpStream) -> Result<Link> {
    let Some(mut ep) = Endpoint::open()? else { return Ok(Link::Tcp(stream)) };
    wire::send(&mut stream, wire::CMD_RDMA, &ep.params().to_bytes())?;
    let body = wire::recv_reply(&mut stream)?;
    if body.is_empty() {
        return Ok(Link::Tcp(stream));
    }
    let remote = Params::from_bytes(&body)?;
    let ready = ep.connect(&remote).is_ok();
    let mut link = Link::Rdma { ctl: stream, ep, inbox: VecDeque::new(), outbox: Vec::new(), timeout: Some(PING_WAIT) };
    let pinged = ready && wire::send(&mut link, wire::CMD_PING, &[]).is_ok() && wire::recv_reply(&mut link).is_ok();
    if pinged {
        link.set_read_timeout(None)?;
        return Ok(link);
    }
    let mut stream = link.into_tcp();
    wire::send(&mut stream, wire::CMD_TCP, &[])?;
    wire::recv_reply(&mut stream)?;
    Ok(Link::Tcp(stream))
}

/// The layers a stage runs with their state, driven by one head at a time
pub struct StageRunner {
    backend: Box<dyn Backend>,
    store: Arc<Store>,
    model: Arc<Model>,
    pub from: usize,
    pub to: usize,
    n_ctx: usize,
    n_batch: usize,
    width: usize,
    n_past: Vec<usize>,
    graphs: HashMap<usize, Arc<Graph>>,
}

impl StageRunner {
    pub fn new(loaded: &Loaded, backend: Box<dyn Backend>, plan: &Plan, from: usize, to: usize) -> Result<Self> {
        let n_layers = loaded.model.n_layers();
        ensure!(from < to && to <= n_layers, "stage range {from}..{to} of a {n_layers} layer model");
        if loaded.model.hands_off() {
            bail!("the model hands tensors between blocks, so no stage can hold a range of its layers");
        }
        let width = loaded.hparams.n_embd * loaded.model.streams;
        Ok(Self { backend, store: loaded.store.clone(), model: loaded.model.clone(), from, to, n_ctx: plan.n_ctx, n_batch: plan.n_batch, width, n_past: vec![0; plan.n_seq.max(1)], graphs: HashMap::new() })
    }

    fn graph(&mut self, n_tokens: usize) -> Result<Arc<Graph>> {
        if !self.graphs.contains_key(&n_tokens) {
            let mut g = GraphBuilder::new();
            let out = self.model.segment(&mut g, n_tokens, Segment { from: self.from, to: self.to, embed: false, inject: false, n_out: None })?;
            self.graphs.insert(n_tokens, Arc::new(g.finish(out)));
        }
        Ok(self.graphs[&n_tokens].clone())
    }

    /// Runs the stage's layers over the rows, each slot's positions following what the stage has seen of it
    #[allow(clippy::too_many_arguments)]
    pub fn forward(&mut self, tokens: &[u32], positions: &[u32], seqs: &[u32], rope: Vec<[u32; 4]>, attend: Vec<u32>, hidden: Vec<f32>, cols: usize) -> Result<Vec<f32>> {
        ensure!(!tokens.is_empty() && tokens.len() <= self.n_batch, "{} tokens for a batch of {}", tokens.len(), self.n_batch);
        ensure!(positions.len() == tokens.len() && seqs.len() == tokens.len() && attend.len() == tokens.len(), "{} positions, {} slots, and {} bounds for {} tokens", positions.len(), seqs.len(), attend.len(), tokens.len());
        ensure!(rope.is_empty() || rope.len() == tokens.len(), "{} rotary rows for {} tokens", rope.len(), tokens.len());
        ensure!(cols == self.width && hidden.len() == tokens.len() * cols, "{} values in rows of {cols} for {} tokens of width {}", hidden.len(), tokens.len(), self.width);
        let runs = kf_core::seq_runs(seqs, tokens.len());
        for (i, (seq, range)) in runs.iter().enumerate() {
            ensure!(*seq < self.n_past.len(), "sequence slot {seq} of {}", self.n_past.len());
            ensure!(!runs[..i].iter().any(|(s, _)| s == seq), "slot {seq} appears twice in one pass");
            let past = self.n_past[*seq];
            ensure!(positions[range.start] as usize == past, "head is at position {} of slot {seq}, this stage at {past}", positions[range.start]);
            ensure!(positions[range.clone()].iter().enumerate().all(|(k, p)| *p as usize == past + k), "positions of slot {seq} are not consecutive");
            ensure!(past + range.len() <= self.n_ctx, "{} tokens past a context of {}", past + range.len(), self.n_ctx);
        }
        let graph = self.graph(tokens.len())?;
        let inputs = Inputs { tokens: tokens.to_vec(), positions: positions.to_vec(), seqs: seqs.to_vec(), hidden, hidden_cols: cols, rope, attend };
        let out = self.backend.run(&graph, &inputs, self.store.as_ref())?;
        for (seq, range) in runs {
            self.n_past[seq] += range.len();
        }
        Ok(out)
    }

    pub fn truncate(&mut self, seq: usize, n: usize) -> Result<()> {
        ensure!(seq < self.n_past.len(), "sequence slot {seq} of {}", self.n_past.len());
        self.backend.truncate(seq, n)?;
        self.n_past[seq] = self.n_past[seq].min(n);
        Ok(())
    }

    /// Drops one slot, or every slot when none is named
    pub fn reset(&mut self, seq: Option<usize>) {
        match seq {
            Some(seq) => {
                self.backend.reset_seq(seq);
                if let Some(n) = self.n_past.get_mut(seq) {
                    *n = 0;
                }
            }
            None => {
                self.backend.reset();
                self.n_past.iter_mut().for_each(|n| *n = 0);
            }
        }
    }

    fn range_reply(&self) -> Vec<u8> {
        let mut p = Vec::new();
        wire::put_u32s(&mut p, &[self.from as u32, self.to as u32, self.n_ctx as u32, self.n_batch as u32, self.n_past.len() as u32, self.width as u32]);
        p
    }

    fn handle(&mut self, cmd: u8, payload: &[u8]) -> Result<Vec<u8>> {
        let mut c = wire::Cursor::new(payload);
        match cmd {
            wire::CMD_RANGE => {
                c.done()?;
                Ok(self.range_reply())
            }
            wire::CMD_FORWARD => {
                let (n, cols, n_rope) = (c.u32()? as usize, c.u32()? as usize, c.u32()? as usize);
                let tokens = c.u32s(n)?;
                let positions = c.u32s(n)?;
                let seqs = c.u32s(n)?;
                let attend = c.u32s(n)?;
                let flat = c.u32s(n_rope * 4)?;
                let rope: Vec<[u32; 4]> = flat.chunks_exact(4).map(|r| [r[0], r[1], r[2], r[3]]).collect();
                let hidden = c.f32s(n * cols)?;
                c.done()?;
                let out = self.forward(&tokens, &positions, &seqs, rope, attend, hidden, cols)?;
                let mut p = Vec::with_capacity(out.len() * 4);
                wire::put_f32s(&mut p, &out);
                Ok(p)
            }
            wire::CMD_TRUNCATE => {
                let (seq, n) = (c.u32()? as usize, c.u32()? as usize);
                c.done()?;
                self.truncate(seq, n)?;
                Ok(Vec::new())
            }
            wire::CMD_RESET => {
                let seq = c.u32()?;
                c.done()?;
                self.reset((seq != wire::ALL_SEQS).then_some(seq as usize));
                Ok(Vec::new())
            }
            other => bail!("unknown command {other}"),
        }
    }
}

/// The addresses a stage admits, from the admit flags with any port stripped
pub fn admitted(list: &[String]) -> Result<Vec<IpAddr>> {
    list.iter()
        .map(|s| {
            let host = s.rsplit_once(':').filter(|(_, p)| p.parse::<u16>().is_ok() && !s.ends_with(']')).map_or(s.as_str(), |(h, _)| h);
            host.trim_matches(|c| c == '[' || c == ']').parse::<IpAddr>().with_context(|| format!("admit {s:?} is not an address"))
        })
        .collect()
}

/// Serves connections one at a time, each admitted by address, answering hellos and chain commands until the stop flag is raised, then hands the runner back
pub fn serve_stage(listener: TcpListener, admit: &[IpAddr], mut runner: StageRunner, stop: Arc<AtomicBool>) -> Result<StageRunner> {
    listener.set_nonblocking(true)?;
    while !stop.load(Ordering::Relaxed) {
        let (stream, peer) = match listener.accept() {
            Ok(x) => x,
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(POLL);
                continue;
            }
            Err(e) => {
                kf_core::event::error("backend", &[("detail", format!("accept: {e}"))]);
                continue;
            }
        };
        if !admit.contains(&peer.ip()) {
            kf_core::event::emit("refused", &[("peer", peer.to_string())]);
            continue;
        }
        stream.set_nonblocking(false)?;
        stream.set_nodelay(true)?;
        if let Err(e) = session(stream, peer, &mut runner, &stop) {
            kf_core::event::error("backend", &[("detail", format!("stage connection from {peer}: {e:#}"))]);
        }
        runner.reset(None);
    }
    Ok(runner)
}

/// The next command byte, waiting through read timeouts until the stop flag is raised, the payload reads after it unbounded
fn next_cmd(link: &mut Link, stop: &AtomicBool) -> Result<Option<u8>> {
    link.set_read_timeout(Some(POLL))?;
    loop {
        match wire::recv_cmd(link) {
            Ok(c) => {
                link.set_read_timeout(None)?;
                return Ok(c);
            }
            Err(e) => match e.downcast_ref::<std::io::Error>().map(|e| e.kind()) {
                Some(std::io::ErrorKind::WouldBlock) | Some(std::io::ErrorKind::TimedOut) if !stop.load(Ordering::Relaxed) => continue,
                Some(std::io::ErrorKind::WouldBlock) | Some(std::io::ErrorKind::TimedOut) => return Ok(None),
                _ => return Err(e),
            },
        }
    }
}

/// The stage's side of the transport choice after the head's parameters arrived, the ping over rdma or the head's word over tcp deciding
fn accept_rdma(mut ctl: TcpStream, head: &[u8]) -> Result<Link> {
    let Ok(remote) = Params::from_bytes(head) else {
        wire::reply(&mut ctl, Ok(Vec::new()))?;
        return Ok(Link::Tcp(ctl));
    };
    let Some(mut ep) = Endpoint::open()? else {
        wire::reply(&mut ctl, Ok(Vec::new()))?;
        return Ok(Link::Tcp(ctl));
    };
    if ep.connect(&remote).is_err() {
        wire::reply(&mut ctl, Ok(Vec::new()))?;
        return Ok(Link::Tcp(ctl));
    }
    wire::reply(&mut ctl, Ok(ep.params().to_bytes()))?;
    let deadline = Instant::now() + PING_HOLD;
    ctl.set_nonblocking(true)?;
    let mut peek = [0u8; 1];
    let mut link = Link::Rdma { ctl, ep, inbox: VecDeque::new(), outbox: Vec::new(), timeout: Some(POLL) };
    while Instant::now() < deadline {
        match wire::recv_cmd(&mut link) {
            Ok(Some(wire::CMD_PING)) => {
                wire::recv_payload(&mut link)?;
                wire::reply(&mut link, Ok(Vec::new()))?;
                if let Link::Rdma { ctl, .. } = &link {
                    ctl.set_nonblocking(false)?;
                }
                link.set_read_timeout(None)?;
                return Ok(link);
            }
            Ok(Some(other)) => bail!("command {other} over rdma before the ping"),
            Ok(None) => bail!("the rdma link closed before the ping"),
            Err(e) => match e.downcast_ref::<std::io::Error>().map(|e| e.kind()) {
                Some(std::io::ErrorKind::WouldBlock) | Some(std::io::ErrorKind::TimedOut) => {}
                _ => return Err(e),
            },
        }
        if let Link::Rdma { ctl, .. } = &link
            && ctl.peek(&mut peek).is_ok_and(|n| n > 0)
        {
            break;
        }
    }
    let mut ctl = link.into_tcp();
    ctl.set_nonblocking(false)?;
    ctl.set_read_timeout(Some(PING_HOLD))?;
    let cmd = wire::recv_cmd(&mut ctl)?;
    ensure!(cmd == Some(wire::CMD_TCP), "the head sent {cmd:?} after the rdma link went unused");
    wire::recv_payload(&mut ctl)?;
    wire::reply(&mut ctl, Ok(Vec::new()))?;
    ctl.set_read_timeout(None)?;
    Ok(Link::Tcp(ctl))
}

fn session(stream: TcpStream, peer: SocketAddr, runner: &mut StageRunner, stop: &AtomicBool) -> Result<()> {
    let mut announced = false;
    let mut link = Link::Tcp(stream);
    while let Some(cmd) = next_cmd(&mut link, stop)? {
        if cmd == wire::HELLO {
            wire::answer_hello(&mut link)?;
            continue;
        }
        if cmd == wire::CMD_RDMA {
            let head = wire::recv_payload(&mut link)?;
            link = accept_rdma(link.into_tcp(), &head)?;
            kf_core::event::emit("transport", &[("kind", link.kind().to_string()), ("peer", peer.to_string())]);
            announced = true;
            continue;
        }
        if !announced {
            kf_core::event::emit("transport", &[("kind", link.kind().to_string()), ("peer", peer.to_string())]);
            announced = true;
        }
        let payload = wire::recv_payload(&mut link)?;
        let result = runner.handle(cmd, &payload);
        wire::reply(&mut link, result)?;
    }
    Ok(())
}

impl StageRunner {
    pub fn backend_ref(&self) -> &dyn Backend {
        self.backend.as_ref()
    }
}

//! RDMA transport for the chain, libibverbs loaded at run time, reliable connected queue pairs carrying the same frames as TCP
use anyhow::{Result, bail, ensure};
use std::ffi::{CStr, c_char, c_int, c_void};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

/// Bytes one send carries, the size of every receive buffer
pub const CHUNK: usize = 1 << 20;
const N_RECV: usize = 32;
const N_SEND: usize = 4;

const WR_SEND: u32 = 2;
const WC_RECV: u32 = 128;
const QPT_RC: u32 = 2;
const SEND_SIGNALED: u32 = 2;
const ACCESS_LOCAL_WRITE: u32 = 1;
const QPS_INIT: u32 = 1;
const QPS_RTR: u32 = 2;
const QPS_RTS: u32 = 3;
const QP_STATE: u32 = 1;
const QP_ACCESS_FLAGS: u32 = 8;
const QP_PKEY_INDEX: u32 = 16;
const QP_PORT: u32 = 32;
const QP_AV: u32 = 128;
const QP_PATH_MTU: u32 = 256;
const QP_TIMEOUT: u32 = 512;
const QP_RETRY_CNT: u32 = 1024;
const QP_RNR_RETRY: u32 = 2048;
const QP_RQ_PSN: u32 = 4096;
const QP_MAX_QP_RD_ATOMIC: u32 = 8192;
const QP_MIN_RNR_TIMER: u32 = 32768;
const QP_SQ_PSN: u32 = 65536;
const QP_MAX_DEST_RD_ATOMIC: u32 = 131072;
const QP_DEST_QPN: u32 = 1 << 20;
const PORT_ACTIVE: u32 = 4;
const LINK_LAYER_ETHERNET: u8 = 2;

type GetDeviceList = unsafe extern "C" fn(*mut c_int) -> *mut *mut c_void;
type FreeDeviceList = unsafe extern "C" fn(*mut *mut c_void);
type GetDeviceName = unsafe extern "C" fn(*mut c_void) -> *const c_char;
type OpenDevice = unsafe extern "C" fn(*mut c_void) -> *mut c_void;
type CloseDevice = unsafe extern "C" fn(*mut c_void) -> c_int;
type QueryPort = unsafe extern "C" fn(*mut c_void, u8, *mut u8) -> c_int;
type QueryGid = unsafe extern "C" fn(*mut c_void, u8, c_int, *mut u8) -> c_int;
type AllocPd = unsafe extern "C" fn(*mut c_void) -> *mut c_void;
type DeallocPd = unsafe extern "C" fn(*mut c_void) -> c_int;
type RegMr = unsafe extern "C" fn(*mut c_void, *mut c_void, usize, c_int) -> *mut c_void;
type DeregMr = unsafe extern "C" fn(*mut c_void) -> c_int;
type CreateCq = unsafe extern "C" fn(*mut c_void, c_int, *mut c_void, *mut c_void, c_int) -> *mut c_void;
type DestroyCq = unsafe extern "C" fn(*mut c_void) -> c_int;
type CreateQp = unsafe extern "C" fn(*mut c_void, *mut u8) -> *mut c_void;
type DestroyQp = unsafe extern "C" fn(*mut c_void) -> c_int;
type ModifyQp = unsafe extern "C" fn(*mut c_void, *mut u8, c_int) -> c_int;
type PostWr = unsafe extern "C" fn(*mut c_void, *mut u8, *mut *mut u8) -> c_int;
type PollCq = unsafe extern "C" fn(*mut c_void, c_int, *mut u8) -> c_int;

/// The verbs library with the entry points the transport calls
struct Verbs {
    _lib: libloading::Library,
    get_device_list: GetDeviceList,
    free_device_list: FreeDeviceList,
    get_device_name: GetDeviceName,
    open_device: OpenDevice,
    close_device: CloseDevice,
    query_port: QueryPort,
    query_gid: QueryGid,
    alloc_pd: AllocPd,
    dealloc_pd: DeallocPd,
    reg_mr: RegMr,
    dereg_mr: DeregMr,
    create_cq: CreateCq,
    destroy_cq: DestroyCq,
    create_qp: CreateQp,
    destroy_qp: DestroyQp,
    modify_qp: ModifyQp,
}

unsafe impl Send for Verbs {}
unsafe impl Sync for Verbs {}

static VERBS: OnceLock<Option<Arc<Verbs>>> = OnceLock::new();

/// Loads libibverbs once, none when the library is not on this machine
fn verbs() -> Option<Arc<Verbs>> {
    VERBS
        .get_or_init(|| {
            let names = ["libibverbs.so.1", "libibverbs.so"];
            let lib = names.iter().find_map(|n| unsafe { libloading::Library::new(*n) }.ok())?;
            macro_rules! sym {
                ($name:literal) => {
                    *lib.get::<unsafe extern "C" fn()>($name).ok()? as *const () as *const c_void
                };
            }
            unsafe {
                let v = Verbs {
                    get_device_list: std::mem::transmute::<*const c_void, GetDeviceList>(sym!(b"ibv_get_device_list\0")),
                    free_device_list: std::mem::transmute::<*const c_void, FreeDeviceList>(sym!(b"ibv_free_device_list\0")),
                    get_device_name: std::mem::transmute::<*const c_void, GetDeviceName>(sym!(b"ibv_get_device_name\0")),
                    open_device: std::mem::transmute::<*const c_void, OpenDevice>(sym!(b"ibv_open_device\0")),
                    close_device: std::mem::transmute::<*const c_void, CloseDevice>(sym!(b"ibv_close_device\0")),
                    query_port: std::mem::transmute::<*const c_void, QueryPort>(sym!(b"ibv_query_port\0")),
                    query_gid: std::mem::transmute::<*const c_void, QueryGid>(sym!(b"ibv_query_gid\0")),
                    alloc_pd: std::mem::transmute::<*const c_void, AllocPd>(sym!(b"ibv_alloc_pd\0")),
                    dealloc_pd: std::mem::transmute::<*const c_void, DeallocPd>(sym!(b"ibv_dealloc_pd\0")),
                    reg_mr: std::mem::transmute::<*const c_void, RegMr>(sym!(b"ibv_reg_mr\0")),
                    dereg_mr: std::mem::transmute::<*const c_void, DeregMr>(sym!(b"ibv_dereg_mr\0")),
                    create_cq: std::mem::transmute::<*const c_void, CreateCq>(sym!(b"ibv_create_cq\0")),
                    destroy_cq: std::mem::transmute::<*const c_void, DestroyCq>(sym!(b"ibv_destroy_cq\0")),
                    create_qp: std::mem::transmute::<*const c_void, CreateQp>(sym!(b"ibv_create_qp\0")),
                    destroy_qp: std::mem::transmute::<*const c_void, DestroyQp>(sym!(b"ibv_destroy_qp\0")),
                    modify_qp: std::mem::transmute::<*const c_void, ModifyQp>(sym!(b"ibv_modify_qp\0")),
                    _lib: lib,
                };
                Some(Arc::new(v))
            }
        })
        .clone()
}

/// What a peer needs to address this endpoint, exchanged over the control connection
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Params {
    pub qpn: u32,
    pub psn: u32,
    pub lid: u16,
    pub gid: [u8; 16],
    pub mtu: u32,
}

impl Params {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut v = Vec::with_capacity(30);
        v.extend_from_slice(&self.qpn.to_le_bytes());
        v.extend_from_slice(&self.psn.to_le_bytes());
        v.extend_from_slice(&self.lid.to_le_bytes());
        v.extend_from_slice(&self.gid);
        v.extend_from_slice(&self.mtu.to_le_bytes());
        v
    }

    pub fn from_bytes(b: &[u8]) -> Result<Self> {
        ensure!(b.len() == 30, "rdma parameters of {} bytes", b.len());
        let mut gid = [0u8; 16];
        gid.copy_from_slice(&b[10..26]);
        Ok(Self { qpn: u32::from_le_bytes(b[0..4].try_into()?), psn: u32::from_le_bytes(b[4..8].try_into()?), lid: u16::from_le_bytes(b[8..10].try_into()?), gid, mtu: u32::from_le_bytes(b[26..30].try_into()?) })
    }
}

/// A registered buffer the device may read and write
struct Region {
    buf: Vec<u8>,
    mr: *mut c_void,
    lkey: u32,
}

/// One side of a reliable connected queue pair with its buffers
pub struct Endpoint {
    v: Arc<Verbs>,
    ctx: *mut c_void,
    pd: *mut c_void,
    send_cq: *mut c_void,
    recv_cq: *mut c_void,
    qp: *mut c_void,
    post_send: PostWr,
    post_recv: PostWr,
    poll_cq: PollCq,
    port: u8,
    gid_index: u8,
    ethernet: bool,
    mtu: u32,
    recv: Vec<Region>,
    send: Vec<Region>,
    /// Send slots with a completion still to collect
    in_flight: usize,
    params: Params,
    pub device: String,
}

unsafe impl Send for Endpoint {}

fn errno() -> std::io::Error {
    std::io::Error::last_os_error()
}

fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().expect("four bytes"))
}

fn put_u32(b: &mut [u8], at: usize, v: u32) {
    b[at..at + 4].copy_from_slice(&v.to_le_bytes());
}

fn put_u16(b: &mut [u8], at: usize, v: u16) {
    b[at..at + 2].copy_from_slice(&v.to_le_bytes());
}

fn put_u64(b: &mut [u8], at: usize, v: u64) {
    b[at..at + 8].copy_from_slice(&v.to_le_bytes());
}

/// The RoCE v2 GID index of a port carrying an IPv4 mapped address, else the first usable index, from sysfs
fn pick_gid(device: &str, port: u8, gids: &[(c_int, [u8; 16])]) -> Option<(u8, [u8; 16])> {
    let kind = |i: c_int| std::fs::read_to_string(format!("/sys/class/infiniband/{device}/ports/{port}/gid_attrs/types/{i}")).unwrap_or_default().trim().to_string();
    let v4 = |g: &[u8; 16]| g[..10].iter().all(|b| *b == 0) && g[10] == 0xff && g[11] == 0xff;
    let usable: Vec<&(c_int, [u8; 16])> = gids.iter().filter(|(_, g)| g.iter().any(|b| *b != 0)).collect();
    usable
        .iter()
        .find(|(i, g)| v4(g) && kind(*i).contains("v2"))
        .or_else(|| usable.iter().find(|(i, _)| kind(*i).contains("v2")))
        .or_else(|| usable.first())
        .map(|(i, g)| (*i as u8, *g))
}

impl Endpoint {
    /// Opens the first device with an active port and makes a queue pair on it, none without a library, a device, or an active port
    pub fn open() -> Result<Option<Endpoint>> {
        let Some(v) = verbs() else { return Ok(None) };
        let mut n: c_int = 0;
        let list = unsafe { (v.get_device_list)(&mut n) };
        if list.is_null() || n <= 0 {
            return Ok(None);
        }
        let devices: Vec<*mut c_void> = (0..n as usize).map(|i| unsafe { *list.add(i) }).collect();
        let mut found = None;
        for dev in devices {
            let name = unsafe { CStr::from_ptr((v.get_device_name)(dev)) }.to_string_lossy().into_owned();
            let ctx = unsafe { (v.open_device)(dev) };
            if ctx.is_null() {
                continue;
            }
            let mut port_attr = [0u8; 64];
            let mut chosen = None;
            for port in 1..=2u8 {
                if unsafe { (v.query_port)(ctx, port, port_attr.as_mut_ptr()) } != 0 {
                    continue;
                }
                let state = u32_at(&port_attr, 0);
                if state != PORT_ACTIVE {
                    continue;
                }
                let gid_tbl_len = u32_at(&port_attr, 12) as c_int;
                let active_mtu = u32_at(&port_attr, 8);
                let lid = u16::from_le_bytes([port_attr[34], port_attr[35]]);
                let link_layer = port_attr[46];
                let mut gids = Vec::new();
                for i in 0..gid_tbl_len.min(32) {
                    let mut g = [0u8; 16];
                    if unsafe { (v.query_gid)(ctx, port, i, g.as_mut_ptr()) } == 0 {
                        gids.push((i, g));
                    }
                }
                let Some((gid_index, gid)) = pick_gid(&name, port, &gids) else { continue };
                chosen = Some((port, lid, link_layer, active_mtu, gid_index, gid));
                break;
            }
            match chosen {
                Some(c) => {
                    found = Some((name, ctx, c));
                    break;
                }
                None => unsafe {
                    (v.close_device)(ctx);
                },
            }
        }
        unsafe { (v.free_device_list)(list) };
        let Some((device, ctx, (port, lid, link_layer, active_mtu, gid_index, gid))) = found else { return Ok(None) };
        let ops = ctx as *const u8;
        let post_send: PostWr = unsafe { std::mem::transmute::<*const c_void, PostWr>(*(ops.add(208) as *const *const c_void)) };
        let post_recv: PostWr = unsafe { std::mem::transmute::<*const c_void, PostWr>(*(ops.add(216) as *const *const c_void)) };
        let poll_cq: PollCq = unsafe { std::mem::transmute::<*const c_void, PollCq>(*(ops.add(96) as *const *const c_void)) };
        let pd = unsafe { (v.alloc_pd)(ctx) };
        ensure!(!pd.is_null(), "rdma protection domain on {device}: {}", errno());
        let send_cq = unsafe { (v.create_cq)(ctx, (N_SEND * 2) as c_int, std::ptr::null_mut(), std::ptr::null_mut(), 0) };
        let recv_cq = unsafe { (v.create_cq)(ctx, (N_RECV * 2) as c_int, std::ptr::null_mut(), std::ptr::null_mut(), 0) };
        ensure!(!send_cq.is_null() && !recv_cq.is_null(), "rdma completion queues on {device}: {}", errno());
        let mut init = [0u8; 64];
        put_u64(&mut init, 8, send_cq as u64);
        put_u64(&mut init, 16, recv_cq as u64);
        put_u32(&mut init, 32, N_SEND as u32);
        put_u32(&mut init, 36, N_RECV as u32);
        put_u32(&mut init, 40, 1);
        put_u32(&mut init, 44, 1);
        put_u32(&mut init, 52, QPT_RC);
        let qp = unsafe { (v.create_qp)(pd, init.as_mut_ptr()) };
        ensure!(!qp.is_null(), "rdma queue pair on {device}: {}", errno());
        let qpn = u32_at(unsafe { std::slice::from_raw_parts(qp as *const u8, 56) }, 52);
        let region = |len: usize| -> Result<Region> {
            let mut buf = vec![0u8; len];
            let mr = unsafe { (v.reg_mr)(pd, buf.as_mut_ptr() as *mut c_void, len, ACCESS_LOCAL_WRITE as c_int) };
            ensure!(!mr.is_null(), "rdma memory registration of {len} bytes: {}", errno());
            let lkey = u32_at(unsafe { std::slice::from_raw_parts(mr as *const u8, 48) }, 36);
            Ok(Region { buf, mr, lkey })
        };
        let recv = (0..N_RECV).map(|_| region(CHUNK)).collect::<Result<Vec<_>>>()?;
        let send = (0..N_SEND).map(|_| region(CHUNK)).collect::<Result<Vec<_>>>()?;
        let psn = (std::process::id().wrapping_mul(2654435761) ^ qpn) & 0xffffff;
        let mtu = active_mtu.clamp(1, 5);
        let params = Params { qpn, psn, lid, gid, mtu };
        let mut ep = Endpoint { v, ctx, pd, send_cq, recv_cq, qp, post_send, post_recv, poll_cq, port, gid_index, ethernet: link_layer == LINK_LAYER_ETHERNET, mtu, recv, send, in_flight: 0, params, device };
        ep.to_init()?;
        for i in 0..N_RECV {
            ep.post_recv(i)?;
        }
        Ok(Some(ep))
    }

    pub fn params(&self) -> Params {
        self.params.clone()
    }

    fn modify(&self, attr: &mut [u8; 160], mask: u32) -> Result<()> {
        let rc = unsafe { (self.v.modify_qp)(self.qp, attr.as_mut_ptr(), mask as c_int) };
        ensure!(rc == 0, "rdma queue pair transition: {}", std::io::Error::from_raw_os_error(rc));
        Ok(())
    }

    fn to_init(&mut self) -> Result<()> {
        let mut a = [0u8; 160];
        put_u32(&mut a, 0, QPS_INIT);
        put_u32(&mut a, 32, ACCESS_LOCAL_WRITE);
        a[129] = self.port;
        self.modify(&mut a, QP_STATE | QP_PKEY_INDEX | QP_PORT | QP_ACCESS_FLAGS)
    }

    /// Brings the queue pair to ready to send against the peer's parameters
    pub fn connect(&mut self, remote: &Params) -> Result<()> {
        let mut a = [0u8; 160];
        put_u32(&mut a, 0, QPS_RTR);
        put_u32(&mut a, 8, self.mtu.min(remote.mtu));
        put_u32(&mut a, 28, remote.qpn);
        put_u32(&mut a, 20, remote.psn);
        a[127] = 1;
        a[128] = 12;
        let ah = 56;
        if self.ethernet || remote.lid == 0 {
            a[ah..ah + 16].copy_from_slice(&remote.gid);
            a[ah + 20] = self.gid_index;
            a[ah + 21] = 64;
            a[ah + 29] = 1;
        } else {
            put_u16(&mut a, ah + 24, remote.lid);
        }
        a[ah + 30] = self.port;
        self.modify(&mut a, QP_STATE | QP_AV | QP_PATH_MTU | QP_DEST_QPN | QP_RQ_PSN | QP_MAX_DEST_RD_ATOMIC | QP_MIN_RNR_TIMER)?;
        let mut a = [0u8; 160];
        put_u32(&mut a, 0, QPS_RTS);
        put_u32(&mut a, 24, self.params.psn);
        a[126] = 1;
        a[130] = 14;
        a[131] = 7;
        a[132] = 7;
        self.modify(&mut a, QP_STATE | QP_TIMEOUT | QP_RETRY_CNT | QP_RNR_RETRY | QP_SQ_PSN | QP_MAX_QP_RD_ATOMIC)
    }

    fn post_recv(&mut self, i: usize) -> Result<()> {
        let r = &self.recv[i];
        let mut sge = [0u8; 16];
        put_u64(&mut sge, 0, r.buf.as_ptr() as u64);
        put_u32(&mut sge, 8, CHUNK as u32);
        put_u32(&mut sge, 12, r.lkey);
        let mut wr = [0u8; 32];
        put_u64(&mut wr, 0, i as u64);
        put_u64(&mut wr, 16, sge.as_mut_ptr() as u64);
        put_u32(&mut wr, 24, 1);
        let mut bad: *mut u8 = std::ptr::null_mut();
        let rc = unsafe { (self.post_recv)(self.qp, wr.as_mut_ptr(), &mut bad) };
        ensure!(rc == 0, "rdma receive post: {}", std::io::Error::from_raw_os_error(rc));
        Ok(())
    }

    /// One completion of a queue within the deadline, the slot and byte count, none when the deadline passes
    fn poll(&self, cq: *mut c_void, deadline: Instant) -> Result<Option<(usize, usize)>> {
        let mut wc = [0u8; 48];
        let mut spins = 0u32;
        loop {
            let n = unsafe { (self.poll_cq)(cq, 1, wc.as_mut_ptr()) };
            ensure!(n >= 0, "rdma completion poll: {}", std::io::Error::from_raw_os_error(-n));
            if n > 0 {
                let status = u32_at(&wc, 8);
                ensure!(status == 0, "rdma work completion failed with status {status}");
                let id = u64::from_le_bytes(wc[0..8].try_into()?) as usize;
                let opcode = u32_at(&wc, 12);
                let bytes = if opcode & WC_RECV != 0 { u32_at(&wc, 20) as usize } else { 0 };
                return Ok(Some((id, bytes)));
            }
            if Instant::now() >= deadline {
                return Ok(None);
            }
            spins += 1;
            if spins > 2000 {
                std::thread::sleep(Duration::from_micros(50));
            } else {
                std::hint::spin_loop();
            }
        }
    }

    fn collect_sends(&mut self, want_free: bool) -> Result<()> {
        while self.in_flight > 0 && (want_free && self.in_flight >= N_SEND || !want_free) {
            let deadline = Instant::now() + Duration::from_secs(30);
            match self.poll(self.send_cq, deadline)? {
                Some(_) => self.in_flight -= 1,
                None => bail!("rdma send did not complete within 30 seconds"),
            }
        }
        Ok(())
    }

    /// Sends bytes as chunks, each a send on a free slot, every completion collected before the slots recycle
    pub fn send_all(&mut self, data: &[u8]) -> Result<()> {
        let chunks: Vec<&[u8]> = if data.is_empty() { vec![&[][..]] } else { data.chunks(CHUNK).collect() };
        for (k, chunk) in chunks.iter().enumerate() {
            self.collect_sends(true)?;
            let slot = k % N_SEND;
            self.send[slot].buf[..chunk.len()].copy_from_slice(chunk);
            let r = &self.send[slot];
            let mut sge = [0u8; 16];
            put_u64(&mut sge, 0, r.buf.as_ptr() as u64);
            put_u32(&mut sge, 8, chunk.len() as u32);
            put_u32(&mut sge, 12, r.lkey);
            let mut wr = [0u8; 128];
            put_u64(&mut wr, 0, slot as u64);
            put_u64(&mut wr, 16, sge.as_mut_ptr() as u64);
            put_u32(&mut wr, 24, 1);
            put_u32(&mut wr, 28, WR_SEND);
            put_u32(&mut wr, 32, SEND_SIGNALED);
            let mut bad: *mut u8 = std::ptr::null_mut();
            let rc = unsafe { (self.post_send)(self.qp, wr.as_mut_ptr(), &mut bad) };
            ensure!(rc == 0, "rdma send post: {}", std::io::Error::from_raw_os_error(rc));
            self.in_flight += 1;
        }
        self.collect_sends(false)
    }

    /// The next chunk the peer sent, none when the deadline passes first
    pub fn recv_chunk(&mut self, deadline: Instant) -> Result<Option<Vec<u8>>> {
        let Some((slot, bytes)) = self.poll(self.recv_cq, deadline)? else { return Ok(None) };
        ensure!(slot < N_RECV, "rdma completion names receive slot {slot}");
        let data = self.recv[slot].buf[..bytes].to_vec();
        self.post_recv(slot)?;
        Ok(Some(data))
    }
}

impl Drop for Endpoint {
    fn drop(&mut self) {
        unsafe {
            (self.v.destroy_qp)(self.qp);
            (self.v.destroy_cq)(self.send_cq);
            (self.v.destroy_cq)(self.recv_cq);
            for r in self.send.iter().chain(self.recv.iter()) {
                (self.v.dereg_mr)(r.mr);
            }
            (self.v.dealloc_pd)(self.pd);
            (self.v.close_device)(self.ctx);
        }
    }
}

/// Whether an endpoint could be opened at all on this machine
pub fn available() -> bool {
    Endpoint::open().ok().flatten().is_some()
}

pub fn gid_text(gid: &[u8; 16]) -> String {
    gid.iter().map(|b| format!("{b:02x}")).collect::<Vec<_>>().join("")
}

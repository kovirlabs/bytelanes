//! [`WgpuBackend`]: runs channel programs on a GPU through wgpu.
//!
//! Every check that decides a completion's status (validation, buffers,
//! bounds, keys, lengths, destination size) runs on the host in exactly the
//! order `bytelanes::cpu` uses, so both report the same status for the same
//! program. All work on the data runs on the GPU: the op itself, writing a
//! result into device memory, and the CRC-32C of the final result.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use std::sync::mpsc::{self, SyncSender};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;

use bytelanes::abi::{
    BufId, BufRange, Caps, ChannelCmd, ChannelProgram, CmpOp, Completion, ElemType, ExtentKind,
    ExtentRef, Features, Flags, KeySlot, NO_INDEX, Op, PathState, Status, Tag,
};
use bytelanes::{Buffers, Busy, ChannelBackend};

use crate::gpu::{Data, FilterArgs, Gpu};
use crate::strategy::{BufferStrategy, CopyingStrategy};

/// Default number of programs queued or finished-but-unpolled before
/// `submit` reports [`Busy`].
pub const DEFAULT_QUEUE_DEPTH: usize = 1024;

/// Errors creating a [`WgpuBackend`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// No adapter matched.
    NoAdapter(String),
    /// The device could not be created.
    Device(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::NoAdapter(why) => write!(f, "no GPU adapter: {why}"),
            Error::Device(why) => write!(f, "could not open the GPU: {why}"),
        }
    }
}

impl std::error::Error for Error {}

struct VramBuf {
    buf: wgpu::Buffer,
    len: u64,
}

#[derive(Default)]
struct State {
    host: HashMap<BufId, Vec<u8>>,
    vram: HashMap<BufId, VramBuf>,
    keys: HashMap<KeySlot, wgpu::Buffer>,
}

struct Shared {
    gpu: Gpu,
    state: Mutex<State>,
    done: Mutex<VecDeque<Completion>>,
    /// Programs accepted and not yet polled (queued, running or done).
    in_flight: AtomicUsize,
    health: AtomicU8,
}

impl Shared {
    fn state(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn done(&self) -> MutexGuard<'_, VecDeque<Completion>> {
        self.done.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// A GPU path. Programs run in submission order on a worker thread; `submit`
/// and `poll` never wait for GPU work.
pub struct WgpuBackend {
    shared: Arc<Shared>,
    queue_depth: usize,
    jobs: Option<SyncSender<(Box<ChannelProgram>, Tag)>>,
    worker: Option<JoinHandle<()>>,
}

const ONLINE: u8 = 0;
const DEGRADED: u8 = 1;

impl WgpuBackend {
    /// Opens `adapter` with the [`CopyingStrategy`] and the default queue
    /// depth.
    pub fn new(adapter: &wgpu::Adapter) -> Result<Self, Error> {
        Self::with_strategy(adapter, Box::new(CopyingStrategy), DEFAULT_QUEUE_DEPTH)
    }

    /// Opens `adapter` with a buffer strategy and queue depth.
    pub fn with_strategy(
        adapter: &wgpu::Adapter,
        strategy: Box<dyn BufferStrategy>,
        queue_depth: usize,
    ) -> Result<Self, Error> {
        let gpu = Gpu::new(adapter, strategy).map_err(Error::Device)?;
        let shared = Arc::new(Shared {
            gpu,
            state: Mutex::new(State::default()),
            done: Mutex::new(VecDeque::new()),
            in_flight: AtomicUsize::new(0),
            health: AtomicU8::new(ONLINE),
        });
        let (tx, rx) = mpsc::sync_channel::<(Box<ChannelProgram>, Tag)>(queue_depth.max(1));
        let worker_shared = Arc::clone(&shared);
        let worker = std::thread::Builder::new()
            .name("bytelanes-wgpu".into())
            .spawn(move || {
                for (prog, tag) in rx {
                    let s = &worker_shared;
                    let c = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        execute(&s.gpu, &mut s.state(), &prog, tag)
                    }))
                    .unwrap_or_else(|_| {
                        s.health.store(DEGRADED, Ordering::Relaxed);
                        Completion::error(tag, NO_INDEX, Status::DeviceError, 0)
                    });
                    s.done().push_back(c);
                }
            })
            .map_err(|e| Error::Device(format!("worker thread: {e}")))?;
        Ok(WgpuBackend {
            shared,
            queue_depth,
            jobs: Some(tx),
            worker: Some(worker),
        })
    }

    /// The adapter this backend runs on: name, driver, backend API.
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.shared.gpu.info
    }

    /// The buffer strategy in use.
    pub fn strategy_name(&self) -> &'static str {
        self.shared.gpu.strategy_name()
    }

    /// Drops every host buffer, device buffer and key, as if the backend
    /// were new, without reopening the device. Programs still queued run
    /// against the empty state.
    pub fn clear(&self) {
        *self.shared.state() = State::default();
    }
}

impl Drop for WgpuBackend {
    fn drop(&mut self) {
        self.jobs.take();
        if let Some(w) = self.worker.take() {
            let _ = w.join();
        }
    }
}

/// The ops this backend runs on the GPU. `Fft` (added to the op format in
/// M7) is not implemented in WGSL yet, so it is not listed and completes
/// with `UnsupportedOp`.
pub const GPU_OPS: [Op; 4] = [Op::Checksum, Op::Encrypt, Op::Decrypt, Op::Filter];

impl ChannelBackend for WgpuBackend {
    fn capabilities(&self) -> Caps {
        Caps::new(
            &GPU_OPS,
            &[ExtentKind::Host, ExtentKind::Vram],
            Features::NONE,
            u32::try_from(self.queue_depth).unwrap_or(u32::MAX),
        )
    }

    fn submit(&self, prog: &ChannelProgram, tag: Tag) -> Result<(), Busy> {
        let s = &self.shared;
        if s.in_flight.fetch_add(1, Ordering::AcqRel) >= self.queue_depth {
            s.in_flight.fetch_sub(1, Ordering::AcqRel);
            return Err(Busy);
        }
        let sent = self
            .jobs
            .as_ref()
            .is_some_and(|tx| tx.try_send((Box::new(*prog), tag)).is_ok());
        if !sent {
            s.in_flight.fetch_sub(1, Ordering::AcqRel);
            return Err(Busy);
        }
        Ok(())
    }

    fn poll(&self, out: &mut [Completion]) -> usize {
        let mut done = self.shared.done();
        let n = out.len().min(done.len());
        for (slot, c) in out.iter_mut().zip(done.drain(..n)) {
            *slot = c;
        }
        self.shared.in_flight.fetch_sub(n, Ordering::AcqRel);
        n
    }

    fn health(&self) -> PathState {
        match self.shared.health.load(Ordering::Relaxed) {
            ONLINE => PathState::Online,
            _ => PathState::Degraded,
        }
    }
}

impl Buffers for WgpuBackend {
    fn put_host(&self, buf: BufId, data: Vec<u8>) -> Result<(), Status> {
        self.shared.state().host.insert(buf, data);
        Ok(())
    }

    fn get_host(&self, buf: BufId) -> Option<Vec<u8>> {
        self.shared.state().host.get(&buf).cloned()
    }

    fn alloc_vram(&self, buf: BufId, len: u64) -> Result<(), Status> {
        let gpu = &self.shared.gpu;
        if !gpu.fits(len) {
            return Err(Status::DeviceError);
        }
        let b = gpu.zeroed("vram", len);
        self.shared
            .state()
            .vram
            .insert(buf, VramBuf { buf: b, len });
        Ok(())
    }

    fn set_key(&self, slot: KeySlot, key: [u8; 32]) -> Result<(), Status> {
        let rk = self.shared.gpu.round_keys(&key);
        self.shared.state().keys.insert(slot, rk);
        Ok(())
    }
}

/// A failure inside one command: status and, for `DstTooSmall`, bytes needed.
struct Fail(Status, u32);

impl From<Status> for Fail {
    fn from(s: Status) -> Self {
        Fail(s, 0)
    }
}

fn device_error(_: String) -> Fail {
    Fail(Status::DeviceError, 0)
}

fn execute(gpu: &Gpu, st: &mut State, prog: &ChannelProgram, tag: Tag) -> Completion {
    if let Err(e) = prog.validate() {
        return Completion::error(tag, e.index, e.status, 0);
    }
    let cmds = prog.commands();
    let mut prev: Option<Data> = None;
    for (i, cmd) in cmds.iter().enumerate() {
        let idx = i as u16;
        let fed = i > 0 && cmds[i - 1].flags().contains(Flags::CHAIN);
        let input = if fed { prev.take() } else { None };
        match run_cmd(gpu, st, cmd, input) {
            Ok(out) => prev = Some(out),
            Err(Fail(status, bytes)) => return Completion::error(tag, idx, status, bytes),
        }
    }
    let last = u16::try_from(cmds.len() - 1).unwrap_or(NO_INDEX);
    // The program validated, so it has at least one command and `prev` is set.
    let Some(result) = prev else {
        return Completion::error(tag, NO_INDEX, Status::DeviceError, 0);
    };
    // The result's CRC is computed on the GPU.
    let crc = gpu.crc(&result);
    match gpu.download(&crc, 0, 4) {
        Ok(b) => Completion::ok(
            tag,
            last,
            result.len as u32,
            u32::from_le_bytes([b[0], b[1], b[2], b[3]]),
        ),
        Err(_) => Completion::error(tag, last, Status::DeviceError, 0),
    }
}

fn range_of(e: ExtentRef) -> BufRange {
    // Callers only pass Host or Vram extents.
    e.buf_range().unwrap_or(BufRange::new(BufId(0), 0, 0))
}

/// Length of the buffer an extent names, or `UnknownBuffer`.
fn buffer_len(st: &State, e: ExtentRef) -> Result<u64, Status> {
    match e {
        ExtentRef::Host(r) => st.host.get(&r.buf).map(|b| b.len() as u64),
        ExtentRef::Vram(r) => st.vram.get(&r.buf).map(|b| b.len),
        _ => return Err(Status::UnsupportedExtent),
    }
    .ok_or(Status::UnknownBuffer)
}

fn check_range(len: u64, r: &BufRange) -> Result<(), Status> {
    match r.end() {
        Some(end) if end <= len => Ok(()),
        _ => Err(Status::OutOfBounds),
    }
}

/// The source bytes on the GPU: uploaded for `Host`, in place for `Vram`.
fn read(gpu: &Gpu, st: &State, e: ExtentRef) -> Result<Data, Fail> {
    let r = range_of(e);
    check_range(buffer_len(st, e)?, &r)?;
    if !gpu.fits(r.len) {
        return Err(Fail(Status::DeviceError, 0));
    }
    match e {
        ExtentRef::Host(_) => {
            let buf = &st.host[&r.buf];
            Ok(gpu.upload(&buf[r.offset as usize..(r.offset + r.len) as usize]))
        }
        _ => Ok(Data {
            buf: st.vram[&r.buf].buf.clone(),
            off: r.offset,
            len: r.len,
        }),
    }
}

fn filter_args(spec: &bytelanes::abi::FilterSpec) -> FilterArgs {
    let (width, kind) = match spec.elem {
        ElemType::U32 => (4, 0),
        ElemType::I32 => (4, 1),
        ElemType::F32 => (4, 2),
        ElemType::U64 => (8, 0),
        ElemType::I64 => (8, 1),
        ElemType::F64 => (8, 2),
    };
    let cmp = match spec.cmp {
        CmpOp::Eq => 0,
        CmpOp::Ne => 1,
        CmpOp::Lt => 2,
        CmpOp::Le => 3,
        CmpOp::Gt => 4,
        CmpOp::Ge => 5,
        CmpOp::Between => 6,
    };
    let (a, a_nan) = ordered_key(spec.elem, spec.a);
    let (b, b_nan) = ordered_key(spec.elem, spec.b);
    FilterArgs {
        width,
        kind,
        cmp,
        a,
        b,
        a_nan,
        b_nan,
    }
}

/// The operand's ordered key (hi, lo) and NaN flag; the host-side twin of
/// `key_of` in `filter.wgsl`.
pub(crate) fn ordered_key(elem: ElemType, bits: u64) -> ((u32, u32), bool) {
    let lo = bits as u32;
    let hi = (bits >> 32) as u32;
    match elem {
        ElemType::U32 => ((0, lo), false),
        ElemType::I32 => ((0, lo ^ 0x8000_0000), false),
        ElemType::U64 => ((hi, lo), false),
        ElemType::I64 => ((hi ^ 0x8000_0000, lo), false),
        ElemType::F32 => {
            let mag = lo & 0x7FFF_FFFF;
            let x = if mag == 0 { 0 } else { lo };
            let k = if x & 0x8000_0000 != 0 {
                !x
            } else {
                x | 0x8000_0000
            };
            ((0, k), mag > 0x7F80_0000)
        }
        ElemType::F64 => {
            let mag = hi & 0x7FFF_FFFF;
            let nan = mag > 0x7FF0_0000 || (mag == 0x7FF0_0000 && lo != 0);
            let hi = if mag == 0 && lo == 0 { 0 } else { hi };
            let k = if hi & 0x8000_0000 != 0 {
                (!hi, !lo)
            } else {
                (hi | 0x8000_0000, lo)
            };
            (k, nan)
        }
    }
}

/// Largest number of rows a `Filter` input may have (indices are `u32`).
const MAX_FILTER_ROWS: u64 = 1 << 32;

fn run_cmd(gpu: &Gpu, st: &mut State, cmd: &ChannelCmd, fed: Option<Data>) -> Result<Data, Fail> {
    // Validation already passed, so check() cannot fail here.
    let (op, src, dst) = cmd.check().map_err(|_| Status::InvalidProgram)?;

    // 1. Op.
    if !GPU_OPS.contains(&op) {
        return Err(Status::UnsupportedOp.into());
    }
    // 2. Extent kinds.
    for e in [src, dst] {
        if !matches!(e, ExtentRef::Host(_) | ExtentRef::Vram(_)) {
            return Err(Status::UnsupportedExtent.into());
        }
    }
    // 3. Buffers and bounds: src (unless fed), then dst.
    let input = match fed {
        Some(d) => d,
        None => read(gpu, st, src)?,
    };
    let dst_range = range_of(dst);
    check_range(buffer_len(st, dst)?, &dst_range)?;

    // 4-5. Key and length, then the op on the GPU.
    let out = match op {
        Op::Checksum => Data {
            buf: gpu.crc(&input),
            off: 0,
            len: 4,
        },
        Op::Encrypt | Op::Decrypt => {
            let p = cmd.crypt_params();
            let rk = st.keys.get(&p.key_slot()).ok_or(Status::UnknownKey)?;
            gpu.aes_ctr(&input, rk, &p.iv)
        }
        Op::Filter => {
            let spec = cmd.filter_spec().map_err(|_| Status::InvalidProgram)?;
            let w = spec.elem.width() as u64;
            if input.len % w != 0 || input.len / w > MAX_FILTER_ROWS {
                return Err(Status::InvalidLength.into());
            }
            gpu.filter(&input, &filter_args(&spec))
                .map_err(device_error)?
        }
        Op::Read | Op::Write | Op::Decompress | Op::Compress | Op::Score | Op::Fft => {
            return Err(Status::UnsupportedOp.into());
        }
    };
    let Ok(out_len) = u32::try_from(out.len) else {
        return Err(Status::InvalidLength.into());
    };

    // 6. Destination size.
    if out.len > dst_range.len {
        return Err(Fail(Status::DstTooSmall, out_len));
    }
    match dst {
        ExtentRef::Host(r) => {
            let bytes = gpu
                .download(&out.buf, out.off, out.len)
                .map_err(device_error)?;
            let host = st.host.get_mut(&r.buf).ok_or(Status::UnknownBuffer)?;
            let start = r.offset as usize;
            host[start..start + bytes.len()].copy_from_slice(&bytes);
        }
        _ => {
            let v = st.vram.get(&dst_range.buf).ok_or(Status::UnknownBuffer)?;
            gpu.place(&out, &v.buf, dst_range.offset);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::ordered_key;
    use bytelanes::abi::ElemType;

    /// The ordered keys sort like the values they come from.
    #[test]
    fn float_keys_order_like_floats() {
        let vals = [
            f64::NEG_INFINITY,
            -1e300,
            -1.0,
            -f64::MIN_POSITIVE,
            -0.0,
            0.0,
            5e-324,
            1.0,
            f64::MAX,
            f64::INFINITY,
        ];
        for w in vals.windows(2) {
            let (a, _) = ordered_key(ElemType::F64, w[0].to_bits());
            let (b, _) = ordered_key(ElemType::F64, w[1].to_bits());
            if w[0] == w[1] {
                assert_eq!(a, b, "{} vs {}", w[0], w[1]);
            } else {
                assert!(a < b, "{} vs {}", w[0], w[1]);
            }
        }
        assert!(ordered_key(ElemType::F64, f64::NAN.to_bits()).1);
        assert!(ordered_key(ElemType::F32, u64::from(f32::NAN.to_bits())).1);
        assert!(!ordered_key(ElemType::F32, u64::from(f32::INFINITY.to_bits())).1);
        let (n, _) = ordered_key(ElemType::F32, u64::from((-0.0f32).to_bits()));
        let (p, _) = ordered_key(ElemType::F32, 0);
        assert_eq!(n, p);
    }

    #[test]
    fn signed_keys_order_like_ints() {
        let vals = [i64::MIN, -2, -1, 0, 1, i64::MAX];
        for w in vals.windows(2) {
            let (a, _) = ordered_key(ElemType::I64, w[0] as u64);
            let (b, _) = ordered_key(ElemType::I64, w[1] as u64);
            assert!(a < b);
        }
        let (a, _) = ordered_key(ElemType::I32, u64::from(-1i32 as u32));
        let (b, _) = ordered_key(ElemType::I32, 0);
        assert!(a < b);
    }
}

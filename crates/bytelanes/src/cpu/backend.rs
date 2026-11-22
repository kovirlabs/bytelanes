//! [`CpuBackend`]: runs channel programs on the CPU reference ops.

use std::collections::{HashMap, VecDeque};
use std::sync::{Mutex, MutexGuard};

use super::ops::{self, Key};
use crate::abi::{
    BufId, BufRange, Caps, ChannelProgram, Completion, ExtentKind, ExtentRef, Features, Flags,
    KeySlot, NO_INDEX, Op, PathState, Status, Tag,
};
use crate::{Buffers, Busy, ChannelBackend};

/// Default number of completions held before `submit` reports [`Busy`].
pub const DEFAULT_QUEUE_DEPTH: usize = 1024;

/// The reference backend. Runs each program to completion inside `submit`,
/// so its completions are ready at the next `poll`.
///
/// It supports `Host` and `Vram` extents (`Vram` buffers are ordinary memory
/// here) and the ops with defined semantics: `Checksum`, `Encrypt`,
/// `Decrypt`, `Filter` and `Fft`.
pub struct CpuBackend {
    state: Mutex<State>,
    queue_depth: usize,
}

#[derive(Default)]
struct State {
    host: HashMap<BufId, Vec<u8>>,
    vram: HashMap<BufId, Vec<u8>>,
    keys: HashMap<KeySlot, Key>,
    done: VecDeque<Completion>,
    health: Option<PathState>,
}

impl Default for CpuBackend {
    fn default() -> Self {
        CpuBackend::new()
    }
}

impl CpuBackend {
    /// A backend with [`DEFAULT_QUEUE_DEPTH`].
    pub fn new() -> Self {
        CpuBackend::with_queue_depth(DEFAULT_QUEUE_DEPTH)
    }

    /// A backend that holds at most `depth` unpolled completions.
    pub fn with_queue_depth(depth: usize) -> Self {
        CpuBackend {
            state: Mutex::new(State::default()),
            queue_depth: depth,
        }
    }

    /// Override the reported health, for failover tests.
    pub fn set_health(&self, health: PathState) {
        self.lock().health = Some(health);
    }

    /// Remove and return host buffer `buf`.
    pub fn take_host(&self, buf: BufId) -> Option<Vec<u8>> {
        self.lock().host.remove(&buf)
    }

    /// Run `prog` now and return its completion, bypassing the queue.
    pub fn run(&self, prog: &ChannelProgram, tag: Tag) -> Completion {
        let mut st = self.lock();
        execute(&mut st, prog, tag)
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        // A panic while holding the lock leaves only plain buffers behind,
        // which stay usable.
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl ChannelBackend for CpuBackend {
    fn capabilities(&self) -> Caps {
        Caps::new(
            &Op::DEFINED,
            &[ExtentKind::Host, ExtentKind::Vram],
            Features::NONE,
            u32::try_from(self.queue_depth).unwrap_or(u32::MAX),
        )
    }

    fn submit(&self, prog: &ChannelProgram, tag: Tag) -> Result<(), Busy> {
        let mut st = self.lock();
        if st.done.len() >= self.queue_depth {
            return Err(Busy);
        }
        let c = execute(&mut st, prog, tag);
        st.done.push_back(c);
        Ok(())
    }

    fn poll(&self, out: &mut [Completion]) -> usize {
        let mut st = self.lock();
        let n = out.len().min(st.done.len());
        for (slot, c) in out.iter_mut().zip(st.done.drain(..n)) {
            *slot = c;
        }
        n
    }

    fn health(&self) -> PathState {
        self.lock().health.unwrap_or(PathState::Online)
    }
}

impl Buffers for CpuBackend {
    fn put_host(&self, buf: BufId, data: Vec<u8>) -> Result<(), Status> {
        self.lock().host.insert(buf, data);
        Ok(())
    }

    fn get_host(&self, buf: BufId) -> Option<Vec<u8>> {
        self.lock().host.get(&buf).cloned()
    }

    fn alloc_vram(&self, buf: BufId, len: u64) -> Result<(), Status> {
        let len = usize::try_from(len).map_err(|_| Status::DeviceError)?;
        self.lock().vram.insert(buf, vec![0; len]);
        Ok(())
    }

    fn set_key(&self, slot: KeySlot, key: [u8; 32]) -> Result<(), Status> {
        self.lock().keys.insert(slot, key);
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

fn execute(st: &mut State, prog: &ChannelProgram, tag: Tag) -> Completion {
    if let Err(e) = prog.validate() {
        return Completion::error(tag, e.index, e.status, 0);
    }
    let cmds = prog.commands();
    let mut prev: Option<Vec<u8>> = None;
    let mut result_crc = 0;
    let mut result_len = 0;
    for (i, cmd) in cmds.iter().enumerate() {
        let idx = i as u16;
        let fed = i > 0 && cmds[i - 1].flags().contains(Flags::CHAIN);
        let input = if fed { prev.take() } else { None };
        match run_cmd(st, cmd, input) {
            Ok(out) => {
                result_len = out.len() as u32;
                result_crc = ops::checksum(&out);
                prev = Some(out);
            }
            Err(Fail(status, bytes)) => return Completion::error(tag, idx, status, bytes),
        }
    }
    let last = u16::try_from(cmds.len() - 1).unwrap_or(NO_INDEX);
    Completion::ok(tag, last, result_len, result_crc)
}

/// Run one command; returns its result bytes. `fed` is the previous
/// command's result when a chain feeds this command.
fn run_cmd(
    st: &mut State,
    cmd: &crate::abi::ChannelCmd,
    fed: Option<Vec<u8>>,
) -> Result<Vec<u8>, Fail> {
    // Validation already passed, so check() cannot fail here.
    let (op, src, dst) = cmd.check().map_err(|_| Status::InvalidProgram)?;

    // 1. Op.
    if !op.is_defined() {
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
        Some(v) => v,
        None => read(st, src)?,
    };
    let dst_range = range_of(dst);
    check_range(region(st, dst)?, &dst_range)?;

    // 4-5. Key and length, inside the op.
    let out = match op {
        Op::Checksum => ops::checksum(&input).to_le_bytes().to_vec(),
        Op::Encrypt | Op::Decrypt => {
            let p = cmd.crypt_params();
            let key = st.keys.get(&p.key_slot()).ok_or(Status::UnknownKey)?;
            let mut out = vec![0; input.len()];
            ops::aes256_ctr(key, &p.iv, &input, &mut out);
            out
        }
        Op::Filter => {
            let spec = cmd.filter_spec().map_err(|_| Status::InvalidProgram)?;
            let mut idx = Vec::new();
            ops::filter(&spec, &input, &mut idx)?;
            ops::indices_to_bytes(&idx)
        }
        Op::Fft => {
            let mut out = Vec::new();
            super::fft::fft(cmd.fft_params().log2n(), &input, &mut out)?;
            out
        }
        Op::Read | Op::Write | Op::Decompress | Op::Compress | Op::Score => {
            return Err(Status::UnsupportedOp.into());
        }
    };
    let Ok(out_len) = u32::try_from(out.len()) else {
        return Err(Status::InvalidLength.into());
    };

    // 6. Destination size.
    if out.len() as u64 > dst_range.len {
        return Err(Fail(Status::DstTooSmall, out_len));
    }
    let buf = region(st, dst)?;
    let start = dst_range.offset as usize;
    buf[start..start + out.len()].copy_from_slice(&out);
    Ok(out)
}

fn range_of(e: ExtentRef) -> BufRange {
    // Callers only pass Host or Vram extents.
    e.buf_range().unwrap_or(BufRange::new(BufId(0), 0, 0))
}

fn region(st: &mut State, e: ExtentRef) -> Result<&mut Vec<u8>, Status> {
    let (map, r) = match e {
        ExtentRef::Host(r) => (&mut st.host, r),
        ExtentRef::Vram(r) => (&mut st.vram, r),
        _ => return Err(Status::UnsupportedExtent),
    };
    map.get_mut(&r.buf).ok_or(Status::UnknownBuffer)
}

fn check_range(buf: &[u8], r: &BufRange) -> Result<(), Status> {
    match r.end() {
        Some(end) if end <= buf.len() as u64 => Ok(()),
        _ => Err(Status::OutOfBounds),
    }
}

fn read(st: &mut State, e: ExtentRef) -> Result<Vec<u8>, Status> {
    let r = range_of(e);
    let buf = region(st, e)?;
    check_range(buf, &r)?;
    Ok(buf[r.offset as usize..(r.offset + r.len) as usize].to_vec())
}

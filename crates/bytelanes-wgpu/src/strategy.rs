//! How host bytes reach the GPU and come back.
//!
//! This trait is the seam between the ops (agent B) and the memory model:
//! [`CopyingStrategy`] stages every transfer through a copy, which is what a
//! discrete GPU on PCIe needs. A unified-memory strategy (agent C, M4) can
//! hand the GPU host-visible memory instead. The ops only ever see a
//! `wgpu::Buffer` with the input at offset 0.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, mpsc};

use crate::pool::{DevicePool, size_class};

/// One range to read back: `dst.len()` bytes of `buffer` from `offset`.
pub struct Readback<'a> {
    /// The GPU buffer to read.
    pub buffer: &'a wgpu::Buffer,
    /// Byte offset in `buffer`; a multiple of 4.
    pub offset: u64,
    /// Where the bytes go; its length is the number of bytes read.
    pub dst: &'a mut [u8],
}

/// Moves bytes between host memory and GPU buffers.
pub trait BufferStrategy: Send + Sync {
    /// Short name, recorded with results.
    fn name(&self) -> &'static str;

    /// A buffer holding `data` at offset 0, at least `capacity` bytes long
    /// (`capacity >= data.len()`, a multiple of 4), with usage
    /// `STORAGE | COPY_SRC | COPY_DST`. The bytes must be visible to work
    /// submitted after this returns.
    fn upload(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data: &[u8],
        capacity: u64,
    ) -> wgpu::Buffer;

    /// `len` bytes of `buffer` starting at `offset`, read after all work
    /// submitted so far has finished. `offset` is a multiple of 4.
    fn download(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffer: &wgpu::Buffer,
        offset: u64,
        len: u64,
    ) -> Result<Vec<u8>, String>;

    /// Reads every range in `parts` into its `dst`, after all work submitted
    /// so far has finished. A strategy should do this with a single wait for
    /// the GPU; the default reads the parts one at a time with `download`.
    fn download_into(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        parts: &mut [Readback<'_>],
    ) -> Result<(), String> {
        for p in parts.iter_mut() {
            let bytes = self.download(device, queue, p.buffer, p.offset, p.dst.len() as u64)?;
            p.dst.copy_from_slice(&bytes);
        }
        Ok(())
    }

    /// Starts reading `parts` back: submits the copies but does not wait, so
    /// the caller can queue more GPU work before [`finish_download`]
    /// completes it. The default defers everything to `finish_download`.
    ///
    /// [`finish_download`]: BufferStrategy::finish_download
    fn begin_download(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        parts: &[ReadRange<'_>],
    ) -> Result<PendingDownload, String> {
        let _ = (device, queue);
        Ok(PendingDownload::deferred(parts))
    }

    /// Completes a readback from `begin_download`: waits for it and fills
    /// `dsts`, one per part, each as long as its part.
    fn finish_download(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pending: PendingDownload,
        dsts: &mut [&mut [u8]],
    ) -> Result<(), String> {
        match pending.0 {
            Pending::Empty => Ok(()),
            Pending::Deferred(parts) => deferred(self, device, queue, &parts, dsts),
            Pending::Staged(s) => s.finish(device, dsts).map(drop),
        }
    }

    /// Hands back a buffer `upload` returned, once every command using it
    /// has been submitted. A strategy may reuse it for a later upload (queue
    /// order makes that safe); the default drops it.
    fn recycle(&self, buffer: wgpu::Buffer) {
        drop(buffer);
    }
}

/// Smallest readback staging buffer: small results share one size class.
const MIN_STAGING: u64 = 64 << 10;
/// Most staging buffers kept for reuse.
const MAX_POOLED: usize = 4;

/// `data` into `buffer` at offset 0 through `queue.write_buffer`, which
/// needs whole 4-byte words, with zeros after it up to `capacity` (the
/// buffer may be reused and hold old bytes).
fn write_via_queue(queue: &wgpu::Queue, buffer: &wgpu::Buffer, data: &[u8], capacity: u64) {
    let n = data.len();
    let whole = n & !3;
    if whole > 0 {
        queue.write_buffer(buffer, 0, &data[..whole]);
    }
    let mut tail = vec![0u8; (capacity as usize).saturating_sub(whole)];
    if !tail.is_empty() {
        tail[..n - whole].copy_from_slice(&data[whole..]);
        queue.write_buffer(buffer, whole as u64, &tail);
    }
}

/// An upload staging buffer and whether it is mapped (ready to write).
#[derive(Debug)]
struct UploadStaging {
    buf: wgpu::Buffer,
    ready: Arc<AtomicBool>,
}

/// Copies through host-visible staging buffers both ways: the input is
/// written into a mapped `MAP_WRITE` buffer and copied to device memory on
/// the GPU; results are copied into a `MAP_READ` buffer and read from it.
/// Staging buffers are kept and reused, so a transfer does not allocate
/// (and the OS does not zero) fresh host memory every time; several ranges
/// come back with one wait.
#[derive(Debug, Default)]
pub struct CopyingStrategy {
    staging: Mutex<Vec<wgpu::Buffer>>,
    uploads: Mutex<Vec<UploadStaging>>,
    inputs: DevicePool,
}

impl CopyingStrategy {
    /// A strategy with empty staging pools.
    pub fn new() -> Self {
        Self::default()
    }

    /// A pooled upload staging buffer of at least `size` bytes, mapped for
    /// writing. Buffers still being re-mapped after earlier uploads are
    /// skipped (every program's readback waits for the GPU, so by the next
    /// program they are normally ready).
    fn take_upload(&self, device: &wgpu::Device, size: u64) -> UploadStaging {
        // Let finished map requests run their callbacks; never blocks.
        let _ = device.poll(wgpu::PollType::Poll);
        let mut pool = self.uploads.lock().unwrap_or_else(|e| e.into_inner());
        let best = pool
            .iter()
            .enumerate()
            .filter(|(_, s)| s.buf.size() >= size && s.ready.load(Ordering::Acquire))
            .min_by_key(|(_, s)| s.buf.size())
            .map(|(i, _)| i);
        match best {
            Some(i) => pool.swap_remove(i),
            None => UploadStaging {
                buf: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("bytelanes upload"),
                    size: size_class(size.max(MIN_STAGING)),
                    usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: true,
                }),
                ready: Arc::new(AtomicBool::new(true)),
            },
        }
    }

    /// Starts re-mapping `s` for the next upload and returns it to the pool.
    fn give_upload(&self, s: UploadStaging) {
        s.ready.store(false, Ordering::Release);
        let ready = Arc::clone(&s.ready);
        s.buf.map_async(wgpu::MapMode::Write, .., move |r| {
            if r.is_ok() {
                ready.store(true, Ordering::Release);
            }
        });
        let mut pool = self.uploads.lock().unwrap_or_else(|e| e.into_inner());
        pool.push(s);
        if pool.len() > MAX_POOLED
            && let Some(i) = pool
                .iter()
                .enumerate()
                .min_by_key(|(_, s)| s.buf.size())
                .map(|(i, _)| i)
        {
            pool.swap_remove(i);
        }
    }

    /// A pooled `MAP_READ` buffer of at least `size` bytes.
    fn take(&self, device: &wgpu::Device, size: u64) -> wgpu::Buffer {
        let mut pool = self.staging.lock().unwrap_or_else(|e| e.into_inner());
        let best = pool
            .iter()
            .enumerate()
            .filter(|(_, b)| b.size() >= size)
            .min_by_key(|(_, b)| b.size())
            .map(|(i, _)| i);
        match best {
            Some(i) => pool.swap_remove(i),
            None => device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("bytelanes readback"),
                size: size_class(size.max(MIN_STAGING)),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        }
    }

    /// Returns `buf` to the pool, dropping the smallest if it is full.
    fn give(&self, buf: wgpu::Buffer) {
        let mut pool = self.staging.lock().unwrap_or_else(|e| e.into_inner());
        pool.push(buf);
        if pool.len() > MAX_POOLED
            && let Some(i) = pool
                .iter()
                .enumerate()
                .min_by_key(|(_, b)| b.size())
                .map(|(i, _)| i)
        {
            pool.swap_remove(i);
        }
    }
}

impl BufferStrategy for CopyingStrategy {
    fn name(&self) -> &'static str {
        "copying"
    }

    fn upload(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data: &[u8],
        capacity: u64,
    ) -> wgpu::Buffer {
        // A pooled device buffer: it may hold an earlier program's bytes, so
        // everything after the data up to `capacity` is zeroed.
        let capacity = capacity.max(16);
        let (buffer, _) = self.inputs.take(device, capacity, "bytelanes input");
        let n = data.len();
        // Copies move whole 4-byte words; the tail word is zero-padded.
        let size = (n as u64).next_multiple_of(4);
        // Small inputs: wgpu's own staging suballocates and is cheaper.
        if size < MIN_STAGING {
            write_via_queue(queue, &buffer, data, capacity);
            return buffer;
        }
        let s = self.take_upload(device, size);
        let written = match s.buf.get_mapped_range_mut(..size) {
            Ok(mut view) => {
                let mut w = view.slice(..);
                w.slice(..n).copy_from_slice(data);
                w.slice(n..).fill(0);
                true
            }
            Err(_) => false,
        };
        if !written {
            // Not mapped after all: fall back to wgpu's own staging.
            drop(s);
            write_via_queue(queue, &buffer, data, capacity);
            return buffer;
        }
        s.buf.unmap();
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("bytelanes upload"),
        });
        enc.copy_buffer_to_buffer(&s.buf, 0, &buffer, 0, size);
        if capacity > size {
            enc.clear_buffer(&buffer, size, Some(capacity - size));
        }
        queue.submit([enc.finish()]);
        self.give_upload(s);
        buffer
    }

    fn recycle(&self, buffer: wgpu::Buffer) {
        self.inputs.give(buffer);
    }

    fn download(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffer: &wgpu::Buffer,
        offset: u64,
        len: u64,
    ) -> Result<Vec<u8>, String> {
        let mut out = vec![0u8; len as usize];
        self.download_into(
            device,
            queue,
            &mut [Readback {
                buffer,
                offset,
                dst: &mut out,
            }],
        )?;
        Ok(out)
    }

    fn download_into(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        parts: &mut [Readback<'_>],
    ) -> Result<(), String> {
        let ranges: Vec<ReadRange<'_>> = parts
            .iter()
            .map(|p| ReadRange {
                buffer: p.buffer,
                offset: p.offset,
                len: p.dst.len() as u64,
            })
            .collect();
        let pending = self.begin_download(device, queue, &ranges)?;
        let mut dsts: Vec<&mut [u8]> = parts.iter_mut().map(|p| &mut *p.dst).collect();
        self.finish_download(device, queue, pending, &mut dsts)
    }

    fn begin_download(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        parts: &[ReadRange<'_>],
    ) -> Result<PendingDownload, String> {
        // Each part's slot in the staging buffer, rounded up to the 4-byte
        // copy alignment.
        let mut slots = Vec::with_capacity(parts.len());
        let mut total = 0u64;
        for p in parts {
            slots.push((total, p.len));
            total += p.len.next_multiple_of(4);
        }
        if total == 0 {
            return Ok(PendingDownload(Pending::Empty));
        }
        let staging = self.take(device, total);
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("bytelanes readback"),
        });
        for (p, &(slot, len)) in parts.iter().zip(&slots) {
            let size = len.next_multiple_of(4);
            if size > 0 {
                enc.copy_buffer_to_buffer(p.buffer, p.offset, &staging, slot, size);
            }
        }
        let index = queue.submit([enc.finish()]);
        // Mapped ranges are 8-byte aligned; the pooled buffer is larger.
        let mapped = total.next_multiple_of(wgpu::MAP_ALIGNMENT);
        let (tx, rx) = mpsc::channel();
        staging.map_async(wgpu::MapMode::Read, ..mapped, move |r| {
            let _ = tx.send(r);
        });
        Ok(PendingDownload(Pending::Staged(Staged {
            staging,
            slots,
            mapped,
            index,
            rx,
        })))
    }

    fn finish_download(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pending: PendingDownload,
        dsts: &mut [&mut [u8]],
    ) -> Result<(), String> {
        match pending.0 {
            Pending::Empty => Ok(()),
            Pending::Deferred(parts) => deferred(self, device, queue, &parts, dsts),
            Pending::Staged(s) => {
                let staging = s.finish(device, dsts)?;
                self.give(staging);
                Ok(())
            }
        }
    }
}

/// A range of a GPU buffer to read back.
pub struct ReadRange<'a> {
    /// The GPU buffer to read.
    pub buffer: &'a wgpu::Buffer,
    /// Byte offset in `buffer`; a multiple of 4.
    pub offset: u64,
    /// Bytes to read.
    pub len: u64,
}

/// A readback started by [`BufferStrategy::begin_download`] and completed
/// by [`BufferStrategy::finish_download`].
pub struct PendingDownload(Pending);

impl PendingDownload {
    /// A readback that does all its work in `finish_download`, which is
    /// what the default `begin_download` returns.
    pub fn deferred(parts: &[ReadRange<'_>]) -> Self {
        PendingDownload(Pending::Deferred(
            parts
                .iter()
                .map(|p| (p.buffer.clone(), p.offset, p.len))
                .collect(),
        ))
    }
}

enum Pending {
    Empty,
    Deferred(Vec<(wgpu::Buffer, u64, u64)>),
    Staged(Staged),
}

/// Copies already submitted into a staging buffer, with its map requested.
struct Staged {
    staging: wgpu::Buffer,
    /// (offset in `staging`, length) per part.
    slots: Vec<(u64, u64)>,
    mapped: u64,
    index: wgpu::SubmissionIndex,
    rx: mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
}

impl Staged {
    /// Waits for the copies, fills `dsts` and returns the unmapped staging
    /// buffer.
    fn finish(self, device: &wgpu::Device, dsts: &mut [&mut [u8]]) -> Result<wgpu::Buffer, String> {
        device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(self.index),
                timeout: None,
            })
            .map_err(|e| format!("device poll failed: {e}"))?;
        self.rx
            .recv()
            .map_err(|_| "map callback never ran".to_owned())?
            .map_err(|e| format!("buffer map failed: {e}"))?;
        {
            let view = self
                .staging
                .get_mapped_range(..self.mapped)
                .map_err(|e| format!("mapped range: {e:?}"))?;
            for (dst, &(slot, len)) in dsts.iter_mut().zip(&self.slots) {
                let (slot, len) = (slot as usize, len as usize);
                dst[..len].copy_from_slice(&view[slot..slot + len]);
            }
        }
        self.staging.unmap();
        Ok(self.staging)
    }
}

/// Completes a deferred readback through `download_into`.
fn deferred<S: BufferStrategy + ?Sized>(
    s: &S,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    parts: &[(wgpu::Buffer, u64, u64)],
    dsts: &mut [&mut [u8]],
) -> Result<(), String> {
    let mut reads: Vec<Readback<'_>> = parts
        .iter()
        .zip(dsts.iter_mut())
        .map(|((buffer, offset, _), dst)| Readback {
            buffer,
            offset: *offset,
            dst,
        })
        .collect();
    s.download_into(device, queue, &mut reads)
}

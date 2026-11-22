//! How host bytes reach the GPU and come back.
//!
//! This trait is the seam between the ops (agent B) and the memory model:
//! [`CopyingStrategy`] stages every transfer through a copy, which is what a
//! discrete GPU on PCIe needs. A unified-memory strategy (agent C, M4) can
//! hand the GPU host-visible memory instead. The ops only ever see a
//! `wgpu::Buffer` with the input at offset 0.

use std::sync::{Mutex, mpsc};

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
}

/// Smallest readback staging buffer: small results share one size class.
const MIN_STAGING: u64 = 64 << 10;
/// Most staging buffers kept for reuse.
const MAX_POOLED: usize = 4;

/// Copies through `queue.write_buffer` on the way in and a mappable staging
/// buffer on the way out. Staging buffers are kept and reused, so a
/// readback does not allocate (and the OS does not zero) fresh host memory
/// every time; several ranges come back with one wait.
#[derive(Debug, Default)]
pub struct CopyingStrategy {
    staging: Mutex<Vec<wgpu::Buffer>>,
}

impl CopyingStrategy {
    /// A strategy with an empty staging pool.
    pub fn new() -> Self {
        Self::default()
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
                size: size.max(MIN_STAGING).next_power_of_two(),
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
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bytelanes input"),
            size: capacity.max(16),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !data.is_empty() {
            // write_buffer needs a multiple of 4 bytes.
            let whole = data.len() & !3;
            if whole > 0 {
                queue.write_buffer(&buffer, 0, &data[..whole]);
            }
            if whole < data.len() {
                let mut tail = [0u8; 4];
                tail[..data.len() - whole].copy_from_slice(&data[whole..]);
                queue.write_buffer(&buffer, whole as u64, &tail);
            }
        }
        buffer
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
        // Each part's slot in the staging buffer, rounded up to the 4-byte
        // copy alignment.
        let mut at = Vec::with_capacity(parts.len());
        let mut total = 0u64;
        for p in parts.iter() {
            at.push(total);
            total += (p.dst.len() as u64).next_multiple_of(4);
        }
        if total == 0 {
            return Ok(());
        }
        let staging = self.take(device, total);
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("bytelanes readback"),
        });
        for (p, &slot) in parts.iter().zip(&at) {
            let size = (p.dst.len() as u64).next_multiple_of(4);
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
        device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(index),
                timeout: None,
            })
            .map_err(|e| format!("device poll failed: {e}"))?;
        rx.recv()
            .map_err(|_| "map callback never ran".to_owned())?
            .map_err(|e| format!("buffer map failed: {e}"))?;
        {
            let view = staging
                .get_mapped_range(..mapped)
                .map_err(|e| format!("mapped range: {e:?}"))?;
            for (p, &slot) in parts.iter_mut().zip(&at) {
                let n = p.dst.len();
                p.dst
                    .copy_from_slice(&view[slot as usize..slot as usize + n]);
            }
        }
        staging.unmap();
        self.give(staging);
        Ok(())
    }
}

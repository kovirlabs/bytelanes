//! How host bytes reach the GPU and come back.
//!
//! This trait is the seam between the ops (agent B) and the memory model:
//! [`CopyingStrategy`] stages every transfer through a copy, which is what a
//! discrete GPU on PCIe needs. [`UnifiedMemoryStrategy`](crate::UnifiedMemoryStrategy)
//! (agent C, M4) hands the GPU memory the CPU maps directly instead. The ops
//! only ever see a `wgpu::Buffer` with the input at offset 0.

use std::sync::mpsc;

/// Moves bytes between host memory and GPU buffers.
pub trait BufferStrategy: Send + Sync {
    /// Short name, recorded with results.
    fn name(&self) -> &'static str;

    /// Device features this strategy needs. The backend requests them when
    /// it opens the device, so opening fails on an adapter without them.
    fn required_features(&self) -> wgpu::Features {
        wgpu::Features::empty()
    }

    /// A zero-filled buffer of `size` bytes (a multiple of 4) with at least
    /// `STORAGE | COPY_SRC | COPY_DST` usage. Every op result and device
    /// (`Vram`) buffer comes from here, so a strategy can choose memory that
    /// its [`download`](Self::download) reads without a copy.
    fn alloc(&self, device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
        // New wgpu buffers are zero-initialised.
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

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
}

/// Copies through `queue.write_buffer` on the way in and a mappable staging
/// buffer on the way out.
#[derive(Clone, Copy, Debug, Default)]
pub struct CopyingStrategy;

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
        if len == 0 {
            return Ok(Vec::new());
        }
        let size = len.next_multiple_of(4);
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bytelanes readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("bytelanes readback"),
        });
        enc.copy_buffer_to_buffer(buffer, offset, &staging, 0, size);
        let index = queue.submit([enc.finish()]);
        let (tx, rx) = mpsc::channel();
        staging.map_async(wgpu::MapMode::Read, .., move |r| {
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
        let bytes = {
            let view = staging
                .get_mapped_range(..)
                .map_err(|e| format!("mapped range: {e:?}"))?;
            view[..len as usize].to_vec()
        };
        staging.unmap();
        Ok(bytes)
    }
}

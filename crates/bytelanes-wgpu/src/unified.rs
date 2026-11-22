//! [`UnifiedMemoryStrategy`]: host and GPU share one memory, so skip the
//! staging copies.
//!
//! On a GPU with unified memory (Apple silicon through Metal), a buffer the
//! CPU can map is the same memory the GPU computes on. wgpu's Metal backend
//! creates every buffer with `MAP_READ` or `MAP_WRITE` usage in
//! `MTLStorageModeShared`, and mapping it returns the buffer's own pointer,
//! with no copy (wgpu-hal 30, `metal/device.rs`, `create_buffer` and
//! `map_buffer`). WebGPU forbids `MAP_*` together with `STORAGE`;
//! [`wgpu::Features::MAPPABLE_PRIMARY_BUFFERS`] lifts that, and Metal offers
//! it.
//!
//! Compared with [`CopyingStrategy`]:
//!
//! | Step | `CopyingStrategy` | `UnifiedMemoryStrategy` |
//! | --- | --- | --- |
//! | Upload | CPU copy into a wgpu staging buffer, then a GPU copy into the input buffer | CPU copy straight into the input buffer |
//! | Download | GPU copy into a new mappable buffer, then a CPU copy out of it | CPU copy straight out of the result buffer |
//!
//! Both GPU copies and both staging buffers go away. One CPU copy remains
//! in each direction, because the backend's host buffers are ordinary
//! `Vec<u8>`s. Removing that copy too means the GPU working on the caller's
//! own memory (Metal's `newBufferWithBytesNoCopy`). wgpu reaches that only
//! through `Device::create_buffer_from_hal`, which is `unsafe`, and this
//! crate forbids `unsafe`.
//!
//! [`CopyingStrategy`]: crate::CopyingStrategy

use std::sync::mpsc;

use crate::strategy::{BufferStrategy, CopyingStrategy};

/// Moves bytes by mapping the GPU's own buffers, for GPUs that share memory
/// with the CPU. Build one with [`UnifiedMemoryStrategy::for_adapter`].
#[derive(Clone, Copy, Debug)]
pub struct UnifiedMemoryStrategy {
    _private: (),
}

impl UnifiedMemoryStrategy {
    /// The strategy for `adapter`, or why it does not apply: the adapter must
    /// be an integrated GPU (wgpu's Metal backend reports Apple silicon's
    /// unified memory this way) and must offer
    /// [`wgpu::Features::MAPPABLE_PRIMARY_BUFFERS`].
    ///
    /// On a discrete GPU, mappable memory sits across the PCIe link, and
    /// running ops on it would be slow, so this refuses one.
    pub fn for_adapter(adapter: &wgpu::Adapter) -> Result<Self, String> {
        let info = adapter.get_info();
        if info.device_type != wgpu::DeviceType::IntegratedGpu {
            return Err(format!(
                "{} ({:?}) is a {:?}, not an integrated GPU with unified memory",
                info.name, info.backend, info.device_type
            ));
        }
        if !adapter
            .features()
            .contains(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS)
        {
            return Err(format!(
                "{} ({:?}) does not offer MAPPABLE_PRIMARY_BUFFERS",
                info.name, info.backend
            ));
        }
        Ok(UnifiedMemoryStrategy { _private: () })
    }
}

/// Usage every buffer handed to the ops must have.
const OP_USAGE: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE
    .union(wgpu::BufferUsages::COPY_SRC)
    .union(wgpu::BufferUsages::COPY_DST);

impl BufferStrategy for UnifiedMemoryStrategy {
    fn name(&self) -> &'static str {
        "unified"
    }

    fn required_features(&self) -> wgpu::Features {
        wgpu::Features::MAPPABLE_PRIMARY_BUFFERS
    }

    /// Result and device buffers are readable by the CPU (`MAP_READ`), so
    /// [`download`](Self::download) maps them in place. Not `MAP_WRITE`:
    /// Metal makes those write-combined, which is slow for the CPU to read.
    fn alloc(&self, device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: OP_USAGE | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        })
    }

    /// The input buffer is created mapped (`MAP_WRITE`), filled through the
    /// mapping, and unmapped: on unified memory that writes the GPU's own
    /// memory once, with no staging buffer and no GPU copy.
    ///
    /// # Panics
    ///
    /// If wgpu refuses a view of a buffer it has just created mapped, which
    /// would be a wgpu bug. The backend turns a panic into `DeviceError`.
    fn upload(
        &self,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        data: &[u8],
        capacity: u64,
    ) -> wgpu::Buffer {
        let size = capacity.max(16).next_multiple_of(4);
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bytelanes input (unified)"),
            size,
            usage: OP_USAGE | wgpu::BufferUsages::MAP_WRITE,
            mapped_at_creation: true,
        });
        if !data.is_empty() {
            // wgpu zero-fills the rest of a buffer mapped at creation.
            let mut view = buffer
                .get_mapped_range_mut(..)
                .expect("a buffer created mapped can be viewed");
            view.slice(..data.len()).copy_from_slice(data);
        }
        buffer.unmap();
        buffer
    }

    /// Maps the result buffer itself and copies the bytes out. A buffer
    /// without `MAP_READ` (not made by [`alloc`](Self::alloc)) goes through
    /// [`CopyingStrategy`] instead.
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
        if !buffer.usage().contains(wgpu::BufferUsages::MAP_READ) {
            return CopyingStrategy::new().download(device, queue, buffer, offset, len);
        }
        // A mapping starts at a multiple of 8 and ends at a multiple of 4.
        let start = offset - offset % wgpu::MAP_ALIGNMENT;
        let end = (offset + len).next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT);
        if end > buffer.size() {
            return Err(format!(
                "download of {offset}..{} past the end of a {}-byte buffer",
                offset + len,
                buffer.size()
            ));
        }
        let (tx, rx) = mpsc::channel();
        buffer.map_async(wgpu::MapMode::Read, start..end, move |r| {
            let _ = tx.send(r);
        });
        // The mapping resolves once all work using the buffer has finished;
        // waiting for everything submitted also covers that.
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|e| format!("device poll failed: {e}"))?;
        rx.recv()
            .map_err(|_| "map callback never ran".to_owned())?
            .map_err(|e| format!("buffer map failed: {e}"))?;
        let bytes = {
            let view = buffer
                .get_mapped_range(start..end)
                .map_err(|e| format!("mapped range: {e:?}"))?;
            let skip = (offset - start) as usize;
            view[skip..skip + len as usize].to_vec()
        };
        buffer.unmap();
        Ok(bytes)
    }
}

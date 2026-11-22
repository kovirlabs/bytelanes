//! A pool of device buffers (`STORAGE | COPY_SRC | COPY_DST`), reused across
//! programs.
//!
//! Measured on the RX 7800 XT (M5): when the device buffers one program
//! allocates add up to more than about 128 MiB, each program paid a fixed
//! 10 to 13 ms, on Vulkan and DX12 alike, consistent with the allocator
//! creating and releasing a fresh memory block on every call. Reusing the
//! buffers removes that cost.
//!
//! Reuse is safe as soon as every command using a buffer has been submitted:
//! the queue runs submissions in order, so a later program's writes to a
//! reused buffer cannot overtake an earlier program's reads of it.

use std::sync::Mutex;

/// Most buffers a pool keeps.
const MAX_BUFFERS: usize = 8;
/// Most bytes a pool keeps.
const MAX_BYTES: u64 = 1536 << 20;
/// Smallest size class.
const MIN_CLASS: u64 = 64 << 10;

/// The size actually allocated for a request of `n` bytes: rounded up to an
/// eighth of the next power of two (at most 12.5% to 25% extra), so nearby
/// sizes share buffers.
pub(crate) fn size_class(n: u64) -> u64 {
    let p = n.max(1).next_power_of_two();
    n.next_multiple_of((p / 8).max(MIN_CLASS))
}

#[derive(Debug, Default)]
pub(crate) struct DevicePool {
    bufs: Mutex<Vec<wgpu::Buffer>>,
}

impl DevicePool {
    /// A buffer of at least `size` bytes, and whether it was reused (its
    /// contents are then whatever the last user left). A new buffer comes
    /// from `alloc`, called with the size class of `size` (at most the
    /// device's largest bindable buffer); it must have at least
    /// `STORAGE | COPY_SRC | COPY_DST` usage.
    pub(crate) fn take(
        &self,
        device: &wgpu::Device,
        size: u64,
        alloc: impl FnOnce(u64) -> wgpu::Buffer,
    ) -> (wgpu::Buffer, bool) {
        {
            let mut pool = self.bufs.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(i) = pool
                .iter()
                .enumerate()
                .filter(|(_, b)| b.size() >= size)
                .min_by_key(|(_, b)| b.size())
                .map(|(i, _)| i)
            {
                return (pool.swap_remove(i), true);
            }
        }
        let limits = device.limits();
        let cap = limits
            .max_buffer_size
            .min(limits.max_storage_buffer_binding_size);
        // Rounded to the 4-byte copy alignment, as `alloc` requires.
        let class = size_class(size).min(cap).max(size).next_multiple_of(4);
        (alloc(class), false)
    }

    /// Keeps `buf` for reuse, dropping the smallest buffers while the pool
    /// is over its count or byte limit.
    pub(crate) fn give(&self, buf: wgpu::Buffer) {
        let mut pool = self.bufs.lock().unwrap_or_else(|e| e.into_inner());
        pool.push(buf);
        while pool.len() > MAX_BUFFERS
            || pool.iter().map(wgpu::Buffer::size).sum::<u64>() > MAX_BYTES
        {
            let Some(i) = pool
                .iter()
                .enumerate()
                .min_by_key(|(_, b)| b.size())
                .map(|(i, _)| i)
            else {
                break;
            };
            pool.swap_remove(i);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::size_class;

    #[test]
    fn size_classes() {
        assert_eq!(size_class(1), 64 << 10);
        assert_eq!(size_class(64 << 20), 64 << 20);
        assert_eq!(size_class((64 << 20) + 16), 80 << 20);
        assert_eq!(size_class((256 << 20) + 16), 320 << 20);
        for n in [1u64, 1000, 65_537, 3 << 20, (100 << 20) + 3] {
            let c = size_class(n);
            assert!(c >= n && c <= n + n / 4 + (64 << 10), "{n} -> {c}");
        }
    }
}

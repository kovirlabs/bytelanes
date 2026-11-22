//! bytelanes-wgpu: the GPU backend, through wgpu.
//!
//! [`WgpuBackend`] implements [`bytelanes::ChannelBackend`] and
//! [`bytelanes::Buffers`] with WGSL compute shaders, on Vulkan or DX12
//! (the Ryzen's RX 7800 XT) and Metal (the MacBook). It reproduces the CPU
//! reference in `bytelanes::cpu` byte for byte:
//!
//! - **Checksum** (CRC-32C): each thread takes the CRC of a 256-byte chunk with
//!   a byte table, then a tree of passes combines neighbours with
//!   carry-less multiplication by powers of x.
//! - **Encrypt / Decrypt** (AES-256-CTR): one thread per 16-byte block, with
//!   round tables in workgroup memory.
//! - **Filter**: every element becomes an integer key that orders like the
//!   value, so NaN, signed zero and subnormal floats behave as in IEEE 754
//!   whatever the GPU does with floats; 64-bit types are pairs of `u32`.
//!   Count, scan and scatter passes write indices in ascending order.
//!
//! Status checks run on the host in the reference's order; every byte of
//! data work, including the CRC-32C of each result, runs on the GPU.
//!
//! How bytes move between host and GPU is a [`BufferStrategy`]. Agent B
//! owns the ops and shaders; agent C owns the unified-memory strategy.

#![forbid(unsafe_code)]

mod backend;
mod gpu;
mod pool;
mod strategy;
mod tables;

pub use backend::{DEFAULT_QUEUE_DEPTH, Error, WgpuBackend};
pub use strategy::{BufferStrategy, CopyingStrategy, Readback};
pub use wgpu;

/// Every adapter wgpu finds for `backends`, software ones included.
pub fn adapters(backends: wgpu::Backends) -> Vec<wgpu::Adapter> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    pollster::block_on(instance.enumerate_adapters(backends))
}

/// The hardware adapters (no software rasterisers) for the native APIs this
/// crate targets: Vulkan, DX12 and Metal. `WGPU_BACKEND` narrows the set,
/// for example `WGPU_BACKEND=dx12`.
pub fn hardware_adapters() -> Vec<wgpu::Adapter> {
    let wanted = wgpu::Backends::VULKAN | wgpu::Backends::DX12 | wgpu::Backends::METAL;
    let backends = wgpu::Backends::from_env().map_or(wanted, |b| b & wanted);
    adapters(backends)
        .into_iter()
        .filter(|a| a.get_info().device_type != wgpu::DeviceType::Cpu)
        .collect()
}

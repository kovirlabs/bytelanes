//! bytelanes-wgpu: GPU backend through wgpu.
//!
//! WGSL compute shaders run through Vulkan or DX12 on the Ryzen and Metal on the
//! MacBook, behind a `BufferStrategy` trait with copying and unified-memory
//! implementations. Agent B owns ops and shaders; agent C owns the unified-memory
//! `BufferStrategy` module.
//!
//! Status: stub. Built in M3 and M4 (spec: `cf-backend-wgpu`).

#![forbid(unsafe_code)]

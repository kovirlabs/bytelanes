//! `UnifiedMemoryStrategy` on its own: byte-exact round trips at every
//! offset and length the backend can ask for, and the fallback for buffers
//! it did not allocate. The ops through this strategy are covered by
//! `conformance.rs` and `large.rs`, which run every applicable strategy.
//!
//! Adapters without unified memory are skipped, saying so.

mod common;

use bytelanes::conformance::SplitMix64;
use bytelanes_wgpu::{BufferStrategy, UnifiedMemoryStrategy, wgpu};

/// Devices opened with the features the strategy needs, one per adapter
/// that has unified memory.
fn devices() -> Vec<(UnifiedMemoryStrategy, wgpu::Device, wgpu::Queue)> {
    let mut out = Vec::new();
    for a in common::adapters() {
        let Ok(s) = UnifiedMemoryStrategy::for_adapter(&a) else {
            eprintln!("SKIPPED: {} has no unified memory", a.get_info().name);
            continue;
        };
        let (device, queue) = pollster::block_on(a.request_device(&wgpu::DeviceDescriptor {
            label: Some("unified test"),
            required_features: s.required_features(),
            ..Default::default()
        }))
        .expect("open device");
        out.push((s, device, queue));
    }
    out
}

#[test]
fn upload_then_download_is_exact_at_every_offset_and_length() {
    for (s, device, queue) in devices() {
        let data = SplitMix64::new(7).bytes(203);
        let buf = s.upload(&device, &queue, &data, 224);
        assert!(buf.size() >= 224);
        assert!(buf.usage().contains(
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST
        ));
        // upload's buffer is MAP_WRITE only, so this exercises the fallback.
        for offset in (0..48).step_by(4) {
            for len in 0..40u64 {
                let got = s
                    .download(&device, &queue, &buf, offset, len)
                    .expect("download");
                let o = offset as usize;
                assert_eq!(got, data[o..o + len as usize], "offset {offset} len {len}");
            }
        }
    }
}

#[test]
fn alloc_buffers_map_in_place_after_gpu_work() {
    for (s, device, queue) in devices() {
        let data = SplitMix64::new(11).bytes(4096);
        let src = s.upload(&device, &queue, &data, 4096);
        let dst = s.alloc(&device, "dst", 4096 + 16);
        assert!(dst.usage().contains(wgpu::BufferUsages::MAP_READ));
        assert!(!dst.usage().contains(wgpu::BufferUsages::MAP_WRITE));

        // A fresh buffer reads as zeros.
        assert_eq!(
            s.download(&device, &queue, &dst, 4, 64).unwrap(),
            vec![0; 64]
        );

        // GPU work writes dst; download waits for it and maps dst directly.
        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&src, 0, &dst, 16, 4096);
        queue.submit([enc.finish()]);
        for offset in [16u64, 20, 24, 28, 1000, 4096] {
            for len in [1u64, 3, 4, 5, 8, 13] {
                let got = s.download(&device, &queue, &dst, offset, len).unwrap();
                let o = offset as usize - 16;
                assert_eq!(got, data[o..o + len as usize], "offset {offset} len {len}");
            }
        }
        // The buffer is unmapped again, so GPU work can use it.
        let mut enc = device.create_command_encoder(&Default::default());
        enc.clear_buffer(&dst, 0, None);
        queue.submit([enc.finish()]);
        assert_eq!(s.download(&device, &queue, &dst, 16, 8).unwrap(), [0; 8]);

        assert!(s.download(&device, &queue, &dst, 4096, 64).is_err());
    }
}

#[test]
fn refuses_adapters_without_unified_memory() {
    for a in common::adapters() {
        let r = UnifiedMemoryStrategy::for_adapter(&a);
        let integrated = a.get_info().device_type == wgpu::DeviceType::IntegratedGpu;
        let feature = a
            .features()
            .contains(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS);
        assert_eq!(r.is_ok(), integrated && feature, "{:?}", a.get_info());
    }
}

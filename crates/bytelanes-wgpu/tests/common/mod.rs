//! Shared by the GPU test files.

// Each test binary uses only some of these.
#![allow(dead_code)]

use bytelanes_wgpu::{
    BufferStrategy, CopyingStrategy, DEFAULT_QUEUE_DEPTH, UnifiedMemoryStrategy, WgpuBackend,
    hardware_adapters, wgpu,
};

/// The hardware adapters to test.
///
/// With no adapter the tests skip and say so, which is right for a machine
/// without a GPU. A GPU runner sets `BYTELANES_REQUIRE_BACKENDS` (for example
/// `vulkan,dx12`) so that a missing driver or adapter fails instead of
/// skipping.
pub fn adapters() -> Vec<wgpu::Adapter> {
    let found = hardware_adapters();
    let required = std::env::var("BYTELANES_REQUIRE_BACKENDS").unwrap_or_default();
    for name in required.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let have = found
            .iter()
            .any(|a| format!("{:?}", a.get_info().backend).eq_ignore_ascii_case(name));
        assert!(
            have,
            "BYTELANES_REQUIRE_BACKENDS names {name}, but no hardware adapter uses it"
        );
    }
    if found.is_empty() {
        eprintln!("SKIPPED: no hardware GPU adapter found");
    }
    found
}

/// Every buffer strategy that applies to `adapter`: always copying, plus
/// unified memory where the adapter allows it.
pub fn strategies(adapter: &wgpu::Adapter) -> Vec<Box<dyn BufferStrategy>> {
    let mut out: Vec<Box<dyn BufferStrategy>> = vec![Box::new(CopyingStrategy)];
    if let Ok(s) = UnifiedMemoryStrategy::for_adapter(adapter) {
        out.push(Box::new(s));
    }
    out
}

/// A backend for every hardware adapter and every strategy that applies to
/// it, so each strategy has to reproduce the reference on its own.
pub fn backends() -> Vec<WgpuBackend> {
    adapters()
        .iter()
        .flat_map(|a| {
            strategies(a).into_iter().map(move |s| {
                WgpuBackend::with_strategy(a, s, DEFAULT_QUEUE_DEPTH).expect("open adapter")
            })
        })
        .collect()
}

/// Adapter, API, driver, device type and strategy, for test output.
pub fn describe(b: &WgpuBackend) -> String {
    let i = b.adapter_info();
    format!(
        "{} | {:?} | driver {} {} | {:?} | {}",
        i.name,
        i.backend,
        i.driver,
        i.driver_info,
        i.device_type,
        b.strategy_name()
    )
}

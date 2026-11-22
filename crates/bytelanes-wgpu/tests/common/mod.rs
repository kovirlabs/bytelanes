//! Shared by the GPU test files.

use bytelanes_wgpu::{hardware_adapters, wgpu};

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

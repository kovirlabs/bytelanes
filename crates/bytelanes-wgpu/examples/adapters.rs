//! Lists the hardware adapters this crate would use, with the limits that
//! bound input sizes.
//!
//! `cargo run -p bytelanes-wgpu --example adapters`

fn main() {
    let found = bytelanes_wgpu::hardware_adapters();
    if found.is_empty() {
        println!("no hardware GPU adapter found");
    }
    for a in found {
        let i = a.get_info();
        let l = a.limits();
        println!(
            "{} | {:?} | {:?} | driver: {} {} | PCI bus: {} | max_buffer_size {} | \
             max_storage_buffer_binding_size {}",
            i.name,
            i.backend,
            i.device_type,
            i.driver,
            i.driver_info,
            i.device_pci_bus_id,
            l.max_buffer_size,
            l.max_storage_buffer_binding_size
        );
    }
}

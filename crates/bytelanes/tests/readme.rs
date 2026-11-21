//! The example in README.md, kept compiling and passing.

use bytelanes::Buffers;
use bytelanes::abi::{BufId, ChannelCmd, ChannelProgram, Extent, Tag};
use bytelanes::cpu::CpuBackend;

#[test]
fn readme_example() {
    let cpu = CpuBackend::new();
    cpu.put_host(BufId(1), b"123456789".to_vec()).unwrap();
    cpu.put_host(BufId(2), vec![0; 4]).unwrap();
    let prog = ChannelProgram::from_commands(&[ChannelCmd::checksum(
        Extent::host(BufId(1), 0, 9),
        Extent::host(BufId(2), 0, 4),
    )])
    .unwrap();
    let done = cpu.run(&prog, Tag(1));
    let result = cpu.get_host(BufId(2)).unwrap();
    bytelanes::verify(&done, &result).unwrap();
    assert_eq!(result, 0xE306_9283u32.to_le_bytes());
}

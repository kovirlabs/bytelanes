//! Reading and writing golden vectors on disk.
//!
//! Layout, one directory per case:
//!
//! ```text
//! vectors/<group>/<case>/
//!   case.json      buffers, keys and expected completion
//!   program.bin    the ChannelProgram wire bytes (1672)
//!   host-<id>.bin  initial contents of host buffer <id>
//!   key-<slot>.bin a 32-byte AES-256 key
//!   expected.bin   the result bytes (only when the status is Ok)
//! ```

use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::{Case, Expect};
use crate::abi::zerocopy::{FromBytes, IntoBytes};
use crate::abi::{BufId, ChannelProgram, Status};

/// Version of the `case.json` format.
pub const CASE_FORMAT: u32 = 1;

/// The `vectors/` directory of this repository.
pub fn vectors_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../vectors")
}

/// Failure to read or write vectors.
#[derive(Debug)]
pub enum VectorError {
    /// File system error, with the path involved.
    Io(PathBuf, std::io::Error),
    /// Malformed `case.json` or binary file.
    Format(PathBuf, String),
}

impl fmt::Display for VectorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VectorError::Io(p, e) => write!(f, "{}: {e}", p.display()),
            VectorError::Format(p, m) => write!(f, "{}: {m}", p.display()),
        }
    }
}

impl std::error::Error for VectorError {}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CaseFile {
    format: u32,
    name: String,
    description: String,
    program: String,
    host: Vec<HostEntry>,
    vram: Vec<VramEntry>,
    keys: Vec<KeyEntry>,
    expect: ExpectFile,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct HostEntry {
    buf: u64,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    file: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    zeroed: Option<u64>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct VramEntry {
    buf: u64,
    len: u64,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct KeyEntry {
    slot: u32,
    file: String,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExpectFile {
    status: String,
    status_code: u16,
    cmd_index: u16,
    bytes: u32,
    crc: u32,
    output_buf: u64,
    output_offset: u64,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    output: Option<String>,
}

fn io<T>(path: &Path, r: std::io::Result<T>) -> Result<T, VectorError> {
    r.map_err(|e| VectorError::Io(path.to_path_buf(), e))
}

fn write(path: &Path, data: &[u8]) -> Result<(), VectorError> {
    io(path, fs::write(path, data))
}

/// Write `cases` under `dir`, one directory per case. Existing files of the
/// same names are replaced; nothing else is removed.
pub fn write_vectors(dir: &Path, cases: &[Case]) -> Result<(), VectorError> {
    for case in cases {
        let cdir = dir.join(&case.name);
        io(&cdir, fs::create_dir_all(&cdir))?;
        write(&cdir.join("program.bin"), case.program.as_bytes())?;
        let mut host = Vec::new();
        for (buf, data) in &case.host {
            if !data.is_empty() && data.iter().all(|&b| b == 0) {
                host.push(HostEntry {
                    buf: buf.0,
                    file: None,
                    zeroed: Some(data.len() as u64),
                });
            } else {
                let f = format!("host-{}.bin", buf.0);
                write(&cdir.join(&f), data)?;
                host.push(HostEntry {
                    buf: buf.0,
                    file: Some(f),
                    zeroed: None,
                });
            }
        }
        let mut keys = Vec::new();
        for (slot, key) in &case.keys {
            let f = format!("key-{slot}.bin");
            write(&cdir.join(&f), key)?;
            keys.push(KeyEntry {
                slot: *slot,
                file: f,
            });
        }
        let e = &case.expect;
        let output = if e.status == Status::Ok {
            write(&cdir.join("expected.bin"), &e.output)?;
            Some("expected.bin".to_string())
        } else {
            None
        };
        let file = CaseFile {
            format: CASE_FORMAT,
            name: case.name.clone(),
            description: case.description.clone(),
            program: "program.bin".to_string(),
            host,
            vram: case
                .vram
                .iter()
                .map(|(b, l)| VramEntry { buf: b.0, len: *l })
                .collect(),
            keys,
            expect: ExpectFile {
                status: format!("{:?}", e.status),
                status_code: e.status.code(),
                cmd_index: e.cmd_index,
                bytes: e.bytes,
                crc: e.crc,
                output_buf: e.output_buf.0,
                output_offset: e.output_offset,
                output,
            },
        };
        let mut json = serde_json::to_string_pretty(&file)
            .map_err(|e| VectorError::Format(cdir.clone(), e.to_string()))?;
        json.push('\n');
        write(&cdir.join("case.json"), json.as_bytes())?;
    }
    Ok(())
}

/// Read every case under `dir`, sorted by name.
pub fn load_vectors(dir: impl AsRef<Path>) -> Result<Vec<Case>, VectorError> {
    let mut paths = Vec::new();
    find_cases(dir.as_ref(), &mut paths)?;
    paths.sort();
    let mut cases = paths
        .iter()
        .map(|p| load_case(p))
        .collect::<Result<Vec<_>, _>>()?;
    cases.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(cases)
}

fn find_cases(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), VectorError> {
    for entry in io(dir, fs::read_dir(dir))? {
        let path = io(dir, entry)?.path();
        if path.is_dir() {
            find_cases(&path, out)?;
        } else if path.file_name().is_some_and(|n| n == "case.json") {
            out.push(path);
        }
    }
    Ok(())
}

fn read(path: &Path) -> Result<Vec<u8>, VectorError> {
    io(path, fs::read(path))
}

fn load_case(json_path: &Path) -> Result<Case, VectorError> {
    let dir = json_path.parent().unwrap_or(Path::new("."));
    let bad = |m: &str| VectorError::Format(json_path.to_path_buf(), m.to_string());
    let text = read(json_path)?;
    let f: CaseFile = serde_json::from_slice(&text).map_err(|e| bad(&e.to_string()))?;
    if f.format != CASE_FORMAT {
        return Err(bad("unsupported case format"));
    }
    let prog_bytes = read(&dir.join(&f.program))?;
    let program = ChannelProgram::read_from_bytes(&prog_bytes)
        .map_err(|_| bad("program.bin has the wrong length"))?;
    let mut host = Vec::new();
    for h in &f.host {
        let data = match (&h.file, h.zeroed) {
            (Some(file), None) => read(&dir.join(file))?,
            (None, Some(n)) => vec![0; usize::try_from(n).map_err(|_| bad("zeroed too large"))?],
            _ => return Err(bad("host entry needs exactly one of file or zeroed")),
        };
        host.push((BufId(h.buf), data));
    }
    let mut keys = Vec::new();
    for k in &f.keys {
        let key: [u8; 32] = read(&dir.join(&k.file))?
            .try_into()
            .map_err(|_| bad("key file is not 32 bytes"))?;
        keys.push((k.slot, key));
    }
    let e = &f.expect;
    let status = Status::from_code(e.status_code).ok_or_else(|| bad("unknown status code"))?;
    let output = match &e.output {
        Some(file) => read(&dir.join(file))?,
        None => Vec::new(),
    };
    Ok(Case {
        name: f.name,
        description: f.description,
        program,
        host,
        vram: f.vram.iter().map(|v| (BufId(v.buf), v.len)).collect(),
        keys,
        expect: Expect {
            status,
            cmd_index: e.cmd_index,
            bytes: e.bytes,
            crc: e.crc,
            output_buf: BufId(e.output_buf),
            output_offset: e.output_offset,
            output,
        },
    })
}

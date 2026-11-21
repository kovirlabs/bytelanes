//! Channel programs and their validation.

use zerocopy::byteorder::little_endian::U16;
use zerocopy::{FromBytes, FromZeros, Immutable, IntoBytes, KnownLayout, Unaligned};

use crate::{ChannelCmd, ExtentRef, FORMAT_VERSION, Flags, MAX_COMMANDS, Op, Status};

/// Command index used in a [`ProgramError`] or [`Completion`](crate::Completion)
/// for a failure of the program as a whole rather than of one command.
pub const NO_INDEX: u16 = u16::MAX;

/// Why a program is invalid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProgramError {
    /// Failing command, or [`NO_INDEX`].
    pub index: u16,
    /// Status to report: always [`Status::InvalidProgram`] from validation.
    pub status: Status,
    /// Human-readable reason. Not part of the wire format.
    pub reason: &'static str,
}

impl ProgramError {
    fn at(index: usize, reason: &'static str) -> Self {
        ProgramError {
            index: index as u16,
            status: Status::InvalidProgram,
            reason,
        }
    }

    fn whole(reason: &'static str) -> Self {
        ProgramError {
            index: NO_INDEX,
            status: Status::InvalidProgram,
            reason,
        }
    }
}

impl core::fmt::Display for ProgramError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.index == NO_INDEX {
            write!(f, "invalid program: {}", self.reason)
        } else {
            write!(
                f,
                "invalid program at command {}: {}",
                self.index, self.reason
            )
        }
    }
}

impl core::error::Error for ProgramError {}

/// A channel program: up to [`MAX_COMMANDS`] commands, 1672 bytes.
///
/// | Bytes | Field |
/// | --- | --- |
/// | 0..2 | format version, [`FORMAT_VERSION`] |
/// | 2..4 | number of commands, 1 to 16 |
/// | 4..8 | reserved, zero |
/// | 8..1672 | 16 command slots of 104 bytes; slots past the count are zero |
///
/// # Rules checked by [`validate`](Self::validate)
///
/// 1. Version, count, reserved bytes and unused slots are as above.
/// 2. Every command passes [`ChannelCmd::check`].
/// 3. Every `dst` except the last command's is `Vram`: intermediate results
///    stay on the device and only the final result may land in host memory.
/// 4. A command with [`Flags::CHAIN`] is not the last, and the next command's
///    `src` equals its `dst`. The last command has no `CHAIN` flag.
/// 5. No `dst` overlaps any `src` in the same program, except the `src` that
///    a chain link feeds. A program therefore never overwrites its own input,
///    so running it twice gives the same result: programs are idempotent by
///    construction and safe to resubmit on another path.
///
/// Backend-specific checks (known buffers, bounds, key slots, supported ops)
/// happen when the program runs and are reported in its completion.
#[derive(
    FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned, Clone, Copy, PartialEq, Eq, Debug,
)]
#[repr(C)]
pub struct ChannelProgram {
    version: U16,
    len: U16,
    reserved: [u8; 4],
    cmds: [ChannelCmd; MAX_COMMANDS],
}

impl Default for ChannelProgram {
    fn default() -> Self {
        ChannelProgram::new()
    }
}

impl ChannelProgram {
    /// An empty program. Push at least one command before submitting it.
    pub fn new() -> Self {
        let mut p = ChannelProgram::new_zeroed();
        p.version = U16::new(FORMAT_VERSION);
        p
    }

    /// A validated program of the given commands.
    pub fn from_commands(cmds: &[ChannelCmd]) -> Result<Self, ProgramError> {
        let mut p = ChannelProgram::new();
        for cmd in cmds {
            p.push(*cmd)?;
        }
        p.validate()?;
        Ok(p)
    }

    /// Append a command. Fails only when the program is full; call
    /// [`validate`](Self::validate) once all commands are in.
    pub fn push(&mut self, cmd: ChannelCmd) -> Result<(), ProgramError> {
        let n = self.len();
        if n >= MAX_COMMANDS {
            return Err(ProgramError::whole("program already has 16 commands"));
        }
        self.cmds[n] = cmd;
        self.len = U16::new(n as u16 + 1);
        Ok(())
    }

    /// Format version.
    pub fn version(&self) -> u16 {
        self.version.get()
    }

    /// Number of commands, clamped to [`MAX_COMMANDS`].
    pub fn len(&self) -> usize {
        usize::from(self.len.get()).min(MAX_COMMANDS)
    }

    /// Whether the program has no commands.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The commands in use.
    pub fn commands(&self) -> &[ChannelCmd] {
        &self.cmds[..self.len()]
    }

    /// View `bytes` as a program and validate it.
    pub fn from_bytes(bytes: &[u8]) -> Result<&Self, ProgramError> {
        let p = ChannelProgram::ref_from_bytes(bytes)
            .map_err(|_| ProgramError::whole("wrong byte length for a ChannelProgram"))?;
        p.validate()?;
        Ok(p)
    }

    /// Check the rules listed on [`ChannelProgram`].
    pub fn validate(&self) -> Result<(), ProgramError> {
        if self.version.get() != FORMAT_VERSION {
            return Err(ProgramError::whole("unsupported format version"));
        }
        let raw_len = usize::from(self.len.get());
        if raw_len == 0 {
            return Err(ProgramError::whole("program has no commands"));
        }
        if raw_len > MAX_COMMANDS {
            return Err(ProgramError::whole("program has more than 16 commands"));
        }
        if self.reserved != [0; 4] {
            return Err(ProgramError::whole("program reserved bytes are not zero"));
        }
        let zero = ChannelCmd::new_zeroed();
        if self.cmds[raw_len..].iter().any(|c| *c != zero) {
            return Err(ProgramError::whole("unused command slots are not zero"));
        }

        let mut decoded = [(Op::Checksum, ExtentRef::None, ExtentRef::None); MAX_COMMANDS];
        let cmds = self.commands();
        for (i, cmd) in cmds.iter().enumerate() {
            decoded[i] = cmd.check().map_err(|r| ProgramError::at(i, r))?;
        }
        let decoded = &decoded[..cmds.len()];
        let last = cmds.len() - 1;

        for (i, cmd) in cmds.iter().enumerate() {
            let (_, _, dst) = decoded[i];
            let chain = cmd.flags().contains(Flags::CHAIN);
            if i < last && !matches!(dst, ExtentRef::Vram(_)) {
                return Err(ProgramError::at(i, "intermediate dst is not Vram"));
            }
            if chain {
                if i == last {
                    return Err(ProgramError::at(i, "last command has CHAIN set"));
                }
                if decoded[i + 1].1 != dst {
                    return Err(ProgramError::at(
                        i + 1,
                        "chained src does not equal previous dst",
                    ));
                }
            }
        }

        for (i, &(_, _, dst)) in decoded.iter().enumerate() {
            let Some(d) = dst.buf_range() else { continue };
            for (j, &(_, src, _)) in decoded.iter().enumerate() {
                let Some(s) = src.buf_range() else { continue };
                if dst.kind() != src.kind() || !d.overlaps(&s) {
                    continue;
                }
                let chain_link = j == i + 1 && cmds[i].flags().contains(Flags::CHAIN) && src == dst;
                if !chain_link {
                    return Err(ProgramError::at(
                        i,
                        "dst overlaps a src of the same program",
                    ));
                }
            }
        }
        Ok(())
    }
}

//! Whisper speech to text in a child process.
//!
//! The child keeps the recording safe: a whisper.cpp abort or a Metal hang ends
//! the child, never the session, and a chunk that hangs can be stopped by
//! killing it. The child is this same program, run as `__whisper-worker`.

pub mod protocol;
pub mod transcriber;
pub mod worker;

/// The hidden subcommand that runs the child.
pub const WORKER_COMMAND: &str = "__whisper-worker";

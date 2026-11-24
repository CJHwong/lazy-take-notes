//! A reader that goes away (`| head`) ends a printing command quietly, as click does.

use std::io::Read;
use std::process::{Command, Stdio};

#[test]
fn a_closed_pipe_ends_transcript_with_exit_1_and_no_message() {
    let base = tempfile::tempdir().unwrap();
    let session = base.path().join("2026-01-01_x");
    std::fs::create_dir_all(&session).unwrap();
    let line = "[00:00:01] a line long enough to fill the pipe quickly\n";
    std::fs::write(session.join("transcript.txt"), line.repeat(10_000)).unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_lazy-take-notes"))
        .arg("-o")
        .arg(base.path())
        .arg("transcript")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    // Read one byte, then close the pipe, as `head -c 1` does.
    let mut first = [0u8; 1];
    let mut stdout = child.stdout.take().unwrap();
    stdout.read_exact(&mut first).unwrap();
    drop(stdout);
    let output = child.wait_with_output().unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(String::from_utf8_lossy(&output.stderr), "");
}

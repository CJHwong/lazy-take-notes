//! The tap source against fake helpers written in sh.

use std::os::unix::fs::PermissionsExt;
use std::path::Path;

use super::*;

fn helper(dir: &Path, body: &str) -> CoreAudioTapSource {
    let path = dir.join("coreaudio-tap");
    std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    CoreAudioTapSource::with_binary(path, false)
}

fn drain_until_end(source: &CoreAudioTapSource) -> (Vec<Vec<f32>>, Option<AudioError>) {
    let mut blocks = Vec::new();
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        match source.read(Duration::from_millis(50)) {
            Ok(Some(block)) => blocks.push(block),
            Ok(None) if source.exhausted() => return (blocks, None),
            Ok(None) => {}
            Err(error) => return (blocks, Some(error)),
        }
    }
    panic!("the helper never ended");
}

#[test]
fn whole_blocks_arrive_and_a_clean_exit_has_no_error() {
    let dir = tempfile::tempdir().unwrap();
    // 1600 zero samples, then one sample of 0.5 (0x3F000000, little endian).
    let source = helper(
        dir.path(),
        r"head -c 6400 /dev/zero; printf '\000\000\000\077'",
    );
    source.open(16_000, 1).unwrap();
    let (blocks, error) = drain_until_end(&source);
    assert_eq!(error, None);
    assert_eq!(blocks.iter().map(Vec::len).collect::<Vec<_>>(), [1600, 1]);
    assert_eq!(blocks[1], [0.5]);
    assert!(source.system_signal_seen());
    source.close().unwrap();
}

#[test]
fn a_failed_helper_reports_its_code_and_stderr_tail() {
    let dir = tempfile::tempdir().unwrap();
    let source = helper(dir.path(), "echo 'tap: no device' >&2; exit 3");
    source.open(16_000, 1).unwrap();
    let (_, error) = drain_until_end(&source);
    assert_eq!(
        error,
        Some(AudioError(
            "coreaudio-tap exited with code 3: tap: no device".into()
        ))
    );
    assert!(source.exhausted());
    source.close().unwrap();
}

#[test]
fn a_partial_sample_at_the_end_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let source = helper(dir.path(), r"printf '\000\000\000'");
    source.open(16_000, 1).unwrap();
    let (_, error) = drain_until_end(&source);
    assert_eq!(
        error,
        Some(AudioError(
            "coreaudio-tap ended with a partial float32 sample".into()
        ))
    );
    source.close().unwrap();
}

#[test]
fn the_permission_wait_follows_the_diagnostics() {
    let dir = tempfile::tempdir().unwrap();
    let source = helper(
        dir.path(),
        "echo 'waiting for permission' >&2; read line; echo 'capturing' >&2; read line",
    );
    source.open(16_000, 1).unwrap();
    let deadline = Instant::now() + Duration::from_secs(2);
    while !source.awaiting_permission() && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(5));
    }
    assert!(source.awaiting_permission());
    source.close().unwrap();
}

#[test]
fn close_kills_a_helper_that_ignores_stdin_and_sigterm() {
    let dir = tempfile::tempdir().unwrap();
    let source = helper(dir.path(), "trap '' TERM; while :; do sleep 0.05; done");
    source.open(16_000, 1).unwrap();
    let started = Instant::now();
    source.close().unwrap();
    let took = started.elapsed();
    assert!(
        took >= Duration::from_secs(2) && took < Duration::from_secs(4),
        "{took:?}"
    );
    assert_eq!(source.pid(), None);
}

#[test]
fn a_missing_helper_names_the_build_script() {
    let source = CoreAudioTapSource::with_binary("/nowhere/coreaudio-tap".into(), false);
    let error = source.open(16_000, 1).unwrap_err();
    assert_eq!(
        error.0,
        "Native binary not found: /nowhere/coreaudio-tap\nRun: bash scripts/build_native.sh"
    );
    assert_eq!(
        source.open(48_000, 1).unwrap_err().0,
        "CoreAudioTapSource requires 16000 Hz mono output"
    );
}

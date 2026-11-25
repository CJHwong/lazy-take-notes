//! Decode any audio or video file to 16 kHz mono f32 through ffmpeg.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use crate::domain::SAMPLE_RATE;

const FFMPEG_TIMEOUT: Duration = Duration::from_secs(300);
const POLL: Duration = Duration::from_millis(50);

/// The text the transcribe screen shows.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct AudioFileError(pub String);

/// Load `path` as float32 mono PCM at 16 kHz. WAV, FLAC, MP3, M4A, OGG, MP4: anything ffmpeg decodes.
pub fn load_audio_file(path: &Path) -> Result<Vec<f32>, AudioFileError> {
    load_with(path, Path::new("ffmpeg"), FFMPEG_TIMEOUT)
}

fn load_with(path: &Path, ffmpeg: &Path, timeout: Duration) -> Result<Vec<f32>, AudioFileError> {
    let shown = path.display();
    if !path.exists() {
        return Err(AudioFileError(format!("Audio file not found: {shown}")));
    }
    let Some(program) = find_program(ffmpeg) else {
        return Err(AudioFileError(
            "ffmpeg is required but not found on PATH.\n  macOS:  brew install ffmpeg\n  Debian: apt install ffmpeg".into(),
        ));
    };
    let rate = SAMPLE_RATE.to_string();
    let mut child = Command::new(program)
        .arg("-i")
        .arg(path)
        .args([
            "-ar", &rate, "-ac", "1", "-f", "f32le", "-v", "quiet", "pipe:1",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| AudioFileError(format!("Failed to launch ffmpeg: {e}")))?;
    // Drain both pipes while waiting: a long file would otherwise fill one and stall ffmpeg.
    let stdout = drain(child.stdout.take());
    let stderr = drain(child.stderr.take());
    let started = Instant::now();
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if started.elapsed() >= timeout => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(AudioFileError(format!(
                    "ffmpeg timed out after {}s processing: {shown}",
                    timeout.as_secs()
                )));
            }
            Ok(None) => thread::sleep(POLL),
            Err(e) => return Err(AudioFileError(format!("Failed to launch ffmpeg: {e}"))),
        }
    };
    let output = stdout.join().unwrap_or_default();
    if !status.success() {
        let code = status.code().unwrap_or(-1);
        let message = String::from_utf8_lossy(&stderr.join().unwrap_or_default())
            .trim()
            .to_string();
        return Err(AudioFileError(format!(
            "ffmpeg exited with code {code} for: {shown}\n{message}"
        )));
    }
    if output.is_empty() {
        return Err(AudioFileError(format!(
            "ffmpeg produced no audio output for: {shown}"
        )));
    }
    // A trailing partial sample is dropped, as numpy would refuse it; ffmpeg never writes one.
    Ok(output
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect())
}

fn drain(pipe: Option<impl Read + Send + 'static>) -> thread::JoinHandle<Vec<u8>> {
    thread::spawn(move || {
        let mut bytes = Vec::new();
        if let Some(mut pipe) = pipe {
            let _ = pipe.read_to_end(&mut bytes);
        }
        bytes
    })
}

/// The program as given when it holds a slash, else the first match on PATH.
fn find_program(program: &Path) -> Option<PathBuf> {
    if program.components().count() > 1 {
        return program.is_file().then(|| program.to_path_buf());
    }
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|dir| dir.join(program))
        .find(|candidate| candidate.is_file())
}

#[cfg(test)]
mod tests {
    use std::os::unix::fs::PermissionsExt;

    use super::*;

    fn script(dir: &Path, body: &str) -> PathBuf {
        let path = dir.join("fake-ffmpeg");
        std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        path
    }

    #[test]
    fn errors_use_the_python_texts() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("none.wav");
        assert_eq!(
            load_with(&missing, Path::new("ffmpeg"), FFMPEG_TIMEOUT)
                .unwrap_err()
                .0,
            format!("Audio file not found: {}", missing.display())
        );
        let input = dir.path().join("in.wav");
        std::fs::write(&input, b"x").unwrap();
        let no_ffmpeg =
            load_with(&input, &dir.path().join("no-such-ffmpeg"), FFMPEG_TIMEOUT).unwrap_err();
        assert!(
            no_ffmpeg
                .0
                .starts_with("ffmpeg is required but not found on PATH.")
        );
        let failing = script(dir.path(), "echo 'bad input' >&2; exit 3");
        assert_eq!(
            load_with(&input, &failing, FFMPEG_TIMEOUT).unwrap_err().0,
            format!(
                "ffmpeg exited with code 3 for: {}\nbad input",
                input.display()
            )
        );
        let silent = script(dir.path(), "exit 0");
        assert_eq!(
            load_with(&input, &silent, FFMPEG_TIMEOUT).unwrap_err().0,
            format!("ffmpeg produced no audio output for: {}", input.display())
        );
    }

    #[test]
    fn a_hung_ffmpeg_is_killed_at_the_timeout() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("in.wav");
        std::fs::write(&input, b"x").unwrap();
        let hung = script(dir.path(), "exec sleep 30");
        let error = load_with(&input, &hung, Duration::from_millis(200)).unwrap_err();
        assert_eq!(
            error.0,
            format!("ffmpeg timed out after 0s processing: {}", input.display())
        );
    }

    #[test]
    fn decodes_little_endian_floats_larger_than_a_pipe_buffer() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("in.wav");
        std::fs::write(&input, b"x").unwrap();
        // 1 MB of samples on stdout and 1 MB of noise on stderr at once.
        let noisy = script(
            dir.path(),
            "head -c 1048576 /dev/zero >&2 & head -c 1048576 /dev/zero; wait",
        );
        let samples = load_with(&input, &noisy, FFMPEG_TIMEOUT).unwrap();
        assert_eq!(samples.len(), 262_144);
        assert!(samples.iter().all(|s| *s == 0.0));
    }
}

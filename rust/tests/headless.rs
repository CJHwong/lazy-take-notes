//! Ported from tests/l4_frameworks_and_drivers/test_headless.py.

mod common;

use std::io::Write;
use std::path::Path;
use std::sync::{Arc, Mutex};

use common::{FakeLlm, FakePersistence, Reply, built_in_template};
use lazy_take_notes::adapters::session_controller::SessionController;
use lazy_take_notes::adapters::session_status::read_status;
use lazy_take_notes::app::events::{Event, WorkerState};
use lazy_take_notes::app::headless::{
    HeadlessSession, Launcher, StatusWriter, banner_lines, drop_empty_session, heartbeat_line,
    summary_lines,
};
use lazy_take_notes::config::build_app_config;
use lazy_take_notes::domain::session_status::SessionStatus;
use lazy_take_notes::domain::transcript::TranscriptSegment;
use lazy_take_notes::ports::llm::LlmError;

#[derive(Clone, Default)]
struct Captured(Arc<Mutex<Vec<u8>>>);

impl Write for Captured {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Captured {
    fn text(&self) -> String {
        String::from_utf8(self.0.lock().unwrap().clone()).unwrap()
    }
}

struct Setup {
    session: HeadlessSession,
    out: Captured,
    dir: tempfile::TempDir,
    llm: Arc<FakeLlm>,
}

fn setup(config_yaml: &str, ai_notes: bool) -> Setup {
    let dir = tempfile::tempdir().unwrap();
    let config = build_app_config(&yaml_serde::from_str(config_yaml).unwrap()).unwrap();
    let llm = Arc::new(FakeLlm::new(Reply::Text("## Notes".into(), 10)));
    let (persistence, _saved) = FakePersistence::new();
    let controller = SessionController::new(
        config,
        built_in_template("default_en"),
        llm.clone(),
        Box::new(persistence),
    );
    let out = Captured::default();
    let session = HeadlessSession::new(
        controller,
        dir.path().to_path_buf(),
        "recording",
        ai_notes,
        None,
        StatusWriter::new(Box::new(out.clone()), false),
    );
    Setup {
        session,
        out,
        dir,
        llm,
    }
}

fn chunk(text: &str) -> Event {
    Event::TranscriptChunk(vec![TranscriptSegment::new(text, 0.0, 1.0)])
}

fn status(state: WorkerState, error: &str) -> Event {
    Event::status_error(state, error)
}

/// A worker that posts these events, then returns.
fn script(events: Vec<Event>) -> Launcher {
    Box::new(move |poster, _cancel| {
        for event in events {
            poster.post(event);
        }
    })
}

fn state_of(dir: &Path) -> SessionStatus {
    read_status(dir).expect("a status file")
}

use WorkerState::{Error, LoadingModel, ModelReady, Recording, Stopped, Warning};

#[tokio::test]
async fn a_session_runs_to_stopped_with_a_final_digest() {
    let mut s = setup("{}", true);
    s.session
        .run(script(vec![
            status(LoadingModel, ""),
            status(ModelReady, ""),
            status(Recording, ""),
            chunk("hello"),
            status(Stopped, ""),
        ]))
        .await;
    let state = state_of(s.dir.path());
    assert_eq!(
        (
            state.state.as_str(),
            state.segment_count,
            state.digest_count
        ),
        ("stopped", 1, 1)
    );
    assert_eq!(
        s.out.text(),
        "… loading model\n✓ model ready, recording\n✓ notes.md updated (digest #1)\n"
    );
}

#[tokio::test]
async fn an_error_before_the_start_ends_the_session_without_notes() {
    let mut s = setup("{}", true);
    s.session
        .run(script(vec![
            status(LoadingModel, ""),
            status(Error, "no mic"),
        ]))
        .await;
    let state = state_of(s.dir.path());
    assert_eq!(
        (state.state.as_str(), state.error.as_str()),
        ("error", "no mic")
    );
    assert!(s.out.text().ends_with("✗ error: no mic\n"));
    assert!(s.llm.chat_calls.lock().unwrap().is_empty());
}

#[tokio::test]
async fn an_error_after_the_start_keeps_recording() {
    let mut s = setup("{}", true);
    s.session
        .run(script(vec![
            status(Recording, ""),
            status(Error, "chunk failed"),
            chunk("after"),
            status(Stopped, ""),
        ]))
        .await;
    assert!(
        s.out
            .text()
            .contains("✗ transcription error (continuing): chunk failed\n")
    );
    assert_eq!(state_of(s.dir.path()).state, "stopped");
}

#[tokio::test]
async fn a_full_buffer_starts_a_digest_during_the_session() {
    let mut s = setup("digest: {min_lines: 1, min_interval: 0}", true);
    s.session
        .run(script(vec![
            status(Recording, ""),
            chunk("one"),
            status(Stopped, ""),
        ]))
        .await;
    // One periodic digest, then the final one.
    assert_eq!(s.llm.chat_calls.lock().unwrap().len(), 2);
    assert_eq!(state_of(s.dir.path()).digest_count, 2);
}

#[tokio::test]
async fn a_failed_ai_check_records_the_transcript_only() {
    let mut s = setup("digest: {min_lines: 1, min_interval: 0}", false);
    s.session
        .run(script(vec![
            status(Recording, ""),
            chunk("one"),
            status(Stopped, ""),
        ]))
        .await;
    assert!(s.llm.chat_calls.lock().unwrap().is_empty());
    let state = state_of(s.dir.path());
    assert_eq!((state.state.as_str(), state.segment_count), ("stopped", 1));
}

#[tokio::test]
async fn a_failed_digest_is_reported_and_saved() {
    let mut s = setup("{}", true);
    s.llm
        .set(Reply::Fail(LlmError::new("ConnectError", "refused")));
    s.session
        .run(script(vec![
            status(Recording, ""),
            chunk("one"),
            status(Stopped, ""),
        ]))
        .await;
    assert!(
        s.out
            .text()
            .contains("✗ digest failed: LLM error: ConnectError: refused\n"),
        "{}",
        s.out.text()
    );
    assert_eq!(
        state_of(s.dir.path()).error,
        "LLM error: ConnectError: refused"
    );
}

#[tokio::test]
async fn download_progress_is_shown_every_twenty_percent() {
    let mut s = setup("{}", true);
    let progress = |percent| Event::ModelDownloadProgress {
        percent,
        model_name: "breeze".into(),
    };
    s.session
        .run(script(vec![
            progress(0),
            progress(5),
            progress(20),
            progress(45),
            status(Stopped, ""),
        ]))
        .await;
    assert_eq!(
        s.out.text(),
        "… downloading breeze 0%\n… downloading breeze 20%\n… downloading breeze 45%\n"
    );
}

#[tokio::test]
async fn a_warning_shows_once_until_audio_flows_again() {
    let mut s = setup("{}", true);
    s.session
        .run(script(vec![
            status(Recording, ""),
            status(Warning, "MIC LOST"),
            status(Warning, "MIC LOST"),
            chunk("back"),
            status(Warning, "MIC LOST"),
            status(Stopped, ""),
        ]))
        .await;
    assert_eq!(s.out.text().matches("⚠ MIC LOST\n").count(), 2);
}

#[tokio::test]
async fn a_stop_is_announced_once_and_a_shutdown_error_ends_cleanly() {
    let mut s = setup("{}", true);
    let stopper = s.session.stopper();
    let launcher: Launcher = Box::new(move |poster, cancel| {
        poster.post(status(Recording, ""));
        poster.post(chunk("one"));
        stopper.stop();
        stopper.stop();
        assert!(cancel.is_cancelled());
        poster.post(status(Error, "stream closed"));
    });
    s.session.run(launcher).await;
    let text = s.out.text();
    assert_eq!(
        text.matches("⏹ stopping… finishing the current chunk, then the final digest\n")
            .count(),
        1
    );
    assert_eq!(state_of(s.dir.path()).state, "stopped");
}

#[tokio::test(start_paused = true)]
async fn a_worker_that_never_stops_is_left_after_the_grace_period() {
    let mut s = setup("{}", false);
    let stopper = s.session.stopper();
    let release = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let held = release.clone();
    let launcher: Launcher = Box::new(move |poster, _cancel| {
        poster.post(status(Recording, ""));
        stopper.stop();
        // Stuck: holds the channel open and never posts stopped.
        while !held.load(std::sync::atomic::Ordering::Relaxed) {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        drop(poster);
    });
    s.session.run(launcher).await;
    release.store(true, std::sync::atomic::Ordering::Relaxed);
    assert!(
        s.out
            .text()
            .ends_with("⏹ worker did not stop in time; ending session\n"),
        "{}",
        s.out.text()
    );
    assert_eq!(state_of(s.dir.path()).state, "stopped");
}

#[test]
fn a_tty_rewrites_the_status_line_and_commits_it_before_a_milestone() {
    let out = Captured::default();
    let mut writer = StatusWriter::new(Box::new(out.clone()), true);
    writer.status("⏺ a");
    writer.status("⏺ b");
    writer.line("✓ done");
    writer.status("⏺ c");
    writer.finish();
    assert_eq!(out.text(), "\r⏺ a\x1b[K\r⏺ b\x1b[K\n✓ done\n\r⏺ c\x1b[K\n");
    let plain = Captured::default();
    let mut writer = StatusWriter::new(Box::new(plain.clone()), false);
    writer.status("⏺ a");
    writer.line("✓ done");
    writer.finish();
    assert_eq!(plain.text(), "✓ done\n");
}

#[test]
fn the_fixed_lines_read_like_python() {
    assert_eq!(
        heartbeat_line("recording", 125.9, 3, 1),
        "⏺ recording | 02:05 | 3 seg | 1 digest"
    );
    assert_eq!(
        heartbeat_line("recording", 0.0, 0, 2),
        "⏺ recording | 00:00 | 0 seg | 2 digests"
    );
    assert_eq!(
        banner_lines("recording", Path::new("/o/s"), 42, true),
        [
            "▶ recording → /o/s",
            "  stop:  Ctrl-C  (or kill -INT 42)",
            "  mute:  kill -USR1 42",
            "  check: lazy-take-notes status",
        ]
    );
    let status = SessionStatus {
        state: "error".into(),
        pid: 1,
        started_at: "2026-11-19T10:00:00".into(),
        updated_at: "2026-11-19T10:01:05".into(),
        segment_count: 2,
        digest_count: 0,
        error: "boom".into(),
    };
    assert_eq!(
        summary_lines(Path::new("/o/s"), &status),
        [
            "■ error | 01:05 | 2 seg | 0 digests",
            "  transcript: /o/s/transcript.txt",
            "  error:      boom"
        ]
    );
}

// --- A failed run with nothing to recover (TestAFailedRun) ---

fn session_dir(files: &[&str]) -> (tempfile::TempDir, std::path::PathBuf) {
    let root = tempfile::tempdir().unwrap();
    let dir = root.path().join("session");
    std::fs::create_dir(&dir).unwrap();
    for name in files {
        std::fs::write(dir.join(name), "x").unwrap();
    }
    (root, dir)
}

#[test]
fn a_folder_with_only_the_status_is_dropped() {
    let (_root, dir) = session_dir(&[".status.json"]);
    assert!(drop_empty_session(&dir));
    assert!(!dir.exists());
}

#[test]
fn a_folder_with_only_the_status_and_its_debug_log_is_dropped() {
    // stderr already has the error, and the same file run again writes the log again.
    let (_root, dir) = session_dir(&[".status.json", "debug.log"]);
    assert!(drop_empty_session(&dir));
    assert!(!dir.exists());
}

#[test]
fn a_folder_whose_debug_log_sits_next_to_a_transcript_is_kept() {
    let (_root, dir) = session_dir(&[".status.json", "debug.log", "transcript.txt"]);
    assert!(!drop_empty_session(&dir));
    assert!(dir.join("debug.log").exists());
    assert!(dir.join("transcript.txt").exists());
}

#[test]
fn a_folder_without_its_status_file_is_kept() {
    let (_root, dir) = session_dir(&["debug.log"]);
    assert!(!drop_empty_session(&dir));
    assert!(dir.join("debug.log").exists());
}

#[cfg(unix)]
#[test]
fn a_folder_that_cannot_be_removed_is_kept() {
    use std::os::unix::fs::PermissionsExt;
    let (_root, dir) = session_dir(&[".status.json"]);
    std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o500)).unwrap();
    let dropped = drop_empty_session(&dir);
    std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700)).unwrap();
    assert!(!dropped);
    assert!(dir.join(".status.json").exists());
}

/// Answers every pass with `turns`, or fails with `error`.
struct FixedDiarizer(Result<Vec<lazy_take_notes::domain::speaker_turn::SpeakerTurn>, String>);

impl lazy_take_notes::ports::diarizer::Diarizer for FixedDiarizer {
    fn diarize(
        &self,
        _path: &Path,
        _timeout: std::time::Duration,
    ) -> Result<Vec<lazy_take_notes::domain::speaker_turn::SpeakerTurn>, String> {
        self.0.clone()
    }
}

/// A recording WAV with one second of silence.
fn write_recording(dir: &Path) {
    let data = 32_000u32;
    let mut bytes = b"RIFF".to_vec();
    bytes.extend((36 + data).to_le_bytes());
    bytes.extend(b"WAVEfmt ");
    bytes.extend([
        16, 0, 0, 0, 1, 0, 1, 0, 0x80, 0x3e, 0, 0, 0, 0x7d, 0, 0, 2, 0, 16, 0,
    ]);
    bytes.extend(b"data");
    bytes.extend(data.to_le_bytes());
    bytes.resize(44 + data as usize, 0);
    std::fs::write(dir.join("recording.wav"), bytes).unwrap();
}

fn stop_after_hello() -> Launcher {
    script(vec![
        status(Recording, ""),
        chunk("hello"),
        status(Stopped, ""),
    ])
}

#[tokio::test]
async fn the_speakers_are_labeled_before_the_final_digest() {
    let s = setup("{}", true);
    write_recording(s.dir.path());
    let turn = lazy_take_notes::domain::speaker_turn::SpeakerTurn {
        start: 0.0,
        end: 1.0,
        speaker: 1,
    };
    let mut session = s.session.with_diarizer(
        Arc::new(FixedDiarizer(Ok(vec![turn]))),
        std::time::Duration::from_secs(5),
    );
    session.run(stop_after_hello()).await;
    let labeled = std::fs::read_to_string(s.dir.path().join("transcript_speakers.txt")).unwrap();
    assert_eq!(labeled, "[00:00:00] S1: hello\n");
    let json = std::fs::read_to_string(s.dir.path().join("speakers.json")).unwrap();
    assert!(json.contains("\"turns\""), "{json}");
    let prompts = s.llm.chat_calls.lock().unwrap();
    let last = &prompts.last().expect("a final digest").1;
    assert!(
        last.iter()
            .any(|message| message.content.contains("S1: hello")),
        "the final notes carry the labels"
    );
    assert!(
        s.out
            .text()
            .contains("… labeling speakers\n✓ transcript_speakers.txt written (1 speakers)\n"),
        "{}",
        s.out.text()
    );
}

#[tokio::test]
async fn a_failed_speaker_pass_still_writes_the_plain_notes() {
    let s = setup("{}", true);
    write_recording(s.dir.path());
    let mut session = s.session.with_diarizer(
        Arc::new(FixedDiarizer(Err("nemo-speech failed".into()))),
        std::time::Duration::from_secs(5),
    );
    session.run(stop_after_hello()).await;
    assert!(!s.dir.path().join("transcript_speakers.txt").exists());
    assert_eq!(state_of(s.dir.path()).digest_count, 1);
    assert!(
        s.out
            .text()
            .contains("⚠ speaker labels failed: nemo-speech failed"),
        "{}",
        s.out.text()
    );
}

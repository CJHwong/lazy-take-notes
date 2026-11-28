//! Ported from TestStreamingWavWriter and TestProcessedRecorderFaults in tests/l4_frameworks_and_drivers/workers/test_audio_worker.py.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use super::*;

fn frames(path: &Path) -> Vec<i16> {
    let raw = std::fs::read(path).unwrap();
    let size = u32::from_le_bytes(raw[40..44].try_into().unwrap()) as usize;
    assert_eq!(
        u32::from_le_bytes(raw[4..8].try_into().unwrap()) as usize,
        36 + size,
        "RIFF size"
    );
    assert_eq!(raw.len(), 44 + size, "data size");
    raw[44..]
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| i16::from_le_bytes(*b))
        .collect()
}

fn pcm(value: f32, count: usize) -> Vec<i16> {
    vec![(value * 32767.0) as i16; count]
}

fn notices() -> (Arc<Mutex<Vec<String>>>, Notice) {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let sink = seen.clone();
    (
        seen,
        Box::new(move |message| sink.lock().unwrap().push(message)),
    )
}

fn full_disk() -> OsError {
    OsError::new(28, "No space left on device", None)
}

fn wav_names(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

#[test]
fn the_header_stays_valid_while_writing() {
    let root = tempfile::tempdir().unwrap();
    let mut writer = StreamingWav::create(root.path().join("recording.wav"), 16_000).unwrap();
    writer.write(&[1, 0].repeat(160)).unwrap();
    assert_eq!(frames(&writer.path).len(), 160);
    writer.write(&[2, 0].repeat(160)).unwrap();
    assert_eq!(frames(&writer.path).len(), 320);
}

#[cfg(unix)]
#[test]
fn os_errors_read_like_python() {
    assert_eq!(
        full_disk().to_string(),
        "[Errno 28] No space left on device"
    );
    let moved = OsError::new(
        2,
        "the recording file was moved or removed",
        Some(Path::new("/a/it's.wav")),
    );
    assert_eq!(
        moved.to_string(),
        "[Errno 2] the recording file was moved or removed: \"/a/it's.wav\""
    );
    let error = File::create("/nonexistent/x.wav").unwrap_err();
    let error = OsError::from_io(&error, Some(Path::new("/nonexistent/x.wav")), None);
    assert_eq!(
        error.to_string(),
        "[Errno 2] No such file or directory: '/nonexistent/x.wav'"
    );
}

#[test]
fn a_full_disk_keeps_the_audio_and_writes_it_to_the_next_file() {
    let root = tempfile::tempdir().unwrap();
    let (seen, notice) = notices();
    let mut recording = RecoveringWav::new(
        Arc::new(SessionFolder::new(root.path())),
        16_000,
        Some(notice),
        None,
    )
    .with_limits(Duration::ZERO, usize::MAX);
    let mut calls = 0;
    // The third chunk fails, then its first retry fails too.
    recording.write_fault = Some(Box::new(move || {
        calls += 1;
        [3, 4].contains(&calls).then(full_disk)
    }));
    for value in 0..6 {
        recording.write(&vec![value as f32 / 10.0; 160]);
    }
    recording.close();
    assert_eq!(wav_names(root.path()), ["recording.wav", "recording_2.wav"]);
    let mut all = frames(&root.path().join("recording.wav"));
    all.extend(frames(&root.path().join("recording_2.wav")));
    let expected: Vec<i16> = (0..6)
        .flat_map(|value| pcm(value as f32 / 10.0, 160))
        .collect();
    assert_eq!(all, expected, "every chunk once, in order");
    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 2);
    assert_eq!(
        seen[0],
        "Audio recording paused: [Errno 28] No space left on device. Audio is kept in memory and saved when the disk works again."
    );
    assert_eq!(seen[1], "Audio recording continues in recording_2.wav.");
}

#[cfg(unix)]
#[test]
fn a_removed_session_folder_is_created_again() {
    for move_to_trash in [false, true] {
        let root = tempfile::tempdir().unwrap();
        let session = root.path().join("session");
        std::fs::create_dir(&session).unwrap();
        let (seen, notice) = notices();
        let mut recording = RecoveringWav::new(
            Arc::new(SessionFolder::new(&session)),
            16_000,
            Some(notice),
            None,
        )
        .with_limits(Duration::ZERO, usize::MAX);
        recording.write(&[0.1; 160]);
        if move_to_trash {
            std::fs::rename(&session, root.path().join("Trash")).unwrap();
        } else {
            std::fs::remove_dir_all(&session).unwrap();
        }
        recording.write(&[0.2; 160]); // lands in the removed file, so it is kept
        recording.write(&[0.3; 160]);
        recording.close();
        let written = frames(&session.join("recording.wav"));
        assert_eq!(written.len(), 320);
        assert_eq!(written[0], (0.2f32 * 32767.0) as i16);
        let seen = seen.lock().unwrap();
        assert!(seen[0].contains("removed"), "{seen:?}");
        assert!(seen[1].contains("recording.wav"), "{seen:?}");
    }
}

#[cfg(unix)]
#[test]
fn a_folder_renamed_by_the_app_keeps_one_recording() {
    let root = tempfile::tempdir().unwrap();
    let folder = Arc::new(SessionFolder::new(root.path().join("2026-02-21_143052")));
    std::fs::create_dir(folder.path()).unwrap();
    let (seen, notice) = notices();
    let mut recording = RecoveringWav::new(folder.clone(), 16_000, Some(notice), None);
    recording.write(&[0.1; 160]);
    folder
        .rename(&root.path().join("2026-02-21_143052_standup"))
        .unwrap();
    recording.write(&[0.2; 160]);
    recording.close();
    assert!(!root.path().join("2026-02-21_143052").exists());
    assert_eq!(wav_names(&folder.path()), ["recording.wav"]);
    assert_eq!(frames(&folder.path().join("recording.wav")).len(), 320);
    assert!(seen.lock().unwrap().is_empty());
}

#[cfg(unix)]
#[test]
fn save_audio_off_deletes_the_file_in_the_renamed_folder() {
    let root = tempfile::tempdir().unwrap();
    let folder = Arc::new(SessionFolder::new(root.path().join("session")));
    std::fs::create_dir(folder.path()).unwrap();
    let mut recording = RecoveringWav::new(folder.clone(), 16_000, None, Some(Box::new(|| false)));
    recording.write(&[0.1; 160]);
    folder.rename(&root.path().join("session_label")).unwrap();
    recording.close();
    assert!(wav_names(&folder.path()).is_empty());
}

#[test]
fn a_disk_that_never_recovers_reports_once_and_bounds_the_backlog() {
    let root = tempfile::tempdir().unwrap();
    let (seen, notice) = notices();
    let mut recording = RecoveringWav::new(
        Arc::new(SessionFolder::new(root.path())),
        16_000,
        Some(notice),
        None,
    )
    .with_limits(Duration::ZERO, 2 * 16_000 * 2);
    recording.write_fault = Some(Box::new(|| Some(full_disk())));
    for _ in 0..5 {
        recording.write(&[0.0; 16_000]);
    }
    assert!(recording.backlog_bytes <= 2 * 16_000 * 2);
    recording.close();
    assert_eq!(seen.lock().unwrap().len(), 1);
    assert!(
        wav_names(root.path()).is_empty(),
        "failed retries hold no audio, so none is left behind"
    );
}

#[test]
fn the_recorder_thread_drains_and_closes() {
    let root = tempfile::tempdir().unwrap();
    let (sender, writer) = start_processed_recorder(
        Arc::new(SessionFolder::new(root.path())),
        16_000,
        None,
        None,
    )
    .unwrap();
    for value in [0.1f32, 2.0, -2.0] {
        sender.send(Some(vec![value; 160])).unwrap();
    }
    sender.send(None).unwrap();
    writer.join().unwrap();
    let written = frames(&root.path().join("recording.wav"));
    let expected: Vec<i16> = [pcm(0.1, 160), pcm(1.0, 160), pcm(-1.0, 160)].concat();
    assert_eq!(written, expected, "samples are clipped to [-1, 1]");
}

#[test]
fn a_snapshot_of_a_file_caught_between_its_data_and_header_writes_is_valid() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("recording.wav");
    let mut wav = StreamingWav::create(source.clone(), 16_000).unwrap();
    let first: Vec<u8> = pcm(0.5, 100).iter().flat_map(|s| s.to_le_bytes()).collect();
    wav.write(&first).unwrap();
    // The next chunk is on disk, its header update is not, plus half a frame.
    let mut file = std::fs::OpenOptions::new()
        .append(true)
        .open(&source)
        .unwrap();
    file.write_all(&[7u8; 61]).unwrap();
    let copy = dir.path().join("copy.wav");
    snapshot(&source, &copy).unwrap();
    let mut expected = pcm(0.5, 100);
    expected.extend(std::iter::repeat_n(i16::from_le_bytes([7, 7]), 30));
    assert_eq!(frames(&copy), expected);
}

#[test]
fn a_snapshot_refuses_a_file_that_is_not_a_recording_wav() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("notes.md");
    std::fs::write(&source, vec![b'#'; 100]).unwrap();
    let error = snapshot(&source, &dir.path().join("copy.wav")).unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::InvalidData);
}

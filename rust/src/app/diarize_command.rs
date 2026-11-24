//! `lazy-take-notes diarize SESSION_DIR`: label a saved transcript with anonymous speakers.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;
use std::sync::{Arc, LazyLock};
use std::time::Duration;

use regex::Regex;

use crate::adapters::assets::store::CachedModelStore;
use crate::adapters::nemo_diarizer::NemoSpeechDiarizer;
use crate::adapters::paths::Paths;
use crate::domain::session_files::SPEAKER_TRANSCRIPT;
use crate::domain::speaker_turn::SpeakerTurn;
use crate::ports::diarizer::Diarizer;
use crate::usecases::speaker_attribution::speaker_at;

static TRANSCRIPT_LINE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\[(\d{2}):(\d{2}):(\d{2})\]\s*(.*)$").expect("valid regex"));

/// Run the command. Returns the report for stdout, or the error for `Error: ...`.
pub fn run(session_dir: &Path, device: &str, timeout: Duration) -> Result<String, String> {
    let recording = session_dir.join("recording.wav");
    let transcript = session_dir.join("transcript.txt");
    if !recording.is_file() {
        return Err(format!("No recording.wav in {}", session_dir.display()));
    }
    if !transcript.is_file() {
        return Err(format!("No transcript.txt in {}", session_dir.display()));
    }
    let store = Paths::for_user().map(|paths| Arc::new(CachedModelStore::new(&paths)));
    let diarizer = NemoSpeechDiarizer::new("", "", device, store);
    let turns = diarizer.diarize(&recording, timeout)?;
    write_outputs(session_dir, &transcript, &turns)
}

fn write_outputs(
    session_dir: &Path,
    transcript: &Path,
    turns: &[SpeakerTurn],
) -> Result<String, String> {
    let json = serde_json::to_string_pretty(turns).map_err(|e| e.to_string())?;
    std::fs::write(session_dir.join("speakers.json"), json).map_err(|e| e.to_string())?;
    let text = std::fs::read_to_string(transcript).map_err(|e| e.to_string())?;
    let labeled: String = text
        .lines()
        .map(|line| label_line(line, turns) + "\n")
        .collect();
    let labeled_path = session_dir.join(SPEAKER_TRANSCRIPT);
    std::fs::write(&labeled_path, labeled).map_err(|e| e.to_string())?;
    Ok(report(turns, &labeled_path))
}

fn label_line(line: &str, turns: &[SpeakerTurn]) -> String {
    let Some(found) = TRANSCRIPT_LINE.captures(line) else {
        return line.to_string();
    };
    let number = |index: usize| found[index].parse::<u32>().unwrap_or(0);
    let at = number(1) * 3600 + number(2) * 60 + number(3);
    match speaker_at(turns, f64::from(at)) {
        Some(speaker) => format!(
            "[{}:{}:{}] S{speaker}: {}",
            &found[1], &found[2], &found[3], &found[4]
        ),
        None => line.to_string(),
    }
}

fn report(turns: &[SpeakerTurn], labeled_path: &Path) -> String {
    let mut speech: BTreeMap<u32, f64> = BTreeMap::new();
    for turn in turns {
        *speech.entry(turn.speaker).or_default() += turn.end - turn.start;
    }
    let mut out = format!("{} turns, {} speakers\n", turns.len(), speech.len());
    for (speaker, seconds) in &speech {
        let _ = writeln!(out, "  S{speaker}: {seconds:.1}s");
    }
    let _ = writeln!(out, "  labeled: {}", labeled_path.display());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turn(start: f64, end: f64, speaker: u32) -> SpeakerTurn {
        SpeakerTurn {
            start,
            end,
            speaker,
        }
    }

    #[test]
    fn labels_each_timed_line_and_keeps_the_rest() {
        let dir = tempfile::tempdir().unwrap();
        let transcript = dir.path().join("transcript.txt");
        std::fs::write(&transcript, "[00:00:01] hi\nnot a line\n[00:00:09] later\n").unwrap();
        let turns = [turn(0.0, 5.0, 1), turn(5.0, 6.5, 0)];
        let out = write_outputs(dir.path(), &transcript, &turns).unwrap();
        let labeled = std::fs::read_to_string(dir.path().join(SPEAKER_TRANSCRIPT)).unwrap();
        assert_eq!(labeled, "[00:00:01] S1: hi\nnot a line\n[00:00:09] later\n");
        assert!(out.starts_with("2 turns, 2 speakers\n  S0: 1.5s\n  S1: 5.0s\n  labeled: "));
        let json = std::fs::read_to_string(dir.path().join("speakers.json")).unwrap();
        assert!(json.starts_with(
            "[\n  {\n    \"start\": 0.0,\n    \"end\": 5.0,\n    \"speaker\": 1\n  },"
        ));
    }
}

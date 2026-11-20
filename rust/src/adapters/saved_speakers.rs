//! A saved session's speakers: `speakers.json` (names and turns) and the labeled
//! transcript. The view screen reads them and renames a speaker after the session.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::domain::session_files::SPEAKER_TRANSCRIPT;
use crate::domain::speaker_turn::SpeakerTurn;

const SPEAKERS_JSON: &str = "speakers.json";

#[derive(Debug, Clone, Default, PartialEq)]
pub struct SavedSpeakers {
    pub names: BTreeMap<u32, String>,
    pub turns: Vec<SpeakerTurn>,
}

impl SavedSpeakers {
    /// The name shown for `speaker`: its name, else `S<n>`.
    pub fn label(&self, speaker: u32) -> String {
        self.names
            .get(&speaker)
            .filter(|name| !name.is_empty())
            .cloned()
            .unwrap_or_else(|| format!("S{speaker}"))
    }

    /// Each speaker with its name ("" when unnamed) and seconds of speech.
    pub fn chips(&self) -> Vec<(u32, String, f64)> {
        let mut speech: BTreeMap<u32, f64> = BTreeMap::new();
        for turn in &self.turns {
            *speech.entry(turn.speaker).or_default() += turn.end - turn.start;
        }
        speech
            .into_iter()
            .map(|(speaker, seconds)| {
                let name = self.names.get(&speaker).cloned().unwrap_or_default();
                (speaker, name, seconds)
            })
            .collect()
    }
}

fn read_json(path: &Path) -> Result<Value, String> {
    let text =
        std::fs::read_to_string(path).map_err(|e| format!("Cannot read {SPEAKERS_JSON}: {e}"))?;
    serde_json::from_str(&text).map_err(|e| format!("Cannot parse {SPEAKERS_JSON}: {e}"))
}

/// The session's speakers. None: the session has no `speakers.json`.
pub fn load(session_dir: &Path) -> Result<Option<SavedSpeakers>, String> {
    let path = session_dir.join(SPEAKERS_JSON);
    if !path.is_file() {
        return Ok(None);
    }
    let json = read_json(&path)?;
    let names = json
        .get("names")
        .and_then(Value::as_object)
        .map(|names| {
            names
                .iter()
                .filter_map(|(key, name)| Some((key.parse().ok()?, name.as_str()?.to_string())))
                .collect()
        })
        .unwrap_or_default();
    let turns = match json.get("turns") {
        Some(turns) => serde_json::from_value(turns.clone())
            .map_err(|e| format!("Cannot parse {SPEAKERS_JSON}: {e}"))?,
        None => Vec::new(),
    };
    Ok(Some(SavedSpeakers { names, turns }))
}

/// Name `speaker` in `speakers.json` (its other keys stay) and in each line of the
/// labeled transcript that starts with its old label.
pub fn rename(session_dir: &Path, speaker: u32, name: &str) -> Result<SavedSpeakers, String> {
    let saved = load(session_dir)?.ok_or_else(|| format!("{SPEAKERS_JSON} is missing"))?;
    let old_label = saved.label(speaker);
    let path = session_dir.join(SPEAKERS_JSON);
    let mut json = read_json(&path)?;
    let Some(object) = json.as_object_mut() else {
        return Err(format!("{SPEAKERS_JSON} is not an object"));
    };
    let names = object
        .entry("names")
        .or_insert_with(|| Value::Object(serde_json::Map::new()));
    let Some(names) = names.as_object_mut() else {
        return Err(format!("{SPEAKERS_JSON}: names is not an object"));
    };
    names.insert(speaker.to_string(), Value::String(name.to_string()));
    let text = serde_json::to_string_pretty(&json).map_err(|e| e.to_string())?;
    std::fs::write(&path, text).map_err(|e| format!("Cannot write {SPEAKERS_JSON}: {e}"))?;
    relabel_transcript(&session_dir.join(SPEAKER_TRANSCRIPT), &old_label, name)?;
    load(session_dir)?.ok_or_else(|| format!("{SPEAKERS_JSON} is missing"))
}

/// `[00:01:02] Old: text` becomes `[00:01:02] New: text`. A missing file is fine.
fn relabel_transcript(path: &PathBuf, old: &str, new: &str) -> Result<(), String> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(format!("Cannot read {SPEAKER_TRANSCRIPT}: {e}")),
    };
    let from = format!("] {old}: ");
    let to = format!("] {new}: ");
    let relabeled: String = text
        .split_inclusive('\n')
        .map(|line| match line.starts_with('[') {
            true => match line.find("] ") {
                Some(at) if line[at..].starts_with(&from) => {
                    format!("{}{to}{}", &line[..at], &line[at + from.len()..])
                }
                _ => line.to_string(),
            },
            false => line.to_string(),
        })
        .collect();
    std::fs::write(path, relabeled).map_err(|e| format!("Cannot write {SPEAKER_TRANSCRIPT}: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join(SPEAKERS_JSON),
            r#"{"names": {}, "turns": [{"start": 0.0, "end": 4.0, "speaker": 1}, {"start": 5.0, "end": 6.0, "speaker": 2}]}"#,
        )
        .unwrap();
        std::fs::write(
            dir.path().join(SPEAKER_TRANSCRIPT),
            "[00:00:00] S1: hello S2: not a label\n[00:00:05] S2: hi\nloose line\n",
        )
        .unwrap();
        dir
    }

    #[test]
    fn a_rename_names_the_speaker_in_both_files_and_twice_works() {
        let dir = session();
        let saved = rename(dir.path(), 2, "Andy").unwrap();
        assert_eq!(saved.label(2), "Andy");
        assert_eq!(saved.turns.len(), 2);
        let text = std::fs::read_to_string(dir.path().join(SPEAKER_TRANSCRIPT)).unwrap();
        assert_eq!(
            text,
            "[00:00:00] S1: hello S2: not a label\n[00:00:05] Andy: hi\nloose line\n"
        );
        rename(dir.path(), 2, "安迪").unwrap();
        let text = std::fs::read_to_string(dir.path().join(SPEAKER_TRANSCRIPT)).unwrap();
        assert!(text.contains("[00:00:05] 安迪: hi\n"), "{text}");
        let json = std::fs::read_to_string(dir.path().join(SPEAKERS_JSON)).unwrap();
        assert!(json.contains("\"2\": \"安迪\""), "{json}");
    }

    #[test]
    fn chips_add_up_each_speakers_talk_time() {
        let saved = load(session().path()).unwrap().unwrap();
        assert_eq!(
            saved.chips(),
            [(1, String::new(), 4.0), (2, String::new(), 1.0)]
        );
    }

    #[test]
    fn no_speakers_file_is_none_and_a_broken_one_says_so() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(load(dir.path()).unwrap(), None);
        std::fs::write(dir.path().join(SPEAKERS_JSON), "{").unwrap();
        assert!(
            load(dir.path())
                .unwrap_err()
                .starts_with("Cannot parse speakers.json")
        );
        assert!(rename(dir.path(), 1, "x").is_err());
    }
}

//! Writes a session's transcript, notes, history and context into its folder.

use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use tracing::debug;

use crate::domain::session_files::{CONTEXT, NOTES, TRANSCRIPT};
use crate::domain::transcript::{TranscriptSegment, format_wall_time};
use crate::ports::persistence::Persistence;

pub struct FilePersistence {
    output_dir: PathBuf,
}

impl FilePersistence {
    pub fn new(output_dir: PathBuf) -> io::Result<Self> {
        fs::create_dir_all(&output_dir)?;
        Ok(Self { output_dir })
    }

    pub fn output_dir(&self) -> &Path {
        &self.output_dir
    }

    /// Point later writes at `new_dir`. The caller already moved the folder.
    pub fn relocate(&mut self, new_dir: PathBuf) {
        self.output_dir = new_dir;
    }
}

fn notes_body(markdown: &str, digest_number: u32) -> String {
    format!("# Digest #{digest_number}\n\n{markdown}\n")
}

impl Persistence for FilePersistence {
    fn save_transcript_lines(
        &mut self,
        segments: &[TranscriptSegment],
        append: bool,
    ) -> io::Result<PathBuf> {
        let path = self.output_dir.join(TRANSCRIPT.name);
        let mut file = OpenOptions::new()
            .create(true)
            .append(append)
            .write(true)
            .truncate(!append)
            .open(&path)?;
        let text: String = segments
            .iter()
            .map(|s| format!("[{}] {}\n", format_wall_time(s.wall_start), s.text))
            .collect();
        file.write_all(text.as_bytes())?;
        let last = segments
            .last()
            .map_or("?".into(), |s| format_wall_time(s.wall_start));
        debug!(target: "ltn.persist", "Wrote {} segments to {} (last_ts={last}, append={append})", segments.len(), TRANSCRIPT.name);
        Ok(path)
    }

    fn save_digest_md(&mut self, markdown: &str, digest_number: u32) -> io::Result<PathBuf> {
        let path = self.output_dir.join(NOTES.name);
        fs::write(&path, notes_body(markdown, digest_number))?;
        Ok(path)
    }

    fn save_history(
        &mut self,
        markdown: &str,
        digest_number: u32,
        is_final: bool,
    ) -> io::Result<PathBuf> {
        let history = self.output_dir.join("history");
        fs::create_dir_all(&history)?;
        let suffix = if is_final { "_final" } else { "" };
        let path = history.join(format!("notes_{digest_number:03}{suffix}.md"));
        fs::write(&path, notes_body(markdown, digest_number))?;
        Ok(path)
    }

    fn save_session_context(&mut self, context: &str) -> io::Result<PathBuf> {
        let path = self.output_dir.join(CONTEXT.name);
        fs::write(&path, context)?;
        Ok(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_the_files_python_writes() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = FilePersistence::new(dir.path().join("s")).unwrap();
        let line = |text: &str, start: f64| TranscriptSegment::new(text, start, start + 1.0);
        store
            .save_transcript_lines(&[line("hello", 3661.0)], true)
            .unwrap();
        store
            .save_transcript_lines(&[line("第二", 5.0)], true)
            .unwrap();
        store.save_digest_md("## A", 2).unwrap();
        store.save_history("## A", 2, false).unwrap();
        store.save_history("## B", 12, true).unwrap();
        store.save_session_context("ctx").unwrap();
        let read = |name: &str| fs::read_to_string(dir.path().join("s").join(name)).unwrap();
        assert_eq!(
            read("transcript.txt"),
            "[01:01:01] hello\n[00:00:05] 第二\n"
        );
        assert_eq!(read("notes.md"), "# Digest #2\n\n## A\n");
        assert_eq!(read("history/notes_002.md"), "# Digest #2\n\n## A\n");
        assert_eq!(read("history/notes_012_final.md"), "# Digest #12\n\n## B\n");
        assert_eq!(read("context.txt"), "ctx");
        store
            .save_transcript_lines(&[line("new", 0.0)], false)
            .unwrap();
        assert_eq!(read("transcript.txt"), "[00:00:00] new\n");
    }
}

use std::io;
use std::path::PathBuf;

use crate::domain::transcript::TranscriptSegment;

/// Writes the session's transcript, notes, history, and context.
pub trait Persistence: Send {
    fn save_transcript_lines(
        &mut self,
        segments: &[TranscriptSegment],
        append: bool,
    ) -> io::Result<PathBuf>;
    fn save_digest_md(&mut self, markdown: &str, digest_number: u32) -> io::Result<PathBuf>;
    fn save_history(
        &mut self,
        markdown: &str,
        digest_number: u32,
        is_final: bool,
    ) -> io::Result<PathBuf>;
    fn save_session_context(&mut self, context: &str) -> io::Result<PathBuf>;

    /// The session folder moved to `new_dir`. Only a folder-backed store cares.
    fn relocate(&mut self, _new_dir: PathBuf) {}
}

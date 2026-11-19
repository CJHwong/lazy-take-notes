//! The session's `debug.log`, in the Python version's line format.
//!
//! One subscriber serves the whole process; each session swaps the file it
//! writes to. The kiosk runs many sessions in one process, and each must log
//! only to its own file. A panic is written to the same file before the
//! process dies, as Python's `faulthandler` does for a native crash.

use std::fmt::Write as _;
use std::fs::{File, OpenOptions};
use std::io::Write as _;
use std::path::Path;
use std::sync::{Mutex, MutexGuard, Once};

use tracing::field::{Field, Visit};
use tracing::subscriber::Interest;
use tracing::{Event, Level, Metadata, Subscriber, info};
use tracing_subscriber::layer::{Context, Layer, SubscriberExt};

use crate::domain::session_files::DEBUG_LOG;

static SESSION_LOG: Mutex<Option<File>> = Mutex::new(None);
static INSTALL: Once = Once::new();

fn session_log() -> MutexGuard<'static, Option<File>> {
    SESSION_LOG
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Point the log at `output_dir/debug.log`, or at nothing when `enabled` is false.
/// A file that cannot be opened logs nothing: losing the log must not stop the recording.
pub fn setup_file_logging(output_dir: &Path, enabled: bool) {
    install();
    let path = output_dir.join(DEBUG_LOG.name);
    let file = enabled
        .then(|| {
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .ok()
        })
        .flatten();
    let writing = file.is_some();
    *session_log() = file;
    if writing {
        info!(target: "ltn.llm", "Debug logging started → {}", path.display());
    }
}

/// Stop writing the session's log.
pub fn close_file_logging() {
    *session_log() = None;
}

/// " (see debug.log)" while a session writes one, else "". A hint must not point at a missing file.
pub fn see_debug_log() -> String {
    if session_log().is_some() {
        format!(" (see {})", DEBUG_LOG.name)
    } else {
        String::new()
    }
}

fn install() {
    INSTALL.call_once(|| {
        let subscriber = tracing_subscriber::registry().with(SessionLayer);
        // A test harness may have set its own; then only that one logs.
        let _ = tracing::subscriber::set_global_default(subscriber);
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |panic| {
            write_line(&format_line(
                "CRITICAL",
                &format!(
                    "panic in thread {}: {panic}\n{}",
                    std::thread::current().name().unwrap_or("?"),
                    std::backtrace::Backtrace::force_capture()
                ),
            ));
            previous(panic);
        }));
    });
}

fn is_ours(metadata: &Metadata<'_>) -> bool {
    metadata.target().starts_with("ltn") && *metadata.level() <= Level::DEBUG
}

/// Python's level names.
fn level_name(level: &Level) -> &'static str {
    match *level {
        Level::ERROR => "ERROR",
        Level::WARN => "WARNING",
        Level::INFO => "INFO",
        _ => "DEBUG",
    }
}

/// `2026-11-19 16:21:08,123 INFO message`, in local time.
fn format_line(level: &str, message: &str) -> String {
    let stamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S,%3f");
    format!("{stamp} {level} {message}\n")
}

fn write_line(line: &str) {
    if let Some(file) = session_log().as_mut() {
        // A full disk loses log lines, never the session.
        let _ = file.write_all(line.as_bytes());
    }
}

struct SessionLayer;

impl<S: Subscriber> Layer<S> for SessionLayer {
    fn register_callsite(&self, metadata: &'static Metadata<'static>) -> Interest {
        // Sometimes, not always: whether a file is open changes during the process.
        if is_ours(metadata) {
            Interest::sometimes()
        } else {
            Interest::never()
        }
    }

    fn enabled(&self, metadata: &Metadata<'_>, _ctx: Context<'_, S>) -> bool {
        is_ours(metadata) && session_log().is_some()
    }

    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let mut text = Text::default();
        event.record(&mut text);
        write_line(&format_line(level_name(event.metadata().level()), &text.0));
    }
}

/// The message, then any other fields as `name=value`.
#[derive(Default)]
struct Text(String);

impl Visit for Text {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            let _ = write!(self.0, "{value:?}");
        } else {
            let _ = write!(self.0, " {}={value:?}", field.name());
        }
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "message" {
            self.0.push_str(value);
        } else {
            let _ = write!(self.0, " {}={value}", field.name());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_session_logs_only_ltn_lines_to_its_own_file() {
        let first = tempfile::tempdir().unwrap();
        let second = tempfile::tempdir().unwrap();
        setup_file_logging(first.path(), true);
        tracing::warn!(target: "ltn.audio", "mic {} lost", "built-in");
        tracing::warn!(target: "hyper", "not ours");
        setup_file_logging(second.path(), true);
        tracing::debug!(target: "ltn.persist", "second session");
        setup_file_logging(second.path(), false);
        assert_eq!(see_debug_log(), "");
        tracing::error!(target: "ltn.audio", "not written");
        // Other tests in this process log too, so check lines, not counts.
        let read = |dir: &Path| std::fs::read_to_string(dir.join(DEBUG_LOG.name)).unwrap();
        let first_log = read(first.path());
        let started = format!(
            " INFO Debug logging started → {}\n",
            first.path().join(DEBUG_LOG.name).display()
        );
        assert!(first_log.contains(&started), "{first_log}");
        let lost = first_log
            .lines()
            .find(|line| line.ends_with(" WARNING mic built-in lost"))
            .expect("the warning line");
        assert_eq!(lost.as_bytes()[19], b',', "{lost}"); // 2026-11-19 16:21:08,123
        assert!(!first_log.contains("not ours"));
        assert!(!first_log.contains("second session"));
        let second_log = read(second.path());
        assert!(
            second_log.contains(" DEBUG second session\n"),
            "{second_log}"
        );
        assert!(!second_log.contains("not written"));
    }
}

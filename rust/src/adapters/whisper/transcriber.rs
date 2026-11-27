//! The parent side: a `Transcriber` that runs whisper in a child process.

use std::ffi::OsString;
use std::io::{BufRead, BufReader};
#[cfg(unix)]
use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::thread;
use std::time::Duration;

use tracing::{info, warn};

use super::WORKER_COMMAND;
use super::protocol::{Reply, write_request};
use crate::domain::transcript::TranscriptSegment;
use crate::ports::transcriber::{TranscribeError, Transcriber};

const TIMEOUT: Duration = Duration::from_secs(120);
const CLOSE_GRACE: Duration = Duration::from_secs(5);

/// The child was gone before it answered, so no late reply can exist: the chunk is safe to retry.
struct ChildDied(String);

enum Failure {
    Died(ChildDied),
    Other(TranscribeError),
}

impl From<TranscribeError> for Failure {
    fn from(error: TranscribeError) -> Self {
        Self::Other(error)
    }
}

struct Worker {
    process: Child,
    stdin: ChildStdin,
    replies: Receiver<Reply>,
}

impl Worker {
    fn alive(&mut self) -> bool {
        matches!(self.process.try_wait(), Ok(None))
    }

    fn kill(mut self) {
        let _ = self.process.kill();
        let _ = self.process.wait();
    }
}

pub struct SubprocessWhisperTranscriber {
    program: PathBuf,
    leading_args: Vec<OsString>,
    timeout: Duration,
    worker: Option<Worker>,
    model_path: Option<String>,
}

impl SubprocessWhisperTranscriber {
    /// Runs this program as the child.
    pub fn new() -> Result<Self, TranscribeError> {
        let program = std::env::current_exe()
            .map_err(|e| TranscribeError(format!("cannot find this program: {e}")))?;
        Ok(Self::with_program(program, vec![WORKER_COMMAND.into()]))
    }

    /// The child is `program leading_args... model_path`.
    pub fn with_program(program: PathBuf, leading_args: Vec<OsString>) -> Self {
        Self {
            program,
            leading_args,
            timeout: TIMEOUT,
            worker: None,
            model_path: None,
        }
    }

    #[cfg(test)]
    fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    fn spawn(&self, model_path: &str) -> Result<Worker, TranscribeError> {
        let mut command = Command::new(&self.program);
        command
            .args(&self.leading_args)
            .arg(model_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        // Its own process group: a terminal Ctrl-C must not kill it mid-chunk in headless mode.
        #[cfg(unix)]
        command.process_group(0);
        let mut process = command
            .spawn()
            .map_err(|e| TranscribeError(format!("Whisper subprocess failed to start: {e}")))?;
        let stdin = process.stdin.take().expect("piped stdin");
        let stdout = process.stdout.take().expect("piped stdout");
        let (sender, replies) = mpsc::channel();
        // Only this thread reads the child's replies. It ends when the child closes its output.
        thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                let Ok(reply) = serde_json::from_str::<Reply>(&line) else {
                    break;
                };
                if sender.send(reply).is_err() {
                    break;
                }
            }
        });
        Ok(Worker {
            process,
            stdin,
            replies,
        })
    }

    fn kill(&mut self) {
        if let Some(worker) = self.worker.take() {
            worker.kill();
        }
    }

    fn transcribe_once(
        &mut self,
        audio: &[f32],
        language: &str,
        hints: &[String],
    ) -> Result<Vec<TranscriptSegment>, Failure> {
        if self.worker.is_none() {
            let Some(path) = self.model_path.clone() else {
                return Err(
                    TranscribeError("Model not loaded. Call load_model() first.".into()).into(),
                );
            };
            warn!(target: "ltn.whisper", "Restarting the whisper subprocess");
            if let Err(error) = self.load_model(&path) {
                self.kill();
                return Err(error.into());
            }
        }
        let timeout = self.timeout;
        let worker = self.worker.as_mut().expect("worker started above");
        // Any failure leaves the pipe in an unknown state: a late reply would be read
        // as the answer to the next chunk. Kill the child; the next call starts a fresh one.
        if let Err(error) = write_request(&mut worker.stdin, language, hints, audio) {
            self.kill();
            return Err(Failure::Died(ChildDied(format!(
                "Whisper subprocess pipe failed: {error}"
            ))));
        }
        let reply = match worker.replies.recv_timeout(timeout) {
            Ok(reply) => reply,
            Err(RecvTimeoutError::Timeout) => {
                self.kill();
                return Err(
                    TranscribeError("Timeout waiting for transcription result".into()).into(),
                );
            }
            Err(RecvTimeoutError::Disconnected) => {
                self.kill();
                return Err(Failure::Died(ChildDied(
                    "Whisper subprocess exited unexpectedly during transcription".into(),
                )));
            }
        };
        match reply {
            Reply::Ok { segments } => Ok(segments
                .into_iter()
                .map(|s| TranscriptSegment::new(s.text, s.start, s.end))
                .collect()),
            Reply::Error { error } => Err(TranscribeError(error).into()),
            Reply::Ready => {
                Err(TranscribeError("Whisper subprocess sent an unexpected reply".into()).into())
            }
        }
    }
}

impl Transcriber for SubprocessWhisperTranscriber {
    /// A no-op when this model is already loaded in a live child: the preflight loads it and the session reuses it.
    fn load_model(&mut self, model_path: &str) -> Result<(), TranscribeError> {
        if self.model_path.as_deref() == Some(model_path)
            && self.worker.as_mut().is_some_and(Worker::alive)
        {
            info!(target: "ltn.whisper", "whisper model already loaded: {model_path}");
            return Ok(());
        }
        self.kill();
        self.model_path = Some(model_path.to_string());
        let worker = self.spawn(model_path)?;
        let reply = worker.replies.recv_timeout(self.timeout);
        self.worker = Some(worker);
        let failure = match reply {
            Ok(Reply::Ready) => return Ok(()),
            Ok(Reply::Error { error }) => format!("Whisper subprocess failed to init: {error}"),
            Ok(Reply::Ok { .. }) => "Whisper subprocess failed to init: unknown".into(),
            Err(RecvTimeoutError::Timeout) => "Timeout waiting for model load".into(),
            Err(RecvTimeoutError::Disconnected) => {
                "Whisper subprocess exited unexpectedly during model load".into()
            }
        };
        self.kill();
        Err(TranscribeError(failure))
    }

    fn transcribe(
        &mut self,
        audio: &[f32],
        language: &str,
        hints: &[String],
    ) -> Result<Vec<TranscriptSegment>, TranscribeError> {
        match self.transcribe_once(audio, language, hints) {
            Ok(segments) => Ok(segments),
            Err(Failure::Other(error)) => Err(error),
            Err(Failure::Died(ChildDied(reason))) => {
                warn!(target: "ltn.whisper", "{reason}; retrying the chunk on a fresh subprocess");
                self.transcribe_once(audio, language, hints)
                    .map_err(|failure| match failure {
                        Failure::Died(ChildDied(reason)) => TranscribeError(reason),
                        Failure::Other(error) => error,
                    })
            }
        }
    }

    /// Close the input so the child exits on its own, then kill it after a grace period.
    fn close(&mut self) {
        let Some(Worker {
            mut process, stdin, ..
        }) = self.worker.take()
        else {
            return;
        };
        drop(stdin);
        let deadline = std::time::Instant::now() + CLOSE_GRACE;
        while std::time::Instant::now() < deadline {
            if !matches!(process.try_wait(), Ok(None)) {
                return;
            }
            thread::sleep(Duration::from_millis(20));
        }
        let _ = process.kill();
        let _ = process.wait();
    }
}

impl Drop for SubprocessWhisperTranscriber {
    fn drop(&mut self) {
        self.kill();
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    use super::*;

    /// A shell child that speaks the protocol. `$1` is the model path; `$COUNT` counts spawns.
    fn fake(dir: &Path, body: &str) -> SubprocessWhisperTranscriber {
        let script = format!(
            r#"echo x >> "{count}"
answer() {{ printf '{{"status":"ok","segments":[{{"text":"hello","start":0.5,"end":1.25}}]}}\n'; }}
skip() {{ n=$(printf '%s' "$1" | sed 's/.*"samples":\([0-9]*\).*/\1/'); dd bs=4 count="$n" >/dev/null 2>&1; }}
{body}"#,
            count = dir.join("spawns").display()
        );
        SubprocessWhisperTranscriber::with_program(
            "/bin/sh".into(),
            vec!["-c".into(), script.into(), "sh".into()],
        )
    }

    fn spawns(dir: &Path) -> usize {
        fs::read_to_string(dir.join("spawns")).map_or(0, |text| text.lines().count())
    }

    const SERVE: &str = r#"printf '{"status":"ready"}\n'
while IFS= read -r line; do skip "$line"; answer; done"#;

    #[test]
    fn transcribes_and_reuses_the_loaded_model() {
        let dir = tempfile::tempdir().unwrap();
        let mut whisper = fake(dir.path(), SERVE);
        whisper.load_model("model.bin").unwrap();
        whisper.load_model("model.bin").unwrap();
        assert_eq!(spawns(dir.path()), 1);
        let segments = whisper
            .transcribe(&[0.1; 1600], "en", &["hint".into()])
            .unwrap();
        assert_eq!(segments, vec![TranscriptSegment::new("hello", 0.5, 1.25)]);
        whisper.load_model("other.bin").unwrap();
        assert_eq!(spawns(dir.path()), 2);
        whisper.close();
    }

    #[test]
    fn load_failures_read_like_python() {
        let dir = tempfile::tempdir().unwrap();
        let mut failing = fake(
            dir.path(),
            r#"printf '{"status":"error","error":"bad model"}\n'"#,
        );
        assert_eq!(
            failing.load_model("m").unwrap_err().0,
            "Whisper subprocess failed to init: bad model"
        );
        let mut exiting = fake(dir.path(), "exit 1");
        assert_eq!(
            exiting.load_model("m").unwrap_err().0,
            "Whisper subprocess exited unexpectedly during model load"
        );
        let mut unloaded = fake(dir.path(), SERVE);
        assert_eq!(
            unloaded.transcribe(&[], "en", &[]).unwrap_err().0,
            "Model not loaded. Call load_model() first."
        );
    }

    #[test]
    fn a_chunk_that_hit_a_dead_child_is_retried_once_on_a_fresh_one() {
        let dir = tempfile::tempdir().unwrap();
        // The first child dies on its first chunk; the second serves.
        let body = format!(
            r#"printf '{{"status":"ready"}}\n'
if [ "$(wc -l < "{}")" -eq 1 ]; then IFS= read -r line; skip "$line"; exit 1; fi
while IFS= read -r line; do skip "$line"; answer; done"#,
            dir.path().join("spawns").display()
        );
        let mut whisper = fake(dir.path(), &body);
        whisper.load_model("m").unwrap();
        assert_eq!(whisper.transcribe(&[0.0; 8], "en", &[]).unwrap().len(), 1);
        assert_eq!(spawns(dir.path()), 2);
    }

    #[test]
    fn a_timed_out_chunk_kills_the_child_and_is_not_retried() {
        let dir = tempfile::tempdir().unwrap();
        let body = r#"printf '{"status":"ready"}\n'
while IFS= read -r line; do skip "$line"; sleep 30; done"#;
        let mut whisper = fake(dir.path(), body).with_timeout(Duration::from_millis(300));
        whisper.load_model("m").unwrap();
        assert_eq!(
            whisper.transcribe(&[0.0; 8], "en", &[]).unwrap_err().0,
            "Timeout waiting for transcription result"
        );
        assert_eq!(spawns(dir.path()), 1);
        assert!(whisper.worker.is_none(), "the hung child is killed");
    }

    #[test]
    fn an_error_reply_keeps_the_child() {
        let dir = tempfile::tempdir().unwrap();
        let body = r#"printf '{"status":"ready"}\n'
while IFS= read -r line; do skip "$line"; printf '{"status":"error","error":"bad audio"}\n'; done"#;
        let mut whisper = fake(dir.path(), body);
        whisper.load_model("m").unwrap();
        assert_eq!(
            whisper.transcribe(&[0.0; 8], "en", &[]).unwrap_err().0,
            "bad audio"
        );
        assert!(whisper.worker.as_mut().is_some_and(Worker::alive));
    }
}

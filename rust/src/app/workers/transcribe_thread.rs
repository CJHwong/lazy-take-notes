//! The transcriber on its own thread, so audio reading continues while whisper runs.
//!
//! One chunk is in flight at a time. The audio loop submits a chunk and polls
//! for the reply. A flush that must finish first goes through `Remote`, which
//! sends the chunk and waits.

use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::thread;
use std::time::Duration;

use crate::domain::transcript::TranscriptSegment;
use crate::ports::transcriber::{TranscribeError, Transcriber};

type Reply = Result<Vec<TranscriptSegment>, TranscribeError>;

struct Job {
    audio: Vec<f32>,
    language: String,
    hints: Vec<String>,
}

pub struct TranscribeThread {
    jobs: Option<Sender<Job>>,
    replies: Receiver<Reply>,
}

impl TranscribeThread {
    /// Take the transcriber. It is closed when this handle is dropped and its last chunk ends.
    pub fn spawn(mut transcriber: Box<dyn Transcriber>) -> std::io::Result<Self> {
        let (jobs, inbox) = mpsc::channel::<Job>();
        let (outbox, replies) = mpsc::channel();
        thread::Builder::new()
            .name("ltn-transcribe".into())
            .spawn(move || {
                while let Ok(job) = inbox.recv() {
                    let reply = transcriber.transcribe(&job.audio, &job.language, &job.hints);
                    if outbox.send(reply).is_err() {
                        break;
                    }
                }
                transcriber.close();
            })?;
        Ok(Self {
            jobs: Some(jobs),
            replies,
        })
    }

    pub fn submit(&self, audio: Vec<f32>, language: &str, hints: Vec<String>) {
        let job = Job {
            audio,
            language: language.to_string(),
            hints,
        };
        if let Some(jobs) = &self.jobs {
            // A dead thread shows up as a failed reply in `wait`.
            let _ = jobs.send(job);
        }
    }

    /// The reply, if it arrived.
    pub fn try_take(&self) -> Option<Reply> {
        self.replies.try_recv().ok()
    }

    /// Wait up to `timeout` for the reply. None when it did not arrive in time.
    pub fn wait(&self, timeout: Duration) -> Option<Reply> {
        match self.replies.recv_timeout(timeout) {
            Ok(reply) => Some(reply),
            Err(RecvTimeoutError::Timeout) => None,
            Err(RecvTimeoutError::Disconnected) => Some(Err(TranscribeError(
                "the transcription thread stopped".into(),
            ))),
        }
    }

    /// Run one chunk and wait for it, through the `Transcriber` interface.
    pub fn remote(&mut self) -> Remote<'_> {
        Remote { thread: self }
    }
}

impl Drop for TranscribeThread {
    fn drop(&mut self) {
        // The thread ends when its job channel closes. It is not joined: a chunk in
        // flight ends at the transcriber's own timeout, which kills the child.
        self.jobs.take();
    }
}

pub struct Remote<'a> {
    thread: &'a mut TranscribeThread,
}

impl Transcriber for Remote<'_> {
    fn load_model(&mut self, _model_path: &str) -> Result<(), TranscribeError> {
        Ok(())
    }

    fn transcribe(
        &mut self,
        audio: &[f32],
        language: &str,
        hints: &[String],
    ) -> Result<Vec<TranscriptSegment>, TranscribeError> {
        self.thread.submit(audio.to_vec(), language, hints.to_vec());
        self.thread
            .wait(Duration::MAX)
            .expect("an unbounded wait always returns")
    }
}

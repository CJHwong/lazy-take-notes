//! What worker threads tell the session. Workers send; the session's loop receives.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use tokio::sync::mpsc::UnboundedSender;

use crate::domain::speaker_turn::SpeakerTurn;
use crate::domain::transcript::TranscriptSegment;

/// The audio and file workers' states. The text is what Python posts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerState {
    LoadingModel,
    ModelReady,
    Recording,
    Warning,
    Stopped,
    Error,
}

impl WorkerState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::LoadingModel => "loading_model",
            Self::ModelReady => "model_ready",
            Self::Recording => "recording",
            Self::Warning => "warning",
            Self::Stopped => "stopped",
            Self::Error => "error",
        }
    }
}

/// `ok`, `lost` or `silent`, as in `MixedAudioSource.health`.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioHealthReport {
    pub mic: String,
    pub system: String,
    pub detail: String,
    pub notice: String,
    /// `error` while the audio loop fails on every chunk.
    pub processing: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    SpeakerTurns(Vec<SpeakerTurn>),
    /// The last diarization pass, over the whole recording, is done (or failed).
    DiarizationFinished,
    TranscriptChunk(Vec<TranscriptSegment>),
    WorkerStatus { state: WorkerState, error: String },
    AudioHealth(AudioHealthReport),
    ModelDownloadProgress { percent: u32, model_name: String },
    TranscriptionStatus { active: bool },
    /// A diarization pass started (true) or ended (false).
    DiarizationStatus { active: bool },
    AudioLevel { rms: f32 },
}

impl Event {
    pub fn status(state: WorkerState) -> Self {
        Self::WorkerStatus {
            state,
            error: String::new(),
        }
    }

    pub fn status_error(state: WorkerState, error: impl Into<String>) -> Self {
        Self::WorkerStatus {
            state,
            error: error.into(),
        }
    }
}

/// A worker's way out. A send after the session ended is dropped.
#[derive(Clone)]
pub struct Poster(UnboundedSender<Event>);

impl Poster {
    pub fn new(sender: UnboundedSender<Event>) -> Self {
        Self(sender)
    }

    pub fn post(&self, event: Event) {
        let _ = self.0.send(event);
    }
}

/// Ask a worker to act now. The worker takes the request, so a second one can follow.
#[derive(Clone, Default)]
pub struct Nudge(Arc<AtomicBool>);

impl Nudge {
    pub fn nudge(&self) {
        self.0.store(true, Ordering::SeqCst);
    }

    pub fn is_set(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }

    /// Whether a request was waiting. Clears it.
    pub fn take(&self) -> bool {
        self.0.swap(false, Ordering::SeqCst)
    }
}

/// Set once to stop a worker. Workers poll it between steps.
#[derive(Clone, Default)]
pub struct Cancel(Arc<AtomicBool>);

impl Cancel {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::SeqCst);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

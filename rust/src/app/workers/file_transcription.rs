//! Transcribe an audio file in chunks, posting each chunk's lines as it finishes.

use std::path::Path;

use tracing::error;

use crate::adapters::audio_file::load_audio_file;
use crate::app::events::{Cancel, Event, Poster, WorkerState};
use crate::domain::SAMPLE_RATE;
use crate::domain::model_asset::AssetProgress;
use crate::domain::transcript::TranscriptSegment;
use crate::ports::model_resolver::ModelResolver;
use crate::ports::transcriber::{TranscribeError, Transcriber};
use crate::usecases::transcribe_audio::{TranscribeAudio, TranscribeSettings};

/// One second of audio per feed, as the live loop receives it.
const FEED_CHUNK: usize = SAMPLE_RATE;

pub struct FileJob<'a> {
    pub audio_path: &'a Path,
    pub model_name: &'a str,
    pub language: &'a str,
    pub settings: TranscribeSettings,
    pub recognition_hints: Vec<String>,
}

/// A download percent from progress, as the Python resolver reports it.
pub fn download_percent(progress: &AssetProgress) -> Option<u32> {
    (progress.total > 0).then(|| (progress.downloaded * 100 / progress.total).min(100) as u32)
}

/// Run the whole file. Returns the lines so far, even after a failure, so they can still be saved.
pub fn run_file_transcription(
    poster: &Poster,
    cancel: &Cancel,
    job: FileJob<'_>,
    resolver: &dyn ModelResolver,
    mut transcriber: Box<dyn Transcriber>,
) -> Vec<TranscriptSegment> {
    let fail = |message: String| {
        poster.post(Event::status_error(WorkerState::Error, message));
        Vec::new()
    };
    let audio = match load_audio_file(job.audio_path) {
        Ok(audio) => audio,
        Err(e) => {
            error!(target: "ltn.file_worker", "Failed to load audio file: {e}");
            return fail(e.to_string());
        }
    };
    poster.post(Event::status(WorkerState::LoadingModel));
    let model_path = match resolver.resolve(job.model_name) {
        Ok(path) => path,
        Err(e) => {
            error!(target: "ltn.file_worker", "Failed to resolve model: {e}");
            return fail(e.to_string());
        }
    };
    if let Err(e) = transcriber.load_model(&model_path) {
        error!(target: "ltn.file_worker", "Failed to load model: {e}");
        transcriber.close();
        return fail(e.to_string());
    }
    poster.post(Event::status(WorkerState::ModelReady));
    let mut use_case = TranscribeAudio::new(job.language, job.settings, job.recognition_hints);
    let mut segments = Vec::new();
    poster.post(Event::status(WorkerState::Recording));
    // A whisper failure ends this file, not the app. The lines so far stay.
    let result = transcribe_chunks(
        &mut use_case,
        &audio,
        cancel,
        poster,
        transcriber.as_mut(),
        &mut segments,
    );
    transcriber.close();
    match result {
        Ok(()) => poster.post(Event::status(WorkerState::Stopped)),
        Err(e) => {
            error!(target: "ltn.file_worker", "Transcription failed: {e}");
            poster.post(Event::status_error(WorkerState::Error, e.to_string()));
        }
    }
    segments
}

fn transcribe_chunks(
    use_case: &mut TranscribeAudio,
    audio: &[f32],
    cancel: &Cancel,
    poster: &Poster,
    transcriber: &mut dyn Transcriber,
    segments: &mut Vec<TranscriptSegment>,
) -> Result<(), TranscribeError> {
    let mut offset = 0;
    while offset < audio.len() && !cancel.is_cancelled() {
        let chunk = &audio[offset..(offset + FEED_CHUNK).min(audio.len())];
        offset += chunk.len();
        use_case.feed_audio(chunk);
        use_case.set_session_offset(offset as f64 / SAMPLE_RATE as f64);
        if use_case.should_trigger() {
            let new = bracketed(poster, || use_case.process_buffer(transcriber, false))?;
            publish(poster, new, segments);
        }
    }
    if !cancel.is_cancelled() {
        let new = bracketed(poster, || use_case.flush(transcriber))?;
        publish(poster, new, segments);
    }
    Ok(())
}

fn bracketed<T>(poster: &Poster, step: impl FnOnce() -> T) -> T {
    poster.post(Event::TranscriptionStatus { active: true });
    let result = step();
    poster.post(Event::TranscriptionStatus { active: false });
    result
}

fn publish(poster: &Poster, new: Vec<TranscriptSegment>, segments: &mut Vec<TranscriptSegment>) {
    if new.is_empty() {
        return;
    }
    segments.extend_from_slice(&new);
    poster.post(Event::TranscriptChunk(new));
}

/// Replay a plugin's subtitle segments as transcript chunks, one by one, with no
/// whisper (Python's `run_subtitle_replay`). The panels and the digest buffer
/// take them as they take a live transcript.
pub fn run_subtitle_replay(
    poster: &Poster,
    cancel: &Cancel,
    segments: Vec<TranscriptSegment>,
) -> Vec<TranscriptSegment> {
    poster.post(Event::status(WorkerState::Recording));
    let mut replayed = Vec::new();
    for segment in segments {
        if cancel.is_cancelled() {
            break;
        }
        replayed.push(segment.clone());
        poster.post(Event::TranscriptChunk(vec![segment]));
    }
    poster.post(Event::status(WorkerState::Stopped));
    replayed
}

//! Buffering, the speech gate, overlap, and prompt chaining around a transcriber.
//!
//! No I/O here: audio comes in through `feed_audio`, segments come out of
//! `apply_result` (or `process_buffer`, which runs the transcriber in place).

use std::sync::LazyLock;

use fancy_regex::Regex as BacktrackingRegex;
use regex::Regex;
use tracing::warn;

use crate::domain::SAMPLE_RATE;
use crate::domain::transcript::TranscriptSegment;
use crate::ports::transcriber::{TranscribeError, Transcriber};

/// Whisper writes a sentence for a click, a cough or steady noise ("Thank you.",
/// "*sad*", a line from the prompt). A chunk is sent only when it holds this much
/// audio clearly above its own noise floor.
pub const MIN_VOICED_S: f64 = 1.0;
const FRAME: usize = SAMPLE_RATE / 10;
/// Hold back a cut chunk's last segment only when another one is sent. One long segment is sent as is.
const MIN_SEGMENTS_TO_HOLD: usize = 2;
/// Audio a buffer needs before a pause can end it, and before a flush sends it.
const MIN_SPEECH_S: f64 = 2.0;

/// A line that is only an annotation (*clap*, [music], (laughs), ♪) or only punctuation.
static NOT_SPEECH: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^(?:\s*(?:[*\[(♪][^\]*)♪]*[*\])♪]\s*|[\W_]*))$").expect("valid regex")
});
/// A phrase said three or more times in a row is a decoding loop, not speech.
static LOOP: LazyLock<BacktrackingRegex> =
    LazyLock::new(|| BacktrackingRegex::new(r"(\S.{8,}?)(?:\s*\1){2,}").expect("valid regex"));

/// One flag per 0.1 s frame: louder than both the threshold and twice the chunk's quietest tenth.
pub fn voiced_frames(audio: &[f32], silence_threshold: f64) -> Vec<bool> {
    let levels: Vec<f64> = audio.chunks_exact(FRAME).map(rms).collect();
    if levels.is_empty() {
        return Vec::new();
    }
    let gate = silence_threshold.max(2.0 * percentile(&levels, 10.0));
    levels.iter().map(|level| *level >= gate).collect()
}

/// Seconds of `audio` clearly louder than both the threshold and the chunk's noise floor.
pub fn voiced_seconds(audio: &[f32], silence_threshold: f64) -> f64 {
    let voiced = voiced_frames(audio, silence_threshold)
        .iter()
        .filter(|flag| **flag)
        .count();
    (voiced * FRAME) as f64 / SAMPLE_RATE as f64
}

/// Where a segment's sound begins, in seconds into the chunk.
///
/// Whisper often starts a segment in the silence before it, as far back as the
/// start of a quiet lead. A start on a voiced frame stays, and so does the start
/// of a segment with no voiced frame.
pub fn speech_start(voiced: &[bool], start: f64, end: f64) -> f64 {
    let first = (start * SAMPLE_RATE as f64) as usize / FRAME;
    let last = ((end * SAMPLE_RATE as f64) / FRAME as f64).ceil() as usize;
    let window = voiced
        .get(first..last.min(voiced.len()))
        .unwrap_or_default();
    match window.iter().position(|flag| *flag) {
        None | Some(0) => start,
        Some(loud) => ((first + loud) * FRAME) as f64 / SAMPLE_RATE as f64,
    }
}

fn rms(samples: &[f32]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let energy: f64 = samples.iter().map(|s| f64::from(*s) * f64::from(*s)).sum();
    (energy / samples.len() as f64).sqrt()
}

/// numpy's default (linear) percentile.
fn percentile(values: &[f64], percent: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let position = percent / 100.0 * (sorted.len() - 1) as f64;
    let low = position.floor() as usize;
    let high = position.ceil() as usize;
    sorted[low] + (sorted[high] - sorted[low]) * (position - low as f64)
}

/// Collapse a decoding loop, then normalize whitespace. A backreference, not a sentence
/// split: whisper's loops come quoted or with no punctuation at all.
pub fn collapse_loops(text: &str) -> String {
    let collapsed = match LOOP.try_replacen(text, 0, "$1") {
        Ok(collapsed) => collapsed.into_owned(),
        Err(e) => {
            // The backtrack limit hit. Keep the line uncleaned rather than lose it.
            warn!(target: "ltn.transcribe", "Loop check skipped: {e}");
            text.to_string()
        }
    };
    collapsed.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub fn is_not_speech(text: &str) -> bool {
    NOT_SPEECH.is_match(text)
}

/// True when `text` is only `last` said one or more times. Prompt chaining feeds the
/// last line back in, and whisper can echo it. `last` is a cleaned, non-empty line.
fn repeats(text: &str, last: &str) -> bool {
    let mut rest = text;
    let mut count = 0;
    loop {
        rest = rest.trim_start();
        if rest.is_empty() {
            return count > 0;
        }
        match rest.strip_prefix(last) {
            Some(after) => {
                rest = after;
                count += 1;
            }
            None => return false,
        }
    }
}

/// Which speech gate decides that a chunk is worth sending.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpeechGate {
    /// At least `MIN_VOICED_S` of frames above the threshold and the noise floor.
    Voiced,
    /// The whole chunk counts once its average level reaches the threshold.
    /// Only for tests that feed constant blocks, which the real gate calls silence.
    LevelOnly,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TranscribeSettings {
    pub chunk_duration: f64,
    pub overlap: f64,
    pub silence_threshold: f64,
    pub pause_duration: f64,
    pub gate: SpeechGate,
}

impl Default for TranscribeSettings {
    fn default() -> Self {
        Self {
            chunk_duration: 25.0,
            overlap: 1.0,
            silence_threshold: 0.01,
            pause_duration: 1.5,
            gate: SpeechGate::Voiced,
        }
    }
}

/// A chunk handed out for transcription.
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedChunk {
    pub audio: Vec<f32>,
    pub hints: Vec<String>,
    /// Session time of the chunk's first sample.
    pub wall_start: f64,
    /// True when the chunk's front repeats nothing of the last chunk, so every
    /// segment is kept: the first chunk, and one that starts with a held segment.
    pub starts_fresh: bool,
}

pub struct TranscribeAudio {
    language: String,
    recognition_hints: Vec<String>,
    current_hints: Vec<String>,
    chunk_samples: usize,
    overlap_samples: usize,
    pause_samples: usize,
    min_speech_samples: usize,
    silence_threshold: f64,
    gate: SpeechGate,
    buffer: Vec<f32>,
    starts_fresh: bool,
    session_offset: f64,
    /// The chunk handed out by `prepare_buffer`.
    sent: Vec<f32>,
    /// The newest line sent out, to drop a repeat of it.
    last_text: Option<String>,
}

fn samples(seconds: f64) -> usize {
    (SAMPLE_RATE as f64 * seconds) as usize
}

impl TranscribeAudio {
    pub fn new(
        language: impl Into<String>,
        settings: TranscribeSettings,
        recognition_hints: Vec<String>,
    ) -> Self {
        Self {
            language: language.into(),
            current_hints: recognition_hints.clone(),
            recognition_hints,
            chunk_samples: samples(settings.chunk_duration),
            overlap_samples: samples(settings.overlap),
            pause_samples: samples(settings.pause_duration),
            min_speech_samples: samples(MIN_SPEECH_S),
            silence_threshold: settings.silence_threshold,
            gate: settings.gate,
            buffer: Vec::with_capacity(SAMPLE_RATE * 10),
            starts_fresh: true,
            session_offset: 0.0,
            sent: Vec::new(),
            last_text: None,
        }
    }

    pub fn language(&self) -> &str {
        &self.language
    }

    pub fn overlap(&self) -> f64 {
        self.overlap_samples as f64 / SAMPLE_RATE as f64
    }

    /// Audio waiting to be sent.
    pub fn buffered(&self) -> &[f32] {
        &self.buffer
    }

    pub fn current_hints(&self) -> &[String] {
        &self.current_hints
    }

    /// Session time, in seconds, at the end of the audio fed so far.
    pub fn set_session_offset(&mut self, offset: f64) {
        self.session_offset = offset;
    }

    pub fn feed_audio(&mut self, data: &[f32]) {
        self.buffer.extend_from_slice(data);
    }

    /// Discard the buffered audio, e.g. after a pause.
    pub fn reset_buffer(&mut self) {
        self.buffer.clear();
    }

    /// True when the buffer is full, or holds speech that a pause just ended.
    pub fn should_trigger(&self) -> bool {
        let length = self.buffer.len();
        if length >= self.chunk_samples {
            return true;
        }
        if length < self.min_speech_samples + self.pause_samples {
            return false;
        }
        let (body, tail) = self.buffer.split_at(length - self.pause_samples);
        rms(tail) < self.silence_threshold && rms(body) >= self.silence_threshold
    }

    fn worth_sending(&self, audio: &[f32]) -> bool {
        match self.gate {
            // Not the chunk average: a quiet lead pulls it under the threshold and loses the speech after it.
            SpeechGate::Voiced => voiced_seconds(audio, self.silence_threshold) >= MIN_VOICED_S,
            SpeechGate::LevelOnly => !audio.is_empty() && rms(audio) >= self.silence_threshold,
        }
    }

    /// Keep only the overlap tail of the buffer.
    fn keep_tail(&mut self) {
        let start = self.buffer.len().saturating_sub(self.overlap_samples);
        self.buffer.drain(..start);
    }

    /// Hand out the buffer for transcription on another thread, and keep its overlap tail.
    /// None when the buffer is empty or holds no speech.
    pub fn prepare_buffer(&mut self) -> Option<PreparedChunk> {
        if self.buffer.is_empty() {
            return None;
        }
        if !self.worth_sending(&self.buffer) {
            self.keep_tail();
            return None;
        }
        self.sent = self.buffer.clone();
        let chunk = PreparedChunk {
            audio: self.sent.clone(),
            hints: self.current_hints.clone(),
            wall_start: self.session_offset - self.buffer.len() as f64 / SAMPLE_RATE as f64,
            starts_fresh: self.starts_fresh,
        };
        self.keep_tail();
        self.starts_fresh = false;
        Some(chunk)
    }

    /// Turn a transcription of a prepared chunk into session lines.
    ///
    /// Drops the overlap region, moves times to the session clock, cleans the text,
    /// and chains the last line into the next prompt. A chunk cut at the size limit
    /// can end mid-word, and whisper then invents an ending. Unless `is_final`, the
    /// last segment of such a chunk waits: its audio goes back to the front of the
    /// buffer and is sent whole next time.
    pub fn apply_result(
        &mut self,
        segments: Vec<TranscriptSegment>,
        wall_start: f64,
        starts_fresh: bool,
        is_final: bool,
    ) -> Vec<TranscriptSegment> {
        let min_start = if starts_fresh { 0.0 } else { self.overlap() };
        let mut segments: Vec<TranscriptSegment> = segments
            .into_iter()
            .filter(|s| s.wall_end > min_start)
            .collect();
        let held_from = if is_final {
            None
        } else {
            self.cut_segment_start(&segments)
        };
        if held_from.is_some() {
            segments.pop();
        }

        let voiced = voiced_frames(&self.sent, self.silence_threshold);
        let mut lines = Vec::new();
        for segment in segments {
            let start = wall_start + speech_start(&voiced, segment.wall_start, segment.wall_end);
            let moved = TranscriptSegment::new(segment.text, start, wall_start + segment.wall_end);
            if let Some(line) = self.clean(moved) {
                lines.push(line);
            }
        }
        if let Some(last) = lines.last() {
            self.current_hints = self.recognition_hints.clone();
            self.current_hints.push(last.text.clone());
        }
        if let Some(start) = held_from {
            self.resend_from(start);
        }
        lines
    }

    /// Drop what whisper writes for non-speech, and collapse a decoding loop.
    fn clean(&mut self, segment: TranscriptSegment) -> Option<TranscriptSegment> {
        let text = collapse_loops(segment.text.trim());
        if is_not_speech(&text)
            || self
                .last_text
                .as_deref()
                .is_some_and(|last| repeats(&text, last))
        {
            return None;
        }
        self.last_text = Some(text.clone());
        Some(TranscriptSegment::new(
            text,
            segment.wall_start,
            segment.wall_end,
        ))
    }

    /// The sample where the last segment starts, when the sent chunk was cut at the size limit.
    fn cut_segment_start(&self, segments: &[TranscriptSegment]) -> Option<usize> {
        if self.sent.len() < self.chunk_samples || segments.len() < MIN_SEGMENTS_TO_HOLD {
            return None;
        }
        let start = segments.last()?.wall_start * SAMPLE_RATE as f64;
        (start >= 1.0 && start < self.sent.len() as f64).then_some(start as usize)
    }

    fn resend_from(&mut self, start: usize) {
        // The buffer holds the overlap tail of the sent chunk, then what arrived since.
        let tail = self
            .overlap_samples
            .min(self.sent.len())
            .min(self.buffer.len());
        let mut resent = self.sent[start..].to_vec();
        resent.extend_from_slice(&self.buffer[tail..]);
        self.buffer = resent;
        // The next chunk starts where the held segment starts, so nothing in it is a repeat.
        self.starts_fresh = true;
    }

    /// Transcribe the buffer now. See `apply_result`.
    pub fn process_buffer(
        &mut self,
        transcriber: &mut dyn Transcriber,
        is_final: bool,
    ) -> Result<Vec<TranscriptSegment>, TranscribeError> {
        let Some(chunk) = self.prepare_buffer() else {
            return Ok(Vec::new());
        };
        let segments = transcriber.transcribe(&chunk.audio, &self.language, &chunk.hints)?;
        Ok(self.apply_result(segments, chunk.wall_start, chunk.starts_fresh, is_final))
    }

    /// Send what is left on shutdown, if it is long enough to hold speech.
    pub fn flush(
        &mut self,
        transcriber: &mut dyn Transcriber,
    ) -> Result<Vec<TranscriptSegment>, TranscribeError> {
        if self.buffer.len() < self.min_speech_samples {
            return Ok(Vec::new());
        }
        self.process_buffer(transcriber, true)
    }
}

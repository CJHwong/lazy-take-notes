//! What the bottom line of a session says: state, digest buffer, elapsed
//! time, audio wave and key hints. Each front end draws it its own way.

use std::collections::VecDeque;
use std::time::Instant;

const WAVE_CHARS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
/// The level meter maps -60 dB (silence) to -11 dB (loud) over 8 bars.
const DB_FLOOR: f64 = -60.0;
const DB_RANGE: f64 = 49.0;

/// Anything the speech gate counts as silent shows as silent too.
pub fn rms_to_char(rms: f64, silence_threshold: f64) -> char {
    if rms.is_nan() || rms < silence_threshold.max(1e-7) {
        return WAVE_CHARS[0];
    }
    if rms.is_infinite() {
        return WAVE_CHARS[7];
    }
    let db = 20.0 * rms.log10();
    let index = ((db - DB_FLOOR) / DB_RANGE * 7.0) as i64;
    WAVE_CHARS[index.clamp(0, 7) as usize]
}

pub struct StatusLine {
    pub recording: bool,
    pub paused: bool,
    pub stopped: bool,
    pub audio_status: String,
    pub transcribing: bool,
    /// A diarization pass runs.
    pub labeling_speakers: bool,
    pub activity: String,
    pub download_percent: i32,
    pub download_model: String,
    pub buf_count: usize,
    pub buf_max: usize,
    /// False in a transcript-only session: no digest will use the buffer.
    pub ai_notes: bool,
    pub silence_threshold: f64,
    pub last_digest_time: Option<Instant>,
    pub mic_muted: bool,
    /// A capture problem that lasts until fixed, such as `MIC LOST`. Shown in red.
    pub audio_alert: String,
    pub mode_label: String,
    /// What the bar shows while audio flows: a recording, or a file being transcribed.
    pub active_label: String,
    pub keybinding_hints: String,
    pub quick_action_hints: String,
    levels: VecDeque<f64>,
    start: Option<Instant>,
    frozen: Option<f64>,
    pause_start: Option<Instant>,
    paused_total: f64,
}

impl Default for StatusLine {
    fn default() -> Self {
        Self {
            recording: false,
            paused: false,
            stopped: false,
            audio_status: String::new(),
            transcribing: false,
            labeling_speakers: false,
            activity: String::new(),
            download_percent: -1,
            download_model: String::new(),
            buf_count: 0,
            buf_max: 15,
            ai_notes: true,
            silence_threshold: 0.0,
            last_digest_time: None,
            mic_muted: false,
            audio_alert: String::new(),
            mode_label: String::new(),
            active_label: "● Rec".into(),
            keybinding_hints: String::new(),
            quick_action_hints: String::new(),
            levels: VecDeque::from(vec![0.0; 6]),
            start: None,
            frozen: None,
            pause_start: None,
            paused_total: 0.0,
        }
    }
}

impl StatusLine {
    /// Catch up with the session, and take the input levels it gathered.
    pub fn sync_from(&mut self, status: &mut super::Status) {
        // Paused first: stopping while paused leaves the pause out of the timer.
        self.set_paused(status.paused);
        self.set_recording(status.recording);
        self.set_stopped(status.stopped);
        self.audio_status = status.audio_status.clone();
        self.mic_muted = status.mic_muted;
        self.audio_alert = status.audio_alert.clone();
        self.transcribing = status.transcribing;
        self.labeling_speakers = status.labeling_speakers;
        self.activity = status.activity.clone();
        self.download_percent = status.download_percent;
        self.download_model = status.download_model.clone();
        self.buf_count = status.buf_count;
        self.last_digest_time = status.last_digest_at;
        for rms in std::mem::take(&mut status.levels) {
            self.push_level(rms);
        }
    }

    /// The elapsed timer starts on the first change to recording.
    pub fn set_recording(&mut self, value: bool) {
        self.recording = value;
        if value && self.start.is_none() {
            self.start = Some(Instant::now());
        }
    }

    /// Paused time is left out of the elapsed timer.
    pub fn set_paused(&mut self, value: bool) {
        if value == self.paused {
            return;
        }
        self.paused = value;
        if value {
            self.pause_start = Some(Instant::now());
        } else if let Some(start) = self.pause_start.take() {
            self.paused_total += start.elapsed().as_secs_f64();
        }
    }

    /// Freeze the elapsed timer, recording time only.
    pub fn set_stopped(&mut self, value: bool) {
        self.stopped = value;
        if !value || self.frozen.is_some() {
            return;
        }
        self.frozen = Some(match self.start {
            None => 0.0,
            Some(_) => self.recording_elapsed(Instant::now()),
        });
    }

    pub fn push_level(&mut self, rms: f64) {
        self.levels.push_back(rms);
        while self.levels.len() > 6 {
            self.levels.pop_front();
        }
    }

    fn recording_elapsed(&self, now: Instant) -> f64 {
        let Some(start) = self.start else {
            return 0.0;
        };
        let mut paused = self.paused_total;
        if let Some(pause_start) = self.pause_start {
            paused += (now - pause_start).as_secs_f64();
        }
        (now - start).as_secs_f64() - paused
    }

    pub fn format_elapsed(&self, now: Instant) -> String {
        if self.start.is_none() && self.frozen.is_none() {
            return "00:00:00".into();
        }
        let elapsed = self
            .frozen
            .unwrap_or_else(|| self.recording_elapsed(now))
            .max(0.0);
        let total = elapsed as u64;
        format!(
            "{:02}:{:02}:{:02}",
            total / 3600,
            (total % 3600) / 60,
            total % 60
        )
    }

    pub fn status_icon(&self) -> String {
        if self.stopped {
            "■ Stopped".into()
        } else if self.paused {
            "❚❚ Paused".into()
        } else if self.recording {
            self.active_label.clone()
        } else if self.download_percent >= 0 {
            format!(
                "⟳ Downloading {}… {}%",
                self.download_model, self.download_percent
            )
        } else if self.audio_status == "loading_model" {
            "⟳ Loading model".into()
        } else if self.audio_status == "error" {
            "✗ Error".into()
        } else {
            "○ Idle".into()
        }
    }

    /// The parts of the bottom line before the hints; the alert is the one in red.
    pub fn parts(&self, now: Instant) -> Vec<(String, bool)> {
        let mut parts = Vec::new();
        if !self.mode_label.is_empty() {
            parts.push((self.mode_label.clone(), false));
        }
        if !self.audio_alert.is_empty() {
            parts.push((format!(" {} ", self.audio_alert), true));
        }
        if self.mic_muted {
            parts.push(("MIC ✕".into(), false));
        }
        parts.push((self.status_icon(), false));
        parts.push((
            if self.ai_notes {
                format!("buf {}/{}", self.buf_count, self.buf_max)
            } else {
                "AI notes off".into()
            },
            false,
        ));
        parts.push((self.format_elapsed(now), false));
        if let Some(last) = self.last_digest_time {
            let since = (now - last).as_secs_f64();
            let text = if since < 60.0 {
                format!("last {}s ago", since as u64)
            } else {
                format!("last {}m ago", (since / 60.0) as u64)
            };
            parts.push((text, false));
        }
        if self.recording {
            let wave: String = self
                .levels
                .iter()
                .map(|level| rms_to_char(*level, self.silence_threshold))
                .collect();
            parts.push((wave, false));
        }
        if self.transcribing {
            parts.push(("⟳ Transcribing…".into(), false));
        }
        if self.labeling_speakers {
            parts.push(("⟳ Labeling speakers…".into(), false));
        }
        if !self.activity.is_empty() {
            parts.push((format!("⟳ {}", self.activity), false));
        }
        parts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_wave_follows_the_db_scale() {
        assert_eq!(rms_to_char(0.0, 0.0), '▁');
        assert_eq!(rms_to_char(0.002, 0.01), '▁');
        assert_eq!(rms_to_char(f64::INFINITY, 0.0), '█');
        assert_eq!(rms_to_char(0.3, 0.0), '█');
        // -40 dB: (20 / 49 * 7) = 2.86 -> index 2.
        assert_eq!(rms_to_char(0.01, 0.0), '▃');
    }
}

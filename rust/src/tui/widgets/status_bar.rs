//! The bottom bar: state, digest buffer, elapsed time, audio wave and key hints.

use std::collections::VecDeque;
use std::time::Instant;

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use crate::tui::text::str_width;
use crate::tui::theme::Theme;

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

pub struct StatusBar {
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

impl Default for StatusBar {
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

impl StatusBar {
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

    fn format_elapsed(&self, now: Instant) -> String {
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

    fn status_icon(&self) -> String {
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
    fn parts(&self, now: Instant) -> Vec<(String, bool)> {
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

    /// The bar's rows as plain text. `width` is the bar's content width, inside its padding.
    pub fn text(&self, width: u16, now: Instant) -> Vec<String> {
        self.lines(width, now)
            .iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|span| span.content.as_ref())
                    .collect()
            })
            .collect()
    }

    pub fn height(&self) -> u16 {
        if self.quick_action_hints.is_empty() {
            1
        } else {
            2
        }
    }

    fn lines(&self, width: u16, now: Instant) -> Vec<Line<'static>> {
        let content_width = usize::from(width).saturating_sub(2);
        let alert = Style::default()
            .fg(Color::White)
            .bg(Color::Red)
            .add_modifier(Modifier::BOLD);
        let mut spans = Vec::new();
        let mut used = 0;
        for (index, (text, is_alert)) in self.parts(now).into_iter().enumerate() {
            if index > 0 {
                spans.push(Span::raw(" │ "));
                used += 3;
            }
            used += str_width(&text);
            spans.push(if is_alert {
                Span::styled(text, alert)
            } else {
                Span::raw(text)
            });
        }
        let hints = &self.keybinding_hints;
        if !hints.is_empty() {
            let gap = content_width as i64 - used as i64 - str_width(hints) as i64;
            if gap >= 2 {
                spans.push(Span::raw(format!("{}{hints}", " ".repeat(gap as usize))));
            }
        }
        let bottom = Line::from(spans);
        if self.quick_action_hints.is_empty() {
            return vec![bottom];
        }
        let quick = &self.quick_action_hints;
        let gap = content_width.saturating_sub(str_width(quick));
        vec![Line::from(format!("{}{quick}", " ".repeat(gap))), bottom]
    }

    pub fn render(&self, frame: &mut Frame, area: Rect, theme: &Theme) {
        let style = Style::default()
            .bg(theme.color("surface"))
            .fg(theme.on("text", "surface"));
        frame.render_widget(Paragraph::new("").style(style), area);
        let inner = Rect::new(
            area.x + 1,
            area.y,
            area.width.saturating_sub(2),
            area.height,
        );
        let lines = self.lines(inner.width, Instant::now());
        frame.render_widget(Paragraph::new(lines).style(style), inner);
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

    #[test]
    fn the_view_bar_reads_like_python() {
        let mut bar = StatusBar {
            mode_label: "View".into(),
            keybinding_hints: "[c] copy  [o] open  [Tab] switch  [q] back".into(),
            ..StatusBar::default()
        };
        bar.set_stopped(true);
        let text = bar.text(100, Instant::now());
        assert_eq!(text.len(), 1);
        assert!(
            text[0].starts_with("View │ ■ Stopped │ buf 0/15 │ 00:00:00 "),
            "{}",
            text[0]
        );
        assert!(text[0].ends_with("[c] copy  [o] open  [Tab] switch  [q] back"));
        assert_eq!(str_width(&text[0]), 98);
    }
}

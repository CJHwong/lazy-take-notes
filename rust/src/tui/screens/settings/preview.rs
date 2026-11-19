//! The Appearance page's recording preview, drawn with the record screen's widgets.

use std::any::Any;

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders};

use super::form::Custom;
use crate::tui::screens::view::{DigestPanel, TranscriptPanel};
use crate::tui::theme::Theme;
use crate::tui::widgets::status_bar::StatusBar;

const SAMPLE_TRANSCRIPT: [(&str, &str); 5] = [
    ("00:01:23", "So the main takeaway from today is..."),
    ("00:02:05", "Right, and we should follow up on the"),
    ("00:02:34", "deployment timeline before next sprint."),
    ("00:03:12", "Agreed. Let me also check the metrics"),
    ("00:03:45", "from last week before we commit to that."),
];

const SAMPLE_DIGEST: &str = "## Meeting Summary

**Key decisions:**
- Follow up on deployment timeline
- Review last week metrics

## Action Items
- Check sprint metrics
";

pub struct Preview {
    transcript: TranscriptPanel,
    digest: DigestPanel,
    status: StatusBar,
}

impl Preview {
    pub fn new() -> Self {
        let mut transcript = TranscriptPanel::default();
        for (time, text) in SAMPLE_TRANSCRIPT {
            transcript.view.write(Line::from(vec![
                Span::styled(
                    format!("[{time}]"),
                    Style::default().add_modifier(Modifier::DIM),
                ),
                Span::raw(format!(" {text}")),
            ]));
        }
        let mut digest = DigestPanel::default();
        digest.update_digest(SAMPLE_DIGEST);
        let mut status = StatusBar::default();
        status.set_recording(true);
        status.mode_label = "Record".into();
        status.buf_count = 5;
        status.buf_max = 15;
        status.keybinding_hints = "[Space] pause  [s] stop  [d] digest  [m] mute  [h] help".into();
        Self {
            transcript,
            digest,
            status,
        }
    }
}

impl Custom for Preview {
    fn height(&self, _width: u16) -> u16 {
        // The page gives the preview the rest of its height (`height: 1fr`).
        3
    }

    fn render(&mut self, frame: &mut Frame, area: Rect, _focus: Option<usize>, theme: &Theme) {
        if area.height < 4 || area.width < 4 {
            return;
        }
        // `border: round $surface-lighten-2` on the page background.
        let block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(
                Style::default()
                    .fg(theme.color("surface-lighten-2"))
                    .bg(theme.color("background")),
            )
            .style(Style::default().bg(theme.color("background")));
        let inner = block.inner(area);
        frame.render_widget(block, area);
        let header = Rect::new(inner.x, inner.y, inner.width, 1);
        let style = Style::default()
            .bg(theme.color("primary"))
            .fg(theme.on("text", "primary"))
            .add_modifier(Modifier::BOLD);
        frame.render_widget(
            ratatui::widgets::Paragraph::new("   lazy-take-notes | default_en").style(style),
            header,
        );
        let status_height = self.status.height().min(inner.height.saturating_sub(1));
        let panels = Rect::new(
            inner.x,
            inner.y + 1,
            inner.width,
            inner.height.saturating_sub(1 + status_height),
        );
        let left = panels.width / 2;
        let transcript = Rect::new(panels.x, panels.y, left, panels.height);
        let digest = Rect::new(
            panels.x + left,
            panels.y,
            panels.width - left,
            panels.height,
        );
        self.transcript.render(frame, transcript, false, theme);
        self.digest.render(frame, digest, false, theme);
        // Without the record screen's CSS, a panel title takes its border color,
        // and the transcript log always shows its scrollbar track.
        title_color(frame, transcript, " Transcript ", theme.color("primary"));
        title_color(frame, digest, " Digest ", theme.color("secondary"));
        let track = theme.color("scrollbar-background");
        let x = transcript.right().saturating_sub(2);
        for y in transcript.y + 1..transcript.bottom().saturating_sub(1) {
            if let Some(cell) = frame.buffer_mut().cell_mut((x, y))
                && cell.symbol() == " "
            {
                cell.set_bg(track);
            }
        }
        let status = Rect::new(inner.x, panels.bottom(), inner.width, status_height);
        self.status.render(frame, status, theme);
    }

    fn as_any(&mut self) -> &mut dyn Any {
        self
    }
}

/// Recolor a panel's border title, drawn after `┌─`.
fn title_color(frame: &mut Frame, panel: Rect, title: &str, color: ratatui::style::Color) {
    let buffer = frame.buffer_mut();
    for offset in 0..title.chars().count() as u16 {
        if let Some(cell) = buffer.cell_mut((panel.x + 2 + offset, panel.y)) {
            cell.set_fg(color);
        }
    }
}

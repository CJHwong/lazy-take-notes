//! The session's modals: help, a quick-action result, the label prompt and the
//! model download notice.

use crate::app::notices::{BUILTIN_NOTICE_BODY, CONSENT_NOTICE_BODY};
use std::path::{Path, PathBuf};
use std::time::Duration;

use crossterm::event::KeyEvent;
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::Line;
use ratatui::widgets::Paragraph;

use crate::app::events::{Event, WorkerState};
use crate::tui::host::{Action, Ctx, MouseInput, MouseKind, Msg, Screen, Severity};
use crate::tui::keys::{Binding, SCREEN_BASE, bind};
use crate::tui::system::copy_to_clipboard;
use crate::tui::text::center;
use crate::tui::theme::Theme;
use crate::tui::widgets::input::Input;
use crate::tui::widgets::scroll::ScrollView;
use crate::tui::widgets::{centered, dialog_box, thick_box};

pub const HELP_BINDINGS: [Binding; 6] = [
    SCREEN_BASE[0],
    SCREEN_BASE[1],
    SCREEN_BASE[2],
    SCREEN_BASE[3],
    bind("escape", "dismiss", "Close"),
    bind("h", "dismiss", "Close"),
];
pub const QUERY_BINDINGS: [Binding; 6] = [
    SCREEN_BASE[0],
    SCREEN_BASE[1],
    SCREEN_BASE[2],
    SCREEN_BASE[3],
    bind("escape", "dismiss", "Close"),
    bind("c", "copy_body", "Copy"),
];
/// The built-in templates notice. Any other key also closes it, through `on_key`.
pub const NOTICE_BINDINGS: [Binding; 5] = [
    SCREEN_BASE[0],
    SCREEN_BASE[1],
    SCREEN_BASE[2],
    SCREEN_BASE[3],
    bind("escape", "dismiss", "Close"),
];
pub const LABEL_BINDINGS: [Binding; 5] = [
    SCREEN_BASE[0],
    SCREEN_BASE[1],
    SCREEN_BASE[2],
    SCREEN_BASE[3],
    bind("escape", "cancel", "Cancel"),
];

/// `width` percent of `total`, at most `max` columns.
fn percent_of(total: u16, percent: u32, max: u16) -> u16 {
    ((u32::from(total) * percent / 100) as u16).min(max)
}

/// One centered row in `style`.
fn centered_row(frame: &mut Frame, area: Rect, text: &str, style: Style) {
    let line = center(Line::from(text.to_string()), area.width);
    frame.render_widget(Paragraph::new(line).style(style), area);
}

/// Which modal a `MarkdownModal` is: its keys and its colors differ.
#[derive(Clone, PartialEq, Eq)]
enum Kind {
    Help,
    Query,
    /// A one-time notice: any key closes it, and `suppress_key` also writes `flag`.
    Notice {
        flag: PathBuf,
        suppress_key: &'static str,
        color: &'static str,
    },
}

/// A `VerticalScroll` modal: a bold title, a Markdown body and a hint docked at the bottom.
pub struct MarkdownModal {
    title: String,
    body_markdown: String,
    body: ScrollView,
    hint: &'static str,
    is_error: bool,
    width_percent: u32,
    max_width: u16,
    max_height_percent: u32,
    kind: Kind,
}

impl MarkdownModal {
    fn new(title: &str, body: &str, hint: &'static str) -> Self {
        let mut view = ScrollView::markdown();
        // The modal's `padding: 1 2` holds the scrollbar; the Markdown keeps its own `0 2`.
        view.padding = 0;
        view.markdown_padding = 2;
        view.bar_width = 2;
        // The hint is docked at the bottom with `margin-top: 1`.
        view.dock_rows = 2;
        view.dock_cover = 1;
        view.set_markdown(body);
        Self {
            title: title.into(),
            body_markdown: body.into(),
            body: view,
            hint,
            is_error: false,
            width_percent: 60,
            max_width: 80,
            max_height_percent: 80,
            kind: Kind::Help,
        }
    }

    pub fn help(body: &str) -> Self {
        Self::new("Help", body, "Press Escape or h to close")
    }

    pub fn query(title: &str, body: &str, is_error: bool) -> Self {
        Self {
            is_error,
            width_percent: 80,
            max_width: 100,
            kind: Kind::Query,
            // Python writes "[Esc] Close  [c] Copy"; Textual reads both brackets as tags.
            ..Self::new(title, body, " Close   Copy")
        }
    }

    /// Python's `BuiltinTemplatesNotice`. Any key closes it; `d` also writes `noticed`.
    pub fn builtin_templates_notice(noticed: PathBuf) -> Self {
        Self {
            max_width: 72,
            max_height_percent: 60,
            kind: Kind::Notice {
                flag: noticed,
                suppress_key: "d",
                color: "accent",
            },
            ..Self::new(
                "Built-in Templates",
                BUILTIN_NOTICE_BODY,
                "Press any key to dismiss, or [d] to stop showing this.",
            )
        }
    }

    /// Python's `ConsentNotice`, shown when a recording starts. `n` writes `noticed`.
    pub fn consent_notice(noticed: PathBuf) -> Self {
        Self {
            max_width: 72,
            max_height_percent: 60,
            kind: Kind::Notice {
                flag: noticed,
                suppress_key: "n",
                color: "warning",
            },
            ..Self::new(
                "Recording Notice",
                CONSENT_NOTICE_BODY,
                "Press any key to dismiss, or [n] to stop showing this.",
            )
        }
    }

    pub fn is_help(&self) -> bool {
        self.kind == Kind::Help
    }

    fn border_var(&self) -> &'static str {
        match (&self.kind, self.is_error) {
            (Kind::Notice { color, .. }, _) => color,
            (_, true) => "error",
            (_, false) => "primary",
        }
    }

    /// A key on a notice: any key closes it, and the suppress key stops it for good.
    fn notice_key(noticed: &Path, suppress: bool, ctx: &mut Ctx) -> Action {
        if suppress {
            let written = noticed
                .parent()
                .map_or(Ok(()), std::fs::create_dir_all)
                .and_then(|()| std::fs::write(noticed, ""));
            if let Err(e) = written {
                ctx.error(
                    format!("Cannot save the setting: {e}"),
                    Duration::from_secs(8),
                );
            }
        }
        Action::Pop(None)
    }
}

impl Screen for MarkdownModal {
    fn render(&mut self, frame: &mut Frame, area: Rect, ctx: &Ctx) {
        let theme = &ctx.theme;
        let surface = theme.color("surface");
        let text = Style::default()
            .fg(theme.on("foreground", "surface"))
            .bg(surface);
        // The title sits in the border; the body and the docked hint fill the box.
        self.body.prefix = Vec::new();
        let width = percent_of(area.width, self.width_percent, self.max_width);
        let content_rows = self.body.content_rows(width.saturating_sub(2 + 2), theme) as u16;
        let wanted = 2 + content_rows;
        let height = wanted.min(percent_of(area.height, self.max_height_percent, u16::MAX));
        let outer = centered(area, width, height);
        let inner = dialog_box(
            frame,
            outer,
            &self.title,
            self.border_var(),
            "surface",
            theme,
        );
        ctx.hit(inner, "body", 0);
        let padded = inner;
        if padded.height < 2 {
            return;
        }
        let hint_y = padded.bottom() - 1;
        self.body.render(frame, padded, text, theme);
        // The docked hint sits inside the scroll viewport, beside the scrollbar.
        let bar = if self.body.overflows() {
            self.body.bar_width
        } else {
            0
        };
        let hint_row = Rect::new(padded.x, hint_y, padded.width.saturating_sub(bar), 1);
        centered_row(
            frame,
            hint_row,
            self.hint,
            text.fg(theme.on("text-muted", "surface")),
        );
    }

    /// The wheel scrolls the body; a click on a notice closes it, as a key does.
    fn mouse(&mut self, input: &MouseInput, _ctx: &mut Ctx) -> Action {
        match input.kind {
            MouseKind::WheelUp => self.body.scroll_by(-1),
            MouseKind::WheelDown => self.body.scroll_by(1),
            MouseKind::Click => {
                if matches!(self.kind, Kind::Notice { .. }) && input.on("body").is_some() {
                    return Action::Pop(None);
                }
            }
        }
        Action::None
    }

    fn key(&mut self, _key: &KeyEvent, name: &str, ctx: &mut Ctx) -> Action {
        if let Kind::Notice {
            flag, suppress_key, ..
        } = &self.kind
        {
            return Self::notice_key(flag, *suppress_key == name, ctx);
        }
        let table: &[Binding] = if self.is_help() {
            &HELP_BINDINGS
        } else {
            &QUERY_BINDINGS
        };
        match crate::tui::keys::action_for(table, name) {
            Some("dismiss") => return Action::Pop(None),
            Some("copy_body") => {
                match copy_to_clipboard(&self.body_markdown) {
                    Ok(()) => ctx.notify_with(
                        "Copied to clipboard",
                        Severity::Information,
                        Duration::from_secs(2),
                        "",
                    ),
                    Err(e) => ctx.error(
                        format!("Cannot copy to the clipboard: {e}"),
                        Duration::from_secs(8),
                    ),
                }
                return Action::None;
            }
            _ => {}
        }
        self.body.key(name);
        Action::None
    }

    fn passes_focus_keys(&self) -> bool {
        matches!(self.kind, Kind::Notice { .. })
    }

    fn is_modal(&self) -> bool {
        true
    }
}

/// The label prompt. Pops with the stripped text, or None for Escape or an empty value.
pub struct LabelModal {
    title: String,
    input: Input,
}

impl LabelModal {
    pub fn new(current: &str, title: &str, placeholder: &str) -> Self {
        let mut input = Input::new(placeholder);
        input.parent_background = Some("surface");
        input.set_value(current);
        input.select_all();
        // A darker row inside the dialog marks the field.
        input.compact = Some(("background", "background"));
        Self {
            title: title.into(),
            input,
        }
    }

    pub fn session(current: &str) -> Self {
        Self::new(current, "Session label", "Session label...")
    }
}

impl Screen for LabelModal {
    fn render(&mut self, frame: &mut Frame, area: Rect, ctx: &Ctx) {
        let theme = &ctx.theme;
        // The title in the border, the input row, a blank row, the hint.
        let outer = centered(area, 60, 2 + 3);
        let inner = dialog_box(frame, outer, &self.title, "primary", "surface", theme);
        if inner.height < 3 {
            return;
        }
        let text = Style::default()
            .fg(theme.on("foreground", "surface"))
            .bg(theme.color("surface"));
        self.input.render_compact(
            frame,
            Rect::new(inner.x + 1, inner.y, inner.width.saturating_sub(2), 1),
            true,
            theme,
        );
        centered_row(
            frame,
            Rect::new(inner.x + 1, inner.y + 2, inner.width.saturating_sub(2), 1),
            "Press Enter to confirm, or Escape to cancel.",
            text.fg(theme.on("text-muted", "surface")),
        );
    }

    fn key(&mut self, key: &KeyEvent, name: &str, _ctx: &mut Ctx) -> Action {
        match name {
            "escape" => Action::Pop(None),
            "enter" => {
                let text = self.input.value().trim().to_string();
                Action::Pop((!text.is_empty()).then(|| Box::new(text) as _))
            }
            _ => {
                self.input.key(key, name);
                Action::None
            }
        }
    }

    fn paste(&mut self, text: &str, _ctx: &mut Ctx) -> Action {
        self.input.paste(text);
        Action::None
    }

    fn is_modal(&self) -> bool {
        true
    }
}

/// Shown while the speech model downloads, then while it loads. It closes itself
/// when the worker reports the model ready, recording, or an error.
pub struct DownloadModal {
    model_name: String,
    status: String,
}

impl DownloadModal {
    pub fn new(model_name: &str) -> Self {
        Self {
            model_name: model_name.into(),
            status: "Downloading model…".into(),
        }
    }

    fn render_box(&self, frame: &mut Frame, area: Rect, theme: &Theme) {
        let widest = crate::tui::text::str_width(&self.status)
            .max(crate::tui::text::str_width(&self.model_name)) as u16;
        let width = (widest + 6 + 2).clamp(40, 60);
        let outer = centered(area, width, 2 + 3);
        let inner = thick_box(frame, outer, "primary", "surface", theme);
        if inner.height < 3 {
            return;
        }
        let padded = Rect::new(inner.x + 1, inner.y, inner.width.saturating_sub(2), 3);
        let text = Style::default()
            .fg(theme.on("foreground", "surface"))
            .bg(theme.color("surface"));
        centered_row(
            frame,
            Rect::new(padded.x, padded.y, padded.width, 1),
            &self.status,
            text.add_modifier(Modifier::BOLD),
        );
        centered_row(
            frame,
            Rect::new(padded.x, padded.y + 2, padded.width, 1),
            &self.model_name,
            text.fg(theme.on("text-muted", "surface")),
        );
    }
}

impl Screen for DownloadModal {
    fn render(&mut self, frame: &mut Frame, area: Rect, ctx: &Ctx) {
        self.render_box(frame, area, &ctx.theme);
    }

    fn key(&mut self, _key: &KeyEvent, _name: &str, _ctx: &mut Ctx) -> Action {
        Action::None
    }

    /// Follow the worker's progress, and pass every event on to the session.
    fn message(&mut self, msg: Msg, _ctx: &mut Ctx) -> (Option<Msg>, Action) {
        let mut action = Action::None;
        match &msg {
            Msg::Worker(Event::ModelDownloadProgress { percent, .. }) => {
                self.status = format!("Downloading model… {percent}%");
            }
            Msg::Worker(Event::WorkerStatus { state, .. }) => match state {
                WorkerState::LoadingModel => self.status = "Loading model…".into(),
                WorkerState::ModelReady | WorkerState::Recording | WorkerState::Error => {
                    action = Action::Pop(None);
                }
                _ => {}
            },
            _ => {}
        }
        (Some(msg), action)
    }

    fn is_modal(&self) -> bool {
        true
    }
}

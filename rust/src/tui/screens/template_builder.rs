//! The template builder: chat with the AI model on the left, the template preview
//! on the right. As Python's `TemplateBuilderApp`.

use std::sync::LazyLock;
use std::time::{Duration, Instant};

use crossterm::event::KeyEvent;
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use crate::adapters::templates::YamlTemplates;
use crate::app::template_builder::{Role, Setup, TemplateBuilderRuntime};
use crate::domain::template::SessionTemplate;
use crate::tui::host::{
    Action, Ctx, MouseInput, MouseKind, Msg, NOTIFICATION_TIMEOUT, Payload, Screen, Severity,
};
use crate::tui::keys::{Binding, SCREEN_BASE, bind};
use crate::tui::markdown::tint;
use crate::tui::screens::template_picker::{Answer, Confirm};
use crate::tui::text::center;
use crate::tui::widgets::button::Variant;
use crate::tui::widgets::input::Input;
use crate::tui::widgets::scroll::ScrollView;
use crate::tui::widgets::{bar, border_color, centered, rounded, split_bars};

pub const BUILDER_BINDINGS: [Binding; 6] = [
    SCREEN_BASE[0],
    SCREEN_BASE[1],
    SCREEN_BASE[2],
    SCREEN_BASE[3],
    bind("escape", "quit_app", "Back").priority(),
    bind("ctrl+s", "save_template", "Save").priority(),
];

pub const SAVE_BINDINGS: [Binding; 5] = [
    SCREEN_BASE[0],
    SCREEN_BASE[1],
    SCREEN_BASE[2],
    SCREEN_BASE[3],
    bind("escape", "cancel", "Cancel"),
];

pub const CONFIRM_QUIT_BINDINGS: [Binding; 7] = [
    SCREEN_BASE[0],
    SCREEN_BASE[1],
    SCREEN_BASE[2],
    SCREEN_BASE[3],
    bind("y", "confirm", "Yes").hidden(),
    bind("n", "dismiss_no", "No").hidden(),
    bind("escape", "dismiss_no", "Cancel"),
];

const THINKING_STEP: Duration = Duration::from_millis(400);

/// Textual reads the markup color names as CSS colors.
const CYAN: Color = Color::Rgb(0x00, 0xff, 0xff);
const GREEN: Color = Color::Rgb(0x00, 0x80, 0x00);

/// Stands in for `[` while the marks are added, so text from the user or the
/// model never reads as a tag (Rich's `escape`).
const OPEN: char = '\u{e000}';

static MARKS: LazyLock<[(fancy_regex::Regex, &str); 4]> = LazyLock::new(|| {
    let regex = |pattern| fancy_regex::Regex::new(pattern).expect("a valid pattern");
    [
        (regex(r"\*\*(.+?)\*\*"), "[b]$1[/b]"),
        (
            regex(r"(?<![*\w])\*(?!\s)(.+?)(?<!\s)\*(?![*\w])"),
            "[i]$1[/i]",
        ),
        (regex(r"`([^`]+)`"), "[bold]$1[/bold]"),
        (regex(r"(?m)^#{1,6} +(.+)$"), "[b]$1[/b]"),
    ]
});

/// The model's bold, italic, code and headings as tags, as Python's `_markdown_marks`.
fn markdown_marks(text: &str) -> String {
    MARKS
        .iter()
        .fold(text.to_string(), |text, (pattern, replacement)| {
            pattern.replace_all(&text, *replacement).into_owned()
        })
}

/// Lines from text with `[b]`, `[i]` and `[bold]` tags; `base` styles all of it.
fn tagged_lines(text: &str, base: Style) -> Vec<Line<'static>> {
    static TAG: LazyLock<regex::Regex> =
        LazyLock::new(|| regex::Regex::new(r"\[(/?)(b|i|bold)\]").expect("a valid pattern"));
    let mut stack: Vec<Modifier> = Vec::new();
    let mut lines = vec![Vec::new()];
    let style_now = |stack: &[Modifier]| {
        stack
            .iter()
            .fold(base, |style, modifier| style.add_modifier(*modifier))
    };
    let push = |piece: &str, style: Style, lines: &mut Vec<Vec<Span<'static>>>| {
        for (index, part) in piece.split('\n').enumerate() {
            if index > 0 {
                lines.push(Vec::new());
            }
            if !part.is_empty() {
                let shown = part.replace(OPEN, "[");
                lines
                    .last_mut()
                    .expect("a line")
                    .push(Span::styled(shown, style));
            }
        }
    };
    let mut rest = 0;
    for tag in TAG.captures_iter(text) {
        let whole = tag.get(0).expect("the match");
        push(&text[rest..whole.start()], style_now(&stack), &mut lines);
        rest = whole.end();
        let modifier = match &tag[2] {
            "i" => Modifier::ITALIC,
            _ => Modifier::BOLD,
        };
        match &tag[1] {
            "/" => {
                if let Some(at) = stack.iter().rposition(|m| *m == modifier) {
                    stack.remove(at);
                }
            }
            _ => stack.push(modifier),
        }
    }
    push(&text[rest..], style_now(&stack), &mut lines);
    lines.into_iter().map(Line::from).collect()
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Focus {
    Chat,
    Input,
    Preview,
}

pub struct TemplateBuilderScreen {
    runtime: TemplateBuilderRuntime,
    thinking_tick: usize,
    /// "Thinking" has no dots until the first step, as in Python.
    thinking_dots: usize,
    thinking_since: Instant,
    header: String,
    /// The runtime's preview revision on show.
    preview_revision: u64,
    chat_view: ScrollView,
    preview: ScrollView,
    input: Input,
    focus: Focus,
}

impl TemplateBuilderScreen {
    /// Create a template, or edit `base` (saved back as `base_key` by default).
    pub fn new(
        setup: Setup,
        templates: YamlTemplates,
        base: Option<(SessionTemplate, String)>,
    ) -> Self {
        let runtime = TemplateBuilderRuntime::new(setup, templates, base);
        let mut chat_view = ScrollView::log();
        chat_view.always_bar = false;
        chat_view.bar_width = 1;
        chat_view.content_padding = 1;
        let mut preview = ScrollView::markdown();
        preview.padding = 0;
        preview.markdown_padding = 1;
        preview.bar_width = 1;
        preview.set_markdown(&runtime.preview());
        let mut screen = Self {
            header: format!("  {}", runtime.title()),
            input: Input::new(runtime.placeholder()),
            preview_revision: runtime.preview_revision(),
            runtime,
            thinking_tick: 0,
            thinking_dots: 0,
            thinking_since: Instant::now(),
            chat_view,
            preview,
            focus: Focus::Input,
        };
        screen.refresh_chat();
        screen
    }

    fn chat_lines(&self) -> Vec<Line<'static>> {
        let mut lines = Vec::new();
        for (index, (role, text)) in self.runtime.chat().iter().enumerate() {
            if index > 0 {
                lines.push(Line::default());
            }
            let escaped = text.replace('[', &OPEN.to_string());
            let (prefix, body) = match role {
                Role::User => ("[bold]You[/bold]", escaped),
                Role::Ai => ("[bold]AI[/bold]", markdown_marks(&escaped)),
            };
            let color = match role {
                Role::User => CYAN,
                Role::Ai => GREEN,
            };
            let mut entry = tagged_lines(&format!("{prefix}: {body}"), Style::default());
            // The prefix is the first span: color it.
            if let Some(first) = entry.first_mut().and_then(|line| line.spans.first_mut()) {
                first.style = first.style.fg(color);
            }
            lines.extend(entry);
        }
        if self.runtime.busy() {
            if !lines.is_empty() {
                lines.push(Line::default());
            }
            lines.push(Line::styled(
                format!("Thinking{}", ".".repeat(self.thinking_dots)),
                Style::default().add_modifier(Modifier::DIM | Modifier::ITALIC),
            ));
        }
        lines
    }

    fn refresh_chat(&mut self) {
        let lines = self.chat_lines();
        self.chat_view.set_lines(lines);
        self.chat_view.scroll_end();
    }

    /// Redraw the preview only when the runtime changed it, so its scroll stays.
    fn update_preview(&mut self) {
        if self.preview_revision != self.runtime.preview_revision() {
            self.preview_revision = self.runtime.preview_revision();
            self.preview.set_markdown(&self.runtime.preview());
        }
    }

    /// Show the runtime after a send or a reply.
    fn sync(&mut self) {
        self.thinking_tick = 0;
        self.thinking_dots = 0;
        self.thinking_since = Instant::now();
        // A disabled input loses the focus to the chat log; it comes back after.
        self.focus = match self.runtime.busy() {
            true => Focus::Chat,
            false => Focus::Input,
        };
        self.update_preview();
        self.refresh_chat();
    }

    fn save(&mut self, ctx: &mut Ctx) -> Action {
        let Some(suggestion) = self.runtime.save_suggestion() else {
            ctx.warn(
                "No template to save — generate one first.",
                NOTIFICATION_TIMEOUT,
            );
            return Action::None;
        };
        Action::Push(Box::new(SaveTemplate::new(
            &suggestion.file_name,
            &suggestion.display_name,
        )))
    }

    fn saved_as(&mut self, request: SaveRequest, ctx: &mut Ctx) {
        let saved = self.runtime.save(&request.file_name, &request.display_name);
        self.update_preview();
        match saved {
            None => {}
            Some(Ok((safe_name, path))) => {
                ctx.notify(format!("Saved as {safe_name} → {}", path.display()));
            }
            Some(Err(e)) => ctx.notify_with(
                format!("Cannot save template: {e}"),
                Severity::Error,
                NOTIFICATION_TIMEOUT,
                "",
            ),
        }
    }

    fn quit(&mut self) -> Action {
        if self.runtime.unsaved() {
            let message = Line::from("You have an unsaved template. Quit anyway?");
            let mut confirm = Confirm::new(message, "warning", Variant::Warning);
            confirm.arrows = false;
            return Action::Push(Box::new(confirm));
        }
        self.leave()
    }

    fn leave(&mut self) -> Action {
        self.runtime.leave();
        Action::Pop(None)
    }

    fn focus_order(&self) -> Vec<Focus> {
        match self.runtime.busy() {
            true => vec![Focus::Chat, Focus::Preview],
            false => vec![Focus::Chat, Focus::Input, Focus::Preview],
        }
    }

    fn move_focus(&mut self, step: isize) {
        let order = self.focus_order();
        let at = order.iter().position(|f| *f == self.focus).unwrap_or(0) as isize;
        let next = (at + step).rem_euclid(order.len() as isize) as usize;
        self.focus = order[next];
    }
}

/// `opacity`: blend each cell of `area` toward `under`, as Textual draws `:disabled`.
fn fade(frame: &mut Frame, area: Rect, under: Color, alpha: f64) {
    let buffer = frame.buffer_mut();
    for y in area.top()..area.bottom() {
        for x in area.left()..area.right() {
            let cell = &mut buffer[(x, y)];
            cell.fg = tint(cell.fg, under, alpha);
            cell.bg = tint(cell.bg, under, alpha);
        }
    }
}

impl Screen for TemplateBuilderScreen {
    fn render(&mut self, frame: &mut Frame, area: Rect, ctx: &Ctx) {
        let theme = &ctx.theme;
        let (header, body, footer) = split_bars(area, 1, 1);
        bar(frame, header, &self.header, "primary", "text", true, theme);
        bar(
            frame,
            footer,
            "[Enter] Send  [Ctrl+S] Save  [Esc] Back",
            "surface",
            "text-muted",
            false,
            theme,
        );
        let background = theme.color("background");
        let text = Style::default()
            .fg(theme.color("foreground"))
            .bg(background);
        let left_width = body.width / 2;
        let left = Rect::new(body.x, body.y, left_width, body.height);
        let right = Rect::new(
            body.x + left_width,
            body.y,
            body.width - left_width,
            body.height,
        );
        let input_height = 3.min(left.height);
        let chat_area = Rect::new(left.x, left.y, left.width, left.height - input_height);
        let input_area = Rect::new(
            left.x,
            left.bottom() - input_height,
            left.width,
            input_height,
        );
        let chat_border = border_color(self.focus == Focus::Chat, theme);
        let chat_inner = rounded(frame, chat_area, "", chat_border, background, theme);
        self.chat_view.render(frame, chat_inner, text, theme);
        ctx.hit(chat_area, "chat", 0);
        self.input
            .render(frame, input_area, self.focus == Focus::Input, theme);
        if self.runtime.busy() {
            fade(frame, input_area, background, 0.7);
        } else {
            ctx.hit(input_area, "input", 0);
        }
        let preview_border = border_color(self.focus == Focus::Preview, theme);
        let preview_inner = rounded(frame, right, "", preview_border, background, theme);
        self.preview.render(frame, preview_inner, text, theme);
        ctx.hit(right, "preview", 0);
    }

    /// A click focuses the chat, the input, or the preview; the wheel scrolls
    /// the chat or the preview under the pointer.
    fn mouse(&mut self, input: &MouseInput, _ctx: &mut Ctx) -> Action {
        let step = match input.kind {
            MouseKind::WheelUp => -1,
            MouseKind::WheelDown => 1,
            MouseKind::Click => {
                if input.on("chat").is_some() {
                    self.focus = Focus::Chat;
                } else if input.on("input").is_some() {
                    self.focus = Focus::Input;
                } else if input.on("preview").is_some() {
                    self.focus = Focus::Preview;
                }
                return Action::None;
            }
        };
        if input.on("chat").is_some() {
            self.chat_view.scroll_by(step);
        } else if input.on("preview").is_some() {
            self.preview.scroll_by(step);
        }
        Action::None
    }

    fn key(&mut self, key: &KeyEvent, name: &str, ctx: &mut Ctx) -> Action {
        match name {
            "escape" => return self.quit(),
            "ctrl+s" => return self.save(ctx),
            "tab" => self.move_focus(1),
            "shift+tab" => self.move_focus(-1),
            _ => match self.focus {
                Focus::Input => {
                    if name == "enter" {
                        let text = self.input.value().trim().to_string();
                        if !text.is_empty() {
                            self.input.set_value("");
                            self.runtime.send(text, &ctx.tasks());
                            self.sync();
                        }
                    } else {
                        self.input.key(key, name);
                    }
                }
                Focus::Chat => {
                    self.chat_view.key(name);
                }
                Focus::Preview => {
                    self.preview.key(name);
                }
            },
        }
        Action::None
    }

    fn paste(&mut self, text: &str, _ctx: &mut Ctx) -> Action {
        if self.focus == Focus::Input {
            self.input.paste(text);
        }
        Action::None
    }

    fn message(&mut self, msg: Msg, _ctx: &mut Ctx) -> (Option<Msg>, Action) {
        match msg {
            Msg::Task(payload) => match self.runtime.task_result(payload) {
                Ok(()) => {
                    self.sync();
                    (None, Action::None)
                }
                Err(payload) => (Some(Msg::Task(payload)), Action::None),
            },
            other => (Some(other), Action::None),
        }
    }

    fn tick(&mut self, _ctx: &mut Ctx) -> Action {
        if self.runtime.busy() && self.thinking_since.elapsed() >= THINKING_STEP {
            // Thinking. → Thinking.. → Thinking...
            self.thinking_since = Instant::now();
            self.thinking_tick = (self.thinking_tick + 1) % 3;
            self.thinking_dots = self.thinking_tick + 1;
            self.refresh_chat();
        }
        Action::None
    }

    fn resumed(&mut self, result: Option<Payload>, ctx: &mut Ctx) -> Action {
        let Some(payload) = result else {
            return Action::None;
        };
        let payload = match payload.downcast::<Answer>() {
            Ok(answer) => {
                return match answer.0 {
                    true => self.leave(),
                    false => Action::None,
                };
            }
            Err(payload) => payload,
        };
        if let Ok(request) = payload.downcast::<SaveRequest>() {
            self.saved_as(*request, ctx);
        }
        Action::None
    }
}

/// What the save dialog returns: the file name, and the name shown in the picker.
pub struct SaveRequest {
    pub file_name: String,
    pub display_name: String,
}

/// "Save Template": the file name and the display name.
pub struct SaveTemplate {
    file_name: Input,
    display_name: Input,
    on_display: bool,
}

impl SaveTemplate {
    pub fn new(suggested: &str, display_name: &str) -> Self {
        let mut file_name = Input::new("template_name (no spaces)");
        file_name.parent_background = Some("surface");
        file_name.set_value(suggested);
        file_name.select_all();
        let mut display = Input::new("Name shown in the picker");
        display.parent_background = Some("surface");
        display.set_value(display_name);
        Self {
            file_name,
            display_name: display,
            on_display: false,
        }
    }
}

impl Screen for SaveTemplate {
    fn render(&mut self, frame: &mut Frame, area: Rect, ctx: &Ctx) {
        let theme = &ctx.theme;
        // A rounded box titled "Save Template": the two titled inputs, a blank row, the hint.
        let outer = centered(area, 60, 2 + 3 + 3 + 1 + 1);
        frame.render_widget(ratatui::widgets::Clear, outer);
        let inner = rounded(
            frame,
            outer,
            "Save Template",
            theme.color("primary"),
            theme.color("surface"),
            theme,
        );
        if inner.height < 8 || inner.width < 10 {
            return;
        }
        let padded = Rect::new(inner.x + 1, inner.y, inner.width - 2, inner.height);
        let muted = Style::default()
            .fg(theme.on("text-muted", "surface"))
            .bg(theme.color("surface"));
        let file_area = Rect::new(padded.x, padded.y, padded.width, 3);
        self.file_name
            .render_titled(frame, file_area, "File name", !self.on_display, theme);
        ctx.hit(file_area, "file_name", 0);
        let display_area = Rect::new(padded.x, padded.y + 3, padded.width, 3);
        self.display_name.render_titled(
            frame,
            display_area,
            "Display name",
            self.on_display,
            theme,
        );
        ctx.hit(display_area, "display_name", 0);
        frame.render_widget(
            Paragraph::new(center(
                Line::from("Enter to save, Esc to cancel"),
                padded.width,
            ))
            .style(muted),
            Rect::new(padded.x, padded.y + 7, padded.width, 1),
        );
    }

    /// A click on an input moves the focus there, as Tab does.
    fn mouse(&mut self, input: &MouseInput, _ctx: &mut Ctx) -> Action {
        if input.kind != MouseKind::Click {
            return Action::None;
        }
        let on_display = match (input.on("file_name"), input.on("display_name")) {
            (Some(_), _) => false,
            (_, Some(_)) => true,
            _ => return Action::None,
        };
        if on_display != self.on_display {
            self.on_display = on_display;
            match on_display {
                true => self.display_name.select_all(),
                false => self.file_name.select_all(),
            }
        }
        Action::None
    }

    fn key(&mut self, key: &KeyEvent, name: &str, _ctx: &mut Ctx) -> Action {
        match name {
            "escape" => Action::Pop(None),
            "tab" | "shift+tab" => {
                self.on_display = !self.on_display;
                match self.on_display {
                    true => self.display_name.select_all(),
                    false => self.file_name.select_all(),
                }
                Action::None
            }
            "enter" => {
                let file_name = self.file_name.value().trim().to_string();
                if file_name.is_empty() {
                    return Action::None;
                }
                let display_name = self.display_name.value().trim().to_string();
                Action::Pop(Some(Box::new(SaveRequest {
                    file_name,
                    display_name,
                })))
            }
            _ => {
                match self.on_display {
                    true => self.display_name.key(key, name),
                    false => self.file_name.key(key, name),
                };
                Action::None
            }
        }
    }

    fn paste(&mut self, text: &str, _ctx: &mut Ctx) -> Action {
        match self.on_display {
            true => self.display_name.paste(text),
            false => self.file_name.paste(text),
        }
        Action::None
    }

    fn is_modal(&self) -> bool {
        true
    }
}

#[cfg(test)]
#[path = "template_builder_tests.rs"]
mod tests;

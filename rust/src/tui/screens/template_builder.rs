//! The template builder: chat with the AI model on the left, the template preview
//! on the right. As Python's `TemplateBuilderApp`.

use std::sync::{Arc, LazyLock};
use std::time::{Duration, Instant};

use crossterm::event::KeyEvent;
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use tracing::error;

use crate::adapters::templates::YamlTemplates;
use crate::domain::chat::ChatMessage;
use crate::domain::template::SessionTemplate;
use crate::ports::llm::{LlmClient, LlmError};
use crate::tui::host::{Action, Ctx, Msg, NOTIFICATION_TIMEOUT, Payload, Screen, Severity};
use crate::tui::keys::{Binding, SCREEN_BASE, bind};
use crate::tui::markdown::tint;
use crate::tui::screens::template_picker::{Answer, Confirm};
use crate::tui::text::center;
use crate::tui::widgets::button::Variant;
use crate::tui::widgets::input::Input;
use crate::tui::widgets::scroll::ScrollView;
use crate::tui::widgets::{bar, centered, split_bars, tall_edges, thick_box};
use crate::usecases::template_builder::{BuilderResult, TemplateBuilder};
use crate::usecases::template_diff::diff_templates;

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

const MAX_AUTO_FIX: usize = 2;
/// A full template takes a slow model a few minutes; past this the call is stuck.
pub const LLM_TIMEOUT: Duration = Duration::from_secs(300);
const THINKING_STEP: Duration = Duration::from_millis(400);

const WELCOME: &str = "Describe the template you want to create. For example:
- \"1-on-1 template, English, track action items and career goals\"
- \"Lecture notes in Chinese, focus on key concepts and exam topics\"

I'll guide you if I need more details, or generate right away if your description is complete enough.
";

const EDIT_WELCOME: &str = "Describe the change you want. For example:
- \"Add a quick action that lists risks\"
- \"Make the summary shorter and write it in Traditional Chinese\"

I'll restate the change and wait for your yes before I edit the template. The preview shows the current version.
";

const NO_TEMPLATE: &str = "*No template yet. Describe what you want.*";

/// Textual reads the markup color names as CSS colors.
const CYAN: Color = Color::Rgb(0x00, 0xff, 0xff);
const GREEN: Color = Color::Rgb(0x00, 0x80, 0x00);

/// What the builder needs from the config, built on the first send as in Python.
pub struct BuilderSetup {
    pub llm: Arc<dyn LlmClient>,
    pub model: String,
    pub provider: String,
    pub example: SessionTemplate,
}

pub type Setup = Box<dyn Fn() -> Result<BuilderSetup, String>>;

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

/// `_slugify`: a safe file name.
pub fn slugify(text: &str) -> String {
    static NOT_WORD: LazyLock<regex::Regex> =
        LazyLock::new(|| regex::Regex::new(r"[^\w]").expect("a valid pattern"));
    static UNDERSCORES: LazyLock<regex::Regex> =
        LazyLock::new(|| regex::Regex::new("_+").expect("a valid pattern"));
    let slug = NOT_WORD
        .replace_all(&text.trim().to_lowercase(), "_")
        .into_owned();
    let slug = UNDERSCORES.replace_all(&slug, "_");
    let slug = slug.trim_matches('_');
    match slug.is_empty() {
        true => "custom_template".into(),
        false => slug.into(),
    }
}

/// The preview pane's Markdown, as Python's `_update_preview`.
pub fn preview_markdown(template: &SessionTemplate) -> String {
    let meta = &template.metadata;
    let name = if meta.name.is_empty() {
        "Untitled"
    } else {
        &meta.name
    };
    let locale = if meta.locale.is_empty() {
        "not set"
    } else {
        &meta.locale
    };
    let mut lines = vec![
        format!("## {name}"),
        String::new(),
        match meta.description.is_empty() {
            true => String::new(),
            false => format!("> {}", meta.description),
        },
        String::new(),
        format!("**Locale:** `{locale}`"),
    ];
    if !template.quick_actions.is_empty() {
        lines.extend([String::new(), "### Quick Actions".into()]);
        for (index, action) in template.quick_actions.iter().enumerate() {
            let description = match action.description.is_empty() {
                true => String::new(),
                false => format!(" — {}", action.description),
            };
            lines.push(format!(
                "- **`{}`** {}{description}",
                index + 1,
                action.label
            ));
        }
    }
    if !template.recognition_hints.is_empty() {
        lines.extend([
            String::new(),
            format!(
                "**Recognition hints:** {}",
                template.recognition_hints.join(", ")
            ),
        ]);
    }
    for (title, body) in [
        ("System Prompt", &template.system_prompt),
        ("Digest Template", &template.digest_user_template),
    ] {
        lines.extend([
            String::new(),
            "---".into(),
            String::new(),
            format!("### {title}"),
            String::new(),
            body.clone(),
        ]);
    }
    lines.join("\n")
}

/// The answer to an LLM error, as Python's `_error_text`.
fn error_text(error: &LlmError, model: &str, provider: &str) -> String {
    let not_found = error.kind == "NotFoundError" || error.message.ends_with("(status code: 404)");
    if not_found {
        let fix = match provider {
            "ollama" => format!("Run: ollama pull {model}"),
            _ => "Pick an available model in Settings > Models.".into(),
        };
        return format!("The AI model {model} is not available. {fix}");
    }
    format!("Error: {}: {}", error.kind, error.message)
}

enum Turn {
    Done(Box<BuilderResult>),
    Failed(LlmError),
    TimedOut,
}

/// One send: the reply, plus the builder and history handed back.
struct TurnDone {
    builder: TemplateBuilder,
    history: Vec<ChatMessage>,
    turn: Turn,
}

async fn run_turn(
    mut builder: TemplateBuilder,
    mut history: Vec<ChatMessage>,
    llm: Arc<dyn LlmClient>,
    model: String,
    text: String,
) -> TurnDone {
    let turn = async {
        let first = builder.generate(llm.as_ref(), &text, &model, &mut history);
        let mut result = match tokio::time::timeout(LLM_TIMEOUT, first).await {
            Err(_) => return Turn::TimedOut,
            Ok(Err(e)) => return Turn::Failed(e),
            Ok(Ok(result)) => result,
        };
        if !result.error.is_empty() || result.validation_errors.is_empty() {
            return Turn::Done(Box::new(result));
        }
        for _ in 0..MAX_AUTO_FIX {
            let errors = result.validation_errors.clone();
            let fix = builder.auto_fix(llm.as_ref(), &errors, &model, &mut history);
            result = match tokio::time::timeout(LLM_TIMEOUT, fix).await {
                Err(_) => return Turn::TimedOut,
                Ok(Err(e)) => return Turn::Failed(e),
                Ok(Ok(result)) => result,
            };
            if result.validation_errors.is_empty() && result.error.is_empty() {
                break;
            }
        }
        Turn::Done(Box::new(result))
    }
    .await;
    TurnDone {
        builder,
        history,
        turn,
    }
}

enum Role {
    User,
    Ai,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Focus {
    Chat,
    Input,
    Preview,
}

/// The model and client, once the config is read.
struct Ready {
    llm: Arc<dyn LlmClient>,
    model: String,
    provider: String,
}

pub struct TemplateBuilderScreen {
    setup: Setup,
    ready: Option<Ready>,
    /// None while a call holds it.
    builder: Option<TemplateBuilder>,
    history: Vec<ChatMessage>,
    base_template: Option<SessionTemplate>,
    base_key: String,
    current: Option<SessionTemplate>,
    saved: bool,
    busy: bool,
    chat: Vec<(Role, String)>,
    thinking_tick: usize,
    /// "Thinking" has no dots until the first step, as in Python.
    thinking_dots: usize,
    thinking_since: Instant,
    task: Option<tokio::task::AbortHandle>,
    templates: YamlTemplates,
    header: String,
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
        let (base_template, base_key) = match base {
            Some((template, key)) => (Some(template), key),
            None => (None, String::new()),
        };
        let header = match &base_template {
            None => "  Template Builder".to_string(),
            Some(template) => {
                let name = match template.metadata.name.is_empty() {
                    true => &base_key,
                    false => &template.metadata.name,
                };
                format!("  Template Builder: Editing: {name}")
            }
        };
        let placeholder = match base_template {
            None => "Describe your template...",
            Some(_) => "Describe the change...",
        };
        let mut chat_view = ScrollView::log();
        chat_view.always_bar = false;
        chat_view.bar_width = 2;
        chat_view.content_padding = 1;
        let mut preview = ScrollView::markdown();
        preview.padding = 0;
        preview.markdown_padding = 1;
        preview.bar_width = 2;
        let mut screen = Self {
            setup,
            ready: None,
            builder: None,
            history: Vec::new(),
            saved: base_template.is_some(),
            current: base_template.clone(),
            base_template,
            base_key,
            busy: false,
            chat: Vec::new(),
            thinking_tick: 0,
            thinking_dots: 0,
            thinking_since: Instant::now(),
            task: None,
            templates,
            header,
            chat_view,
            preview,
            input: Input::new(placeholder),
            focus: Focus::Input,
        };
        match screen.base_template.is_some() {
            false => {
                screen.preview.set_markdown(NO_TEMPLATE);
                screen.append(Role::Ai, WELCOME);
            }
            true => {
                screen.append(Role::Ai, EDIT_WELCOME);
                screen.update_preview();
            }
        }
        screen
    }

    fn append(&mut self, role: Role, text: &str) {
        self.chat.push((role, text.to_string()));
        self.refresh_chat();
    }

    fn chat_lines(&self) -> Vec<Line<'static>> {
        let mut lines = Vec::new();
        for (index, (role, text)) in self.chat.iter().enumerate() {
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
        if self.busy {
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

    fn update_preview(&mut self) {
        if let Some(template) = &self.current {
            self.preview.set_markdown(&preview_markdown(template));
        }
    }

    fn set_busy(&mut self, busy: bool) {
        self.busy = busy;
        self.thinking_tick = 0;
        self.thinking_dots = 0;
        self.thinking_since = Instant::now();
        // A disabled input loses the focus to the chat log; it comes back after.
        self.focus = if busy { Focus::Chat } else { Focus::Input };
        self.refresh_chat();
    }

    fn send(&mut self, text: String, ctx: &mut Ctx) {
        if self.busy {
            return;
        }
        self.append(Role::User, &text);
        self.set_busy(true);
        if self.ready.is_none() {
            match (self.setup)() {
                Ok(setup) => {
                    self.builder = Some(TemplateBuilder::new(
                        &setup.example,
                        self.base_template.clone(),
                    ));
                    self.ready = Some(Ready {
                        llm: setup.llm,
                        model: setup.model,
                        provider: setup.provider,
                    });
                }
                Err(e) => {
                    // A bad config must leave the builder usable for a retry.
                    error!(target: "ltn.template_builder", "Template generation failed: {e}");
                    self.append(Role::Ai, &format!("Error: {e}"));
                    self.set_busy(false);
                    return;
                }
            }
        }
        let ready = self.ready.as_ref().expect("set above");
        let builder = self.builder.take().expect("no call is running");
        let history = std::mem::take(&mut self.history);
        self.task = Some(ctx.spawn(run_turn(
            builder,
            history,
            ready.llm.clone(),
            ready.model.clone(),
            text,
        )));
    }

    fn turn_done(&mut self, done: TurnDone) {
        self.task = None;
        self.builder = Some(done.builder);
        self.history = done.history;
        let (model, provider) = self
            .ready
            .as_ref()
            .map(|ready| (ready.model.clone(), ready.provider.clone()))
            .unwrap_or_default();
        match done.turn {
            Turn::TimedOut => {
                error!(
                    target: "ltn.template_builder",
                    "Template builder LLM call got no answer within {}s",
                    LLM_TIMEOUT.as_secs()
                );
                let minutes = LLM_TIMEOUT.as_secs_f64() / 60.0;
                self.append(
                    Role::Ai,
                    &format!(
                        "Error: the AI model did not answer within {} minutes. Send it again.",
                        minutes.round()
                    ),
                );
            }
            Turn::Failed(e) => {
                error!(target: "ltn.template_builder", "Template generation failed: {}: {}", e.kind, e.message);
                self.append(Role::Ai, &error_text(&e, &model, &provider));
            }
            Turn::Done(result) => self.show_result(*result),
        }
        self.set_busy(false);
    }

    fn show_result(&mut self, result: BuilderResult) {
        if !result.error.is_empty() {
            let text = format!(
                "{}\n\n*(Error: {})*",
                result.assistant_message, result.error
            );
            self.append(Role::Ai, &text);
            return;
        }
        let Some(template) = result.template else {
            self.append(Role::Ai, &result.assistant_message);
            return;
        };
        let previous = self.current.replace(template.clone());
        self.saved = false;
        self.update_preview();
        let message = match result.assistant_message.is_empty() {
            true => "Template generated.",
            false => &result.assistant_message,
        };
        self.append(Role::Ai, message);
        if let (Some(_), Some(previous)) = (&self.base_template, previous) {
            let changes = diff_templates(&previous, &template);
            let text = match changes.is_empty() {
                true => "No field changed.".to_string(),
                false => format!("Changed: {}", changes.join(", ")),
            };
            self.append(Role::Ai, &text);
        }
    }

    fn save(&mut self, ctx: &mut Ctx) -> Action {
        let Some(current) = &self.current else {
            ctx.warn(
                "No template to save — generate one first.",
                NOTIFICATION_TIMEOUT,
            );
            return Action::None;
        };
        let suggested = if !self.base_key.is_empty() {
            self.base_key.clone()
        } else if !current.metadata.key.is_empty() {
            current.metadata.key.clone()
        } else {
            slugify(&current.metadata.name)
        };
        Action::Push(Box::new(SaveTemplate::new(
            &suggested,
            &current.metadata.name,
        )))
    }

    fn saved_as(&mut self, request: SaveRequest, ctx: &mut Ctx) {
        let Some(current) = &mut self.current else {
            return;
        };
        if !request.display_name.is_empty() && request.display_name != current.metadata.name {
            current.metadata.name = request.display_name;
            self.update_preview();
        }
        let Some(current) = &self.current else {
            return;
        };
        let safe_name = slugify(&request.file_name);
        match self.templates.save_user_template(current, &safe_name) {
            Ok(path) => {
                self.saved = true;
                ctx.notify(format!("Saved as {safe_name} → {}", path.display()));
            }
            Err(e) => {
                error!(target: "ltn.template_builder", "Cannot save template {safe_name}: {e}");
                ctx.notify_with(
                    format!("Cannot save template: {e}"),
                    Severity::Error,
                    NOTIFICATION_TIMEOUT,
                    "",
                );
            }
        }
    }

    fn quit(&mut self) -> Action {
        if self.current.is_some() && !self.saved {
            let message = Line::from("You have an unsaved template. Quit anyway?");
            let mut confirm = Confirm::new(message, "warning", Variant::Warning);
            confirm.arrows = false;
            return Action::Push(Box::new(confirm));
        }
        self.leave()
    }

    /// Close the builder; a running call ends with it, as Textual cancels the worker.
    fn leave(&mut self) -> Action {
        if let Some(task) = self.task.take() {
            task.abort();
        }
        Action::Pop(None)
    }

    fn focus_order(&self) -> Vec<Focus> {
        match self.busy {
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

/// A `border: tall` box around a scroll container. Returns the inside.
fn tall_box(frame: &mut Frame, area: Rect, border: Color, inner: Color, outer: Color) -> Rect {
    if area.width < 2 || area.height < 2 {
        return Rect::default();
    }
    let edges = tall_edges(border, inner, outer);
    let width = usize::from(area.width - 2);
    let mut lines = Vec::new();
    for row in 0..area.height {
        let (fill, style) = match row {
            0 => ("▔", edges.fill),
            r if r == area.height - 1 => ("▁", edges.fill),
            _ => (" ", Style::default().bg(inner)),
        };
        lines.push(Line::from(vec![
            Span::styled("▊", edges.left),
            Span::styled(fill.repeat(width), style),
            Span::styled("▎", edges.right),
        ]));
    }
    frame.render_widget(Paragraph::new(lines), area);
    Rect::new(area.x + 1, area.y + 1, area.width - 2, area.height - 2)
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
        let border = theme.color("surface-lighten-2");
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
        let chat_inner = tall_box(frame, chat_area, border, background, background);
        self.chat_view.render(frame, chat_inner, text, theme);
        self.input
            .render(frame, input_area, self.focus == Focus::Input, theme);
        if self.busy {
            fade(frame, input_area, background, 0.7);
        }
        let preview_inner = tall_box(frame, right, border, background, background);
        self.preview.render(frame, preview_inner, text, theme);
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
                            self.send(text, ctx);
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
            Msg::Task(payload) => match payload.downcast::<TurnDone>() {
                Ok(done) => {
                    self.turn_done(*done);
                    (None, Action::None)
                }
                Err(payload) => (Some(Msg::Task(payload)), Action::None),
            },
            other => (Some(other), Action::None),
        }
    }

    fn tick(&mut self, _ctx: &mut Ctx) -> Action {
        if self.busy && self.thinking_since.elapsed() >= THINKING_STEP {
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
        // Border, padding, then: title + margin, label, input + margin, label, input + margin, hint.
        let outer = centered(area, 60, 2 + 2 + 2 + 1 + 4 + 1 + 4 + 1);
        let inner = thick_box(frame, outer, "primary", "surface", theme);
        if inner.height < 15 || inner.width < 10 {
            return;
        }
        let padded = Rect::new(inner.x + 2, inner.y + 1, inner.width - 4, inner.height - 2);
        let text = Style::default()
            .fg(theme.on("foreground", "surface"))
            .bg(theme.color("surface"));
        let muted = text.fg(theme.on("text-muted", "surface"));
        let row = |y: u16| Rect::new(padded.x, padded.y + y, padded.width, 1);
        frame.render_widget(
            Paragraph::new(center(Line::from("Save Template"), padded.width)).style(text),
            row(0),
        );
        frame.render_widget(Paragraph::new("File name").style(muted), row(2));
        self.file_name.render(
            frame,
            Rect::new(padded.x, padded.y + 3, padded.width, 3),
            !self.on_display,
            theme,
        );
        frame.render_widget(Paragraph::new("Display name").style(muted), row(7));
        self.display_name.render(
            frame,
            Rect::new(padded.x, padded.y + 8, padded.width, 3),
            self.on_display,
            theme,
        );
        frame.render_widget(
            Paragraph::new(center(
                Line::from("Enter to save, Esc to cancel"),
                padded.width,
            ))
            .style(muted),
            row(12),
        );
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

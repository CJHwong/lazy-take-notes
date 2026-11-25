//! The checks before a recording starts (Python's `PreflightScreen`).

use crate::app::preflight::{Check, PreflightPlan, Status};
use crate::app::preflight_runtime::{PreflightRuntime, Summary};
use crate::ports::transcriber::Transcriber;
use crate::tui::host::{Action, Ctx, MouseInput, MouseKind, Msg, Screen};
use crate::tui::keys::{Binding, SCREEN_BASE, bind};
use crate::tui::text::wrap_line;
use crate::tui::widgets::button::{Variant, button_width, draw_button, draw_disabled_button};
use crate::tui::widgets::scroll::draw_scrollbar;
use crossterm::event::KeyEvent;
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

pub const PREFLIGHT_BINDINGS: [Binding; 6] = [
    SCREEN_BASE[0],
    SCREEN_BASE[1],
    SCREEN_BASE[2],
    SCREEN_BASE[3],
    bind("r", "retry", "Retry"),
    bind("escape", "back", "Back"),
];

/// Textual reads the markup color names as CSS colors.
/// The heading, shown in the border.
const TITLE: &str = "Checking everything before the meeting starts";

const GREEN: Color = Color::Rgb(0x00, 0x80, 0x00);
const YELLOW: Color = Color::Rgb(0xff, 0xff, 0x00);
const RED: Color = Color::Rgb(0xff, 0x00, 0x00);
const CYAN: Color = Color::Rgb(0x00, 0xff, 0xff);

pub use crate::app::preflight::{PreflightOutcome, Reload};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Focus {
    Box,
    Retry,
    Override,
    Back,
}

pub struct PreflightScreen {
    runtime: PreflightRuntime,
    focus: Focus,
    offset: usize,
    /// Rows past the window at the last draw, for the scroll keys.
    max_offset: usize,
    page: usize,
}

fn pop(outcome: PreflightOutcome) -> Action {
    Action::Pop(Some(Box::new(outcome) as crate::tui::host::Payload))
}

impl PreflightScreen {
    pub fn new(
        plan: PreflightPlan,
        transcriber: Box<dyn Transcriber>,
        reload: Option<Reload>,
    ) -> Self {
        Self {
            runtime: PreflightRuntime::new(plan, transcriber, reload),
            focus: Focus::Box,
            offset: 0,
            max_offset: 0,
            page: 1,
        }
    }

    /// The buttons that go away while the checks run lose the focus.
    fn after_start(&mut self) {
        if self.runtime.running && matches!(self.focus, Focus::Retry | Focus::Override) {
            self.focus = Focus::Box;
        }
    }

    fn retry(&mut self, ctx: &mut Ctx) {
        self.runtime.retry(&ctx.tasks());
        self.after_start();
    }

    fn back(&mut self) -> Action {
        match self.runtime.back() {
            true => Action::Pop(None),
            false => Action::None,
        }
    }

    fn start_anyway(&mut self) -> Action {
        self.runtime.start_anyway().map_or(Action::None, pop)
    }

    /// The focus order Tab follows: the box, then each button that can take the focus.
    fn focusable(&self) -> Vec<Focus> {
        let mut order = vec![Focus::Box];
        if !self.runtime.running {
            order.push(Focus::Retry);
        }
        if self.runtime.show_override {
            order.push(Focus::Override);
        }
        order.push(Focus::Back);
        order
    }

    fn move_focus(&mut self, forward: bool) {
        let order = self.focusable();
        let at = order
            .iter()
            .position(|focus| *focus == self.focus)
            .unwrap_or(0);
        let next = match forward {
            true => (at + 1) % order.len(),
            false => (at + order.len() - 1) % order.len(),
        };
        self.focus = order[next];
    }

    fn press(&mut self, ctx: &mut Ctx) -> Action {
        match self.focus {
            Focus::Box => Action::None,
            Focus::Retry => {
                self.retry(ctx);
                Action::None
            }
            Focus::Override => self.start_anyway(),
            Focus::Back => self.back(),
        }
    }

    fn scroll_key(&mut self, name: &str) -> bool {
        let page = self.page.max(1);
        self.offset = match name {
            "up" => self.offset.saturating_sub(1),
            "down" => self.offset + 1,
            "pageup" => self.offset.saturating_sub(page),
            "pagedown" => self.offset + page,
            "home" => 0,
            "end" => self.max_offset,
            _ => return false,
        }
        .min(self.max_offset);
        true
    }

    /// `✓ Title  detail`, then `    Fix: …` for a failure or warning with a fix.
    fn check_lines(check: &Check, text: Style) -> Vec<Line<'static>> {
        let (mark, color) = match check.status {
            Status::Pass => ("✓", Some(GREEN)),
            Status::Warn => ("!", Some(YELLOW)),
            Status::Fail => ("✗", Some(RED)),
            Status::Skip => ("-", None),
            Status::Running => ("…", Some(CYAN)),
            Status::Pending => ("·", None),
        };
        let mark_style = match color {
            Some(color) => text.fg(color),
            None => text.add_modifier(Modifier::DIM),
        };
        let mut spans = vec![
            Span::styled(mark.to_string(), mark_style),
            Span::styled(" ", text),
            Span::styled(check.title.to_string(), text.add_modifier(Modifier::BOLD)),
        ];
        if !check.detail.is_empty() {
            spans.push(Span::styled(format!("  {}", check.detail), text));
        }
        let mut lines = vec![Line::from(spans)];
        if matches!(check.status, Status::Fail | Status::Warn) && !check.fix.is_empty() {
            lines.push(Line::from(vec![
                Span::styled("    ", text),
                Span::styled("Fix:", text.add_modifier(Modifier::DIM)),
                Span::styled(format!(" {}", check.fix), text),
            ]));
        }
        lines
    }

    /// Every row the box scrolls through, wrapped to `width`.
    fn content(&self, width: u16, text: Style) -> Vec<Line<'static>> {
        // The heading sits in the border (`TITLE`).
        let mut lines = Vec::new();
        for check in &self.runtime.checks {
            lines.extend(Self::check_lines(check, text));
        }
        // `#preflight-summary { margin-top: 1 }`. An empty `Static` is still one row high.
        lines.push(Line::default());
        match &self.runtime.summary {
            Summary::None => lines.push(Line::default()),
            Summary::Red(message) => lines.push(Line::styled(message.clone(), text.fg(RED))),
            Summary::Yellow(message) => lines.push(Line::styled(message.clone(), text.fg(YELLOW))),
        }
        lines
            .iter()
            .flat_map(|line| {
                let wrapped = wrap_line(line, width);
                if wrapped.is_empty() {
                    vec![Line::default()]
                } else {
                    wrapped
                }
            })
            .collect()
    }

    /// The buttons that show, with their labels as Textual draws them: the
    /// `[r]` and `[Esc]` in Python's labels read as markup tags, and the space before them stays.
    fn focus_of(index: usize) -> Focus {
        match index {
            1 => Focus::Retry,
            2 => Focus::Override,
            3 => Focus::Back,
            _ => Focus::Box,
        }
    }

    fn index_of(focus: Focus) -> usize {
        match focus {
            Focus::Box => 0,
            Focus::Retry => 1,
            Focus::Override => 2,
            Focus::Back => 3,
        }
    }

    fn buttons(&self) -> Vec<(Focus, &'static str, Variant, bool)> {
        let mut buttons = vec![(
            Focus::Retry,
            "Retry ",
            Variant::Primary,
            !self.runtime.running,
        )];
        if self.runtime.show_override {
            buttons.push((Focus::Override, "Start anyway", Variant::Warning, true));
        }
        buttons.push((Focus::Back, "Back ", Variant::Default, true));
        buttons
    }
}

impl Screen for PreflightScreen {
    fn mounted(&mut self, ctx: &mut Ctx) -> Action {
        self.runtime.begin(&ctx.tasks());
        self.after_start();
        Action::None
    }

    fn render(&mut self, frame: &mut Frame, area: Rect, ctx: &Ctx) {
        let theme = &ctx.theme;
        let text = Style::default()
            .fg(theme.color("foreground"))
            .bg(theme.color("background"));
        // `width: 90%; max-width: 110`, a rounded border, `padding: 0 2`.
        let width = (u32::from(area.width) * 90 / 100).min(110) as u16;
        if width < 8 || area.height < 4 {
            return;
        }
        let inner_width = width - 2 - 4;
        let mut lines = self.content(inner_width, text);
        // A blank row, then the one-row buttons.
        let wanted = lines.len() + 2;
        let max_height = (u32::from(area.height) * 90 / 100) as usize;
        let mut height = (wanted + 2).min(max_height);
        let mut window = height.saturating_sub(2);
        let overflow = wanted > window;
        if overflow {
            // The scrollbar takes a column; wrap again in what is left.
            lines = self.content(inner_width.saturating_sub(1), text);
            height = (lines.len() + 2 + 2).min(max_height);
            window = height.saturating_sub(2);
        }
        let total = lines.len() + 2;
        self.page = window;
        self.max_offset = total.saturating_sub(window);
        self.offset = self.offset.min(self.max_offset);
        let x = area.x + (area.width - width) / 2;
        let y = area.y + (area.height - height as u16) / 2;
        let outer = Rect::new(x, y, width, height as u16);
        let body = crate::tui::widgets::rounded(
            frame,
            outer,
            TITLE,
            theme.color("primary"),
            theme.color("background"),
            theme,
        );
        ctx.hit(body, "box", 0);
        let content = Rect::new(
            body.x + 2,
            body.y,
            inner_width - u16::from(overflow),
            window as u16,
        );
        let text_rows = lines.len();
        let shown: Vec<Line<'static>> = lines.into_iter().skip(self.offset).take(window).collect();
        frame.render_widget(Paragraph::new(shown).style(text), content);
        // The buttons scroll with the rest: they start one row after the text.
        let button_row = (text_rows + 1).checked_sub(self.offset);
        if let Some(row) = button_row.filter(|row| *row < window) {
            let mut bx = content.x;
            for (focus, label, variant, enabled) in self.buttons() {
                let bw = button_width(label);
                if bx + bw > content.right() {
                    break;
                }
                let rect = Rect::new(bx, content.y + row as u16, bw, 1);
                match enabled {
                    true => draw_button(frame, rect, label, variant, self.focus == focus, theme),
                    false => draw_disabled_button(frame, rect, label, variant, theme),
                }
                if enabled {
                    ctx.hit(rect, "button", Self::index_of(focus));
                }
                bx += bw + 1;
            }
        }
        if overflow {
            let bar = Rect::new(body.right() - 1, body.y, 1, window as u16);
            draw_scrollbar(frame, bar, self.offset, total, window, theme);
        }
    }

    fn mouse(&mut self, input: &MouseInput, ctx: &mut Ctx) -> Action {
        match input.kind {
            MouseKind::WheelUp => {
                self.scroll_key("up");
                Action::None
            }
            MouseKind::WheelDown => {
                self.scroll_key("down");
                Action::None
            }
            MouseKind::Click => match input.on("button") {
                Some(index) => {
                    self.focus = Self::focus_of(index);
                    self.press(ctx)
                }
                None => Action::None,
            },
        }
    }

    fn key(&mut self, _key: &KeyEvent, name: &str, ctx: &mut Ctx) -> Action {
        match name {
            "r" => {
                self.retry(ctx);
                Action::None
            }
            "escape" => self.back(),
            "tab" => {
                self.move_focus(true);
                Action::None
            }
            "shift+tab" | "backtab" => {
                self.move_focus(false);
                Action::None
            }
            "enter" => self.press(ctx),
            name if self.focus == Focus::Box => {
                self.scroll_key(name);
                Action::None
            }
            _ => Action::None,
        }
    }

    fn message(&mut self, msg: Msg, _ctx: &mut Ctx) -> (Option<Msg>, Action) {
        let Msg::Task(payload) = msg else {
            return (Some(msg), Action::None);
        };
        match self.runtime.task_result(payload) {
            Ok(Some(outcome)) => (None, pop(outcome)),
            Ok(None) => (None, Action::None),
            Err(payload) => (Some(Msg::Task(payload)), Action::None),
        }
    }

    fn stop_for_shutdown(&mut self, _ctx: &mut Ctx) -> bool {
        // The checks close their own sources; the whisper child dies with the process.
        false
    }
}

/// The row inside the window where the buttons start, if they are in view.
/// `text_rows` is the content above them, buttons take their margin plus 3 rows.
#[cfg(test)]
mod tests {
    use crossterm::event::{KeyCode, KeyModifiers};

    use std::sync::atomic::Ordering;

    use super::*;
    use crate::app::preflight::PreflightReport;
    use crate::app::preflight::new_checks;
    use crate::app::preflight::tests::{Provider, Whisper, plan, runtime};
    use crate::app::preflight_runtime::{CheckUpdate, ChecksDone};
    use crate::tui::host::Host;

    fn report(failed: &[&str]) -> PreflightReport {
        let mut checks = new_checks();
        for check in &mut checks {
            match failed.contains(&check.key) {
                true => check.set(Status::Fail, "broken", "fix it"),
                false => check.set(Status::Pass, "", ""),
            }
        }
        PreflightReport {
            checks,
            overridden: false,
        }
    }

    struct Run {
        screen: PreflightScreen,
        host: Host,
        _runtime: tokio::runtime::Runtime,
        _dir: tempfile::TempDir,
    }

    fn start() -> Run {
        let dir = tempfile::tempdir().unwrap();
        let runtime = runtime();
        let plan = plan(dir.path(), "{}", Provider::Answers, &runtime);
        let screen = PreflightScreen::new(plan, Box::new(Whisper::default()), None);
        let host = Host::new("textual-dark", runtime.handle().clone());
        Run {
            screen,
            host,
            _runtime: runtime,
            _dir: dir,
        }
    }

    impl Run {
        fn done(&mut self, report: PreflightReport) -> Action {
            let done = ChecksDone {
                result: Ok(report),
                transcriber: Box::new(Whisper::default()),
            };
            self.screen.runtime.running = true;
            let (rest, action) = self
                .screen
                .message(Msg::Task(Box::new(done)), self.host.ctx());
            assert!(rest.is_none());
            action
        }

        fn press(&mut self, name: &str) -> Action {
            let code = match name {
                "tab" => KeyCode::Tab,
                "enter" => KeyCode::Enter,
                "escape" => KeyCode::Esc,
                other => KeyCode::Char(other.chars().next().unwrap()),
            };
            self.screen.key(
                &KeyEvent::new(code, KeyModifiers::NONE),
                name,
                self.host.ctx(),
            )
        }
    }

    fn outcome(action: Action) -> Option<PreflightOutcome> {
        match action {
            Action::Pop(Some(payload)) => payload.downcast::<PreflightOutcome>().ok().map(|o| *o),
            _ => None,
        }
    }

    #[test]
    fn a_clean_report_starts_the_session_at_once() {
        let mut run = start();
        let outcome = outcome(run.done(report(&[]))).expect("the screen leaves with the report");
        assert!(outcome.report.ok());
        assert!(!outcome.report.overridden);
    }

    #[test]
    fn a_failed_ai_check_offers_start_anyway_behind_the_retry_button() {
        let mut run = start();
        assert!(matches!(run.done(report(&["llm"])), Action::None));
        assert!(run.screen.runtime.show_override);
        assert!(matches!(&run.screen.runtime.summary, Summary::Yellow(text)
            if text == "You can start without AI notes: the transcript is still recorded."));
        // Tab: the box, Retry, then Start anyway.
        run.press("tab");
        run.press("tab");
        let outcome = outcome(run.press("enter")).expect("Start anyway leaves");
        assert!(outcome.report.overridden);
        assert!(!outcome.report.llm_ok());
    }

    #[test]
    fn a_blocking_failure_has_no_way_past_it_and_back_cancels() {
        let mut run = start();
        assert!(matches!(run.done(report(&["output"])), Action::None));
        assert!(!run.screen.runtime.show_override);
        assert!(matches!(&run.screen.runtime.summary, Summary::Red(text)
            if text == "Fix the items marked ✗, then press Retry."));
        assert!(matches!(run.press("escape"), Action::Pop(None)));
        assert!(run.screen.runtime.plan.cancel.load(Ordering::SeqCst));
    }

    #[test]
    fn a_check_update_replaces_that_line() {
        let mut run = start();
        let mut update = new_checks().remove(2);
        update.set(Status::Running, "listening", "");
        run.screen
            .message(Msg::Task(Box::new(CheckUpdate(update))), run.host.ctx());
        assert_eq!(run.screen.runtime.checks[2].detail, "listening");
        assert_eq!(run.screen.runtime.checks[2].status, Status::Running);
    }
}

//! The settings dialogs: discard unsaved changes, and remove a model.

use crossterm::event::KeyEvent;
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use crate::domain::model_asset::RemovalEstimate;
use crate::tui::host::{Action, Ctx, MouseInput, MouseKind, Screen};
use crate::tui::keys::{Binding, SCREEN_BASE, action_for, bind};
use crate::tui::text::wrap_line;
use crate::tui::widgets::button::{Variant, button_width, draw_button};
use crate::tui::widgets::{centered, rounded};

/// Escape keeps editing. Enter presses the focused button, and the safe one has the focus.
pub const DISCARD_BINDINGS: [Binding; 5] = [
    SCREEN_BASE[0],
    SCREEN_BASE[1],
    SCREEN_BASE[2],
    SCREEN_BASE[3],
    bind("escape", "keep_editing", "Keep editing").priority(),
];

/// What a dialog popped with.
pub struct Answer {
    pub confirmed: bool,
    pub delete_legacy: bool,
}

/// `_DiscardChangesModal` and `RemoveModelModal`: a title, a body, an optional
/// checkbox, then Cancel and the confirm button at the right. Cancel has the focus,
/// so a reflex Enter never discards or deletes.
pub struct Dialog {
    title: String,
    body: String,
    checkbox: Option<String>,
    checked: bool,
    cancel: &'static str,
    confirm: &'static str,
    /// 0 the checkbox, 1 Cancel, 2 the confirm button.
    focus: usize,
}

fn megabytes(size: u64) -> String {
    format!("{:.0} MB", size as f64 / (1024.0 * 1024.0))
}

fn home_relative(path: &str) -> String {
    let Some(home) = dirs::home_dir() else {
        return path.to_string();
    };
    let home = home.to_string_lossy().to_string();
    if path == home || path.starts_with(&format!("{home}/")) {
        return format!("~{}", &path[home.len()..]);
    }
    path.to_string()
}

impl Dialog {
    pub fn discard() -> Self {
        Self {
            title: "Unsaved changes".into(),
            body: "Your changes to Settings have not been saved. Leave anyway?".into(),
            checkbox: None,
            checked: false,
            cancel: "Keep editing",
            confirm: "Discard and exit",
            focus: 1,
        }
    }

    pub fn remove(label: &str, estimate: &RemovalEstimate) -> Self {
        let mut body = format!("This frees {}.", megabytes(estimate.frees));
        let checkbox = estimate.legacy_path.as_ref().map(|legacy| {
            let extra = estimate.frees_with_legacy.saturating_sub(estimate.frees);
            body += &format!(
                " An older copy in {} uses {} more.",
                home_relative(legacy),
                megabytes(extra)
            );
            format!(
                "Also delete it (frees {} in total)",
                megabytes(estimate.frees_with_legacy)
            )
        });
        body += " It downloads again if you need it.";
        Self {
            title: format!("Remove {label}?"),
            body,
            checkbox,
            checked: false,
            cancel: "Cancel",
            confirm: "Remove",
            focus: 1,
        }
    }

    fn answer(&self, confirmed: bool) -> Action {
        Action::Pop(Some(Box::new(Answer {
            confirmed,
            delete_legacy: self.checked,
        })))
    }

    fn stops(&self) -> Vec<usize> {
        match self.checkbox {
            Some(_) => vec![0, 1, 2],
            None => vec![1, 2],
        }
    }
}

impl Dialog {
    /// Enter on the focused stop.
    fn press(&mut self) -> Action {
        match self.focus {
            0 => {
                self.checked = !self.checked;
                Action::None
            }
            1 => self.answer(false),
            _ => self.answer(true),
        }
    }
}

impl Screen for Dialog {
    fn render(&mut self, frame: &mut Frame, area: Rect, ctx: &crate::tui::host::Ctx) {
        let theme = &ctx.theme;
        // 54 wide: a rounded border with the title in it, `padding: 0 2`.
        let text_width = 54 - 2 - 4;
        let body = wrap_line(&Line::from(self.body.clone()), text_width);
        let checkbox_rows = if self.checkbox.is_some() { 2 } else { 0 };
        let height = 2 + body.len() as u16 + checkbox_rows + 2;
        let outer = centered(area, 54, height);
        frame.render_widget(ratatui::widgets::Clear, outer);
        let surface = theme.color("surface");
        let inner = rounded(
            frame,
            outer,
            &self.title,
            theme.color("warning"),
            surface,
            theme,
        );
        if inner.height < 3 {
            return;
        }
        let text = Style::default()
            .fg(theme.on("foreground", "surface"))
            .bg(surface);
        let x = inner.x + 2;
        let mut y = inner.y;
        for line in body {
            frame.render_widget(
                Paragraph::new(line).style(text),
                Rect::new(x, y, text_width, 1),
            );
            y += 1;
        }
        if let Some(label) = &self.checkbox {
            y += 1;
            let mark = match self.checked {
                true => Span::styled(
                    "[x]",
                    text.fg(theme.color("success")).add_modifier(Modifier::BOLD),
                ),
                false => Span::styled("[ ]", text),
            };
            let mut label_style = text;
            if self.focus == 0 {
                label_style = label_style
                    .fg(theme.color("primary"))
                    .add_modifier(Modifier::BOLD);
            }
            let row = Rect::new(x, y, text_width, 1);
            frame.render_widget(
                Paragraph::new(Line::from(vec![
                    mark,
                    Span::styled(" ", text),
                    Span::styled(label.clone(), label_style),
                ])),
                row,
            );
            ctx.hit(row, "button", 0);
        }
        y += 1;
        // The buttons sit at the right, one column apart.
        let confirm_width = button_width(self.confirm);
        let cancel_width = button_width(self.cancel);
        let right = x + text_width;
        let confirm = Rect::new(right - confirm_width, y, confirm_width, 1);
        let cancel = Rect::new(confirm.x - 1 - cancel_width, y, cancel_width, 1);
        draw_button(
            frame,
            cancel,
            self.cancel,
            Variant::Default,
            self.focus == 1,
            theme,
        );
        draw_button(
            frame,
            confirm,
            self.confirm,
            Variant::Error,
            self.focus == 2,
            theme,
        );
        ctx.hit(cancel, "button", 1);
        ctx.hit(confirm, "button", 2);
    }

    /// A click focuses and presses the checkbox or a button. Clicks outside do nothing.
    fn mouse(&mut self, input: &MouseInput, _ctx: &mut Ctx) -> Action {
        match (input.kind, input.on("button")) {
            (MouseKind::Click, Some(stop)) if self.stops().contains(&stop) => {
                self.focus = stop;
                self.press()
            }
            _ => Action::None,
        }
    }

    fn key(&mut self, _key: &KeyEvent, name: &str, _ctx: &mut Ctx) -> Action {
        let stops = self.stops();
        let at = stops
            .iter()
            .position(|stop| *stop == self.focus)
            .unwrap_or(0);
        match name {
            _ if action_for(&DISCARD_BINDINGS, name) == Some("keep_editing") => self.answer(false),
            "tab" => {
                self.focus = stops[(at + 1) % stops.len()];
                Action::None
            }
            "shift+tab" => {
                self.focus = stops[(at + stops.len() - 1) % stops.len()];
                Action::None
            }
            "enter" | "space" => self.press(),
            _ => Action::None,
        }
    }

    fn is_modal(&self) -> bool {
        true
    }
}

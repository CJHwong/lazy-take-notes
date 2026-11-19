//! The settings dialogs: discard unsaved changes, and remove a model.

use crossterm::event::KeyEvent;
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use crate::domain::model_asset::RemovalEstimate;
use crate::tui::host::{Action, Ctx, Screen};
use crate::tui::keys::{Binding, SCREEN_BASE, action_for, bind};
use crate::tui::text::wrap_line;
use crate::tui::widgets::button::{Variant, button_width, draw_button};
use crate::tui::widgets::{centered, thick_box};

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

impl Screen for Dialog {
    fn render(&mut self, frame: &mut Frame, area: Rect, ctx: &crate::tui::host::Ctx) {
        let theme = &ctx.theme;
        // `width: 54; border: thick $warning; padding: 1 2`.
        let text_width = 54 - 2 - 4;
        let body = wrap_line(&Line::from(self.body.clone()), text_width);
        let checkbox_rows = if self.checkbox.is_some() { 3 } else { 0 };
        let height = 2 + 2 + 2 + body.len() as u16 + 1 + checkbox_rows + 3;
        let outer = centered(area, 54, height);
        let inner = thick_box(frame, outer, "warning", "surface", theme);
        if inner.height < 6 {
            return;
        }
        let surface = theme.color("surface");
        let text = Style::default()
            .fg(theme.on("foreground", "surface"))
            .bg(surface);
        let x = inner.x + 2;
        let mut y = inner.y + 1;
        frame.render_widget(
            Paragraph::new(Line::styled(
                self.title.clone(),
                text.add_modifier(Modifier::BOLD),
            )),
            Rect::new(x, y, text_width, 1),
        );
        y += 2;
        for line in body {
            frame.render_widget(
                Paragraph::new(line).style(text),
                Rect::new(x, y, text_width, 1),
            );
            y += 1;
        }
        y += 1;
        if let Some(label) = &self.checkbox {
            // Textual's Checkbox: `▐X▌` and the label, in a `tall` border.
            let mark = Style::default()
                .fg(theme.color(if self.checked { "success" } else { "panel" }))
                .bg(theme.color("panel"));
            let mut label_style = text;
            if self.focus == 0 {
                label_style = label_style.add_modifier(Modifier::REVERSED);
            }
            frame.render_widget(
                Paragraph::new(Line::from(vec![
                    Span::styled(" ▐", text),
                    Span::styled("X", mark),
                    Span::styled("▌ ", text),
                    Span::styled(label.clone(), label_style),
                ])),
                Rect::new(x, y + 1, text_width, 1),
            );
            y += 3;
        }
        // `#discard-actions { align: right middle }`, each button after `margin-left: 2`.
        let confirm_width = button_width(self.confirm);
        let cancel_width = button_width(self.cancel);
        let right = x + text_width;
        let confirm_x = right - confirm_width;
        let cancel_x = confirm_x - 2 - cancel_width;
        draw_button(
            frame,
            Rect::new(cancel_x, y, cancel_width, 3),
            self.cancel,
            Variant::Default,
            self.focus == 1,
            theme,
        );
        draw_button(
            frame,
            Rect::new(confirm_x, y, confirm_width, 3),
            self.confirm,
            Variant::Error,
            self.focus == 2,
            theme,
        );
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
            "enter" | "space" => match self.focus {
                0 => {
                    self.checked = !self.checked;
                    Action::None
                }
                1 => self.answer(false),
                _ => self.answer(true),
            },
            _ => Action::None,
        }
    }

    fn is_modal(&self) -> bool {
        true
    }
}

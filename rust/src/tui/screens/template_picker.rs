//! Pick the template that configures a session (Python's `TemplatePicker`).

use std::path::PathBuf;
use std::time::Duration;

use crossterm::event::KeyEvent;
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use tracing::warn;

use crate::adapters::templates::YamlTemplates;
use crate::app::template_catalog::{NO_TEMPLATES, Row, TemplateCatalog};
use crate::tui::host::{Action, Ctx, MouseInput, MouseKind, Msg, Payload, Screen};
use crate::tui::keys::{Binding, SCREEN_BASE, bind};
use crate::tui::screens::modals::MarkdownModal;
use crate::tui::screens::picker::{Picker, PickerKey};
use crate::tui::text::center;
use crate::tui::widgets::button::{Variant, button_width, draw_button};
use crate::tui::widgets::{centered, thick_box};

/// What the picker closed with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TemplateChoice {
    /// The key of the template to use.
    Key(String),
    /// `n`: build a new template, then show the picker again.
    Create,
    /// `a`: edit this template with the AI builder, then show the picker again.
    EditWithAi(String),
}

/// The editor command: `$VISUAL`, then `$EDITOR`, then the platform's opener.
pub fn resolve_editor() -> Option<Vec<String>> {
    for variable in ["VISUAL", "EDITOR"] {
        let value = std::env::var(variable).unwrap_or_default();
        if !value.trim().is_empty() {
            return Some(value.split_whitespace().map(str::to_string).collect());
        }
    }
    let fallback: &[&str] = if cfg!(target_os = "macos") {
        &["open", "-t"]
    } else if cfg!(target_os = "linux") {
        &["xdg-open"]
    } else if cfg!(windows) {
        &["notepad"]
    } else {
        return None;
    };
    Some(fallback.iter().map(|part| part.to_string()).collect())
}

/// The editor ended; reload this template.
struct Edited(String);

/// The delete confirmation's answer.
pub struct Answer(pub bool);

pub struct TemplatePicker {
    picker: Picker,
    catalog: TemplateCatalog,
    rows: Vec<Row>,
    current: Option<String>,
    /// Set after `e`: a GUI editor returns at once, so reload again on focus.
    pending_reload: Option<String>,
    noticed: PathBuf,
}

impl TemplatePicker {
    pub fn new(templates: YamlTemplates, show_builtins: bool, noticed: PathBuf) -> Self {
        let catalog = TemplateCatalog::new(templates, show_builtins);
        let mut picker = Picker::new("Filter templates...", 40);
        picker.list.wrap = true;
        // Nothing to filter, and a focused filter would take the n key.
        picker.list_focused = catalog.loaded.is_empty();
        let mut screen = Self {
            picker,
            catalog,
            rows: Vec::new(),
            current: None,
            pending_reload: None,
            noticed,
        };
        screen.rebuild("");
        screen
    }

    fn query(&self) -> String {
        self.picker.search.value().trim().to_lowercase()
    }

    fn rebuild(&mut self, query: &str) {
        self.rows = self.catalog.rows(query);
        let disabled = self
            .rows
            .iter()
            .map(|row| matches!(row, Row::Locale(_)))
            .collect();
        self.picker.list.reset_with(disabled);
        self.current = self.highlighted();
        match self.current.clone() {
            Some(name) => self.show_preview(&name),
            None if self.catalog.loaded.is_empty() => {
                self.picker.preview.set_markdown(NO_TEMPLATES)
            }
            None => self.picker.preview.set_markdown("*No matching templates*"),
        }
    }

    fn highlighted(&self) -> Option<String> {
        match self.rows.get(self.picker.list.index?)? {
            Row::Template(name) => Some(name.clone()),
            Row::Locale(_) => None,
        }
    }

    fn show_preview(&mut self, name: &str) {
        if let Some(markdown) = self.catalog.preview_markdown(name) {
            self.picker.preview.set_markdown(&markdown);
        }
    }

    fn row_line(&self, row: &Row) -> Line<'static> {
        let dim = Style::default().add_modifier(Modifier::DIM);
        match row {
            Row::Locale(locale) => Line::from(Span::styled(
                locale.to_uppercase(),
                Style::default().add_modifier(Modifier::BOLD),
            )),
            Row::Template(name) => {
                let template = &self.catalog.loaded[name];
                let label = match template.metadata.name.is_empty() {
                    true => name.clone(),
                    false => template.metadata.name.clone(),
                };
                let mut spans = vec![
                    Span::raw(format!("{label}  ")),
                    Span::styled(format!("({})", template.metadata.locale), dim),
                ];
                if self.catalog.user_names.contains(name) {
                    spans.push(Span::raw("  "));
                    spans.push(Span::styled("[user]", dim));
                }
                Line::from(spans)
            }
        }
    }

    fn edit_template(&mut self, ctx: &mut Ctx) {
        let Some(name) = self.current.clone() else {
            return;
        };
        let Some(editor) = resolve_editor() else {
            ctx.error(
                "No editor found ($VISUAL / $EDITOR)",
                Duration::from_secs(5),
            );
            return;
        };
        let path = match self.catalog.templates.ensure_user_copy(&name) {
            Ok(path) => path,
            Err(e) => {
                ctx.error(
                    format!("Cannot copy the template: {e}"),
                    Duration::from_secs(8),
                );
                return;
            }
        };
        ctx.suspend(move || {
            let status = std::process::Command::new(&editor[0])
                .args(&editor[1..])
                .arg(&path)
                .status();
            if let Err(e) = status {
                warn!(target: "ltn.app", "Cannot run the editor {}: {e}", editor[0]);
            }
            Edited(name)
        });
    }

    /// Reload a template after an outside edit, rebuild the list, keep it highlighted.
    fn reload_after_edit(&mut self, name: &str, ctx: &mut Ctx) {
        if self.catalog.reload(name).is_err() {
            ctx.warn(
                format!("Template \"{name}\" has invalid YAML \u{2014} changes ignored"),
                Duration::from_secs(5),
            );
            return;
        }
        let query = self.query();
        self.rebuild(&query);
        let at = self
            .rows
            .iter()
            .position(|row| matches!(row, Row::Template(key) if key == name));
        if let Some(at) = at {
            self.picker.list.select(at);
            self.current = Some(name.to_string());
            self.show_preview(name);
        }
    }

    fn delete_template(&mut self, ctx: &mut Ctx) -> Action {
        let Some(name) = self.current.clone() else {
            return Action::None;
        };
        if !self.catalog.user_names.contains(&name) {
            ctx.warn("Only user templates can be deleted", Duration::from_secs(5));
            return Action::None;
        }
        Action::Push(Box::new(Confirm::delete(&name)))
    }

    fn on_delete_confirmed(&mut self, ctx: &mut Ctx) {
        let Some(name) = self.current.clone() else {
            return;
        };
        if self.catalog.delete(&name).is_err() {
            ctx.error(
                format!("\"{name}\" is not a user template"),
                Duration::from_secs(5),
            );
            return;
        }
        let query = self.query();
        self.rebuild(&query);
        ctx.notify(format!("Deleted user template \"{name}\""));
    }

    /// The keys that act on the highlighted template while the list has the focus.
    fn list_command(&mut self, name: &str, ctx: &mut Ctx) -> Action {
        match name {
            "e" => self.edit_template(ctx),
            "n" => return Action::Pop(Some(Box::new(TemplateChoice::Create))),
            "a" => {
                if let Some(key) = self.current.clone() {
                    return Action::Pop(Some(Box::new(TemplateChoice::EditWithAi(key))));
                }
            }
            "x" => return self.delete_template(ctx),
            _ => {}
        }
        Action::None
    }

    fn after(&mut self, outcome: PickerKey, name: &str, ctx: &mut Ctx) -> Action {
        match outcome {
            PickerKey::Cancel => Action::Pop(None),
            PickerKey::Select => match self.current.clone() {
                Some(key) => Action::Pop(Some(Box::new(TemplateChoice::Key(key)))),
                None => Action::None,
            },
            PickerKey::Filter(query) => {
                self.rebuild(&query);
                Action::None
            }
            PickerKey::Moved => {
                if let Some(key) = self.highlighted() {
                    self.current = Some(key.clone());
                    self.show_preview(&key);
                }
                Action::None
            }
            PickerKey::Unused if self.picker.list_focused => self.list_command(name, ctx),
            PickerKey::Used | PickerKey::Unused => Action::None,
        }
    }
}

impl Screen for TemplatePicker {
    fn mounted(&mut self, _ctx: &mut Ctx) -> Action {
        if self.noticed.exists() {
            return Action::None;
        }
        Action::Push(Box::new(MarkdownModal::builtin_templates_notice(
            self.noticed.clone(),
        )))
    }

    fn render(&mut self, frame: &mut Frame, area: Rect, ctx: &Ctx) {
        let header = format!(
            "  Select a template ({} available)",
            self.catalog.loaded.len()
        );
        let rows: Vec<Line<'static>> = self.rows.iter().map(|row| self.row_line(row)).collect();
        self.picker.render(
            frame,
            area,
            &header,
            "[Enter] Select  [↑/↓] Navigate  [n] New  [e] Edit  [a] Edit with AI  [x] Delete  [Esc] Cancel",
            &rows,
            ctx,
        );
    }

    fn key(&mut self, key: &KeyEvent, name: &str, ctx: &mut Ctx) -> Action {
        let outcome = self.picker.key(key, name);
        self.after(outcome, name, ctx)
    }

    fn paste(&mut self, text: &str, ctx: &mut Ctx) -> Action {
        let outcome = self.picker.paste(text);
        self.after(outcome, "", ctx)
    }

    fn mouse(&mut self, input: &MouseInput, ctx: &mut Ctx) -> Action {
        let outcome = self.picker.mouse(input);
        self.after(outcome, "", ctx)
    }

    fn message(&mut self, msg: Msg, ctx: &mut Ctx) -> (Option<Msg>, Action) {
        let Msg::Task(payload) = msg else {
            return (Some(msg), Action::None);
        };
        match payload.downcast::<Edited>() {
            Ok(edited) => {
                // A terminal editor has closed; a GUI one reloads again on focus.
                self.pending_reload = Some(edited.0.clone());
                self.reload_after_edit(&edited.0, ctx);
                (None, Action::None)
            }
            Err(payload) => (Some(Msg::Task(payload)), Action::None),
        }
    }

    fn resumed(&mut self, result: Option<Payload>, ctx: &mut Ctx) -> Action {
        let confirmed = result
            .and_then(|payload| payload.downcast::<Answer>().ok())
            .is_some_and(|answer| answer.0);
        if confirmed {
            self.on_delete_confirmed(ctx);
        }
        Action::None
    }

    fn app_focus(&mut self, ctx: &mut Ctx) -> Action {
        if let Some(name) = self.pending_reload.take() {
            self.reload_after_edit(&name, ctx);
        }
        Action::None
    }
}

pub const CONFIRM_DELETE_BINDINGS: [Binding; 9] = [
    SCREEN_BASE[0],
    SCREEN_BASE[1],
    SCREEN_BASE[2],
    SCREEN_BASE[3],
    bind("y", "confirm", "Yes"),
    bind("n", "dismiss_no", "No"),
    bind("escape", "dismiss_no", "Cancel"),
    bind("left", "app.focus_previous", "Focus previous").hidden(),
    bind("right", "app.focus_next", "Focus next").hidden(),
];

/// A question with a No and a Yes button, stacked. No has the focus. Pops with `Answer`.
pub struct Confirm {
    message: Line<'static>,
    border: &'static str,
    yes: Variant,
    yes_focused: bool,
    /// The arrow keys move the focus too (the delete dialog binds them).
    pub arrows: bool,
}

impl Confirm {
    pub fn new(message: Line<'static>, border: &'static str, yes: Variant) -> Self {
        Self {
            message,
            border,
            yes,
            yes_focused: false,
            arrows: true,
        }
    }

    /// "Delete user template?"
    pub fn delete(name: &str) -> Self {
        let message = Line::from(vec![
            Span::raw("Delete user template "),
            Span::styled(
                format!("\"{name}\""),
                Style::default().add_modifier(Modifier::BOLD),
            ),
            Span::raw("?"),
        ]);
        Self::new(message, "error", Variant::Error)
    }

    fn answer(yes: bool) -> Action {
        Action::Pop(Some(Box::new(Answer(yes))))
    }
}

impl Screen for Confirm {
    fn render(&mut self, frame: &mut Frame, area: Rect, ctx: &Ctx) {
        let theme = &ctx.theme;
        // The message, a blank row, then No and Yes side by side.
        let outer = centered(area, 50, 2 + 3);
        let inner = thick_box(frame, outer, self.border, "surface", theme);
        if inner.height < 3 || inner.width < 20 {
            return;
        }
        let text = Style::default()
            .fg(theme.on("foreground", "surface"))
            .bg(theme.color("surface"));
        frame.render_widget(
            Paragraph::new(center(self.message.clone(), inner.width)).style(text),
            Rect::new(inner.x, inner.y, inner.width, 1),
        );
        let no_width = button_width("No (n)");
        let yes_width = button_width("Yes (y)");
        let x = inner.x + inner.width.saturating_sub(no_width + 2 + yes_width) / 2;
        let no = Rect::new(x, inner.y + 2, no_width, 1);
        let yes = Rect::new(x + no_width + 2, inner.y + 2, yes_width, 1);
        draw_button(
            frame,
            no,
            "No (n)",
            Variant::Default,
            !self.yes_focused,
            theme,
        );
        draw_button(frame, yes, "Yes (y)", self.yes, self.yes_focused, theme);
        ctx.hit(no, "button", 0);
        ctx.hit(yes, "button", 1);
    }

    fn mouse(&mut self, input: &MouseInput, _ctx: &mut Ctx) -> Action {
        match (input.kind, input.on("button")) {
            (MouseKind::Click, Some(index)) => Self::answer(index == 1),
            _ => Action::None,
        }
    }

    fn key(&mut self, _key: &KeyEvent, name: &str, _ctx: &mut Ctx) -> Action {
        match name {
            "y" => Self::answer(true),
            "n" | "escape" => Self::answer(false),
            "enter" | "space" => Self::answer(self.yes_focused),
            "tab" | "shift+tab" => {
                self.yes_focused = !self.yes_focused;
                Action::None
            }
            "left" | "right" | "up" | "down" if self.arrows => {
                self.yes_focused = !self.yes_focused;
                Action::None
            }
            _ => Action::None,
        }
    }

    fn is_modal(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crossterm::event::{KeyCode, KeyModifiers};

    fn runtime() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
    }

    fn picker(root: &std::path::Path) -> TemplatePicker {
        let paths = crate::adapters::paths::Paths::under(root);
        let noticed = root.join("noticed");
        std::fs::write(&noticed, "").unwrap();
        TemplatePicker::new(YamlTemplates::new(&paths), true, noticed)
    }

    fn press(screen: &mut TemplatePicker, name: &str, ctx: &mut Ctx) -> Action {
        let code = match name {
            "down" => KeyCode::Down,
            other => KeyCode::Char(other.chars().next().unwrap()),
        };
        screen.key(&KeyEvent::new(code, KeyModifiers::NONE), name, ctx)
    }

    fn choice(action: Action) -> Option<TemplateChoice> {
        match action {
            Action::Pop(Some(payload)) => payload.downcast::<TemplateChoice>().ok().map(|c| *c),
            _ => None,
        }
    }

    #[test]
    fn the_template_keys_type_into_the_search_box_while_it_has_the_focus() {
        let root = tempfile::tempdir().unwrap();
        let runtime = runtime();
        let mut host = crate::tui::host::Host::new("textual-dark", runtime.handle().clone());
        let ctx = host.ctx();
        let mut screen = picker(root.path());
        for name in ["n", "e", "a", "x"] {
            assert!(choice(press(&mut screen, name, ctx)).is_none(), "{name}");
        }
        assert_eq!(screen.picker.search.value(), "neax");
        assert!(ctx.take_suspend_job().is_none());
    }

    #[test]
    fn the_template_keys_act_once_the_list_has_the_focus() {
        let root = tempfile::tempdir().unwrap();
        let runtime = runtime();
        let mut host = crate::tui::host::Host::new("textual-dark", runtime.handle().clone());
        let ctx = host.ctx();
        let mut screen = picker(root.path());
        press(&mut screen, "down", ctx);
        assert!(matches!(
            choice(press(&mut screen, "n", ctx)),
            Some(TemplateChoice::Create)
        ));
        let key = screen.current.clone().unwrap();
        match choice(press(&mut screen, "a", ctx)) {
            Some(TemplateChoice::EditWithAi(edited)) => assert_eq!(edited, key),
            _ => panic!("a must ask the builder to edit {key}"),
        }
        // A built-in template cannot be deleted; the picker says so.
        assert!(matches!(press(&mut screen, "x", ctx), Action::None));
        assert_eq!(ctx.toast_texts(), ["Only user templates can be deleted"]);
        assert!(screen.picker.search.value().is_empty());
    }

    #[test]
    fn a_click_highlights_a_template_and_a_second_click_picks_it() {
        use crossterm::event::{MouseButton, MouseEvent, MouseEventKind};
        let root = tempfile::tempdir().unwrap();
        let runtime = runtime();
        let mut host = crate::tui::host::Host::new("textual-dark", runtime.handle().clone());
        host.push(Box::new(picker(root.path())));
        let mut terminal =
            ratatui::Terminal::new(ratatui::backend::TestBackend::new(100, 30)).unwrap();
        let click = MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column: 6,
            row: 7,
            modifiers: KeyModifiers::NONE,
        };
        let row_at = |terminal: &ratatui::Terminal<ratatui::backend::TestBackend>| -> String {
            let buffer = terminal.backend().buffer();
            (0..30)
                .map(|x| buffer[(x, 7)].symbol().to_string())
                .collect()
        };
        terminal.draw(|frame| host.draw(frame)).unwrap();
        assert!(!row_at(&terminal).contains('▶'), "{}", row_at(&terminal));
        host.mouse(&click);
        terminal.draw(|frame| host.draw(frame)).unwrap();
        assert!(row_at(&terminal).contains('▶'), "{}", row_at(&terminal));
        assert!(!host.finished());
        host.mouse(&click);
        assert!(host.finished());
    }

    #[test]
    fn the_editor_comes_from_visual_then_editor() {
        // Both variables are read; only check that a set one splits into argv.
        let editor = resolve_editor().unwrap();
        assert!(!editor.is_empty());
    }
}

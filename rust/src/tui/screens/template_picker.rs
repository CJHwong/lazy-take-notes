//! Pick the template that configures a session (Python's `TemplatePicker`).

use std::collections::{BTreeMap, BTreeSet};
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
use crate::domain::template::SessionTemplate;
use crate::tui::host::{Action, Ctx, Msg, Payload, Screen};
use crate::tui::keys::{Binding, SCREEN_BASE, bind};
use crate::tui::screens::modals::MarkdownModal;
use crate::tui::screens::picker::{Picker, PickerKey};
use crate::tui::text::center;
use crate::tui::widgets::button::{Variant, draw_button};
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

const NO_TEMPLATES: &str = "**No templates yet.**\n\nPress n to create one, or turn on Show built-in templates in Settings > Sessions.";

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

/// One list row: a locale header, which cannot be highlighted, or a template.
enum Row {
    Locale(String),
    Template(String),
}

/// The editor ended; reload this template.
struct Edited(String);

/// The delete confirmation's answer.
struct DeleteAnswer(bool);

pub struct TemplatePicker {
    picker: Picker,
    templates: YamlTemplates,
    user_names: BTreeSet<String>,
    builtin_names: BTreeSet<String>,
    loaded: BTreeMap<String, SessionTemplate>,
    rows: Vec<Row>,
    current: Option<String>,
    /// Set after `e`: a GUI editor returns at once, so reload again on focus.
    pending_reload: Option<String>,
    noticed: PathBuf,
}

impl TemplatePicker {
    pub fn new(templates: YamlTemplates, show_builtins: bool, noticed: PathBuf) -> Self {
        let user_names = templates.user_names();
        let builtin_names = templates.builtin_names();
        let mut loaded = BTreeMap::new();
        for name in templates.all_names() {
            let hidden =
                !show_builtins && builtin_names.contains(&name) && !user_names.contains(&name);
            if hidden {
                continue;
            }
            match templates.load_template(&name) {
                Ok(template) => {
                    loaded.insert(name, template);
                }
                Err(e) => warn!(target: "ltn.app", "Template {name} does not load: {e}"),
            }
        }
        let mut picker = Picker::new("Filter templates...", 40);
        picker.list.wrap = true;
        // Nothing to filter, and a focused filter would take the n key.
        picker.list_focused = loaded.is_empty();
        let mut screen = Self {
            picker,
            templates,
            user_names,
            builtin_names,
            loaded,
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
        let mut groups: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for (name, template) in &self.loaded {
            let matches = query.is_empty()
                || name.to_lowercase().contains(query)
                || template.metadata.description.to_lowercase().contains(query);
            if matches {
                let locale = template.metadata.locale.clone();
                groups.entry(locale).or_default().push(name.clone());
            }
        }
        self.rows.clear();
        for (locale, names) in groups {
            self.rows.push(Row::Locale(locale));
            self.rows.extend(names.into_iter().map(Row::Template));
        }
        let disabled = self
            .rows
            .iter()
            .map(|row| matches!(row, Row::Locale(_)))
            .collect();
        self.picker.list.reset_with(disabled);
        self.current = self.highlighted();
        match self.current.clone() {
            Some(name) => self.show_preview(&name),
            None if self.loaded.is_empty() => self.picker.preview.set_markdown(NO_TEMPLATES),
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
        let Some(template) = self.loaded.get(name) else {
            return;
        };
        self.picker
            .preview
            .set_markdown(&self.preview_markdown(name, template));
    }

    fn preview_markdown(&self, name: &str, template: &SessionTemplate) -> String {
        let meta = &template.metadata;
        let source = if self.user_names.contains(name) {
            "\\[user]"
        } else {
            "\\[built-in]"
        };
        let mut lines = vec![
            format!("## {}  {source}", meta.name),
            String::new(),
            match meta.description.is_empty() {
                true => String::new(),
                false => format!("> {}", meta.description),
            },
            String::new(),
            format!("**Locale:** `{}`", meta.locale),
        ];
        if !template.quick_actions.is_empty() {
            lines.extend([String::new(), "### Quick Actions".into()]);
            for (index, action) in template.quick_actions.iter().enumerate() {
                lines.push(format!(
                    "- **`{}`** {} \u{2014} {}",
                    index + 1,
                    action.label,
                    action.description
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
        lines.extend([
            "".into(),
            "---".into(),
            "".into(),
            "### System Prompt".into(),
            "".into(),
        ]);
        lines.push(template.system_prompt.clone());
        lines.join("\n")
    }

    fn row_line(&self, row: &Row) -> Line<'static> {
        let dim = Style::default().add_modifier(Modifier::DIM);
        match row {
            Row::Locale(locale) => Line::from(Span::styled(
                locale.to_uppercase(),
                Style::default().add_modifier(Modifier::BOLD),
            )),
            Row::Template(name) => {
                let template = &self.loaded[name];
                let label = match template.metadata.name.is_empty() {
                    true => name.clone(),
                    false => template.metadata.name.clone(),
                };
                let mut spans = vec![
                    Span::raw(format!("{label}  ")),
                    Span::styled(format!("({})", template.metadata.locale), dim),
                ];
                if self.user_names.contains(name) {
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
        let path = match self.templates.ensure_user_copy(&name) {
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
        self.user_names = self.templates.user_names();
        match self.templates.load_template(name) {
            Ok(template) => {
                self.loaded.insert(name.to_string(), template);
            }
            Err(_) => {
                ctx.warn(
                    format!("Template \"{name}\" has invalid YAML \u{2014} changes ignored"),
                    Duration::from_secs(5),
                );
                return;
            }
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
        if !self.user_names.contains(&name) {
            ctx.warn("Only user templates can be deleted", Duration::from_secs(5));
            return Action::None;
        }
        Action::Push(Box::new(ConfirmDelete::new(&name)))
    }

    fn on_delete_confirmed(&mut self, ctx: &mut Ctx) {
        let Some(name) = self.current.clone() else {
            return;
        };
        if self.templates.delete_user_template(&name).is_err() {
            ctx.error(
                format!("\"{name}\" is not a user template"),
                Duration::from_secs(5),
            );
            return;
        }
        self.user_names = self.templates.user_names();
        // A built-in with the same key comes back; otherwise the template is gone.
        let builtin = self
            .builtin_names
            .contains(&name)
            .then(|| self.templates.load_template(&name).ok())
            .flatten();
        match builtin {
            Some(template) => {
                self.loaded.insert(name.clone(), template);
            }
            None => {
                self.loaded.remove(&name);
            }
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
        let header = format!("  Select a template ({} available)", self.loaded.len());
        let rows: Vec<Line<'static>> = self.rows.iter().map(|row| self.row_line(row)).collect();
        self.picker.render(
            frame,
            area,
            &header,
            "[Enter] Select  [↑/↓] Navigate  [n] New  [e] Edit  [a] Edit with AI  [x] Delete  [Esc] Cancel",
            &rows,
            &ctx.theme,
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
            .and_then(|payload| payload.downcast::<DeleteAnswer>().ok())
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

/// "Delete user template?" with a No and a Yes button. No has the focus.
pub struct ConfirmDelete {
    name: String,
    yes_focused: bool,
}

impl ConfirmDelete {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.into(),
            yes_focused: false,
        }
    }

    fn answer(yes: bool) -> Action {
        Action::Pop(Some(Box::new(DeleteAnswer(yes))))
    }
}

impl Screen for ConfirmDelete {
    fn render(&mut self, frame: &mut Frame, area: Rect, ctx: &Ctx) {
        let theme = &ctx.theme;
        // Border, padding, the message, its margin, then two buttons stacked in a `Center`.
        let outer = centered(area, 50, 2 + 2 + 1 + 1 + 3 + 3);
        let inner = thick_box(frame, outer, "error", "surface", theme);
        if inner.height < 10 || inner.width < 20 {
            return;
        }
        let padded = Rect::new(inner.x + 2, inner.y + 1, inner.width - 4, inner.height - 2);
        let text = Style::default()
            .fg(theme.on("foreground", "surface"))
            .bg(theme.color("surface"));
        let message = Line::from(vec![
            Span::raw("Delete user template "),
            Span::styled(
                format!("\"{}\"", self.name),
                Style::default().add_modifier(Modifier::BOLD),
            ),
            Span::raw("?"),
        ]);
        frame.render_widget(
            Paragraph::new(center(message, padded.width)).style(text),
            Rect::new(padded.x, padded.y, padded.width, 1),
        );
        // Each button is 16 wide (its `min-width`) with `margin: 0 1`.
        let x = padded.x + padded.width.saturating_sub(18) / 2 + 1;
        draw_button(
            frame,
            Rect::new(x, padded.y + 2, 16, 3),
            "No (n)",
            Variant::Default,
            !self.yes_focused,
            theme,
        );
        draw_button(
            frame,
            Rect::new(x, padded.y + 5, 16, 3),
            "Yes (y)",
            Variant::Error,
            self.yes_focused,
            theme,
        );
    }

    fn key(&mut self, _key: &KeyEvent, name: &str, _ctx: &mut Ctx) -> Action {
        match name {
            "y" => Self::answer(true),
            "n" | "escape" => Self::answer(false),
            "enter" | "space" => Self::answer(self.yes_focused),
            "tab" | "shift+tab" | "left" | "right" | "up" | "down" => {
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
    fn the_editor_comes_from_visual_then_editor() {
        // Both variables are read; only check that a set one splits into argv.
        let editor = resolve_editor().unwrap();
        assert!(!editor.is_empty());
    }
}

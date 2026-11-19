//! The layout and keys every searchable picker shares (Python's `SearchablePicker`).

use crossterm::event::KeyEvent;
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::text::Line;

use crate::tui::keys::{Binding, SCREEN_BASE, bind};
use crate::tui::theme::Theme;
use crate::tui::widgets::input::Input;
use crate::tui::widgets::list::ListCursor;
use crate::tui::widgets::scroll::ScrollView;
use crate::tui::widgets::{bar, panel, split_bars};

pub const PICKER_BINDINGS: [Binding; 7] = [
    SCREEN_BASE[0],
    SCREEN_BASE[1],
    SCREEN_BASE[2],
    SCREEN_BASE[3],
    bind("escape", "cancel", "Cancel").priority(),
    bind("q", "cancel", "Cancel"),
    bind("enter", "select_item", "Select").priority(),
];

/// What a key did in the picker.
#[derive(Debug, PartialEq, Eq)]
pub enum PickerKey {
    Cancel,
    Select,
    /// The filter text changed; rebuild the list with this lowercased query.
    Filter(String),
    /// The highlight moved; show the preview of the new item.
    Moved,
    Used,
    Unused,
}

pub struct Picker {
    pub search: Input,
    pub list: ListCursor,
    pub list_focused: bool,
    pub preview: ScrollView,
    /// `max-width` of the list pane.
    pub max_list_width: u16,
    /// The `fr` widths of the list pane and the preview.
    pub fractions: (u16, u16),
}

impl Picker {
    pub fn new(placeholder: &str, max_list_width: u16) -> Self {
        Self {
            search: Input::new(placeholder),
            list: ListCursor::default(),
            list_focused: false,
            preview: {
                // The preview box's `padding: 1 2`; the Markdown inside it has its own `0 2`.
                let mut preview = ScrollView::markdown();
                preview.padding = 2;
                preview.padding_y = 1;
                preview.content_padding = 2;
                preview
            },
            max_list_width,
            fractions: (1, 3),
        }
    }

    pub fn key(&mut self, key: &KeyEvent, name: &str) -> PickerKey {
        match name {
            "escape" => return PickerKey::Cancel,
            "enter" => return PickerKey::Select,
            "tab" | "shift+tab" => {
                self.list_focused = !self.list_focused;
                return PickerKey::Used;
            }
            _ => {}
        }
        if self.list_focused {
            return self.list_key(name);
        }
        if name == "down" {
            self.list_focused = true;
            self.list.select(self.list.first_enabled().unwrap_or(0));
            return PickerKey::Moved;
        }
        let before = self.search.value();
        if !self.search.key(key, name) {
            return PickerKey::Unused;
        }
        let after = self.search.value();
        if after == before {
            PickerKey::Used
        } else {
            PickerKey::Filter(after.trim().to_lowercase())
        }
    }

    fn list_key(&mut self, name: &str) -> PickerKey {
        if name == "up" && self.list.index.is_some() && self.list.index == self.list.first_enabled()
        {
            self.list_focused = false;
            return PickerKey::Used;
        }
        if name == "q" {
            return PickerKey::Cancel;
        }
        let before = self.list.index;
        if !self.list.key(name) {
            return PickerKey::Unused;
        }
        if self.list.index == before {
            PickerKey::Used
        } else {
            PickerKey::Moved
        }
    }

    pub fn paste(&mut self, text: &str) -> PickerKey {
        if self.list_focused {
            return PickerKey::Unused;
        }
        self.search.paste(text);
        PickerKey::Filter(self.search.value().trim().to_lowercase())
    }

    pub fn render(
        &mut self,
        frame: &mut Frame,
        area: Rect,
        header: &str,
        footer: &str,
        rows: &[Line<'_>],
        theme: &Theme,
    ) {
        let (top, body, bottom) = split_bars(area, 1, 1);
        bar(frame, top, header, "primary", "text", true, theme);
        bar(frame, bottom, footer, "surface", "text-muted", false, theme);
        // The list pane and the preview share the width by `fr`; the list keeps
        // between 24 and its max.
        let (list_fr, preview_fr) = self.fractions;
        let list_width = (body.width * list_fr / (list_fr + preview_fr))
            .clamp(24.min(body.width), self.max_list_width.max(24))
            .min(body.width);
        let pane = Rect::new(body.x, body.y, list_width, body.height);
        let preview_area = Rect::new(
            body.x + list_width,
            body.y,
            body.width - list_width,
            body.height,
        );
        self.search.render(
            frame,
            Rect::new(pane.x, pane.y, pane.width, 3.min(pane.height)),
            !self.list_focused,
            theme,
        );
        let list_top = (pane.y + 4).min(pane.bottom());
        let list_area = Rect::new(pane.x, list_top, pane.width, pane.bottom() - list_top);
        if list_area.height >= 2 {
            let surface = theme.color("surface");
            let background = if self.list_focused {
                theme.tinted("surface", "foreground", 0.05)
            } else {
                surface
            };
            let inner = panel(frame, list_area, "", "primary", background, theme);
            self.list
                .render(frame, inner, rows, self.list_focused, background, theme);
        }
        let background = theme.color("background");
        let inner = panel(frame, preview_area, "", "secondary", background, theme);
        let style = Style::default()
            .bg(background)
            .fg(theme.color("foreground"));
        self.preview.render(frame, inner, style, theme);
    }
}

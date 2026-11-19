//! The settings pages as item tables. Each item carries Python's widget id;
//! layout, drawing, focus order and the saved data all read the same table.

use std::any::Any;

use crossterm::event::KeyEvent;
use ratatui::Frame;
use ratatui::backend::TestBackend;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use crate::tui::text::wrap_line;
use crate::tui::theme::Theme;
use crate::tui::widgets::input::Input;
use crate::tui::widgets::select::{
    Select, SelectKey, compact_button_width, draw_compact_button, draw_switch,
};
use crate::tui::widgets::text_area::TextArea;

/// `.field-label { width: 24 }`.
pub const LABEL_WIDTH: u16 = 24;
/// `scrollbar-size-vertical: 2`.
const SCROLLBAR: u16 = 2;
/// `#cfg-recognition-hints { min-height: 5; max-height: 16 }`.
const HINTS_MIN: u16 = 5;
const HINTS_MAX: u16 = 16;

/// What the page reports to the screen.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FormEvent {
    Pressed(String),
    /// A select changed: its id and new value.
    Changed(&'static str, String),
    /// An item that is not a plain widget asked the screen to act.
    Custom(&'static str, String),
}

/// A part of the page drawn by its owner (model rows, the level meters, the preview).
pub trait Custom {
    fn height(&self, width: u16) -> u16;
    fn render(&mut self, frame: &mut Frame, area: Rect, focus: Option<usize>, theme: &Theme);
    /// How many focus stops it has.
    fn stops(&self) -> usize {
        0
    }
    /// A key at focus stop `stop`. None when unused; the item may move `stop`
    /// (a removed row hands the focus on).
    fn key(
        &mut self,
        _stop: &mut usize,
        _key: &KeyEvent,
        _name: &str,
    ) -> Option<Option<FormEvent>> {
        None
    }
    fn paste(&mut self, _stop: usize, _text: &str) {}
    /// The focus arrived at `stop` (an input selects its value).
    fn focused(&mut self, _stop: usize) {}
    /// An open select list, drawn over the page. `origin` is the item's top left on screen.
    fn render_overlay(&mut self, _frame: &mut Frame, _origin: (u16, u16), _theme: &Theme) {}
    fn expanded(&self) -> bool {
        false
    }
    fn collapse(&mut self) {}
    /// The rows (top, height) of the widget that holds `stop`. Focus centers it.
    fn region(&self, _stop: usize) -> Option<(u16, u16)> {
        None
    }
    fn as_any(&mut self) -> &mut dyn Any;
}

pub struct Button {
    pub id: String,
    pub label: String,
    pub shown: bool,
    pub disabled: bool,
}

impl Button {
    pub fn new(id: &str, label: &str) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            shown: true,
            disabled: false,
        }
    }
}

pub enum Kind {
    /// `.section-title`: bold accent.
    Title(String),
    /// `.section-rule`.
    Rule,
    /// Muted text that wraps (`.section-desc`, `.field-help`).
    Desc(String),
    /// One bold line (`.field-label` on its own).
    Label(String),
    /// A line in one color variable, or the foreground.
    Text(String, Option<&'static str>),
    Field {
        label: String,
        input: Input,
        help: String,
    },
    Switch {
        label: String,
        on: bool,
        help: String,
    },
    /// A select with a label column, or across the width when `label` is None.
    Select {
        label: Option<String>,
        select: Select,
        help: String,
    },
    /// Buttons in a row with `gap` columns after each.
    Buttons {
        gap: u16,
        buttons: Vec<Button>,
    },
    /// An input across the width.
    Input(Input),
    TextArea(TextArea),
    Custom(Box<dyn Custom>),
}

pub struct Item {
    pub id: &'static str,
    pub kind: Kind,
    pub shown: bool,
    pub margin_top: u16,
    pub margin_bottom: u16,
    /// `max-width`, for a row narrower than the page.
    pub max_width: Option<u16>,
}

impl Item {
    pub fn new(id: &'static str, kind: Kind) -> Self {
        let (margin_top, margin_bottom) = match &kind {
            Kind::Title(_) => (1, 0),
            Kind::Rule
            | Kind::Desc(_)
            | Kind::Field { .. }
            | Kind::Switch { .. }
            | Kind::Select { label: Some(_), .. }
            | Kind::Buttons { .. }
            | Kind::TextArea(_) => (0, 1),
            _ => (0, 0),
        };
        Self {
            id,
            kind,
            shown: true,
            margin_top,
            margin_bottom,
            max_width: None,
        }
    }

    pub fn margins(mut self, top: u16, bottom: u16) -> Self {
        self.margin_top = top;
        self.margin_bottom = bottom;
        self
    }

    pub fn hidden(mut self, hidden: bool) -> Self {
        self.shown = !hidden;
        self
    }

    fn stops(&self) -> usize {
        match &self.kind {
            Kind::Field { .. }
            | Kind::Switch { .. }
            | Kind::Select { .. }
            | Kind::Input(_)
            | Kind::TextArea(_) => 1,
            Kind::Buttons { buttons, .. } => buttons
                .iter()
                .filter(|button| button.shown && !button.disabled)
                .count(),
            Kind::Custom(custom) => custom.stops(),
            _ => 0,
        }
    }

    fn height(&self, width: u16) -> u16 {
        let control = width.saturating_sub(LABEL_WIDTH);
        let help_rows = |help: &str| match help.is_empty() {
            true => 0,
            false => wrap_line(&Line::from(help.to_string()), control).len() as u16,
        };
        match &self.kind {
            Kind::Title(_) | Kind::Rule | Kind::Label(_) | Kind::Text(..) | Kind::Input(_) => 1,
            Kind::Buttons { .. } => 1,
            Kind::Desc(text) => wrap_line(&Line::from(text.clone()), width).len().max(1) as u16,
            Kind::Field { help, .. } | Kind::Switch { help, .. } => 1 + help_rows(help),
            Kind::Select { label, help, .. } => match label {
                Some(_) => 3 + help_rows(help),
                None => 3,
            },
            Kind::TextArea(area) => (area.line_count() as u16 + 2).clamp(HINTS_MIN, HINTS_MAX),
            Kind::Custom(custom) => custom.height(width),
        }
    }
}

/// One page: its items, the scroll offset, and the focused stop.
pub struct Page {
    pub items: Vec<Item>,
    pub offset: usize,
    /// (item index, stop within it).
    pub focus: Option<(usize, usize)>,
    /// `max-width` of the page column.
    pub max_width: u16,
    /// The last item fills the height and the page does not scroll (Appearance).
    pub fill_last: bool,
    /// Rows the window showed at the last draw.
    window: usize,
    /// The content height at the last layout.
    total: u16,
    /// Where each shown item sat at the last layout: (index, top, height).
    placed: Vec<(usize, u16, u16)>,
}

impl Page {
    pub fn new(items: Vec<Item>) -> Self {
        Self {
            items,
            offset: 0,
            focus: None,
            max_width: 120,
            fill_last: false,
            window: 1,
            total: 0,
            placed: Vec::new(),
        }
    }

    pub fn item(&self, id: &str) -> Option<&Item> {
        self.items.iter().find(|item| item.id == id)
    }

    pub fn item_mut(&mut self, id: &str) -> Option<&mut Item> {
        self.items.iter_mut().find(|item| item.id == id)
    }

    /// The focus stops in order: (item, stop).
    fn stops(&self) -> Vec<(usize, usize)> {
        self.items
            .iter()
            .enumerate()
            .filter(|(_, item)| item.shown)
            .flat_map(|(index, item)| (0..item.stops()).map(move |stop| (index, stop)))
            .collect()
    }

    pub fn first_stop(&self) -> Option<(usize, usize)> {
        self.stops().first().copied()
    }

    pub fn last_stop(&self) -> Option<(usize, usize)> {
        self.stops().last().copied()
    }

    /// Move the focus one stop. None when it leaves the page at either end.
    pub fn step_focus(&mut self, forward: bool) -> Option<(usize, usize)> {
        let stops = self.stops();
        let at = self
            .focus
            .and_then(|focus| stops.iter().position(|stop| *stop == focus));
        let next = match (at, forward) {
            (None, true) => stops.first().copied(),
            (None, false) => stops.last().copied(),
            (Some(at), true) => stops.get(at + 1).copied(),
            (Some(at), false) => at.checked_sub(1).and_then(|at| stops.get(at).copied()),
        };
        self.set_focus(next);
        next
    }

    pub fn set_focus(&mut self, focus: Option<(usize, usize)>) {
        self.close_selects();
        self.focus = focus;
        if let Some((index, stop)) = focus {
            match self.items.get_mut(index).map(|item| &mut item.kind) {
                // Textual selects an input's value when it gets the focus.
                Some(Kind::Input(input) | Kind::Field { input, .. }) => input.select_all(),
                Some(Kind::Custom(custom)) => custom.focused(stop),
                _ => {}
            }
            self.scroll_to(index, stop);
        }
    }

    pub fn close_selects(&mut self) {
        for item in &mut self.items {
            match &mut item.kind {
                Kind::Select { select, .. } => select.expanded = false,
                Kind::Custom(custom) => custom.collapse(),
                _ => {}
            }
        }
    }

    /// An open select list, if any.
    pub fn expanded(&self) -> bool {
        self.items.iter().any(|item| match &item.kind {
            Kind::Select { select, .. } => select.expanded,
            Kind::Custom(custom) => custom.expanded(),
            _ => false,
        })
    }

    /// Scroll the least that shows the item, as `scroll_to_widget`.
    /// Center a focused widget that is not in full view, as Textual's
    /// `scroll_to_center(origin_visible=True)`.
    fn scroll_to(&mut self, index: usize, stop: usize) {
        let Some(&(_, item_top, item_height)) = self.placed.iter().find(|(at, _, _)| *at == index)
        else {
            return;
        };
        let (top, height) = match &self.items[index].kind {
            // The control row, not its help text below.
            Kind::Select { .. } => (0, 3),
            Kind::Field { .. } | Kind::Switch { .. } | Kind::Input(_) | Kind::Buttons { .. } => {
                (0, 1)
            }
            Kind::Custom(custom) => custom.region(stop).unwrap_or((0, item_height)),
            _ => (0, item_height),
        };
        let first = usize::from(item_top + top);
        // `can_view_entire`: a widget in full view does not scroll the page.
        if first >= self.offset && first + usize::from(height) <= self.offset + self.window {
            return;
        }
        let top = first as f64;
        let window = self.window as f64;
        let center = top + f64::from(height) / 2.0;
        let delta = (center - (self.offset as f64 + window / 2.0) + 0.5).floor();
        let max_offset = f64::from(self.total.saturating_sub(self.window as u16));
        let mut offset = (self.offset as f64 + delta).clamp(0.0, max_offset);
        // `origin_visible`: the widget's first row stays in the window.
        if top < offset || top >= offset + window {
            offset = top.min(max_offset);
        }
        self.offset = offset as usize;
    }

    /// The column the items use inside a page of `width`: `padding: 1 2`,
    /// `max-width`, and the scrollbar when the page overflows.
    fn content_width(&self, width: u16, scrollbar: bool) -> u16 {
        let bar = if scrollbar { SCROLLBAR } else { 0 };
        width.min(self.max_width).saturating_sub(4 + bar)
    }

    /// Lay the shown items out top to bottom. Adjacent vertical margins collapse.
    /// Returns the content height. The padding stays outside: it never scrolls.
    fn layout(&mut self, width: u16) -> u16 {
        let mut placed = Vec::new();
        let mut y = 0u16;
        let mut gap_before = 0u16;
        let mut first = true;
        for (index, item) in self.items.iter().enumerate() {
            if !item.shown {
                continue;
            }
            let margin = match first {
                true => item.margin_top,
                false => gap_before.max(item.margin_top),
            };
            first = false;
            y += margin;
            let item_width = item.max_width.map_or(width, |max| max.min(width));
            let height = item.height(item_width);
            placed.push((index, y, height));
            y += height;
            gap_before = item.margin_bottom;
        }
        self.placed = placed;
        self.total = y + gap_before;
        self.total
    }

    /// Key for the focused item. None when the page does not use it.
    pub fn key(&mut self, key: &KeyEvent, name: &str) -> Option<Option<FormEvent>> {
        let (index, stop) = self.focus?;
        let item = self.items.get_mut(index)?;
        let id = item.id;
        let used = match &mut item.kind {
            Kind::Field { input, .. } | Kind::Input(input) => input.key(key, name).then_some(None),
            Kind::Switch { on, .. } => match name {
                "enter" | "space" => {
                    *on = !*on;
                    Some(None)
                }
                _ => None,
            },
            Kind::Select { select, .. } => match select.key(name) {
                SelectKey::Changed => Some(Some(FormEvent::Changed(id, select.value()))),
                SelectKey::Used => Some(None),
                SelectKey::Unused => None,
            },
            Kind::Buttons { buttons, .. } => match name {
                "enter" | "space" => {
                    let pressed = buttons
                        .iter()
                        .filter(|button| button.shown && !button.disabled)
                        .nth(stop)
                        .map(|button| FormEvent::Pressed(button.id.clone()));
                    Some(pressed)
                }
                _ => None,
            },
            Kind::TextArea(area) => area.key(key, name).then_some(None),
            Kind::Custom(custom) => {
                let mut moved = stop;
                let used = custom.key(&mut moved, key, name);
                if moved != stop {
                    let stops = custom.stops();
                    self.focus = match stops {
                        0 => None,
                        _ => Some((index, moved.min(stops - 1))),
                    };
                    if self.focus.is_none() {
                        self.step_focus(true);
                    }
                }
                used
            }
            _ => None,
        };
        if used.is_some() {
            return used;
        }
        self.scroll_key(name).then_some(None)
    }

    /// `VerticalScroll`'s own keys. True when the key scrolls.
    pub fn scroll_key(&mut self, name: &str) -> bool {
        let max_offset = usize::from(self.total).saturating_sub(self.window);
        match name {
            "up" => self.offset = self.offset.saturating_sub(1),
            "down" => self.offset = (self.offset + 1).min(max_offset),
            "pageup" => self.offset = self.offset.saturating_sub(self.window),
            "pagedown" => self.offset = (self.offset + self.window).min(max_offset),
            "home" => self.offset = 0,
            "end" => self.offset = max_offset,
            _ => return false,
        }
        true
    }

    pub fn paste(&mut self, text: &str) {
        let Some((index, _)) = self.focus else {
            return;
        };
        match self.items.get_mut(index).map(|item| &mut item.kind) {
            Some(Kind::Field { input, .. } | Kind::Input(input)) => input.paste(text),
            Some(Kind::TextArea(area)) => area.paste(text),
            Some(Kind::Custom(custom)) => {
                custom.paste(self.focus.map_or(0, |(_, stop)| stop), text)
            }
            _ => {}
        }
    }

    /// Draw the page into `area`. The open select's list is drawn last, over the rest.
    pub fn render(&mut self, frame: &mut Frame, area: Rect, theme: &Theme) {
        let background = theme.color("background");
        frame.render_widget(
            Paragraph::new("").style(Style::default().bg(background)),
            area,
        );
        let mut width = self.content_width(area.width, false);
        if width == 0 || area.height == 0 {
            return;
        }
        // `padding: 1 2`: the viewport and its scrollbar sit inside the padding.
        let window = area.height.saturating_sub(2);
        let mut total = self.layout(width);
        let overflow = !self.fill_last && total > window;
        if overflow {
            width = self.content_width(area.width, true);
            total = self.layout(width);
        }
        if self.fill_last {
            if let Some(last) = self.placed.last_mut() {
                last.2 = window.saturating_sub(last.1);
            }
            total = window;
        }
        self.window = usize::from(window);
        self.offset = self
            .offset
            .min(usize::from(total).saturating_sub(self.window));
        // Draw the whole column off screen, then copy the visible rows.
        let Ok(mut canvas) = ratatui::Terminal::new(TestBackend::new(width, total.max(1)));
        let mut overlay = None;
        let mut custom_overlay = None;
        let drawn = canvas.draw(|canvas_frame| {
            let full = canvas_frame.area();
            canvas_frame.render_widget(
                Paragraph::new("").style(Style::default().bg(background)),
                full,
            );
            for &(index, top, height) in &self.placed {
                let item = &mut self.items[index];
                let item_width = item.max_width.map_or(width, |max| max.min(width));
                let rect = Rect::new(0, top, item_width, height);
                let focus = self
                    .focus
                    .filter(|(at, _)| *at == index)
                    .map(|(_, stop)| stop);
                draw_item(canvas_frame, rect, item, focus, theme);
                match &item.kind {
                    Kind::Select { select, label, .. } if select.expanded => {
                        let x = if label.is_some() { LABEL_WIDTH } else { 0 };
                        overlay = Some((index, x, top + 3, item_width - x));
                    }
                    Kind::Custom(custom) if custom.expanded() => {
                        custom_overlay = Some((index, top))
                    }
                    _ => {}
                }
            }
        });
        if drawn.is_err() {
            return;
        }
        let source = canvas.backend().buffer().clone();
        let left = area.x + 2;
        let top = area.y + 1;
        let target = frame.buffer_mut();
        for row in 0..window {
            let from_y = self.offset as u16 + row;
            if from_y >= total {
                break;
            }
            for x in 0..width {
                if let Some(cell) = target.cell_mut((left + x, top + row)) {
                    *cell = source[(x, from_y)].clone();
                }
            }
        }
        let column = Rect::new(area.x, top, width + 2 + SCROLLBAR, window);
        if overflow {
            crate::tui::widgets::scroll::draw_scrollbar_columns(
                frame,
                column,
                SCROLLBAR,
                self.offset,
                usize::from(total),
                self.window,
                theme,
            );
        }
        if let Some((index, x, top, list_width)) = overlay {
            let y = (i32::from(area.y + 1) + i32::from(top) - self.offset as i32).max(0) as u16;
            let below = Rect::new(left + x, y, list_width, 1);
            let screen = frame.area();
            if let Kind::Select { select, .. } = &mut self.items[index].kind {
                select.render_overlay(frame, below, screen, "surface-lighten-1", theme);
            }
        }
        if let Some((index, top)) = custom_overlay {
            let y = (i32::from(area.y + 1) + i32::from(top) - self.offset as i32).max(0) as u16;
            if let Kind::Custom(custom) = &mut self.items[index].kind {
                custom.render_overlay(frame, (left, y), theme);
            }
        }
    }

    // --- Values by id, for the saved data ---

    pub fn input(&self, id: &str) -> String {
        match self.item(id).map(|item| &item.kind) {
            Some(Kind::Field { input, .. } | Kind::Input(input)) => input.value(),
            _ => String::new(),
        }
    }

    pub fn set_input(&mut self, id: &str, value: &str) {
        if let Some(Kind::Field { input, .. } | Kind::Input(input)) =
            self.item_mut(id).map(|item| &mut item.kind)
        {
            input.set_value(value);
        }
    }

    pub fn switch(&self, id: &str) -> bool {
        matches!(
            self.item(id).map(|item| &item.kind),
            Some(Kind::Switch { on: true, .. })
        )
    }

    pub fn set_switch(&mut self, id: &str, value: bool) {
        if let Some(Kind::Switch { on, .. }) = self.item_mut(id).map(|item| &mut item.kind) {
            *on = value;
        }
    }

    pub fn select(&self, id: &str) -> String {
        match self.item(id).map(|item| &item.kind) {
            Some(Kind::Select { select, .. }) => select.value(),
            _ => String::new(),
        }
    }

    pub fn select_mut(&mut self, id: &str) -> Option<&mut Select> {
        match self.item_mut(id).map(|item| &mut item.kind) {
            Some(Kind::Select { select, .. }) => Some(select),
            _ => None,
        }
    }

    pub fn text_area(&self, id: &str) -> String {
        match self.item(id).map(|item| &item.kind) {
            Some(Kind::TextArea(area)) => area.text(),
            _ => String::new(),
        }
    }

    pub fn text_area_mut(&mut self, id: &str) -> Option<&mut TextArea> {
        match self.item_mut(id).map(|item| &mut item.kind) {
            Some(Kind::TextArea(area)) => Some(area),
            _ => None,
        }
    }

    pub fn button_mut(&mut self, id: &str) -> Option<&mut Button> {
        self.items.iter_mut().find_map(|item| match &mut item.kind {
            Kind::Buttons { buttons, .. } => buttons.iter_mut().find(|button| button.id == id),
            _ => None,
        })
    }

    pub fn custom<T: 'static>(&mut self, id: &str) -> Option<&mut T> {
        match self.item_mut(id).map(|item| &mut item.kind) {
            Some(Kind::Custom(custom)) => custom.as_any().downcast_mut::<T>(),
            _ => None,
        }
    }

    pub fn set_text(&mut self, id: &str, value: &str) {
        if let Some(item) = self.item_mut(id) {
            match &mut item.kind {
                Kind::Desc(text) | Kind::Label(text) | Kind::Text(text, _) | Kind::Title(text) => {
                    *text = value.to_string();
                }
                _ => {}
            }
        }
    }

    pub fn show(&mut self, id: &str, shown: bool) {
        if let Some(item) = self.item_mut(id) {
            item.shown = shown;
        }
    }
}

fn line_at(frame: &mut Frame, area: Rect, line: Line<'static>) {
    frame.render_widget(Paragraph::new(line), area);
}

fn muted(theme: &Theme) -> Style {
    Style::default()
        .fg(theme.on("text-muted", "background"))
        .bg(theme.color("background"))
}

fn draw_wrapped(frame: &mut Frame, area: Rect, text: &str, style: Style) {
    let lines = wrap_line(&Line::styled(text.to_string(), style), area.width);
    for (row, line) in lines.into_iter().enumerate().take(usize::from(area.height)) {
        line_at(
            frame,
            Rect::new(area.x, area.y + row as u16, area.width, 1),
            line,
        );
    }
}

fn draw_label(frame: &mut Frame, area: Rect, label: &str, theme: &Theme) {
    let style = Style::default()
        .fg(theme.on("foreground", "background"))
        .bg(theme.color("background"))
        .add_modifier(Modifier::BOLD);
    let rect = Rect::new(area.x, area.y, LABEL_WIDTH.min(area.width), 1);
    line_at(frame, rect, Line::styled(label.to_string(), style));
}

fn draw_item(frame: &mut Frame, area: Rect, item: &mut Item, focus: Option<usize>, theme: &Theme) {
    let control = Rect::new(
        area.x + LABEL_WIDTH,
        area.y,
        area.width.saturating_sub(LABEL_WIDTH),
        area.height,
    );
    let help_area = |rows: u16| {
        Rect::new(
            control.x,
            control.y + rows,
            control.width,
            area.height.saturating_sub(rows),
        )
    };
    match &mut item.kind {
        Kind::Title(text) => {
            let style = Style::default()
                .fg(theme.on("accent", "background"))
                .bg(theme.color("background"))
                .add_modifier(Modifier::BOLD);
            line_at(frame, area, Line::styled(text.clone(), style));
        }
        Kind::Rule => {
            let style = Style::default()
                .fg(theme.color("surface-lighten-2"))
                .bg(theme.color("background"));
            line_at(
                frame,
                area,
                Line::styled("─".repeat(usize::from(area.width)), style),
            );
        }
        Kind::Desc(text) => draw_wrapped(frame, area, text, muted(theme)),
        Kind::Label(text) => {
            draw_label(frame, Rect::new(area.x, area.y, area.width, 1), text, theme)
        }
        Kind::Text(text, color) => {
            let fg = theme.on(color.unwrap_or("foreground"), "background");
            let style = Style::default().fg(fg).bg(theme.color("background"));
            line_at(frame, area, Line::styled(text.clone(), style));
        }
        Kind::Field { label, input, help } => {
            draw_label(frame, area, label, theme);
            input.render_compact(
                frame,
                Rect::new(control.x, control.y, control.width, 1),
                focus.is_some(),
                theme,
            );
            draw_wrapped(frame, help_area(1), help, muted(theme));
        }
        Kind::Switch { label, on, help } => {
            draw_label(frame, area, label, theme);
            draw_switch(
                frame,
                Rect::new(control.x, control.y, control.width, 1),
                *on,
                focus.is_some(),
                theme,
            );
            draw_wrapped(frame, help_area(1), help, muted(theme));
        }
        Kind::Select {
            label,
            select,
            help,
        } => {
            let box_area = match label {
                Some(text) => {
                    draw_label(frame, area, text, theme);
                    Rect::new(control.x, control.y, control.width, 3)
                }
                None => Rect::new(area.x, area.y, area.width, 3),
            };
            select.render(frame, box_area, focus.is_some(), "surface-lighten-1", theme);
            if label.is_some() {
                draw_wrapped(frame, help_area(3), help, muted(theme));
            }
        }
        Kind::Buttons { gap, buttons } => {
            let mut x = area.x;
            let mut stop = 0;
            for button in buttons.iter().filter(|button| button.shown) {
                let width = compact_button_width(&button.label);
                if x + width > area.right() {
                    break;
                }
                let focused = !button.disabled && focus == Some(stop);
                draw_compact_button(
                    frame,
                    Rect::new(x, area.y, width, 1),
                    &button.label,
                    focused,
                    button.disabled,
                    theme,
                );
                if !button.disabled {
                    stop += 1;
                }
                x += width + *gap;
            }
        }
        Kind::Input(input) => input.render_compact(frame, area, focus.is_some(), theme),
        Kind::TextArea(text_area) => {
            let border = if focus.is_some() {
                "border"
            } else {
                "border-blurred"
            };
            text_area.render(frame, area, focus.is_some(), "", border, theme);
        }
        Kind::Custom(custom) => custom.render(frame, area, focus, theme),
    }
}

/// A line of spans at `area`, for custom items.
pub fn spans_at(frame: &mut Frame, area: Rect, spans: Vec<Span<'static>>) {
    line_at(frame, area, Line::from(spans));
}

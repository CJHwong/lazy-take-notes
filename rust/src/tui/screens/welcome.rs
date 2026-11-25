//! The welcome picker: choose a mode when `lazy-take-notes` runs with no command.

use crossterm::event::KeyEvent;
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use crate::tui::host::{Action, Ctx, MouseInput, MouseKind, Screen};
use crate::tui::keys::{Binding, SCREEN_BASE, action_for, bind};
use crate::tui::system::saved_theme;
use crate::tui::widgets::list::ListCursor;
use crate::tui::widgets::{bar, panel};

pub const WELCOME_BINDINGS: [Binding; 7] = [
    SCREEN_BASE[0],
    SCREEN_BASE[1],
    SCREEN_BASE[2],
    SCREEN_BASE[3],
    bind("escape", "cancel", "Cancel").priority(),
    bind("q", "cancel", "Cancel"),
    bind("enter", "select_mode", "Select").priority(),
];

pub use crate::app::modes::MODES;

/// The mode the picker popped with.
pub struct ChosenMode(pub &'static str);

const PIG: Color = Color::Rgb(0xE8, 0xA8, 0x20);
const TEAL: Color = Color::Rgb(0x4A, 0x9E, 0x9E);
const OUTLINE: Color = Color::Rgb(0x2A, 0x18, 0x18);
const SNOUT: Color = Color::Rgb(0xC4, 0x7A, 0x10);

/// The banner pixels: two rows per text row. `.` is the screen background.
const PIXELS: [&str; 12] = [
    "..TTTTTTTTTTTTTTTT..",
    ".TT.DDDDDDDDDDDD.TT.",
    "TT.DPPPPPPPPPPPPD.TT",
    "TT.DPDDPPPPPPDDPD.TT",
    "TT.DPPPPPPPPPPPPD.TT",
    "TT.DPPPSSSSSSPPPD.TT",
    ".T.DPPPSDSSDSPPPD.T.",
    "...DPPPSSSSSSPPPD...",
    "...DPPPPPPPPPPPPD...",
    "....DPPPPPPPPPPD....",
    ".....DPPPPPPPPD.....",
    "......DDDDDDDD......",
];

fn pixel(code: u8) -> Option<Color> {
    match code {
        b'P' => Some(PIG),
        b'T' => Some(TEAL),
        b'D' => Some(OUTLINE),
        b'S' => Some(SNOUT),
        _ => None,
    }
}

/// One text cell from a top and a bottom pixel, as Python's `_cell`.
fn cell(top: Option<Color>, bottom: Option<Color>) -> Span<'static> {
    match (top, bottom) {
        (None, None) => Span::raw(" "),
        (Some(top), None) => Span::styled("▀", Style::default().fg(top)),
        (None, Some(bottom)) => Span::styled("▄", Style::default().fg(bottom)),
        (Some(top), Some(bottom)) if top == bottom => Span::styled("█", Style::default().fg(top)),
        (Some(top), Some(bottom)) => Span::styled("▀", Style::default().fg(top).bg(bottom)),
    }
}

fn banner() -> Vec<Line<'static>> {
    let mut lines: Vec<Line<'static>> = PIXELS
        .chunks(2)
        .map(|pair| {
            let (top, bottom) = (pair[0].as_bytes(), pair[1].as_bytes());
            Line::from(
                (0..top.len())
                    .map(|x| cell(pixel(top[x]), pixel(bottom[x])))
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    let bold = Style::default().add_modifier(Modifier::BOLD);
    lines.push(Line::from(vec![
        Span::styled("lazy-", bold.fg(PIG)),
        Span::styled("take-notes", bold.fg(TEAL)),
    ]));
    lines
}

pub struct WelcomePicker {
    list: ListCursor,
}

impl WelcomePicker {
    /// Open with the highlight on `last_mode`, so returning from a mode keeps its place.
    pub fn new(last_mode: Option<&str>) -> Self {
        let mut list = ListCursor::default();
        list.reset(MODES.len());
        let index = MODES
            .iter()
            .position(|(mode, _, _)| Some(*mode) == last_mode)
            .unwrap_or(0);
        list.select(index);
        Self { list }
    }
}

impl Screen for WelcomePicker {
    fn mounted(&mut self, ctx: &mut Ctx) -> Action {
        // A theme saved in Settings shows from here on, as Python's `load_theme()`.
        ctx.set_theme(&saved_theme());
        Action::None
    }

    fn render(&mut self, frame: &mut Frame, area: Rect, ctx: &Ctx) {
        let theme = &ctx.theme;
        let background = theme.color("background");
        frame.render_widget(
            Paragraph::new("").style(Style::default().bg(background)),
            area,
        );
        let footer = Rect::new(
            area.x,
            area.bottom().saturating_sub(1),
            area.width,
            1.min(area.height),
        );
        bar(
            frame,
            footer,
            "[Enter] Select  [↑/↓] Navigate  [Esc] Cancel",
            "surface",
            "text-muted",
            false,
            theme,
        );
        crate::tui::widgets::hint_targets(frame, footer, ctx);
        // `align: center middle`: the banner (with its top padding) and the list
        // (with `margin: 1 2`) sit together in the middle of the rest.
        let body = Rect::new(area.x, area.y, area.width, area.height.saturating_sub(1));
        let banner = banner();
        let list_height = (MODES.len() as u16 + 2).min(8);
        let height = 1 + banner.len() as u16 + 1 + list_height + 1;
        let top = body.y + body.height.saturating_sub(height) / 2;
        let text_style = Style::default()
            .bg(background)
            .fg(theme.color("foreground"));
        for (row, line) in banner.into_iter().enumerate() {
            let y = top + 1 + row as u16;
            if y >= body.bottom() {
                break;
            }
            let rect = Rect::new(body.x, y, body.width, 1);
            let line = crate::tui::text::center(line, body.width);
            frame.render_widget(Paragraph::new(line).style(text_style), rect);
        }
        let list_y = top + 1 + 7 + 1;
        if list_y + list_height > body.bottom() || body.width < 6 {
            return;
        }
        let list_width = (body.width - 4).min(64);
        let list_area = Rect::new(
            body.x + (body.width - list_width) / 2,
            list_y,
            list_width,
            list_height,
        );
        let surface = background;
        let inner = panel(frame, list_area, "", "primary", surface, theme);
        ctx.hit(inner, "list", 0);
        let dim = Style::default().add_modifier(Modifier::DIM);
        let rows: Vec<Line<'static>> = MODES
            .iter()
            .map(|(_, label, description)| {
                Line::from(vec![
                    Span::styled(*label, Style::default().add_modifier(Modifier::BOLD)),
                    Span::raw("  "),
                    Span::styled(*description, dim),
                ])
            })
            .collect();
        self.list.render(frame, inner, &rows, true, surface, theme);
    }

    /// A click opens the mode under the pointer, as a menu does.
    fn mouse(&mut self, input: &MouseInput, ctx: &mut Ctx) -> Action {
        match input.kind {
            MouseKind::Click if input.on("list").is_some() => match self.list.click(input.row) {
                Some((index, _)) => Action::Pop(Some(Box::new(ChosenMode(MODES[index].0)))),
                None => Action::None,
            },
            _ => match input.wheel_key() {
                Some((key, name)) => self.key(&key, name, ctx),
                None => Action::None,
            },
        }
    }

    fn key(&mut self, _key: &KeyEvent, name: &str, _ctx: &mut Ctx) -> Action {
        match action_for(&WELCOME_BINDINGS, name) {
            Some("cancel") => return Action::Pop(None),
            Some("select_mode") => {
                return match self.list.index {
                    Some(index) => Action::Pop(Some(Box::new(ChosenMode(MODES[index].0)))),
                    None => Action::None,
                };
            }
            _ => {}
        }
        self.list.key(name);
        Action::None
    }
}

#[cfg(test)]
#[path = "welcome_tests.rs"]
mod tests;

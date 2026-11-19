//! The Models page parts that are more than one widget: the speech model rows,
//! the language overrides, and the speaker assets row.

use std::any::Any;
use std::collections::HashMap;

use crossterm::event::KeyEvent;
use indexmap::IndexMap;
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::Span;

use super::form::{Custom, FormEvent, spans_at};
use crate::adapters::model_catalog::{asset_for_model, expand_model_alias};
use crate::app::download_progress::{describe, megabytes};
use crate::domain::model_asset::{AssetProgress, AssetState, AssetStatus, ModelAsset};
use crate::tui::text::{str_width, truncate};
use crate::tui::theme::Theme;
use crate::tui::widgets::input::Input;
use crate::tui::widgets::select::{Select, SelectKey, compact_button_width, draw_compact_button};

/// The canonical `hf://` URI of a cataloged asset.
pub fn model_uri(asset: &ModelAsset) -> String {
    format!("hf://{}/{}", asset.hf_repo, asset.entry)
}

fn status_text(status: AssetStatus) -> &'static str {
    match status {
        AssetStatus::Ready => "Ready",
        AssetStatus::Missing => "Missing",
        AssetStatus::Downloading | AssetStatus::Busy => "Downloading",
        AssetStatus::Corrupt => "Damaged",
        AssetStatus::Unsupported => "Unsupported",
    }
}

fn status_color(status: AssetStatus) -> Option<&'static str> {
    match status {
        AssetStatus::Ready => Some("success"),
        AssetStatus::Missing => Some("warning"),
        AssetStatus::Corrupt => Some("error"),
        AssetStatus::Downloading | AssetStatus::Busy => Some("accent"),
        AssetStatus::Unsupported => None,
    }
}

fn text_style(theme: &Theme, color: Option<&str>) -> Style {
    Style::default()
        .fg(theme.on(color.unwrap_or("foreground"), "background"))
        .bg(theme.color("background"))
}

/// One cell run of `width` columns: the text, cut or padded.
fn cell(text: &str, width: usize, style: Style) -> Span<'static> {
    let shown = truncate(text, width);
    let pad = width.saturating_sub(str_width(&shown));
    Span::styled(format!("{shown}{}", " ".repeat(pad)), style)
}

/// Draw buttons left to right in the one-row `row`, each after `gap` columns.
/// `focus` is the index among them. Returns where each drawn button sits.
fn draw_buttons(
    frame: &mut Frame,
    row: Rect,
    gap: u16,
    labels: &[&str],
    focus: Option<usize>,
    width: Option<u16>,
    theme: &Theme,
) -> Vec<Rect> {
    let (mut x, y, right) = (row.x, row.y, row.right());
    let mut rects = Vec::new();
    for (index, label) in labels.iter().enumerate() {
        x += gap;
        let w = width.unwrap_or_else(|| compact_button_width(label));
        if x + w > right {
            break;
        }
        let rect = Rect::new(x, y, w, 1);
        draw_compact_button(frame, rect, label, focus == Some(index), false, theme);
        rects.push(rect);
        x += w;
    }
    rects
}

/// Textual's `ProgressBar(show_eta=False)`: a bar and the percentage.
fn draw_progress(frame: &mut Frame, area: Rect, done: u64, total: u64, theme: &Theme) {
    let bar_width = usize::from(area.width.saturating_sub(5).min(32));
    let ratio = if total == 0 {
        0.0
    } else {
        (done as f64 / total as f64).min(1.0)
    };
    let filled = (ratio * bar_width as f64).round() as usize;
    let background = theme.color("background");
    let on = Style::default()
        .fg(theme.on("warning", "background"))
        .bg(background);
    let off = Style::default()
        .fg(theme.tinted("background", "foreground", 0.1))
        .bg(background);
    let percent = format!("{:>4}%", (ratio * 100.0) as u32);
    spans_at(
        frame,
        area,
        vec![
            Span::styled("━".repeat(filled), on),
            Span::styled("━".repeat(bar_width - filled), off),
            Span::styled(percent, text_style(theme, None)),
        ],
    );
}

#[derive(Clone)]
struct RowState {
    status: AssetStatus,
    progress: Option<AssetProgress>,
    /// Bytes kept after a cancel; the status then reads "Paused".
    kept: u64,
    paused: bool,
}

/// The speech model rows. The marked row is the default model.
pub struct ModelChooser {
    assets: Vec<ModelAsset>,
    states: HashMap<&'static str, RowState>,
    selected: Option<&'static str>,
    /// Each focus stop at the last draw.
    hits: Vec<(Rect, usize)>,
}

impl ModelChooser {
    pub fn new(assets: Vec<ModelAsset>, selected_uri: &str) -> Self {
        let selected = assets
            .iter()
            .find(|asset| model_uri(asset) == selected_uri)
            .map(|asset| asset.id);
        let states = assets
            .iter()
            .map(|asset| {
                (
                    asset.id,
                    RowState {
                        status: AssetStatus::Missing,
                        progress: None,
                        kept: 0,
                        paused: false,
                    },
                )
            })
            .collect();
        Self {
            assets,
            states,
            selected,
            hits: Vec::new(),
        }
    }

    pub fn selected_uri(&self) -> String {
        self.assets
            .iter()
            .find(|asset| Some(asset.id) == self.selected)
            .map(model_uri)
            .unwrap_or_default()
    }

    pub fn select_by_uri(&mut self, uri: &str) {
        if let Some(asset) = self.assets.iter().find(|asset| model_uri(asset) == uri) {
            self.selected = Some(asset.id);
        }
    }

    pub fn set_state(&mut self, state: &AssetState) {
        if let Some(row) = self.states.get_mut(state.asset.id) {
            row.status = state.status;
            row.paused = false;
            if state.status != AssetStatus::Downloading {
                row.progress = None;
            }
        }
    }

    pub fn mark_downloading(&mut self, asset_id: &str) {
        if let Some(row) = self.states.get_mut(asset_id) {
            row.kept = 0;
            row.status = AssetStatus::Downloading;
            row.progress = None;
            row.paused = false;
        }
    }

    pub fn set_progress(&mut self, asset_id: &str, progress: AssetProgress) {
        let size = self
            .assets
            .iter()
            .find(|a| a.id == asset_id)
            .map_or(1, |a| a.size.max(1));
        if let Some(row) = self.states.get_mut(asset_id) {
            let total = if progress.total > 0 {
                progress.total
            } else {
                size
            };
            row.progress = Some(AssetProgress { total, ..progress });
            row.kept = progress.downloaded;
        }
    }

    /// After Cancel: the `.part` stays, and Download continues from it.
    pub fn mark_paused(&mut self, asset_id: &str) {
        if let Some(row) = self.states.get_mut(asset_id) {
            row.paused = row.kept > 0;
        }
    }

    /// The buttons a row shows, by its state.
    fn actions(&self, asset: &ModelAsset) -> Vec<(&'static str, &'static str)> {
        let status = self.states[asset.id].status;
        let mut actions = Vec::new();
        if matches!(
            status,
            AssetStatus::Missing | AssetStatus::Corrupt | AssetStatus::Busy
        ) {
            actions.push(("download", "Download"));
        }
        if matches!(status, AssetStatus::Ready | AssetStatus::Corrupt) {
            actions.push(("remove", "Remove"));
        }
        if status == AssetStatus::Downloading {
            actions.push(("cancel", "Cancel"));
        }
        actions
    }

    /// The detail under a row: the download's progress, or why it waits.
    fn detail(&self, asset: &ModelAsset) -> String {
        let row = &self.states[asset.id];
        if row.paused {
            format!(
                "{} kept. Download continues from there.",
                megabytes(row.kept)
            )
        } else if row.status == AssetStatus::Busy {
            "In another window. Press Download to follow it.".into()
        } else {
            row.progress.as_ref().map(describe).unwrap_or_default()
        }
    }

    /// Rows a model takes: its line, and a detail line when it has one.
    fn rows_of(&self, asset: &ModelAsset) -> u16 {
        let row = &self.states[asset.id];
        let detail = row.status == AssetStatus::Downloading || !self.detail(asset).is_empty();
        1 + u16::from(detail)
    }

    /// The first row of each model.
    fn tops(&self) -> Vec<u16> {
        let mut top = 0;
        self.assets
            .iter()
            .map(|asset| {
                let at = top;
                top += self.rows_of(asset);
                at
            })
            .collect()
    }

    /// The row and the stop within it: 0 is the row itself, then its buttons.
    fn locate(&self, stop: usize) -> Option<(usize, usize)> {
        let mut first = 0;
        for (index, asset) in self.assets.iter().enumerate() {
            let count = 1 + self.actions(asset).len();
            if stop < first + count {
                return Some((index, stop - first));
            }
            first += count;
        }
        None
    }
}

impl Custom for ModelChooser {
    fn height(&self, _width: u16) -> u16 {
        self.assets.iter().map(|asset| self.rows_of(asset)).sum()
    }

    fn stops(&self) -> usize {
        self.assets
            .iter()
            .map(|asset| 1 + self.actions(asset).len())
            .sum()
    }

    fn region(&self, stop: usize) -> Option<(u16, u16)> {
        self.locate(stop).map(|(row, _)| (self.tops()[row], 1))
    }

    fn hits(&self) -> Vec<(Rect, usize)> {
        self.hits.clone()
    }

    fn render(&mut self, frame: &mut Frame, area: Rect, focus: Option<usize>, theme: &Theme) {
        let focus = focus.and_then(|stop| self.locate(stop));
        let muted = text_style(theme, Some("text-muted"));
        self.hits.clear();
        let mut first_stop = 0;
        let tops = self.tops();
        for (index, asset) in self.assets.iter().enumerate() {
            let y = area.y + tops[index];
            let row_stop = first_stop;
            first_stop += 1 + self.actions(asset).len();
            if y >= area.bottom() {
                break;
            }
            let row = &self.states[asset.id];
            let mark = if Some(asset.id) == self.selected {
                "●"
            } else {
                "○"
            };
            let mut label_style = text_style(theme, None).add_modifier(Modifier::BOLD);
            if focus == Some((index, 0)) {
                // The row takes the focus; Enter or a click marks it.
                label_style = label_style.add_modifier(Modifier::REVERSED);
            }
            let status = match row.paused {
                true => "Paused",
                false => status_text(row.status),
            };
            let line = vec![
                cell(mark, 2, text_style(theme, Some("accent"))),
                cell(asset.label, 30, label_style),
                cell(&megabytes(asset.size), 9, muted),
                cell(status, 12, text_style(theme, status_color(row.status))),
            ];
            spans_at(frame, Rect::new(area.x, y, area.width, 1), line);
            self.hits
                .push((Rect::new(area.x, y, 53.min(area.width), 1), row_stop));
            let labels: Vec<&str> = self
                .actions(asset)
                .iter()
                .map(|(_, label)| *label)
                .collect();
            let button_focus = match focus {
                Some((at, stop)) if at == index && stop > 0 => Some(stop - 1),
                _ => None,
            };
            let buttons = draw_buttons(
                frame,
                Rect::new(area.x + 53, y, area.width.saturating_sub(53), 1),
                1,
                &labels,
                button_focus,
                None,
                theme,
            );
            for (button, rect) in buttons.into_iter().enumerate() {
                self.hits.push((rect, row_stop + 1 + button));
            }
            // The progress line: `margin-left: 2`, a 34-column bar, then the detail.
            let line_y = y + 1;
            if self.rows_of(asset) < 2 || line_y >= area.bottom() {
                continue;
            }
            let detail = self.detail(asset);
            let mut x = area.x + 2;
            if row.status == AssetStatus::Downloading {
                let (done, total) = row
                    .progress
                    .map_or((0, asset.size.max(1)), |p| (p.downloaded, p.total));
                draw_progress(
                    frame,
                    Rect::new(x, line_y, 34.min(area.width - 2), 1),
                    done,
                    total,
                    theme,
                );
                x += 34;
            }
            if x < area.right() {
                spans_at(
                    frame,
                    Rect::new(x, line_y, area.right() - x, 1),
                    vec![Span::styled(detail, muted)],
                );
            }
        }
    }

    fn key(&mut self, stop: &mut usize, _key: &KeyEvent, name: &str) -> Option<Option<FormEvent>> {
        if !matches!(name, "enter" | "space") {
            return None;
        }
        let (index, at) = self.locate(*stop)?;
        let asset = self.assets[index].clone();
        if at == 0 {
            self.selected = Some(asset.id);
            return Some(Some(FormEvent::Custom(
                "model-chosen",
                asset.id.to_string(),
            )));
        }
        let (action, _) = self.actions(&asset)[at - 1];
        Some(Some(FormEvent::Pressed(format!("{action}-{}", asset.id))))
    }

    fn as_any(&mut self) -> &mut dyn Any {
        self
    }
}

/// One locale override: the locale, its model, and Remove.
struct LanguageRow {
    locale: Input,
    model: Select,
}

/// `LanguageModelsEditor`: rows, then Add language.
pub struct LanguageEditor {
    rows: Vec<LanguageRow>,
    assets: Vec<ModelAsset>,
    /// The model box width at the last draw; the open list takes the same.
    model_width: u16,
    /// Each focus stop at the last draw.
    hits: Vec<(Rect, usize)>,
}

/// Focus stops in a row: locale, model, Remove.
const ROW_STOPS: usize = 3;

impl LanguageEditor {
    pub fn new(models: &[(String, String)], assets: Vec<ModelAsset>) -> Self {
        let mut editor = Self {
            rows: Vec::new(),
            assets,
            model_width: 0,
            hits: Vec::new(),
        };
        editor.reload(models);
        editor
    }

    pub fn reload(&mut self, models: &[(String, String)]) {
        self.rows = models
            .iter()
            .map(|(locale, uri)| self.make_row(locale, uri))
            .collect();
    }

    fn make_row(&self, locale: &str, uri: &str) -> LanguageRow {
        let uri = match uri.is_empty() {
            true => String::new(),
            false => asset_for_model(&expand_model_alias(uri), &self.assets)
                .map(model_uri)
                .unwrap_or_else(|| uri.to_string()),
        };
        let mut choices: Vec<(String, String)> = self
            .assets
            .iter()
            .map(|asset| (asset.label.to_string(), model_uri(asset)))
            .collect();
        if !uri.is_empty() && !choices.iter().any(|(_, value)| *value == uri) {
            choices.push((uri.clone(), uri.clone()));
        }
        let value = match uri.is_empty() {
            true => choices.first().map(|(_, v)| v.clone()).unwrap_or_default(),
            false => uri,
        };
        let mut input = Input::new("zh");
        input.compact = Some(("surface-lighten-1", "surface-lighten-2"));
        input.set_value(locale);
        LanguageRow {
            locale: input,
            model: Select::new(choices, &value),
        }
    }

    /// The overrides in the form, in row order. A row without a locale or a model is left out.
    pub fn values(&self) -> IndexMap<String, String> {
        let mut values = IndexMap::new();
        for row in &self.rows {
            let locale = row.locale.value().trim().to_string();
            let model = row.model.value();
            if !locale.is_empty() && !model.is_empty() {
                values.insert(locale, model);
            }
        }
        values
    }

    /// Remove the row for `locale`, as its Remove button does.
    #[cfg(test)]
    pub fn remove(&mut self, locale: &str) {
        self.rows.retain(|row| row.locale.value() != locale);
    }

    fn open_row(&self) -> Option<usize> {
        self.rows.iter().position(|row| row.model.expanded)
    }
}

impl Custom for LanguageEditor {
    fn height(&self, _width: u16) -> u16 {
        // Rows of 3, then a blank row and Add language.
        match self.rows.len() as u16 {
            0 => 2,
            rows => rows * 3 + 2,
        }
    }

    fn stops(&self) -> usize {
        self.rows.len() * ROW_STOPS + 1
    }

    fn region(&self, stop: usize) -> Option<(u16, u16)> {
        let row = (stop / ROW_STOPS) as u16;
        match stop / ROW_STOPS < self.rows.len() {
            true => Some((row * 3, 3)),
            false => Some((row * 3 + 1, 1)),
        }
    }

    fn hits(&self) -> Vec<(Rect, usize)> {
        self.hits.clone()
    }

    fn open_select(&self) -> Option<(Vec<(Rect, usize)>, usize)> {
        let select = &self.rows[self.open_row()?].model;
        Some((select.drawn_rows(), select.options().len()))
    }

    fn render(&mut self, frame: &mut Frame, area: Rect, focus: Option<usize>, theme: &Theme) {
        let remove_width = compact_button_width("Remove");
        self.model_width = area.width.saturating_sub(14 + 1 + remove_width);
        self.hits.clear();
        for (index, row) in self.rows.iter().enumerate() {
            let y = area.y + index as u16 * 3;
            if y + 3 > area.bottom() {
                break;
            }
            let first = index * ROW_STOPS;
            let at = |offset: usize| focus == Some(first + offset);
            // Locale: 14 wide; model: the rest; Remove after one column. The
            // one-row parts sit on the model box's middle row.
            let locale = Rect::new(area.x, y + 1, 13, 1);
            row.locale.render_compact(frame, locale, at(0), theme);
            let model_width = self.model_width;
            let model = Rect::new(area.x + 14, y, model_width, 3);
            row.model.render(frame, model, at(1), "background", theme);
            let remove = Rect::new(area.x + 14 + model_width + 1, y + 1, remove_width, 1);
            draw_compact_button(frame, remove, "Remove", at(2), false, theme);
            self.hits
                .extend([(locale, first), (model, first + 1), (remove, first + 2)]);
        }
        let y = match self.rows.len() as u16 {
            0 => area.y + 1,
            rows => area.y + rows * 3 + 1,
        };
        if y < area.bottom() {
            let width = compact_button_width("Add language");
            let stop = self.rows.len() * ROW_STOPS;
            let rect = Rect::new(area.x, y, width, 1);
            draw_compact_button(
                frame,
                rect,
                "Add language",
                focus == Some(stop),
                false,
                theme,
            );
            self.hits.push((rect, stop));
        }
    }

    fn render_overlay(&mut self, frame: &mut Frame, origin: (u16, u16), theme: &Theme) {
        let Some(index) = self.open_row() else {
            return;
        };
        let screen = frame.area();
        let below = Rect::new(
            origin.0 + 14,
            origin.1 + index as u16 * 3 + 3,
            self.model_width,
            1,
        );
        self.rows[index]
            .model
            .render_overlay(frame, below, screen, "background", theme);
    }

    fn expanded(&self) -> bool {
        self.open_row().is_some()
    }

    fn collapse(&mut self) {
        for row in &mut self.rows {
            row.model.expanded = false;
        }
    }

    fn focused(&mut self, stop: usize) {
        if let Some(row) = self.rows.get_mut(stop / ROW_STOPS)
            && stop.is_multiple_of(ROW_STOPS)
        {
            row.locale.select_all();
        }
    }

    fn paste(&mut self, stop: usize, text: &str) {
        if let Some(row) = self.rows.get_mut(stop / ROW_STOPS)
            && stop.is_multiple_of(ROW_STOPS)
        {
            row.locale.paste(text);
        }
    }

    fn key(&mut self, stop: &mut usize, key: &KeyEvent, name: &str) -> Option<Option<FormEvent>> {
        let index = *stop / ROW_STOPS;
        if index == self.rows.len() {
            if !matches!(name, "enter" | "space") {
                return None;
            }
            let row = self.make_row("", "");
            self.rows.push(row);
            // The focus stays on Add language, which moved down one row.
            *stop += ROW_STOPS;
            return Some(None);
        }
        let row = &mut self.rows[index];
        match *stop % ROW_STOPS {
            0 => row.locale.key(key, name).then_some(None),
            1 => match row.model.key(name) {
                SelectKey::Unused => None,
                _ => Some(None),
            },
            _ => {
                if !matches!(name, "enter" | "space") {
                    return None;
                }
                self.rows.remove(index);
                // The focus goes to the next row's locale, or Add language.
                *stop = index * ROW_STOPS;
                Some(None)
            }
        }
    }

    fn as_any(&mut self) -> &mut dyn Any {
        self
    }
}

/// The speaker runtime and model, installed and removed together.
#[derive(Default)]
pub struct SpeakerState {
    pub status: String,
    /// Which buttons show: Install, Remove, Cancel.
    pub install: bool,
    pub remove: bool,
    pub cancel: bool,
    /// Each button at the last draw.
    hits: Vec<(Rect, usize)>,
}

impl SpeakerState {
    fn buttons(&self) -> Vec<(&'static str, &'static str)> {
        let mut list = Vec::new();
        if self.install {
            list.push(("speaker-install", "Install"));
        }
        if self.remove {
            list.push(("speaker-remove", "Remove"));
        }
        if self.cancel {
            list.push(("speaker-cancel", "Cancel"));
        }
        list
    }
}

impl Custom for SpeakerState {
    fn height(&self, _width: u16) -> u16 {
        1
    }

    fn stops(&self) -> usize {
        self.buttons().len()
    }

    fn render(&mut self, frame: &mut Frame, area: Rect, focus: Option<usize>, theme: &Theme) {
        let status = match self.status.is_empty() {
            true => "Checking...",
            false => self.status.as_str(),
        };
        spans_at(
            frame,
            area,
            vec![
                cell(
                    "Runtime and model",
                    18,
                    text_style(theme, None).add_modifier(Modifier::BOLD),
                ),
                cell(status, 26, text_style(theme, None)),
            ],
        );
        let labels: Vec<&str> = self.buttons().iter().map(|(_, label)| *label).collect();
        let buttons = draw_buttons(
            frame,
            Rect::new(area.x + 44, area.y, area.width.saturating_sub(44), 1),
            0,
            &labels,
            focus,
            Some(12),
            theme,
        );
        self.hits = buttons
            .into_iter()
            .enumerate()
            .map(|(stop, rect)| (rect, stop))
            .collect();
    }

    fn hits(&self) -> Vec<(Rect, usize)> {
        self.hits.clone()
    }

    fn key(&mut self, stop: &mut usize, _key: &KeyEvent, name: &str) -> Option<Option<FormEvent>> {
        if !matches!(name, "enter" | "space") {
            return None;
        }
        let (id, _) = *self.buttons().get(*stop)?;
        Some(Some(FormEvent::Pressed(id.into())))
    }

    fn as_any(&mut self) -> &mut dyn Any {
        self
    }
}

//! Browse the filesystem and pick an audio file (Python's `FilePicker`).

use std::path::{Path, PathBuf};

use crossterm::event::KeyEvent;
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};

use crate::adapters::os_error::OsError;
use crate::tui::host::{Action, Ctx, MouseInput, Msg, Screen};
use crate::tui::keys::{Binding, bind};
use crate::tui::screens::picker::{PICKER_BINDINGS, Picker, PickerKey};

pub const FILE_PICKER_BINDINGS: [Binding; 10] = [
    PICKER_BINDINGS[0],
    PICKER_BINDINGS[1],
    PICKER_BINDINGS[2],
    PICKER_BINDINGS[3],
    PICKER_BINDINGS[4],
    PICKER_BINDINGS[5],
    PICKER_BINDINGS[6],
    bind("backspace", "parent", "Parent"),
    bind("left", "parent", "Parent"),
    bind("right", "enter_dir", "Enter dir"),
];

/// Formats ffmpeg decodes that people record meetings and memos in.
pub const AUDIO_EXTS: [&str; 13] = [
    "mp3", "wav", "m4a", "mp4", "aac", "aiff", "aif", "flac", "ogg", "opus", "webm", "mov", "mkv",
];

/// Typed text that starts like this opens a path instead of filtering.
const PATH_STARTS: [&str; 4] = ["/", "~", "./", "../"];
const DIR_COUNT_CAP: usize = 1000;

/// Python's `human_size`: one decimal from KB up.
pub fn human_size(bytes: u64) -> String {
    let mut size = bytes as f64;
    for unit in ["B", "KB", "MB", "GB"] {
        if size < 1024.0 {
            return match unit {
                "B" => format!("{size:.0} {unit}"),
                _ => format!("{size:.1} {unit}"),
            };
        }
        size /= 1024.0;
    }
    format!("{size:.1} TB")
}

/// The item count of a folder, capped, or why it cannot be read.
pub fn count_dir_items(path: &Path) -> String {
    let entries = match std::fs::read_dir(path) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
            return "Permission denied".into();
        }
        Err(e) => {
            // The folder can vanish or turn into a file between listing and counting.
            return format!(
                "Cannot read: {}",
                OsError::from_io(&e, None, None).strerror()
            );
        }
    };
    let count = entries.take(DIR_COUNT_CAP + 1).count();
    if count > DIR_COUNT_CAP {
        format!("{DIR_COUNT_CAP}+")
    } else {
        count.to_string()
    }
}

fn is_audio(path: &Path) -> bool {
    path.extension()
        .map(|ext| ext.to_string_lossy().to_lowercase())
        .is_some_and(|ext| AUDIO_EXTS.contains(&ext.as_str()))
}

fn name_of(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// `Path.parent`: the root is its own parent.
fn parent_of(path: &Path) -> PathBuf {
    path.parent().unwrap_or(path).to_path_buf()
}

/// `Path.expanduser()` for a leading `~`.
fn expand_user(typed: &str) -> PathBuf {
    match (typed.strip_prefix('~'), dirs::home_dir()) {
        (Some(rest), Some(home)) if rest.is_empty() || rest.starts_with('/') => {
            home.join(rest.trim_start_matches('/'))
        }
        _ => PathBuf::from(typed),
    }
}

enum Entry {
    Parent,
    Dir(PathBuf),
    File { path: PathBuf, size: String },
}

/// A finished folder count, for the folder that was highlighted when it started.
struct DirCount {
    path: PathBuf,
    label: String,
}

fn bold(text: String) -> Line<'static> {
    Line::from(Span::styled(
        text,
        Style::default().add_modifier(Modifier::BOLD),
    ))
}

pub struct FilePicker {
    picker: Picker,
    current_dir: PathBuf,
    entries: Vec<Entry>,
    /// The folder whose count the preview waits for.
    highlighted_dir: Option<PathBuf>,
    /// A count to start after this event, for the newly highlighted folder.
    count_pending: bool,
}

impl FilePicker {
    pub fn new(start_dir: &Path) -> Self {
        let mut picker = Picker::new("Filter, or type a path and press Enter", 9999);
        picker.fractions = (2, 1);
        // The preview is a `Static` with the box's `padding: 1 2`, not Markdown.
        picker.preview.content_padding = 0;
        picker.list_focused = true;
        let mut screen = Self {
            picker,
            current_dir: start_dir
                .canonicalize()
                .unwrap_or_else(|_| start_dir.to_path_buf()),
            entries: Vec::new(),
            highlighted_dir: None,
            count_pending: false,
        };
        screen.rebuild("");
        screen
    }

    fn set_info(&mut self, lines: Vec<Line<'static>>) {
        self.picker.preview.set_lines(lines);
    }

    fn rebuild(&mut self, query: &str) {
        self.entries = vec![Entry::Parent];
        let children = match std::fs::read_dir(&self.current_dir) {
            Ok(children) => children,
            Err(e) => {
                let text = if e.kind() == std::io::ErrorKind::PermissionDenied {
                    "Permission denied".to_string()
                } else {
                    let error = OsError::from_io(&e, None, None);
                    format!(
                        "Cannot open {}: {}",
                        self.current_dir.display(),
                        error.strerror()
                    )
                };
                self.picker.list.reset_unselected(1);
                self.set_info(vec![Line::from(text)]);
                return;
            }
        };
        let paths: Vec<PathBuf> = children.flatten().map(|entry| entry.path()).collect();
        let by_name = |path: &PathBuf| name_of(path).to_lowercase();
        let mut dirs: Vec<PathBuf> = paths
            .iter()
            .filter(|path| path.is_dir() && !name_of(path).starts_with('.'))
            .cloned()
            .collect();
        let mut files: Vec<PathBuf> = paths
            .iter()
            .filter(|path| path.is_file() && is_audio(path))
            .cloned()
            .collect();
        dirs.sort_by_key(by_name);
        files.sort_by_key(by_name);
        if !query.is_empty() && !PATH_STARTS.iter().any(|start| query.starts_with(start)) {
            dirs.retain(|path| by_name(path).contains(query));
            files.retain(|path| by_name(path).contains(query));
        }
        self.entries.extend(dirs.into_iter().map(Entry::Dir));
        self.entries.extend(files.into_iter().map(|path| {
            let size = match path.metadata() {
                Ok(meta) => human_size(meta.len()),
                Err(_) => "unknown".into(),
            };
            Entry::File { path, size }
        }));
        self.picker.list.reset(self.entries.len());
        self.update_info();
    }

    fn current(&self) -> Option<&Entry> {
        self.entries.get(self.picker.list.index?)
    }

    fn update_info(&mut self) {
        let (highlighted, lines) = match self.current() {
            None => (None, Vec::new()),
            Some(Entry::Parent) => (
                None,
                vec![
                    bold("..".into()),
                    Line::from("Parent directory"),
                    Line::from(parent_of(&self.current_dir).display().to_string()),
                ],
            ),
            Some(Entry::Dir(path)) => (
                Some(path.clone()),
                vec![
                    bold(format!("{}/", name_of(path))),
                    Line::from("(counting…)"),
                ],
            ),
            Some(Entry::File { path, size }) => {
                let (size, modified) = match path.metadata() {
                    Ok(meta) => (human_size(meta.len()), modified_text(&meta)),
                    Err(_) => (size.clone(), "unknown".into()),
                };
                (
                    None,
                    vec![
                        bold(name_of(path)),
                        Line::from(format!("Size: {size}")),
                        Line::from(format!("Modified: {modified}")),
                    ],
                )
            }
        };
        self.count_pending = highlighted.is_some();
        self.highlighted_dir = highlighted;
        self.set_info(lines);
    }

    fn start_dir_count(&mut self, ctx: &mut Ctx) {
        if !std::mem::take(&mut self.count_pending) {
            return;
        }
        let Some(path) = self.highlighted_dir.clone() else {
            return;
        };
        ctx.spawn_thread("dir-count", move || DirCount {
            label: count_dir_items(&path),
            path,
        });
    }

    fn navigate_to(&mut self, directory: PathBuf) {
        self.current_dir = directory;
        // A filter typed in one folder would hide everything in the next one.
        self.picker.search.set_value("");
        self.rebuild("");
    }

    fn open_typed_path(&mut self, typed: &str) -> Action {
        let path = self.current_dir.join(expand_user(typed));
        match path.canonicalize() {
            Ok(path) if path.is_dir() => self.navigate_to(path),
            Ok(path) if path.is_file() => return Action::Pop(Some(Box::new(path))),
            _ => self.set_info(vec![Line::from(format!("Not found: {typed}"))]),
        }
        Action::None
    }

    fn select_item(&mut self) -> Action {
        let typed = self.picker.search.value().trim().to_string();
        if !self.picker.list_focused && PATH_STARTS.iter().any(|start| typed.starts_with(start)) {
            return self.open_typed_path(&typed);
        }
        let target = match self.current() {
            Some(Entry::Parent) => parent_of(&self.current_dir),
            Some(Entry::Dir(path)) => path.clone(),
            Some(Entry::File { path, .. }) => return Action::Pop(Some(Box::new(path.clone()))),
            None => return Action::None,
        };
        self.navigate_to(target);
        Action::None
    }

    fn list_binding(&mut self, name: &str) {
        match name {
            "backspace" | "left" => self.navigate_to(parent_of(&self.current_dir)),
            "right" => {
                if let Some(Entry::Dir(path)) = self.current() {
                    let path = path.clone();
                    self.navigate_to(path);
                }
            }
            _ => {}
        }
    }

    fn after(&mut self, outcome: PickerKey, name: &str, ctx: &mut Ctx) -> Action {
        let action = match outcome {
            PickerKey::Cancel => Action::Pop(None),
            PickerKey::Select => self.select_item(),
            PickerKey::Filter(query) => {
                self.rebuild(&query);
                Action::None
            }
            PickerKey::Moved => {
                self.update_info();
                Action::None
            }
            PickerKey::Unused if self.picker.list_focused => {
                self.list_binding(name);
                Action::None
            }
            PickerKey::Used | PickerKey::Unused => Action::None,
        };
        self.start_dir_count(ctx);
        action
    }
}

fn modified_text(meta: &std::fs::Metadata) -> String {
    match meta.modified() {
        Ok(time) => chrono::DateTime::<chrono::Local>::from(time)
            .format("%Y-%m-%d %H:%M")
            .to_string(),
        Err(_) => "unknown".into(),
    }
}

fn row(entry: &Entry) -> Line<'static> {
    match entry {
        Entry::Parent => Line::from("[..]"),
        Entry::Dir(path) => Line::from(format!("[{}/]", name_of(path))),
        Entry::File { path, size } => Line::from(vec![
            Span::raw(format!("{}  ", name_of(path))),
            Span::styled(size.clone(), Style::default().add_modifier(Modifier::DIM)),
        ]),
    }
}

impl Screen for FilePicker {
    fn mounted(&mut self, ctx: &mut Ctx) -> Action {
        self.start_dir_count(ctx);
        Action::None
    }

    fn render(&mut self, frame: &mut Frame, area: Rect, ctx: &Ctx) {
        let header = format!("  Select Audio File  {}", self.current_dir.display());
        let rows: Vec<Line<'static>> = self.entries.iter().map(row).collect();
        self.picker.render(
            frame,
            area,
            &header,
            "[Enter] Open/Select  [Left/Backspace] Parent  [Right] Enter dir  [↑/↓] Navigate  [Esc] Cancel",
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

    fn message(&mut self, msg: Msg, _ctx: &mut Ctx) -> (Option<Msg>, Action) {
        let Msg::Task(payload) = msg else {
            return (Some(msg), Action::None);
        };
        let count = match payload.downcast::<DirCount>() {
            Ok(count) => count,
            Err(payload) => return (Some(Msg::Task(payload)), Action::None),
        };
        // A stale result: the user has moved on.
        if self.highlighted_dir.as_ref() == Some(&count.path) {
            let detail = if count.label.starts_with(|c: char| c.is_ascii_digit()) {
                format!("{} items", count.label)
            } else {
                count.label
            };
            self.set_info(vec![
                bold(format!("{}/", name_of(&count.path))),
                Line::from(detail),
            ]);
        }
        (None, Action::None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_read_like_python() {
        assert_eq!(human_size(0), "0 B");
        assert_eq!(human_size(1023), "1023 B");
        assert_eq!(human_size(1536), "1.5 KB");
        assert_eq!(human_size(5 * 1024 * 1024), "5.0 MB");
        assert_eq!(human_size(3 * 1024u64.pow(4)), "3.0 TB");
    }

    #[test]
    fn a_large_folder_count_is_capped() {
        let dir = tempfile::tempdir().unwrap();
        for index in 0..=DIR_COUNT_CAP {
            std::fs::write(dir.path().join(format!("{index}")), "").unwrap();
        }
        assert_eq!(count_dir_items(dir.path()), "1000+");
        assert_eq!(
            count_dir_items(&dir.path().join("missing")),
            "Cannot read: No such file or directory"
        );
    }
}

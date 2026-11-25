//! What the screens ask of the OS: the saved theme, the clipboard, the file manager.

use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

pub use crate::app::theme::saved_theme;

fn clipboard_command() -> Vec<&'static str> {
    if cfg!(target_os = "macos") {
        vec!["pbcopy"]
    } else if std::env::var_os("WAYLAND_DISPLAY").is_some() {
        vec!["wl-copy"]
    } else {
        vec!["xclip", "-selection", "clipboard"]
    }
}

/// Put `text` on the system clipboard.
pub fn copy_to_clipboard(text: &str) -> Result<(), String> {
    let command = clipboard_command();
    let mut child = Command::new(command[0])
        .args(&command[1..])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("{}: {e}", command[0]))?;
    child
        .stdin
        .take()
        .ok_or("no stdin")?
        .write_all(text.as_bytes())
        .map_err(|e| e.to_string())?;
    let status = child.wait().map_err(|e| e.to_string())?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("{} exited with {status}", command[0]))
    }
}

/// Open `path` in the file manager, without waiting for it.
pub fn open_folder(path: &Path) -> Result<(), String> {
    let opener = if cfg!(target_os = "macos") {
        "open"
    } else {
        "xdg-open"
    };
    let mut child = Command::new(opener)
        .arg(path)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| e.to_string())?;
    // Reap it, so it does not stay a zombie until the app exits.
    std::thread::spawn(move || child.wait());
    Ok(())
}

//! Open a file in the system's text editor.

use std::path::Path;
use std::process::Child;

/// The command that opens `path`. `text_editor` and `wait` mean `open`'s flags;
/// `xdg-open` returns at once and has no equivalent.
pub fn command(path: &Path, text_editor: bool, wait: bool, os: &str) -> Vec<String> {
    let mut args: Vec<String> = if os == "macos" {
        let mut args = vec!["open".to_string()];
        if wait {
            args.push("-W".into());
        }
        if text_editor {
            args.push("-t".into());
        }
        args
    } else {
        vec!["xdg-open".into()]
    };
    args.push(path.display().to_string());
    args
}

pub fn open_path(path: &Path, text_editor: bool, wait: bool) -> std::io::Result<Child> {
    let args = command(path, text_editor, wait, std::env::consts::OS);
    std::process::Command::new(&args[0])
        .args(&args[1..])
        .spawn()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_macos_command_uses_open_and_its_flags() {
        let path = Path::new("/tmp/template.yaml");
        assert_eq!(
            command(path, true, true, "macos"),
            ["open", "-W", "-t", "/tmp/template.yaml"]
        );
        assert_eq!(
            command(path, false, false, "macos"),
            ["open", "/tmp/template.yaml"]
        );
    }

    #[test]
    fn linux_uses_xdg_open_and_drops_the_flags() {
        let path = Path::new("/tmp/template.yaml");
        assert_eq!(
            command(path, true, true, "linux"),
            ["xdg-open", "/tmp/template.yaml"]
        );
    }
}

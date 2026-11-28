//! The executable of the macOS .app. A Finder or Spotlight start has no
//! terminal, so it opens the window. A start from a terminal opens the
//! terminal app. Any argument runs the command line, so the whisper child
//! (this same program plus a subcommand) works inside the bundle.

use std::io::IsTerminal;

fn main() {
    if std::env::args_os().len() > 1 {
        std::process::exit(lazy_take_notes::cli::entry());
    }
    if std::io::stdin().is_terminal() {
        std::process::exit(lazy_take_notes::cli::entry());
    }
    std::process::exit(lazy_take_notes::gui::run());
}

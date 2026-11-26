//! The executable of the macOS .app. A Finder start passes no arguments and
//! opens the window. Any argument runs the command line, so the whisper child
//! (this same program plus a subcommand) works inside the bundle.

fn main() {
    if std::env::args_os().len() > 1 {
        std::process::exit(lazy_take_notes::cli::entry());
    }
    std::process::exit(lazy_take_notes::gui::run());
}

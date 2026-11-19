//! The `lazy-take-notes` program. The commands arrive with the session and TUI steps;
//! for now it runs only the whisper child.

use lazy_take_notes::adapters::whisper::{WORKER_COMMAND, worker};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some(WORKER_COMMAND) if args.len() == 3 => std::process::exit(worker::run(&args[2])),
        _ => {
            eprintln!("lazy-take-notes (Rust): not usable yet");
            std::process::exit(2);
        }
    }
}

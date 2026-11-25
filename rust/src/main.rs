//! The `lazy-take-notes` program.

use lazy_take_notes::adapters::whisper::{WORKER_COMMAND, worker};

fn main() {
    // The whisper child is this same program; it never goes through the CLI parser.
    let args: Vec<String> = std::env::args().collect();
    if args.len() == 3 && args[1] == WORKER_COMMAND {
        std::process::exit(worker::run(&args[2]));
    }
    std::process::exit(lazy_take_notes::cli::main());
}

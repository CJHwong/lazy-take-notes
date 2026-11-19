//! The command line. Commands that are not ported yet say so and exit 2.

use std::path::PathBuf;
use std::time::Duration;

use clap::{Parser, Subcommand};

use crate::app::diarize_command;

#[derive(Parser)]
#[command(
    name = "lazy-take-notes",
    version,
    about = "lazy-take-notes -- live transcription & AI summaries in your terminal."
)]
pub struct Cli {
    /// Path to YAML config file.
    #[arg(short = 'c', long = "config", global = true)]
    pub config: Option<PathBuf>,
    /// Base output directory (session subfolder created automatically).
    #[arg(
        short = 'o',
        long = "output-dir",
        env = "LTN_OUTPUT_DIR",
        global = true
    )]
    pub output_dir: Option<String>,
    #[command(subcommand)]
    pub command: Option<Command>,
}

#[derive(Subcommand)]
pub enum Command {
    /// Label a saved session's transcript with anonymous speakers.
    ///
    /// Needs the speaker runtime and model: install them in Settings > Models >
    /// Speaker assets. Writes speakers.json and transcript_speakers.txt into the
    /// session directory.
    Diarize {
        session_dir: PathBuf,
        /// Diarization backend: auto, cpu, or metal.
        #[arg(long, default_value = "auto")]
        device: String,
        /// Seconds before diarization is killed.
        #[arg(long = "timeout", default_value_t = 300.0)]
        timeout_s: f64,
    },
}

/// Parse the arguments and run the command. Returns the exit code.
pub fn main() -> i32 {
    let cli = Cli::parse();
    match cli.command {
        None => {
            eprintln!("lazy-take-notes (Rust): the TUI is not usable yet");
            2
        }
        Some(Command::Diarize {
            session_dir,
            device,
            timeout_s,
        }) => {
            if !session_dir.is_dir() {
                // A bad argument is a usage error: click exits 2.
                eprintln!(
                    "Error: Invalid value for 'SESSION_DIR': Directory '{}' does not exist.",
                    session_dir.display()
                );
                return 2;
            }
            let timeout = Duration::from_secs_f64(timeout_s.max(0.0));
            match diarize_command::run(&session_dir, &device, timeout) {
                Ok(report) => {
                    print!("{report}");
                    0
                }
                Err(e) => fail(&e),
            }
        }
    }
}

/// Print a click-style error and return its exit code.
fn fail(message: &str) -> i32 {
    eprintln!("Error: {message}");
    1
}

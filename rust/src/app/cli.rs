//! The command line. Commands that are not ported yet say so and exit 2.

use std::path::{Path, PathBuf};
use std::time::Duration;

use clap::{Args, Parser, Subcommand};

use crate::adapters::pyrepr::py_repr_json;
use crate::app::container::{load_config, resolve_base_dir};
use crate::app::headless_run::{self, HeadlessArgs};
use crate::app::{diarize_command, status_command};

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

#[derive(Args)]
pub struct SessionOptions {
    /// Session label appended to the timestamp folder (e.g. 'sprint-review').
    #[arg(short = 'l', long)]
    label: Option<String>,
    /// Run without the TUI; write files only. Track it with `status`.
    #[arg(long)]
    headless: bool,
    /// Template key, e.g. lecture_notes_en. Skips the picker.
    #[arg(long = "template")]
    template_name: Option<String>,
    /// Language/locale, e.g. en or zh-TW. Uses that locale's default template.
    #[arg(long)]
    language: Option<String>,
}

#[derive(Subcommand)]
pub enum Command {
    /// Start a live recording session with transcription and digest.
    Record {
        #[command(flatten)]
        options: SessionOptions,
        /// Start with the microphone muted (capture system audio only).
        #[arg(long)]
        mute_mic: bool,
    },
    /// Transcribe an audio file with streaming TUI and generate a final digest.
    Transcribe {
        audio_file: Option<String>,
        #[command(flatten)]
        options: SessionOptions,
    },
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
    /// Show a headless session's status (newest by default).
    Status { session: Option<String> },
    /// Browse a previously saved session (transcript + digest, read-only).
    View,
}

/// Parse the arguments and run the command. Returns the exit code.
pub fn main() -> i32 {
    let cli = Cli::parse();
    if let Some(path) = cli.config.as_deref().filter(|path| !path.exists()) {
        return usage_error(&format!(
            "Invalid value for '-c' / '--config': Path {} does not exist.",
            quote_path(path)
        ));
    }
    let config = cli.config.as_deref();
    let output_dir = cli.output_dir.as_deref();
    let result = match cli.command {
        None => return not_ported("the TUI"),
        Some(Command::Record { options, mute_mic }) => {
            record(config, output_dir, &options, mute_mic)
        }
        Some(Command::Transcribe {
            audio_file,
            options,
        }) => transcribe(config, output_dir, audio_file.as_deref(), &options),
        Some(Command::Diarize {
            session_dir,
            device,
            timeout_s,
        }) => return diarize(&session_dir, &device, timeout_s),
        Some(Command::Status { session }) => status(config, output_dir, session.as_deref()),
        Some(Command::View) => view(config, output_dir),
    };
    result.unwrap_or_else(|message| fail(&message))
}

fn quote_path(path: &Path) -> String {
    py_repr_json(&serde_json::Value::String(path.display().to_string()))
}

fn check_template_language(options: &SessionOptions) -> Result<(), String> {
    if options.template_name.is_some() && options.language.is_some() {
        return Err("Use --template or --language, not both.".into());
    }
    Ok(())
}

/// Headless has no picker, so the language must be explicit.
fn require_headless_template(options: &SessionOptions) -> Result<(), String> {
    if options.template_name.is_none() && options.language.is_none() {
        return Err("--headless requires --language (e.g. --language en) or --template.".into());
    }
    Ok(())
}

fn headless_args<'a>(
    config: Option<&'a Path>,
    output_dir: Option<&'a str>,
    options: &'a SessionOptions,
) -> HeadlessArgs<'a> {
    HeadlessArgs {
        config_path: config,
        output_dir,
        template_name: options.template_name.as_deref(),
        language: options.language.as_deref(),
        label: options.label.as_deref(),
    }
}

fn record(
    config: Option<&Path>,
    output_dir: Option<&str>,
    options: &SessionOptions,
    mute_mic: bool,
) -> Result<i32, String> {
    check_template_language(options)?;
    if !options.headless {
        return Ok(not_ported("the recording TUI"));
    }
    require_headless_template(options)?;
    headless_run::record(&headless_args(config, output_dir, options), mute_mic)
}

fn transcribe(
    config: Option<&Path>,
    output_dir: Option<&str>,
    audio_file: Option<&str>,
    options: &SessionOptions,
) -> Result<i32, String> {
    check_template_language(options)?;
    if !options.headless {
        return transcribe_tui(config, output_dir, audio_file, options);
    }
    let Some(audio_file) = audio_file else {
        return Err("--headless requires an audio file argument (no interactive picker).".into());
    };
    require_headless_template(options)?;
    if !Path::new(audio_file).is_file() {
        let quoted = py_repr_json(&serde_json::Value::String(audio_file.into()));
        return Err(format!("{quoted} is not a valid file."));
    }
    headless_run::transcribe(
        &headless_args(config, output_dir, options),
        Path::new(audio_file),
    )
}

fn transcribe_tui(
    config: Option<&Path>,
    output_dir: Option<&str>,
    audio_file: Option<&str>,
    options: &SessionOptions,
) -> Result<i32, String> {
    let picked;
    let audio_file = match audio_file {
        Some(audio_file) => audio_file,
        None => match crate::app::tui_run::pick_file()? {
            Some(path) => {
                picked = path.display().to_string();
                picked.as_str()
            }
            None => return Ok(0),
        },
    };
    if !Path::new(audio_file).is_file() {
        let quoted = py_repr_json(&serde_json::Value::String(audio_file.into()));
        return Err(format!("{quoted} is not a valid file."));
    }
    crate::app::tui_run::transcribe(
        &headless_args(config, output_dir, options),
        Path::new(audio_file),
    )
}

fn status(
    config: Option<&Path>,
    output_dir: Option<&str>,
    session: Option<&str>,
) -> Result<i32, String> {
    let loaded = load_config(config, output_dir)?;
    let base_dir = resolve_base_dir(&loaded.paths, output_dir, &loaded.config);
    let session_dir = status_command::resolve_status_dir(&base_dir, session)?;
    for line in status_command::status_lines(&session_dir)? {
        println!("{line}");
    }
    Ok(0)
}

fn view(config: Option<&Path>, output_dir: Option<&str>) -> Result<i32, String> {
    let loaded = load_config(config, output_dir)?;
    let base_dir = resolve_base_dir(&loaded.paths, output_dir, &loaded.config);
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| e.to_string())?;
    let root = Box::new(crate::tui::screens::view::ViewLoop::new(base_dir));
    crate::tui::host::run_screen(
        root,
        &crate::tui::system::saved_theme(),
        runtime.handle().clone(),
    )
    .map_err(|e| e.to_string())?;
    Ok(0)
}

fn diarize(session_dir: &Path, device: &str, timeout_s: f64) -> i32 {
    if !session_dir.is_dir() {
        return usage_error(&format!(
            "Invalid value for 'SESSION_DIR': Directory '{}' does not exist.",
            session_dir.display()
        ));
    }
    let timeout = Duration::from_secs_f64(timeout_s.max(0.0));
    match diarize_command::run(session_dir, device, timeout) {
        Ok(report) => {
            print!("{report}");
            0
        }
        Err(e) => fail(&e),
    }
}

fn not_ported(what: &str) -> i32 {
    eprintln!("lazy-take-notes (Rust): {what} is not usable yet; use --headless");
    2
}

/// A bad argument: click prints it and exits 2.
fn usage_error(message: &str) -> i32 {
    eprintln!("Error: {message}");
    2
}

/// Print a click-style error and return its exit code.
fn fail(message: &str) -> i32 {
    eprintln!("Error: {message}");
    1
}

//! `plugin add/remove/list` and the commands plugins add (PLAN.md, "Plugins").

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::Duration;

use serde_json::{Value, json};

use crate::adapters::paths::Paths;
use crate::adapters::plugins::host::{Incoming, PluginHost};
use crate::adapters::plugins::manifest::{Manifest, parse_spec_name};
use crate::app::headless_run::HeadlessArgs;
use crate::app::tui_run::{TranscribeSource, transcribe_source};
use crate::domain::session_files::NOTES;
use crate::domain::transcript::TranscriptSegment;

/// Say what is happening when a host start takes longer than this.
const START_NOTICE_AFTER: Duration = Duration::from_secs(2);

fn manifest() -> Result<Manifest, String> {
    let paths = Paths::for_user().ok_or("cannot find the home folder")?;
    Ok(Manifest::in_dir(&paths.config_dir))
}

/// Start the host for `specs`. A start that takes a while (the first one
/// installs Python and the plugins) says so on stderr.
fn start_host(specs: &[String]) -> Result<PluginHost, String> {
    let (done, waiting) = mpsc::channel::<()>();
    let notice = std::thread::spawn(move || {
        if waiting.recv_timeout(START_NOTICE_AFTER).is_err() {
            eprintln!(
                "Starting plugins (the first start installs them; this can take a minute)..."
            );
        }
    });
    let started = PluginHost::start(specs);
    let _ = done.send(());
    let _ = notice.join();
    let host = started.map_err(|e| e.to_string())?;
    for warning in host.hello.warnings() {
        eprintln!("{warning}");
    }
    Ok(host)
}

pub fn add(spec: &str) -> Result<i32, String> {
    let manifest = manifest()?;
    let name = parse_spec_name(spec);
    let mut specs = manifest.load()?;
    if specs
        .iter()
        .any(|existing| parse_spec_name(existing) == name)
    {
        println!("Plugin {name} is already added.");
        return Ok(0);
    }
    print!("Validating {name}...");
    let _ = std::io::stdout().flush();
    let mut trial = specs.clone();
    trial.push(spec.to_string());
    let host = match start_host(&trial) {
        Ok(host) => host,
        Err(error) => {
            println!(" failed");
            eprintln!("Validation failed: {error}");
            return Ok(1);
        }
    };
    if let Some(error) = host
        .hello
        .load_errors
        .iter()
        .find(|e| e.from_package(&name))
    {
        println!(" failed");
        eprintln!("Validation failed: {}", error.error);
        return Ok(1);
    }
    specs.push(spec.to_string());
    manifest.save(&specs)?;
    println!(" ok\nPlugin {name} added.");
    Ok(0)
}

pub fn remove(name: &str) -> Result<i32, String> {
    let manifest = manifest()?;
    let specs = manifest.load()?;
    let kept: Vec<String> = specs
        .iter()
        .filter(|spec| parse_spec_name(spec) != name)
        .cloned()
        .collect();
    if kept.len() == specs.len() {
        eprintln!("Plugin {name} not found.");
        return Ok(0);
    }
    manifest.save(&kept)?;
    println!("Plugin {name} removed.");
    Ok(0)
}

pub fn list() -> Result<i32, String> {
    let specs = manifest()?.load()?;
    if specs.is_empty() {
        println!("No plugins installed.");
    }
    for spec in specs {
        println!("  {}  ({spec})", parse_spec_name(&spec));
    }
    Ok(0)
}

/// What a plugin command needs from the app's own options.
pub struct Options<'a> {
    pub config_path: Option<&'a Path>,
    pub output_dir: Option<&'a str>,
}

/// `lazy-take-notes <name> <args>` for a command a plugin adds. None: no plugin
/// has that command, and the caller reports the unknown command. An error: the
/// plugins could not start.
pub fn run(argv: &[String], options: &Options<'_>) -> Result<Option<i32>, String> {
    let Some((name, args)) = argv.split_first() else {
        return Ok(None);
    };
    let specs = manifest()?.launch_specs();
    if specs.is_empty() {
        return Ok(None);
    }
    let host = start_host(&specs)?;
    if !host
        .hello
        .commands
        .iter()
        .any(|command| &command.name == name)
    {
        return Ok(None);
    }
    // The plugin's own output goes to the terminal, as under Python.
    host.echo_stderr(true);
    let incoming = host.take_incoming();
    let params = json!({
        "name": name,
        "argv": args,
        "config_path": options.config_path.map(|p| p.display().to_string()),
        "output_dir": options.output_dir,
    });
    let reply = host.call_serving("command.run", params, &incoming, |request| {
        serve(&host, request, options)
    });
    let reply = match reply {
        Ok(reply) => reply,
        Err(error) => {
            eprintln!("Error: {error}");
            return Ok(Some(1));
        }
    };
    let code = reply.get("exit_code").and_then(Value::as_i64).unwrap_or(1);
    Ok(Some(i32::try_from(code).unwrap_or(1)))
}

/// A request from the plugin while its command runs.
fn serve(
    host: &PluginHost,
    request: Incoming,
    options: &Options<'_>,
) -> Result<Value, (i64, String)> {
    match request.method.as_str() {
        "session.transcribe" => {
            // The TUI owns the terminal now; the host's stderr goes to the log.
            host.echo_stderr(false);
            let result = transcribe(&request.params, options);
            host.echo_stderr(true);
            result.map_err(|message| (-32000, message))
        }
        other => Err((
            -32601,
            format!("The app does not run {other} from a plugin yet."),
        )),
    }
}

fn text<'a>(params: &'a Value, key: &str) -> Option<&'a str> {
    params.get(key).and_then(Value::as_str)
}

fn transcribe(params: &Value, options: &Options<'_>) -> Result<Value, String> {
    // Python replays the subtitles when both are given.
    let source = match (params.get("segments"), text(params, "audio_path")) {
        (Some(segments), _) if !segments.is_null() => {
            let segments: Vec<TranscriptSegment> = serde_json::from_value(segments.clone())
                .map_err(|e| format!("Bad segments from the plugin: {e}"))?;
            TranscribeSource::Subtitles(segments)
        }
        (_, Some(audio)) => TranscribeSource::Audio(PathBuf::from(audio)),
        _ => return Err("run_transcribe needs audio_path or subtitle_segments.".into()),
    };
    let args = HeadlessArgs {
        config_path: options.config_path,
        output_dir: options.output_dir,
        template_name: text(params, "template_name"),
        language: text(params, "language"),
        label: text(params, "label"),
    };
    let Some(session_dir) = transcribe_source(&args, source)? else {
        return Ok(Value::Null);
    };
    Ok(json!({
        "session_dir": session_dir.display().to_string(),
        "notes_path": NOTES.resolve(&session_dir).map(|p| p.display().to_string()),
    }))
}

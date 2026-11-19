//! Check every requirement of a live recording before the meeting starts.
//!
//! A problem found here costs a retry. The same problem found mid-meeting costs
//! the meeting. Each check uses the objects the session then uses: the same LLM
//! client, and the same transcriber, whose loaded model carries into the session.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use tracing::{info, warn};

use crate::adapters::assets::resolver::CachedModelResolver;
use crate::adapters::assets::store::CachedModelStore;
use crate::adapters::audio::coreaudio_tap::CoreAudioTapSource;
use crate::adapters::audio::microphone::{MicrophoneSource, play_test_tone};
use crate::adapters::nemo_diarizer::NemoSpeechDiarizer;
use crate::app::download_progress;
use crate::domain::SAMPLE_RATE;
use crate::domain::config::AppConfig;
use crate::domain::template::SessionTemplate;
use crate::ports::audio_source::AudioSource;
use crate::ports::llm::LlmClient;
use crate::ports::model_resolver::ModelResolver;
use crate::ports::transcriber::Transcriber;
use crate::usecases::template_validator::validate_template;

/// 16 kHz int16 mono WAV is about 115 MB per hour; this covers a 4-hour meeting.
pub const MIN_FREE_BYTES_WITH_AUDIO: u64 = 500 * 1024 * 1024;
pub const MIN_FREE_BYTES: u64 = 50 * 1024 * 1024;
/// A cold Ollama model load can take a minute.
pub const LLM_PROBE_TIMEOUT: Duration = Duration::from_secs(120);
const LLM_PROBE_PROMPT: &str = "Reply with the single word OK.";
const MIC_READ: Duration = Duration::from_millis(1500);
const SYSTEM_FIRST_BUFFER: Duration = Duration::from_secs(5);
const SYSTEM_PERMISSION_WAIT: Duration = Duration::from_secs(45);
const SYSTEM_TONE_WAIT: Duration = Duration::from_millis(1500);
const READ_STEP: Duration = Duration::from_millis(100);

const MIC_PERMISSION_FIX: &str = "Allow microphone access in System Settings > Privacy & Security > Microphone, then check the input volume in System Settings > Sound.";
const SYSTEM_PERMISSION_FIX: &str = "Allow this app in System Settings > Privacy & Security > Screen & System Audio Recording, then press Retry.";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Pending,
    Running,
    Pass,
    Warn,
    Fail,
    Skip,
}

impl Status {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Running => "running",
            Self::Pass => "pass",
            Self::Warn => "warn",
            Self::Fail => "fail",
            Self::Skip => "skip",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Check {
    pub key: &'static str,
    pub title: &'static str,
    pub status: Status,
    pub detail: String,
    pub fix: String,
    /// A failed overridable check offers "Start anyway" instead of blocking.
    pub overridable: bool,
}

impl Check {
    pub fn new(key: &'static str, title: &'static str, overridable: bool) -> Self {
        Self {
            key,
            title,
            status: Status::Pending,
            detail: String::new(),
            fix: String::new(),
            overridable,
        }
    }

    pub fn set(&mut self, status: Status, detail: impl Into<String>, fix: impl Into<String>) {
        self.status = status;
        self.detail = detail.into();
        self.fix = fix.into();
    }

    fn fail(&mut self, detail: impl Into<String>, fix: impl Into<String>) {
        self.set(Status::Fail, detail, fix);
    }

    fn pass(&mut self, detail: impl Into<String>) {
        self.set(Status::Pass, detail, "");
    }
}

pub fn new_checks() -> Vec<Check> {
    vec![
        Check::new("output", "Output folder", false),
        Check::new("template", "Template", false),
        Check::new("mic", "Microphone", false),
        Check::new("system", "System audio", false),
        Check::new("whisper", "Speech model", false),
        Check::new("diarization", "Speaker labels", false),
        Check::new("llm", "AI notes model", true),
    ]
}

#[derive(Debug, Clone, PartialEq)]
pub struct PreflightReport {
    pub checks: Vec<Check>,
    pub overridden: bool,
}

impl PreflightReport {
    pub fn get(&self, key: &str) -> &Check {
        self.checks
            .iter()
            .find(|check| check.key == key)
            .expect("every check key exists")
    }

    pub fn failures(&self) -> Vec<&Check> {
        self.checks
            .iter()
            .filter(|check| check.status == Status::Fail)
            .collect()
    }

    pub fn ok(&self) -> bool {
        self.failures().is_empty()
    }

    pub fn blocked(&self) -> bool {
        self.failures().iter().any(|check| !check.overridable)
    }

    pub fn llm_ok(&self) -> bool {
        self.get("llm").status != Status::Fail
    }
}

/// Everything the checks need. The session reuses these objects.
pub struct PreflightPlan {
    pub config: AppConfig,
    pub template: SessionTemplate,
    pub base_dir: PathBuf,
    pub llm_client: Arc<dyn LlmClient>,
    pub llm_provider: String,
    pub asset_store: Arc<CachedModelStore>,
    /// Runs the LLM probe. The session's runtime, so the client's connections stay valid.
    pub runtime: tokio::runtime::Handle,
    /// --mute-mic: the mic must still open, but what it hears does not matter.
    pub mic_muted: bool,
    /// Set when the user leaves the checks; a running download stops and keeps its partial file.
    pub cancel: Arc<AtomicBool>,
}

/// Called with a check each time it changes. Called from more than one thread.
pub type Notify<'a> = &'a (dyn Fn(&Check) + Sync);

/// A check while it runs: changes go straight to the screen.
struct Running<'a> {
    check: &'a mut Check,
    notify: Notify<'a>,
}

impl Running<'_> {
    fn update(&mut self, detail: impl Into<String>) {
        self.check.detail = detail.into();
        (self.notify)(self.check);
    }
}

type CheckFn = fn(&mut Running<'_>, &PreflightPlan, &mut dyn Transcriber) -> Result<(), String>;

/// Run every check. The slow network checks run beside the audio checks.
pub fn run_preflight(
    plan: &PreflightPlan,
    transcriber: &mut dyn Transcriber,
    notify: Notify<'_>,
) -> PreflightReport {
    let started = Instant::now();
    let mut checks = new_checks();
    let [output, template, mic, system, whisper, diarization, llm] = checks.as_mut_slice() else {
        unreachable!("seven checks");
    };
    std::thread::scope(|scope| {
        scope.spawn(|| run_check(whisper, check_whisper, plan, transcriber, notify));
        scope.spawn(|| {
            run_check(llm, check_llm, plan, &mut NoTranscriber, notify);
        });
        let fast: [(&mut Check, CheckFn); 5] = [
            (output, check_output),
            (template, check_template),
            (mic, check_microphone),
            (system, check_system_audio),
            (diarization, check_diarization),
        ];
        for (check, check_fn) in fast {
            run_check(check, check_fn, plan, &mut NoTranscriber, notify);
        }
    });
    let report = PreflightReport {
        checks,
        overridden: false,
    };
    let summary: Vec<String> = report
        .checks
        .iter()
        .map(|check| format!("{}={}", check.key, check.status.as_str()))
        .collect();
    info!(
        target: "ltn.preflight",
        "preflight finished in {:.1}s: {}",
        started.elapsed().as_secs_f64(),
        summary.join(", ")
    );
    report
}

/// Only the speech model check gets the session's transcriber.
struct NoTranscriber;

impl Transcriber for NoTranscriber {
    fn load_model(
        &mut self,
        _path: &str,
    ) -> Result<(), crate::ports::transcriber::TranscribeError> {
        Ok(())
    }

    fn transcribe(
        &mut self,
        _audio: &[f32],
        _language: &str,
        _hints: &[String],
    ) -> Result<
        Vec<crate::domain::transcript::TranscriptSegment>,
        crate::ports::transcriber::TranscribeError,
    > {
        Ok(Vec::new())
    }
}

fn run_check(
    check: &mut Check,
    check_fn: CheckFn,
    plan: &PreflightPlan,
    transcriber: &mut dyn Transcriber,
    notify: Notify<'_>,
) {
    check.set(Status::Running, "", "");
    notify(check);
    let mut running = Running { check, notify };
    let result = check_fn(&mut running, plan, transcriber);
    let check = running.check;
    if let Err(error) = result {
        warn!(target: "ltn.preflight", "preflight check {} failed: {error}", check.key);
        let fix = std::mem::take(&mut check.fix);
        check.fail(error, fix);
    }
    if check.status == Status::Running {
        let detail = std::mem::take(&mut check.detail);
        check.pass(detail);
    }
    notify(check);
}

/// Python's `strerror`: the OS text without Rust's " (os error N)".
fn strerror(error: &std::io::Error) -> String {
    let text = error.to_string();
    match error.raw_os_error() {
        Some(code) => text
            .strip_suffix(&format!(" (os error {code})"))
            .unwrap_or(&text)
            .to_string(),
        None => text,
    }
}

fn free_bytes(path: &Path) -> std::io::Result<u64> {
    use std::os::unix::ffi::OsStrExt;
    let name = std::ffi::CString::new(path.as_os_str().as_bytes())
        .map_err(|_| std::io::Error::from_raw_os_error(libc::EINVAL))?;
    let mut stats: libc::statvfs = unsafe { std::mem::zeroed() };
    // SAFETY: `name` is a valid C string and `stats` a writable statvfs.
    if unsafe { libc::statvfs(name.as_ptr(), &mut stats) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(u64::from(stats.f_bavail) * stats.f_frsize)
}

fn probe_output(base: &Path) -> std::io::Result<u64> {
    std::fs::create_dir_all(base)?;
    let probe = base.join(format!(".ltn-preflight-{}", std::process::id()));
    std::fs::write(&probe, "ok")?;
    std::fs::remove_file(&probe)?;
    free_bytes(base)
}

fn check_output(
    running: &mut Running<'_>,
    plan: &PreflightPlan,
    _transcriber: &mut dyn Transcriber,
) -> Result<(), String> {
    let base = &plan.base_dir;
    let free = match probe_output(base) {
        Ok(free) => free,
        Err(error) => {
            running.check.fail(
                format!("Cannot write to {}: {}", base.display(), strerror(&error)),
                "Choose a writable output.directory in Settings > Sessions.",
            );
            return Ok(());
        }
    };
    let need = if plan.config.output.save_audio {
        MIN_FREE_BYTES_WITH_AUDIO
    } else {
        MIN_FREE_BYTES
    };
    if free < need {
        running.check.fail(
            format!("Only {} MB free in {}", free >> 20, base.display()),
            format!(
                "Free at least {} MB, or choose another output.directory.",
                need >> 20
            ),
        );
        return Ok(());
    }
    running.check.pass(format!(
        "{} ({:.1} GB free)",
        base.display(),
        free as f64 / f64::from(1u32 << 30)
    ));
    Ok(())
}

fn check_template(
    running: &mut Running<'_>,
    plan: &PreflightPlan,
    _transcriber: &mut dyn Transcriber,
) -> Result<(), String> {
    let errors = validate_template(&plan.template).errors;
    if errors.is_empty() {
        running.check.pass(plan.template.metadata.name.clone());
    } else {
        running.check.fail(
            errors.join("; "),
            "Fix the template in the template builder, or pick another template.",
        );
    }
    Ok(())
}

fn check_whisper(
    running: &mut Running<'_>,
    plan: &PreflightPlan,
    transcriber: &mut dyn Transcriber,
) -> Result<(), String> {
    let locale = plan.template.metadata.locale.clone();
    let name = plan
        .config
        .transcription
        .model_for_locale(&locale)
        .to_string();
    running.check.fix =
        "Check the network for the first download, or install the model in Settings > Models."
            .into();
    running.update(format!("Finding {name}"));
    let path = resolve_showing_progress(running, plan, &name);
    let path = match path {
        Ok(path) => path,
        Err(error) => {
            running.check.fail(
                error.to_string(),
                "Set transcription.model to a model listed in Settings > Models.",
            );
            return Ok(());
        }
    };
    running.update(format!("Loading {name}"));
    transcriber
        .load_model(&path)
        .map_err(|e| format!("RuntimeError: {e}"))?;
    running.update(format!("Testing {name}"));
    let language = locale.split('-').next().unwrap_or("").to_lowercase();
    transcriber
        .transcribe(&vec![0.0; SAMPLE_RATE], &language, &[])
        .map_err(|e| format!("RuntimeError: {e}"))?;
    running.check.pass(name);
    Ok(())
}

/// Resolve the model on a helper thread, and show each download step here as it comes.
fn resolve_showing_progress(
    running: &mut Running<'_>,
    plan: &PreflightPlan,
    name: &str,
) -> Result<String, crate::domain::errors::ModelResolutionError> {
    let (lines, inbox) = std::sync::mpsc::channel::<String>();
    let label = name.to_string();
    let mut resolver = CachedModelResolver::new(
        plan.asset_store.clone(),
        Some(Box::new(move |step| {
            let detail = download_progress::describe(&step);
            let _ = lines.send(format!("Downloading {label}: {detail}"));
        })),
    );
    resolver.cancel = plan.cancel.clone();
    std::thread::scope(|scope| {
        let resolving = scope.spawn(|| {
            let result = resolver.resolve(name);
            drop(resolver); // closes the channel, which ends the loop below
            result
        });
        for line in inbox {
            running.update(line);
        }
        resolving
            .join()
            .expect("the resolver thread does not panic")
    })
}

/// (frames received, non-zero samples) over the window.
fn listen(source: &dyn AudioSource, window: Duration) -> Result<(usize, usize), String> {
    let (mut frames, mut nonzero) = (0, 0);
    let end = Instant::now() + window;
    while Instant::now() < end {
        let Some(chunk) = source.read(READ_STEP).map_err(|e| e.to_string())? else {
            continue;
        };
        frames += chunk.len();
        nonzero += chunk.iter().filter(|&&s| s != 0.0).count();
    }
    Ok((frames, nonzero))
}

fn check_microphone(
    running: &mut Running<'_>,
    plan: &PreflightPlan,
    _transcriber: &mut dyn Transcriber,
) -> Result<(), String> {
    let mic = MicrophoneSource::new(plan.config.transcription.input_device.clone());
    running.check.fix = MIC_PERMISSION_FIX.into();
    running.update("Listening");
    mic.open(SAMPLE_RATE as u32, 1)
        .map_err(|e| format!("PortAudioError: {e}"))?;
    let heard = listen(&mic, MIC_READ);
    let _ = mic.close();
    let (frames, nonzero) = heard.map_err(|e| format!("PortAudioError: {e}"))?;
    let notice = mic.device_notice();
    let check = &mut *running.check;
    if plan.mic_muted {
        check.pass("Opened; muted for this session");
    } else if frames == 0 {
        check.fail(
            "The microphone opened but sent no audio",
            MIC_PERMISSION_FIX,
        );
    } else if nonzero == 0 {
        check.fail(
            "The microphone sends only digital silence",
            MIC_PERMISSION_FIX,
        );
    } else if !notice.is_empty() {
        check.set(
            Status::Warn,
            notice,
            "Connect that microphone, or pick another one in Settings > Audio.",
        );
    } else {
        check.pass(format!(
            "{:.1}s of audio received",
            frames as f64 / SAMPLE_RATE as f64
        ));
    }
    Ok(())
}

fn check_system_audio(
    running: &mut Running<'_>,
    _plan: &PreflightPlan,
    _transcriber: &mut dyn Transcriber,
) -> Result<(), String> {
    let system = CoreAudioTapSource::new(false);
    running.check.fix = SYSTEM_PERMISSION_FIX.into();
    running.update("Starting capture");
    system
        .open(SAMPLE_RATE as u32, 1)
        .map_err(|e| format!("RuntimeError: {e}"))?;
    let result = verify_system_capture(running, &system);
    let _ = system.close();
    result
}

fn verify_system_capture(
    running: &mut Running<'_>,
    system: &dyn AudioSource,
) -> Result<(), String> {
    if !system.continuous_capture() {
        // A loopback that delivers only while audio plays cannot be proven idle.
        running
            .check
            .pass("Loopback opened; verified when audio plays");
        return Ok(());
    }
    if !wait_for_first_buffer(running, system)? {
        let fix = running.check.fix.clone();
        running
            .check
            .fail("System audio capture did not start", fix);
        return Ok(());
    }
    system.drain();
    running.update("Playing an inaudible test tone");
    if !play_test_tone() {
        running.check.set(
            Status::Warn,
            "No output device to play the test tone; system audio is not verified",
            "",
        );
        return Ok(());
    }
    let (_frames, nonzero) = listen(system, SYSTEM_TONE_WAIT)?;
    if nonzero == 0 {
        running.check.fail(
            "System audio is silent: the capture permission is missing",
            SYSTEM_PERMISSION_FIX,
        );
        return Ok(());
    }
    running.check.pass("Test tone captured");
    Ok(())
}

fn wait_for_first_buffer(
    running: &mut Running<'_>,
    source: &dyn AudioSource,
) -> Result<bool, String> {
    let started = Instant::now();
    loop {
        let limit = if source.awaiting_permission() {
            SYSTEM_PERMISSION_WAIT
        } else {
            SYSTEM_FIRST_BUFFER
        };
        if started.elapsed() >= limit {
            return Ok(false);
        }
        let chunk = source
            .read(READ_STEP)
            .map_err(|e| format!("RuntimeError: {e}"))?;
        if chunk.is_some_and(|chunk| !chunk.is_empty()) {
            return Ok(true);
        }
        if source.exhausted() {
            return Ok(false);
        }
        if source.awaiting_permission() {
            running.update(format!("Waiting for permission. {SYSTEM_PERMISSION_FIX}"));
        }
    }
}

fn check_diarization(
    running: &mut Running<'_>,
    plan: &PreflightPlan,
    _transcriber: &mut dyn Transcriber,
) -> Result<(), String> {
    let config = &plan.config.diarization;
    let check = &mut *running.check;
    if !config.enabled {
        check.set(Status::Skip, "Off", "");
        return Ok(());
    }
    if !plan.config.output.save_audio {
        check.fail(
            "Speaker labels need the recording saved",
            "Turn on save_audio in Settings > Sessions, or turn off speaker labels in Settings > Audio.",
        );
        return Ok(());
    }
    let diarizer = NemoSpeechDiarizer::new(
        &config.binary,
        &config.model,
        &config.device,
        Some(plan.asset_store.clone()),
    );
    let binary = match diarizer.resolve_paths() {
        Ok((binary, _model)) => binary,
        Err(error) => {
            check.fail(error, "Install the speaker assets in Settings > Models.");
            return Ok(());
        }
    };
    if !is_executable(&binary) {
        check.fail(
            format!("{} is not executable", binary.display()),
            "Reinstall the speaker assets in Settings > Models.",
        );
        return Ok(());
    }
    check.pass("Assets installed");
    Ok(())
}

fn is_executable(path: &Path) -> bool {
    use std::os::unix::ffi::OsStrExt;
    let Ok(name) = std::ffi::CString::new(path.as_os_str().as_bytes()) else {
        return false;
    };
    // SAFETY: `name` is a valid C string.
    unsafe { libc::access(name.as_ptr(), libc::X_OK) == 0 }
}

fn check_llm(
    running: &mut Running<'_>,
    plan: &PreflightPlan,
    _transcriber: &mut dyn Transcriber,
) -> Result<(), String> {
    let models = [
        plan.config.digest.model.clone(),
        plan.config.interactive.model.clone(),
    ];
    let client = plan.llm_client.clone();
    let check = &mut *running.check;
    let notify = running.notify;
    plan.runtime.block_on(check_ai(
        check,
        client.as_ref(),
        &plan.llm_provider,
        &models,
        &|check: &Check| notify(check),
    ));
    Ok(())
}

/// Reach the provider, find the models, and get one tiny reply from each. Settings > Test runs it too.
pub async fn check_ai(
    check: &mut Check,
    client: &dyn LlmClient,
    provider: &str,
    models: &[String],
    notify: &(dyn Fn(&Check) + Sync),
) {
    if let Err(error) = client.check_connectivity().await {
        check.fail(
            format!("Cannot reach the AI provider: {error}"),
            provider_fix(provider),
        );
        return;
    }
    let mut unique: Vec<String> = Vec::new();
    for model in models {
        if !unique.contains(model) {
            unique.push(model.clone());
        }
    }
    let missing = client.check_models(&unique).await;
    if !missing.is_empty() {
        let fix = if provider == "ollama" {
            let pulls: Vec<String> = missing.iter().map(|m| format!("ollama pull {m}")).collect();
            format!("Run: {}", pulls.join("; "))
        } else {
            "Pick an available model in Settings > Models.".into()
        };
        check.fail(format!("Model not available: {}", missing.join(", ")), fix);
        return;
    }
    for model in &unique {
        check.detail = format!(
            "Asking {model} for a one-word reply (up to {}s)",
            LLM_PROBE_TIMEOUT.as_secs()
        );
        notify(check);
        if let Some(problem) = probe_model(client, model).await {
            check.fail(
                format!("{model}: {problem}"),
                "Check the API key, the usage limit, and the model name.",
            );
            return;
        }
    }
    check.pass(unique.join(", "));
}

/// One tiny request. Catches auth errors and usage limits that a model list cannot.
async fn probe_model(client: &dyn LlmClient, model: &str) -> Option<String> {
    match tokio::time::timeout(
        LLM_PROBE_TIMEOUT,
        client.chat_single(model, LLM_PROBE_PROMPT),
    )
    .await
    {
        Err(_) => Some(format!("no reply within {}s", LLM_PROBE_TIMEOUT.as_secs())),
        // LlmError shows as "Kind: message", as Python's `{type(error).__name__}: {error}`.
        Ok(Err(error)) => Some(error.to_string()),
        Ok(Ok(_)) => None,
    }
}

fn provider_fix(provider: &str) -> &'static str {
    if provider == "ollama" {
        return "Start Ollama (ollama serve), or check ollama.host in Settings > Models.";
    }
    "Check the provider address and network in Settings > Models."
}

/// One plain-text line per check, for the CLI and headless output.
pub fn format_report_line(check: &Check) -> String {
    let mark = match check.status {
        Status::Pass => "ok",
        Status::Warn => "warn",
        Status::Fail => "FAIL",
        Status::Skip => "skip",
        other => other.as_str(),
    };
    let mut line = format!("[{mark:>4}] {}: {}", check.title, check.detail);
    if matches!(check.status, Status::Fail | Status::Warn) && !check.fix.is_empty() {
        line += &format!("\n       fix: {}", check.fix);
    }
    line
}

#[cfg(test)]
#[path = "preflight_tests.rs"]
pub(crate) mod tests;

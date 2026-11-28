//! Ported from tests/l4_frameworks_and_drivers/test_preflight.py, minus the hardware checks.

use std::sync::Mutex;

use async_trait::async_trait;

use super::*;
use crate::adapters::assets::store::LegacyDirs;
use crate::domain::chat::ChatMessage;
use crate::domain::transcript::TranscriptSegment;
use crate::ports::audio_source::AudioError;
use crate::ports::llm::{ChatResponse, LlmError};
use crate::ports::transcriber::TranscribeError;

/// What the fake provider does.
#[derive(Clone)]
pub(crate) enum Provider {
    Down,
    Missing(Vec<&'static str>),
    Refuses,
    Answers,
}

struct FakeClient(Provider);

#[async_trait]
impl LlmClient for FakeClient {
    async fn chat(
        &self,
        _model: &str,
        _messages: &[ChatMessage],
    ) -> Result<ChatResponse, LlmError> {
        match self.0 {
            Provider::Refuses => Err(LlmError::new(
                "ResponseError",
                "unauthorized (status code: 401)",
            )),
            _ => Ok(ChatResponse {
                content: "OK".into(),
                prompt_tokens: 1,
            }),
        }
    }

    async fn check_connectivity(&self) -> Result<(), String> {
        match self.0 {
            Provider::Down => Err("Connection refused".into()),
            _ => Ok(()),
        }
    }

    async fn check_models(&self, _models: &[String]) -> Vec<String> {
        match &self.0 {
            Provider::Missing(names) => names.iter().map(|n| n.to_string()).collect(),
            _ => Vec::new(),
        }
    }
}

fn store(dir: &Path) -> Arc<CachedModelStore> {
    let legacy = LegacyDirs {
        pywhispercpp_models: dir.join("p"),
        nemotron_store: dir.join("n"),
    };
    Arc::new(CachedModelStore::with(
        dir.join("cache"),
        vec![],
        "test-os",
        legacy,
    ))
}

pub(crate) fn plan(
    dir: &Path,
    config_yaml: &str,
    provider: Provider,
    runtime: &tokio::runtime::Runtime,
) -> PreflightPlan {
    let raw = yaml_serde::from_str(config_yaml).unwrap();
    let template_text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../src/lazy_take_notes/templates/default_en.yaml"
    ))
    .unwrap();
    PreflightPlan {
        config: crate::config::build_app_config(&raw).unwrap(),
        template: crate::adapters::templates::parse_template(&template_text, "default_en").unwrap(),
        base_dir: dir.join("out"),
        llm_client: Arc::new(FakeClient(provider)),
        llm_provider: "ollama".into(),
        asset_store: store(dir),
        runtime: runtime.handle().clone(),
        mic_muted: false,
        cancel: Arc::default(),
    }
}

pub(crate) fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_all()
        .build()
        .unwrap()
}

/// Run one check function the way `run_preflight` does.
fn run_one(
    check_fn: CheckFn,
    plan: &PreflightPlan,
    transcriber: &mut dyn Transcriber,
) -> (Check, Vec<String>) {
    let seen = Mutex::new(Vec::new());
    let notify = |check: &Check| seen.lock().unwrap().push(check.detail.clone());
    let mut check = Check::new("x", "X", false);
    run_check(&mut check, check_fn, plan, transcriber, &notify);
    (check, seen.into_inner().unwrap())
}

#[test]
fn a_report_line_reads_like_python() {
    let mut check = Check::new("output", "Output folder", false);
    check.set(
        Status::Fail,
        "Cannot write to /x: Permission denied",
        "Choose a folder.",
    );
    assert_eq!(
        format_report_line(&check),
        "[FAIL] Output folder: Cannot write to /x: Permission denied\n       fix: Choose a folder."
    );
    check.set(Status::Pass, "/x (1.0 GB free)", "ignored");
    assert_eq!(
        format_report_line(&check),
        "[  ok] Output folder: /x (1.0 GB free)"
    );
}

#[test]
fn only_a_failed_ai_check_can_be_overridden() {
    let mut checks = new_checks();
    checks.iter_mut().find(|c| c.key == "llm").unwrap().status = Status::Fail;
    let report = PreflightReport {
        checks,
        overridden: false,
    };
    assert!(!report.ok());
    assert!(!report.blocked());
    assert!(!report.llm_ok());
    let mut checks = new_checks();
    checks[0].status = Status::Fail;
    assert!(
        PreflightReport {
            checks,
            overridden: false
        }
        .blocked()
    );
}

#[cfg(unix)]
#[test]
fn an_unwritable_output_folder_fails_with_the_os_text() {
    let dir = tempfile::tempdir().unwrap();
    let runtime = runtime();
    let mut plan = plan(dir.path(), "{}", Provider::Answers, &runtime);
    std::fs::write(dir.path().join("file"), "x").unwrap();
    plan.base_dir = dir.path().join("file/out");
    let (check, _) = run_one(check_output, &plan, &mut NoTranscriber);
    assert_eq!(check.status, Status::Fail);
    assert_eq!(
        check.detail,
        format!(
            "Cannot write to {}: Not a directory",
            plan.base_dir.display()
        )
    );
}

#[test]
fn a_writable_output_folder_passes_with_its_free_space() {
    let dir = tempfile::tempdir().unwrap();
    let runtime = runtime();
    let plan = plan(dir.path(), "{}", Provider::Answers, &runtime);
    let (check, _) = run_one(check_output, &plan, &mut NoTranscriber);
    assert_eq!(check.status, Status::Pass, "{}", check.detail);
    assert!(check.detail.ends_with(" GB free)"));
    assert_eq!(
        std::fs::read_dir(&plan.base_dir).unwrap().count(),
        0,
        "the probe file is gone"
    );
}

#[test]
fn speaker_labels_off_is_skipped() {
    let dir = tempfile::tempdir().unwrap();
    let runtime = runtime();
    let off = plan(dir.path(), "{}", Provider::Answers, &runtime);
    assert_eq!(
        run_one(check_diarization, &off, &mut NoTranscriber)
            .0
            .status,
        Status::Skip
    );
}

#[cfg(not(windows))]
#[test]
fn speaker_labels_without_audio_or_a_binary_fail() {
    let dir = tempfile::tempdir().unwrap();
    let runtime = runtime();
    let no_audio = plan(
        dir.path(),
        "{diarization: {enabled: true}, output: {save_audio: false}}",
        Provider::Answers,
        &runtime,
    );
    let (check, _) = run_one(check_diarization, &no_audio, &mut NoTranscriber);
    assert_eq!(check.detail, "Speaker labels need the recording saved");
    let missing = plan(
        dir.path(),
        "{diarization: {enabled: true, binary: /nowhere/nemo}, output: {save_audio: true}}",
        Provider::Answers,
        &runtime,
    );
    let (check, _) = run_one(check_diarization, &missing, &mut NoTranscriber);
    assert_eq!(check.status, Status::Fail);
    assert!(
        check
            .detail
            .starts_with("nemo-speech binary not found at /nowhere/nemo.")
    );
}

#[cfg(windows)]
#[test]
fn speaker_labels_are_not_available_on_windows() {
    let dir = tempfile::tempdir().unwrap();
    let runtime = runtime();
    let enabled = plan(
        dir.path(),
        "{diarization: {enabled: true, binary: /nowhere/nemo}, output: {save_audio: true}}",
        Provider::Answers,
        &runtime,
    );
    let (check, _) = run_one(check_diarization, &enabled, &mut NoTranscriber);
    assert_eq!(check.status, Status::Skip);
    assert_eq!(check.detail, "Not available on Windows");
    let no_audio = plan(
        dir.path(),
        "{diarization: {enabled: true}, output: {save_audio: false}}",
        Provider::Answers,
        &runtime,
    );
    let (check, _) = run_one(check_diarization, &no_audio, &mut NoTranscriber);
    assert_eq!(check.status, Status::Skip);
}

fn ai(provider: Provider) -> Check {
    let dir = tempfile::tempdir().unwrap();
    let runtime = runtime();
    let plan = plan(dir.path(), "{}", provider, &runtime);
    run_one(check_llm, &plan, &mut NoTranscriber).0
}

#[test]
fn the_ai_check_names_the_failure_and_its_fix() {
    let down = ai(Provider::Down);
    assert_eq!(
        down.detail,
        "Cannot reach the AI provider: Connection refused"
    );
    assert_eq!(
        down.fix,
        "Start Ollama (ollama serve), or check ollama.host in Settings > Models."
    );
    let missing = ai(Provider::Missing(vec!["a", "b"]));
    assert_eq!(missing.detail, "Model not available: a, b");
    assert_eq!(missing.fix, "Run: ollama pull a; ollama pull b");
    let refused = ai(Provider::Refuses);
    assert!(
        refused
            .detail
            .ends_with(": ResponseError: unauthorized (status code: 401)"),
        "{}",
        refused.detail
    );
    let answers = ai(Provider::Answers);
    assert_eq!(answers.status, Status::Pass);
}

/// Loads any path; fails the load when asked to; counts transcribe calls.
#[derive(Default)]
pub(crate) struct Whisper {
    loaded: Option<String>,
    fail_load: bool,
    transcripts: usize,
}

impl Transcriber for Whisper {
    fn load_model(&mut self, path: &str) -> Result<(), TranscribeError> {
        if self.fail_load {
            return Err(TranscribeError("model load failed".into()));
        }
        self.loaded = Some(path.into());
        Ok(())
    }

    fn transcribe(
        &mut self,
        _a: &[f32],
        _l: &str,
        _h: &[String],
    ) -> Result<Vec<TranscriptSegment>, TranscribeError> {
        self.transcripts += 1;
        Ok(Vec::new())
    }
}

#[test]
fn the_speech_model_check_loads_the_model_and_does_not_transcribe() {
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("model.bin");
    std::fs::write(&model, "m").unwrap();
    let runtime = runtime();
    let yaml = format!(
        "{{transcription: {{model: {}, models: {{}}}}}}",
        model.display()
    );
    let plan = plan(dir.path(), &yaml, Provider::Answers, &runtime);
    let mut whisper = Whisper::default();
    let (check, details) = run_one(check_whisper, &plan, &mut whisper);
    assert_eq!(check.status, Status::Pass, "{}", check.detail);
    assert_eq!(whisper.loaded.as_deref(), model.to_str());
    assert_eq!(whisper.transcripts, 0, "the check loads only");
    assert!(details.iter().any(|d| d.starts_with("Loading ")));
    whisper.fail_load = true;
    let (check, _) = run_one(check_whisper, &plan, &mut whisper);
    assert_eq!(check.detail, "RuntimeError: model load failed");
    assert!(
        check
            .fix
            .starts_with("Check the network for the first download")
    );
}

#[test]
fn an_unknown_speech_model_names_the_setting() {
    let dir = tempfile::tempdir().unwrap();
    let runtime = runtime();
    let plan = plan(
        dir.path(),
        "{transcription: {model: no-such-model, models: {}}}",
        Provider::Answers,
        &runtime,
    );
    let (check, _) = run_one(check_whisper, &plan, &mut Whisper::default());
    assert_eq!(check.status, Status::Fail);
    assert_eq!(
        check.fix,
        "Set transcription.model to a model listed in Settings > Models."
    );
}

#[cfg(target_os = "macos")]
struct NoHelper;

#[cfg(target_os = "macos")]
impl AudioSource for NoHelper {
    fn open(&self, _sample_rate: u32, _channels: u16) -> Result<(), AudioError> {
        Err(AudioError(
            "Native binary not found: /nowhere/coreaudio-tap\nRun: bash scripts/build_native.sh"
                .into(),
        ))
    }
    fn read(&self, _timeout: Duration) -> Result<Option<Vec<f32>>, AudioError> {
        Ok(None)
    }
    fn close(&self) -> Result<(), AudioError> {
        Ok(())
    }
}

#[cfg(target_os = "macos")]
#[test]
fn a_missing_tap_helper_says_to_restore_it_not_to_grant_permission() {
    let mut check = Check::new("system", "System audio", false);
    let notify: Notify<'_> = &|_: &Check| {};
    let mut running = Running {
        check: &mut check,
        notify,
    };
    let error = check_system_source(&mut running, &NoHelper).unwrap_err();
    assert!(error.starts_with("RuntimeError: Native binary not found"));
    assert!(check.fix.contains("coreaudio-tap"));
    assert!(!check.fix.contains("Privacy"));
}

#[cfg(target_os = "linux")]
#[test]
fn the_linux_hints_name_no_macos_settings_or_helper() {
    for text in [MIC_PERMISSION_FIX, SYSTEM_PERMISSION_FIX, SYSTEM_HELPER_FIX] {
        assert!(!text.contains("System Settings"), "{text}");
        assert!(!text.contains("coreaudio-tap"), "{text}");
    }
}

/// A fixed system source: every read returns `block`, and the tests count reads.
struct FixedSource {
    block: Option<Vec<f32>>,
    reads: std::sync::atomic::AtomicUsize,
    continuous: bool,
}

impl AudioSource for FixedSource {
    fn open(&self, _sample_rate: u32, _channels: u16) -> Result<(), AudioError> {
        Ok(())
    }

    fn read(&self, _timeout: Duration) -> Result<Option<Vec<f32>>, AudioError> {
        self.reads.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(self.block.clone())
    }

    fn close(&self) -> Result<(), AudioError> {
        Ok(())
    }

    fn continuous_capture(&self) -> bool {
        self.continuous
    }
}

const TONE_WINDOW: Duration = Duration::from_millis(30);

/// Run the Windows check against a fake source and return the final check.
fn windows_check(source: &FixedSource, play_tone: impl Fn() -> bool) -> Check {
    let mut check = Check::new("system", "System audio", false);
    check.fix = SYSTEM_PERMISSION_FIX.into();
    let notify: Notify<'_> = &|_: &Check| {};
    let mut running = Running {
        check: &mut check,
        notify,
    };
    verify_windows_capture(&mut running, source, TONE_WINDOW, play_tone).unwrap();
    running.check.clone()
}

#[test]
fn the_windows_check_plays_the_tone_before_it_reads_the_loopback() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let source = FixedSource {
        block: Some(vec![0.5; 160]),
        reads: AtomicUsize::new(0),
        continuous: true,
    };
    let reads_at_tone = AtomicUsize::new(usize::MAX);
    let check = windows_check(&source, || {
        reads_at_tone.store(source.reads.load(Ordering::SeqCst), Ordering::SeqCst);
        true
    });
    assert_eq!(
        reads_at_tone.load(Ordering::SeqCst),
        0,
        "the tone must play before the first read"
    );
    assert_eq!(check.status, Status::Pass);
    assert_eq!(check.detail, "Test tone captured");
}

#[test]
fn a_windows_tone_that_never_arrives_names_its_cause() {
    let no_device = FixedSource {
        block: Some(vec![0.5; 160]),
        reads: Default::default(),
        continuous: true,
    };
    let check = windows_check(&no_device, || false);
    assert_eq!(check.status, Status::Warn);
    assert!(check.detail.starts_with("No output device"));

    let none = FixedSource {
        block: None,
        reads: Default::default(),
        continuous: true,
    };
    let check = windows_check(&none, || true);
    assert_eq!(check.status, Status::Fail);
    assert_eq!(check.detail, "System audio capture did not start");
    assert_eq!(check.fix, SYSTEM_PERMISSION_FIX);

    let zeros = FixedSource {
        block: Some(vec![0.0; 160]),
        reads: Default::default(),
        continuous: true,
    };
    let check = windows_check(&zeros, || true);
    assert_eq!(check.status, Status::Fail);
    assert_eq!(
        check.detail,
        "System audio is silent: the capture permission is missing"
    );
}

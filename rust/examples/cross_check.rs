//! Reads and writes real files for `scripts/cross_check.py`, which compares the results with Python.

use std::path::{Path, PathBuf};

use lazy_take_notes::adapters::assets::store::{CachedModelStore, LegacyDirs};
use lazy_take_notes::adapters::config_file::{load_raw, write_config};
use lazy_take_notes::adapters::model_catalog;
use lazy_take_notes::adapters::paths::Paths;
use lazy_take_notes::adapters::session_reader::list_sessions;
use lazy_take_notes::adapters::templates::{YamlTemplates, parse_template};
use lazy_take_notes::config::{build_app_config, build_infra_config};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let arg = |i: usize| PathBuf::from(&args[i]);
    let paths = Paths::for_user().expect("home folder");
    let out = match args[0].as_str() {
        "config" => {
            let raw = load_raw(&paths, Some(&arg(1)), None).unwrap();
            serde_json::json!({"app": build_app_config(&raw).unwrap(), "infra": build_infra_config(&raw).unwrap()})
        }
        // Rewrite a copy of a config file through the Rust writer.
        "rewrite-config" => {
            write_config(
                &paths,
                &yaml_serde::Value::Mapping(Default::default()),
                Some(&arg(1)),
                &[],
            )
            .unwrap();
            serde_json::json!("ok")
        }
        "template" => serde_json::to_value(read_template(&arg(1))).unwrap(),
        // Save a template through the Rust writer into <dir>/config/templates/<name>.yaml.
        "save-template" => {
            let templates = YamlTemplates::new(&Paths::under(&arg(2)));
            let path = templates
                .save_user_template(&read_template(&arg(1)), &args[3])
                .unwrap();
            serde_json::json!(path)
        }
        "sessions" => {
            let names: Vec<String> = list_sessions(&arg(1)).into_iter().map(|s| s.name).collect();
            serde_json::json!(names)
        }
        "assets" => {
            // cache root, pywhispercpp models folder, legacy nemotron store
            let platform = model_catalog::platform_key();
            let legacy = LegacyDirs {
                pywhispercpp_models: arg(2),
                nemotron_store: arg(3),
            };
            let store = CachedModelStore::with(
                arg(1),
                model_catalog::catalog(&platform),
                &platform,
                legacy,
            );
            let mut states = serde_json::Map::new();
            for state in store.list_assets().unwrap() {
                let path = state.path.map(|p| {
                    Path::new(&p)
                        .strip_prefix(arg(1))
                        .unwrap()
                        .display()
                        .to_string()
                });
                let row = serde_json::json!([
                    state.status,
                    state.installed_bytes,
                    state.legacy_bytes,
                    path
                ]);
                states.insert(state.asset.id.to_string(), row);
            }
            serde_json::Value::Object(states)
        }
        "install" => {
            // cache root, asset id. No legacy folders, so the asset is downloaded.
            let platform = model_catalog::platform_key();
            let legacy = LegacyDirs {
                pywhispercpp_models: arg(1).join("none"),
                nemotron_store: arg(1).join("none"),
            };
            let store = CachedModelStore::with(
                arg(1),
                model_catalog::catalog(&platform),
                &platform,
                legacy,
            );
            let mut reports = 0;
            let path = store
                .install(&args[2], &mut |_| reports += 1, None)
                .unwrap();
            let state = store.status(&args[2]).unwrap();
            serde_json::json!({"path": path, "status": state.status, "bytes": state.installed_bytes, "progress_reports": reports})
        }
        "llm" => llm_case(&args[1], &args[2], &args[3]),
        "llm-real" => llm_real(&paths),
        "whisper" => whisper_chunk(&args[1..]),
        "transcribe-file" => transcribe_file(&args[1..]),
        "tone" => tone_reaches_tap(),
        "list-devices" => {
            // Print the input list every 0.5 s, to see devices that appear while running.
            let started = std::time::Instant::now();
            while started.elapsed().as_secs_f64() < args[1].parse::<f64>().unwrap() {
                let names = lazy_take_notes::adapters::audio::microphone::list_input_devices();
                eprintln!("{:5.1}s {names:?}", started.elapsed().as_secs_f64());
                std::thread::sleep(std::time::Duration::from_millis(500));
            }
            serde_json::Value::Null
        }
        "mic-levels" => mic_levels(&args[1], args[2].parse().unwrap()),
        "record" => record(
            &paths,
            &arg(1),
            args[2].parse().unwrap(),
            args.get(3).cloned(),
        ),
        "decode" => {
            let samples = lazy_take_notes::adapters::audio_file::load_audio_file(&arg(1)).unwrap();
            let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
            std::fs::write(arg(2), bytes).unwrap();
            serde_json::json!(samples.len())
        }
        other => panic!("unknown command {other}"),
    };
    println!("{out}");
}

fn read_template(path: &Path) -> lazy_take_notes::domain::template::SessionTemplate {
    parse_template(
        &std::fs::read_to_string(path).unwrap(),
        &path.display().to_string(),
    )
    .unwrap()
}

/// One LLM client call against a test server, as JSON: {"ok": ...} or {"err": "..."}.
fn llm_case(provider: &str, operation: &str, base: &str) -> serde_json::Value {
    use lazy_take_notes::adapters::llm::{ollama::OllamaLlmClient, openai::OpenAiCompatClient};
    use lazy_take_notes::ports::llm::LlmClient;
    let client: Box<dyn LlmClient> = match provider {
        "ollama" => Box::new(OllamaLlmClient::new(base)),
        _ => Box::new(OpenAiCompatClient::new(
            std::env::var("LTN_TEST_KEY").ok(),
            &format!("{base}/v1"),
        )),
    };
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        match operation {
            "chat" => match client.chat_single("m", "p").await {
                Ok(text) => serde_json::json!({"ok": text}),
                Err(error) => serde_json::json!({"err": error.to_string()}),
            },
            "conn" => match client.check_connectivity().await {
                Ok(()) => serde_json::json!({"ok": null}),
                Err(text) => serde_json::json!({"err": text}),
            },
            _ => serde_json::json!({"ok": client.check_models(&["x".to_string()]).await}),
        }
    })
}

/// One tiny request to the provider in the user's real config. Prints no secret.
fn llm_real(paths: &Paths) -> serde_json::Value {
    use lazy_take_notes::adapters::llm::{ollama::OllamaLlmClient, openai::OpenAiCompatClient};
    use lazy_take_notes::ports::llm::LlmClient;
    let raw = load_raw(paths, None, None).unwrap();
    let (app, infra) = (
        build_app_config(&raw).unwrap(),
        build_infra_config(&raw).unwrap(),
    );
    // LTN_VIA_OPENAI=1 drives the OpenAI client against Ollama's OpenAI-compatible /v1 API.
    let via_openai = std::env::var("LTN_VIA_OPENAI").is_ok() && infra.llm_provider == "ollama";
    let client: Box<dyn LlmClient> = match infra.llm_provider.as_str() {
        _ if via_openai => {
            let base = format!("{}/v1", OllamaLlmClient::new(&infra.ollama.host).base_url());
            Box::new(OpenAiCompatClient::new(Some("ollama".into()), &base))
        }
        "ollama" => Box::new(OllamaLlmClient::new(&infra.ollama.host)),
        _ => Box::new(OpenAiCompatClient::new(
            infra.openai.api_key.clone(),
            &infra.openai.base_url,
        )),
    };
    let model = app.interactive.model.clone();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let started = std::time::Instant::now();
        let connectivity = client.check_connectivity().await;
        let missing = client.check_models(std::slice::from_ref(&model)).await;
        let reply = client.chat(&model, &[lazy_take_notes::domain::chat::ChatMessage::user("Reply with the word ok.")]).await;
        serde_json::json!({
            "provider": if via_openai { "openai client on ollama /v1" } else { infra.llm_provider.as_str() },
            "connectivity": connectivity.is_ok(),
            "model_missing": !missing.is_empty(),
            "chat": reply.as_ref().map(|_| "ok").unwrap_or("fail"),
            "error_kind": reply.as_ref().err().map(|e| e.kind.clone()),
            "reply_chars": reply.as_ref().map(|r| r.content.len()).unwrap_or(0),
            "prompt_tokens": reply.as_ref().map(|r| r.prompt_tokens).unwrap_or(0),
            "seconds": started.elapsed().as_secs_f64(),
        })
    })
}

/// Transcribe 25 s of a file from an offset: program, model, file, language, offset seconds.
fn whisper_chunk(args: &[String]) -> serde_json::Value {
    use lazy_take_notes::adapters::whisper::{
        WORKER_COMMAND, transcriber::SubprocessWhisperTranscriber,
    };
    use lazy_take_notes::ports::transcriber::Transcriber;
    let audio =
        lazy_take_notes::adapters::audio_file::load_audio_file(Path::new(&args[2])).unwrap();
    let start = args[4].parse::<usize>().unwrap() * 16_000;
    let chunk = &audio[start..(start + 25 * 16_000).min(audio.len())];
    let mut whisper = SubprocessWhisperTranscriber::with_program(
        PathBuf::from(&args[0]),
        vec![WORKER_COMMAND.into()],
    );
    let started = std::time::Instant::now();
    whisper.load_model(&args[1]).unwrap();
    let loaded = started.elapsed().as_secs_f64();
    let started = std::time::Instant::now();
    let segments = whisper.transcribe(chunk, &args[3], &[]).unwrap();
    let chunk_seconds = started.elapsed().as_secs_f64();
    whisper.close();
    serde_json::json!({"load_s": loaded, "chunk_s": chunk_seconds, "segments": segments})
}

/// Transcribe the start of a file the way the file worker does: one second per feed,
/// a chunk when the use case triggers, a flush at the end.
/// Args: program, model, audio file, language, seconds, template file (for its hints).
fn transcribe_file(args: &[String]) -> serde_json::Value {
    use lazy_take_notes::adapters::whisper::{
        WORKER_COMMAND, transcriber::SubprocessWhisperTranscriber,
    };
    use lazy_take_notes::ports::transcriber::Transcriber;
    use lazy_take_notes::usecases::transcribe_audio::{TranscribeAudio, TranscribeSettings};
    const SAMPLE_RATE: usize = 16_000;
    let audio =
        lazy_take_notes::adapters::audio_file::load_audio_file(Path::new(&args[2])).unwrap();
    let audio = &audio[..(args[4].parse::<usize>().unwrap() * SAMPLE_RATE).min(audio.len())];
    let hints = read_template(Path::new(&args[5])).recognition_hints;
    let mut whisper = SubprocessWhisperTranscriber::with_program(
        PathBuf::from(&args[0]),
        vec![WORKER_COMMAND.into()],
    );
    whisper.load_model(&args[1]).unwrap();
    let mut use_case = TranscribeAudio::new(args[3].clone(), TranscribeSettings::default(), hints);
    let mut segments = Vec::new();
    let started = std::time::Instant::now();
    let mut offset = 0;
    while offset < audio.len() {
        let chunk = &audio[offset..(offset + SAMPLE_RATE).min(audio.len())];
        offset += chunk.len();
        use_case.feed_audio(chunk);
        use_case.set_session_offset(offset as f64 / SAMPLE_RATE as f64);
        if use_case.should_trigger() {
            segments.extend(use_case.process_buffer(&mut whisper, false).unwrap());
        }
    }
    segments.extend(use_case.flush(&mut whisper).unwrap());
    let total = started.elapsed().as_secs_f64();
    let started = std::time::Instant::now();
    whisper.close();
    serde_json::json!({"segments": segments, "total_s": total, "close_s": started.elapsed().as_secs_f64()})
}

/// Record the real mic and system audio into DIR for SECONDS, as a session does.
/// Prints the health on each change and the delivered samples against wall time.
/// DEVICE, when given, replaces the configured mic.
fn record(paths: &Paths, dir: &Path, seconds: f64, device: Option<String>) -> serde_json::Value {
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use lazy_take_notes::adapters::audio::coreaudio_tap::CoreAudioTapSource;
    use lazy_take_notes::adapters::audio::microphone::{MicrophoneSource, play_test_tone};
    use lazy_take_notes::adapters::audio::mixed::{MAX_MIC_GAIN, MixTiming, MixedAudioSource};
    use lazy_take_notes::adapters::audio::session_folder::SessionFolder;
    use lazy_take_notes::adapters::audio::wav::start_processed_recorder;
    use lazy_take_notes::ports::audio_source::AudioSource;

    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_target(true)
        .init();
    let app = build_app_config(&load_raw(paths, None, None).unwrap()).unwrap();
    let tap = Arc::new(CoreAudioTapSource::new(false));
    let mic = Arc::new(MicrophoneSource::new(
        device.or(app.transcription.input_device.clone()),
    ));
    let source = MixedAudioSource::new(
        mic,
        tap.clone(),
        app.transcription.silence_threshold,
        MAX_MIC_GAIN,
        Some(Box::new(play_test_tone)),
        MixTiming::default(),
    );
    std::fs::create_dir_all(dir).unwrap();
    let folder = Arc::new(SessionFolder::new(dir));
    let notice = Box::new(|message: String| eprintln!("NOTICE {message}"));
    let (sender, writer) = start_processed_recorder(folder, 16_000, Some(notice), None).unwrap();
    source.open(16_000, 1).unwrap();
    let started = Instant::now();
    let (mut delivered, mut health, mut pid) = (0usize, None, None);
    while started.elapsed().as_secs_f64() < seconds {
        if let Some(block) = source.read(Duration::from_millis(100)).unwrap() {
            delivered += block.len();
            sender.send(Some(block)).unwrap();
        }
        let now = source.health().expect("the mixer reports health");
        if health != Some(now) {
            eprintln!(
                "HEALTH {:.1}s {now:?} errors={:?}",
                started.elapsed().as_secs_f64(),
                source.source_errors()
            );
            health = Some(now);
        }
        if tap.pid() != pid {
            pid = tap.pid();
            eprintln!("TAP PID {pid:?}");
        }
    }
    let wall = started.elapsed().as_secs_f64();
    source.close().unwrap();
    sender.send(None).unwrap();
    writer.join().unwrap();
    serde_json::json!({
        "wall_s": wall,
        "delivered_s": delivered as f64 / 16_000.0,
        "pace": delivered as f64 / 16_000.0 / wall,
        "dropped_frames": source.dropped_frames(),
        "mic_notice": source.device_notice(),
    })
}

/// Play the test tone and measure 997 Hz in the tap before and during it.
fn tone_reaches_tap() -> serde_json::Value {
    use std::time::{Duration, Instant};

    use lazy_take_notes::adapters::audio::coreaudio_tap::CoreAudioTapSource;
    use lazy_take_notes::adapters::audio::microphone::play_test_tone;
    use lazy_take_notes::ports::audio_source::AudioSource;

    // Amplitude of one frequency (Goertzel), robust to other sound in the mix.
    fn level(samples: &[f32], hz: f64) -> f64 {
        let w = std::f64::consts::TAU * hz / 16_000.0;
        let (mut s1, mut s2) = (0.0f64, 0.0f64);
        for &x in samples {
            let s0 = f64::from(x) + 2.0 * w.cos() * s1 - s2;
            s2 = s1;
            s1 = s0;
        }
        (s1 * s1 + s2 * s2 - 2.0 * w.cos() * s1 * s2).sqrt() * 2.0 / samples.len() as f64
    }
    let collect = |tap: &CoreAudioTapSource, seconds: f64| {
        let (started, mut out) = (Instant::now(), Vec::new());
        while started.elapsed().as_secs_f64() < seconds {
            if let Some(block) = tap.read(Duration::from_millis(50)).unwrap() {
                out.extend(block);
            }
        }
        out
    };
    let tap = CoreAudioTapSource::new(false);
    tap.open(16_000, 1).unwrap();
    collect(&tap, 1.0);
    let before = collect(&tap, 0.6);
    let player = std::thread::spawn(play_test_tone);
    let during = collect(&tap, 0.6);
    let played = player.join().unwrap();
    tap.close().unwrap();
    serde_json::json!({
        "played": played,
        "before_997hz": level(&before, 997.0),
        "during_997hz": level(&during, 997.0),
        "expected": 0.0005,
    })
}

/// RMS of SECONDS from one input device through MicrophoneSource, with channels=1.
fn mic_levels(device: &str, seconds: f64) -> serde_json::Value {
    use std::time::{Duration, Instant};

    use lazy_take_notes::adapters::audio::microphone::MicrophoneSource;
    use lazy_take_notes::ports::audio_source::AudioSource;

    let mic = MicrophoneSource::new(Some(device.to_string()));
    mic.open(16_000, 1).unwrap();
    let (started, mut samples) = (Instant::now(), Vec::new());
    while started.elapsed().as_secs_f64() < seconds {
        if let Some(block) = mic.read(Duration::from_millis(50)).unwrap() {
            samples.extend(block);
        }
    }
    mic.close().unwrap();
    let rms = (samples
        .iter()
        .map(|&s| f64::from(s) * f64::from(s))
        .sum::<f64>()
        / samples.len().max(1) as f64)
        .sqrt();
    serde_json::json!({"samples": samples.len(), "rms": rms, "notice": mic.device_notice()})
}

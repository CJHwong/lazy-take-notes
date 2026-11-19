//! The real program as the whisper child: the hidden subcommand, the protocol on
//! stdout while whisper.cpp prints its errors, and the init error text.

use lazy_take_notes::adapters::whisper::WORKER_COMMAND;
use lazy_take_notes::adapters::whisper::transcriber::SubprocessWhisperTranscriber;
use lazy_take_notes::ports::transcriber::Transcriber;

fn program() -> std::path::PathBuf {
    env!("CARGO_BIN_EXE_lazy-take-notes").into()
}

#[test]
fn a_bad_model_fails_to_init_with_the_python_text() {
    let dir = tempfile::tempdir().unwrap();
    let bad = dir.path().join("not-a-model.bin");
    std::fs::write(&bad, b"not a ggml file").unwrap();
    let mut whisper =
        SubprocessWhisperTranscriber::with_program(program(), vec![WORKER_COMMAND.into()]);
    let error = whisper.load_model(bad.to_str().unwrap()).unwrap_err();
    assert!(
        error.0.starts_with("Whisper subprocess failed to init: "),
        "{}",
        error.0
    );
}

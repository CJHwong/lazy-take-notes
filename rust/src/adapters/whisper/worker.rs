//! The child side: load one model, then answer requests until the input closes.

use std::fs::{File, OpenOptions};
use std::io::{self, BufReader};
use std::os::fd::{AsRawFd, FromRawFd};

use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState,
};

use super::protocol::{Reply, Segment, read_request, write_reply};

/// The parameters pywhispercpp 1.4.1 passed, set one by one so a whisper.cpp
/// update with other defaults does not change the transcripts.
fn params<'a>(language: &'a str, prompt: &'a str) -> FullParams<'a, 'a> {
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 5 });
    params.set_n_threads(4);
    params.set_n_max_text_ctx(16384);
    params.set_no_context(true);
    params.set_single_segment(false);
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(true);
    params.set_token_timestamps(false);
    params.set_suppress_blank(true);
    params.set_temperature(0.0);
    params.set_temperature_inc(0.2);
    params.set_entropy_thold(2.4);
    params.set_logprob_thold(-1.0);
    params.set_no_speech_thold(0.6);
    params.set_max_initial_ts(1.0);
    params.set_length_penalty(-1.0);
    params.set_language(Some(language));
    if !prompt.is_empty() {
        params.set_initial_prompt(prompt);
    }
    params
}

/// Flash attention on, as in the whisper.cpp pywhispercpp bundles. Off, a 25 s chunk
/// took 25 to 50 % longer; on, chunks match Python in time and in every segment.
fn context_params() -> WhisperContextParameters<'static> {
    let mut params = WhisperContextParameters::default();
    params.flash_attn(true);
    params
}

struct Engine {
    // The state holds a reference to the context; the context must outlive it.
    _context: WhisperContext,
    state: WhisperState,
}

impl Engine {
    fn load(model_path: &str) -> Result<Self, String> {
        let context = WhisperContext::new_with_params(model_path, context_params())
            .map_err(|e| e.to_string())?;
        let state = context.create_state().map_err(|e| e.to_string())?;
        Ok(Self {
            _context: context,
            state,
        })
    }

    fn transcribe(
        &mut self,
        audio: &[f32],
        language: &str,
        hints: &[String],
    ) -> Result<Vec<Segment>, String> {
        let prompt = hints.join(" ");
        self.state
            .full(params(language, &prompt), audio)
            .map_err(|e| e.to_string())?;
        let mut segments = Vec::new();
        for index in 0..self.state.full_n_segments() {
            let Some(segment) = self.state.get_segment(index) else {
                continue;
            };
            let text = segment.to_str_lossy().map_err(|e| e.to_string())?;
            let text = text.trim();
            if !text.is_empty() {
                // whisper.cpp counts in centiseconds.
                let (start, end) = (
                    segment.start_timestamp() as f64 / 100.0,
                    segment.end_timestamp() as f64 / 100.0,
                );
                segments.push(Segment {
                    text: text.to_string(),
                    start,
                    end,
                });
            }
        }
        Ok(segments)
    }
}

/// Keep stdout for the protocol, then send fds 1 and 2 to /dev/null: whisper.cpp
/// and ggml print with fprintf, which would corrupt the replies and the TUI.
fn protocol_output() -> io::Result<File> {
    let null = OpenOptions::new().write(true).open("/dev/null")?;
    // SAFETY: dup and dup2 on descriptors this process owns; the new descriptor gets one owner.
    unsafe {
        let kept = libc::dup(1);
        if kept < 0 {
            return Err(io::Error::last_os_error());
        }
        if libc::dup2(null.as_raw_fd(), 1) < 0 || libc::dup2(null.as_raw_fd(), 2) < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(File::from_raw_fd(kept))
    }
}

/// The child's main. Returns the process exit code.
pub fn run(model_path: &str) -> i32 {
    let Ok(mut out) = protocol_output() else {
        return 1;
    };
    let mut engine = match Engine::load(model_path) {
        Ok(engine) => engine,
        Err(error) => {
            let _ = write_reply(&mut out, &Reply::Error { error });
            return 1;
        }
    };
    if write_reply(&mut out, &Reply::Ready).is_err() {
        return 1;
    }
    let mut input = BufReader::new(io::stdin().lock());
    loop {
        let (request, audio) = match read_request(&mut input) {
            Ok(Some(request)) => request,
            Ok(None) => return 0,
            Err(_) => return 1,
        };
        let reply = match engine.transcribe(&audio, &request.language, &request.hints) {
            Ok(segments) => Reply::Ok { segments },
            Err(error) => Reply::Error { error },
        };
        if write_reply(&mut out, &reply).is_err() {
            return 1;
        }
    }
}

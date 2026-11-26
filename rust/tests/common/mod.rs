//! Fakes that implement the ports, for use case tests. No library is mocked.
#![allow(dead_code)]

use std::sync::Mutex;

use async_trait::async_trait;
use lazy_take_notes::domain::chat::ChatMessage;
use lazy_take_notes::domain::template::SessionTemplate;
use lazy_take_notes::domain::transcript::TranscriptSegment;
use lazy_take_notes::ports::llm::{ChatResponse, LlmClient, LlmError};
use lazy_take_notes::ports::transcriber::{TranscribeError, Transcriber};

/// What the fake LLM does on the next call.
#[derive(Clone)]
pub enum Reply {
    Text(String, u64),
    Fail(LlmError),
    /// Never answers, like a stalled server.
    Hang,
    /// A bug in the client: the call panics.
    Panic(&'static str),
}

pub struct FakeLlm {
    reply: Mutex<Reply>,
    pub chat_calls: Mutex<Vec<(String, Vec<ChatMessage>)>>,
    pub single_calls: Mutex<Vec<(String, String)>>,
}

impl FakeLlm {
    pub fn new(reply: Reply) -> Self {
        Self {
            reply: Mutex::new(reply),
            chat_calls: Mutex::default(),
            single_calls: Mutex::default(),
        }
    }

    pub fn text(content: &str) -> Self {
        Self::new(Reply::Text(content.into(), 0))
    }

    pub fn set(&self, reply: Reply) {
        *self.reply.lock().unwrap() = reply;
    }

    pub fn last_messages(&self) -> Vec<ChatMessage> {
        self.chat_calls
            .lock()
            .unwrap()
            .last()
            .expect("a chat call")
            .1
            .clone()
    }

    async fn answer(&self) -> Result<ChatResponse, LlmError> {
        let reply = self.reply.lock().unwrap().clone();
        match reply {
            Reply::Text(content, prompt_tokens) => Ok(ChatResponse {
                content,
                prompt_tokens,
            }),
            Reply::Fail(error) => Err(error),
            Reply::Hang => std::future::pending().await,
            Reply::Panic(message) => panic!("{message}"),
        }
    }
}

#[async_trait]
impl LlmClient for FakeLlm {
    async fn chat(&self, model: &str, messages: &[ChatMessage]) -> Result<ChatResponse, LlmError> {
        self.chat_calls
            .lock()
            .unwrap()
            .push((model.into(), messages.to_vec()));
        self.answer().await
    }

    async fn chat_single(&self, model: &str, prompt: &str) -> Result<String, LlmError> {
        self.single_calls
            .lock()
            .unwrap()
            .push((model.into(), prompt.into()));
        Ok(self.answer().await?.content)
    }

    async fn check_connectivity(&self) -> Result<(), String> {
        Ok(())
    }

    async fn check_models(&self, _models: &[String]) -> Vec<String> {
        Vec::new()
    }
}

/// Returns queued replies in order, then the default segments.
#[derive(Default)]
pub struct FakeTranscriber {
    pub segments: Vec<TranscriptSegment>,
    pub queued: Vec<Vec<TranscriptSegment>>,
    /// (audio, language, hints) of each call.
    pub calls: Vec<(Vec<f32>, String, Vec<String>)>,
}

impl FakeTranscriber {
    pub fn returning(segments: Vec<TranscriptSegment>) -> Self {
        Self {
            segments,
            ..Default::default()
        }
    }

    pub fn queue(replies: Vec<Vec<TranscriptSegment>>) -> Self {
        Self {
            queued: replies,
            ..Default::default()
        }
    }
}

impl Transcriber for FakeTranscriber {
    fn load_model(&mut self, _model_path: &str) -> Result<(), TranscribeError> {
        Ok(())
    }

    fn transcribe(
        &mut self,
        audio: &[f32],
        language: &str,
        hints: &[String],
    ) -> Result<Vec<TranscriptSegment>, TranscribeError> {
        self.calls
            .push((audio.to_vec(), language.into(), hints.to_vec()));
        if self.queued.is_empty() {
            return Ok(self.segments.clone());
        }
        Ok(self.queued.remove(0))
    }
}

/// Noise that rises and falls at syllable rate, like speech: 0.4 s on, 0.1 s off.
pub fn speech_like(samples: usize, amplitude: f32) -> Vec<f32> {
    let mut state: u64 = 0x2545_f491;
    (0..samples)
        .map(|index| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let on = (index as f64 / 16000.0) % 0.5 < 0.4;
            // Sum of uniforms: close enough to Gaussian noise for a level gate.
            let noise = ((state >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 3.46;
            if on { (noise as f32) * amplitude } else { 0.0 }
        })
        .collect()
}

pub fn built_in_template(name: &str) -> SessionTemplate {
    let path = format!(
        "{}/tests/fixtures/templates.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let all: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    serde_json::from_value(all[name].clone()).unwrap()
}

pub fn segment(text: &str, start: f64, end: f64) -> TranscriptSegment {
    TranscriptSegment::new(text, start, end)
}

/// What the fake persistence was asked to write. Shared, because the controller owns the fake.
#[derive(Default)]
pub struct Saved {
    pub transcript: Vec<(Vec<TranscriptSegment>, bool)>,
    pub digest: Vec<(String, u32)>,
    pub history: Vec<(String, u32, bool)>,
    pub context: Vec<String>,
    pub template: Vec<String>,
}

/// Records each write. A method named in `failing` fails like a full disk instead.
pub struct FakePersistence {
    pub saved: std::sync::Arc<Mutex<Saved>>,
    pub failing: Vec<&'static str>,
}

impl FakePersistence {
    pub fn new() -> (Self, std::sync::Arc<Mutex<Saved>>) {
        Self::failing(&[])
    }

    pub fn failing(methods: &[&'static str]) -> (Self, std::sync::Arc<Mutex<Saved>>) {
        let saved = std::sync::Arc::new(Mutex::new(Saved::default()));
        let fake = Self {
            saved: saved.clone(),
            failing: methods.to_vec(),
        };
        (fake, saved)
    }

    fn check(&self, method: &str) -> std::io::Result<std::path::PathBuf> {
        if self.failing.contains(&method) {
            return Err(std::io::Error::other("[Errno 28] No space left on device"));
        }
        Ok(std::path::PathBuf::from("/fake/output"))
    }
}

impl lazy_take_notes::ports::persistence::Persistence for FakePersistence {
    fn save_transcript_lines(
        &mut self,
        segments: &[TranscriptSegment],
        append: bool,
    ) -> std::io::Result<std::path::PathBuf> {
        let path = self.check("save_transcript_lines")?;
        self.saved
            .lock()
            .unwrap()
            .transcript
            .push((segments.to_vec(), append));
        Ok(path)
    }

    fn save_digest_md(
        &mut self,
        markdown: &str,
        number: u32,
    ) -> std::io::Result<std::path::PathBuf> {
        let path = self.check("save_digest_md")?;
        self.saved
            .lock()
            .unwrap()
            .digest
            .push((markdown.into(), number));
        Ok(path)
    }

    fn save_history(
        &mut self,
        markdown: &str,
        number: u32,
        is_final: bool,
    ) -> std::io::Result<std::path::PathBuf> {
        let path = self.check("save_history")?;
        self.saved
            .lock()
            .unwrap()
            .history
            .push((markdown.into(), number, is_final));
        Ok(path)
    }

    fn save_session_context(&mut self, context: &str) -> std::io::Result<std::path::PathBuf> {
        let path = self.check("save_session_context")?;
        self.saved.lock().unwrap().context.push(context.into());
        Ok(path)
    }

    fn save_template_name(&mut self, name: &str) -> std::io::Result<std::path::PathBuf> {
        let path = self.check("save_template_name")?;
        self.saved.lock().unwrap().template.push(name.into());
        Ok(path)
    }
}

/// Seeded uniform noise of `amplitude`, as Python's fixtures make it. `syllables`
/// silences the last 0.1 s of every 0.5 s, like the gaps between words.
pub fn lcg_audio(seed: u64, seconds: f64, amplitude: f64, syllables: bool) -> Vec<f32> {
    let count = (seconds * 16000.0) as usize;
    let mut state = seed;
    (0..count)
        .map(|index| {
            state = (1_664_525 * state + 1_013_904_223) % (1 << 32);
            let mut value = state as f64 / (1u64 << 32) as f64 * 2.0 - 1.0;
            if syllables && (index as f64 / 16000.0) % 0.5 >= 0.4 {
                value = 0.0;
            }
            (value * amplitude) as f32
        })
        .collect()
}

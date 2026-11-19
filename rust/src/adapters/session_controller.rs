//! The session's business state: transcript, digest history, notes and speakers.
//!
//! The controller never holds its state across an await. A digest runs in three
//! calls: `begin_digest` adds the prompt, the caller sends `request.messages`
//! to the LLM, and `finish_digest` applies the reply, saves the notes and
//! compacts the history. A replaced digest calls `abandon_digest`. Headless
//! mode owns the controller in one task, so it can use `run_digest`.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use tracing::{error, warn};

use crate::domain::config::AppConfig;
use crate::domain::digest_state::DigestState;
use crate::domain::speaker_turn::SpeakerTurn;
use crate::domain::template::SessionTemplate;
use crate::domain::transcript::TranscriptSegment;
use crate::ports::llm::{ChatResponse, LlmClient, LlmError};
use crate::ports::persistence::Persistence;
use crate::usecases::compact::compact_messages;
use crate::usecases::digest::{self, DigestInput, DigestRequest, DigestResult};
use crate::usecases::label::generate_label;
use crate::usecases::quick_action::{QuickActionInput, RECENT_LINES, run_quick_action};
use crate::usecases::speaker_attribution::speaker_for_span;

/// How far a line may sit from a turn and still take its speaker, in seconds.
const SPEAKER_TOLERANCE_S: f64 = 1.5;

pub struct SessionController {
    config: AppConfig,
    template: SessionTemplate,
    llm: Arc<dyn LlmClient>,
    persistence: Box<dyn Persistence>,
    pub digest_state: DigestState,
    pub all_segments: Vec<TranscriptSegment>,
    pub latest_digest: Option<String>,
    pub user_context: String,
    pub speaker_turns: Vec<SpeakerTurn>,
    pub speaker_names: HashMap<u32, String>,
    persist_error: Option<String>,
}

impl SessionController {
    pub fn new(
        config: AppConfig,
        template: SessionTemplate,
        llm: Arc<dyn LlmClient>,
        persistence: Box<dyn Persistence>,
    ) -> Self {
        let digest_state = DigestState::new(&template.system_prompt);
        Self {
            config,
            template,
            llm,
            persistence,
            digest_state,
            all_segments: Vec::new(),
            latest_digest: None,
            user_context: String::new(),
            speaker_turns: Vec::new(),
            speaker_names: HashMap::new(),
            persist_error: None,
        }
    }

    pub fn config(&self) -> &AppConfig {
        &self.config
    }

    pub fn template(&self) -> &SessionTemplate {
        &self.template
    }

    pub fn llm(&self) -> Arc<dyn LlmClient> {
        self.llm.clone()
    }

    pub fn persistence_mut(&mut self) -> &mut dyn Persistence {
        self.persistence.as_mut()
    }

    /// Run one disk write. A full or vanished disk must not end the recording.
    fn persist<T>(
        &mut self,
        what: &str,
        write: impl FnOnce(&mut dyn Persistence) -> std::io::Result<T>,
    ) {
        if let Err(e) = write(self.persistence.as_mut()) {
            error!(target: "ltn.controller", "Could not save {what}: {e}");
            self.persist_error = Some(format!("Could not save {what}: {e}"));
        }
    }

    /// The newest unreported write failure. Reading it clears it.
    pub fn take_persist_error(&mut self) -> Option<String> {
        self.persist_error.take()
    }

    /// Save the session context outside a final digest, e.g. on a shutdown signal.
    pub fn save_context_now(&mut self) {
        if self.user_context.trim().is_empty() || !self.config.output.save_context {
            return;
        }
        let context = self.user_context.clone();
        self.persist("session context", |p| p.save_session_context(&context));
    }

    pub fn set_speaker_turns(&mut self, turns: Vec<SpeakerTurn>) {
        self.speaker_turns = turns;
    }

    pub fn set_speaker_name(&mut self, speaker: u32, name: impl Into<String>) {
        self.speaker_names.insert(speaker, name.into());
    }

    /// The mapped name, else `S<n>`, else None when no turn covers the line.
    pub fn speaker_label(&self, segment: &TranscriptSegment) -> Option<String> {
        let speaker = speaker_for_span(
            &self.speaker_turns,
            segment.wall_start,
            segment.wall_end,
            SPEAKER_TOLERANCE_S,
        )?;
        Some(
            self.speaker_names
                .get(&speaker)
                .filter(|name| !name.is_empty())
                .cloned()
                .unwrap_or_else(|| format!("S{speaker}")),
        )
    }

    fn labeled_lines(&self, segments: &[TranscriptSegment]) -> Vec<String> {
        segments
            .iter()
            .map(|segment| match self.speaker_label(segment) {
                Some(label) => format!("{label}: {}", segment.text),
                None => segment.text.clone(),
            })
            .collect()
    }

    /// Keep and save new lines. True when a digest should start now.
    pub fn on_transcript_segments(&mut self, segments: &[TranscriptSegment]) -> bool {
        self.all_segments.extend_from_slice(segments);
        for segment in segments {
            self.digest_state.buffer.push(segment.text.clone());
            self.digest_state.all_lines.push(segment.text.clone());
        }
        self.persist("transcript", |p| p.save_transcript_lines(segments, true));
        let digest = &self.config.digest;
        digest::should_trigger_digest(
            &self.digest_state,
            digest.min_lines,
            digest.min_interval,
            digest.max_lines,
            Instant::now(),
        )
    }

    /// Add the digest prompt to the history. Send `request.messages` to `digest_model`.
    pub fn begin_digest(&mut self, is_final: bool) -> Result<DigestRequest, String> {
        // The prompt carries the same speaker labels the screen shows.
        let buffer_lines = (!self.speaker_turns.is_empty()).then(|| {
            let start = self
                .all_segments
                .len()
                .saturating_sub(self.digest_state.buffer.len());
            self.labeled_lines(&self.all_segments[start..])
        });
        let full_transcript = match (is_final, self.speaker_turns.is_empty()) {
            (false, _) => String::new(),
            (true, true) => self.digest_state.all_lines.join("\n"),
            (true, false) => self.labeled_lines(&self.all_segments).join("\n"),
        };
        let input = DigestInput {
            is_final,
            full_transcript: &full_transcript,
            user_context: &self.user_context,
            buffer_lines: buffer_lines.as_deref(),
        };
        digest::begin(&mut self.digest_state, &self.template, input)
    }

    pub fn digest_model(&self) -> &str {
        &self.config.digest.model
    }

    /// Apply the reply. A success saves the notes and compacts a long history.
    pub fn finish_digest(
        &mut self,
        request: &DigestRequest,
        is_final: bool,
        reply: Result<ChatResponse, LlmError>,
    ) -> DigestResult {
        let markdown = digest::finish(&mut self.digest_state, request, reply)?;
        self.latest_digest = Some(markdown.clone());
        let number = self.digest_state.digest_count;
        self.persist("notes", |p| p.save_digest_md(&markdown, number));
        if self.config.output.save_notes_history {
            self.persist("notes history", |p| {
                p.save_history(&markdown, number, is_final)
            });
        }
        if is_final {
            self.save_context_now();
        }
        if self.digest_state.prompt_tokens > self.config.digest.compact_token_threshold {
            compact_messages(
                &mut self.digest_state,
                &markdown,
                &self.template.system_prompt,
            );
        }
        Ok(markdown)
    }

    /// A newer digest replaced this one.
    pub fn abandon_digest(&mut self, request: &DigestRequest) {
        digest::abandon(&mut self.digest_state, request);
    }

    /// A whole digest cycle, for a caller that owns the controller in one task.
    pub async fn run_digest(&mut self, is_final: bool) -> DigestResult {
        let request = self.begin_digest(is_final)?;
        let llm = self.llm.clone();
        let reply = llm.chat(&self.config.digest.model, &request.messages).await;
        self.finish_digest(&request, is_final, reply)
    }

    /// A short folder label from the latest notes. None without notes or on failure.
    pub async fn generate_label(&self) -> Option<String> {
        self.label_job()?.await
    }

    /// The label request with its own copy of the inputs, to run on another task.
    pub fn label_job(&self) -> Option<impl Future<Output = Option<String>> + Send + 'static> {
        let latest = self.latest_digest.clone()?;
        let llm = self.llm.clone();
        let metadata = self.template.metadata.clone();
        let model = self.config.interactive.model.clone();
        Some(async move {
            let label = generate_label(
                llm.as_ref(),
                &metadata.name,
                &metadata.description,
                &latest,
                &model,
            )
            .await;
            match label {
                Ok(label) if !label.is_empty() => Some(label),
                Ok(_) => None,
                Err(e) => {
                    warn!(target: "ltn.controller", "Auto-label generation failed: {e}");
                    None
                }
            }
        })
    }

    /// Run the quick action on `key`: (reply, label), or None for a key with no action.
    pub async fn run_quick_action(&self, key: &str) -> Option<Result<(String, String), String>> {
        self.quick_action_job(key).await
    }

    /// The quick action with its own copy of the inputs, to run on another task.
    pub fn quick_action_job(
        &self,
        key: &str,
    ) -> impl Future<Output = Option<Result<(String, String), String>>> + Send + 'static {
        let llm = self.llm.clone();
        let template = self.template.clone();
        let model = self.config.interactive.model.clone();
        let latest = self.latest_digest.clone();
        let start = self.all_segments.len().saturating_sub(RECENT_LINES);
        let segments = self.all_segments[start..].to_vec();
        let context = self.user_context.clone();
        let key = key.to_string();
        async move {
            let input = QuickActionInput {
                latest_digest: latest.as_deref(),
                segments: &segments,
                user_context: &context,
            };
            run_quick_action(llm.as_ref(), &key, &template, &model, input).await
        }
    }

    /// Point later writes at the renamed session folder.
    pub fn relocate(&mut self, new_dir: std::path::PathBuf) {
        self.persistence.relocate(new_dir);
    }
}

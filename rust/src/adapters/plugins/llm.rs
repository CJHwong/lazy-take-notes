//! `LlmClient` for a provider that a Python plugin registers.
//!
//! The host starts on the first call, not when the client is built: the first
//! start can install Python and packages, and building a client must not block
//! the screen. One host serves every client in the process.

use std::sync::Mutex;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::sync::OnceCell;
use tracing::warn;

use super::host::{ErrorKind, PluginError, PluginHost};
use crate::domain::chat::ChatMessage;
use crate::ports::llm::{ChatResponse, LLM_REQUEST_TIMEOUT, LlmClient, LlmError};

/// The plugin's factory and its checks; a model call has `LLM_REQUEST_TIMEOUT`.
const SETUP_LIMIT: Duration = Duration::from_secs(60);

/// The running host and the specs it was started with.
static SHARED: Mutex<Option<(Vec<String>, PluginHost)>> = Mutex::new(None);

/// The host for `specs`: the running one, or a new one when there is none, it
/// died, or the plugin list changed. Blocks while a host starts.
pub fn shared_host(specs: &[String]) -> Result<PluginHost, PluginError> {
    let mut shared = SHARED.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((started_with, host)) = shared.as_ref()
        && started_with.as_slice() == specs
        && host.is_alive()
    {
        return Ok(host.clone());
    }
    // The old host, if any, ends when its last client goes.
    *shared = None;
    let host = PluginHost::start(specs)?;
    *shared = Some((specs.to_vec(), host.clone()));
    Ok(host)
}

type Connection = Result<(PluginHost, Value), PluginError>;

pub struct PluginLlmClient {
    provider: String,
    /// The `InfraConfig` the plugin's factory gets, plugin keys included.
    config: Value,
    specs: Vec<String>,
    /// The built-in provider names, for the "Unknown provider" message.
    builtin: Vec<String>,
    /// A host to use instead of the shared one.
    preset: Option<PluginHost>,
    connection: OnceCell<Connection>,
}

impl PluginLlmClient {
    pub fn new(provider: &str, config: Value, specs: Vec<String>, builtin: Vec<String>) -> Self {
        Self {
            provider: provider.to_string(),
            config,
            specs,
            builtin,
            preset: None,
            connection: OnceCell::new(),
        }
    }

    /// A client on a host that already runs (tests use a fake host).
    pub fn on_host(provider: &str, host: PluginHost, builtin: Vec<String>) -> Self {
        let mut client = Self::new(provider, json!({}), Vec::new(), builtin);
        client.preset = Some(host);
        client
    }

    /// The host and this client's id there. A failure is kept: each call then
    /// gives the same message at once, and a new session tries again.
    async fn connect(&self) -> Result<&(PluginHost, Value), PluginError> {
        let connection = self
            .connection
            .get_or_init(|| async {
                let host = match &self.preset {
                    Some(host) => host.clone(),
                    None => {
                        let specs = self.specs.clone();
                        tokio::task::spawn_blocking(move || shared_host(&specs))
                            .await
                            .map_err(|e| {
                                PluginError::new(
                                    ErrorKind::Start,
                                    format!("The plugin host start failed: {e}"),
                                )
                            })??
                    }
                };
                if !host.hello.llm_providers.contains(&self.provider) {
                    return Err(PluginError::new(ErrorKind::Start, self.unknown(&host)));
                }
                let params = json!({"provider": self.provider, "config": self.config});
                let result = host.call("llm.create", params, SETUP_LIMIT).await?;
                let client = result.get("client").cloned().unwrap_or(Value::Null);
                Ok((host, client))
            })
            .await;
        connection.as_ref().map_err(Clone::clone)
    }

    /// Python's message: the built-in names, then the plugins'.
    fn unknown(&self, host: &PluginHost) -> String {
        let quoted = crate::adapters::pyrepr::py_repr_json(&Value::String(self.provider.clone()));
        let available: Vec<&str> = self
            .builtin
            .iter()
            .chain(&host.hello.llm_providers)
            .map(String::as_str)
            .collect();
        format!(
            "Unknown provider {quoted}. Available: {}",
            available.join(", ")
        )
    }

    async fn call(
        &self,
        method: &str,
        mut params: Value,
        limit: Duration,
    ) -> Result<Value, PluginError> {
        let (host, client) = self.connect().await?;
        params["client"] = client.clone();
        host.call(method, params, limit).await
    }
}

fn llm_error(error: PluginError) -> LlmError {
    match error.kind {
        ErrorKind::Raised if !error.python_type.is_empty() => {
            LlmError::new(error.python_type, error.message)
        }
        _ => LlmError::new("PluginError", error.message),
    }
}

fn text(value: &Value, key: &str) -> String {
    value
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string()
}

#[async_trait]
impl LlmClient for PluginLlmClient {
    async fn chat(&self, model: &str, messages: &[ChatMessage]) -> Result<ChatResponse, LlmError> {
        let params = json!({"model": model, "messages": messages});
        let result = self
            .call("llm.chat", params, LLM_REQUEST_TIMEOUT)
            .await
            .map_err(llm_error)?;
        Ok(ChatResponse {
            content: text(&result, "content"),
            prompt_tokens: result
                .get("prompt_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0),
        })
    }

    async fn chat_single(&self, model: &str, prompt: &str) -> Result<String, LlmError> {
        let params = json!({"model": model, "prompt": prompt});
        let result = self
            .call("llm.chat_single", params, LLM_REQUEST_TIMEOUT)
            .await
            .map_err(llm_error)?;
        Ok(text(&result, "content"))
    }

    async fn check_connectivity(&self) -> Result<(), String> {
        let result = self
            .call("llm.check_connectivity", json!({}), SETUP_LIMIT)
            .await
            .map_err(|e| e.to_string())?;
        match result.get("ok").and_then(Value::as_bool) {
            Some(true) => Ok(()),
            _ => Err(text(&result, "message")),
        }
    }

    /// A failed check names no model: the connectivity check reports the cause.
    async fn check_models(&self, models: &[String]) -> Vec<String> {
        let params = json!({"models": models});
        match self.call("llm.check_models", params, SETUP_LIMIT).await {
            Ok(result) => result
                .get("missing")
                .and_then(Value::as_array)
                .map(|names| {
                    names
                        .iter()
                        .filter_map(|n| n.as_str().map(str::to_string))
                        .collect()
                })
                .unwrap_or_default(),
            Err(error) => {
                warn!(target: "ltn.plugins", "check_models failed: {error}");
                Vec::new()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HELLO: &str = r#"{"jsonrpc":"2.0","method":"hello","params":{"protocol":1,"llm_providers":["claude-code"]}}"#;

    fn client(provider: &str, replies: &[&str]) -> PluginLlmClient {
        let mut script = format!("echo '{HELLO}'; read line; ");
        for reply in replies {
            script.push_str(&format!("echo '{reply}'; read line; "));
        }
        let mut command = std::process::Command::new("/bin/sh");
        command.arg("-c").arg(script);
        let host = PluginHost::start_command(command, Duration::from_secs(5)).unwrap();
        PluginLlmClient::on_host(provider, host, vec!["ollama".into(), "openai".into()])
    }

    fn block_on<T>(future: impl std::future::Future<Output = T>) -> T {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(future)
    }

    #[test]
    fn a_plugin_provider_chats_and_checks_through_the_host() {
        let llm = client(
            "claude-code",
            &[
                r#"{"jsonrpc":"2.0","id":1,"result":{"client":"c1"}}"#,
                r#"{"jsonrpc":"2.0","id":2,"result":{"ok":true,"message":""}}"#,
                r#"{"jsonrpc":"2.0","id":3,"result":{"content":"notes","prompt_tokens":42}}"#,
                r#"{"jsonrpc":"2.0","id":4,"result":{"missing":["opus"]}}"#,
            ],
        );
        block_on(async {
            assert_eq!(llm.check_connectivity().await, Ok(()));
            let response = llm
                .chat("sonnet", &[ChatMessage::user("hi")])
                .await
                .unwrap();
            assert_eq!(
                response,
                ChatResponse {
                    content: "notes".into(),
                    prompt_tokens: 42
                }
            );
            assert_eq!(llm.check_models(&["opus".into()]).await, ["opus"]);
        });
    }

    #[test]
    fn a_name_no_plugin_provides_lists_every_provider() {
        let llm = client("claude", &[]);
        let error = block_on(llm.check_connectivity()).unwrap_err();
        assert_eq!(
            error,
            "Unknown provider 'claude'. Available: ollama, openai, claude-code"
        );
    }

    #[test]
    fn a_plugin_exception_keeps_its_python_class() {
        let llm = client(
            "claude-code",
            &[
                r#"{"jsonrpc":"2.0","id":1,"result":{"client":"c1"}}"#,
                r#"{"jsonrpc":"2.0","id":2,"error":{"code":-32000,"message":"usage limit","data":{"type":"RuntimeError"}}}"#,
            ],
        );
        let error = block_on(llm.chat("sonnet", &[ChatMessage::user("hi")])).unwrap_err();
        assert_eq!(error, LlmError::new("RuntimeError", "usage limit"));
    }
}

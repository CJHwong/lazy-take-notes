//! Ollama over its REST API, with the requests and error texts of `ollama-python`.

use async_trait::async_trait;
use serde_json::{Value, json};

use super::http_client;
use crate::adapters::pyrepr::py_str_json;
use crate::domain::chat::ChatMessage;
use crate::ports::llm::{ChatResponse, LLM_REQUEST_TIMEOUT, LlmClient, LlmError};

const DEFAULT_PORT: u16 = 11434;
const CONNECTION_ERROR_MESSAGE: &str = "Failed to connect to Ollama. Please check that Ollama is downloaded, running and accessible. https://ollama.com/download";

/// One event of `ollama pull`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PullProgress {
    pub status: String,
    pub digest: Option<String>,
    pub total: Option<u64>,
    pub completed: Option<u64>,
}

pub struct OllamaLlmClient {
    base: String,
    /// `OLLAMA_API_KEY`, sent as a Bearer token, as `ollama-python` does.
    api_key: Option<String>,
}

impl OllamaLlmClient {
    /// An empty host falls back to `OLLAMA_HOST`, then to 127.0.0.1:11434.
    pub fn new(host: &str) -> Self {
        let env_host = std::env::var("OLLAMA_HOST").unwrap_or_default();
        let host = if host.is_empty() {
            env_host.as_str()
        } else {
            host
        };
        let api_key = std::env::var("OLLAMA_API_KEY")
            .ok()
            .filter(|key| !key.is_empty());
        Self {
            base: parse_host(host),
            api_key,
        }
    }

    pub fn base_url(&self) -> &str {
        &self.base
    }

    /// `(name, size in bytes)` of every installed model.
    pub async fn list_models(&self) -> Result<Vec<(String, u64)>, LlmError> {
        let tags = self
            .request(reqwest::Method::GET, "/api/tags", None, false)
            .await?;
        let models = tags["models"].as_array().cloned().unwrap_or_default();
        Ok(models
            .iter()
            .filter_map(|model| {
                let name = model["model"]
                    .as_str()
                    .filter(|n| !n.is_empty())
                    .or(model["name"].as_str())?;
                (!name.is_empty()).then(|| (name.to_string(), model["size"].as_u64().unwrap_or(0)))
            })
            .collect())
    }

    /// Pull a model and report each progress event. An `error` event fails the pull.
    pub async fn pull_model(
        &self,
        model: &str,
        on_progress: &mut (dyn FnMut(PullProgress) + Send),
    ) -> Result<(), LlmError> {
        let body = json!({"model": model, "stream": true, "insecure": false});
        let mut response = self
            .send(reqwest::Method::POST, "/api/pull", Some(body), false)
            .await?;
        let mut pending = Vec::new();
        while let Some(chunk) = response.chunk().await.map_err(transport_error)? {
            pending.extend_from_slice(&chunk);
            while let Some(end) = pending.iter().position(|&byte| byte == b'\n') {
                let line: Vec<u8> = pending.drain(..=end).collect();
                pull_event(&line, on_progress)?;
            }
        }
        pull_event(&pending, on_progress)
    }

    async fn send(
        &self,
        method: reqwest::Method,
        path: &str,
        body: Option<Value>,
        timeout: bool,
    ) -> Result<reqwest::Response, LlmError> {
        // The chat calls time out like the Python async client. The others had no timeout.
        let limit = timeout.then_some(LLM_REQUEST_TIMEOUT);
        let client = http_client(limit, limit)?;
        let mut request = client
            .request(method, format!("{}{path}", self.base))
            .header("content-type", "application/json")
            .header("accept", "application/json");
        if let Some(key) = &self.api_key {
            request = request.bearer_auth(key);
        }
        if let Some(body) = body {
            request = request.body(body.to_string());
        }
        let response = request.send().await.map_err(transport_error)?;
        let status = response.status();
        if status.is_success() {
            return Ok(response);
        }
        let text = response.text().await.unwrap_or_default();
        Err(response_error(&text, i64::from(status.as_u16())))
    }

    async fn request(
        &self,
        method: reqwest::Method,
        path: &str,
        body: Option<Value>,
        timeout: bool,
    ) -> Result<Value, LlmError> {
        let response = self.send(method, path, body, timeout).await?;
        let text = response.text().await.map_err(transport_error)?;
        serde_json::from_str(&text).map_err(|e| LlmError::new("JSONDecodeError", e.to_string()))
    }
}

#[async_trait]
impl LlmClient for OllamaLlmClient {
    async fn chat(&self, model: &str, messages: &[ChatMessage]) -> Result<ChatResponse, LlmError> {
        let body = json!({"model": model, "stream": false, "messages": messages, "tools": []});
        let reply = self
            .request(reqwest::Method::POST, "/api/chat", Some(body), true)
            .await?;
        Ok(ChatResponse {
            content: reply["message"]["content"]
                .as_str()
                .unwrap_or("")
                .to_string(),
            prompt_tokens: reply["prompt_eval_count"].as_u64().unwrap_or(0),
        })
    }

    async fn check_connectivity(&self) -> Result<(), String> {
        match self
            .request(reqwest::Method::GET, "/api/tags", None, false)
            .await
        {
            // Another server on the port can answer 200 too. Only Ollama sends a model list.
            Ok(tags) if tags["models"].is_array() => Ok(()),
            Ok(_) => Err("Cannot connect to Ollama: the server sent no model list".into()),
            Err(error) => Err(format!("Cannot connect to Ollama: {}", error.message)),
        }
    }

    /// A model the server answers with an HTTP error is missing. Any other failure
    /// reports none missing: the connectivity check reports that one.
    async fn check_models(&self, models: &[String]) -> Vec<String> {
        let mut missing = Vec::new();
        for model in models {
            let body = json!({"model": model});
            match self
                .send(reqwest::Method::POST, "/api/show", Some(body), false)
                .await
            {
                Ok(_) => {}
                Err(error) if error.kind == "ResponseError" => missing.push(model.clone()),
                Err(_) => return Vec::new(),
            }
        }
        missing
    }
}

fn pull_event(
    line: &[u8],
    on_progress: &mut (dyn FnMut(PullProgress) + Send),
) -> Result<(), LlmError> {
    let text = String::from_utf8_lossy(line);
    if text.trim().is_empty() {
        return Ok(());
    }
    let event: Value = serde_json::from_str(text.trim())
        .map_err(|e| LlmError::new("JSONDecodeError", e.to_string()))?;
    let error = &event["error"];
    if !error.is_null() && error != "" && error != false {
        // A stream error has no HTTP status; ollama-python reports -1.
        return Err(response_error(&py_str_json(error), -1));
    }
    on_progress(PullProgress {
        status: event["status"].as_str().unwrap_or("").to_string(),
        digest: event["digest"].as_str().map(str::to_string),
        total: event["total"].as_u64(),
        completed: event["completed"].as_u64(),
    });
    Ok(())
}

/// `ResponseError`: the body's `error` field when it is JSON, else the raw body.
fn response_error(text: &str, status: i64) -> LlmError {
    let reason = match serde_json::from_str::<Value>(text) {
        Ok(Value::Object(map)) => map
            .get("error")
            .map_or_else(|| text.to_string(), py_str_json),
        _ => text.to_string(),
    };
    LlmError::new("ResponseError", format!("{reason} (status code: {status})"))
}

fn transport_error(error: reqwest::Error) -> LlmError {
    if error.is_timeout() {
        let kind = if error.is_connect() {
            "ConnectTimeout"
        } else {
            "ReadTimeout"
        };
        return LlmError::new(kind, "timed out");
    }
    if error.is_connect() {
        return LlmError::new("ConnectionError", CONNECTION_ERROR_MESSAGE);
    }
    LlmError::new("ReadError", error.to_string())
}

/// The host forms `ollama-python` accepts: a bare host, `host:port`, `:port`, a URL, or IPv6 in brackets.
pub fn parse_host(host: &str) -> String {
    let (mut scheme, mut hostport, mut port) = ("http", host, DEFAULT_PORT);
    match host.split_once("://") {
        Some((_, "")) | None => {}
        Some((given, rest)) => {
            (scheme, hostport) = (given, rest);
            port = match given {
                "http" => 80,
                "https" => 443,
                _ => DEFAULT_PORT,
            };
        }
    }
    let netloc_end = hostport.find(['/', '?', '#']).unwrap_or(hostport.len());
    let (netloc, rest) = hostport.split_at(netloc_end);
    let path = &rest[..rest.find(['?', '#']).unwrap_or(rest.len())];
    let netloc = netloc.rsplit_once('@').map_or(netloc, |(_, after)| after);
    let (name, given_port, ipv6) = match netloc
        .strip_prefix('[')
        .and_then(|inner| inner.split_once(']'))
    {
        Some((address, after)) => (address.to_string(), after.strip_prefix(':'), true),
        None => match netloc.rsplit_once(':') {
            Some((name, given)) => (name.to_string(), Some(given), false),
            None => (netloc.to_string(), None, false),
        },
    };
    let mut name = name.to_lowercase();
    if name.is_empty() {
        name = "127.0.0.1".into();
    } else if ipv6 {
        name = format!("[{name}]");
    }
    let port = given_port
        .and_then(|p| p.parse::<u16>().ok())
        .unwrap_or(port);
    match path.trim_matches('/') {
        "" => format!("{scheme}://{name}:{port}"),
        path => format!("{scheme}://{name}:{port}/{path}"),
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_server::{dead_base, reply, serve};
    use super::*;

    #[test]
    fn parse_host_matches_the_ollama_python_doctests() {
        let cases = [
            ("", "http://127.0.0.1:11434"),
            ("1.2.3.4", "http://1.2.3.4:11434"),
            (":56789", "http://127.0.0.1:56789"),
            ("1.2.3.4:56789", "http://1.2.3.4:56789"),
            ("http://1.2.3.4", "http://1.2.3.4:80"),
            ("https://1.2.3.4", "https://1.2.3.4:443"),
            ("https://1.2.3.4:56789", "https://1.2.3.4:56789"),
            ("example.com", "http://example.com:11434"),
            ("example.com:56789", "http://example.com:56789"),
            ("http://example.com", "http://example.com:80"),
            ("https://example.com", "https://example.com:443"),
            ("https://example.com:56789", "https://example.com:56789"),
            ("example.com/", "http://example.com:11434"),
            ("example.com:56789/", "http://example.com:56789"),
            ("example.com/path", "http://example.com:11434/path"),
            ("example.com:56789/path", "http://example.com:56789/path"),
            (
                "https://example.com:56789/path",
                "https://example.com:56789/path",
            ),
            ("example.com:56789/path/", "http://example.com:56789/path"),
            (
                "[0001:002:003:0004::1]",
                "http://[0001:002:003:0004::1]:11434",
            ),
            (
                "[0001:002:003:0004::1]:56789",
                "http://[0001:002:003:0004::1]:56789",
            ),
            (
                "http://[0001:002:003:0004::1]",
                "http://[0001:002:003:0004::1]:80",
            ),
            (
                "https://[0001:002:003:0004::1]",
                "https://[0001:002:003:0004::1]:443",
            ),
            (
                "https://[0001:002:003:0004::1]:56789",
                "https://[0001:002:003:0004::1]:56789",
            ),
            (
                "[0001:002:003:0004::1]/",
                "http://[0001:002:003:0004::1]:11434",
            ),
            (
                "[0001:002:003:0004::1]:56789/",
                "http://[0001:002:003:0004::1]:56789",
            ),
            (
                "[0001:002:003:0004::1]/path",
                "http://[0001:002:003:0004::1]:11434/path",
            ),
            (
                "[0001:002:003:0004::1]:56789/path",
                "http://[0001:002:003:0004::1]:56789/path",
            ),
            (
                "https://[0001:002:003:0004::1]:56789/path",
                "https://[0001:002:003:0004::1]:56789/path",
            ),
            (
                "[0001:002:003:0004::1]:56789/path/",
                "http://[0001:002:003:0004::1]:56789/path",
            ),
            ("http://localhost:11434", "http://localhost:11434"),
        ];
        for (input, expected) in cases {
            assert_eq!(parse_host(input), expected, "{input}");
        }
    }

    fn client(base: &str) -> OllamaLlmClient {
        OllamaLlmClient {
            base: base.into(),
            api_key: Some("test-key".into()),
        }
    }

    #[tokio::test]
    async fn chat_sends_the_python_body_and_reads_content_and_tokens() {
        let server = serve(vec![reply(
            200,
            r#"{"message":{"role":"assistant","content":"hi"},"prompt_eval_count":7}"#,
        )]);
        let messages = [ChatMessage::system("sys"), ChatMessage::user("hello")];
        let response = client(&server.base).chat("m", &messages).await.unwrap();
        assert_eq!(
            response,
            ChatResponse {
                content: "hi".into(),
                prompt_tokens: 7
            }
        );
        let seen = server.seen.lock().unwrap()[0].clone();
        assert_eq!(
            (seen.method.as_str(), seen.path.as_str()),
            ("POST", "/api/chat")
        );
        assert_eq!(
            seen.body,
            r#"{"model":"m","stream":false,"messages":[{"role":"system","content":"sys"},{"role":"user","content":"hello"}],"tools":[]}"#
        );
        assert_eq!(seen.authorization.as_deref(), Some("Bearer test-key"));
    }

    #[tokio::test]
    async fn errors_read_like_ollama_python() {
        let server = serve(vec![
            reply(200, r#"{"status":"ok"}"#),
            reply(500, r#"{"error":"boom"}"#),
            reply(401, r#"{"error":{"message":"bad key","param":null}}"#),
            reply(400, "plain text error"),
        ]);
        let ollama = client(&server.base);
        assert_eq!(
            ollama.check_connectivity().await.unwrap_err(),
            "Cannot connect to Ollama: the server sent no model list"
        );
        assert_eq!(
            ollama.chat_single("m", "p").await.unwrap_err().to_string(),
            "ResponseError: boom (status code: 500)"
        );
        assert_eq!(
            ollama.chat_single("m", "p").await.unwrap_err().to_string(),
            "ResponseError: {'message': 'bad key', 'param': None} (status code: 401)"
        );
        assert_eq!(
            ollama.check_connectivity().await.unwrap_err(),
            "Cannot connect to Ollama: plain text error (status code: 400)"
        );
        let dead = client(&dead_base());
        assert_eq!(
            dead.chat_single("m", "p").await.unwrap_err().to_string(),
            format!("ConnectionError: {CONNECTION_ERROR_MESSAGE}")
        );
        assert!(dead.check_models(&["x".into()]).await.is_empty());
    }

    #[tokio::test]
    async fn check_models_counts_any_http_error_as_missing() {
        let server = serve(vec![reply(200, "{}"), reply(500, r#"{"error":"boom"}"#)]);
        let missing = client(&server.base)
            .check_models(&["here".into(), "gone".into()])
            .await;
        assert_eq!(missing, vec!["gone".to_string()]);
        assert_eq!(server.seen.lock().unwrap()[1].body, r#"{"model":"gone"}"#);
    }

    #[tokio::test]
    async fn list_and_pull() {
        let tags = r#"{"models":[{"model":"llama3:8b","name":"llama3:8b","size":123},{"name":"old","size":5}]}"#;
        let stream = "{\"status\":\"pulling\",\"digest\":\"sha\",\"total\":10,\"completed\":4}\n{\"status\":\"success\"}\n";
        let server = serve(vec![
            reply(200, tags),
            reply(200, stream),
            reply(200, "{\"status\":\"x\"}\n{\"error\":\"disk full\"}\n"),
        ]);
        let ollama = client(&server.base);
        assert_eq!(
            ollama.list_models().await.unwrap(),
            vec![("llama3:8b".into(), 123), ("old".into(), 5)]
        );
        let mut events = Vec::new();
        ollama
            .pull_model("llama3", &mut |event| events.push(event))
            .await
            .unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(
            (
                events[0].completed,
                events[0].total,
                events[1].status.as_str()
            ),
            (Some(4), Some(10), "success")
        );
        assert_eq!(
            server.seen.lock().unwrap()[1].body,
            r#"{"model":"llama3","stream":true,"insecure":false}"#
        );
        let error = ollama.pull_model("llama3", &mut |_| {}).await.unwrap_err();
        assert_eq!(error.message, "disk full (status code: -1)");
    }
}

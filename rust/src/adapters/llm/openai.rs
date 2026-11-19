//! Any OpenAI-compatible API (OpenAI, Gemini, Groq, vLLM...), with the requests,
//! retries and error texts of the `openai` Python SDK.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use serde_json::{Value, json};

use super::http_client;
use super::pyrepr::py_repr_json;
use crate::domain::chat::ChatMessage;
use crate::ports::llm::{ChatResponse, LLM_REQUEST_TIMEOUT, LlmClient, LlmError};

const MAX_RETRIES: u32 = 2;
const INITIAL_RETRY_DELAY: f64 = 0.5;
const MAX_RETRY_DELAY: f64 = 8.0;
/// The SDK's default connect timeout, used by the checks that pass no timeout.
const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
const MISSING_KEY: &str = "The api_key client option must be set either by passing api_key to the client or by setting the OPENAI_API_KEY environment variable";

pub struct OpenAiCompatClient {
    /// None reads `OPENAI_API_KEY` at each call, as a new SDK client does.
    api_key: Option<String>,
    base_url: String,
}

/// A failed call, with what the checks need beyond the user-facing error.
struct Failure {
    error: LlmError,
    status: Option<u16>,
    /// The `error` member of a JSON body, else the whole body.
    body: Option<Value>,
}

impl From<LlmError> for Failure {
    fn from(error: LlmError) -> Self {
        Self {
            error,
            status: None,
            body: None,
        }
    }
}

impl OpenAiCompatClient {
    pub fn new(api_key: Option<String>, base_url: &str) -> Self {
        Self {
            api_key,
            base_url: base_url.trim_end_matches('/').to_string(),
        }
    }

    fn key(&self) -> Result<String, LlmError> {
        self.api_key
            .clone()
            .or_else(|| std::env::var("OPENAI_API_KEY").ok())
            .ok_or_else(|| LlmError::new("OpenAIError", MISSING_KEY))
    }

    /// One API call with the SDK's retries: two more tries after a connection
    /// error, a timeout, or a 408, 409, 429 or 5xx answer.
    async fn call(
        &self,
        method: reqwest::Method,
        path: &str,
        body: Option<&Value>,
        chat: bool,
    ) -> Result<Value, Failure> {
        let key = self.key()?;
        let (connect, read) = if chat {
            (LLM_REQUEST_TIMEOUT, LLM_REQUEST_TIMEOUT)
        } else {
            (DEFAULT_CONNECT_TIMEOUT, LLM_REQUEST_TIMEOUT)
        };
        let client = http_client(Some(connect), Some(read))?;
        let url = format!("{}{path}", self.base_url);
        let mut retries_taken = 0;
        loop {
            let mut request = client
                .request(method.clone(), &url)
                .bearer_auth(&key)
                .header("accept", "application/json");
            if let Some(body) = body {
                request = request
                    .header("content-type", "application/json")
                    .body(body.to_string());
            }
            let (failure, retry, headers) = match request.send().await {
                Err(error) => (transport_failure(&error), true, None),
                Ok(response) if response.status().is_success() => {
                    let text = response.text().await.map_err(|e| transport_failure(&e))?;
                    return serde_json::from_str(&text)
                        .map_err(|e| LlmError::new("JSONDecodeError", e.to_string()).into());
                }
                Ok(response) => {
                    let headers = response.headers().clone();
                    let status = response.status().as_u16();
                    let text = response.text().await.unwrap_or_default();
                    (
                        status_failure(status, &text),
                        should_retry(status, &headers),
                        Some(headers),
                    )
                }
            };
            if !retry || retries_taken >= MAX_RETRIES {
                return Err(failure);
            }
            tokio::time::sleep(retry_delay(retries_taken, headers.as_ref())).await;
            retries_taken += 1;
        }
    }
}

#[async_trait]
impl LlmClient for OpenAiCompatClient {
    async fn chat(&self, model: &str, messages: &[ChatMessage]) -> Result<ChatResponse, LlmError> {
        let body = json!({"messages": messages, "model": model});
        let reply = self
            .call(
                reqwest::Method::POST,
                "/chat/completions",
                Some(&body),
                true,
            )
            .await
            .map_err(|f| f.error)?;
        let Some(choice) = reply["choices"]
            .as_array()
            .and_then(|choices| choices.first())
        else {
            return Err(LlmError::new("IndexError", "list index out of range"));
        };
        Ok(ChatResponse {
            content: choice["message"]["content"]
                .as_str()
                .unwrap_or("")
                .to_string(),
            prompt_tokens: reply["usage"]["prompt_tokens"].as_u64().unwrap_or(0),
        })
    }

    async fn check_connectivity(&self) -> Result<(), String> {
        match self
            .call(reqwest::Method::GET, "/models", None, false)
            .await
        {
            Ok(_) => Ok(()),
            Err(failure) if failure.status == Some(401) => Err(format!(
                "Authentication failed: {}",
                server_message(&failure)
            )),
            Err(failure) => Err(format!(
                "Cannot connect to OpenAI-compatible API: {}",
                failure.error.message
            )),
        }
    }

    /// A 404 means missing. Any other failure reports none missing: many
    /// compatible providers have no models endpoint.
    async fn check_models(&self, models: &[String]) -> Vec<String> {
        let mut missing = Vec::new();
        for model in models {
            match self
                .call(
                    reqwest::Method::GET,
                    &format!("/models/{model}"),
                    None,
                    false,
                )
                .await
            {
                Ok(_) => {}
                Err(failure) if failure.status == Some(404) => missing.push(model.clone()),
                Err(_) => return Vec::new(),
            }
        }
        missing
    }
}

fn transport_failure(error: &reqwest::Error) -> Failure {
    if error.is_timeout() {
        return LlmError::new("APITimeoutError", "Request timed out.").into();
    }
    LlmError::new("APIConnectionError", "Connection error.").into()
}

/// The SDK's exception class and message for an HTTP error answer.
fn status_failure(status: u16, text: &str) -> Failure {
    let text = text.trim();
    let parsed = serde_json::from_str::<Value>(text).ok();
    let message = match &parsed {
        Some(body) => format!("Error code: {status} - {}", py_repr_json(body)),
        None if text.is_empty() => format!("Error code: {status}"),
        None => text.to_string(),
    };
    let body = match parsed {
        Some(Value::Object(mut map)) => Some(map.remove("error").unwrap_or(Value::Object(map))),
        Some(other) => Some(other),
        None => Some(Value::String(text.to_string())),
    };
    let kind = match status {
        400 => "BadRequestError",
        401 => "AuthenticationError",
        403 => "PermissionDeniedError",
        404 => "NotFoundError",
        409 => "ConflictError",
        422 => "UnprocessableEntityError",
        429 => "RateLimitError",
        500.. => "InternalServerError",
        _ => "APIStatusError",
    };
    Failure {
        error: LlmError::new(kind, message),
        status: Some(status),
        body,
    }
}

/// The server's own sentence. The error text holds the whole JSON body.
fn server_message(failure: &Failure) -> String {
    let inner = match &failure.body {
        Some(Value::Object(map)) => map
            .get("error")
            .unwrap_or(failure.body.as_ref().unwrap_or(&Value::Null)),
        other => other.as_ref().unwrap_or(&Value::Null),
    };
    match inner["message"].as_str() {
        Some(message) if !message.is_empty() => message.to_string(),
        _ => failure.error.message.clone(),
    }
}

fn should_retry(status: u16, headers: &reqwest::header::HeaderMap) -> bool {
    match headers.get("x-should-retry").and_then(|v| v.to_str().ok()) {
        Some("true") => return true,
        Some("false") => return false,
        _ => {}
    }
    matches!(status, 408 | 409 | 429) || status >= 500
}

/// The server's `retry-after-ms` or `retry-after` when it is 0 to 60 s, else
/// exponential backoff from 0.5 s, capped at 8 s, less up to 25 % jitter.
fn retry_delay(retries_taken: u32, headers: Option<&reqwest::header::HeaderMap>) -> Duration {
    let header = |name: &str| {
        headers?
            .get(name)?
            .to_str()
            .ok()?
            .trim()
            .parse::<f64>()
            .ok()
    };
    let asked = header("retry-after-ms")
        .map(|ms| ms / 1000.0)
        .or_else(|| header("retry-after"));
    if let Some(seconds) = asked.filter(|s| *s > 0.0 && *s <= 60.0) {
        return Duration::from_secs_f64(seconds);
    }
    let backoff =
        (INITIAL_RETRY_DELAY * 2f64.powi(retries_taken.min(1000) as i32)).min(MAX_RETRY_DELAY);
    Duration::from_secs_f64((backoff * (1.0 - 0.25 * jitter())).max(0.0))
}

/// A number in [0, 1). Spreading retries needs no strong randomness.
fn jitter() -> f64 {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.subsec_nanos());
    f64::from(nanos % 1_000_000) / 1_000_000.0
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::super::test_server::{Reply, dead_base, reply, serve};
    use super::*;

    fn client(base: &str) -> OpenAiCompatClient {
        OpenAiCompatClient::new(Some("test-key".into()), &format!("{base}/v1/"))
    }

    fn fast_retry(status: u16, body: &str) -> Reply {
        Reply {
            headers: vec![("retry-after-ms", "10".into())],
            ..reply(status, body)
        }
    }

    #[tokio::test]
    async fn chat_sends_the_sdk_body_and_reads_the_reply() {
        let body = r#"{"choices":[{"message":{"role":"assistant","content":"hi"}}],"usage":{"prompt_tokens":7}}"#;
        let server = serve(vec![reply(200, body)]);
        let messages = [ChatMessage::system("sys"), ChatMessage::user("hello")];
        let response = client(&server.base).chat("m", &messages).await.unwrap();
        assert_eq!(
            (response.content.as_str(), response.prompt_tokens),
            ("hi", 7)
        );
        let seen = server.seen.lock().unwrap()[0].clone();
        assert_eq!(
            (seen.method.as_str(), seen.path.as_str()),
            ("POST", "/v1/chat/completions")
        );
        assert_eq!(
            seen.body,
            r#"{"messages":[{"role":"system","content":"sys"},{"role":"user","content":"hello"}],"model":"m"}"#
        );
        assert_eq!(seen.authorization.as_deref(), Some("Bearer test-key"));
    }

    #[tokio::test]
    async fn status_errors_read_like_the_sdk_and_retry_like_it() {
        let auth = r#"{"error":{"message":"Incorrect API key provided","param":null}}"#;
        let server = serve(vec![fast_retry(500, r#"{"error":"boom"}"#)]);
        let openai = client(&server.base);
        let error = openai.chat_single("m", "p").await.unwrap_err();
        assert_eq!(
            error.to_string(),
            "InternalServerError: Error code: 500 - {'error': 'boom'}"
        );
        assert_eq!(server.seen.lock().unwrap().len(), 3);

        let server = serve(vec![reply(401, auth)]);
        let openai = client(&server.base);
        assert_eq!(
            openai.chat_single("m", "p").await.unwrap_err().to_string(),
            "AuthenticationError: Error code: 401 - {'error': {'message': 'Incorrect API key provided', 'param': None}}"
        );
        assert_eq!(
            openai.check_connectivity().await.unwrap_err(),
            "Authentication failed: Incorrect API key provided"
        );
        assert!(openai.check_models(&["x".into()]).await.is_empty());
        assert_eq!(server.seen.lock().unwrap().len(), 3, "a 401 is not retried");

        let server = serve(vec![reply(400, "plain text error"), reply(418, "")]);
        let openai = client(&server.base);
        assert_eq!(
            openai.chat_single("m", "p").await.unwrap_err().to_string(),
            "BadRequestError: plain text error"
        );
        assert_eq!(
            openai.check_connectivity().await.unwrap_err(),
            "Cannot connect to OpenAI-compatible API: Error code: 418"
        );
    }

    #[tokio::test]
    async fn check_models_404_is_missing() {
        let server = serve(vec![
            reply(200, "{}"),
            reply(404, r#"{"error":{"message":"nope"}}"#),
        ]);
        let missing = client(&server.base)
            .check_models(&["llama3:8b".into(), "gone".into()])
            .await;
        assert_eq!(missing, vec!["gone".to_string()]);
        assert_eq!(server.seen.lock().unwrap()[0].path, "/v1/models/llama3:8b");
    }

    #[tokio::test]
    async fn empty_choices_and_missing_key_and_dead_server() {
        let server = serve(vec![reply(200, r#"{"choices":[]}"#)]);
        assert_eq!(
            client(&server.base)
                .chat_single("m", "p")
                .await
                .unwrap_err()
                .to_string(),
            "IndexError: list index out of range"
        );
        let started = Instant::now();
        let dead = client(&dead_base());
        assert_eq!(
            dead.chat_single("m", "p").await.unwrap_err().to_string(),
            "APIConnectionError: Connection error."
        );
        // Two retries back off about 0.5 s and 1 s, less jitter.
        assert!(
            started.elapsed() >= Duration::from_millis(1100),
            "{:?}",
            started.elapsed()
        );
        let keyless = OpenAiCompatClient {
            api_key: None,
            base_url: dead_base(),
        };
        if std::env::var("OPENAI_API_KEY").is_err() {
            assert_eq!(
                keyless.check_connectivity().await.unwrap_err(),
                format!("Cannot connect to OpenAI-compatible API: {MISSING_KEY}")
            );
        }
    }

    #[test]
    fn retry_delay_honors_the_server_then_backs_off() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("retry-after", "3".parse().unwrap());
        assert_eq!(retry_delay(0, Some(&headers)), Duration::from_secs(3));
        headers.insert("retry-after", "120".parse().unwrap());
        let delay = retry_delay(1, Some(&headers));
        assert!(
            delay >= Duration::from_millis(750) && delay <= Duration::from_secs(1),
            "{delay:?}"
        );
        assert!(retry_delay(10, None) <= Duration::from_secs(8));
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("x-should-retry", "false".parse().unwrap());
        assert!(
            !should_retry(500, &headers) && should_retry(409, &reqwest::header::HeaderMap::new())
        );
    }
}

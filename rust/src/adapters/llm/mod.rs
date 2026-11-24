//! LLM clients for the `LlmClient` port: Ollama and any OpenAI-compatible API.
//!
//! Both send the same requests as the Python SDKs they replace, and report
//! failures with the same class names and text, because the user reads them
//! in the status bar, the preflight and Settings.

pub mod ollama;
pub mod openai;
#[cfg(test)]
mod test_server;

use std::time::Duration;

use crate::ports::llm::LlmError;

/// The Python SDKs use httpx, which speaks HTTP/1.1 only.
fn http_client(
    connect: Option<Duration>,
    read: Option<Duration>,
) -> Result<reqwest::Client, LlmError> {
    let mut builder = reqwest::Client::builder().http1_only();
    if let Some(connect) = connect {
        builder = builder.connect_timeout(connect);
    }
    if let Some(read) = read {
        builder = builder.read_timeout(read);
    }
    builder.build().map_err(|e| {
        LlmError::new(
            "RuntimeError",
            format!("could not start the HTTP client: {e}"),
        )
    })
}

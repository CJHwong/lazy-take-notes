//! OS errors printed as Python prints `str(OSError)`, because the user reads them in notices.

use std::fmt;
use std::io;
use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::adapters::pyrepr::py_repr_json;

/// An OS error with the paths Python would name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OsError {
    errno: Option<i32>,
    text: String,
    path: Option<PathBuf>,
    path2: Option<PathBuf>,
}

impl OsError {
    pub fn new(errno: i32, text: &str, path: Option<&Path>) -> Self {
        Self {
            errno: Some(errno),
            text: text.to_string(),
            path: path.map(Path::to_path_buf),
            path2: None,
        }
    }

    pub fn from_io(error: &io::Error, path: Option<&Path>, path2: Option<&Path>) -> Self {
        let text = error.to_string();
        let text = match error.raw_os_error() {
            Some(code) => text
                .strip_suffix(&format!(" (os error {code})"))
                .unwrap_or(&text)
                .to_string(),
            None => text,
        };
        Self {
            errno: error.raw_os_error(),
            text,
            path: path.map(Path::to_path_buf),
            path2: path2.map(Path::to_path_buf),
        }
    }
}

impl OsError {
    /// The error text alone, as Python's `OSError.strerror`.
    pub fn strerror(&self) -> &str {
        &self.text
    }
}

impl fmt::Display for OsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let quote = |path: &Path| py_repr_json(&Value::String(path.to_string_lossy().into_owned()));
        match self.errno {
            Some(code) => write!(f, "[Errno {code}] {}", self.text)?,
            None => write!(f, "{}", self.text)?,
        }
        if let Some(path) = &self.path {
            write!(f, ": {}", quote(path))?;
            if let Some(path2) = &self.path2 {
                write!(f, " -> {}", quote(path2))?;
            }
        }
        Ok(())
    }
}

impl std::error::Error for OsError {}

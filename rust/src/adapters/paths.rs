//! Where the app keeps its config, templates and model cache.
//!
//! Every gateway takes a `Paths` instead of reading a global, so tests point
//! them at temporary folders.

use std::path::{Path, PathBuf};

const APP_DIR: &str = "lazy-take-notes";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Paths {
    pub config_dir: PathBuf,
    pub cache_dir: PathBuf,
    pub home: PathBuf,
}

impl Paths {
    /// The same folders as Python's `platformdirs.user_config_path` and `user_cache_path`.
    /// macOS: `~/Library/Application Support/lazy-take-notes`, `~/Library/Caches/lazy-take-notes`.
    pub fn for_user() -> Option<Self> {
        Some(Self {
            config_dir: dirs::config_dir()?.join(APP_DIR),
            cache_dir: dirs::cache_dir()?.join(APP_DIR),
            home: dirs::home_dir()?,
        })
    }

    /// All folders under one root, for tests.
    pub fn under(root: &Path) -> Self {
        Self {
            config_dir: root.join("config"),
            cache_dir: root.join("cache"),
            home: root.join("home"),
        }
    }

    pub fn user_templates_dir(&self) -> PathBuf {
        self.config_dir.join("templates")
    }

    pub fn consent_noticed(&self) -> PathBuf {
        self.config_dir.join(".consent_noticed")
    }

    pub fn builtin_templates_noticed(&self) -> PathBuf {
        self.config_dir.join(".builtin_templates_noticed")
    }

    /// `config.yaml` first, then `config.yml`.
    pub fn default_config_paths(&self) -> [PathBuf; 2] {
        [
            self.config_dir.join("config.yaml"),
            self.config_dir.join("config.yml"),
        ]
    }

    pub fn legacy_nemotron_store(&self) -> PathBuf {
        self.home.join(".cache").join("nemotron-diarization")
    }

    /// Expand a leading `~` the way Python's `Path.expanduser` does.
    pub fn expand_user(&self, path: &str) -> PathBuf {
        match path.strip_prefix('~') {
            Some("") => self.home.clone(),
            Some(rest) if rest.starts_with('/') => self.home.join(&rest[1..]),
            _ => PathBuf::from(path),
        }
    }
}

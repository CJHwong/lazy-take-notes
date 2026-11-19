//! The `ModelResolver` port over the shared cache, for the transcription worker.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use super::store::CachedModelStore;
use crate::domain::errors::ModelResolutionError;
use crate::domain::model_asset::AssetProgress;
use crate::ports::model_resolver::ModelResolver;

pub type OnProgress = Box<dyn Fn(AssetProgress) + Send + Sync>;

pub struct CachedModelResolver {
    store: Arc<CachedModelStore>,
    on_progress: Option<OnProgress>,
    /// Set by the caller to stop a download. The partial file stays for the next try.
    pub cancel: Arc<AtomicBool>,
}

impl CachedModelResolver {
    pub fn new(store: Arc<CachedModelStore>, on_progress: Option<OnProgress>) -> Self {
        Self {
            store,
            on_progress,
            cancel: Arc::default(),
        }
    }
}

impl ModelResolver for CachedModelResolver {
    fn resolve(&self, model_name: &str) -> Result<String, ModelResolutionError> {
        let mut report = |progress: AssetProgress| {
            if let Some(callback) = &self.on_progress {
                callback(progress);
            }
        };
        self.store
            .resolve(model_name, &mut report, Some(&self.cancel))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;
    use crate::adapters::assets::store::LegacyDirs;

    #[test]
    fn resolves_a_file_path_and_reports_no_progress_for_it() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("model.bin");
        std::fs::write(&file, b"x").unwrap();
        let legacy = LegacyDirs {
            pywhispercpp_models: dir.path().join("p"),
            nemotron_store: dir.path().join("n"),
        };
        let store = Arc::new(CachedModelStore::with(
            dir.path().join("cache"),
            vec![],
            "test-os",
            legacy,
        ));
        let seen = Arc::new(Mutex::new(0));
        let counter = seen.clone();
        let resolver = CachedModelResolver::new(
            store,
            Some(Box::new(move |_| *counter.lock().unwrap() += 1)),
        );
        assert_eq!(
            resolver.resolve(file.to_str().unwrap()).unwrap(),
            file.display().to_string()
        );
        assert_eq!(*seen.lock().unwrap(), 0);
        assert!(resolver.resolve("no-such-model").is_err());
    }
}

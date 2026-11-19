use crate::domain::errors::ModelResolutionError;

/// Maps a model name to a usable local path.
pub trait ModelResolver: Send + Sync {
    fn resolve(&self, model_name: &str) -> Result<String, ModelResolutionError>;
}

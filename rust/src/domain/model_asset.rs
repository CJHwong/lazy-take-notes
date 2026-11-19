//! Entities for the shared model download cache.

use serde::{Deserialize, Serialize};

/// Lifecycle state of one downloadable asset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AssetStatus {
    Ready,
    Missing,
    Downloading,
    /// Another process or store holds the download lock.
    Busy,
    Corrupt,
    Unsupported,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssetGroup {
    Transcription,
    Speakers,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssetKind {
    Hf,
    Release,
}

/// One file or runtime tree in the shared cache.
///
/// `subdir` and `entry` locate the asset under the cache root. A file asset uses
/// its file name as `entry`. A runtime archive uses the executable path inside
/// the extracted tree, for example `bin/nemo-speech`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelAsset {
    pub id: &'static str,
    pub label: &'static str,
    pub purpose: &'static str,
    pub group: AssetGroup,
    pub kind: AssetKind,
    pub url: &'static str,
    pub size: u64,
    pub sha256: &'static str,
    pub subdir: &'static str,
    pub entry: &'static str,
    pub hf_repo: &'static str,
    pub legacy_dirs: &'static [&'static str],
    pub archive: bool,
    pub version: &'static str,
    pub platforms: &'static [&'static str],
    pub entry_sha256: &'static str,
}

/// Progress of one install, in bytes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AssetProgress {
    pub downloaded: u64,
    pub total: u64,
    pub bytes_per_second: f64,
}

/// Resolved state of one asset. `path` points to the entry on disk.
#[derive(Debug, Clone, PartialEq)]
pub struct AssetState {
    pub asset: ModelAsset,
    pub status: AssetStatus,
    pub installed_bytes: u64,
    pub legacy_bytes: u64,
    pub path: Option<String>,
}

/// Bytes a Remove frees. A hard-linked file frees nothing while its other name exists.
///
/// `legacy_path` is the older copy that Remove can delete too, if there is one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemovalEstimate {
    pub frees: u64,
    pub frees_with_legacy: u64,
    pub legacy_path: Option<String>,
}

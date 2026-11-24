//! The shared cache of downloaded models and runtimes.
//!
//! Same layout and `state.json` as the Python version, so either version reuses
//! what the other downloaded: `assets/`, `downloads/`, `locks/`, `state.json`.
//! The store adopts legacy copies from other tools with a hard link or a copy.
//! It deletes one only when `remove` asks to, and a removed asset stays removed
//! (not adopted again) until `install`.

use std::collections::HashMap;
use std::fs;
use std::io;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use serde_json::{Map, Value, json};
use tracing::{info, warn};

use super::download::{DownloadError, Expected, download_file, extract_archive, sha256_file};
use super::file_lock;
use crate::adapters::model_catalog::{self, RUNTIME_ID};
use crate::adapters::paths::Paths;
use crate::domain::errors::ModelResolutionError;
use crate::domain::model_asset::{
    AssetProgress, AssetState, AssetStatus, ModelAsset, RemovalEstimate,
};

const SCHEMA: &str = "ltn-assets/v1";
const DISK_MARGIN: u64 = 64 * 1024 * 1024;
const WHISPER_CPP_REPO: &str = "ggerganov/whisper.cpp";

/// Model names pywhispercpp downloaded by itself from whisper.cpp's repo.
const PYWHISPERCPP_NAMES: &[&str] = &[
    "base",
    "base-q5_1",
    "base-q8_0",
    "base.en",
    "base.en-q5_1",
    "base.en-q8_0",
    "large-v1",
    "large-v2",
    "large-v2-q5_0",
    "large-v2-q8_0",
    "large-v3",
    "large-v3-q5_0",
    "large-v3-turbo",
    "large-v3-turbo-q5_0",
    "large-v3-turbo-q8_0",
    "medium",
    "medium-q5_0",
    "medium-q8_0",
    "medium.en",
    "medium.en-q5_0",
    "medium.en-q8_0",
    "small",
    "small-q5_1",
    "small-q8_0",
    "small.en",
    "small.en-q5_1",
    "small.en-q8_0",
    "tiny",
    "tiny-q5_1",
    "tiny-q8_0",
    "tiny.en",
    "tiny.en-q5_1",
    "tiny.en-q8_0",
];

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("unknown model asset: '{0}'")]
    UnknownAsset(String),
    #[error(transparent)]
    Download(#[from] DownloadError),
    #[error(transparent)]
    Io(#[from] io::Error),
}

pub type Progress<'a> = &'a mut dyn FnMut(AssetProgress);

/// Where other tools left copies the store can adopt.
#[derive(Debug, Clone)]
pub struct LegacyDirs {
    pub pywhispercpp_models: PathBuf,
    pub nemotron_store: PathBuf,
}

pub struct CachedModelStore {
    root: PathBuf,
    assets: Vec<ModelAsset>,
    platform: String,
    legacy: LegacyDirs,
    /// The `assets` map of `state.json`, as this store last saw it.
    state: Mutex<Map<String, Value>>,
}

impl CachedModelStore {
    pub fn new(paths: &Paths) -> Self {
        let platform = model_catalog::platform_key();
        let legacy = LegacyDirs {
            pywhispercpp_models: paths.pywhispercpp_models.clone(),
            nemotron_store: paths.legacy_nemotron_store(),
        };
        Self::with(
            paths.cache_dir.clone(),
            model_catalog::catalog(&platform),
            &platform,
            legacy,
        )
    }

    pub fn with(
        root: PathBuf,
        assets: Vec<ModelAsset>,
        platform: &str,
        legacy: LegacyDirs,
    ) -> Self {
        let store = Self {
            root,
            assets,
            platform: platform.into(),
            legacy,
            state: Mutex::default(),
        };
        *store.state.lock().expect("state lock") = store.load_state();
        store
    }

    pub fn cache_root(&self) -> &Path {
        &self.root
    }

    /// The catalog, without touching the disk. The UI builds its rows from it first.
    pub fn catalog(&self) -> &[ModelAsset] {
        &self.assets
    }

    pub fn list_assets(&self) -> Result<Vec<AssetState>, StoreError> {
        self.assets
            .iter()
            .map(|asset| self.status(asset.id))
            .collect()
    }

    pub fn status(&self, asset_id: &str) -> Result<AssetState, StoreError> {
        let asset = self.require(asset_id)?;
        let state = |status, installed, legacy, path: Option<&Path>| AssetState {
            asset: asset.clone(),
            status,
            installed_bytes: installed,
            legacy_bytes: legacy,
            path: path.map(|p| p.display().to_string()),
        };
        if !asset.platforms.is_empty() && !asset.platforms.contains(&self.platform.as_str()) {
            return Ok(state(AssetStatus::Unsupported, 0, 0, None));
        }
        let entry = self.entry_path(asset);
        if entry.exists() {
            let size = tree_size(&self.installed_path(asset));
            let status = if self.final_matches(asset, &entry)? {
                AssetStatus::Ready
            } else {
                AssetStatus::Corrupt
            };
            return Ok(state(status, size, 0, Some(&entry)));
        }
        // A download in another process or store shows only as a .part file. Its lock tells.
        let Some(_free) = file_lock::try_exclusive(&self.lock_path(asset.id))? else {
            return Ok(state(AssetStatus::Busy, 0, 0, None));
        };
        if self.is_removed(asset) {
            let legacy = self
                .legacy_entry(asset)
                .map_or(0, |path| legacy_size(asset, &path));
            return Ok(state(AssetStatus::Missing, 0, legacy, None));
        }
        if entry.exists() {
            // The download that held the lock has just finished.
            return Ok(state(
                AssetStatus::Ready,
                tree_size(&self.installed_path(asset)),
                0,
                Some(&entry),
            ));
        }
        let Some(legacy) = self.legacy_entry(asset) else {
            return Ok(state(AssetStatus::Missing, 0, 0, None));
        };
        let legacy_bytes = legacy_size(asset, &legacy);
        match self.legacy_digest(asset, &legacy)? {
            Some(digest) => {
                self.adopt(asset, &legacy, &digest)?;
                Ok(state(
                    AssetStatus::Ready,
                    tree_size(&self.installed_path(asset)),
                    legacy_bytes,
                    Some(&entry),
                ))
            }
            None => Ok(state(AssetStatus::Missing, 0, legacy_bytes, None)),
        }
    }

    /// Make the asset present and return its entry path. A removed asset counts as wanted again.
    pub fn install(
        &self,
        asset_id: &str,
        on_progress: Progress<'_>,
        cancel: Option<&AtomicBool>,
    ) -> Result<PathBuf, StoreError> {
        let asset = self.require(asset_id)?.clone();
        self.clear_removed(&asset);
        let state = self.status(asset_id)?;
        if let (AssetStatus::Ready, Some(path)) = (state.status, &state.path) {
            return Ok(PathBuf::from(path));
        }
        // Only the file lock spans the download, so status() still answers the UI meanwhile.
        let _lock = self.download_lock(
            asset.id,
            &self.part_path(&asset),
            asset.size,
            on_progress,
            cancel,
        )?;
        // Another store or process may have finished it while this one waited.
        let state = self.status(asset_id)?;
        if let (AssetStatus::Ready, Some(path)) = (state.status, &state.path) {
            return Ok(PathBuf::from(path));
        }
        self.install_asset(&asset, on_progress, cancel)?;
        Ok(self.entry_path(&asset))
    }

    /// What `remove` frees, with and without the older copy.
    pub fn removal(&self, asset_id: &str) -> Result<RemovalEstimate, StoreError> {
        let asset = self.require(asset_id)?;
        let target = self.installed_path(asset);
        let frees = freed_bytes(std::slice::from_ref(&target));
        Ok(match self.deletable_legacy(asset) {
            None => RemovalEstimate {
                frees,
                frees_with_legacy: frees,
                legacy_path: None,
            },
            Some(legacy) => RemovalEstimate {
                frees,
                frees_with_legacy: freed_bytes(&[target, legacy.clone()]),
                legacy_path: Some(legacy.display().to_string()),
            },
        })
    }

    /// Delete an installed asset and keep it removed until `install`. Returns the bytes freed.
    pub fn remove(&self, asset_id: &str, delete_legacy: bool) -> Result<u64, StoreError> {
        let asset = self.require(asset_id)?;
        let mut targets = vec![self.installed_path(asset)];
        let legacy = if delete_legacy {
            self.deletable_legacy(asset)
        } else {
            None
        };
        targets.extend(legacy.clone());
        let freed = freed_bytes(&targets);
        for target in &targets {
            delete(target)?;
        }
        if let Some(legacy) = legacy {
            info!(target: "ltn.assets", "deleted legacy copy of {}: {}", asset.id, legacy.display());
        }
        // Without the mark, the next status() would adopt the legacy copy straight back.
        self.update_state(|state| {
            state.insert(asset.id.into(), json!({"removed": true}));
        })?;
        Ok(freed)
    }

    /// A speech model name as a local path: a catalog asset, an alias, an `hf://` URI,
    /// a file path, or a name pywhispercpp used to download by itself.
    pub fn resolve(
        &self,
        name: &str,
        on_progress: Progress<'_>,
        cancel: Option<&AtomicBool>,
    ) -> Result<String, ModelResolutionError> {
        let error = |e: StoreError| {
            ModelResolutionError(format!("Could not get speech model '{name}': {e}"))
        };
        if let Some(asset) = model_catalog::asset_for_model(name, &self.assets) {
            let id = asset.id;
            return self
                .install(id, on_progress, cancel)
                .map(|p| p.display().to_string())
                .map_err(error);
        }
        if let Some((repo, file)) =
            model_catalog::parse_hf_uri(&model_catalog::expand_model_alias(name))
        {
            return self
                .install_custom(&repo, &file, cancel)
                .map(|p| p.display().to_string())
                .map_err(error);
        }
        if name.starts_with("hf://") {
            return Err(ModelResolutionError(format!(
                "Malformed hf:// URI (expected hf://owner/repo/file): {name}"
            )));
        }
        let path = expand_home(name);
        if path.is_absolute() {
            if path.is_file() {
                return Ok(path.display().to_string());
            }
            return Err(ModelResolutionError(format!(
                "Model file not found: {name}"
            )));
        }
        if path.is_file() {
            return Ok(name.to_string());
        }
        if PYWHISPERCPP_NAMES.contains(&name) {
            let file = format!("ggml-{name}.bin");
            let kept = self.legacy.pywhispercpp_models.join(&file);
            if kept.is_file() {
                return Ok(kept.display().to_string());
            }
            return self
                .install_custom(WHISPER_CPP_REPO, &file, cancel)
                .map(|p| p.display().to_string())
                .map_err(error);
        }
        Err(ModelResolutionError(format!(
            "Unknown speech model '{name}'"
        )))
    }

    fn install_asset(
        &self,
        asset: &ModelAsset,
        on_progress: Progress<'_>,
        cancel: Option<&AtomicBool>,
    ) -> Result<(), StoreError> {
        self.check_disk(asset)?;
        let entry = self.entry_path(asset);
        if asset.archive {
            let directory = self.asset_dir(asset);
            if directory.exists() {
                fs::remove_dir_all(&directory)?;
            }
            let archive = self.root.join("downloads").join(archive_name(asset));
            download_file(
                asset.url,
                &archive,
                Expected {
                    size: asset.size,
                    sha256: asset.sha256,
                },
                on_progress,
                cancel,
            )?;
            let entry = extract_archive(
                &archive,
                &directory,
                asset.entry,
                asset.entry_sha256,
                cancel,
            )?;
            let _ = fs::remove_file(&archive);
            let digest = if asset.entry_sha256.is_empty() {
                sha256_file(&entry, None)?
            } else {
                asset.entry_sha256.into()
            };
            self.record_asset(asset, &digest)?;
        } else {
            download_file(
                asset.url,
                &entry,
                Expected {
                    size: asset.size,
                    sha256: asset.sha256,
                },
                on_progress,
                cancel,
            )?;
            let digest = if asset.sha256.is_empty() {
                sha256_file(&entry, None)?
            } else {
                asset.sha256.into()
            };
            self.record_asset(asset, &digest)?;
        }
        info!(target: "ltn.assets", "installed model asset {} at {}", asset.id, entry.display());
        Ok(())
    }

    /// A model outside the catalog, by `hf://` URI. Its size and digest are unknown, so only the download is checked.
    fn install_custom(
        &self,
        repo: &str,
        file: &str,
        cancel: Option<&AtomicBool>,
    ) -> Result<PathBuf, StoreError> {
        let id = format!("custom:{repo}/{file}");
        let url = format!("https://huggingface.co/{repo}/resolve/main/{file}");
        let entry = self
            .root
            .join("assets/whisper/custom")
            .join(repo.replace('/', "__"))
            .join(file);
        let part = PathBuf::from(format!("{}.part", entry.display()));
        let mut no_progress = |_: AssetProgress| {};
        let _lock = self.download_lock(&id, &part, 0, &mut no_progress, cancel)?;
        if !entry.is_file() {
            download_file(&url, &entry, Expected::default(), &mut no_progress, cancel)?;
            self.record(&id, &entry, &url, "", &sha256_file(&entry, None)?)?;
            info!(target: "ltn.assets", "installed model asset {id} at {}", entry.display());
        }
        Ok(entry)
    }

    fn download_lock(
        &self,
        id: &str,
        part: &Path,
        size: u64,
        on_progress: Progress<'_>,
        cancel: Option<&AtomicBool>,
    ) -> Result<file_lock::LockGuard, StoreError> {
        let mut logged = false;
        let mut waiting = || {
            if !logged {
                info!(target: "ltn.assets", "waiting for another download of {id}");
                logged = true;
            }
            // The other download may have just finished and renamed its .part file.
            if let Ok(meta) = fs::metadata(part) {
                on_progress(AssetProgress {
                    downloaded: meta.len(),
                    total: size,
                    bytes_per_second: 0.0,
                });
            }
        };
        let cancelled = || cancel.is_some_and(|flag| flag.load(Ordering::Relaxed));
        file_lock::exclusive(&self.lock_path(id), &cancelled, &mut waiting).map_err(|e| {
            if e.kind() == io::ErrorKind::Interrupted {
                StoreError::Download(DownloadError::Cancelled(e.to_string()))
            } else {
                StoreError::Io(e)
            }
        })
    }

    fn check_disk(&self, asset: &ModelAsset) -> Result<(), StoreError> {
        if asset.size == 0 {
            return Ok(());
        }
        // A new account may have no cache folder yet; measure the nearest one that exists.
        let probe = self
            .root
            .ancestors()
            .find(|p| p.exists())
            .unwrap_or(Path::new("/"));
        let free = free_bytes(probe)?;
        if free < asset.size + DISK_MARGIN {
            return Err(StoreError::Download(DownloadError::Failed(format!(
                "not enough disk space for {}: need {}, have {}",
                asset.label,
                megabytes(asset.size),
                megabytes(free)
            ))));
        }
        Ok(())
    }

    // --- Verification and state ---

    /// True when the entry is the expected file. A matching path, size and mtime in
    /// `state.json` skips the hash, so a 1.6 GB model is hashed once.
    fn final_matches(&self, asset: &ModelAsset, entry: &Path) -> Result<bool, StoreError> {
        let meta = fs::metadata(entry)?;
        let recorded = self
            .state
            .lock()
            .expect("state lock")
            .get(asset.id)
            .cloned()
            .unwrap_or(Value::Null);
        if recorded["path"] == json!(self.relative(entry))
            && recorded["size"] == json!(meta.len())
            && recorded["mtime_ns"] == json!(mtime_ns(&meta))
        {
            return Ok(true);
        }
        let digest = sha256_file(entry, None)?;
        let expected = if asset.archive {
            asset.entry_sha256
        } else {
            asset.sha256
        };
        if !expected.is_empty() && digest != expected {
            return Ok(false);
        }
        self.record_asset(asset, &digest)?;
        Ok(true)
    }

    fn legacy_digest(
        &self,
        asset: &ModelAsset,
        legacy: &Path,
    ) -> Result<Option<String>, StoreError> {
        let digest = sha256_file(legacy, None)?;
        let expected = if asset.archive {
            asset.entry_sha256
        } else {
            asset.sha256
        };
        if !expected.is_empty() && digest != expected {
            warn!(target: "ltn.assets", "ignoring legacy copy with a bad digest: {}", legacy.display());
            return Ok(None);
        }
        Ok(Some(digest))
    }

    fn adopt(&self, asset: &ModelAsset, legacy: &Path, digest: &str) -> Result<(), StoreError> {
        let entry = self.entry_path(asset);
        if asset.archive {
            let tree = self.asset_dir(asset);
            // Adopt runs only when the entry is missing, so a tree here is from an earlier failed attempt.
            if tree.exists() {
                fs::remove_dir_all(&tree)?;
            }
            let source = runtime_tree(asset, legacy).unwrap_or_else(|| legacy.to_path_buf());
            copy_tree(&source, &tree)?;
        } else {
            if let Some(parent) = entry.parent() {
                fs::create_dir_all(parent)?;
            }
            if fs::hard_link(legacy, &entry).is_err() {
                fs::copy(legacy, &entry)?;
            }
        }
        info!(target: "ltn.assets", "adopted legacy asset {} from {}", asset.id, legacy.display());
        self.record_asset(asset, digest)
    }

    fn record_asset(&self, asset: &ModelAsset, digest: &str) -> Result<(), StoreError> {
        self.record(
            asset.id,
            &self.entry_path(asset),
            asset.url,
            asset.version,
            digest,
        )
    }

    /// Record a verified entry, so a later status() can skip the hash.
    fn record(
        &self,
        id: &str,
        entry: &Path,
        source: &str,
        version: &str,
        digest: &str,
    ) -> Result<(), StoreError> {
        let meta = fs::metadata(entry)?;
        let record = json!({
            "path": self.relative(entry),
            "size": meta.len(),
            "mtime_ns": mtime_ns(&meta),
            "sha256": digest,
            "source": source,
            "version": version,
        });
        self.update_state(|state| {
            state.insert(id.into(), record);
        })
    }

    /// Change the state and save it. Another store or process may have saved since
    /// this one loaded, so the file's entries merge under this store's.
    fn update_state(&self, change: impl FnOnce(&mut Map<String, Value>)) -> Result<(), StoreError> {
        let mut state = self.state.lock().expect("state lock");
        let mut merged = self.load_state();
        merged.extend(std::mem::take(&mut *state));
        change(&mut merged);
        *state = merged;
        self.save_state(&state)?;
        Ok(())
    }

    /// The `assets` map of `state.json`, or empty when it is missing or of another schema.
    fn load_state(&self) -> Map<String, Value> {
        let Ok(text) = fs::read_to_string(self.state_path()) else {
            return Map::new();
        };
        let Ok(Value::Object(payload)) = serde_json::from_str::<Value>(&text) else {
            return Map::new();
        };
        if payload.get("schema") != Some(&json!(SCHEMA)) {
            return Map::new();
        }
        match payload.get("assets") {
            Some(Value::Object(assets)) => assets.clone(),
            _ => Map::new(),
        }
    }

    fn save_state(&self, state: &Map<String, Value>) -> io::Result<()> {
        fs::create_dir_all(&self.root)?;
        let payload = json!({
            "schema": SCHEMA,
            "updated": chrono::Utc::now().format("%Y-%m-%dT%H:%M:%S+00:00").to_string(),
            "assets": state,
        });
        // One temporary file per writer, so two processes never replace each other's half-written file.
        let thread =
            format!("{:?}", std::thread::current().id()).replace(|c: char| !c.is_ascii_digit(), "");
        let temporary = self
            .root
            .join(format!("state.json.{}.{thread}.tmp", std::process::id()));
        fs::write(
            &temporary,
            serde_json::to_string_pretty(&payload).map_err(io::Error::other)?,
        )?;
        fs::rename(temporary, self.state_path())
    }

    // Read the file, not the cached map: Settings and the preflight each hold a store,
    // and either may remove or install.
    fn is_removed(&self, asset: &ModelAsset) -> bool {
        self.load_state()
            .get(asset.id)
            .and_then(|record| record.get("removed"))
            == Some(&json!(true))
    }

    fn clear_removed(&self, asset: &ModelAsset) {
        if !self.is_removed(asset) {
            return;
        }
        let result = self.update_state(|state| {
            state.remove(asset.id);
        });
        if let Err(e) = result {
            warn!(target: "ltn.assets", "could not clear the removed mark of {}: {e}", asset.id);
        }
    }

    // --- Legacy copies ---

    fn legacy_entry(&self, asset: &ModelAsset) -> Option<PathBuf> {
        self.legacy_candidates(asset)
            .into_iter()
            .find(|path| path.is_file())
    }

    fn legacy_candidates(&self, asset: &ModelAsset) -> Vec<PathBuf> {
        let mut candidates = Vec::new();
        if !asset.hf_repo.is_empty() {
            let root = &self.legacy.pywhispercpp_models;
            candidates.extend(
                asset
                    .legacy_dirs
                    .iter()
                    .map(|dir| root.join(dir).join(asset.entry)),
            );
            let (owner, repo) = asset.hf_repo.split_once('/').unwrap_or((asset.hf_repo, ""));
            candidates.push(
                root.join("hf")
                    .join(format!("{owner}__{repo}"))
                    .join(asset.entry),
            );
        }
        // The older diarization setup recorded its binary and its model.
        let key = match asset.id {
            RUNTIME_ID => Some("binary"),
            model_catalog::NEMOTRON_ID => Some("model"),
            _ => None,
        };
        if let Some(key) = key
            && let Some(Value::String(path)) = self
                .runtime_record()
                .and_then(|record| record.get(key).cloned())
            && !path.is_empty()
        {
            candidates.push(expand_home(&path));
        }
        candidates
    }

    fn runtime_record(&self) -> Option<Map<String, Value>> {
        let text = fs::read_to_string(self.legacy.nemotron_store.join("runtime.json")).ok()?;
        match serde_json::from_str(&text).ok()? {
            Value::Object(record) => Some(record),
            _ => None,
        }
    }

    /// The legacy file, or the runtime tree that holds it, when its layout is the expected one.
    fn deletable_legacy(&self, asset: &ModelAsset) -> Option<PathBuf> {
        let legacy = self.legacy_entry(asset)?;
        if !asset.archive {
            return Some(legacy);
        }
        let tree = runtime_tree(asset, &legacy)?;
        let expected_name = Path::new(asset.subdir).file_name()?;
        (tree.file_name() == Some(expected_name) && !tree.starts_with(&self.root)).then_some(tree)
    }

    // --- Paths ---

    fn require(&self, asset_id: &str) -> Result<&ModelAsset, StoreError> {
        self.assets
            .iter()
            .find(|asset| asset.id == asset_id)
            .ok_or_else(|| StoreError::UnknownAsset(asset_id.into()))
    }

    fn state_path(&self) -> PathBuf {
        self.root.join("state.json")
    }

    fn asset_dir(&self, asset: &ModelAsset) -> PathBuf {
        self.root.join("assets").join(asset.subdir)
    }

    fn entry_path(&self, asset: &ModelAsset) -> PathBuf {
        self.asset_dir(asset).join(asset.entry)
    }

    /// An archive extracts a whole tree (bin/, lib/, share/); the entry is one file in it.
    fn installed_path(&self, asset: &ModelAsset) -> PathBuf {
        if asset.archive {
            self.asset_dir(asset)
        } else {
            self.entry_path(asset)
        }
    }

    fn lock_path(&self, id: &str) -> PathBuf {
        let name: String = id
            .chars()
            .map(|c| {
                if c.is_alphanumeric() || "-_.".contains(c) {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        self.root.join("locks").join(format!("{name}.lock"))
    }

    fn part_path(&self, asset: &ModelAsset) -> PathBuf {
        if asset.archive {
            return self
                .root
                .join("downloads")
                .join(format!("{}.part", archive_name(asset)));
        }
        PathBuf::from(format!("{}.part", self.entry_path(asset).display()))
    }

    fn relative(&self, path: &Path) -> String {
        path.strip_prefix(&self.root)
            .unwrap_or(path)
            .display()
            .to_string()
    }
}

fn archive_name(asset: &ModelAsset) -> &str {
    asset.url.rsplit('/').next().unwrap_or(asset.url)
}

/// The runtime tree that holds an entry such as `bin/nemo-speech`.
fn runtime_tree(asset: &ModelAsset, entry: &Path) -> Option<PathBuf> {
    let depth = Path::new(asset.entry).components().count();
    entry.ancestors().nth(depth).map(Path::to_path_buf)
}

fn legacy_size(asset: &ModelAsset, legacy: &Path) -> u64 {
    if asset.archive {
        runtime_tree(asset, legacy).map_or(0, |tree| tree_size(&tree))
    } else {
        tree_size(legacy)
    }
}

fn expand_home(path: &str) -> PathBuf {
    match (path.strip_prefix("~/"), dirs::home_dir()) {
        (Some(rest), Some(home)) => home.join(rest),
        _ => PathBuf::from(path),
    }
}

fn mtime_ns(meta: &fs::Metadata) -> i64 {
    meta.mtime() * 1_000_000_000 + meta.mtime_nsec()
}

/// Regular files under `path`, not following links. A link takes no space of its own.
fn files(path: &Path) -> Vec<PathBuf> {
    let Ok(meta) = fs::symlink_metadata(path) else {
        return Vec::new();
    };
    if meta.file_type().is_symlink() {
        return Vec::new();
    }
    if meta.is_file() {
        return vec![path.to_path_buf()];
    }
    let Ok(entries) = fs::read_dir(path) else {
        return Vec::new();
    };
    entries
        .filter_map(Result::ok)
        .flat_map(|entry| files(&entry.path()))
        .collect()
}

fn tree_size(path: &Path) -> u64 {
    files(path)
        .iter()
        .filter_map(|file| fs::metadata(file).ok())
        .map(|meta| meta.len())
        .sum()
}

/// Bytes that deleting all of `paths` frees. A file with another hard link outside them frees nothing.
fn freed_bytes(paths: &[PathBuf]) -> u64 {
    let mut inodes: HashMap<(u64, u64), (u64, u64, u64)> = HashMap::new();
    for file in paths.iter().flat_map(|path| files(path)) {
        let Ok(meta) = fs::metadata(&file) else {
            continue;
        };
        let seen = inodes
            .entry((meta.dev(), meta.ino()))
            .or_insert((meta.len(), meta.nlink(), 0));
        seen.2 += 1;
    }
    inodes
        .values()
        .filter(|(_, links, seen)| seen >= links)
        .map(|(size, _, _)| size)
        .sum()
}

fn delete(path: &Path) -> io::Result<()> {
    match fs::symlink_metadata(path) {
        Ok(meta) if meta.is_dir() => fs::remove_dir_all(path),
        Ok(_) => fs::remove_file(path),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e),
    }
}

/// Copy a tree, keeping symlinks as links: a runtime library keeps its second name as a link.
fn copy_tree(source: &Path, target: &Path) -> io::Result<()> {
    fs::create_dir_all(target)?;
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let destination = target.join(entry.file_name());
        let kind = entry.file_type()?;
        if kind.is_symlink() {
            std::os::unix::fs::symlink(fs::read_link(entry.path())?, &destination)?;
        } else if kind.is_dir() {
            copy_tree(&entry.path(), &destination)?;
        } else {
            fs::copy(entry.path(), &destination)?;
        }
    }
    Ok(())
}

fn free_bytes(path: &Path) -> io::Result<u64> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    let c_path = CString::new(path.as_os_str().as_bytes()).map_err(io::Error::other)?;
    // SAFETY: statvfs fills a zeroed struct for a valid NUL-terminated path.
    let mut stats: libc::statvfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::statvfs(c_path.as_ptr(), &mut stats) } != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(stats.f_bavail as u64 * stats.f_frsize)
}

fn megabytes(size: u64) -> String {
    format!("{:.0} MB", size as f64 / (1024.0 * 1024.0))
}

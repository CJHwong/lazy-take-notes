//! Behavior of the model cache, ported from the Python store tests.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::thread;
use std::time::Duration;

use serde_json::Value;
use sha2::{Digest, Sha256};

use super::download::test_server::serve;
use super::file_lock;
use super::store::{CachedModelStore, LegacyDirs, StoreError};
use crate::domain::model_asset::{AssetGroup, AssetKind, AssetStatus, ModelAsset};

fn leak(text: String) -> &'static str {
    Box::leak(text.into_boxed_str())
}

fn sha(bytes: &[u8]) -> &'static str {
    leak(
        Sha256::digest(bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect(),
    )
}

const BODY: &[u8] = b"model weights for the test";

fn file_asset(url: &str) -> ModelAsset {
    ModelAsset {
        id: "tiny",
        label: "Tiny",
        purpose: "Test",
        group: AssetGroup::Transcription,
        kind: AssetKind::Hf,
        url: leak(url.into()),
        size: BODY.len() as u64,
        sha256: sha(BODY),
        subdir: "whisper/tiny",
        entry: "ggml-tiny.bin",
        hf_repo: "owner/tiny-repo",
        legacy_dirs: &["tiny-legacy"],
        archive: false,
        version: "",
        platforms: &[],
        entry_sha256: "",
    }
}

fn runtime_asset() -> ModelAsset {
    ModelAsset {
        id: "nemo-runtime",
        label: "Runtime",
        purpose: "Test",
        group: AssetGroup::Speakers,
        kind: AssetKind::Release,
        url: "http://127.0.0.1:9/rt-1.tar.gz",
        size: 1,
        sha256: "",
        subdir: "diarization/runtime/rt-1",
        entry: "bin/tool",
        hf_repo: "",
        legacy_dirs: &[],
        archive: true,
        version: "1",
        platforms: &["test-os"],
        entry_sha256: sha(b"#!tool"),
    }
}

struct Fixture {
    dir: tempfile::TempDir,
    assets: Vec<ModelAsset>,
}

impl Fixture {
    fn new(assets: Vec<ModelAsset>) -> Self {
        Self {
            dir: tempfile::tempdir().unwrap(),
            assets,
        }
    }

    fn root(&self) -> PathBuf {
        self.dir.path().join("cache")
    }

    fn legacy(&self) -> LegacyDirs {
        LegacyDirs {
            pywhispercpp_models: self.dir.path().join("pyw"),
            nemotron_store: self.dir.path().join("nemotron"),
        }
    }

    fn store(&self) -> CachedModelStore {
        CachedModelStore::with(self.root(), self.assets.clone(), "test-os", self.legacy())
    }

    fn entry(&self) -> PathBuf {
        self.root().join("assets/whisper/tiny/ggml-tiny.bin")
    }

    fn put_legacy_file(&self, bytes: &[u8]) -> PathBuf {
        let path = self.dir.path().join("pyw/tiny-legacy/ggml-tiny.bin");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, bytes).unwrap();
        path
    }

    fn state(&self) -> Value {
        serde_json::from_str(&fs::read_to_string(self.root().join("state.json")).unwrap()).unwrap()
    }
}

fn no_progress() -> impl FnMut(crate::domain::model_asset::AssetProgress) {
    |_| {}
}

#[test]
fn install_downloads_records_and_reuses() {
    let server = serve(BODY.to_vec());
    let fixture = Fixture::new(vec![file_asset(&server.url)]);
    let store = fixture.store();
    let path = store.install("tiny", &mut no_progress(), None).unwrap();
    assert_eq!(path, fixture.entry());
    let record = &fixture.state()["assets"]["tiny"];
    assert_eq!(fixture.state()["schema"], "ltn-assets/v1");
    assert_eq!(record["path"], "assets/whisper/tiny/ggml-tiny.bin");
    assert_eq!(record["sha256"], sha(BODY));
    assert_eq!(record["size"], BODY.len());
    store.install("tiny", &mut no_progress(), None).unwrap();
    assert_eq!(server.requests.lock().unwrap().len(), 1);
    assert_eq!(store.status("tiny").unwrap().status, AssetStatus::Ready);
}

#[test]
fn a_changed_file_reads_corrupt() {
    let server = serve(BODY.to_vec());
    let fixture = Fixture::new(vec![file_asset(&server.url)]);
    fixture
        .store()
        .install("tiny", &mut no_progress(), None)
        .unwrap();
    fs::write(fixture.entry(), b"tampered weights").unwrap();
    assert_eq!(
        fixture.store().status("tiny").unwrap().status,
        AssetStatus::Corrupt
    );
}

#[test]
fn unsupported_and_unknown_assets() {
    let fixture = Fixture::new(vec![ModelAsset {
        platforms: &["other-os"],
        ..runtime_asset()
    }]);
    assert_eq!(
        fixture.store().status("nemo-runtime").unwrap().status,
        AssetStatus::Unsupported
    );
    assert!(matches!(
        fixture.store().status("nope"),
        Err(StoreError::UnknownAsset(_))
    ));
}

#[test]
fn a_set_cancel_stops_the_download_and_installs_nothing() {
    let server = serve(BODY.to_vec());
    let fixture = Fixture::new(vec![file_asset(&server.url)]);
    let cancel = AtomicBool::new(true);
    assert!(
        fixture
            .store()
            .install("tiny", &mut no_progress(), Some(&cancel))
            .is_err()
    );
    assert!(!fixture.entry().exists());
}

mod resolve {
    use super::*;

    #[test]
    fn by_id_and_by_uri() {
        let server = serve(BODY.to_vec());
        let fixture = Fixture::new(vec![file_asset(&server.url)]);
        let store = fixture.store();
        let by_id = store.resolve("tiny", &mut no_progress(), None).unwrap();
        let by_uri = store
            .resolve(
                "hf://owner/tiny-repo/ggml-tiny.bin",
                &mut no_progress(),
                None,
            )
            .unwrap();
        assert_eq!(by_id, by_uri);
    }

    #[test]
    fn paths_and_bad_names() {
        let fixture = Fixture::new(vec![]);
        let store = fixture.store();
        let file = fixture.dir.path().join("model.bin");
        fs::write(&file, b"x").unwrap();
        let resolve = |name: &str| store.resolve(name, &mut no_progress(), None);
        assert_eq!(
            resolve(file.to_str().unwrap()).unwrap(),
            file.display().to_string()
        );
        assert!(
            resolve("/no/such/model.bin")
                .unwrap_err()
                .0
                .starts_with("Model file not found")
        );
        assert!(
            resolve("hf://only-owner")
                .unwrap_err()
                .0
                .starts_with("Malformed hf:// URI")
        );
        assert_eq!(
            resolve("nonsense-model").unwrap_err().0,
            "Unknown speech model 'nonsense-model'"
        );
    }

    #[test]
    fn a_pywhispercpp_name_uses_its_old_download() {
        let fixture = Fixture::new(vec![]);
        let kept = fixture.dir.path().join("pyw/ggml-base.en.bin");
        fs::create_dir_all(kept.parent().unwrap()).unwrap();
        fs::write(&kept, b"x").unwrap();
        assert_eq!(
            fixture
                .store()
                .resolve("base.en", &mut no_progress(), None)
                .unwrap(),
            kept.display().to_string()
        );
    }
}

mod legacy {
    use super::*;

    #[test]
    fn a_matching_legacy_file_is_adopted_by_hard_link() {
        let fixture = Fixture::new(vec![file_asset("http://127.0.0.1:9/never")]);
        fixture.put_legacy_file(BODY);
        let state = fixture.store().status("tiny").unwrap();
        assert_eq!(
            (state.status, state.legacy_bytes),
            (AssetStatus::Ready, BODY.len() as u64)
        );
        assert_eq!(fs::read(fixture.entry()).unwrap(), BODY);
        assert_eq!(fixture.store().removal("tiny").unwrap().frees, 0);
    }

    #[test]
    fn a_legacy_file_with_a_bad_digest_is_ignored() {
        let fixture = Fixture::new(vec![file_asset("http://127.0.0.1:9/never")]);
        fixture.put_legacy_file(b"something else");
        let state = fixture.store().status("tiny").unwrap();
        assert_eq!(
            (state.status, state.legacy_bytes),
            (AssetStatus::Missing, 14)
        );
    }

    fn put_legacy_runtime(fixture: &Fixture, folder: &str) -> PathBuf {
        let tree = fixture.dir.path().join("old").join(folder);
        fs::create_dir_all(tree.join("bin")).unwrap();
        fs::create_dir_all(tree.join("lib")).unwrap();
        fs::write(tree.join("bin/tool"), b"#!tool").unwrap();
        fs::write(tree.join("lib/a.so"), b"library").unwrap();
        std::os::unix::fs::symlink("a.so", tree.join("lib/b.so")).unwrap();
        fs::create_dir_all(fixture.dir.path().join("nemotron")).unwrap();
        let record = serde_json::json!({"binary": tree.join("bin/tool")});
        fs::write(
            fixture.dir.path().join("nemotron/runtime.json"),
            record.to_string(),
        )
        .unwrap();
        tree
    }

    #[test]
    fn a_legacy_runtime_tree_is_copied_with_its_links_and_can_be_deleted() {
        let fixture = Fixture::new(vec![runtime_asset()]);
        let tree = put_legacy_runtime(&fixture, "rt-1");
        let store = fixture.store();
        assert_eq!(
            store.status("nemo-runtime").unwrap().status,
            AssetStatus::Ready
        );
        let installed = fixture.root().join("assets/diarization/runtime/rt-1");
        assert!(
            fs::symlink_metadata(installed.join("lib/b.so"))
                .unwrap()
                .file_type()
                .is_symlink()
        );
        let estimate = store.removal("nemo-runtime").unwrap();
        assert_eq!(estimate.legacy_path, Some(tree.display().to_string()));
        assert_eq!((estimate.frees, estimate.frees_with_legacy), (13, 26));
        store.remove("nemo-runtime", true).unwrap();
        assert!(!tree.exists() && !installed.exists());
    }

    #[test]
    fn a_legacy_runtime_in_an_unexpected_folder_is_not_offered_for_deletion() {
        let fixture = Fixture::new(vec![runtime_asset()]);
        put_legacy_runtime(&fixture, "renamed");
        assert_eq!(
            fixture.store().removal("nemo-runtime").unwrap().legacy_path,
            None
        );
    }
}

mod remove {
    use super::*;

    #[test]
    fn a_removed_asset_stays_removed_but_reports_its_old_copy() {
        let fixture = Fixture::new(vec![file_asset("http://127.0.0.1:9/never")]);
        let legacy = fixture.put_legacy_file(BODY);
        let store = fixture.store();
        store.status("tiny").unwrap();
        assert_eq!(store.remove("tiny", false).unwrap(), 0);
        let state = fixture.store().status("tiny").unwrap();
        assert_eq!(
            (state.status, state.legacy_bytes),
            (AssetStatus::Missing, BODY.len() as u64)
        );
        assert!(legacy.exists() && !fixture.entry().exists());
        // Install clears the mark and adopts the old copy, with no download.
        assert_eq!(
            store.install("tiny", &mut no_progress(), None).unwrap(),
            fixture.entry()
        );
    }

    #[test]
    fn remove_with_the_old_copy_deletes_both_and_frees_the_bytes() {
        let fixture = Fixture::new(vec![file_asset("http://127.0.0.1:9/never")]);
        let legacy = fixture.put_legacy_file(BODY);
        let store = fixture.store();
        store.status("tiny").unwrap();
        let estimate = store.removal("tiny").unwrap();
        assert_eq!(
            (estimate.frees, estimate.frees_with_legacy),
            (0, BODY.len() as u64)
        );
        assert_eq!(store.remove("tiny", true).unwrap(), BODY.len() as u64);
        assert!(!legacy.exists());
    }

    #[test]
    fn no_old_copy_means_no_offer() {
        let fixture = Fixture::new(vec![file_asset("http://127.0.0.1:9/never")]);
        assert_eq!(fixture.store().removal("tiny").unwrap().legacy_path, None);
    }
}

mod state_file {
    use super::*;

    #[test]
    fn a_corrupt_or_foreign_state_file_is_ignored() {
        let fixture = Fixture::new(vec![file_asset("http://127.0.0.1:9/never")]);
        fs::create_dir_all(fixture.root()).unwrap();
        for text in [
            "{not json",
            "[1, 2]",
            r#"{"schema": "other", "assets": {}}"#,
        ] {
            fs::write(fixture.root().join("state.json"), text).unwrap();
            assert_eq!(
                fixture.store().status("tiny").unwrap().status,
                AssetStatus::Missing
            );
        }
    }

    #[test]
    fn a_second_store_keeps_the_first_store_s_record() {
        let server = serve(BODY.to_vec());
        let other = ModelAsset {
            id: "other",
            subdir: "whisper/other",
            ..file_asset(&server.url)
        };
        let fixture = Fixture::new(vec![file_asset(&server.url), other]);
        let first = fixture.store();
        let second = fixture.store();
        first.install("tiny", &mut no_progress(), None).unwrap();
        second.install("other", &mut no_progress(), None).unwrap();
        let assets = &fixture.state()["assets"];
        assert!(assets.get("tiny").is_some() && assets.get("other").is_some());
    }

    #[test]
    fn a_python_record_skips_the_hash() {
        // A record with matching path, size and mtime means "verified", whoever wrote it.
        let server = serve(BODY.to_vec());
        let fixture = Fixture::new(vec![file_asset(&server.url)]);
        fixture
            .store()
            .install("tiny", &mut no_progress(), None)
            .unwrap();
        let mut state = fixture.state();
        state["assets"]["tiny"]["sha256"] = "recorded-by-python".into();
        fs::write(
            fixture.root().join("state.json"),
            serde_json::to_string_pretty(&state).unwrap(),
        )
        .unwrap();
        fixture.store().status("tiny").unwrap();
        assert_eq!(
            fixture.state()["assets"]["tiny"]["sha256"],
            "recorded-by-python"
        );
    }
}

mod two_downloads {
    use super::*;

    #[test]
    fn a_held_lock_reads_busy_and_a_cancel_stops_the_wait() {
        let fixture = Fixture::new(vec![file_asset("http://127.0.0.1:9/never")]);
        let held = file_lock::try_exclusive(&fixture.root().join("locks/tiny.lock"))
            .unwrap()
            .unwrap();
        assert_eq!(
            fixture.store().status("tiny").unwrap().status,
            AssetStatus::Busy
        );
        // A legacy copy is not adopted while another holder has the lock.
        fixture.put_legacy_file(BODY);
        assert_eq!(
            fixture.store().status("tiny").unwrap().status,
            AssetStatus::Busy
        );
        let cancel = Arc::new(AtomicBool::new(false));
        let flag = cancel.clone();
        let root = fixture.root();
        let assets = fixture.assets.clone();
        let legacy = fixture.legacy();
        let waiter = thread::spawn(move || {
            let store = CachedModelStore::with(root, assets, "test-os", legacy);
            store.install("tiny", &mut |_| {}, Some(&flag))
        });
        thread::sleep(Duration::from_millis(300));
        cancel.store(true, std::sync::atomic::Ordering::Relaxed);
        assert!(waiter.join().unwrap().is_err());
        drop(held);
    }

    #[test]
    fn the_second_install_waits_and_reuses_the_first() {
        let server = serve(BODY.to_vec());
        let fixture = Fixture::new(vec![file_asset(&server.url)]);
        let held = file_lock::try_exclusive(&fixture.root().join("locks/tiny.lock"))
            .unwrap()
            .unwrap();
        let (root, assets, legacy) = (fixture.root(), fixture.assets.clone(), fixture.legacy());
        let waiter = thread::spawn(move || {
            CachedModelStore::with(root, assets, "test-os", legacy).install(
                "tiny",
                &mut |_| {},
                None,
            )
        });
        thread::sleep(Duration::from_millis(300));
        // The "other process" finishes the download, then releases the lock.
        fs::create_dir_all(fixture.entry().parent().unwrap()).unwrap();
        fs::write(fixture.entry(), BODY).unwrap();
        drop(held);
        assert_eq!(waiter.join().unwrap().unwrap(), fixture.entry());
        assert!(server.requests.lock().unwrap().is_empty());
    }
}

#[test]
fn a_missing_cache_parent_still_checks_the_disk() {
    let server = serve(BODY.to_vec());
    let fixture = Fixture::new(vec![file_asset(&server.url)]);
    let root = fixture.dir.path().join("no/such/parent/cache");
    let store = CachedModelStore::with(
        root.clone(),
        fixture.assets.clone(),
        "test-os",
        fixture.legacy(),
    );
    assert!(
        store
            .install("tiny", &mut no_progress(), None)
            .unwrap()
            .starts_with(&root)
    );
    assert!(Path::new(&root).join("state.json").exists());
}

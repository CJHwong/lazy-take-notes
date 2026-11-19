//! Pinned catalog of downloadable models and runtimes.
//!
//! Versions and digests are fixed here. The app never reads a remote catalog: a
//! remote manifest would let a server change what the app executes.

use crate::domain::model_asset::{AssetGroup, AssetKind, ModelAsset};

pub const RUNTIME_ID: &str = "nemo-runtime";
pub const NEMOTRON_ID: &str = "nemotron-3-diarization";
pub const RUNTIME_VERSION: &str = "0.2.0";

const NO_DIRS: &[&str] = &[];

pub const WHISPER_MODELS: [ModelAsset; 3] = [
    ModelAsset {
        id: "whisper-large-v3-turbo-q8_0",
        label: "Whisper large-v3-turbo q8",
        purpose: "Default transcript",
        group: AssetGroup::Transcription,
        kind: AssetKind::Hf,
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q8_0.bin",
        size: 874_188_075,
        sha256: "317eb69c11673c9de1e1f0d459b253999804ec71ac4c23c17ecf5fbe24e259a1",
        subdir: "whisper/whisper-large-v3-turbo-q8_0",
        entry: "ggml-large-v3-turbo-q8_0.bin",
        hf_repo: "ggerganov/whisper.cpp",
        legacy_dirs: &["whisper-cpp"],
        archive: false,
        version: "",
        platforms: NO_DIRS,
        entry_sha256: "",
    },
    ModelAsset {
        id: "breeze25-q8",
        label: "Breeze ASR 25 q8 (zh)",
        purpose: "Chinese transcript",
        group: AssetGroup::Transcription,
        kind: AssetKind::Hf,
        url: "https://huggingface.co/alan314159/Breeze-ASR-25-whispercpp/resolve/main/ggml-model-q8_0.bin",
        size: 1_656_129_708,
        sha256: "669eb226a0e23b42465a6d2f60ce1902fbd534e19faab59511730420eb25e90d",
        subdir: "whisper/breeze25-q8",
        entry: "ggml-model-q8_0.bin",
        hf_repo: "alan314159/Breeze-ASR-25-whispercpp",
        legacy_dirs: &["breeze"],
        archive: false,
        version: "",
        platforms: NO_DIRS,
        entry_sha256: "",
    },
    ModelAsset {
        id: "breeze26-q8",
        label: "Breeze ASR 26 q8 (zh-min-nan)",
        purpose: "Hokkien transcript",
        group: AssetGroup::Transcription,
        kind: AssetKind::Hf,
        url: "https://huggingface.co/phate334/Breeze-ASR-26-GGML/resolve/main/ggml-model-q8_0.bin",
        size: 1_656_129_708,
        sha256: "e878f611bf3ef407cff3c1e2ecf5f60dc92bc6dd09a85b22166731c0d92764a1",
        subdir: "whisper/breeze26-q8",
        entry: "ggml-model-q8_0.bin",
        hf_repo: "phate334/Breeze-ASR-26-GGML",
        legacy_dirs: &["breeze26"],
        archive: false,
        version: "",
        platforms: NO_DIRS,
        entry_sha256: "",
    },
];

pub const NEMOTRON_MODEL: ModelAsset = ModelAsset {
    id: NEMOTRON_ID,
    label: "Nemotron 3 diarization",
    purpose: "Speaker labels",
    group: AssetGroup::Speakers,
    kind: AssetKind::Hf,
    url: "https://huggingface.co/nvidia/Nemotron-3-Diarization/resolve/f667ed73aee57d40cc39428eb768b4fd87a0a29e/Nemotron-3-Diarization.q8_0.gguf",
    size: 107_012_128,
    sha256: "08456d9e22cd9a323c0364d98375f3746d6e68507ebb705cd46438c534c7a3a1",
    subdir: "diarization/models",
    entry: "Nemotron-3-Diarization.q8_0.gguf",
    hf_repo: "nvidia/Nemotron-3-Diarization",
    legacy_dirs: NO_DIRS,
    archive: false,
    version: "",
    platforms: NO_DIRS,
    entry_sha256: "",
};

const fn runtime(
    platform: &'static [&'static str],
    file: &'static str,
    url: &'static str,
    size: u64,
    sha256: &'static str,
    entry_sha256: &'static str,
) -> ModelAsset {
    ModelAsset {
        id: RUNTIME_ID,
        label: "NeMo Speech runtime",
        purpose: "Speaker labels",
        group: AssetGroup::Speakers,
        kind: AssetKind::Release,
        url,
        size,
        sha256,
        subdir: file,
        entry: "bin/nemo-speech",
        hf_repo: "",
        legacy_dirs: NO_DIRS,
        archive: true,
        version: RUNTIME_VERSION,
        platforms: platform,
        entry_sha256,
    }
}

const RUNTIME_ASSETS: [(&str, ModelAsset); 4] = [
    (
        "darwin-arm64",
        runtime(
            &["darwin-arm64"],
            "diarization/runtime/nemo-speech-0.2.0-macos-aarch64-metal",
            "https://github.com/NVIDIA/NeMo-Speech.cpp/releases/download/v0.2.0/nemo-speech-0.2.0-macos-aarch64-metal.tar.gz",
            5_031_510,
            "5cb02ba7c04f0b5585ce5cde9c830be5f0b7dfb4c83083c500109c381b4f2da9",
            "be11c8abb3eb17c891489e7058a61bf764035d471cdb78f6bf413fb181144b73",
        ),
    ),
    (
        "darwin-x86_64",
        runtime(
            &["darwin-x86_64"],
            "diarization/runtime/nemo-speech-0.2.0-macos-x86_64-cpu",
            "https://github.com/NVIDIA/NeMo-Speech.cpp/releases/download/v0.2.0/nemo-speech-0.2.0-macos-x86_64-cpu.tar.gz",
            5_256_846,
            "d52c318d8197511d4f7eee02bcf05511f7e91b300555a130b562e571695a0f39",
            "",
        ),
    ),
    (
        "linux-x86_64",
        runtime(
            &["linux-x86_64"],
            "diarization/runtime/nemo-speech-0.2.0-linux-x86_64-cpu",
            "https://github.com/NVIDIA/NeMo-Speech.cpp/releases/download/v0.2.0/nemo-speech-0.2.0-linux-x86_64-cpu.tar.gz",
            6_683_477,
            "f396057150f1b774935c7414fd32ebc04195a795a34120d78d8c9ecefa1b7507",
            "",
        ),
    ),
    (
        "linux-arm64",
        runtime(
            &["linux-arm64"],
            "diarization/runtime/nemo-speech-0.2.0-linux-aarch64-cpu",
            "https://github.com/NVIDIA/NeMo-Speech.cpp/releases/download/v0.2.0/nemo-speech-0.2.0-linux-aarch64-cpu.tar.gz",
            6_309_769,
            "ac50694d4729f9eae390a9851d38f51269d9af9829b535325539090a3b95c787",
            "",
        ),
    ),
];

const BREEZE25_REPO: &str = "alan314159/Breeze-ASR-25-whispercpp";
const BREEZE25_VARIANTS: [(&str, &str); 8] = [
    ("breeze25", "ggml-model.bin"),
    ("breeze25-q8", "ggml-model-q8_0.bin"),
    ("breeze25-q5", "ggml-model-q5_k.bin"),
    ("breeze25-q4", "ggml-model-q4_k.bin"),
    // Pre-26 aliases, from before breeze26 forced the version suffix.
    ("breeze", "ggml-model.bin"),
    ("breeze-q8", "ggml-model-q8_0.bin"),
    ("breeze-q5", "ggml-model-q5_k.bin"),
    ("breeze-q4", "ggml-model-q4_k.bin"),
];
/// Taiwanese Hokkien (zh-min-nan) fine-tune.
const BREEZE26_REPO: &str = "phate334/Breeze-ASR-26-GGML";
const BREEZE26_VARIANTS: [(&str, &str); 4] = [
    ("breeze26", "ggml-model-q8_0.bin"),
    ("breeze26-q8", "ggml-model-q8_0.bin"),
    ("breeze26-q5", "ggml-model-q5_0.bin"),
    ("breeze26-q4", "ggml-model-q4_0.bin"),
];
const WHISPER_CPP_REPO: &str = "ggerganov/whisper.cpp";
/// Model names whisper.cpp publishes, by short name.
pub const WHISPER_CPP_MODELS: [(&str, &str); 11] = [
    ("large-v3-turbo-q8_0", "ggml-large-v3-turbo-q8_0.bin"),
    ("large-v3-turbo-q5_0", "ggml-large-v3-turbo-q5_0.bin"),
    ("large-v3-turbo", "ggml-large-v3-turbo.bin"),
    ("large-v3-q5_0", "ggml-large-v3-q5_0.bin"),
    ("large-v3", "ggml-large-v3.bin"),
    ("large-v2-q8_0", "ggml-large-v2-q8_0.bin"),
    ("large-v2-q5_0", "ggml-large-v2-q5_0.bin"),
    ("medium-q8_0", "ggml-medium-q8_0.bin"),
    ("medium-q5_0", "ggml-medium-q5_0.bin"),
    ("small-q8_0", "ggml-small-q8_0.bin"),
    ("small-q5_1", "ggml-small-q5_1.bin"),
];

fn lookup(table: &[(&str, &'static str)], name: &str) -> Option<&'static str> {
    table
        .iter()
        .find(|(alias, _)| *alias == name)
        .map(|(_, file)| *file)
}

/// A short alias as its `hf://` URI. Other names pass through.
pub fn expand_model_alias(name: &str) -> String {
    let tables: [(&[(&str, &str)], &str); 3] = [
        (&BREEZE25_VARIANTS, BREEZE25_REPO),
        (&BREEZE26_VARIANTS, BREEZE26_REPO),
        (&WHISPER_CPP_MODELS, WHISPER_CPP_REPO),
    ];
    for (table, repo) in tables {
        if let Some(file) = lookup(table, name) {
            return format!("hf://{repo}/{file}");
        }
    }
    name.to_string()
}

/// The release-asset key for this machine, like `darwin-arm64`.
pub fn platform_key() -> String {
    platform_key_for(std::env::consts::OS, std::env::consts::ARCH)
}

pub fn platform_key_for(os: &str, arch: &str) -> String {
    let arm = matches!(arch, "aarch64" | "arm64");
    match os {
        "macos" | "darwin" => if arm { "darwin-arm64" } else { "darwin-x86_64" }.into(),
        "linux" => if arm { "linux-arm64" } else { "linux-x86_64" }.into(),
        other => format!("{other}-{arch}"),
    }
}

/// The diarization runtime for a platform, or None where there is none.
pub fn runtime_asset(platform: &str) -> Option<ModelAsset> {
    RUNTIME_ASSETS
        .iter()
        .find(|(key, _)| *key == platform)
        .map(|(_, asset)| asset.clone())
}

/// Every asset that applies to a platform, in display order.
pub fn catalog(platform: &str) -> Vec<ModelAsset> {
    let mut assets: Vec<ModelAsset> = WHISPER_MODELS.to_vec();
    assets.push(NEMOTRON_MODEL);
    assets.extend(runtime_asset(platform));
    assets
}

/// `hf://owner/repo/file` as (`owner/repo`, `file`).
pub fn parse_hf_uri(uri: &str) -> Option<(String, String)> {
    let parts: Vec<&str> = uri.strip_prefix("hf://")?.split('/').collect();
    if parts.len() < 3 {
        return None;
    }
    Some((format!("{}/{}", parts[0], parts[1]), parts[2..].join("/")))
}

/// The catalog asset for an asset id, an alias, or an `hf://` URI.
pub fn asset_for_model<'a>(name: &str, assets: &'a [ModelAsset]) -> Option<&'a ModelAsset> {
    if let Some(asset) = assets.iter().find(|asset| asset.id == name) {
        return Some(asset);
    }
    let (repo, file) = parse_hf_uri(&expand_model_alias(name))?;
    assets
        .iter()
        .find(|asset| asset.hf_repo == repo && asset.entry == file)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aliases_expand_and_other_names_pass_through() {
        assert_eq!(
            expand_model_alias("breeze"),
            "hf://alan314159/Breeze-ASR-25-whispercpp/ggml-model.bin"
        );
        assert_eq!(
            expand_model_alias("breeze26-q5"),
            "hf://phate334/Breeze-ASR-26-GGML/ggml-model-q5_0.bin"
        );
        assert_eq!(
            expand_model_alias("large-v3"),
            "hf://ggerganov/whisper.cpp/ggml-large-v3.bin"
        );
        assert_eq!(expand_model_alias("/models/x.bin"), "/models/x.bin");
    }

    #[test]
    fn assets_resolve_by_id_alias_and_uri() {
        let assets = catalog("darwin-arm64");
        assert_eq!(assets.len(), 5);
        assert_eq!(
            asset_for_model("breeze25-q8", &assets).unwrap().id,
            "breeze25-q8"
        );
        assert_eq!(
            asset_for_model("breeze-q8", &assets).unwrap().id,
            "breeze25-q8"
        );
        let uri = "hf://ggerganov/whisper.cpp/ggml-large-v3-turbo-q8_0.bin";
        assert_eq!(
            asset_for_model(uri, &assets).unwrap().id,
            "whisper-large-v3-turbo-q8_0"
        );
        assert!(asset_for_model("large-v3", &assets).is_none());
        assert_eq!(
            asset_for_model(RUNTIME_ID, &assets)
                .unwrap()
                .entry_sha256
                .len(),
            64
        );
    }

    #[test]
    fn platform_keys_and_unsupported_platforms() {
        assert_eq!(platform_key_for("macos", "aarch64"), "darwin-arm64");
        assert_eq!(platform_key_for("linux", "x86_64"), "linux-x86_64");
        assert_eq!(platform_key_for("windows", "x86_64"), "windows-x86_64");
        assert_eq!(catalog("windows-x86_64").len(), 4);
    }

    #[test]
    fn hf_uris_need_owner_repo_and_file() {
        assert_eq!(
            parse_hf_uri("hf://a/b/c/d.bin"),
            Some(("a/b".into(), "c/d.bin".into()))
        );
        assert_eq!(parse_hf_uri("hf://a/b"), None);
        assert_eq!(parse_hf_uri("a/b/c"), None);
    }
}

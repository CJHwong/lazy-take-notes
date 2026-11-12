"""Pinned catalog of downloadable models and runtimes.

Versions and digests are fixed here. The app never reads a remote catalog. A
remote manifest would let a server change what the app executes.
"""

from __future__ import annotations

import platform
import sys

from lazy_take_notes.l1_entities.model_asset import ModelAsset

RUNTIME_ID = 'nemo-runtime'
NEMOTRON_ID = 'nemotron-3-diarization'
NEMOTRON_REVISION = 'f667ed73aee57d40cc39428eb768b4fd87a0a29e'
RUNTIME_VERSION = '0.2.0'
_RUNTIME_RELEASE = f'https://github.com/NVIDIA/NeMo-Speech.cpp/releases/download/v{RUNTIME_VERSION}'

WHISPER_MODELS: tuple[ModelAsset, ...] = (
    ModelAsset(
        id='whisper-large-v3-turbo-q8_0',
        label='Whisper large-v3-turbo q8',
        purpose='Default transcript',
        group='transcription',
        kind='hf',
        url='https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q8_0.bin',
        size=874_188_075,
        sha256='317eb69c11673c9de1e1f0d459b253999804ec71ac4c23c17ecf5fbe24e259a1',
        subdir='whisper/whisper-large-v3-turbo-q8_0',
        entry='ggml-large-v3-turbo-q8_0.bin',
        hf_repo='ggerganov/whisper.cpp',
        legacy_dirs=('whisper-cpp',),
    ),
    ModelAsset(
        id='breeze25-q8',
        label='Breeze ASR 25 q8 (zh)',
        purpose='Chinese transcript',
        group='transcription',
        kind='hf',
        url='https://huggingface.co/alan314159/Breeze-ASR-25-whispercpp/resolve/main/ggml-model-q8_0.bin',
        size=1_656_129_708,
        sha256='669eb226a0e23b42465a6d2f60ce1902fbd534e19faab59511730420eb25e90d',
        subdir='whisper/breeze25-q8',
        entry='ggml-model-q8_0.bin',
        hf_repo='alan314159/Breeze-ASR-25-whispercpp',
        legacy_dirs=('breeze',),
    ),
    ModelAsset(
        id='breeze26-q8',
        label='Breeze ASR 26 q8 (zh-min-nan)',
        purpose='Hokkien transcript',
        group='transcription',
        kind='hf',
        url='https://huggingface.co/phate334/Breeze-ASR-26-GGML/resolve/main/ggml-model-q8_0.bin',
        size=1_656_129_708,
        sha256='e878f611bf3ef407cff3c1e2ecf5f60dc92bc6dd09a85b22166731c0d92764a1',
        subdir='whisper/breeze26-q8',
        entry='ggml-model-q8_0.bin',
        hf_repo='phate334/Breeze-ASR-26-GGML',
        legacy_dirs=('breeze26',),
    ),
)

NEMOTRON_MODEL = ModelAsset(
    id=NEMOTRON_ID,
    label='Nemotron 3 diarization',
    purpose='Speaker labels',
    group='speakers',
    kind='hf',
    url=(
        'https://huggingface.co/nvidia/Nemotron-3-Diarization/resolve/'
        f'{NEMOTRON_REVISION}/Nemotron-3-Diarization.q8_0.gguf'
    ),
    size=107_012_128,
    sha256='08456d9e22cd9a323c0364d98375f3746d6e68507ebb705cd46438c534c7a3a1',
    subdir='diarization/models',
    entry='Nemotron-3-Diarization.q8_0.gguf',
    hf_repo='nvidia/Nemotron-3-Diarization',
)

_RUNTIME_ASSETS: dict[str, ModelAsset] = {
    'darwin-arm64': ModelAsset(
        id=RUNTIME_ID,
        label='NeMo Speech runtime',
        purpose='Speaker labels',
        group='speakers',
        kind='release',
        url=f'{_RUNTIME_RELEASE}/nemo-speech-{RUNTIME_VERSION}-macos-aarch64-metal.tar.gz',
        size=5_031_510,
        sha256='5cb02ba7c04f0b5585ce5cde9c830be5f0b7dfb4c83083c500109c381b4f2da9',
        subdir=f'diarization/runtime/nemo-speech-{RUNTIME_VERSION}-macos-aarch64-metal',
        entry='bin/nemo-speech',
        archive=True,
        version=RUNTIME_VERSION,
        platforms=('darwin-arm64',),
        entry_sha256='be11c8abb3eb17c891489e7058a61bf764035d471cdb78f6bf413fb181144b73',
    ),
    'darwin-x86_64': ModelAsset(
        id=RUNTIME_ID,
        label='NeMo Speech runtime',
        purpose='Speaker labels',
        group='speakers',
        kind='release',
        url=f'{_RUNTIME_RELEASE}/nemo-speech-{RUNTIME_VERSION}-macos-x86_64-cpu.tar.gz',
        size=5_256_846,
        sha256='d52c318d8197511d4f7eee02bcf05511f7e91b300555a130b562e571695a0f39',
        subdir=f'diarization/runtime/nemo-speech-{RUNTIME_VERSION}-macos-x86_64-cpu',
        entry='bin/nemo-speech',
        archive=True,
        version=RUNTIME_VERSION,
        platforms=('darwin-x86_64',),
    ),
    'linux-x86_64': ModelAsset(
        id=RUNTIME_ID,
        label='NeMo Speech runtime',
        purpose='Speaker labels',
        group='speakers',
        kind='release',
        url=f'{_RUNTIME_RELEASE}/nemo-speech-{RUNTIME_VERSION}-linux-x86_64-cpu.tar.gz',
        size=6_683_477,
        sha256='f396057150f1b774935c7414fd32ebc04195a795a34120d78d8c9ecefa1b7507',
        subdir=f'diarization/runtime/nemo-speech-{RUNTIME_VERSION}-linux-x86_64-cpu',
        entry='bin/nemo-speech',
        archive=True,
        version=RUNTIME_VERSION,
        platforms=('linux-x86_64',),
    ),
    'linux-arm64': ModelAsset(
        id=RUNTIME_ID,
        label='NeMo Speech runtime',
        purpose='Speaker labels',
        group='speakers',
        kind='release',
        url=f'{_RUNTIME_RELEASE}/nemo-speech-{RUNTIME_VERSION}-linux-aarch64-cpu.tar.gz',
        size=6_309_769,
        sha256='ac50694d4729f9eae390a9851d38f51269d9af9829b535325539090a3b95c787',
        subdir=f'diarization/runtime/nemo-speech-{RUNTIME_VERSION}-linux-aarch64-cpu',
        entry='bin/nemo-speech',
        archive=True,
        version=RUNTIME_VERSION,
        platforms=('linux-arm64',),
    ),
}

BREEZE25_REPO = 'alan314159/Breeze-ASR-25-whispercpp'
BREEZE25_VARIANTS = {
    'breeze25': 'ggml-model.bin',
    'breeze25-q8': 'ggml-model-q8_0.bin',
    'breeze25-q5': 'ggml-model-q5_k.bin',
    'breeze25-q4': 'ggml-model-q4_k.bin',
    # Back-compat: pre-26 aliases, before breeze26 forced the version suffix.
    'breeze': 'ggml-model.bin',
    'breeze-q8': 'ggml-model-q8_0.bin',
    'breeze-q5': 'ggml-model-q5_k.bin',
    'breeze-q4': 'ggml-model-q4_k.bin',
}

# Breeze-ASR-26: Taiwanese Hokkien (zh-min-nan) fine-tune.
BREEZE26_REPO = 'phate334/Breeze-ASR-26-GGML'
BREEZE26_VARIANTS = {
    'breeze26': 'ggml-model-q8_0.bin',
    'breeze26-q8': 'ggml-model-q8_0.bin',
    'breeze26-q5': 'ggml-model-q5_0.bin',
    'breeze26-q4': 'ggml-model-q4_0.bin',
}

WHISPER_CPP_REPO = 'ggerganov/whisper.cpp'
WHISPER_CPP_MODELS = {
    'large-v3-turbo-q8_0': 'ggml-large-v3-turbo-q8_0.bin',
    'large-v3-turbo-q5_0': 'ggml-large-v3-turbo-q5_0.bin',
    'large-v3-turbo': 'ggml-large-v3-turbo.bin',
    'large-v3-q5_0': 'ggml-large-v3-q5_0.bin',
    'large-v3': 'ggml-large-v3.bin',
    'large-v2-q8_0': 'ggml-large-v2-q8_0.bin',
    'large-v2-q5_0': 'ggml-large-v2-q5_0.bin',
    'medium-q8_0': 'ggml-medium-q8_0.bin',
    'medium-q5_0': 'ggml-medium-q5_0.bin',
    'small-q8_0': 'ggml-small-q8_0.bin',
    'small-q5_1': 'ggml-small-q5_1.bin',
}


def expand_model_alias(name: str) -> str:
    """Convert a short alias to its ``hf://`` URI. Unknown names pass through."""
    if name in BREEZE25_VARIANTS:
        return f'hf://{BREEZE25_REPO}/{BREEZE25_VARIANTS[name]}'
    if name in BREEZE26_VARIANTS:
        return f'hf://{BREEZE26_REPO}/{BREEZE26_VARIANTS[name]}'
    if name in WHISPER_CPP_MODELS:
        return f'hf://{WHISPER_CPP_REPO}/{WHISPER_CPP_MODELS[name]}'
    return name


def platform_key(system: str | None = None, machine: str | None = None) -> str:
    """Return the release-asset key for this machine."""
    system = system or sys.platform
    machine = (machine or platform.machine()).lower()
    if system == 'darwin':
        return 'darwin-arm64' if machine in ('arm64', 'aarch64') else 'darwin-x86_64'
    if system.startswith('linux'):
        return 'linux-arm64' if machine in ('aarch64', 'arm64') else 'linux-x86_64'
    return f'{system}-{machine}'


def runtime_asset(platform_name: str | None = None) -> ModelAsset | None:
    """Return the runtime asset for *platform_name*, or None when unsupported."""
    return _RUNTIME_ASSETS.get(platform_name or platform_key())


def catalog(platform_name: str | None = None) -> dict[str, ModelAsset]:
    """Return every asset that applies to this platform."""
    assets: dict[str, ModelAsset] = {asset.id: asset for asset in WHISPER_MODELS}
    assets[NEMOTRON_MODEL.id] = NEMOTRON_MODEL
    runtime = runtime_asset(platform_name)
    if runtime is not None:
        assets[runtime.id] = runtime
    return assets


def parse_hf_uri(uri: str) -> tuple[str, str] | None:
    """Split ``hf://owner/repo/file`` into ``(owner/repo, file)``."""
    if not uri.startswith('hf://'):
        return None
    parts = uri.removeprefix('hf://').split('/')
    if len(parts) < 3:  # noqa: PLR2004 -- owner, repo, and at least one file segment
        return None
    return f'{parts[0]}/{parts[1]}', '/'.join(parts[2:])


def asset_for_model(name: str, assets: dict[str, ModelAsset] | None = None) -> ModelAsset | None:
    """Find an asset by id, alias, or ``hf://`` URI."""
    available = assets if assets is not None else catalog()
    if name in available:
        return available[name]
    parsed = parse_hf_uri(expand_model_alias(name))
    if parsed is None:
        return None
    repo, filename = parsed
    for asset in available.values():
        if asset.hf_repo == repo and asset.entry == filename:
            return asset
    return None

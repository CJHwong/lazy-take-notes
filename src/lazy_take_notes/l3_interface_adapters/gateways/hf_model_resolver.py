"""Gateway: HuggingFace model resolver — implements ModelResolver port."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from huggingface_hub import hf_hub_download
from pywhispercpp.constants import MODELS_DIR

from lazy_take_notes.l1_entities.errors import ModelResolutionError

BREEZE_REPO = 'alan314159/Breeze-ASR-25-whispercpp'
BREEZE_VARIANTS = {
    'breeze': 'ggml-model.bin',
    'breeze-q8': 'ggml-model-q8_0.bin',
    'breeze-q5': 'ggml-model-q5_k.bin',
    'breeze-q4': 'ggml-model-q4_k.bin',
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
    if name in BREEZE_VARIANTS:
        return f'hf://{BREEZE_REPO}/{BREEZE_VARIANTS[name]}'
    if name in WHISPER_CPP_MODELS:
        return f'hf://{WHISPER_CPP_REPO}/{WHISPER_CPP_MODELS[name]}'
    return name


def _make_progress_class(callback: Callable[[int], None]) -> type:
    """Create a tqdm-compatible class that reports download progress via *callback*."""

    class _ProgressReporter:
        def __init__(self, *args, **kwargs):
            self.total: int = kwargs.get('total', 0) or 0
            self.n: int = 0
            if self.total > 0:
                callback(0)

        def update(self, n: int = 1) -> None:
            self.n += n
            if self.total > 0:
                callback(min(int(self.n / self.total * 100), 100))

        def close(self) -> None:
            pass

        def set_description(self, *a, **kw) -> None:
            pass

        def set_description_str(self, *a, **kw) -> None:
            pass

        def refresh(self) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    return _ProgressReporter


class HfModelResolver:
    """Resolves whisper model names to local file paths, downloading from HF if needed."""

    def __init__(self, on_progress: Callable[[int], None] | None = None) -> None:
        self._on_progress = on_progress

    def resolve(self, model_name: str) -> str:
        if Path(model_name).is_absolute():
            if not Path(model_name).exists():
                raise ModelResolutionError(f'Model file not found: {model_name}')
            return model_name

        tqdm_class = _make_progress_class(self._on_progress) if self._on_progress else None

        if model_name.startswith('hf://'):
            return _download_hf_uri(model_name, tqdm_class=tqdm_class)

        if model_name in BREEZE_VARIANTS:
            return _download_breeze(model_name, tqdm_class=tqdm_class)

        if model_name in WHISPER_CPP_MODELS:
            return _download_whisper_cpp(model_name, tqdm_class=tqdm_class)

        return model_name


def _download_from_hf(
    repo_id: str,
    filename: str,
    cache_dir: Path,
    *,
    tqdm_class: type | None = None,
) -> str:
    """Download a model file from HuggingFace Hub into *cache_dir*."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    kwargs: dict = dict(repo_id=repo_id, filename=filename, local_dir=cache_dir)
    if tqdm_class is not None:
        kwargs['tqdm_class'] = tqdm_class
    return hf_hub_download(**kwargs)


def _download_hf_uri(uri: str, *, tqdm_class: type | None = None) -> str:
    """Download a model specified by ``hf://owner/repo/filename`` URI."""
    path = uri.removeprefix('hf://')
    parts = path.split('/')
    if len(parts) < 3:  # noqa: PLR2004 -- need owner, repo, and at least one filename segment
        raise ModelResolutionError(f'Malformed hf:// URI (expected hf://owner/repo/file): {uri}')
    repo_id = f'{parts[0]}/{parts[1]}'
    filename = '/'.join(parts[2:])
    cache_dir = Path(MODELS_DIR) / 'hf' / f'{parts[0]}__{parts[1]}'
    return _download_from_hf(repo_id, filename, cache_dir, tqdm_class=tqdm_class)


def _download_breeze(name: str, *, tqdm_class: type | None = None) -> str:
    return _download_from_hf(
        BREEZE_REPO,
        BREEZE_VARIANTS[name],
        Path(MODELS_DIR) / 'breeze',
        tqdm_class=tqdm_class,
    )


def _download_whisper_cpp(name: str, *, tqdm_class: type | None = None) -> str:
    return _download_from_hf(
        WHISPER_CPP_REPO,
        WHISPER_CPP_MODELS[name],
        Path(MODELS_DIR) / 'whisper-cpp',
        tqdm_class=tqdm_class,
    )

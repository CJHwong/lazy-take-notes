"""Shared fixtures for gateway tests."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def no_real_legacy_copies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the user's real old caches out: a test that removes with delete_legacy must never reach them."""
    monkeypatch.setattr(
        'lazy_take_notes.l3_interface_adapters.gateways.cached_model_store.LEGACY_NEMOTRON_STORE',
        tmp_path / 'no-legacy-store',
    )
    monkeypatch.setattr('pywhispercpp.constants.MODELS_DIR', str(tmp_path / 'no-pywhispercpp'))

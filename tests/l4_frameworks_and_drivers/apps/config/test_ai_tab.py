"""Tests for the AI tab actions."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from textual.widgets import Button, Static

import lazy_take_notes.l3_interface_adapters.gateways.yaml_config_loader as loader_mod
import lazy_take_notes.l3_interface_adapters.gateways.yaml_config_writer as writer_mod
from lazy_take_notes.l4_frameworks_and_drivers.apps.config import ConfigApp


@pytest.fixture
def app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ConfigApp:
    config_path = tmp_path / 'config.yaml'
    config_path.write_text('{}')
    monkeypatch.setattr(loader_mod, 'DEFAULT_CONFIG_PATHS', [config_path])
    monkeypatch.setattr(writer_mod, 'DEFAULT_CONFIG_PATHS', [config_path])
    monkeypatch.setattr(writer_mod, 'CONFIG_DIR', tmp_path)
    return ConfigApp()


class TestRefreshModels:
    async def test_refresh_lists_models_from_the_client(self, app: ConfigApp, monkeypatch: pytest.MonkeyPatch):
        from lazy_take_notes.l4_frameworks_and_drivers.container import DependencyContainer

        class FakeClient:
            def list_models(self):
                return [('gpt-oss:20b', 1), ('qwen3:8b', 2)]

        monkeypatch.setattr(
            DependencyContainer,
            'resolve_llm_client',
            staticmethod(lambda _infra: FakeClient()),
        )
        async with app.run_test(size=(120, 60)) as pilot:
            await pilot.pause()
            app.query_one('#cfg-ollama-refresh', Button).press()
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()

            text = str(app.query_one('#cfg-ollama-models', Static).render())
            assert 'gpt-oss:20b' in text
            assert 'Cannot list models' not in text


class TestPullModel:
    async def test_pull_calls_the_client_and_reports(self, app: ConfigApp, monkeypatch: pytest.MonkeyPatch):
        from lazy_take_notes.l4_frameworks_and_drivers.container import DependencyContainer

        calls: list[str] = []

        class FakeClient:
            def pull_model(self, model, on_progress=None):
                calls.append(model)
                if on_progress is not None:
                    on_progress(type('Progress', (), {'status': 'success'})())

        monkeypatch.setattr(
            DependencyContainer,
            'resolve_llm_client',
            staticmethod(lambda _infra: FakeClient()),
        )
        async with app.run_test(size=(120, 60)) as pilot:
            await pilot.pause()
            app.query_one('#cfg-ollama-pull-digest', Button).press()
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()

        assert calls == ['gpt-oss:20b']

    @pytest.mark.parametrize(
        ('event', 'text'),
        [
            ({'status': 'pulling ca7a9654b546', 'completed': 46_000_000, 'total': 92_000_000}, '44 MB of 88 MB (50%)'),
            ({'status': 'verifying sha256 digest'}, 'verifying sha256 digest'),
            ({'status': 'pulling 62fbfd9ed093', 'completed': 182, 'total': 182}, 'pulling 62fbfd9ed093'),
        ],
        ids=['layer', 'no-bytes', 'tiny-layer'],
    )
    async def test_pull_shows_how_much_is_done(self, app: ConfigApp, monkeypatch: pytest.MonkeyPatch, event, text):
        from lazy_take_notes.l4_frameworks_and_drivers.container import DependencyContainer

        seen: list[str] = []
        hold = threading.Event()

        class FakeClient:
            def pull_model(self, model, on_progress=None):
                assert on_progress is not None
                on_progress(type('Progress', (), event)())
                hold.wait(5)

        monkeypatch.setattr(DependencyContainer, 'resolve_llm_client', staticmethod(lambda _infra: FakeClient()))
        async with app.run_test(size=(120, 60)) as pilot:
            await pilot.pause()
            app.query_one('#cfg-ollama-pull-digest', Button).press()
            for _ in range(20):
                await pilot.pause(0.05)
                seen.append(str(app.query_one('#cfg-ollama-models', Static).render()))
                if text in seen[-1]:
                    break
            hold.set()
            await app.workers.wait_for_complete()

        assert any(f'Pulling gpt-oss:20b: {text}' == line for line in seen)

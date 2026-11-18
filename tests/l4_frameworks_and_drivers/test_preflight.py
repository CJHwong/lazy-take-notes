"""Preflight: every requirement fails before the meeting, with a fix, never during it."""

from __future__ import annotations

import errno
import threading
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from lazy_take_notes.l4_frameworks_and_drivers import preflight
from lazy_take_notes.l4_frameworks_and_drivers.preflight import FAIL, PASS, SKIP, WARN, PreflightPlan, run_preflight
from tests.conftest import FakeAudioSource, FakeLLMClient, FakeTranscriber

_CONTAINER = 'lazy_take_notes.l4_frameworks_and_drivers.container.DependencyContainer.build_audio_sources'
_GATEWAY = 'lazy_take_notes.l3_interface_adapters.gateways.sounddevice_audio_source'


class LiveSource(FakeAudioSource):
    """Delivers `value` forever, like the Core Audio tap."""

    continuous_capture = True

    def __init__(self, value: float = 0.01) -> None:
        super().__init__()
        self.value = value
        self.device_notice = ''
        self.awaiting_permission = False

    def read(self, timeout=0.1):
        return np.full(320, self.value, dtype=np.float32)


class DeadTap(FakeAudioSource):
    """A continuous source that never delivers."""

    continuous_capture = True


class Resolver:
    def __init__(self, error: Exception | None = None, on_progress=None) -> None:
        self._error = error
        self._on_progress = on_progress

    def resolve(self, name: str) -> str:
        if self._error is not None:
            raise self._error
        if self._on_progress:
            self._on_progress(50)
        return f'/models/{name}.bin'


@pytest.fixture(autouse=True)
def _fast_audio_windows(monkeypatch):
    monkeypatch.setattr(preflight, 'MIC_READ_S', 0.02)
    monkeypatch.setattr(preflight, 'SYSTEM_FIRST_BUFFER_S', 0.05)
    monkeypatch.setattr(preflight, 'SYSTEM_PERMISSION_WAIT_S', 0.05)
    monkeypatch.setattr(preflight, 'SYSTEM_TONE_WAIT_S', 0.02)
    monkeypatch.setattr(f'{_GATEWAY}.refresh_device_list', lambda: None)


@pytest.fixture
def plan(tmp_path, default_config, default_template):
    return PreflightPlan(
        config=default_config,
        template=default_template,
        base_dir=tmp_path / 'sessions',
        llm_client=FakeLLMClient(),
        llm_provider='ollama',
        transcriber=FakeTranscriber(),
        model_resolver_factory=lambda on_progress: Resolver(on_progress=on_progress),
    )


def _run(plan, mic=None, system=None, tone=True, platform='darwin'):
    mic = mic or LiveSource()
    system = system or LiveSource()
    with (
        patch(_CONTAINER, return_value=(mic, system)),
        patch(f'{_GATEWAY}.play_test_tone', return_value=tone),
        patch.object(preflight.sys, 'platform', platform),
    ):
        return run_preflight(plan)


class TestAllGood:
    def test_every_check_passes_and_the_model_is_loaded_for_the_session(self, plan):
        updates = []
        with (
            patch(_CONTAINER, return_value=(LiveSource(), LiveSource())),
            patch(f'{_GATEWAY}.play_test_tone', return_value=True),
            patch.object(preflight.sys, 'platform', 'darwin'),
        ):
            report = run_preflight(plan, on_update=lambda check: updates.append((check.key, check.status)))

        assert report.ok, [preflight.format_report_line(check) for check in report.failures]
        assert report.get('diarization').status == SKIP
        expected = plan.config.transcription.model_for_locale(plan.template.metadata.locale)
        assert plan.transcriber.load_model_calls == [f'/models/{expected}.bin']
        assert plan.report is report
        assert ('whisper', 'running') in updates
        assert not list(plan.base_dir.iterdir())  # the write probe leaves nothing behind

    def test_the_probe_asks_each_distinct_model_once(self, plan):
        _run(plan)
        assert len(plan.llm_client.chat_single_calls) == len({plan.config.digest.model, plan.config.interactive.model})


class TestBlocking:
    def test_an_unwritable_output_folder_blocks(self, plan):
        with patch.object(preflight.Path, 'write_text', side_effect=PermissionError(errno.EACCES, 'Permission denied')):
            report = _run(plan)
        check = report.get('output')
        assert check.status == FAIL
        assert report.blocked
        assert 'output.directory' in check.fix

    def test_too_little_disk_space_blocks(self, plan):
        with patch.object(preflight.shutil, 'disk_usage', return_value=type('U', (), {'free': 10 * 2**20})()):
            report = _run(plan)
        assert report.get('output').status == FAIL
        assert '10 MB' in report.get('output').detail

    def test_a_broken_template_blocks(self, plan):
        plan.template = plan.template.model_copy(update={'digest_user_template': 'Notes {nonexistent_field}'})
        report = _run(plan)
        assert report.get('template').status == FAIL
        assert report.blocked

    def test_a_model_that_cannot_download_blocks(self, plan):
        plan.model_resolver_factory = lambda _progress: Resolver(error=OSError('network is unreachable'))
        report = _run(plan)
        check = report.get('whisper')
        assert check.status == FAIL
        assert 'network is unreachable' in check.detail
        assert 'Settings > Models' in check.fix

    def test_an_unknown_model_name_says_which_setting_to_fix(self, plan):
        from lazy_take_notes.l1_entities.errors import ModelResolutionError

        error = ModelResolutionError("Unknown speech model 'not-a-real-model'")
        plan.model_resolver_factory = lambda _progress: Resolver(error=error)
        check = _run(plan).get('whisper')
        assert check.status == FAIL
        assert check.detail == "Unknown speech model 'not-a-real-model'"
        assert 'transcription.model' in check.fix
        assert 'network' not in check.fix

    def test_a_model_that_cannot_load_blocks(self, plan):
        class BrokenTranscriber(FakeTranscriber):
            def load_model(self, model_path):
                raise RuntimeError('Whisper subprocess failed to init: bad file')

        plan.transcriber = BrokenTranscriber()
        assert _run(plan).get('whisper').status == FAIL

    def test_a_mic_that_sends_nothing_blocks(self, plan):
        report = _run(plan, mic=FakeAudioSource())
        assert report.get('mic').status == FAIL
        assert not report.get('mic').overridable

    def test_a_mic_that_cannot_open_blocks(self, plan):
        class Unplugged(FakeAudioSource):
            def open(self, sample_rate, channels):
                raise RuntimeError('Error opening InputStream: Invalid device')

        assert _run(plan, mic=Unplugged()).get('mic').status == FAIL

    def test_system_audio_that_never_starts_blocks(self, plan):
        report = _run(plan, system=DeadTap())
        assert report.get('system').status == FAIL
        assert 'Screen & System Audio Recording' in report.get('system').fix

    def test_system_audio_that_misses_the_tone_blocks(self, plan):
        report = _run(plan, system=LiveSource(value=0.0))
        assert report.get('system').status == FAIL
        assert 'permission' in report.get('system').detail

    def test_diarization_without_saved_audio_blocks(self, plan):
        plan.config = plan.config.model_copy(
            update={
                'diarization': plan.config.diarization.model_copy(update={'enabled': True}),
                'output': plan.config.output.model_copy(update={'save_audio': False}),
            }
        )
        assert _run(plan).get('diarization').status == FAIL

    def test_diarization_without_assets_blocks(self, plan, tmp_path):
        plan.config = plan.config.model_copy(
            update={
                'diarization': plan.config.diarization.model_copy(
                    update={'enabled': True, 'binary': str(tmp_path / 'missing-binary')}
                ),
            }
        )
        report = _run(plan)
        assert report.get('diarization').status == FAIL
        assert 'missing-binary' in report.get('diarization').detail


class TestLeaving:
    @pytest.mark.parametrize('model', ['large-v3-turbo-q8_0', 'hf://owner/repo/custom.bin'])
    def test_the_download_gets_the_plan_s_cancel_event(self, plan, model):
        from lazy_take_notes.l3_interface_adapters.gateways.download_engine import DownloadCancelledError
        from lazy_take_notes.l3_interface_adapters.gateways.hf_model_resolver import HfModelResolver

        seen: list = []

        class Store:
            def install(self, asset_id, on_progress=None, cancel=None):
                seen.append(cancel)
                raise DownloadCancelledError('download cancelled')

            def resolve(self, name, cancel=None):
                seen.append(cancel)
                raise DownloadCancelledError('download cancelled')

        plan.config = plan.config.model_copy(
            update={'transcription': plan.config.transcription.model_copy(update={'model': model, 'models': {}})}
        )
        plan.model_resolver_factory = lambda on_progress: HfModelResolver(on_progress, store=Store())
        check = _run(plan).get('whisper')

        assert seen == [plan.cancel]
        assert check.status == FAIL
        assert plan.transcriber.load_model_calls == []

    def test_the_slow_checks_never_keep_the_process_alive(self, plan):
        # Python joins non-daemon threads at exit, so a 120 s probe would hold a closed app open.
        seen: list[bool] = []

        class Watching(Resolver):
            def resolve(self, name: str) -> str:
                seen.append(threading.current_thread().daemon)
                return super().resolve(name)

        plan.model_resolver_factory = lambda on_progress: Watching(on_progress=on_progress)
        _run(plan)

        assert seen == [True]


class TestOverridable:
    def test_an_unreachable_llm_can_be_overridden(self, plan):
        plan.llm_client.set_connectivity(False, 'Connection refused')
        report = _run(plan)
        assert report.get('llm').status == FAIL
        assert not report.blocked
        assert not report.llm_ok
        assert 'ollama serve' in report.get('llm').fix

    def test_a_missing_model_names_the_pull_command(self, plan):
        plan.llm_client.set_missing_models([plan.config.digest.model])
        report = _run(plan)
        assert f'ollama pull {plan.config.digest.model}' in report.get('llm').fix

    def test_a_usage_limit_is_caught_by_the_probe(self, plan):
        class LimitedLLM(FakeLLMClient):
            async def chat_single(self, model, prompt):
                raise RuntimeError('429 You have reached your usage limit')

        plan.llm_client = LimitedLLM()
        report = _run(plan)
        assert report.get('llm').status == FAIL
        assert 'usage limit' in report.get('llm').detail

    def test_a_hung_llm_times_out(self, plan, monkeypatch):
        import asyncio

        class HungLLM(FakeLLMClient):
            async def chat_single(self, model, prompt):
                await asyncio.sleep(10)
                return 'late'

        monkeypatch.setattr(preflight, 'LLM_PROBE_TIMEOUT_S', 0.05)
        plan.llm_client = HungLLM()
        report = _run(plan)
        assert 'no reply within' in report.get('llm').detail

    def test_a_mic_sending_exact_zeros_blocks(self, plan):
        report = _run(plan, mic=LiveSource(value=0.0))
        check = report.get('mic')
        assert check.status == FAIL
        assert report.blocked
        assert 'Privacy & Security > Microphone' in check.fix

    def test_a_muted_start_needs_the_mic_to_open_but_not_to_hear(self, plan):
        plan.mic_muted = True
        assert _run(plan, mic=LiveSource(value=0.0)).get('mic').status == PASS
        assert _run(plan, mic=FakeAudioSource()).get('mic').status == PASS

        class Unplugged(FakeAudioSource):
            def open(self, sample_rate, channels):
                raise RuntimeError('Invalid device')

        assert _run(plan, mic=Unplugged()).get('mic').status == FAIL


class TestWarnings:
    def test_a_tone_that_cannot_play_warns_instead_of_failing(self, plan):
        report = _run(plan, system=LiveSource(value=0.0), tone=False)
        assert report.get('system').status == WARN
        assert report.ok

    def test_a_fallback_microphone_warns(self, plan):
        mic = LiveSource()
        mic.device_notice = "Microphone 'USB' is unavailable; using Built-in"
        report = _run(plan, mic=mic)
        assert report.get('mic').status == WARN
        assert 'USB' in report.get('mic').detail

    def test_an_idle_loopback_that_delivers_nothing_still_passes(self, plan):
        class IdleLoopback(FakeAudioSource):
            continuous_capture = False

        report = _run(plan, system=IdleLoopback(), platform='linux')
        assert report.get('system').status == PASS
        assert 'when audio plays' in report.get('system').detail

    def test_linux_loopback_needs_no_tone(self, plan):
        report = _run(plan, system=LiveSource(value=0.0), platform='linux')
        assert report.get('system').status == PASS

    def test_an_injected_audio_source_is_not_checked(self, plan):
        plan.check_audio = False
        report = _run(plan, mic=FakeAudioSource(), system=FakeAudioSource())
        assert report.get('mic').status == SKIP
        assert report.get('system').status == SKIP
        assert report.ok


class TestNeverCrashes:
    def test_an_unexpected_error_in_a_check_becomes_a_failed_check(self, plan):
        with patch.object(preflight, 'validate_template', side_effect=ZeroDivisionError('bug')):
            report = _run(plan)
        assert report.get('template').status == FAIL
        assert 'ZeroDivisionError' in report.get('template').detail

    def test_audio_sources_are_closed_after_the_checks(self, plan):
        mic, system = LiveSource(), LiveSource()
        _run(plan, mic=mic, system=system)
        assert mic.close_calls == 1
        assert system.close_calls == 1


class TestDownloadProgress:
    def test_the_first_download_shows_size_speed_and_time_left(self, plan):
        from lazy_take_notes.l1_entities.model_asset import AssetProgress
        from lazy_take_notes.l3_interface_adapters.gateways.hf_model_resolver import HfModelResolver
        from lazy_take_notes.l4_frameworks_and_drivers import preflight

        megabyte = 1024 * 1024

        class Store:
            def install(self, asset_id, on_progress=None, cancel=None):
                assert on_progress is not None
                on_progress(AssetProgress(143 * megabyte, 834 * megabyte, 2.3 * megabyte))
                return Path('/tmp/model.bin')

            def resolve(self, name, cancel=None):
                return name

        plan.config = plan.config.model_copy(
            update={
                'transcription': plan.config.transcription.model_copy(
                    update={'model': 'large-v3-turbo-q8_0', 'models': {}}
                )
            }
        )
        plan.model_resolver_factory = lambda on_progress: HfModelResolver(on_progress, store=Store())
        updates: list[str] = []
        check = preflight.Check('whisper', 'Speech model')
        preflight._check_whisper(check, plan, updates.append)  # noqa: SLF001 -- the live text is not kept in the report

        assert 'Downloading large-v3-turbo-q8_0: 143 MB of 834 MB (17%), 2.3 MB/s, about 5 min left' in updates

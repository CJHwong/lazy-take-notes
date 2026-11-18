"""Check every requirement of a live recording before the meeting starts.

A problem found here costs a retry. The same problem found mid-meeting costs
the meeting. Each check uses the same objects the session will use: the same
LLM client, the same transcriber (its loaded model carries into the session),
and the same model resolver, so a check that passes cannot fail differently
later for a reason it did not see.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from lazy_take_notes.l1_entities.audio_constants import SAMPLE_RATE
from lazy_take_notes.l1_entities.config import AppConfig
from lazy_take_notes.l1_entities.errors import ModelResolutionError
from lazy_take_notes.l1_entities.template import SessionTemplate
from lazy_take_notes.l2_use_cases.ports.llm_client import LLMClient
from lazy_take_notes.l2_use_cases.ports.transcriber import Transcriber
from lazy_take_notes.l2_use_cases.utils.template_validator import validate_template
from lazy_take_notes.l3_interface_adapters.gateways.hf_model_resolver import HfModelResolver
from lazy_take_notes.l4_frameworks_and_drivers import download_progress

if TYPE_CHECKING:
    from lazy_take_notes.l3_interface_adapters.gateways.cached_model_store import CachedModelStore

log = logging.getLogger('ltn.preflight')

PENDING, RUNNING, PASS, WARN, FAIL, SKIP = 'pending', 'running', 'pass', 'warn', 'fail', 'skip'

# 16 kHz int16 mono WAV is about 115 MB per hour; this covers a 4-hour meeting.
MIN_FREE_BYTES_WITH_AUDIO = 500 * 1024 * 1024
MIN_FREE_BYTES = 50 * 1024 * 1024
# A cold Ollama model load can take a minute.
LLM_PROBE_TIMEOUT_S = 120.0
LLM_PROBE_PROMPT = 'Reply with the single word OK.'
MIC_READ_S = 1.5
SYSTEM_FIRST_BUFFER_S = 5.0
SYSTEM_PERMISSION_WAIT_S = 45.0
SYSTEM_TONE_WAIT_S = 1.5

_MIC_PERMISSION_FIX = (
    'Allow microphone access in System Settings > Privacy & Security > Microphone, '
    'then check the input volume in System Settings > Sound.'
)
_SYSTEM_PERMISSION_FIX = (
    'Allow this app in System Settings > Privacy & Security > Screen & System Audio Recording, then press Retry.'
)


@dataclass
class Check:
    key: str
    title: str
    status: str = PENDING
    detail: str = ''
    fix: str = ''
    # A failed overridable check offers "Start anyway" instead of blocking.
    overridable: bool = False

    def set(self, status: str, detail: str = '', fix: str = '') -> None:
        self.status = status
        self.detail = detail
        self.fix = fix


@dataclass
class PreflightReport:
    checks: list[Check]
    overridden: bool = False

    def get(self, key: str) -> Check:
        return next(check for check in self.checks if check.key == key)

    @property
    def failures(self) -> list[Check]:
        return [check for check in self.checks if check.status == FAIL]

    @property
    def ok(self) -> bool:
        return not self.failures

    @property
    def blocked(self) -> bool:
        return any(not check.overridable for check in self.failures)

    @property
    def llm_ok(self) -> bool:
        return self.get('llm').status != FAIL


@dataclass
class PreflightPlan:
    """Everything the checks need. The session reuses these objects."""

    config: AppConfig
    template: SessionTemplate
    base_dir: Path
    llm_client: LLMClient
    llm_provider: str
    transcriber: Transcriber
    model_resolver_factory: Callable
    asset_store: CachedModelStore | None = None
    check_audio: bool = True
    # --mute-mic: the mic must still open (the mixer opens it), but what it hears does not matter.
    mic_muted: bool = False
    # False when a plugin supplied the transcriber; only an owned one is closed on Back.
    owns_transcriber: bool = True
    report: PreflightReport | None = field(default=None, repr=False)
    # The InfraConfig the plan was built from; Retry rebuilds both from disk.
    infra: object = field(default=None, repr=False)
    # Set when the user leaves the checks; a running download stops and keeps its .part file.
    cancel: threading.Event = field(default_factory=threading.Event, repr=False)


def new_checks() -> list[Check]:
    return [
        Check('output', 'Output folder'),
        Check('template', 'Template'),
        Check('mic', 'Microphone'),
        Check('system', 'System audio'),
        Check('whisper', 'Speech model'),
        Check('diarization', 'Speaker labels'),
        Check('llm', 'AI notes model', overridable=True),
    ]


def run_preflight(plan: PreflightPlan, on_update: Callable[[Check], None] | None = None) -> PreflightReport:
    """Run every check. Slow network checks run beside the audio checks."""
    notify = on_update or (lambda _check: None)
    checks = {check.key: check for check in new_checks()}
    started = time.monotonic()
    # Daemon threads: Python joins any other thread at exit, so a user who leaves
    # the checks would wait for a download or a 120 s model probe.
    slow = [
        threading.Thread(
            target=_run, args=(checks[key], check_fn, plan, notify), daemon=True, name=f'ltn-preflight-{key}'
        )
        for key, check_fn in (('whisper', _check_whisper), ('llm', _check_llm))
    ]
    for thread in slow:
        thread.start()
    for key, check_fn in (
        ('output', _check_output),
        ('template', _check_template),
        ('mic', _check_microphone),
        ('system', _check_system_audio),
        ('diarization', _check_diarization),
    ):
        _run(checks[key], check_fn, plan, notify)
    for thread in slow:
        thread.join()
    report = PreflightReport(list(checks.values()))
    log.info(
        'preflight finished in %.1fs: %s',
        time.monotonic() - started,
        ', '.join(f'{check.key}={check.status}' for check in report.checks),
    )
    plan.report = report
    return report


def _run(check: Check, check_fn: Callable, plan: PreflightPlan, notify: Callable[[Check], None]) -> None:
    check.set(RUNNING)
    notify(check)

    def update(detail: str) -> None:
        check.detail = detail
        notify(check)

    try:
        check_fn(check, plan, update)
    except Exception as error:
        log.warning('preflight check %s failed', check.key, exc_info=True)
        check.set(FAIL, f'{type(error).__name__}: {error}', check.fix)
    if check.status == RUNNING:
        check.set(PASS, check.detail)
    notify(check)


def _check_output(check: Check, plan: PreflightPlan, _update: Callable[[str], None]) -> None:
    base = plan.base_dir
    fix = 'Choose a writable output.directory in Settings > Sessions.'
    try:
        base.mkdir(parents=True, exist_ok=True)
        probe = base / f'.ltn-preflight-{os.getpid()}'
        probe.write_text('ok', encoding='utf-8')
        probe.unlink()
        free = shutil.disk_usage(base).free
    except OSError as error:
        check.set(FAIL, f'Cannot write to {base}: {error.strerror or error}', fix)
        return
    need = MIN_FREE_BYTES_WITH_AUDIO if plan.config.output.save_audio else MIN_FREE_BYTES
    if free < need:
        check.set(
            FAIL,
            f'Only {free // 2**20} MB free in {base}',
            f'Free at least {need // 2**20} MB, or choose another output.directory.',
        )
        return
    check.set(PASS, f'{base} ({free / 2**30:.1f} GB free)')


def _check_template(check: Check, plan: PreflightPlan, _update: Callable[[str], None]) -> None:
    errors = validate_template(plan.template).errors
    if errors:
        check.set(FAIL, '; '.join(errors), 'Fix the template in the template builder, or pick another template.')
        return
    check.set(PASS, plan.template.metadata.name)


def _check_whisper(check: Check, plan: PreflightPlan, update: Callable[[str], None]) -> None:
    locale = plan.template.metadata.locale
    name = plan.config.transcription.model_for_locale(locale)
    check.fix = 'Check the network for the first download, or install the model in Settings > Models.'
    update(f'Finding {name}')
    resolver = plan.model_resolver_factory(lambda percent: update(f'Downloading {name}: {percent}%'))
    if isinstance(resolver, HfModelResolver):
        # A plugin resolver has no cancel; its download ends with the daemon check thread.
        resolver.cancel = plan.cancel
        # The first download can take most of an hour; show how much is left.
        resolver.on_detail = lambda progress: update(f'Downloading {name}: {download_progress.describe(progress)}')
    try:
        path = resolver.resolve(name)
    except ModelResolutionError as error:
        check.set(FAIL, str(error), 'Set transcription.model to a model listed in Settings > Models.')
        return
    update(f'Loading {name}')
    plan.transcriber.load_model(path)
    update(f'Testing {name}')
    plan.transcriber.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), locale.split('-')[0].lower(), None)
    check.set(PASS, name)


def _check_microphone(check: Check, plan: PreflightPlan, update: Callable[[str], None]) -> None:
    if not plan.check_audio:
        check.set(SKIP, 'Supplied by a plugin; not checked')
        return
    from lazy_take_notes.l3_interface_adapters.gateways.sounddevice_audio_source import (  # noqa: PLC0415 -- deferred: audio stack loaded only when needed
        refresh_device_list,
    )
    from lazy_take_notes.l4_frameworks_and_drivers.container import (  # noqa: PLC0415 -- deferred: avoid a module cycle
        DependencyContainer,
    )

    # The kiosk lives across meetings: see a headset plugged in since the last one.
    refresh_device_list()
    mic, _system = DependencyContainer.build_audio_sources(plan.config.transcription.input_device)
    check.fix = _MIC_PERMISSION_FIX
    update('Listening')
    mic.open(SAMPLE_RATE, 1)
    try:
        frames, nonzero = _listen(mic, MIC_READ_S)
    finally:
        mic.close()
    notice = getattr(mic, 'device_notice', '')
    if plan.mic_muted:
        check.set(PASS, 'Opened; muted for this session')
    elif frames == 0:
        check.set(FAIL, 'The microphone opened but sent no audio', _MIC_PERMISSION_FIX)
    elif nonzero == 0:
        check.set(FAIL, 'The microphone sends only digital silence', _MIC_PERMISSION_FIX)
    elif notice:
        check.set(WARN, notice, 'Connect that microphone, or pick another one in Settings > Audio.')
    else:
        check.set(PASS, f'{frames / SAMPLE_RATE:.1f}s of audio received')


def _check_system_audio(check: Check, plan: PreflightPlan, update: Callable[[str], None]) -> None:
    if not plan.check_audio:
        check.set(SKIP, 'Supplied by a plugin; not checked')
        return
    from lazy_take_notes.l4_frameworks_and_drivers.container import (  # noqa: PLC0415 -- deferred: avoid a module cycle
        DependencyContainer,
    )

    _mic, system = DependencyContainer.build_audio_sources(plan.config.transcription.input_device)
    check.fix = _SYSTEM_PERMISSION_FIX if sys.platform == 'darwin' else ''
    update('Starting capture')
    system.open(SAMPLE_RATE, 1)
    try:
        _verify_system_capture(check, system, update)
    finally:
        system.close()


def _verify_system_capture(check: Check, system, update: Callable[[str], None]) -> None:
    if not getattr(system, 'continuous_capture', False):
        # A loopback that delivers only while audio plays cannot be proven idle.
        check.set(PASS, 'Loopback opened; verified when audio plays')
        return
    if not _wait_for_first_buffer(system, update):
        check.set(FAIL, 'System audio capture did not start', check.fix)
        return
    if sys.platform != 'darwin':
        check.set(PASS, 'Loopback capture started')
        return
    from lazy_take_notes.l3_interface_adapters.gateways.sounddevice_audio_source import (  # noqa: PLC0415 -- deferred: audio stack loaded only when needed
        play_test_tone,
    )

    system.drain()
    update('Playing an inaudible test tone')
    if not play_test_tone():
        check.set(WARN, 'No output device to play the test tone; system audio is not verified')
        return
    _frames, nonzero = _listen(system, SYSTEM_TONE_WAIT_S)
    if nonzero == 0:
        check.set(FAIL, 'System audio is silent: the capture permission is missing', _SYSTEM_PERMISSION_FIX)
        return
    check.set(PASS, 'Test tone captured')


def _wait_for_first_buffer(source, update: Callable[[str], None]) -> bool:
    deadline = time.monotonic() + SYSTEM_FIRST_BUFFER_S
    permission_deadline = time.monotonic() + SYSTEM_PERMISSION_WAIT_S
    while time.monotonic() < max(deadline, permission_deadline if getattr(source, 'awaiting_permission', False) else 0):
        chunk = source.read(timeout=0.1)
        if chunk is not None and chunk.size:
            return True
        if getattr(source, 'exhausted', False):
            return False
        if getattr(source, 'awaiting_permission', False):
            update(f'Waiting for permission. {_SYSTEM_PERMISSION_FIX}')
    return False


def _listen(source, seconds: float) -> tuple[int, int]:
    """Return (frames received, non-zero samples) over the window."""
    frames = nonzero = 0
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        chunk = source.read(timeout=0.1)
        if chunk is None or not chunk.size:
            continue
        frames += chunk.size
        nonzero += int(np.count_nonzero(chunk))
    return frames, nonzero


def _check_diarization(check: Check, plan: PreflightPlan, _update: Callable[[str], None]) -> None:
    cfg = plan.config.diarization
    if not cfg.enabled:
        check.set(SKIP, 'Off')
        return
    if not plan.config.output.save_audio:
        check.set(
            FAIL,
            'Speaker labels need the recording saved',
            'Turn on save_audio in Settings > Sessions, or turn off speaker labels in Settings > Audio.',
        )
        return
    from lazy_take_notes.l3_interface_adapters.gateways.nemo_speech_diarizer import (  # noqa: PLC0415 -- deferred: loaded only when enabled
        NemoSpeechDiarizer,
    )

    diarizer = NemoSpeechDiarizer(
        binary=cfg.binary or None, model=cfg.model or None, device=cfg.device, store=plan.asset_store
    )
    try:
        binary, _model = diarizer.resolve_paths()
    except RuntimeError as error:
        check.set(FAIL, str(error), 'Install the speaker assets in Settings > Models.')
        return
    if not os.access(binary, os.X_OK):
        check.set(FAIL, f'{binary} is not executable', 'Reinstall the speaker assets in Settings > Models.')
        return
    check.set(PASS, 'Assets installed')


def _check_llm(check: Check, plan: PreflightPlan, update: Callable[[str], None]) -> None:
    check_ai(
        check, plan.llm_client, plan.llm_provider, [plan.config.digest.model, plan.config.interactive.model], update
    )


def check_ai(
    check: Check,
    client: LLMClient,
    provider: str,
    models: list[str],
    update: Callable[[str], None],
) -> None:
    """Reach the provider, find the models, and get one tiny reply from each. Settings > Test runs it too."""
    reachable, error = client.check_connectivity()
    if not reachable:
        check.set(FAIL, f'Cannot reach the AI provider: {error}', _provider_fix(provider))
        return
    models = list(dict.fromkeys(models))
    missing = client.check_models(models)
    if missing:
        pull = '; '.join(f'ollama pull {model}' for model in missing)
        fix = f'Run: {pull}' if provider == 'ollama' else 'Pick an available model in Settings > Models.'
        check.set(FAIL, f'Model not available: {", ".join(missing)}', fix)
        return
    for model in models:
        update(f'Asking {model} for a one-word reply (up to {LLM_PROBE_TIMEOUT_S:.0f}s)')
        problem = _probe_model(client, model)
        if problem:
            check.set(FAIL, f'{model}: {problem}', 'Check the API key, the usage limit, and the model name.')
            return
    check.set(PASS, ', '.join(models))


def _probe_model(client: LLMClient, model: str) -> str:
    """One tiny request. Catches auth errors and usage limits that a model list cannot."""
    try:
        asyncio.run(asyncio.wait_for(client.chat_single(model, LLM_PROBE_PROMPT), timeout=LLM_PROBE_TIMEOUT_S))
    except TimeoutError:
        return f'no reply within {LLM_PROBE_TIMEOUT_S:.0f}s'
    except Exception as error:
        return f'{type(error).__name__}: {error}'
    return ''


def _provider_fix(provider: str) -> str:
    if provider == 'ollama':
        return 'Start Ollama (ollama serve), or check ollama.host in Settings > Models.'
    return 'Check the provider address and network in Settings > Models.'


def format_report_line(check: Check) -> str:
    """One plain-text line per check, for the CLI and headless output."""
    mark = {PASS: 'ok', WARN: 'warn', FAIL: 'FAIL', SKIP: 'skip'}.get(check.status, check.status)
    line = f'[{mark:>4}] {check.title}: {check.detail}'
    if check.status in (FAIL, WARN) and check.fix:
        line += f'\n       fix: {check.fix}'
    return line

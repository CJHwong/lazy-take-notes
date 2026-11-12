"""Audio page: live input check, microphone choice, and audio processing."""

from __future__ import annotations

import subprocess  # noqa: S404 -- opens the system privacy pane, not a shell command
import sys
import threading

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Button, Rule, Select, Static

from lazy_take_notes.l3_interface_adapters.gateways.sounddevice_audio_source import SounddeviceAudioSource
from lazy_take_notes.l4_frameworks_and_drivers.apps.config.fields import FieldRow, SelectRow, SwitchRow, format_number
from lazy_take_notes.l4_frameworks_and_drivers.apps.config.sections import SettingsSection, section_title
from lazy_take_notes.l4_frameworks_and_drivers.container import DependencyContainer
from lazy_take_notes.l4_frameworks_and_drivers.messages import (
    AudioProbeDone,
    AudioProbeError,
    AudioProbeLevel,
    AudioProbePermission,
)
from lazy_take_notes.l4_frameworks_and_drivers.widgets.status_bar import (
    _rms_to_char,  # noqa: PLC2701 -- share the dB meter mapping
)
from lazy_take_notes.l4_frameworks_and_drivers.workers.audio_probe_worker import run_audio_probe

_DEFAULT_DEVICE = '__default__'
_PERMISSION_TEXT = {
    'allowed': 'Allowed',
    'waiting': 'Waiting for permission',
    'no_signal': 'No signal',
}


class LevelMeter(Static):
    """A rolling block meter, using the status bar's dB scale."""

    def __init__(self, silence_threshold: float, **kwargs) -> None:
        super().__init__('▁' * 12, **kwargs)
        self._silence = silence_threshold
        self._levels = [0.0] * 12

    def push(self, rms: float) -> None:
        self._levels = [*self._levels[1:], rms]
        self.update(''.join(_rms_to_char(value, self._silence) for value in self._levels))


class AudioTab(SettingsSection):
    """Diagnose both audio sources and configure the microphone and processing."""

    DEFAULT_CSS = """
    AudioTab .audio-actions {
        height: auto;
        margin: 0 0 1 0;
    }
    AudioTab .audio-actions Button {
        margin-right: 2;
    }
    AudioTab .audio-permission {
        height: 1;
    }
    AudioTab .audio-permission.allowed {
        color: $success;
    }
    AudioTab .audio-permission.waiting {
        color: $warning;
    }
    AudioTab .audio-permission.no-signal {
        color: $error;
    }
    AudioTab LevelMeter {
        height: 1;
        margin: 0 0 1 0;
        color: $accent;
    }
    """

    def __init__(self, raw: dict, **kwargs) -> None:
        super().__init__(raw, **kwargs)
        self._stop = threading.Event()
        self._probe_running = False

    def compose(self) -> ComposeResult:
        transcription = self._raw.get('transcription', {})
        diarization = self._raw.get('diarization', {})
        silence = float(transcription.get('silence_threshold', 0.01))

        yield section_title('Input check')
        yield Rule(classes='section-rule')
        yield Static('Start a 15 second test and make a sound. Nothing is saved.', classes='section-desc')
        with Horizontal(classes='audio-actions'):
            yield Button('Start test', id='audio-start', variant='primary')
            yield Button('Stop', id='audio-stop')
        yield Static('Microphone', classes='field-label')
        yield LevelMeter(silence, id='audio-meter-mic')
        yield Static('Not tested', id='audio-perm-mic', classes='audio-permission')
        yield Static('System audio', classes='field-label')
        yield LevelMeter(silence, id='audio-meter-system')
        yield Static('Not tested', id='audio-perm-system', classes='audio-permission')
        with Horizontal(classes='audio-actions'):
            yield Button('Grant system audio', id='audio-grant-system')
            yield Button('Open microphone settings', id='audio-grant-mic')

        yield section_title('Microphone')
        yield Rule(classes='section-rule')
        current = transcription.get('input_device') or _DEFAULT_DEVICE
        yield Select(
            [('System default', _DEFAULT_DEVICE), (str(current), str(current))],
            value=str(current),
            id='cfg-audio-device',
            allow_blank=False,
        )
        yield Static('A removed device falls back to the system default with a warning.', classes='section-desc')

        yield section_title('Audio processing')
        yield Rule(classes='section-rule')
        yield FieldRow(
            'Chunk duration',
            'cfg-trans-chunk',
            format_number(transcription.get('chunk_duration', '')),
            placeholder='25.0',
        )
        yield FieldRow(
            'Overlap',
            'cfg-trans-overlap',
            format_number(transcription.get('overlap', '')),
            placeholder='1.0',
        )
        yield FieldRow(
            'Silence threshold',
            'cfg-trans-silence',
            str(transcription.get('silence_threshold', '')),
            placeholder='0.01',
        )
        yield FieldRow(
            'Pause duration',
            'cfg-trans-pause',
            format_number(transcription.get('pause_duration', '')),
            placeholder='1.5',
        )

        yield section_title('Speaker labels')
        yield Rule(classes='section-rule')
        yield SwitchRow(
            'Label speakers',
            'cfg-diarization-enabled',
            bool(diarization.get('enabled', False)),
        )
        yield SelectRow(
            'Device',
            'cfg-diarization-device',
            ['auto', 'cpu', 'metal', 'cuda', 'vulkan'],
            str(diarization.get('device', 'auto')),
        )
        yield FieldRow(
            'Update every',
            'cfg-diarization-interval',
            format_number(diarization.get('interval_s', 60)),
        )

    def on_mount(self) -> None:
        self._refresh_devices(self._raw.get('transcription', {}).get('input_device'))
        self._set_running(False)

    def reload(self, raw: dict) -> None:
        self._raw = raw
        self._refresh_devices(raw.get('transcription', {}).get('input_device'))

    def device_value(self) -> str | None:
        value = str(self.query_one('#cfg-audio-device', Select).value)
        return None if value == _DEFAULT_DEVICE else value

    def _refresh_devices(self, current: str | None) -> None:
        try:
            devices = SounddeviceAudioSource.list_input_devices()
        except Exception:  # noqa: BLE001 -- a missing PortAudio is not fatal to Settings
            devices = []
        choices = [('System default', _DEFAULT_DEVICE), *[(name, name) for name in devices]]
        if current and current not in devices:
            choices.append((current, current))
        select = self.query_one('#cfg-audio-device', Select)
        select.set_options(choices)
        select.value = current or _DEFAULT_DEVICE

    # --- Probe ----------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ''
        if button_id == 'audio-start':
            self._start_probe()
        elif button_id == 'audio-stop':
            self._stop.set()
        elif button_id == 'audio-grant-system':
            self._open_privacy('Privacy_ScreenCapture')
        elif button_id == 'audio-grant-mic':
            self._open_privacy('Privacy_Microphone')

    def _start_probe(self) -> None:
        if self._probe_running:
            return
        self._stop = threading.Event()
        mic, system = DependencyContainer.build_audio_sources(self.device_value(), ensure_permission=True)
        self._probe_running = True
        self._set_running(True)
        self.run_worker(
            lambda: run_audio_probe(mic, system, post_message=self.post_message, stop=self._stop),
            thread=True,
            group='audio-probe',
        )

    @on(AudioProbeLevel)
    def _on_level(self, message: AudioProbeLevel) -> None:
        meter = '#audio-meter-mic' if message.source == 'mic' else '#audio-meter-system'
        self.query_one(meter, LevelMeter).push(message.rms)

    @on(AudioProbePermission)
    def _on_permission(self, message: AudioProbePermission) -> None:
        target = '#audio-perm-mic' if message.source == 'mic' else '#audio-perm-system'
        label = self.query_one(target, Static)
        label.update(_PERMISSION_TEXT.get(message.state, message.state))
        for name in ('allowed', 'waiting', 'no-signal'):
            label.remove_class(name)
        label.add_class(message.state.replace('_', '-'))

    @on(AudioProbeDone)
    def _on_done(self, _message: AudioProbeDone) -> None:
        self._probe_running = False
        self._set_running(False)

    @on(AudioProbeError)
    def _on_error(self, message: AudioProbeError) -> None:
        self._probe_running = False
        self._set_running(False)
        self.notify(f'Audio test failed: {message.error}', severity='error', timeout=10)

    def _set_running(self, running: bool) -> None:
        self.query_one('#audio-start', Button).disabled = running
        self.query_one('#audio-stop', Button).disabled = not running

    @staticmethod
    def _open_privacy(pane: str) -> None:
        if sys.platform != 'darwin':
            return
        subprocess.Popen(  # noqa: S603 -- fixed arg list, not shell=True
            ['/usr/bin/open', f'x-apple.systempreferences:com.apple.preference.security?{pane}']
        )

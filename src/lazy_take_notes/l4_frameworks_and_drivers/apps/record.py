"""RecordApp — live audio recording TUI."""

from __future__ import annotations

import logging
import threading

from textual.binding import Binding

from lazy_take_notes.l2_use_cases.ports.audio_source import AudioSource
from lazy_take_notes.l2_use_cases.ports.transcriber import Transcriber
from lazy_take_notes.l4_frameworks_and_drivers.apps.base import BaseApp
from lazy_take_notes.l4_frameworks_and_drivers.messages import (
    AudioLevel,
    AudioWorkerStatus,
    ModelDownloadProgress,
)
from lazy_take_notes.l4_frameworks_and_drivers.widgets.status_bar import StatusBar

log = logging.getLogger('ltn.app')


class RecordApp(BaseApp):
    """Live audio recording TUI — microphone capture, transcription, and digest."""

    BINDINGS = [
        Binding('space', 'toggle_pause', 'Pause/Resume', priority=True),
        Binding('s', 'stop_recording', 'Stop', priority=True),
        Binding('d', 'force_digest', 'Digest now', show=False),
    ]

    def __init__(
        self,
        *args,
        audio_source: AudioSource | None = None,
        transcriber: Transcriber | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._audio_source = audio_source
        self._transcriber = transcriber

        # Audio control state
        self._audio_paused = threading.Event()
        self._audio_shutdown = threading.Event()
        self._audio_stopped = False
        self._pending_quit = False

    def _hints_for_state(self, state: str) -> str:
        if state == 'recording':
            return r'\[Space] pause  \[s] stop  \[d] digest  \[c] copy  \[l] label  \[h] help'
        if state == 'paused':
            return r'\[Space] resume  \[s] stop  \[c] copy  \[l] label  \[h] help'
        return r'\[c] copy  \[l] label  \[h] help  \[q] quit'

    def _help_keybindings(self) -> list[str]:
        return [
            '| Key | Action |',
            '|-----|--------|',
            '| `Space` | Pause / Resume |',
            '| `s` | Stop recording |',
            '| `d` | Force digest now |',
            '| `c` | Copy focused panel |',
            '| `Tab` | Switch panel focus |',
            '| `l` | Rename session |',
            '| `h` | Toggle this help |',
            '| `q` | Quit |',
        ]

    def on_mount(self) -> None:
        super().on_mount()
        self.query_one('#status-bar', StatusBar).mode_label = 'Record'
        self._start_audio_worker()

    def _start_audio_worker(self) -> None:  # pragma: no cover -- thin thread launcher; patched out in tests
        tc = self._config.transcription
        locale = self._template.metadata.locale
        self._audio_model_name = tc.model_for_locale(locale)
        self._audio_language = locale.split('-')[0].lower()
        self.run_worker(
            self._audio_worker_thread,
            thread=True,
            group='audio',
        )

    def _report_download_progress(self, percent: int) -> None:
        self.post_message(ModelDownloadProgress(percent=percent, model_name=self._audio_model_name))

    def _audio_worker_thread(
        self,
    ):  # pragma: no cover -- thread body; tested independently
        from lazy_take_notes.l3_interface_adapters.gateways.hf_model_resolver import (  # noqa: PLC0415 -- deferred: runs in worker thread, loaded only when audio starts
            HfModelResolver,
        )
        from lazy_take_notes.l4_frameworks_and_drivers.workers.audio_worker import (  # noqa: PLC0415 -- deferred: audio module loaded only when session starts
            run_audio_worker,
        )

        # Resolve model in the worker thread so downloads don't block the TUI.
        try:
            resolver = HfModelResolver(on_progress=self._report_download_progress)
            model_path = resolver.resolve(self._audio_model_name)
        except Exception as e:
            self.post_message(AudioWorkerStatus(status='error', error=str(e)))
            return []

        tc = self._config.transcription
        return run_audio_worker(
            post_message=self.post_message,
            is_cancelled=lambda: self._audio_shutdown.is_set(),
            model_path=model_path,
            language=self._audio_language,
            chunk_duration=tc.chunk_duration,
            overlap=tc.overlap,
            silence_threshold=tc.silence_threshold,
            pause_duration=tc.pause_duration,
            recognition_hints=self._template.recognition_hints,
            pause_event=self._audio_paused,
            output_dir=self._output_dir,
            save_audio=self._config.output.save_audio,
            transcriber=self._transcriber,
            audio_source=self._audio_source,
        )

    def _cancel_audio_workers(self) -> None:
        self._audio_shutdown.set()

    # --- Message Handlers ---

    def on_audio_worker_status(self, message: AudioWorkerStatus) -> None:
        bar = self.query_one('#status-bar', StatusBar)
        bar.audio_status = message.status
        bar.download_percent = -1

        if message.status == 'loading_model' and self._download_modal is not None:
            self._download_modal.switch_to_loading()
        elif message.status in ('model_ready', 'recording'):
            self._dismiss_download_modal()

        if message.status == 'recording':
            bar.recording = True
            bar.paused = False
            bar.stopped = False
            self.screen.refresh(layout=True)
            self._update_hints('recording')
        elif message.status == 'stopped':
            bar.recording = False
            self._update_hints('stopped')
            if self._pending_quit:
                # Quit was triggered before recording started; flush is now complete.
                has_content = self._controller.digest_state.buffer or self._controller.digest_state.digest_count > 0
                if has_content and not self._final_digest_done:
                    self._run_final_digest()
                else:
                    self.exit()
            elif self._audio_stopped:
                has_content = self._controller.digest_state.buffer or self._controller.digest_state.digest_count > 0
                if has_content and not self._final_digest_done:
                    self._run_digest_worker(is_final=True)
        elif message.status == 'error':
            self._dismiss_download_modal()
            self._update_hints('error')
            if message.error:
                self.notify(
                    f'Audio error: {message.error}\n(see ltn_debug.log)',
                    severity='error',
                    timeout=12,
                )
        elif message.status == 'loading_model':
            self._update_hints('idle')
        elif message.status == 'model_ready':
            self._update_hints('idle')

    def on_audio_level(self, message: AudioLevel) -> None:
        try:
            bar = self.query_one('#status-bar', StatusBar)
            bar.audio_level = message.rms
        except Exception:  # noqa: S110 -- TUI race guard; widget may not exist during startup  # pragma: no cover
            pass

    # --- Actions ---

    def action_toggle_pause(self) -> None:
        if self._audio_stopped:
            self.notify('Recording already stopped', severity='warning', timeout=3)
            return

        bar = self.query_one('#status-bar', StatusBar)

        if self._audio_paused.is_set():
            self._audio_paused.clear()
            bar.paused = False
            bar.recording = True
            self.notify('Recording resumed', timeout=2)
            self._update_hints('recording')
        else:
            self._audio_paused.set()
            bar.paused = True
            bar.recording = False
            self.notify('Recording paused', timeout=2)
            self._update_hints('paused')

    def action_stop_recording(self) -> None:
        if self._audio_stopped:
            return

        self._audio_stopped = True
        self._cancel_audio_workers()

        bar = self.query_one('#status-bar', StatusBar)
        bar.recording = False
        bar.paused = False
        bar.stopped = True
        self._update_hints('stopped')

        self.notify('Recording stopped. You can still browse and run quick actions.', timeout=5)

    def action_force_digest(self) -> None:
        if self._pending_quit:
            return
        super().action_force_digest()

    def action_quit_app(self) -> None:
        bar = self.query_one('#status-bar', StatusBar)
        if bar.recording or bar.paused:
            return  # q is intentionally disabled while recording/paused; press s first

        if self._pending_quit:
            if self._digest_running:
                self.notify('Final digest in progress \u2014 please wait', severity='warning', timeout=3)
                return
            self.exit()
            return

        if self._digest_running:
            self.notify('Digest in progress \u2014 please wait', severity='warning', timeout=3)
            return

        self._cancel_audio_workers()

        was_already_stopped = self._audio_stopped
        if not self._audio_stopped:
            self._audio_stopped = True
            bar = self.query_one('#status-bar', StatusBar)
            bar.recording = False
            bar.paused = False
            bar.stopped = True
            self._update_hints('stopped')

        if not was_already_stopped:
            # Audio worker is still flushing — defer final digest/exit until
            # AudioWorkerStatus(stopped) confirms the flush is complete.
            self._pending_quit = True
            return

        # Audio was already stopped — flush already happened, buffer is up to date.
        has_content = self._controller.digest_state.buffer or self._controller.digest_state.digest_count > 0
        if has_content and not self._final_digest_done:
            self._pending_quit = True
            self._run_final_digest()
        else:
            self.exit()

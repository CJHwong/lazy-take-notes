"""TranscribeApp — file transcription TUI with streaming output."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from textual.binding import Binding

from lazy_take_notes.l2_use_cases.ports.transcriber import Transcriber
from lazy_take_notes.l4_frameworks_and_drivers.apps.base import BaseApp
from lazy_take_notes.l4_frameworks_and_drivers.messages import (
    AudioWorkerStatus,
)
from lazy_take_notes.l4_frameworks_and_drivers.widgets.status_bar import StatusBar

log = logging.getLogger('ltn.app')


class TranscribeApp(BaseApp):
    """File transcription TUI — streams transcription from an audio file, then digests."""

    BINDINGS = [
        Binding('s', 'stop_transcription', 'Stop', priority=True),
        Binding('d', 'force_digest', 'Digest now', show=False),
    ]

    def __init__(
        self,
        *args,
        audio_path: Path,
        transcriber: Transcriber | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._audio_path = audio_path
        self._transcriber = transcriber
        self._file_shutdown = threading.Event()
        self._worker_done = False
        self._pending_quit = False

    def _hints_for_state(self, state: str) -> str:
        if state == 'recording':
            return r'\[s] stop  \[d] digest  \[c] copy  \[l] label  \[h] help'
        return r'\[c] copy  \[l] label  \[h] help  \[q] quit'

    def _help_keybindings(self) -> list[str]:
        return [
            '| Key | Action |',
            '|-----|--------|',
            '| `s` | Stop transcription |',
            '| `d` | Force digest now |',
            '| `c` | Copy focused panel |',
            '| `Tab` | Switch panel focus |',
            '| `l` | Rename session |',
            '| `h` | Toggle this help |',
            '| `q` | Quit |',
        ]

    def on_mount(self) -> None:
        super().on_mount()
        self.query_one('#status-bar', StatusBar).mode_label = 'Transcribe'
        self._start_file_worker()

    def _start_file_worker(self) -> None:  # pragma: no cover -- thin thread launcher; patched out in tests
        tc = self._config.transcription
        locale = self._template.metadata.locale
        self._model_name = tc.model_for_locale(locale)
        self._language = locale.split('-')[0].lower()
        self.run_worker(
            self._file_worker_thread,
            thread=True,
            group='file-transcription',
        )

    def _file_worker_thread(self):  # pragma: no cover -- thread body; tested independently
        from lazy_take_notes.l4_frameworks_and_drivers.workers.file_transcription_worker import (  # noqa: PLC0415 -- deferred: loaded only when session starts
            run_file_transcription,
        )

        tc = self._config.transcription
        return run_file_transcription(
            post_message=self.post_message,
            is_cancelled=lambda: self._file_shutdown.is_set(),
            audio_path=self._audio_path,
            model_name=self._model_name,
            language=self._language,
            chunk_duration=tc.chunk_duration,
            overlap=tc.overlap,
            silence_threshold=tc.silence_threshold,
            pause_duration=tc.pause_duration,
            recognition_hints=self._template.recognition_hints,
            transcriber=self._transcriber,
        )

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
            bar.stopped = False
            self._update_hints('recording')
        elif message.status == 'stopped':
            bar.recording = False
            bar.stopped = True
            self._worker_done = True
            self._update_hints('stopped')
            # Auto-trigger final digest when transcription completes
            has_content = self._controller.digest_state.buffer or self._controller.digest_state.digest_count > 0
            if has_content and not self._final_digest_done:
                self._run_final_digest()
            elif self._pending_quit:
                self.exit()
        elif message.status == 'error':
            self._dismiss_download_modal()
            self._worker_done = True
            self._update_hints('error')
            if message.error:
                self.notify(
                    f'Transcription error: {message.error}\n(see ltn_debug.log)',
                    severity='error',
                    timeout=12,
                )
        elif message.status in ('loading_model', 'model_ready'):
            self._update_hints('idle')

    # --- Actions ---

    def action_stop_transcription(self) -> None:
        if self._worker_done:
            return
        self._file_shutdown.set()
        self.notify('Stopping transcription...', timeout=3)

    def action_force_digest(self) -> None:
        if self._pending_quit:
            return
        super().action_force_digest()

    def action_quit_app(self) -> None:
        if self._digest_running:
            self.notify('Digest in progress \u2014 please wait', severity='warning', timeout=3)
            return

        if not self._worker_done:
            self._file_shutdown.set()
            self._pending_quit = True
            return

        has_content = self._controller.digest_state.buffer or self._controller.digest_state.digest_count > 0
        if has_content and not self._final_digest_done:
            self._pending_quit = True
            self._run_final_digest()
        else:
            self.exit()

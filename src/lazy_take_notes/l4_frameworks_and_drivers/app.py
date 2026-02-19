"""Textual App — thin TUI shell: compose + message routing only."""

from __future__ import annotations

import threading
from pathlib import Path

from textual.app import App as TextualApp
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import Static

from lazy_take_notes.l1_entities.config import AppConfig
from lazy_take_notes.l1_entities.template import SessionTemplate
from lazy_take_notes.l2_use_cases.ports.audio_source import AudioSource
from lazy_take_notes.l3_interface_adapters.controllers.session_controller import SessionController
from lazy_take_notes.l4_frameworks_and_drivers.logging_setup import setup_file_logging
from lazy_take_notes.l4_frameworks_and_drivers.messages import (
    AudioLevel,
    AudioWorkerStatus,
    DigestError,
    DigestReady,
    ModelDownloadProgress,
    QueryResult,
    TranscriptChunk,
)
from lazy_take_notes.l4_frameworks_and_drivers.widgets.digest_panel import DigestPanel
from lazy_take_notes.l4_frameworks_and_drivers.widgets.download_modal import DownloadModal
from lazy_take_notes.l4_frameworks_and_drivers.widgets.help_modal import HelpModal
from lazy_take_notes.l4_frameworks_and_drivers.widgets.query_modal import QueryModal
from lazy_take_notes.l4_frameworks_and_drivers.widgets.status_bar import StatusBar
from lazy_take_notes.l4_frameworks_and_drivers.widgets.transcript_panel import TranscriptPanel


class App(TextualApp):
    """Main TUI application for transcription and digest."""

    CSS_PATH = 'app.tcss'

    BINDINGS = [
        Binding('q', 'quit_app', 'Quit', priority=True),
        Binding('space', 'toggle_pause', 'Pause/Resume', priority=True),
        Binding('s', 'stop_recording', 'Stop', priority=True),
        Binding('h', 'show_help', 'Help', priority=True),
        Binding('tab', 'focus_next', 'Switch Panel', show=False),
    ]

    def __init__(
        self,
        config: AppConfig,
        template: SessionTemplate,
        output_dir: Path,
        controller: SessionController | None = None,
        audio_source: AudioSource | None = None,
        missing_digest_models: list[str] | None = None,
        missing_interactive_models: list[str] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._config = config
        self._template = template
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

        setup_file_logging(self._output_dir)

        # Controller (injected or created with default wiring)
        if controller is not None:
            self._controller = controller
        else:
            from lazy_take_notes.l4_frameworks_and_drivers.container import (  # noqa: PLC0415 -- deferred: only wired when no controller injected (non-test path)
                DependencyContainer,
            )

            container = DependencyContainer(config, template, output_dir)
            self._controller = container.controller

        self._audio_source = audio_source

        self._missing_digest_models: list[str] = missing_digest_models or []
        self._missing_interactive_models: list[str] = missing_interactive_models or []

        # Audio control state
        self._audio_paused = threading.Event()
        self._audio_shutdown = threading.Event()
        self._audio_stopped = False
        self._pending_quit = False
        self._download_modal: DownloadModal | None = None

        # Register dynamic quick action bindings
        for qa in template.quick_actions:
            self._bindings.bind(
                qa.key,
                f"quick_action('{qa.key}')",
                description=qa.label,
                show=False,
            )

    def compose(self) -> ComposeResult:
        meta = self._template.metadata
        header_text = f'  lazy-take-notes | {meta.name}'
        if meta.description:
            header_text += f' - {meta.description}'
        if meta.locale:
            header_text += f' [{meta.locale}]'
        yield Static(header_text, id='header')
        with Horizontal(id='main-panels'):
            yield TranscriptPanel(id='transcript-panel')
            yield DigestPanel(id='digest-panel')
        yield StatusBar(id='status-bar')

    def _hints_for_state(self, state: str) -> str:
        qa_hints = '  '.join(rf'\[{qa.key}] {qa.label}' for qa in self._template.quick_actions)
        if state == 'recording':
            parts = [r'\[Space] pause', r'\[s] stop']
            if qa_hints:
                parts.append(qa_hints)
            parts.append(r'\[h] help')
            return '  '.join(parts)
        if state == 'paused':
            return r'\[Space] resume  \[s] stop  \[h] help'
        if state == 'stopped':
            parts = []
            if qa_hints:
                parts.append(qa_hints)
            parts += [r'\[h] help', r'\[q] quit']
            return '  '.join(parts)
        return r'\[h] help  \[q] quit'  # idle / loading / downloading / error

    def _update_hints(self, state: str) -> None:
        try:
            bar = self.query_one('#status-bar', StatusBar)
            bar.keybinding_hints = self._hints_for_state(state)
        except Exception:  # noqa: S110 — widget may not exist during startup
            pass

    def on_mount(self) -> None:
        self._update_hints('idle')
        bar = self.query_one('#status-bar', StatusBar)
        bar.buf_max = self._config.digest.min_lines
        if self._missing_digest_models:
            panel = self.query_one('#digest-panel', DigestPanel)
            pull_cmds = '\n\n'.join(f'`ollama pull {m}`' for m in self._missing_digest_models)
            panel.update_digest(
                f'**LLM model unavailable**\n\nDigests are disabled. To enable:\n\n{pull_cmds}\n\nThen restart.'
            )
        if self._missing_interactive_models:
            models_str = ', '.join(self._missing_interactive_models)
            self.notify(
                f'Quick actions disabled: model {models_str} not found. Run: ollama pull {models_str}',
                severity='warning',
                timeout=10,
            )
        self._start_audio_worker()
        self.set_interval(1.0, self._refresh_status_bar)

    def _start_audio_worker(self) -> None:
        tc = self._config.transcription
        locale = self._template.metadata.locale
        self._audio_model_name = tc.model_for_locale(locale)
        self._audio_language = locale.split('-')[0].lower()
        self.run_worker(
            self._audio_worker_thread,
            thread=True,
            group='audio',
        )

    def _on_model_download_progress(self, percent: int) -> None:
        self.post_message(ModelDownloadProgress(percent=percent, model_name=self._audio_model_name))

    def _audio_worker_thread(self):
        from lazy_take_notes.l3_interface_adapters.gateways.hf_model_resolver import (  # noqa: PLC0415 -- deferred: runs in worker thread, loaded only when audio starts
            HfModelResolver,
        )
        from lazy_take_notes.l4_frameworks_and_drivers.audio_worker import (  # noqa: PLC0415 -- deferred: audio module loaded only when session starts
            run_audio_worker,
        )

        # Resolve model in the worker thread so downloads don't block the TUI.
        try:
            resolver = HfModelResolver(on_progress=self._on_model_download_progress)
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
            whisper_prompt=self._template.whisper_prompt,
            pause_event=self._audio_paused,
            output_dir=self._output_dir,
            save_audio=self._config.output.save_audio,
            audio_source=self._audio_source,
        )

    def _cancel_audio_workers(self) -> None:
        self._audio_shutdown.set()

    def _refresh_status_bar(self) -> None:
        try:
            bar = self.query_one('#status-bar', StatusBar)
            bar.refresh()
        except Exception:  # noqa: S110 — widget may not exist during startup
            pass

    # --- Message Handlers ---

    def on_transcript_chunk(self, message: TranscriptChunk) -> None:
        panel = self.query_one('#transcript-panel', TranscriptPanel)
        panel.append_segments(message.segments)

        should_digest = self._controller.on_transcript_segments(message.segments)

        bar = self.query_one('#status-bar', StatusBar)
        bar.buf_count = len(self._controller.digest_state.buffer)

        if should_digest:
            self._run_digest_worker()

    def _dismiss_download_modal(self) -> None:
        if self._download_modal is not None:
            self._download_modal.dismiss()
            self._download_modal = None

    def on_model_download_progress(self, message: ModelDownloadProgress) -> None:
        bar = self.query_one('#status-bar', StatusBar)
        bar.download_percent = message.percent
        bar.download_model = message.model_name

        if self._download_modal is None:
            self._download_modal = DownloadModal(model_name=message.model_name)
            self.push_screen(self._download_modal)
        else:
            self._download_modal.update_progress(message.percent)

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
                # Quit was triggered while audio was running; flush is now complete.
                if self._controller.digest_state.buffer or self._controller.digest_state.digest_count > 0:
                    self._run_final_digest()
                else:
                    self.exit()
            elif self._audio_stopped and self._controller.digest_state.buffer:
                self._run_digest_worker(is_final=False)
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

    def on_digest_ready(self, message: DigestReady) -> None:
        panel = self.query_one('#digest-panel', DigestPanel)
        panel.update_digest(message.markdown)

        bar = self.query_one('#status-bar', StatusBar)
        bar.buf_count = len(self._controller.digest_state.buffer)
        bar.activity = ''

    def on_digest_error(self, message: DigestError) -> None:
        bar = self.query_one('#status-bar', StatusBar)
        bar.activity = ''
        self.notify(
            f'Digest failed: {message.error} (see ltn_debug.log)',
            severity='error',
            timeout=8,
        )

    def on_query_result(self, message: QueryResult) -> None:
        bar = self.query_one('#status-bar', StatusBar)
        bar.activity = ''
        self.push_screen(QueryModal(title=message.action_label, body=message.result))

    def on_audio_level(self, message: AudioLevel) -> None:
        try:
            bar = self.query_one('#status-bar', StatusBar)
            bar.audio_level = message.rms
        except Exception:  # noqa: S110 — widget may not exist during startup
            pass

    # --- Workers ---

    def _run_digest_worker(self, is_final: bool = False) -> None:
        bar = self.query_one('#status-bar', StatusBar)
        bar.activity = 'Final digest...' if is_final else 'Digesting...'

        async def _digest_task() -> None:
            result = await self._controller.run_digest(is_final=is_final)
            if result.data is not None:
                self.post_message(
                    DigestReady(
                        markdown=result.data,
                        digest_number=self._controller.digest_state.digest_count,
                        is_final=is_final,
                    )
                )
            else:
                self.post_message(
                    DigestError(
                        error=result.error,
                        consecutive_failures=self._controller.digest_state.consecutive_failures,
                    )
                )

        self.run_worker(_digest_task, exclusive=True, group='digest')

    def _run_query_worker(self, key: str) -> None:
        bar = self.query_one('#status-bar', StatusBar)
        # Find label for the status bar
        label = key
        for qa in self._template.quick_actions:
            if qa.key == key:
                label = qa.label
                break
        bar.activity = f'{label}...'

        async def _query_task() -> None:
            result = await self._controller.run_quick_action(key)
            if result is not None:
                text, action_label = result
                self.post_message(QueryResult(result=text, action_label=action_label))

        self.run_worker(_query_task, exclusive=True, group='query')

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

    def action_quick_action(self, key: str) -> None:
        self._run_query_worker(key)

    def action_show_help(self) -> None:
        if isinstance(self.screen, HelpModal):
            self.screen.dismiss()
            return

        meta = self._template.metadata
        lines: list[str] = []

        # Template info
        if meta.name:
            lines.append(f'**Template:** {meta.name}\n')
        if meta.description:
            lines.append(f'**Description:** {meta.description}\n')
        if meta.locale:
            lines.append(f'**Locale:** {meta.locale}\n')
        if lines:
            lines.append('')

        # Quick actions
        if self._template.quick_actions:
            lines.append('### Quick Actions')
            for qa in self._template.quick_actions:
                desc = f' - {qa.description}' if qa.description else ''
                lines.append(f'- `{qa.key}` **{qa.label}**{desc}')
            lines.append('')

        # Status bar
        min_lines = self._config.digest.min_lines
        lines.extend(
            [
                '### Status Bar',
                '| Indicator | Meaning |',
                '|-----------|---------|',
                '| `● Rec` `❚❚ Paused` `■ Stopped` `○ Idle` | Recording state |',
                f'| `buf N/{min_lines}` | Lines buffered toward next digest (fires at {min_lines}) |',
                '| `00:00:00` | Recording time, pauses excluded |',
                '| `▁▂▄█▄▂` | Mic input level — flat means silence detected |',
                '| `⟳ Digesting…` | LLM digest cycle in progress |',
                '',
            ]
        )

        # Keybindings
        lines.append('### Keybindings')
        lines.extend(
            [
                '| Key | Action |',
                '|-----|--------|',
                '| `Space` | Pause / Resume |',
                '| `s` | Stop recording |',
                '| `h` | Toggle this help |',
            ]
        )
        for qa in self._template.quick_actions:
            lines.append(f'| `{qa.key}` | {qa.label} |')
        lines.extend(
            [
                '| `c` | Copy focused panel |',
                '| `Tab` | Switch panel focus |',
                '| `q` | Quit |',
            ]
        )

        self.push_screen(HelpModal(body_md='\n'.join(lines)))

    def action_quit_app(self) -> None:
        if self._pending_quit:
            self.exit()
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
        if self._controller.digest_state.buffer or self._controller.digest_state.digest_count > 0:
            self._pending_quit = True
            self._run_final_digest()
        else:
            self.exit()

    def _run_final_digest(self) -> None:
        bar = self.query_one('#status-bar', StatusBar)
        bar.activity = 'Final digest...'

        async def _final_task() -> None:
            result = await self._controller.run_digest(is_final=True)
            if result.data is not None:
                self.post_message(
                    DigestReady(
                        markdown=result.data,
                        digest_number=self._controller.digest_state.digest_count,
                        is_final=True,
                    )
                )
                self.notify('Final digest ready. Press q to quit.', timeout=10)
            else:
                self.post_message(
                    DigestError(
                        error=result.error,
                        consecutive_failures=self._controller.digest_state.consecutive_failures,
                    )
                )
                self.notify(
                    'Final digest failed. Press q to quit.',
                    severity='error',
                    timeout=10,
                )

        self.run_worker(_final_task, exclusive=True, group='final')

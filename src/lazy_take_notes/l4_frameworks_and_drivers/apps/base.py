"""BaseApp — shared TUI shell: compose, digest/query routing, session management."""

from __future__ import annotations

import logging
import re
import subprocess  # noqa: S404 -- used for fire-and-forget OS file manager launch
import sys
import time
from pathlib import Path

from textual.app import App as TextualApp
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Static, TextArea

from lazy_take_notes.l1_entities.config import AppConfig
from lazy_take_notes.l1_entities.template import SessionTemplate
from lazy_take_notes.l3_interface_adapters.controllers.session_controller import SessionController
from lazy_take_notes.l4_frameworks_and_drivers.logging_setup import setup_file_logging
from lazy_take_notes.l4_frameworks_and_drivers.messages import (
    DigestError,
    DigestReady,
    ModelDownloadProgress,
    QueryResult,
    TranscriptChunk,
    TranscriptionStatus,
)
from lazy_take_notes.l4_frameworks_and_drivers.widgets.digest_panel import DigestPanel
from lazy_take_notes.l4_frameworks_and_drivers.widgets.download_modal import DownloadModal
from lazy_take_notes.l4_frameworks_and_drivers.widgets.help_modal import HelpModal
from lazy_take_notes.l4_frameworks_and_drivers.widgets.label_modal import LabelModal
from lazy_take_notes.l4_frameworks_and_drivers.widgets.query_modal import QueryModal
from lazy_take_notes.l4_frameworks_and_drivers.widgets.status_bar import StatusBar
from lazy_take_notes.l4_frameworks_and_drivers.widgets.transcript_panel import TranscriptPanel

log = logging.getLogger('ltn.app')


class BaseApp(TextualApp):
    """Shared TUI shell — compose, digest/query routing, session management."""

    CSS_PATH = 'app.tcss'

    BINDINGS = [
        Binding('q', 'quit_app', 'Quit', priority=True),
        Binding('l', 'rename_session', 'Label', show=False),
        Binding('o', 'open_session_dir', 'Open', show=False),
        Binding('h', 'show_help', 'Help', priority=True),
        Binding('tab', 'focus_next', 'Switch Panel', show=False),
    ]

    def __init__(
        self,
        config: AppConfig,
        template: SessionTemplate,
        output_dir: Path,
        controller: SessionController | None = None,
        missing_digest_models: list[str] | None = None,
        missing_interactive_models: list[str] | None = None,
        label: str = '',
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._config = config
        self._template = template
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._session_label = label

        setup_file_logging(self._output_dir)

        # Controller (injected or created with default wiring)
        if controller is not None:
            self._controller = controller
        else:  # pragma: no cover -- composition-root wiring; controller always injected in tests
            from lazy_take_notes.l4_frameworks_and_drivers.container import (  # noqa: PLC0415 -- deferred: only wired when no controller injected (non-test path)
                DependencyContainer,
            )

            container = DependencyContainer(config, template, output_dir)
            self._controller = container.controller

        self._missing_digest_models: list[str] = missing_digest_models or []
        self._missing_interactive_models: list[str] = missing_interactive_models or []

        self._download_modal: DownloadModal | None = None
        self._digest_running = False
        self._query_running = False
        self._final_digest_done = False

        # Register dynamic quick action bindings (positional: 1-5)
        for i, qa in enumerate(template.quick_actions):
            num = str(i + 1)
            self._bindings.bind(
                num,
                f"quick_action('{num}')",
                description=qa.label,
                show=False,
            )

    def _build_header_text(self) -> str:
        meta = self._template.metadata
        header_text = f'  lazy-take-notes | {meta.name}'
        if meta.description:
            header_text += f' - {meta.description}'
        if meta.locale:
            header_text += f' [{meta.locale}]'
        if self._session_label:
            header_text += f' \u2014 {self._session_label}'
        return header_text

    def compose(self) -> ComposeResult:
        yield Static(self._build_header_text(), id='header')
        with Horizontal(id='main-panels'):
            yield TranscriptPanel(id='transcript-panel')
            with Vertical(id='digest-col'):
                yield DigestPanel(id='digest-panel')
                yield TextArea(id='context-input')
        yield StatusBar(id='status-bar')

    def _hints_for_state(self, state: str) -> str:
        return r'\[c] copy  \[l] label  \[o] open  \[h] help  \[q] quit'

    def _update_hints(self, state: str) -> None:
        try:
            bar = self.query_one('#status-bar', StatusBar)
            bar.keybinding_hints = self._hints_for_state(state)
            bar.quick_action_hints = '  '.join(
                rf'\[{i + 1}] {qa.label}' for i, qa in enumerate(self._template.quick_actions)
            )
        except Exception:  # noqa: S110 -- TUI race guard; widget may not exist during startup  # pragma: no cover
            pass

    def on_mount(self) -> None:
        self._update_hints('idle')
        bar = self.query_one('#status-bar', StatusBar)
        bar.buf_max = self._config.digest.min_lines
        self.query_one('#context-input', TextArea).border_title = 'Session Context'
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
        self.set_interval(0.2, self._refresh_status_bar)

    def _refresh_status_bar(self) -> None:
        try:
            bar = self.query_one('#status-bar', StatusBar)
            bar.refresh()
        except Exception:  # noqa: S110 -- TUI race guard; widget may not exist during startup  # pragma: no cover
            pass

    # --- Message Handlers ---

    def on_transcription_status(self, message: TranscriptionStatus) -> None:
        bar = self.query_one('#status-bar', StatusBar)
        bar.transcribing = message.active

    def on_transcript_chunk(self, message: TranscriptChunk) -> None:
        panel = self.query_one('#transcript-panel', TranscriptPanel)
        panel.append_segments(message.segments)

        should_digest = self._controller.on_transcript_segments(message.segments)

        bar = self.query_one('#status-bar', StatusBar)
        bar.buf_count = len(self._controller.digest_state.buffer)

        if should_digest and not self._digest_running:
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

    def on_digest_ready(self, message: DigestReady) -> None:
        panel = self.query_one('#digest-panel', DigestPanel)
        panel.update_digest(message.markdown)

        bar = self.query_one('#status-bar', StatusBar)
        bar.buf_count = len(self._controller.digest_state.buffer)
        bar.last_digest_time = time.monotonic()
        bar.activity = ''

        if message.is_final:
            self._final_digest_done = True

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
        self.push_screen(QueryModal(title=message.action_label, body=message.result, is_error=message.is_error))

    # --- Workers ---

    def _run_digest_worker(self, is_final: bool = False) -> None:
        bar = self.query_one('#status-bar', StatusBar)
        bar.activity = 'Final digest...' if is_final else 'Digesting...'
        self._digest_running = True

        async def _digest_task() -> None:
            try:
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
            finally:
                self._digest_running = False

        self.run_worker(_digest_task, exclusive=True, group='digest')

    def _run_query_worker(self, key: str) -> None:
        bar = self.query_one('#status-bar', StatusBar)
        # Find label for the status bar
        label = key
        for i, qa in enumerate(self._template.quick_actions):
            if str(i + 1) == key:
                label = qa.label
                break
        bar.activity = f'{label}...'
        self._query_running = True

        async def _query_task() -> None:
            try:
                result = await self._controller.run_quick_action(key)
            except Exception as e:
                log.error('Quick action failed: %s', e, exc_info=True)
                self.post_message(QueryResult(result=f'Error: {e}', action_label=label, is_error=True))
                return
            finally:
                self._query_running = False
            if result is not None:
                text, action_label = result
                self.post_message(QueryResult(result=text, action_label=action_label))

        self.run_worker(_query_task, exclusive=True, group='query')

    def _run_final_digest(self) -> None:
        bar = self.query_one('#status-bar', StatusBar)
        bar.activity = 'Final digest...'
        self._digest_running = True

        async def _final_task() -> None:
            try:
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
            finally:
                self._digest_running = False

        self.run_worker(_final_task, exclusive=True, group='final')

    # --- Actions ---

    def action_force_digest(self) -> None:
        if self._digest_running:
            self.notify('Digest already in progress', severity='warning', timeout=3)
            return
        if not self._controller.digest_state.buffer:
            self.notify('Nothing in buffer to digest yet', timeout=3)
            return
        self._run_digest_worker(is_final=False)

    def action_rename_session(self) -> None:
        self.push_screen(
            LabelModal(current_label=self._session_label),
            callback=self._on_label_result,
        )

    def _on_label_result(self, label: str | None) -> None:
        if not label:
            return
        safe_label = re.sub(r'[^\w\-]', '_', label.strip())
        if not safe_label:
            return

        # Preserve the YYYY-MM-DD_HHMMSS timestamp prefix from the current dir name.
        current_name = self._output_dir.name
        # Timestamp is always the first 17 chars: "2026-02-21_143052"
        timestamp_prefix = current_name[:17]
        new_name = f'{timestamp_prefix}_{safe_label}'
        new_path = self._output_dir.parent / new_name

        if new_path != self._output_dir:
            self._output_dir.rename(new_path)
            self._output_dir = new_path

            from lazy_take_notes.l3_interface_adapters.gateways.file_persistence import (  # noqa: PLC0415 -- L4 reaches into concrete adapter for relocate
                FilePersistenceGateway,
            )

            persistence = self._controller._persistence  # noqa: SLF001 -- L4 composition root reaches through
            if isinstance(persistence, FilePersistenceGateway):
                persistence.relocate(new_path)

        self._session_label = safe_label
        self.query_one('#header', Static).update(self._build_header_text())
        self.notify(f'Session: {safe_label}', timeout=3)

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if event.text_area.id == 'context-input':
            self._controller.user_context = event.text_area.text

    def action_quick_action(self, key: str) -> None:
        if self._digest_running:
            self.notify('Digest in progress \u2014 please wait', severity='warning', timeout=3)
            return
        if self._query_running:
            self.notify('Quick action already running \u2014 please wait', severity='warning', timeout=3)
            return
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

        # Status bar
        min_lines = self._config.digest.min_lines
        lines.extend(
            [
                '### Status Bar',
                '| Indicator | Meaning |',
                '|-----------|---------|',
                '| `\u25cf Rec` `\u275a\u275a Paused` `\u25a0 Stopped` `\u25cb Idle` | Recording state |',
                f'| `buf N/{min_lines}` | Lines buffered toward next digest (fires at {min_lines}) |',
                '| `00:00:00` | Recording time, pauses excluded |',
                '| `last Xs ago` | Time elapsed since the last digest |',
                '| `\u2581\u2582\u2584\u2588\u2584\u2582` | Mic input level \u2014 flat means silence detected |',
                '| `\u27f3 Transcribing\u2026` | Speech-to-text in progress |',
                '| `\u27f3 Digesting\u2026` | LLM digest cycle in progress |',
                '',
            ]
        )

        # Keybindings
        lines.append('### Keybindings')
        lines.extend(self._help_keybindings())

        self.push_screen(HelpModal(body_md='\n'.join(lines)))

    def _help_keybindings(self) -> list[str]:
        """Return keybinding help table rows. Subclasses override for mode-specific bindings."""
        return [
            '| Key | Action |',
            '|-----|--------|',
            '| `d` | Force digest now |',
            '| `c` | Copy focused panel |',
            '| `Tab` | Switch panel focus |',
            '| `l` | Rename session |',
            '| `o` | Open session directory |',
            '| `h` | Toggle this help |',
            '| `q` | Quit |',
        ]

    def action_open_session_dir(self) -> None:
        bar = self.query_one('#status-bar', StatusBar)
        if not bar.stopped:
            return
        if sys.platform == 'darwin':
            opener = 'open'
        elif sys.platform == 'win32':
            opener = 'explorer'
        else:
            opener = 'xdg-open'
        subprocess.Popen([opener, str(self._output_dir)])  # noqa: S603 -- fixed arg list, not shell=True

    def action_quit_app(self) -> None:
        self.exit()

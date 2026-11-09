"""ShellApp — one Textual App for the kiosk loop.

The standalone commands still run a single Screen on a minimal host. The
kiosk runs this shell instead, so moving between the mode picker and a mode
does not tear down and rebuild the Textual app.
"""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING

from textual.app import App

from lazy_take_notes.l1_entities.session_files import NOTES, SessionArtifacts

if TYPE_CHECKING:
    from lazy_take_notes.l1_entities.config import AppConfig
    from lazy_take_notes.l3_interface_adapters.gateways.yaml_template_loader import YamlTemplateLoader
    from lazy_take_notes.l4_frameworks_and_drivers.config import InfraConfig

    RecordContext = tuple[AppConfig, InfraConfig, YamlTemplateLoader]
    TranscribeContext = tuple[AppConfig, InfraConfig, YamlTemplateLoader]


class ShellApp(App):
    """Kiosk shell: Welcome, pickers, and modes are screens on one App."""

    def __init__(
        self,
        *,
        config_path=None,
        output_dir: str | None = None,
        initial_mode: str | None = None,
        record_options: dict | None = None,
        transcribe_options: dict | None = None,
        kiosk: bool = True,
    ) -> None:
        super().__init__()
        self._config_path = config_path
        self._output_dir = output_dir
        self._initial_mode = initial_mode
        self._record_options = record_options or {}
        self._transcribe_options = transcribe_options or {}
        self._last_mode: str | None = None
        self._kiosk = kiosk
        self._single_shot = initial_mode is not None
        self._record_state: RecordContext | None = None
        self._transcribe_state: TranscribeContext | None = None
        self._exit_stack: ExitStack | None = None
        self.session_artifacts: SessionArtifacts | None = None

    def on_mount(self) -> None:
        if self._initial_mode is not None:
            self._open_mode(self._initial_mode)
        else:
            self._show_welcome()

    # --- Routing ---------------------------------------------------------

    def _show_welcome(self) -> None:
        from lazy_take_notes.l4_frameworks_and_drivers.pickers.welcome_picker import (  # noqa: PLC0415 -- deferred
            WelcomePicker,
        )

        self.push_screen(WelcomePicker(initial_mode=self._last_mode), self._on_mode_selected)

    def _on_mode_selected(self, mode: str | None) -> None:
        if mode is None:
            self.exit()
            return
        self._last_mode = mode
        self._open_mode(mode)

    def _open_mode(self, mode: str) -> None:
        if mode == 'record':
            self._open_record()
        elif mode == 'transcribe':
            self._open_transcribe()
        elif mode == 'view':
            self._open_session_picker()
        elif mode == 'config':
            self._open_config()
        elif mode == 'create-template':
            self._open_template_builder()

    def _finished(self, _result=None) -> None:
        if self._exit_stack is not None:
            self._exit_stack.close()
            self._exit_stack = None
        if self._kiosk and not self._single_shot:
            self._show_welcome()
        else:
            self.exit()

    def _record_context(self) -> RecordContext:
        if self._record_state is None:
            raise RuntimeError('record context requested before the record flow started')
        return self._record_state

    def _transcribe_context(self) -> TranscribeContext:
        if self._transcribe_state is None:
            raise RuntimeError('transcribe context requested before the transcribe flow started')
        return self._transcribe_state

    # --- Record ----------------------------------------------------------

    def _open_record(self) -> None:
        from lazy_take_notes.l4_frameworks_and_drivers.cli_helpers import (  # noqa: PLC0415 -- deferred
            load_config,
            select_template,
        )
        from lazy_take_notes.l4_frameworks_and_drivers.pickers.template_picker import (  # noqa: PLC0415 -- deferred
            TemplatePicker,
        )

        config, infra, template_loader = load_config(self._config_path, self._output_dir)
        self._record_state = (config, infra, template_loader)
        opts = self._record_options
        if opts.get('template_name') or opts.get('language'):
            template = select_template(
                template_loader,
                template_name=opts.get('template_name'),
                language=opts.get('language'),
                show_builtins=infra.show_builtin_templates,
            )
            if template is None:
                self._finished()
                return
            self._start_record(template)
            return

        self.push_screen(
            TemplatePicker(show_builtins=infra.show_builtin_templates),
            self._on_record_template,
        )

    def _on_record_template(self, name: str | None) -> None:
        if name is None:
            self._finished()
            return
        if name == '__create_template__':
            self._open_template_builder(after=self._open_record)
            return
        _config, _infra, template_loader = self._record_context()
        self._start_record(template_loader.load(name))

    def _start_record(self, template) -> None:
        from lazy_take_notes.l4_frameworks_and_drivers.cli_helpers import (  # noqa: PLC0415 -- deferred
            build_record_session,
        )
        from lazy_take_notes.l4_frameworks_and_drivers.keep_awake import keep_awake  # noqa: PLC0415 -- deferred

        config, infra, _template_loader = self._record_context()
        opts = self._record_options
        screen, out_dir = build_record_session(
            config,
            infra,
            template,
            self._output_dir,
            opts.get('label'),
            llm_client=opts.get('llm_client'),
            transcriber=opts.get('transcriber'),
            audio_source=opts.get('audio_source'),
            mute_mic=opts.get('mute_mic', False),
        )
        self.session_artifacts = SessionArtifacts(session_dir=out_dir, notes_path=NOTES.resolve(out_dir))
        stack = ExitStack()
        stack.enter_context(keep_awake())
        self._exit_stack = stack
        self.push_screen(screen, self._finished)

    # --- Transcribe ------------------------------------------------------

    def _open_transcribe(self) -> None:
        opts = self._transcribe_options
        if opts.get('audio_path') is None and opts.get('subtitle_segments') is None:
            from lazy_take_notes.l4_frameworks_and_drivers.pickers.file_picker import (  # noqa: PLC0415 -- deferred
                FilePicker,
            )

            self.push_screen(FilePicker(), self._on_transcribe_file)
        else:
            self._on_transcribe_file(True)

    def _on_transcribe_file(self, selected) -> None:
        if selected is None:
            self._finished()
            return
        if selected is not True:
            self._transcribe_options['audio_path'] = Path(str(selected))

        from lazy_take_notes.l4_frameworks_and_drivers.cli_helpers import (  # noqa: PLC0415 -- deferred
            load_config,
            select_template,
        )
        from lazy_take_notes.l4_frameworks_and_drivers.pickers.template_picker import (  # noqa: PLC0415 -- deferred
            TemplatePicker,
        )

        config, infra, template_loader = load_config(self._config_path, self._output_dir)
        self._transcribe_state = (config, infra, template_loader)
        opts = self._transcribe_options
        if opts.get('template_name') or opts.get('language'):
            template = select_template(
                template_loader,
                template_name=opts.get('template_name'),
                language=opts.get('language'),
                show_builtins=infra.show_builtin_templates,
            )
            if template is None:
                self._finished()
                return
            self._start_transcribe(template)
            return

        self.push_screen(
            TemplatePicker(show_builtins=infra.show_builtin_templates),
            self._on_transcribe_template,
        )

    def _on_transcribe_template(self, name: str | None) -> None:
        if name is None:
            self._finished()
            return
        if name == '__create_template__':
            self._open_template_builder(after=self._open_transcribe)
            return
        _config, _infra, template_loader = self._transcribe_context()
        self._start_transcribe(template_loader.load(name))

    def _start_transcribe(self, template) -> None:
        from lazy_take_notes.l4_frameworks_and_drivers.cli_helpers import (  # noqa: PLC0415 -- deferred
            build_transcribe_session,
        )

        config, infra, _template_loader = self._transcribe_context()
        opts = self._transcribe_options
        screen, out_dir = build_transcribe_session(
            config,
            infra,
            template,
            self._output_dir,
            opts.get('label'),
            audio_path=opts.get('audio_path'),
            subtitle_segments=opts.get('subtitle_segments'),
            llm_client=opts.get('llm_client'),
            transcriber=opts.get('transcriber'),
        )
        self.session_artifacts = SessionArtifacts(session_dir=out_dir, notes_path=NOTES.resolve(out_dir))
        self.push_screen(screen, self._finished)

    # --- View, settings, template builder --------------------------------

    def _open_session_picker(self) -> None:
        from lazy_take_notes.l4_frameworks_and_drivers.cli_helpers import (  # noqa: PLC0415 -- deferred
            load_config,
            resolve_base_dir,
        )
        from lazy_take_notes.l4_frameworks_and_drivers.pickers.session_picker import (  # noqa: PLC0415 -- deferred
            SessionPicker,
        )

        config, _infra, _loader = load_config(self._config_path, self._output_dir)
        base_dir = resolve_base_dir(self._output_dir, config)
        self.push_screen(SessionPicker(sessions_dir=base_dir), self._on_session_selected)

    def _on_session_selected(self, session_dir: Path | None) -> None:
        if session_dir is None:
            self._finished()
            return
        from lazy_take_notes.l4_frameworks_and_drivers.apps.view import ViewApp  # noqa: PLC0415 -- deferred

        self.push_screen(ViewApp(session_dir), self._finished)

    def _open_config(self) -> None:
        from lazy_take_notes.l4_frameworks_and_drivers.apps.config import ConfigApp  # noqa: PLC0415 -- deferred

        self.push_screen(ConfigApp(), self._finished)

    def _open_template_builder(self, after=None) -> None:
        from lazy_take_notes.l4_frameworks_and_drivers.apps.template_builder import (  # noqa: PLC0415 -- deferred
            TemplateBuilderApp,
        )

        self.push_screen(TemplateBuilderApp(), after or self._finished)


def run_shell(config_path=None, output_dir: str | None = None, **kwargs) -> ShellApp:
    """Create and run a shell. Returns it so callers can read session artifacts."""
    shell = ShellApp(config_path=config_path, output_dir=output_dir, **kwargs)
    shell.run()
    return shell

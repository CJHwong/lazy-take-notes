"""ConfigApp — TUI form for editing app configuration."""

from __future__ import annotations

import subprocess  # noqa: S404 -- used for launching $EDITOR, not shell commands

from pydantic import ValidationError
from textual.app import ComposeResult, SuspendNotSupported
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import (
    Button,
    ContentSwitcher,
    Input,
    ListItem,
    ListView,
    Select,
    Static,
    Switch,
    TextArea,
)

from lazy_take_notes.l2_use_cases.ports.asset_store import ModelAdminStore
from lazy_take_notes.l3_interface_adapters.gateways.cached_model_store import CachedModelStore
from lazy_take_notes.l3_interface_adapters.gateways.yaml_config_loader import YamlConfigLoader, deep_merge
from lazy_take_notes.l3_interface_adapters.gateways.yaml_config_writer import config_file_path, write_config
from lazy_take_notes.l4_frameworks_and_drivers.apps.config.appearance_tab import AppearanceTab
from lazy_take_notes.l4_frameworks_and_drivers.apps.config.audio_tab import AudioTab
from lazy_take_notes.l4_frameworks_and_drivers.apps.config.dialogs import _DiscardChangesModal
from lazy_take_notes.l4_frameworks_and_drivers.apps.config.digest_tab import DigestTab
from lazy_take_notes.l4_frameworks_and_drivers.apps.config.fields import (
    plugin_manages_models,
    resolve_editor,
    to_float,
    to_int,
)
from lazy_take_notes.l4_frameworks_and_drivers.apps.config.models_tab import ModelsTab
from lazy_take_notes.l4_frameworks_and_drivers.apps.config.sessions_tab import SessionsTab
from lazy_take_notes.l4_frameworks_and_drivers.config import (
    APP_CONFIG_DEFAULTS,
    InfraConfig,
    build_app_config,
)
from lazy_take_notes.l4_frameworks_and_drivers.container import (
    BUILTIN_LLM_PROVIDERS,
    BUILTIN_TRANSCRIPTION_PROVIDERS,
    LLM_PROVIDERS_GROUP,
    TRANSCRIPTION_PROVIDERS_GROUP,
    DependencyContainer,
)
from lazy_take_notes.l4_frameworks_and_drivers.screen_base import ShellScreen

# ── Theme preview ─────────────────────────────────────────────────────────────


class ConfigApp(ShellScreen):
    """TUI form for editing lazy-take-notes configuration."""

    CSS = """
    Input {
        height: 1;
        border: none;
        background: $surface-lighten-1;
        padding: 0 1;
    }
    Input:focus {
        background: $surface-lighten-2;
    }
    Select {
        height: 3;
        border: none;
        background: $surface-lighten-1;
    }
    Button {
        height: 1;
        min-width: 8;
        border: none;
        background: $surface-lighten-2;
        color: $text;
        padding: 0 1;
    }
    Button:hover {
        background: $primary;
        color: $text;
    }
    Switch {
        height: 1;
        border: none;
    }
    #cfg-title {
        height: 1;
        background: $primary;
        color: $text;
        text-style: bold;
        padding: 0 1;
    }
    #cfg-path {
        height: 1;
        background: $surface;
        color: $text-muted;
        padding: 0 1;
    }
    #cfg-shell {
        height: 1fr;
    }
    #cfg-nav {
        width: 20;
        height: 1fr;
        background: $surface;
        border-right: solid $surface-lighten-2;
    }
    #cfg-nav ListItem {
        height: 1;
        padding: 0 1;
    }
    #cfg-nav ListItem.-highlight {
        background: $primary;
        color: $text;
    }
    #cfg-sections {
        width: 1fr;
        height: 1fr;
    }
    #cfg-footer {
        dock: bottom;
        height: 1;
        background: $surface;
        color: $text-muted;
        padding: 0 1;
    }
    .section-title {
        height: 1;
        text-style: bold;
        color: $accent;
        margin: 1 0 0 0;
    }
    .section-rule {
        height: 1;
        color: $surface-lighten-2;
        margin: 0 0 1 0;
    }
    .section-desc {
        color: $text-muted;
        height: auto;
        margin: 0 0 1 0;
    }
    #cfg-recognition-hints {
        min-height: 5;
        max-height: 16;
        margin: 0 0 1 0;
    }
    """

    BINDINGS = [
        Binding('escape', 'quit_app', 'Quit', priority=True),
        Binding('ctrl+s', 'save_config', 'Save', priority=True),
    ]

    def __init__(self, asset_store: ModelAdminStore | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._asset_store = asset_store or CachedModelStore()
        self._saved = False
        self._clear_api_key = False
        self._raw: dict = {}
        self._baseline: dict | None = None
        self._load_current_config()

    def _load_current_config(self) -> None:
        """Load existing config merged with defaults."""
        import copy  # noqa: PLC0415 -- one-shot use

        loader = YamlConfigLoader()
        user_raw = loader.load()
        merged = copy.deepcopy(APP_CONFIG_DEFAULTS)
        deep_merge(merged, user_raw)
        self._raw = merged
        try:
            self._infra = InfraConfig.model_validate(user_raw)
        except ValidationError:
            self._infra = InfraConfig()

    NAV = (
        ('models', 'Models'),
        ('digest', 'Digest'),
        ('audio', 'Audio'),
        ('sessions', 'Sessions'),
        ('appearance', 'Appearance'),
    )

    def compose(self) -> ComposeResult:
        yield Static('Settings', id='cfg-title')
        yield Static(str(config_file_path()), id='cfg-path')
        with Horizontal(id='cfg-shell'):
            yield ListView(
                *[ListItem(Static(label), id=f'nav-{key}') for key, label in self.NAV],
                id='cfg-nav',
            )
            with ContentSwitcher(initial='section-models', id='cfg-sections'):
                yield ModelsTab(self._asset_store, self._raw, self._infra, id='section-models')
                yield DigestTab(self._raw, id='section-digest')
                yield AudioTab(self._raw, id='section-audio')
                yield SessionsTab(self._raw, self._infra, id='section-sessions')
                yield AppearanceTab(self._infra, id='section-appearance')
        yield Static(
            r'\[Ctrl+S] Save  \[t] Test AI  \[e] Edit raw file  \[Esc] Back',
            id='cfg-footer',
            markup=True,
        )

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        item = event.item
        if item is not None and item.id:
            self.query_one('#cfg-sections', ContentSwitcher).current = f'section-{item.id.removeprefix("nav-")}'

    def on_key(self, event) -> None:
        if event.key == 't' and not isinstance(self.focused, Input):
            self._test_connection()
            event.prevent_default()
        elif event.key == 'e' and not isinstance(self.focused, Input):
            self._open_raw_editor()
            event.prevent_default()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ''
        if button_id == 'cfg-openai-clear-key':
            self.query_one('#cfg-openai-api-key', Input).value = ''
            self._clear_api_key = True
            self.notify('The stored key is removed on save.', timeout=4)
        elif button_id == 'cfg-ai-test':
            self._test_connection()
        elif button_id == 'cfg-ollama-refresh':
            self._refresh_ollama()
        elif button_id == 'cfg-ollama-pull-digest':
            self._pull_ollama('#cfg-digest-model')
        elif button_id == 'cfg-ollama-pull-interactive':
            self._pull_ollama('#cfg-interactive-model')

    def _refresh_ollama(self) -> None:
        try:
            infra = InfraConfig.model_validate(self._collect_form_data())
        except ValidationError as exc:
            self.notify(f'Invalid provider config: {exc}', severity='error')
            return
        self.query_one('#cfg-ollama-models', Static).update('Loading...')
        self.run_worker(lambda: self._refresh_ollama_worker(infra), thread=True, group='ollama-list')

    def _refresh_ollama_worker(self, infra: InfraConfig) -> None:
        try:
            client = DependencyContainer.resolve_llm_client(infra)
            list_models = getattr(client, 'list_models', None)
            if list_models is None:
                text = 'This provider does not list models.'
            else:
                models = list_models()
                text = ', '.join(name for name, _size in models) or 'No models installed.'
        except Exception as exc:  # noqa: BLE001 -- surface the server error in the tab
            text = f'Cannot list models: {exc}'
        self.app.call_from_thread(self._set_ollama_models, text)

    def _set_ollama_models(self, text: str) -> None:
        self.query_one('#cfg-ollama-models', Static).update(text)

    def _pull_ollama(self, input_id: str) -> None:
        model = self.query_one(input_id, Input).value.strip()
        if not model:
            self.notify('Enter a model name first.', severity='warning', timeout=4)
            return
        try:
            infra = InfraConfig.model_validate(self._collect_form_data())
        except ValidationError as exc:
            self.notify(f'Invalid provider config: {exc}', severity='error')
            return
        self.query_one('#cfg-ollama-models', Static).update(f'Pulling {model}...')
        self.run_worker(lambda: self._pull_ollama_worker(model, infra), thread=True, group='ollama-pull')

    def _pull_ollama_worker(self, model: str, infra: InfraConfig) -> None:
        try:
            client = DependencyContainer.resolve_llm_client(infra)
            pull = getattr(client, 'pull_model', None)
            if pull is None:
                raise RuntimeError('This provider cannot pull models.')

            def report(event) -> None:
                status = getattr(event, 'status', '')
                self.app.call_from_thread(self._set_ollama_models, f'Pulling {model}: {status}')

            pull(model, on_progress=report)
        except Exception as exc:  # noqa: BLE001 -- surface the server error in the tab
            self.app.call_from_thread(self._pull_failed, model, str(exc))
        else:
            self.app.call_from_thread(self._pull_done, model)

    def _pull_failed(self, model: str, error: str) -> None:
        self.query_one('#cfg-ollama-models', Static).update(f'Pull failed: {error}')
        self.notify(f'Pull failed for {model}: {error}', severity='error', timeout=10)

    def _pull_done(self, model: str) -> None:
        self.notify(f'{model} is ready.', timeout=5)
        self._refresh_ollama()

    def _model_value(self) -> str:
        """Return the custom path when set, otherwise the selected model."""
        custom = self.query_one('#cfg-trans-custom-path', Input).value.strip()
        if custom:
            return custom
        return self.query_one(ModelsTab).model_value()

    def _input_device(self) -> str | None:
        value = str(self.query_one('#cfg-audio-device', Select).value)
        return None if value == '__default__' else value

    def _collect_form_data(self) -> dict:
        """Read all form fields and build a config dict."""
        provider = self.query_one('#cfg-llm-provider', Select).value
        trans_provider = self.query_one('#cfg-trans-provider', Select).value
        diarization_raw = self._raw.get('diarization', {})
        openai_config: dict[str, object] = {
            'base_url': self.query_one('#cfg-openai-base-url', Input).value.strip(),
        }
        api_key = self.query_one('#cfg-openai-api-key', Input).value.strip()
        if self._clear_api_key:
            openai_config['api_key'] = None
        elif api_key:
            openai_config['api_key'] = api_key
        data: dict = {
            'llm_provider': provider,
            'transcription_provider': str(trans_provider),
            'theme': str(self.query_one('#cfg-theme', Select).value),
            'show_builtin_templates': self.query_one('#cfg-show-builtin-templates', Switch).value,
            'ollama': {
                'host': self.query_one('#cfg-ollama-host', Input).value.strip(),
            },
            'openai': openai_config,
            'transcription': {
                'model': self._model_value(),
                'models': self.query_one(ModelsTab).language_models(),
                'input_device': self._input_device(),
                'chunk_duration': to_float(self.query_one('#cfg-trans-chunk', Input).value),
                'overlap': to_float(self.query_one('#cfg-trans-overlap', Input).value),
                'silence_threshold': to_float(self.query_one('#cfg-trans-silence', Input).value),
                'pause_duration': to_float(self.query_one('#cfg-trans-pause', Input).value),
            },
            'digest': {
                'model': self.query_one('#cfg-digest-model', Input).value.strip(),
                'min_lines': to_int(self.query_one('#cfg-digest-min-lines', Input).value),
                'min_interval': to_float(self.query_one('#cfg-digest-min-interval', Input).value),
                'compact_token_threshold': to_int(self.query_one('#cfg-digest-compact', Input).value),
            },
            'diarization': {
                'enabled': self.query_one('#cfg-diarization-enabled', Switch).value,
                'binary': str(diarization_raw.get('binary', '')),
                'model': str(diarization_raw.get('model', '')),
                'device': str(self.query_one('#cfg-diarization-device', Select).value),
                'interval_s': to_float(self.query_one('#cfg-diarization-interval', Input).value),
            },
            'interactive': {
                'model': self.query_one('#cfg-interactive-model', Input).value.strip(),
            },
            'output': {
                'directory': self.query_one('#cfg-output-dir', Input).value.strip(),
                'save_audio': self.query_one('#cfg-output-save-audio', Switch).value,
                'save_notes_history': self.query_one('#cfg-output-save-notes-history', Switch).value,
                'save_context': self.query_one('#cfg-output-save-context', Switch).value,
                'save_debug_log': self.query_one('#cfg-output-save-debug-log', Switch).value,
                'auto_label': self.query_one('#cfg-output-auto-label', Switch).value,
            },
        }
        max_lines_text = self.query_one('#cfg-digest-max-lines', Input).value.strip()
        if max_lines_text:
            data['digest']['max_lines'] = to_int(max_lines_text)

        hints_text = self.query_one('#cfg-recognition-hints', TextArea).text.strip()
        if hints_text:
            data['recognition_hints'] = [line.strip() for line in hints_text.splitlines() if line.strip()]
        else:
            data['recognition_hints'] = []

        return data

    def on_mount(self) -> None:
        """Apply the saved theme, wire the nav, and snapshot the form."""
        self.app.theme = self._infra.theme
        self._sync_model_fields_visibility(self._infra.llm_provider)
        self._sync_transcription_model_fields_visibility(self._infra.transcription_provider)
        nav = self.query_one('#cfg-nav', ListView)
        nav.index = 0
        nav.focus()
        self._baseline = self._collect_form_data()

    def on_select_changed(self, event: Select.Changed) -> None:
        """Toggle model fields when provider changes; preview theme on change."""
        if event.select.id == 'cfg-llm-provider':
            self._sync_model_fields_visibility(str(event.value))
        elif event.select.id == 'cfg-trans-provider':
            self._sync_transcription_model_fields_visibility(str(event.value))
        elif event.select.id == 'cfg-theme':
            self.app.theme = str(event.value)

    def _sync_model_fields_visibility(self, provider: str) -> None:
        self._sync_provider_visibility(
            provider,
            LLM_PROVIDERS_GROUP,
            BUILTIN_LLM_PROVIDERS,
            'model-fields',
            'plugin-model-note',
        )

    def _sync_transcription_model_fields_visibility(self, provider: str) -> None:
        hide = plugin_manages_models(provider, TRANSCRIPTION_PROVIDERS_GROUP, BUILTIN_TRANSCRIPTION_PROVIDERS)
        self.query_one('#models-transcription-fields').display = not hide
        self.query_one('#models-language-fields').display = not hide
        self.query_one('#plugin-trans-model-note').display = hide

    def _sync_provider_visibility(
        self,
        provider: str,
        group: str,
        builtins: tuple[str, ...],
        fields_id: str,
        note_id: str,
    ) -> None:
        hide = plugin_manages_models(provider, group, builtins)
        self.query_one(f'#{fields_id}').display = not hide
        self.query_one(f'#{note_id}').display = hide

    def action_save_config(self) -> None:
        """Validate form data, then write to config.yaml."""
        errors = self._numeric_errors()
        if errors:
            self.notify('\n'.join(errors), severity='error', timeout=10)
            return
        data = self._collect_form_data()
        if data['diarization']['enabled'] and not data['output']['save_audio']:
            self.notify(
                'Speaker labels need Save Audio Recording turned on.',
                severity='error',
                timeout=8,
            )
            return
        try:
            build_app_config(data)
            InfraConfig.model_validate(data)
        except (ValidationError, ValueError) as exc:
            self.notify(f'Validation error: {exc}', severity='error', timeout=8)
            return

        path = write_config(data)
        self._saved = True
        self._clear_api_key = False
        self._baseline = self._collect_form_data()
        self.notify(f'Saved to {path}', severity='information')

    def _numeric_errors(self) -> list[str]:
        """Reject a non-numeric or out-of-range field instead of storing 0."""
        checks = (
            ('#cfg-trans-chunk', 'Chunk Duration', 'float', 0.0, None),
            ('#cfg-trans-overlap', 'Overlap', 'float', 0.0, None),
            ('#cfg-trans-silence', 'Silence Threshold', 'float', 0.0, 1.0),
            ('#cfg-trans-pause', 'Pause Duration', 'float', 0.0, None),
            ('#cfg-digest-min-lines', 'Minimum Lines', 'int', 1, None),
            ('#cfg-digest-min-interval', 'Minimum Interval', 'float', 0.0, None),
            ('#cfg-digest-compact', 'Compaction Threshold', 'int', 1, None),
            ('#cfg-digest-max-lines', 'Force After Lines', 'int', 1, None),
            ('#cfg-diarization-interval', 'Diarization Interval', 'float', 0.0, None),
        )
        errors: list[str] = []
        for widget_id, label, kind, minimum, maximum in checks:
            text = self.query_one(widget_id, Input).value.strip()
            if not text:
                continue
            try:
                value = int(text) if kind == 'int' else float(text)
            except ValueError:
                errors.append(f'{label} must be a number.')
                continue
            if value < minimum:
                errors.append(f'{label} must be at least {minimum}.')
            if maximum is not None and value > maximum:
                errors.append(f'{label} must be at most {maximum}.')
        return errors

    def _test_connection(self) -> None:
        """Check LLM connectivity using current form values."""
        data = self._collect_form_data()
        try:
            infra = InfraConfig.model_validate(data)
        except ValidationError as exc:
            self.notify(f'Invalid provider config: {exc}', severity='error')
            return

        from lazy_take_notes.l4_frameworks_and_drivers.container import (  # noqa: PLC0415 -- deferred: only needed on test
            DependencyContainer,
        )

        try:
            client = DependencyContainer.resolve_llm_client(infra)
        except ValueError as exc:
            self.notify(str(exc), severity='error', timeout=8)
            return

        ok, err = client.check_connectivity()
        if ok:
            self.notify('Connection OK', severity='information')
        else:
            self.notify(f'Connection failed: {err}', severity='error', timeout=8)

    def _open_raw_editor(self) -> None:
        """Open config.yaml in $EDITOR."""
        editor_argv = resolve_editor()
        if editor_argv is None:
            self.notify('No editor found ($VISUAL / $EDITOR)', severity='error')
            return
        path = config_file_path()
        if not path.exists():
            from lazy_take_notes.l3_interface_adapters.gateways.paths import CONFIG_DIR  # noqa: PLC0415 -- deferred

            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            import yaml as _yaml  # noqa: PLC0415 -- one-shot use

            path.write_text(
                _yaml.dump(self._raw, default_flow_style=False, allow_unicode=True, sort_keys=False),
                encoding='utf-8',
            )
        try:
            with self.app.suspend():
                subprocess.run([*editor_argv, str(path)], check=False)  # noqa: S603 -- argv from env/platform, not user shell
        except SuspendNotSupported:
            self.notify('Cannot open editor in this environment', severity='error')
            return
        self._load_current_config()
        self._repopulate_fields()
        self._baseline = self._collect_form_data()
        self.notify('Reloaded from disk')

    def _repopulate_provider_select(self, widget_id: str, provider: str) -> None:
        """Update a provider Select widget, adding the value if not in options."""
        select = self.query_one(widget_id, Select)
        current_values = [str(v) for _, v in select._options]
        if provider not in current_values:
            select.set_options([(n, n) for n in [*current_values, provider]])
        select.value = provider

    def _repopulate_fields(self) -> None:
        """Push current self._raw / self._infra values back into form widgets."""
        self._repopulate_provider_select('#cfg-llm-provider', self._infra.llm_provider)
        self._repopulate_provider_select('#cfg-trans-provider', self._infra.transcription_provider)

        self.query_one('#cfg-ollama-host', Input).value = self._infra.ollama.host
        self.query_one('#cfg-openai-base-url', Input).value = self._infra.openai.base_url
        self.query_one('#cfg-openai-api-key', Input).value = self._infra.openai.api_key or ''

        trans = self._raw.get('transcription', {})
        self.query_one(ModelsTab).reload(self._raw)
        self.query_one(AudioTab).reload(self._raw)
        self.query_one('#cfg-trans-chunk', Input).value = str(trans.get('chunk_duration', ''))
        self.query_one('#cfg-trans-overlap', Input).value = str(trans.get('overlap', ''))
        self.query_one('#cfg-trans-silence', Input).value = str(trans.get('silence_threshold', ''))
        self.query_one('#cfg-trans-pause', Input).value = str(trans.get('pause_duration', ''))

        digest = self._raw.get('digest', {})
        self.query_one('#cfg-digest-model', Input).value = str(digest.get('model', ''))
        self.query_one('#cfg-digest-min-lines', Input).value = str(digest.get('min_lines', ''))
        self.query_one('#cfg-digest-min-interval', Input).value = str(digest.get('min_interval', ''))
        self.query_one('#cfg-digest-compact', Input).value = str(digest.get('compact_token_threshold', ''))

        max_lines = digest.get('max_lines')
        self.query_one('#cfg-digest-max-lines', Input).value = '' if max_lines is None else str(max_lines)

        diarization = self._raw.get('diarization', {})
        self.query_one('#cfg-diarization-enabled', Switch).value = bool(diarization.get('enabled', False))
        self.query_one('#cfg-diarization-device', Select).value = str(diarization.get('device', 'auto'))
        self.query_one('#cfg-diarization-interval', Input).value = str(diarization.get('interval_s', 60))

        interactive = self._raw.get('interactive', {})
        self.query_one('#cfg-interactive-model', Input).value = str(interactive.get('model', ''))

        output = self._raw.get('output', {})
        self.query_one('#cfg-output-dir', Input).value = str(output.get('directory', ''))
        self.query_one('#cfg-output-save-audio', Switch).value = output.get('save_audio', True)
        self.query_one('#cfg-output-save-notes-history', Switch).value = output.get('save_notes_history', True)
        self.query_one('#cfg-output-save-context', Switch).value = output.get('save_context', True)
        self.query_one('#cfg-output-save-debug-log', Switch).value = output.get('save_debug_log', False)
        self.query_one('#cfg-output-auto-label', Switch).value = output.get('auto_label', True)
        self.query_one('#cfg-recognition-hints', TextArea).text = '\n'.join(self._raw.get('recognition_hints', []))
        self.query_one('#cfg-theme', Select).value = self._infra.theme

    def _has_unsaved_changes(self) -> bool:
        """True when the form no longer matches the values loaded or last saved."""
        return self._baseline is not None and self._collect_form_data() != self._baseline

    def action_quit_app(self) -> None:
        if isinstance(self.app.screen, _DiscardChangesModal):
            self.app.screen.dismiss(False)
            return
        if self._has_unsaved_changes():
            self.app.push_screen(_DiscardChangesModal(), self._after_discard_prompt)
            return
        self.dismiss(self._saved)

    def _after_discard_prompt(self, discard: bool | None) -> None:
        if discard:
            self.dismiss(self._saved)

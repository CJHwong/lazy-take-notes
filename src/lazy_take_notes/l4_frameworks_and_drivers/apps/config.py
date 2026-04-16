"""ConfigApp — TUI form for editing app configuration."""

from __future__ import annotations

import subprocess  # noqa: S404 -- used for launching $EDITOR, not shell commands
import sys

from pydantic import ValidationError
from textual.app import App as TextualApp
from textual.app import ComposeResult, SuspendNotSupported
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Collapsible,
    Input,
    Label,
    Select,
    Static,
    Switch,
    TabbedContent,
    TabPane,
    TextArea,
)

from lazy_take_notes.l3_interface_adapters.gateways.yaml_config_loader import YamlConfigLoader, deep_merge
from lazy_take_notes.l3_interface_adapters.gateways.yaml_config_writer import config_file_path, write_config
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
)
from lazy_take_notes.l4_frameworks_and_drivers.widgets.digest_panel import DigestPanel
from lazy_take_notes.l4_frameworks_and_drivers.widgets.status_bar import StatusBar
from lazy_take_notes.l4_frameworks_and_drivers.widgets.transcript_panel import TranscriptPanel


def _resolve_editor() -> list[str] | None:
    """Resolve the user's preferred editor command (reuses logic from template_picker)."""
    import os  # noqa: PLC0415 -- deferred: only needed when editing
    import platform  # noqa: PLC0415 -- deferred: only needed when editing

    for var in ('VISUAL', 'EDITOR'):
        value = os.environ.get(var, '').strip()
        if value:
            return value.split()
    fallbacks = {
        'darwin': ['open', '-t'],
        'linux': ['xdg-open'],
        'win32': ['notepad'],
    }
    plat = sys.platform if sys.platform in fallbacks else platform.system().lower()
    return fallbacks.get(plat)


# ── Field widgets ────────────────────────────────────────────────────────────


class _FieldRow(Vertical):
    """Labeled input field with help text."""

    DEFAULT_CSS = """
    _FieldRow {
        height: auto;
        margin: 0 0 0 0;
        padding: 0 0;
    }
    _FieldRow .field-label {
        height: 1;
        padding: 0;
        text-style: bold;
    }
    _FieldRow .field-help {
        height: auto;
        padding: 0;
        color: $text-muted;
    }
    _FieldRow Input {
        margin: 0 0 1 0;
    }
    """

    def __init__(
        self,
        label: str,
        field_id: str,
        value: str,
        *,
        help_text: str = '',
        placeholder: str = '',
        password: bool = False,
    ) -> None:
        super().__init__()
        self._label = label
        self._field_id = field_id
        self._value = value
        self._help_text = help_text
        self._placeholder = placeholder
        self._password = password

    def compose(self) -> ComposeResult:
        yield Label(self._label, classes='field-label')
        if self._help_text:
            yield Static(self._help_text, classes='field-help')
        yield Input(
            value=self._value,
            id=self._field_id,
            placeholder=self._placeholder,
            password=self._password,
        )


class _SwitchRow(Horizontal):
    """Label + Switch side-by-side, with help text."""

    DEFAULT_CSS = """
    _SwitchRow {
        height: auto;
        margin: 0 0 1 0;
    }
    _SwitchRow .switch-group {
        width: 1fr;
        height: auto;
    }
    _SwitchRow .field-label {
        height: 1;
        padding: 0;
        text-style: bold;
    }
    _SwitchRow .field-help {
        height: auto;
        color: $text-muted;
    }
    _SwitchRow Switch {
        margin: 0 1 0 0;
    }
    """

    def __init__(self, label: str, field_id: str, value: bool, *, help_text: str = '') -> None:
        super().__init__()
        self._label = label
        self._field_id = field_id
        self._value = value
        self._help_text = help_text

    def compose(self) -> ComposeResult:
        yield Switch(value=self._value, id=self._field_id)
        with Vertical(classes='switch-group'):
            yield Label(self._label, classes='field-label')
            if self._help_text:
                yield Static(self._help_text, classes='field-help')


class _SelectRow(Vertical):
    """Label + Select dropdown with help text."""

    DEFAULT_CSS = """
    _SelectRow {
        height: auto;
        margin: 0 0 0 0;
    }
    _SelectRow .field-label {
        height: 1;
        padding: 0;
        text-style: bold;
    }
    _SelectRow .field-help {
        height: auto;
        color: $text-muted;
    }
    _SelectRow Select {
        margin: 0 0 1 0;
    }
    """

    def __init__(
        self,
        label: str,
        field_id: str,
        options: list[str],
        value: str,
        *,
        help_text: str = '',
    ) -> None:
        super().__init__()
        self._label = label
        self._field_id = field_id
        self._options = options
        self._value = value
        self._help_text = help_text

    def compose(self) -> ComposeResult:
        yield Label(self._label, classes='field-label')
        if self._help_text:
            yield Static(self._help_text, classes='field-help')
        options = self._options
        value = self._value
        if value not in options:
            options = [*options, value]
        choices = [(opt, opt) for opt in options]
        yield Select(choices, value=value, id=self._field_id, allow_blank=False)


def _discover_providers(group: str, builtins: tuple[str, ...]) -> list[str]:
    """Return built-in + plugin-registered provider names for an entry point group."""
    from importlib.metadata import entry_points  # noqa: PLC0415 -- deferred: not needed on --help

    plugin_names = [ep.name for ep in entry_points(group=group)]
    return [*builtins, *(n for n in plugin_names if n not in builtins)]


def _plugin_manages_models(provider: str, group: str, builtins: tuple[str, ...]) -> bool:
    """Check if a plugin provider manages its own model selection.

    Returns True when the provider's factory has ``manages_models = True``.
    Built-in providers always return False (they use the model fields).
    """
    if provider in builtins:
        return False

    from importlib.metadata import entry_points  # noqa: PLC0415 -- deferred: only on provider change

    for ep in entry_points(group=group):
        if ep.name == provider:
            factory = ep.load()
            return getattr(factory, 'manages_models', False)
    return False


# ── Theme preview ─────────────────────────────────────────────────────────────


_SAMPLE_TRANSCRIPT = [
    '[dim]\\[00:01:23][/dim] So the main takeaway from today is...',
    '[dim]\\[00:02:05][/dim] Right, and we should follow up on the',
    '[dim]\\[00:02:34][/dim] deployment timeline before next sprint.',
    '[dim]\\[00:03:12][/dim] Agreed. Let me also check the metrics',
    '[dim]\\[00:03:45][/dim] from last week before we commit to that.',
]

_SAMPLE_DIGEST = """\
## Meeting Summary

**Key decisions:**
- Follow up on deployment timeline
- Review last week metrics

## Action Items
- Check sprint metrics
"""


class RecordingPreview(Vertical):
    """Preview of the recording layout using the real widgets."""

    DEFAULT_CSS = """
    RecordingPreview {
        height: auto;
        margin: 1 0;
        border: round $surface-lighten-2;
        padding: 0;
    }
    RecordingPreview .preview-header {
        height: 1;
        background: $primary;
        color: $text;
        text-style: bold;
        padding: 0 1;
    }
    RecordingPreview .preview-panels {
        height: 12;
    }
    RecordingPreview .preview-panels > TranscriptPanel {
        width: 1fr;
    }
    RecordingPreview .preview-panels > DigestPanel {
        width: 1fr;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static('  lazy-take-notes | default_en', classes='preview-header')
        with Horizontal(classes='preview-panels'):
            yield TranscriptPanel(id='preview-transcript')
            yield DigestPanel(id='preview-digest')
        yield StatusBar(id='preview-status')

    def on_mount(self) -> None:
        transcript = self.query_one('#preview-transcript', TranscriptPanel)
        for line in _SAMPLE_TRANSCRIPT:
            transcript.write(line)

        digest = self.query_one('#preview-digest', DigestPanel)
        digest.update_digest(_SAMPLE_DIGEST)

        bar = self.query_one('#preview-status', StatusBar)
        bar.recording = True
        bar.mode_label = 'Record'
        bar.buf_count = 5
        bar.buf_max = 15
        bar.keybinding_hints = r'\[Space] pause  \[s] stop  \[d] digest  \[m] mute  \[h] help'


# ── ConfigApp ────────────────────────────────────────────────────────────────


class ConfigApp(TextualApp):
    """TUI form for editing lazy-take-notes configuration."""

    CSS = """
    #cfg-header {
        dock: top;
        height: 1;
        background: $primary;
        color: $text;
        text-align: center;
        text-style: bold;
        padding: 0 1;
    }
    #cfg-footer {
        dock: bottom;
        height: 1;
        background: $surface;
        color: $text-muted;
        text-align: center;
        padding: 0 1;
    }
    #cfg-tabs {
        height: 1fr;
    }
    TabPane {
        padding: 1 2;
    }
    /* Cap form width so it doesn't stretch across ultra-wide terminals */
    .form-scroll {
        max-width: 100;
    }
    .section-desc {
        color: $text-muted;
        height: auto;
        margin: 0 0 1 0;
    }
    .field-group {
        border: round $surface-lighten-2;
        padding: 1 2;
        margin: 0 0 1 0;
        height: auto;
    }
    .field-group-title {
        text-style: bold;
        height: 1;
        margin: 0 0 0 0;
        color: $text;
    }
    /* Collapsible for advanced sections — collapsed by default */
    Collapsible {
        margin: 0 0 1 0;
        padding: 0;
    }
    /* Recognition hints text area — sized to content, capped */
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

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._saved = False
        self._raw: dict = {}
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

    def compose(self) -> ComposeResult:
        cfg_path = config_file_path()
        yield Static(f'  Settings — {cfg_path}', id='cfg-header')

        with TabbedContent(id='cfg-tabs'):
            # ── Tab 1: AI Provider ───────────────────────────────────
            with TabPane('AI Provider', id='tab-provider'):
                with VerticalScroll(classes='form-scroll'):
                    yield Static(
                        'Choose how the app connects to an AI model for summaries and Q&A.',
                        classes='section-desc',
                    )

                    yield _SelectRow(
                        'AI Provider',
                        'cfg-llm-provider',
                        _discover_providers(LLM_PROVIDERS_GROUP, BUILTIN_LLM_PROVIDERS),
                        self._infra.llm_provider,
                        help_text=(
                            '"ollama" runs models locally (free, private). '
                            '"openai" uses a cloud API (needs an API key). '
                            'Plugin providers appear when installed.'
                        ),
                    )

                    with Vertical(classes='field-group'):
                        yield Static('Ollama (local AI)', classes='field-group-title')
                        yield _FieldRow(
                            'Server Address',
                            'cfg-ollama-host',
                            self._infra.ollama.host,
                            help_text='Leave as-is if Ollama runs on this machine.',
                            placeholder='http://localhost:11434',
                        )

                    with Vertical(classes='field-group'):
                        yield Static('OpenAI / Compatible API', classes='field-group-title')
                        yield _FieldRow(
                            'API Base URL',
                            'cfg-openai-base-url',
                            self._infra.openai.base_url,
                            help_text='Default for OpenAI. Change for Groq, Together, vLLM, etc.',
                            placeholder='https://api.openai.com/v1',
                        )
                        yield _FieldRow(
                            'API Key',
                            'cfg-openai-api-key',
                            self._infra.openai.api_key or '',
                            help_text='Leave blank to use the OPENAI_API_KEY environment variable.',
                            placeholder='sk-...',
                            password=True,
                        )

                    digest = self._raw.get('digest', {})
                    interactive = self._raw.get('interactive', {})

                    with Vertical(id='model-fields', classes='field-group'):
                        yield Static('AI Models', classes='field-group-title')
                        yield _FieldRow(
                            'Summary Model',
                            'cfg-digest-model',
                            str(digest.get('model', '')),
                            help_text='Generates rolling summaries from your transcript.',
                            placeholder='gpt-oss:20b',
                        )
                        yield _FieldRow(
                            'Quick-Action Model',
                            'cfg-interactive-model',
                            str(interactive.get('model', '')),
                            help_text='Handles keyboard-shortcut queries (keys 1–5). Can be the same model.',
                            placeholder='gpt-oss:20b',
                        )

                    yield Static(
                        'Models are managed by the plugin. Edit config.yaml to configure.',
                        id='plugin-model-note',
                        classes='section-desc',
                    )

            # ── Tab 2: Transcription ─────────────────────────────────
            with TabPane('Speech-to-Text', id='tab-transcription'):
                with VerticalScroll(classes='form-scroll'):
                    trans = self._raw.get('transcription', {})

                    yield Static(
                        'How spoken audio is converted to text. Defaults work for most cases.',
                        classes='section-desc',
                    )

                    yield _SelectRow(
                        'Speech-to-Text Engine',
                        'cfg-trans-provider',
                        _discover_providers(TRANSCRIPTION_PROVIDERS_GROUP, BUILTIN_TRANSCRIPTION_PROVIDERS),
                        self._infra.transcription_provider,
                        help_text=(
                            '"whisper-cpp" uses local Whisper inference (default). '
                            'Plugin engines appear when installed.'
                        ),
                    )

                    with Vertical(id='trans-model-fields', classes='field-group'):
                        yield Static('Model', classes='field-group-title')
                        yield _FieldRow(
                            'Whisper Model',
                            'cfg-trans-model',
                            str(trans.get('model', '')),
                            help_text=(
                                'Use hf://owner/repo/file for any HuggingFace GGUF model. '
                                'Short aliases (e.g. large-v3-turbo-q8_0) also work.'
                            ),
                            placeholder='hf://ggerganov/whisper.cpp/ggml-large-v3-turbo-q8_0.bin',
                        )
                        yield _FieldRow(
                            'Language-specific Models',
                            'cfg-trans-models',
                            _dict_to_inline(trans.get('models', {})),
                            help_text=('Override per language. Format: "zh: hf://owner/repo/file.bin, ja: model-name"'),
                            placeholder='zh: hf://alan314159/Breeze-ASR-25-whispercpp/ggml-model-q8_0.bin',
                        )

                    yield Static(
                        'Models are managed by the engine plugin. Edit config.yaml to configure.',
                        id='plugin-trans-model-note',
                        classes='section-desc',
                    )

                    with Collapsible(title='Advanced audio settings (usually no need to change)', collapsed=True):
                        yield _FieldRow(
                            'Chunk Duration (seconds)',
                            'cfg-trans-chunk',
                            str(trans.get('chunk_duration', '')),
                            help_text='Audio processed per batch. Longer = more context, more delay.',
                            placeholder='25.0',
                        )
                        yield _FieldRow(
                            'Overlap (seconds)',
                            'cfg-trans-overlap',
                            str(trans.get('overlap', '')),
                            help_text='Overlap between chunks to avoid cutting words.',
                            placeholder='1.0',
                        )
                        yield _FieldRow(
                            'Silence Threshold (0.0–1.0)',
                            'cfg-trans-silence',
                            str(trans.get('silence_threshold', '')),
                            help_text='Below this volume = silence. Lower = more sensitive.',
                            placeholder='0.01',
                        )
                        yield _FieldRow(
                            'Pause Duration (seconds)',
                            'cfg-trans-pause',
                            str(trans.get('pause_duration', '')),
                            help_text='Silence length that triggers a text break.',
                            placeholder='1.5',
                        )

            # ── Tab 3: Summaries ─────────────────────────────────────
            with TabPane('Summaries', id='tab-digest'):
                with VerticalScroll(classes='form-scroll'):
                    digest = self._raw.get('digest', {})

                    yield Static(
                        'Controls live summaries and quick-action Q&A.',
                        classes='section-desc',
                    )

                    with Vertical(classes='field-group'):
                        yield Static('When to summarize', classes='field-group-title')
                        yield _FieldRow(
                            'After at least N lines',
                            'cfg-digest-min-lines',
                            str(digest.get('min_lines', '')),
                            help_text='Minimum transcript lines before a summary is generated.',
                            placeholder='15',
                        )
                        yield _FieldRow(
                            'Wait at least N seconds',
                            'cfg-digest-min-interval',
                            str(digest.get('min_interval', '')),
                            help_text='Minimum gap between consecutive summaries.',
                            placeholder='60',
                        )

                    with Collapsible(title='Advanced (rarely needed)', collapsed=True):
                        yield _FieldRow(
                            'Token Compaction Threshold',
                            'cfg-digest-compact',
                            str(digest.get('compact_token_threshold', '')),
                            help_text=(
                                'Compress history when it exceeds this token count. '
                                'Change only if you hit context-length errors.'
                            ),
                            placeholder='100000',
                        )

                    with Vertical(classes='field-group'):
                        yield Static('Templates', classes='field-group-title')
                        yield _SwitchRow(
                            'Show Built-in Templates',
                            'cfg-show-builtin-templates',
                            self._infra.show_builtin_templates,
                            help_text='Show bundled example templates in the picker. Disable to see only your own.',
                        )

            # ── Tab 4: Output ────────────────────────────────────────
            with TabPane('Output', id='tab-output'):
                with VerticalScroll(classes='form-scroll'):
                    output = self._raw.get('output', {})

                    yield Static(
                        'Where sessions are saved and what gets recorded.',
                        classes='section-desc',
                    )

                    with Vertical(classes='field-group'):
                        yield Static('Session storage', classes='field-group-title')
                        yield _FieldRow(
                            'Output Directory',
                            'cfg-output-dir',
                            str(output.get('directory', '')),
                            help_text='Base folder. A timestamped subfolder is created per session.',
                            placeholder='./output',
                        )
                        yield _SwitchRow(
                            'Save Audio Recording',
                            'cfg-output-save-audio',
                            output.get('save_audio', True),
                            help_text='Save a WAV alongside the transcript. Disable to save disk space.',
                        )
                        yield _SwitchRow(
                            'Save Notes History',
                            'cfg-output-save-notes-history',
                            output.get('save_notes_history', True),
                            help_text='Keep numbered snapshots in a history/ folder.',
                        )
                        yield _SwitchRow(
                            'Save Session Context',
                            'cfg-output-save-context',
                            output.get('save_context', True),
                            help_text='Save the context text you typed during the session.',
                        )
                        yield _SwitchRow(
                            'Save Debug Log',
                            'cfg-output-save-debug-log',
                            output.get('save_debug_log', False),
                            help_text='Write a debug.log file. Useful for troubleshooting.',
                        )
                        yield _SwitchRow(
                            'Auto-Label Sessions',
                            'cfg-output-auto-label',
                            output.get('auto_label', True),
                            help_text='Use AI to name unlabeled sessions after the final digest.',
                        )

                    with Vertical(classes='field-group'):
                        yield Static('Custom vocabulary', classes='field-group-title')
                        yield Static(
                            'Names, acronyms, or jargon the recognizer should watch for.\nOne entry per line.',
                            classes='field-help',
                        )
                        yield TextArea(
                            '\n'.join(self._raw.get('recognition_hints', [])),
                            id='cfg-recognition-hints',
                        )

            # ── Tab 5: Appearance ─────────────────────────────────
            with TabPane('Appearance', id='tab-appearance'):
                with VerticalScroll(classes='form-scroll'):
                    yield Static(
                        'Customize the look and feel of the app.',
                        classes='section-desc',
                    )
                    yield _SelectRow(
                        'Theme',
                        'cfg-theme',
                        list(self.available_themes.keys()),
                        self._infra.theme,
                        help_text='Choose a color theme. Preview updates instantly.',
                    )
                    yield RecordingPreview()

        yield Static(
            r'\[Ctrl+S] Save  \[t] Test AI connection  \[e] Edit raw file  \[Esc] Back',
            id='cfg-footer',
            markup=True,
        )

    def on_key(self, event) -> None:
        if event.key == 't' and not isinstance(self.focused, Input):
            self._test_connection()
            event.prevent_default()
        elif event.key == 'e' and not isinstance(self.focused, Input):
            self._open_raw_editor()
            event.prevent_default()

    def _collect_form_data(self) -> dict:
        """Read all form fields and build a config dict."""
        provider = self.query_one('#cfg-llm-provider', Select).value
        trans_provider = self.query_one('#cfg-trans-provider', Select).value
        data: dict = {
            'llm_provider': provider,
            'transcription_provider': str(trans_provider),
            'theme': str(self.query_one('#cfg-theme', Select).value),
            'show_builtin_templates': self.query_one('#cfg-show-builtin-templates', Switch).value,
            'ollama': {
                'host': self.query_one('#cfg-ollama-host', Input).value.strip(),
            },
            'openai': {
                'base_url': self.query_one('#cfg-openai-base-url', Input).value.strip(),
            },
            'transcription': {
                'model': self.query_one('#cfg-trans-model', Input).value.strip(),
                'models': _inline_to_dict(self.query_one('#cfg-trans-models', Input).value.strip()),
                'chunk_duration': _to_float(self.query_one('#cfg-trans-chunk', Input).value),
                'overlap': _to_float(self.query_one('#cfg-trans-overlap', Input).value),
                'silence_threshold': _to_float(self.query_one('#cfg-trans-silence', Input).value),
                'pause_duration': _to_float(self.query_one('#cfg-trans-pause', Input).value),
            },
            'digest': {
                'model': self.query_one('#cfg-digest-model', Input).value.strip(),
                'min_lines': _to_int(self.query_one('#cfg-digest-min-lines', Input).value),
                'min_interval': _to_float(self.query_one('#cfg-digest-min-interval', Input).value),
                'compact_token_threshold': _to_int(self.query_one('#cfg-digest-compact', Input).value),
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
        api_key = self.query_one('#cfg-openai-api-key', Input).value.strip()
        if api_key:
            data['openai']['api_key'] = api_key

        hints_text = self.query_one('#cfg-recognition-hints', TextArea).text.strip()
        if hints_text:
            data['recognition_hints'] = [line.strip() for line in hints_text.splitlines() if line.strip()]
        else:
            data['recognition_hints'] = []

        return data

    def on_mount(self) -> None:
        """Apply saved theme and set initial visibility of model fields."""
        self.theme = self._infra.theme
        self._sync_model_fields_visibility(self._infra.llm_provider)
        self._sync_transcription_model_fields_visibility(self._infra.transcription_provider)

    def on_select_changed(self, event: Select.Changed) -> None:
        """Toggle model fields when provider changes; preview theme on change."""
        if event.select.id == 'cfg-llm-provider':
            self._sync_model_fields_visibility(str(event.value))
        elif event.select.id == 'cfg-trans-provider':
            self._sync_transcription_model_fields_visibility(str(event.value))
        elif event.select.id == 'cfg-theme':
            self.theme = str(event.value)

    def _sync_model_fields_visibility(self, provider: str) -> None:
        self._sync_provider_visibility(
            provider,
            LLM_PROVIDERS_GROUP,
            BUILTIN_LLM_PROVIDERS,
            'model-fields',
            'plugin-model-note',
        )

    def _sync_transcription_model_fields_visibility(self, provider: str) -> None:
        self._sync_provider_visibility(
            provider,
            TRANSCRIPTION_PROVIDERS_GROUP,
            BUILTIN_TRANSCRIPTION_PROVIDERS,
            'trans-model-fields',
            'plugin-trans-model-note',
        )

    def _sync_provider_visibility(
        self,
        provider: str,
        group: str,
        builtins: tuple[str, ...],
        fields_id: str,
        note_id: str,
    ) -> None:
        hide = _plugin_manages_models(provider, group, builtins)
        self.query_one(f'#{fields_id}').display = not hide
        self.query_one(f'#{note_id}').display = hide

    def action_save_config(self) -> None:
        """Validate form data, then write to config.yaml."""
        data = self._collect_form_data()
        try:
            build_app_config(data)
            InfraConfig.model_validate(data)
        except (ValidationError, ValueError) as exc:
            self.notify(f'Validation error: {exc}', severity='error', timeout=8)
            return

        path = write_config(data)
        self._saved = True
        self.notify(f'Saved to {path}', severity='information')

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
        editor_argv = _resolve_editor()
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
            with self.suspend():
                subprocess.run([*editor_argv, str(path)], check=False)  # noqa: S603 -- argv from env/platform, not user shell
        except SuspendNotSupported:
            self.notify('Cannot open editor in this environment', severity='error')
            return
        self._load_current_config()
        self._repopulate_fields()
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
        self.query_one('#cfg-trans-model', Input).value = str(trans.get('model', ''))
        self.query_one('#cfg-trans-models', Input).value = _dict_to_inline(trans.get('models', {}))
        self.query_one('#cfg-trans-chunk', Input).value = str(trans.get('chunk_duration', ''))
        self.query_one('#cfg-trans-overlap', Input).value = str(trans.get('overlap', ''))
        self.query_one('#cfg-trans-silence', Input).value = str(trans.get('silence_threshold', ''))
        self.query_one('#cfg-trans-pause', Input).value = str(trans.get('pause_duration', ''))

        digest = self._raw.get('digest', {})
        self.query_one('#cfg-digest-model', Input).value = str(digest.get('model', ''))
        self.query_one('#cfg-digest-min-lines', Input).value = str(digest.get('min_lines', ''))
        self.query_one('#cfg-digest-min-interval', Input).value = str(digest.get('min_interval', ''))
        self.query_one('#cfg-digest-compact', Input).value = str(digest.get('compact_token_threshold', ''))

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

    def action_quit_app(self) -> None:
        self.exit(self._saved)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _to_float(value: str) -> float:
    """Parse string to float, returning 0.0 on failure."""
    try:
        return float(value.strip())
    except (ValueError, AttributeError):
        return 0.0


def _to_int(value: str) -> int:
    """Parse string to int, returning 0 on failure."""
    try:
        return int(value.strip())
    except (ValueError, AttributeError):
        return 0


def _dict_to_inline(mapping: dict) -> str:
    """Convert {'zh': 'breeze-q8'} to 'zh: breeze-q8, en: large-v3'."""
    return ', '.join(f'{k}: {v}' for k, v in mapping.items())


def _inline_to_dict(text: str) -> dict:
    """Parse 'zh: breeze-q8, en: large-v3' back to dict."""
    if not text:
        return {}
    result = {}
    for pair in text.split(','):
        pair = pair.strip()
        if ':' in pair:
            key, _, value = pair.partition(':')
            key = key.strip()
            value = value.strip()
            if key:
                result[key] = value
    return result

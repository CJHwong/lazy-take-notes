"""TemplateBuilderApp — chat-based TUI for LLM-powered template creation."""

from __future__ import annotations

from textual.app import App as TextualApp
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Center, Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Markdown, Static

from lazy_take_notes.l1_entities.chat_message import ChatMessage
from lazy_take_notes.l1_entities.template import SessionTemplate

# Standalone TextualApp — no BaseApp inheritance (like ViewApp, ConfigApp).
# Constructs its own LLM client from InfraConfig.

_MAX_AUTO_FIX = 2

_WELCOME = """\
Describe the template you want to create. For example:
- "1-on-1 template, English, track action items and career goals"
- "Lecture notes in Chinese, focus on key concepts and exam topics"

I'll guide you if I need more details, or generate right away if your \
description is complete enough.
"""


class _SaveTemplateScreen(ModalScreen[str | None]):
    """Modal to name the template before saving."""

    CSS = """
    _SaveTemplateScreen {
        align: center middle;
    }
    #save-dialog {
        width: 60;
        height: auto;
        border: thick $primary;
        background: $surface;
        padding: 1 2;
    }
    #save-label {
        width: 1fr;
        text-align: center;
        margin-bottom: 1;
    }
    #save-input {
        margin-bottom: 1;
    }
    #save-hint {
        color: $text-muted;
        text-align: center;
    }
    """

    BINDINGS = [
        Binding('escape', 'cancel', 'Cancel'),
    ]

    def __init__(self, suggested_name: str = '') -> None:
        super().__init__()
        self._suggested = suggested_name

    def compose(self) -> ComposeResult:
        with Vertical(id='save-dialog'):
            yield Static('Save Template', id='save-label')
            yield Input(
                value=self._suggested,
                placeholder='template_name (no spaces)',
                id='save-input',
            )
            yield Static(
                r'Enter to save, Esc to cancel',
                id='save-hint',
            )

    def on_mount(self) -> None:
        self.query_one('#save-input', Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        name = event.value.strip()
        if name:
            self.dismiss(name)

    def action_cancel(self) -> None:
        self.dismiss(None)


class _ConfirmQuitScreen(ModalScreen[bool]):
    """Modal warning about unsaved template on quit."""

    CSS = """
    _ConfirmQuitScreen {
        align: center middle;
    }
    #quit-dialog {
        width: 50;
        height: auto;
        border: thick $warning;
        background: $surface;
        padding: 1 2;
    }
    #quit-msg {
        width: 1fr;
        text-align: center;
        margin-bottom: 1;
    }
    #quit-buttons {
        width: 100%;
        align-horizontal: center;
    }
    #quit-buttons Button {
        margin: 0 1;
    }
    """

    BINDINGS = [
        Binding('y', 'confirm', 'Yes', show=False),
        Binding('n', 'dismiss_no', 'No', show=False),
        Binding('escape', 'dismiss_no', 'Cancel'),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id='quit-dialog'):
            yield Static(
                'You have an unsaved template. Quit anyway?',
                id='quit-msg',
            )
            with Center(id='quit-buttons'):
                yield Button('No (n)', id='quit-no', variant='default')
                yield Button('Yes (y)', id='quit-yes', variant='warning')

    def on_mount(self) -> None:
        self.query_one('#quit-no', Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == 'quit-yes')

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_dismiss_no(self) -> None:
        self.dismiss(False)


class TemplateBuilderApp(TextualApp):
    """Chat + preview TUI for building session templates with an LLM."""

    CSS = """
    #tb-header {
        dock: top;
        height: 1;
        background: $primary;
        color: $text;
        text-align: center;
        text-style: bold;
        padding: 0 1;
    }
    #tb-footer {
        dock: bottom;
        height: 1;
        background: $surface;
        color: $text-muted;
        text-align: center;
        padding: 0 1;
    }
    #tb-main {
        layout: horizontal;
        height: 1fr;
    }
    #tb-chat-col {
        width: 1fr;
    }
    #tb-chat-scroll {
        height: 1fr;
        border: tall $surface-lighten-2;
    }
    #tb-chat-log {
        height: auto;
        padding: 0 1;
    }
    #tb-input {
        dock: bottom;
        margin: 0 0;
    }
    #tb-preview-col {
        width: 1fr;
    }
    #tb-preview-scroll {
        height: 1fr;
        border: tall $surface-lighten-2;
    }
    #tb-preview {
        padding: 0 1;
    }
    """

    BINDINGS = [
        Binding('escape', 'quit_app', 'Back', priority=True),
        Binding('ctrl+s', 'save_template', 'Save', priority=True),
    ]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._history: list[ChatMessage] = []
        self._current_template: SessionTemplate | None = None
        self._use_case = None  # built lazily on first send
        self._model: str = ''
        self._busy = False
        self._chat_lines: list[str] = []
        self._thinking_timer = None
        self._thinking_tick = 0
        self._saved = False

    def compose(self) -> ComposeResult:
        yield Static('  Template Builder', id='tb-header')
        with Horizontal(id='tb-main'):
            with Vertical(id='tb-chat-col'):
                with VerticalScroll(id='tb-chat-scroll'):
                    yield Static('', id='tb-chat-log')
                yield Input(placeholder='Describe your template...', id='tb-input')
            with Vertical(id='tb-preview-col'):
                with VerticalScroll(id='tb-preview-scroll'):
                    yield Markdown('*No template yet — describe what you want.*', id='tb-preview')
        yield Static(
            r'\[Enter] Send  \[Ctrl+S] Save  \[Esc] Back',
            id='tb-footer',
            markup=True,
        )

    def on_mount(self) -> None:
        from lazy_take_notes.l4_frameworks_and_drivers.config import (  # noqa: PLC0415 -- deferred: avoid import at module level
            load_theme,
        )

        self.theme = load_theme()
        self._append_chat('assistant', _WELCOME)
        self.query_one('#tb-input', Input).focus()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        event.input.value = ''
        self._send_message(text)

    def _send_message(self, text: str) -> None:
        if self._busy:
            return
        self._append_chat('user', text)
        self._set_busy(True)
        self.run_worker(self._do_generate(text), exclusive=True, group='builder')

    def _set_busy(self, busy: bool) -> None:
        """Toggle the animated thinking indicator and input disabled state."""
        self._busy = busy
        inp = self.query_one('#tb-input', Input)
        if busy:
            self._thinking_tick = 0
            self._chat_lines.append('[dim italic]Thinking[/]')
            self._refresh_chat_log()
            inp.disabled = True
            self._thinking_timer = self.set_interval(0.4, self._animate_thinking)
        else:
            if self._thinking_timer is not None:
                self._thinking_timer.stop()
                self._thinking_timer = None
            # Remove the thinking line
            if self._chat_lines and 'Thinking' in self._chat_lines[-1]:
                self._chat_lines.pop()
                self._refresh_chat_log()
            inp.disabled = False
            inp.focus()

    def _animate_thinking(self) -> None:
        """Cycle the thinking indicator: Thinking. → Thinking.. → Thinking..."""
        if not self._chat_lines or 'Thinking' not in self._chat_lines[-1]:
            return
        self._thinking_tick = (self._thinking_tick + 1) % 3
        dots = '.' * (self._thinking_tick + 1)
        self._chat_lines[-1] = f'[dim italic]Thinking{dots}[/]'
        self._refresh_chat_log()

    async def _do_generate(self, user_message: str) -> None:
        """Worker: call LLM, handle auto-fix loop."""
        try:
            use_case = self._ensure_use_case()
            if use_case is None:
                self._append_chat('assistant', 'Error: could not initialize LLM client. Check your config.')
                return

            result = await use_case.generate(user_message, self._model, self._history)

            if result.error:
                self._append_chat('assistant', f'{result.assistant_message}\n\n*(Error: {result.error})*')
                return

            # Auto-fix loop for validation errors
            if result.validation_errors:
                for _attempt in range(_MAX_AUTO_FIX):
                    result = await use_case.auto_fix(result.validation_errors, self._model, self._history)
                    if not result.validation_errors and not result.error:
                        break
                if result.error:
                    self._append_chat('assistant', f'{result.assistant_message}\n\n*(Error: {result.error})*')
                    return

            if result.template:
                self._current_template = result.template
                self._saved = False
                self._update_preview()
                msg = result.assistant_message or 'Template generated.'
                self._append_chat('assistant', msg)
            else:
                self._append_chat('assistant', result.assistant_message)
        finally:
            self._set_busy(False)

    def _ensure_use_case(self):
        """Lazily build the use case + LLM client from InfraConfig."""
        if self._use_case is not None:
            return self._use_case

        from lazy_take_notes.l3_interface_adapters.gateways.yaml_config_loader import (  # noqa: PLC0415 -- deferred
            YamlConfigLoader,
        )
        from lazy_take_notes.l3_interface_adapters.gateways.yaml_template_loader import (  # noqa: PLC0415 -- deferred
            YamlTemplateLoader,
        )
        from lazy_take_notes.l4_frameworks_and_drivers.config import (  # noqa: PLC0415 -- deferred
            InfraConfig,
            build_app_config,
        )

        loader = YamlConfigLoader()
        raw = loader.load()
        config = build_app_config(raw)
        try:
            infra = InfraConfig.model_validate(raw)
        except Exception:  # noqa: BLE001 -- fallback to defaults on bad config
            infra = InfraConfig()

        self._model = config.digest.model

        # Build LLM client (same pattern as ConfigApp._test_connection)
        if infra.llm_provider == 'openai':
            from lazy_take_notes.l3_interface_adapters.gateways.openai_llm_client import (  # noqa: PLC0415 -- deferred
                OpenAICompatLLMClient,
            )

            client = OpenAICompatLLMClient(api_key=infra.openai.api_key, base_url=infra.openai.base_url)
        else:
            from lazy_take_notes.l3_interface_adapters.gateways.ollama_llm_client import (  # noqa: PLC0415 -- deferred
                OllamaLLMClient,
            )

            client = OllamaLLMClient(host=infra.ollama.host)

        # Load example template for the system prompt
        example = YamlTemplateLoader().load('default_en')

        from lazy_take_notes.l2_use_cases.template_builder_use_case import (  # noqa: PLC0415 -- deferred
            TemplateBuildUseCase,
        )

        self._use_case = TemplateBuildUseCase(llm_client=client, example_template=example)
        return self._use_case

    def _append_chat(self, role: str, text: str) -> None:
        """Append a message to the chat log display."""
        prefix = '[bold cyan]You[/]' if role == 'user' else '[bold green]AI[/]'
        self._chat_lines.append(f'{prefix}: {text}')
        self._refresh_chat_log()

    def _refresh_chat_log(self) -> None:
        """Re-render chat log from _chat_lines and scroll to bottom."""
        log = self.query_one('#tb-chat-log', Static)
        log.update('\n\n'.join(self._chat_lines))
        scroll = self.query_one('#tb-chat-scroll', VerticalScroll)
        scroll.scroll_end(animate=False)

    def _update_preview(self) -> None:
        """Render current template as Markdown in the preview pane."""
        tmpl = self._current_template
        if tmpl is None:
            return
        meta = tmpl.metadata
        lines = [
            f'## {meta.name or "Untitled"}',
            '',
            f'> {meta.description}' if meta.description else '',
            '',
            f'**Locale:** `{meta.locale or "not set"}`',
        ]

        if tmpl.quick_actions:
            lines += ['', '### Quick Actions']
            for i, qa in enumerate(tmpl.quick_actions):
                desc = f' — {qa.description}' if qa.description else ''
                lines.append(f'- **`{i + 1}`** {qa.label}{desc}')

        if tmpl.recognition_hints:
            lines += ['', f'**Recognition hints:** {", ".join(tmpl.recognition_hints)}']

        lines += ['', '---', '', '### System Prompt', '', tmpl.system_prompt]
        lines += ['', '---', '', '### Digest Template', '', tmpl.digest_user_template]

        self.query_one('#tb-preview', Markdown).update('\n'.join(lines))

    def action_save_template(self) -> None:
        if self._current_template is None:
            self.notify('No template to save — generate one first.', severity='warning')
            return
        suggested = self._current_template.metadata.key or _slugify(self._current_template.metadata.name)
        self.push_screen(
            _SaveTemplateScreen(suggested_name=suggested),
            callback=self._on_save_name,
        )

    def _on_save_name(self, name: str | None) -> None:
        if not name or self._current_template is None:
            return
        from lazy_take_notes.l3_interface_adapters.gateways.template_writer import (  # noqa: PLC0415 -- deferred
            save_user_template,
        )

        safe_name = _slugify(name)
        path = save_user_template(self._current_template, safe_name)
        self._saved = True
        self.notify(f'Saved as {safe_name} → {path}', severity='information')

    def action_quit_app(self) -> None:
        if self._current_template is not None and not self._saved:
            self.push_screen(_ConfirmQuitScreen(), callback=self._on_quit_confirmed)
        else:
            self.exit()

    def _on_quit_confirmed(self, confirmed: bool | None) -> None:
        if confirmed:
            self.exit()


def _slugify(text: str) -> str:
    """Convert text to a safe filename slug."""
    import re  # noqa: PLC0415 -- one-shot use

    slug = re.sub(r'[^\w]', '_', text.strip().lower())
    slug = re.sub(r'_+', '_', slug).strip('_')
    return slug or 'custom_template'

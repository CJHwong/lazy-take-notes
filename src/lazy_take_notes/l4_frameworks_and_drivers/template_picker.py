"""Template picker — small TUI to select a template before recording."""

from __future__ import annotations

from collections import defaultdict

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Input, ListItem, ListView, Markdown, Static

from lazy_take_notes.l1_entities.template import SessionTemplate
from lazy_take_notes.l3_interface_adapters.gateways.yaml_template_loader import (
    YamlTemplateLoader,
    all_template_names,
    user_template_names,
)


class LocaleHeader(ListItem):
    """Non-interactive group header showing a locale name (e.g. 'EN')."""

    def __init__(self, locale: str) -> None:
        super().__init__(disabled=True)
        self._locale = locale.upper()

    def compose(self) -> ComposeResult:
        yield Static(f'[bold]{self._locale}[/bold]', markup=True)


class TemplateItem(ListItem):
    """Selectable row representing a single template."""

    def __init__(self, name: str, locale: str, *, is_user: bool = False) -> None:
        super().__init__()
        self.template_name = name
        badge = '  [dim]\\[user][/dim]' if is_user else ''
        yield_text = f'{name}  [dim]({locale})[/dim]{badge}'
        self._label_text = yield_text

    def compose(self) -> ComposeResult:
        yield Static(self._label_text, markup=True)


class TemplatePicker(App[str | None]):
    CSS = """
    #picker-header {
        dock: top;
        height: 1;
        background: $primary;
        color: $text;
        text-align: center;
        text-style: bold;
        padding: 0 1;
    }
    #picker-footer {
        dock: bottom;
        height: 1;
        background: $surface;
        color: $text-muted;
        text-align: center;
        padding: 0 1;
    }
    #picker-layout {
        height: 1fr;
    }
    #list-pane {
        width: 1fr;
        min-width: 24;
        max-width: 40;
    }
    #search-input {
        dock: top;
        margin: 0 0 1 0;
    }
    #template-list {
        border: solid $primary;
        scrollbar-size: 1 1;
    }
    #template-preview {
        width: 3fr;
        border: solid $secondary;
        padding: 1 2;
        scrollbar-size: 1 1;
    }
    #preview-md {
        height: auto;
    }
    """

    BINDINGS = [
        Binding('escape', 'cancel', 'Cancel', priority=True),
        Binding('q', 'cancel', 'Cancel'),
        Binding('enter', 'select_template', 'Select', priority=True),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        loader = YamlTemplateLoader()
        self._user_names = user_template_names()
        self._templates: dict[str, SessionTemplate] = {name: loader.load(name) for name in sorted(all_template_names())}
        self._current_name: str | None = None

    def compose(self) -> ComposeResult:
        count = len(self._templates)
        yield Static(f'  Select a template ({count} available)', id='picker-header')
        with Horizontal(id='picker-layout'):
            with Vertical(id='list-pane'):
                yield Input(placeholder='Filter templates...', id='search-input')
                yield ListView(id='template-list')
            with VerticalScroll(id='template-preview', can_focus=False):
                yield Markdown('', id='preview-md')
        yield Static(
            r'\[Enter] Select  \[↑/↓] Navigate  \[Esc] Cancel',
            id='picker-footer',
        )

    def on_mount(self) -> None:
        self._rebuild_list()
        self.query_one('#search-input', Input).focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        self._rebuild_list(event.value.strip().lower())

    def _rebuild_list(self, query: str = '') -> None:
        """Rebuild the ListView contents, optionally filtered by *query*."""
        list_view = self.query_one('#template-list', ListView)
        list_view.clear()

        # Group templates by locale.
        groups: dict[str, list[str]] = defaultdict(list)
        for name, tmpl in self._templates.items():
            if query and query not in name.lower() and query not in tmpl.metadata.description.lower():
                continue
            groups[tmpl.metadata.locale].append(name)

        first_item: TemplateItem | None = None
        for locale in sorted(groups):
            list_view.append(LocaleHeader(locale))
            for name in sorted(groups[locale]):
                item = TemplateItem(
                    name,
                    locale,
                    is_user=name in self._user_names,
                )
                list_view.append(item)
                if first_item is None:
                    first_item = item

        if first_item is not None:
            list_view.index = list_view.children.index(first_item)
            self._current_name = first_item.template_name
            self._show_preview(first_item.template_name)
        else:
            self._current_name = None
            self.query_one('#preview-md', Markdown).update('*No matching templates*')

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        if isinstance(event.item, TemplateItem):
            self._current_name = event.item.template_name
            self._show_preview(event.item.template_name)

    def _show_preview(self, name: str) -> None:
        tmpl = self._templates[name]
        meta = tmpl.metadata
        source = '\\[user]' if name in self._user_names else '\\[built-in]'
        lines = [
            f'## {meta.name}  {source}',
            '',
            f'> {meta.description}' if meta.description else '',
            '',
            f'**Locale:** `{meta.locale}`',
        ]

        if tmpl.quick_actions:
            lines += ['', '### Quick Actions']
            for qa in tmpl.quick_actions:
                lines.append(f'- **`{qa.key}`** {qa.label} — {qa.description}')

        if tmpl.whisper_prompt:
            lines += ['', f'**Whisper hint:** {tmpl.whisper_prompt}']

        lines += ['', '---', '', '### System Prompt', '']
        lines.append(tmpl.system_prompt)

        self.query_one('#preview-md', Markdown).update('\n'.join(lines))

    def action_select_template(self) -> None:
        self.exit(self._current_name)

    def action_cancel(self) -> None:
        self.exit(None)

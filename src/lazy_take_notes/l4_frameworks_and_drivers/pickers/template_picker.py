"""Template picker — small TUI to select a template before recording."""

from __future__ import annotations

import os
import platform
import subprocess  # noqa: S404 -- used for launching $EDITOR, not shell commands
import sys
from collections import defaultdict

from textual.app import ComposeResult, SuspendNotSupported
from textual.binding import Binding
from textual.containers import Center, Vertical
from textual.events import AppFocus, Key
from textual.screen import ModalScreen
from textual.widgets import Button, Input, ListItem, Markdown, Static

from lazy_take_notes.l1_entities.template import SessionTemplate
from lazy_take_notes.l3_interface_adapters.gateways.paths import BUILTIN_TEMPLATES_NOTICED_PATH
from lazy_take_notes.l3_interface_adapters.gateways.yaml_template_loader import (
    YamlTemplateLoader,
    all_template_names,
    builtin_names,
    delete_user_template,
    ensure_user_copy,
    user_template_names,
)
from lazy_take_notes.l4_frameworks_and_drivers.pickers.base import (
    PickerListView,
    SearchablePicker,
)
from lazy_take_notes.l4_frameworks_and_drivers.widgets.builtin_templates_notice import (
    BuiltinTemplatesNotice,
)


def resolve_editor() -> list[str] | None:
    """Resolve the user's preferred editor command as an argv list.

    Priority: $VISUAL → $EDITOR → platform fallback (open -t / xdg-open / notepad).
    Returns None only if nothing is found (should never happen with platform fallback).
    """
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


class _ConfirmDeleteScreen(ModalScreen[bool]):
    """Modal yes/no dialog for template deletion — overlays the picker."""

    CSS = """
    _ConfirmDeleteScreen {
        align: center middle;
    }
    #confirm-dialog {
        width: 50;
        height: auto;
        border: thick $error;
        background: $surface;
        padding: 1 2;
    }
    #confirm-msg {
        width: 1fr;
        text-align: center;
        margin-bottom: 1;
    }
    #confirm-buttons {
        width: 100%;
        align-horizontal: center;
    }
    #confirm-buttons Button {
        margin: 0 1;
    }
    """

    BINDINGS = [
        Binding('y', 'confirm', 'Yes'),
        Binding('n', 'dismiss_no', 'No'),
        Binding('escape', 'dismiss_no', 'Cancel'),
        Binding('left', 'focus_previous', 'Focus previous', show=False),
        Binding('right', 'focus_next', 'Focus next', show=False),
    ]

    def __init__(self, template_name: str) -> None:
        super().__init__()
        self._template_name = template_name

    def compose(self) -> ComposeResult:
        with Vertical(id='confirm-dialog'):
            yield Static(
                f'Delete user template [bold]"{self._template_name}"[/bold]?',
                id='confirm-msg',
                markup=True,
            )
            with Center(id='confirm-buttons'):
                yield Button('No (n)', id='confirm-no', variant='default')
                yield Button('Yes (y)', id='confirm-yes', variant='error')

    def on_mount(self) -> None:
        self.query_one('#confirm-no', Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == 'confirm-yes')

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_dismiss_no(self) -> None:
        self.dismiss(False)


class LocaleHeader(ListItem):
    """Non-interactive group header showing a locale name (e.g. 'EN')."""

    def __init__(self, locale: str) -> None:
        super().__init__(disabled=True)
        self._locale = locale.upper()

    def compose(self) -> ComposeResult:
        yield Static(f'[bold]{self._locale}[/bold]', markup=True)


class TemplateItem(ListItem):
    """Selectable row representing a single template."""

    def __init__(self, name: str, locale: str, display_name: str = '', *, is_user: bool = False) -> None:
        super().__init__()
        self.template_name = name
        badge = '  [dim]\\[user][/dim]' if is_user else ''
        label = display_name or name
        self._label_text = f'{label}  [dim]({locale})[/dim]{badge}'

    def compose(self) -> ComposeResult:
        yield Static(self._label_text, markup=True)


class _TemplateListView(PickerListView):
    """ListView that pops focus back to #sp-search when ↑ is pressed on the first TemplateItem."""

    _selectable_type = TemplateItem


class TemplatePicker(SearchablePicker[str]):
    CSS = """
    #sp-list-pane { max-width: 40; }
    """

    def __init__(self, *, show_builtins: bool = True, **kwargs):
        super().__init__(**kwargs)
        loader = YamlTemplateLoader()
        self._user_names = user_template_names()
        self._builtin_names = builtin_names()
        self._show_builtins = show_builtins

        all_names = sorted(all_template_names())
        if not show_builtins:
            all_names = [n for n in all_names if n in self._user_names or n not in self._builtin_names]

        self._templates: dict[str, SessionTemplate] = {name: loader.load(name) for name in all_names}
        self._current_name: str | None = None
        # Set after [e] edit — cleared on AppFocus reload so GUI editors
        # (which return from subprocess.run immediately) get a second reload
        # when the user switches back to the terminal.
        self._pending_reload_name: str | None = None

    def on_mount(self) -> None:  # noqa: D102 -- base class has the docstring; Textual dispatches both via MRO
        if not BUILTIN_TEMPLATES_NOTICED_PATH.exists():
            self.push_screen(
                BuiltinTemplatesNotice(on_suppress=self._suppress_builtin_notice),
            )

    def _suppress_builtin_notice(self) -> None:
        """Write flag file so the notice is not shown again."""
        BUILTIN_TEMPLATES_NOTICED_PATH.parent.mkdir(parents=True, exist_ok=True)
        BUILTIN_TEMPLATES_NOTICED_PATH.touch()

    def _make_list_view(self) -> _TemplateListView:
        return _TemplateListView(id='sp-list')

    def _compose_preview(self) -> ComposeResult:
        yield Markdown('', id='sp-preview-md')

    def _header_text(self) -> str:
        return f'  Select a template ({len(self._templates)} available)'

    def _footer_text(self) -> str:
        return r'\[Enter] Select  \[↑/↓] Navigate  \[n] New  \[e] Edit  \[x] Delete  \[Esc] Cancel'

    def _search_placeholder(self) -> str:
        return 'Filter templates...'

    def on_key(self, event: Key) -> None:
        if event.key == 'e' and not isinstance(self.focused, Input):
            self.action_edit_template()
            event.prevent_default()
            return
        if event.key == 'n' and not isinstance(self.focused, Input):
            self.action_new_template()
            event.prevent_default()
            return
        if event.key == 'x' and not isinstance(self.focused, Input):
            self.action_delete_template()
            event.prevent_default()
            return
        super().on_key(event)

    def action_new_template(self) -> None:
        """Exit picker with a sentinel value to launch the template builder."""
        self.exit('__create_template__')

    def _rebuild_list(self, query: str = '') -> None:
        """Rebuild the ListView contents, optionally filtered by *query*."""
        list_view = self.query_one('#sp-list', _TemplateListView)
        list_view.clear()

        # Group templates by locale.
        groups: dict[str, list[str]] = defaultdict(list)
        for name, tmpl in self._templates.items():
            if query and query not in name.lower() and query not in tmpl.metadata.description.lower():
                continue
            groups[tmpl.metadata.locale].append(name)

        first_item: TemplateItem | None = None
        first_item_index: int = 0
        insert_idx: int = 0
        for locale in sorted(groups):
            list_view.append(LocaleHeader(locale))
            insert_idx += 1
            for name in sorted(groups[locale]):
                item = TemplateItem(
                    name,
                    locale,
                    display_name=self._templates[name].metadata.name,
                    is_user=name in self._user_names,
                )
                list_view.append(item)
                if first_item is None:
                    first_item = item
                    first_item_index = insert_idx
                insert_idx += 1

        if first_item is not None:
            list_view.index = first_item_index
            self._current_name = first_item.template_name
            self._show_preview(first_item.template_name)
        else:
            self._current_name = None
            self.query_one('#sp-preview-md', Markdown).update('*No matching templates*')

    def _on_item_highlighted(self, item: ListItem) -> None:
        if isinstance(item, TemplateItem):
            self._current_name = item.template_name
            self._show_preview(item.template_name)

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
            for i, qa in enumerate(tmpl.quick_actions):
                lines.append(f'- **`{i + 1}`** {qa.label} — {qa.description}')

        if tmpl.recognition_hints:
            lines += ['', f'**Recognition hints:** {", ".join(tmpl.recognition_hints)}']

        lines += ['', '---', '', '### System Prompt', '']
        lines.append(tmpl.system_prompt)

        self.query_one('#sp-preview-md', Markdown).update('\n'.join(lines))

    def action_edit_template(self) -> None:
        """Open the highlighted template in $EDITOR (copies built-in to user dir first)."""
        if self._current_name is None:
            return
        editor_argv = resolve_editor()
        if editor_argv is None:
            self.notify('No editor found ($VISUAL / $EDITOR)', severity='error')
            return
        path = ensure_user_copy(self._current_name)
        edit_name = self._current_name
        try:
            with self.suspend():
                subprocess.run([*editor_argv, str(path)], check=False)  # noqa: S603 -- argv built from env/platform, not user shell input
        except SuspendNotSupported:
            self.notify('Cannot open editor in this environment', severity='error')
            return
        # Reload now (covers terminal editors that block until close).
        # For GUI editors that return immediately, AppFocus will re-reload.
        self._pending_reload_name = edit_name
        self._reload_after_edit(edit_name)

    def on_app_focus(self, _event: AppFocus) -> None:
        """Re-reload template when terminal regains focus (handles GUI editors)."""
        if self._pending_reload_name is not None:
            name = self._pending_reload_name
            self._pending_reload_name = None
            self._reload_after_edit(name)

    def action_delete_template(self) -> None:
        """Delete the highlighted user template after confirmation."""
        if self._current_name is None:
            return
        if self._current_name not in self._user_names:
            self.notify('Only user templates can be deleted', severity='warning')
            return
        self.push_screen(
            _ConfirmDeleteScreen(self._current_name),
            callback=self._on_delete_confirmed,
        )

    def _on_delete_confirmed(self, confirmed: bool | None) -> None:
        if not confirmed or self._current_name is None:
            return
        name = self._current_name
        try:
            delete_user_template(name)
        except ValueError:
            self.notify(f'"{name}" is not a user template', severity='error')
            return
        self._user_names = user_template_names()
        # If a built-in with the same name exists, reload it; otherwise drop it entirely
        if name in builtin_names():
            self._templates[name] = YamlTemplateLoader().load(name)
        else:
            del self._templates[name]
        query = self.query_one('#sp-search', Input).value.strip().lower()
        self._rebuild_list(query)
        self.notify(f'Deleted user template "{name}"')

    def _reload_after_edit(self, name: str) -> None:
        """Refresh template data after external edit, rebuild the list."""
        self._user_names = user_template_names()
        loader = YamlTemplateLoader()
        try:
            self._templates[name] = loader.load(name)
        except Exception:  # noqa: BLE001 -- invalid YAML after user edit; surface as warning, don't crash
            self.notify(f'Template "{name}" has invalid YAML — changes ignored', severity='warning')
            return
        query = self.query_one('#sp-search', Input).value.strip().lower()
        self._rebuild_list(query)
        # Restore highlight to the edited template
        list_view = self.query_one('#sp-list', _TemplateListView)
        for idx, child in enumerate(list_view.children):
            if isinstance(child, TemplateItem) and child.template_name == name:
                list_view.index = idx
                break

    def action_select_item(self) -> None:
        if len(self.screen_stack) > 1:
            # Modal is active — don't let priority enter binding leak through.
            # Dismiss the notice if it's showing; leave other modals alone.
            if isinstance(self.screen, BuiltinTemplatesNotice):
                self.screen.dismiss()
            return
        if self._current_name is None:
            return
        self.exit(self._current_name)

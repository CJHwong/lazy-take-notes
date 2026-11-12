"""Reusable form widgets and helpers for the settings screens."""

from __future__ import annotations

import os
import platform
import sys

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Input, Label, Select, Static, Switch


def resolve_editor() -> list[str] | None:
    """Resolve the user's preferred editor command (reuses logic from template_picker)."""
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


class FieldRow(Horizontal):
    """Label on the left, input and optional help on the right."""

    DEFAULT_CSS = """
    FieldRow {
        height: auto;
        margin: 0 0 1 0;
    }
    FieldRow .field-label {
        width: 24;
        height: auto;
        padding: 0;
        text-style: bold;
    }
    FieldRow .field-control {
        width: 1fr;
        height: auto;
    }
    FieldRow Input {
        width: 100%;
    }
    FieldRow .field-help {
        height: auto;
        color: $text-muted;
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
        with Vertical(classes='field-control'):
            yield Input(
                value=self._value,
                id=self._field_id,
                placeholder=self._placeholder,
                password=self._password,
            )
            if self._help_text:
                yield Static(self._help_text, classes='field-help')


class SwitchRow(Horizontal):
    """Label on the left, switch and optional help on the right."""

    DEFAULT_CSS = """
    SwitchRow {
        height: auto;
        margin: 0 0 1 0;
    }
    SwitchRow .field-label {
        width: 24;
        height: auto;
        padding: 0;
        text-style: bold;
    }
    SwitchRow .field-control {
        width: 1fr;
        height: auto;
    }
    SwitchRow .field-help {
        height: auto;
        color: $text-muted;
    }
    SwitchRow Switch {
        margin: 0;
    }
    """

    def __init__(self, label: str, field_id: str, value: bool, *, help_text: str = '') -> None:
        super().__init__()
        self._label = label
        self._field_id = field_id
        self._value = value
        self._help_text = help_text

    def compose(self) -> ComposeResult:
        yield Label(self._label, classes='field-label')
        with Vertical(classes='field-control'):
            yield Switch(value=self._value, id=self._field_id)
            if self._help_text:
                yield Static(self._help_text, classes='field-help')


class SelectRow(Horizontal):
    """Label on the left, select and optional help on the right."""

    DEFAULT_CSS = """
    SelectRow {
        height: auto;
        margin: 0 0 1 0;
    }
    SelectRow .field-label {
        width: 24;
        height: auto;
        padding: 0;
        text-style: bold;
    }
    SelectRow .field-control {
        width: 1fr;
        height: auto;
    }
    SelectRow Select {
        width: 100%;
    }
    SelectRow .field-help {
        height: auto;
        color: $text-muted;
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
        with Vertical(classes='field-control'):
            options = self._options
            value = self._value
            if value not in options:
                options = [*options, value]
            choices = [(opt, opt) for opt in options]
            yield Select(choices, value=value, id=self._field_id, allow_blank=False)
            if self._help_text:
                yield Static(self._help_text, classes='field-help')


def discover_providers(group: str, builtins: tuple[str, ...]) -> list[str]:
    """Return built-in and plugin-registered provider names for an entry point group."""
    from importlib.metadata import entry_points  # noqa: PLC0415 -- deferred: not needed on --help

    plugin_names = [ep.name for ep in entry_points(group=group)]
    return [*builtins, *(name for name in plugin_names if name not in builtins)]


def plugin_manages_models(provider: str, group: str, builtins: tuple[str, ...]) -> bool:
    """Check if a plugin provider manages its own model selection."""
    if provider in builtins:
        return False

    from importlib.metadata import entry_points  # noqa: PLC0415 -- deferred: only on provider change

    for ep in entry_points(group=group):
        if ep.name == provider:
            factory = ep.load()
            return getattr(factory, 'manages_models', False)
    return False


def format_number(value) -> str:
    """Render a config number without a trailing .0 for whole values."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return str(int(number)) if number.is_integer() else str(number)


def to_float(value: str) -> float:
    """Parse a string to float, returning 0.0 on failure."""
    try:
        return float(value.strip())
    except (ValueError, AttributeError):
        return 0.0


def to_int(value: str) -> int:
    """Parse a string to int, returning 0 on failure."""
    try:
        return int(value.strip())
    except (ValueError, AttributeError):
        return 0


def dict_to_inline(mapping: dict) -> str:
    """Convert {'zh': 'breeze-q8'} to 'zh: breeze-q8'."""
    return ', '.join(f'{key}: {value}' for key, value in mapping.items())


def inline_to_dict(text: str) -> dict:
    """Parse 'zh: breeze-q8, en: large-v3' back to a dict."""
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

"""Shared layout and the language model editor for settings sections."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.widget import Widget
from textual.widgets import Button, Input, Select, Static

from lazy_take_notes.l1_entities.model_asset import ModelAsset
from lazy_take_notes.l3_interface_adapters.gateways import model_catalog


class SettingsSection(VerticalScroll):
    """A settings page as one padded vertical column."""

    DEFAULT_CSS = """
    SettingsSection {
        height: 1fr;
        max-width: 120;
        padding: 1 2;
    }
    """

    def __init__(self, raw: dict, **kwargs) -> None:
        super().__init__(**kwargs)
        self._raw = raw


def section_title(text: str) -> Static:
    return Static(text, classes='section-title')


def model_uri(asset: ModelAsset) -> str:
    """Return the canonical ``hf://`` URI for a cataloged asset."""
    return f'hf://{asset.hf_repo}/{asset.entry}'


class LanguageRow(Horizontal):
    """One locale and its model."""

    DEFAULT_CSS = """
    LanguageRow {
        height: 3;
        margin: 0 0 1 0;
    }
    LanguageRow .lang-locale {
        width: 14;
    }
    LanguageRow .lang-model {
        width: 1fr;
    }
    LanguageRow Button {
        margin-left: 1;
    }
    """

    def __init__(self, locale: str, uri: str, catalog: list[ModelAsset], row_id: int) -> None:
        super().__init__()
        self.locale = locale
        asset = model_catalog.asset_for_model(model_catalog.expand_model_alias(uri)) if uri else None
        self.uri = model_uri(asset) if asset is not None else uri
        self._catalog = catalog
        self._row_id = row_id

    def compose(self) -> ComposeResult:
        choices = [(asset.label, model_uri(asset)) for asset in self._catalog]
        values = {value for _, value in choices}
        if self.uri and self.uri not in values:
            choices.append((self.uri, self.uri))
        value = self.uri or (choices[0][1] if choices else '')
        yield Input(value=self.locale, placeholder='zh', classes='lang-locale')
        yield Select(choices, value=value, id=f'lang-model-{self._row_id}', classes='lang-model', allow_blank=False)
        yield Button('Remove', id=f'lang-remove-{self._row_id}')


class LanguageModelsEditor(Vertical):
    """Add and remove per-locale model overrides."""

    DEFAULT_CSS = """
    LanguageModelsEditor {
        height: auto;
    }
    LanguageModelsEditor #cfg-lang-add {
        margin-top: 1;
    }
    """

    def __init__(self, models: dict[str, str], catalog: list[ModelAsset]) -> None:
        super().__init__()
        self._models = dict(models)
        self._catalog = catalog
        self._next_id = 0

    def compose(self) -> ComposeResult:
        for locale, uri in self._models.items():
            yield self._make_row(locale, uri)
        yield Button('Add language', id='cfg-lang-add')

    def _make_row(self, locale: str, uri: str) -> LanguageRow:
        row = LanguageRow(locale, uri, self._catalog, self._next_id)
        self._next_id += 1
        return row

    def add_row(self) -> None:
        self.mount(self._make_row('', ''), before=self.query_one('#cfg-lang-add'))

    def reload(self, models: dict[str, str]) -> None:
        """Replace every row with the values from a reloaded config."""
        for row in list(self.query(LanguageRow)):
            row.remove()
        for locale, uri in models.items():
            self.mount(self._make_row(locale, uri), before=self.query_one('#cfg-lang-add'))

    def values(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for row in self.query(LanguageRow):
            try:
                locale = row.query_one('.lang-locale', Input).value.strip()
                model = str(row.query_one('.lang-model', Select).value)
            except NoMatches:
                locale, model = row.locale, row.uri
            if locale and model:
                result[locale] = model
        return result

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ''
        if button_id == 'cfg-lang-add':
            self.add_row()
        elif button_id.startswith('lang-remove-'):
            row = event.button.parent
            if isinstance(row, Widget):
                row.remove()

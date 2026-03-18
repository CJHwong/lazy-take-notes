"""Shared base class for searchable TUI pickers.

All pickers that display a filterable list alongside a preview pane
(SessionPicker, TemplatePicker, FilePicker) inherit from SearchablePicker.
"""

from __future__ import annotations

from typing import ClassVar, TypeVar

T = TypeVar('T')

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.events import Key
from textual.widgets import Input, ListItem, ListView, Static


class PickerListView(ListView):
    """ListView that pops focus back to #sp-search when ↑ is pressed on the first selectable item."""

    _selectable_type: ClassVar[type[ListItem]]

    def on_key(self, event: Key) -> None:
        if event.key != 'up':
            return
        first_selectable = next(
            (i for i, child in enumerate(self.children) if isinstance(child, self._selectable_type)),
            None,
        )
        if first_selectable is not None and self.index == first_selectable:
            self.app.query_one('#sp-search', Input).focus()
            event.prevent_default()


class SearchablePicker(App[T]):
    """Abstract base for searchable pickers with a filter input and preview pane.

    Standardised widget IDs:
        #sp-header, #sp-footer, #sp-layout, #sp-list-pane,
        #sp-search, #sp-list, #sp-preview

    Subclasses must implement all methods that raise NotImplementedError.
    """

    BINDINGS = [
        Binding('escape', 'cancel', 'Cancel', priority=True),
        Binding('q', 'cancel', 'Cancel'),
        Binding('enter', 'select_item', 'Select', priority=True),
    ]

    DEFAULT_CSS = """
    #sp-header  { dock: top; height: 1; background: $primary; color: $text; text-align: center; text-style: bold; padding: 0 1; }
    #sp-footer  { dock: bottom; height: 1; background: $surface; color: $text-muted; text-align: center; padding: 0 1; }
    #sp-layout  { height: 1fr; }
    #sp-list-pane { width: 1fr; min-width: 24; }
    #sp-search  { dock: top; margin: 0 0 1 0; }
    #sp-list    { border: solid $primary; scrollbar-size: 1 1; }
    #sp-preview { width: 3fr; border: solid $secondary; padding: 1 2; scrollbar-size: 1 1; }
    """

    # ── Methods subclasses must override ──────────────────────────────────────

    def _make_list_view(self) -> PickerListView:
        raise NotImplementedError

    def _compose_preview(self) -> ComposeResult:
        """Yield the widget(s) to place inside the preview VerticalScroll.

        Override this in the subclass to yield a Markdown, Static, etc.
        The base yields nothing so that compose() does not crash on instantiation.
        """
        yield from ()

    def _header_text(self) -> str:
        raise NotImplementedError

    def _footer_text(self) -> str:
        raise NotImplementedError

    def _rebuild_list(self, query: str = '') -> None:
        raise NotImplementedError

    def _on_item_highlighted(self, item: ListItem) -> None:
        raise NotImplementedError

    def action_select_item(self) -> None:
        raise NotImplementedError

    # ── Concrete methods provided by the base ────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Static(self._header_text(), id='sp-header')
        with Horizontal(id='sp-layout'):
            with Vertical(id='sp-list-pane'):
                yield Input(placeholder=self._search_placeholder(), id='sp-search')
                yield self._make_list_view()
            with VerticalScroll(id='sp-preview', can_focus=False):
                yield from self._compose_preview()
        yield Static(self._footer_text(), id='sp-footer', markup=True)

    def on_mount(self) -> None:
        self._rebuild_list()
        self.query_one('#sp-search', Input).focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        self._rebuild_list(event.value.strip().lower())

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        if event.item is not None:
            self._on_item_highlighted(event.item)

    def on_key(self, event: Key) -> None:
        if event.key == 'down' and self.focused is self.query_one('#sp-search', Input):
            self.query_one('#sp-list', PickerListView).focus()
            event.prevent_default()

    def action_cancel(self) -> None:
        self.exit(None)

    def _search_placeholder(self) -> str:
        return 'Filter...'

    def _refresh_header(self) -> None:
        """Dynamically update the header (e.g. when FilePicker navigates directories)."""
        self.query_one('#sp-header', Static).update(self._header_text())

"""Built-in templates notice -- shown in the template picker until dismissed."""

from __future__ import annotations

from collections.abc import Callable

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Markdown, Static

NOTICE_BODY = """\
Built-in templates are **example sessions** for common use cases \
(meetings, standups, lectures, podcasts, etc.).

- Press **[n]** in the picker to **create your own** template.
- To hide built-in templates, set `show_builtin_templates: false` \
in settings or config.yaml.
"""


class BuiltinTemplatesNotice(ModalScreen[None]):
    """Non-blocking notice explaining built-in templates. Shown until user opts out."""

    DEFAULT_CSS = """
    BuiltinTemplatesNotice {
        align: center middle;
    }

    BuiltinTemplatesNotice > VerticalScroll {
        width: 60%;
        max-width: 72;
        height: auto;
        max-height: 60%;
        background: $surface;
        border: thick $accent;
        padding: 1 2;
    }

    BuiltinTemplatesNotice > VerticalScroll > #btn-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }

    BuiltinTemplatesNotice > VerticalScroll > #btn-body {
        height: auto;
    }

    BuiltinTemplatesNotice > VerticalScroll > #btn-hint {
        dock: bottom;
        text-align: center;
        color: $text-muted;
        margin-top: 1;
    }
    """

    BINDINGS = [
        ('escape', 'dismiss', 'Close'),
    ]

    def __init__(
        self,
        *args,
        on_suppress: Callable[[], None] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._on_suppress = on_suppress

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield Static('Built-in Templates', id='btn-title')
            yield Markdown(NOTICE_BODY, id='btn-body')
            yield Static(
                "Press any key to dismiss \u00b7 \\[d] don't show again",
                id='btn-hint',
            )

    def on_key(self, event) -> None:  # noqa: ANN001 -- Textual Key event; type not needed
        if event.key == 'd' and self._on_suppress is not None:
            self._on_suppress()
        self.dismiss()

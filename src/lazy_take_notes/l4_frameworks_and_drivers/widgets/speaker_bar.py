"""Speaker bar — one clickable chip per speaker heard so far."""

from __future__ import annotations

from textual.containers import Horizontal
from textual.widgets import Static


def format_spoken_time(seconds: float) -> str:
    """Readable speech total: 42s, 4m58s, 1h12m."""
    total = int(seconds)
    if total >= 3600:
        return f'{total // 3600}h{(total % 3600) // 60}m'
    if total >= 60:
        return f'{total // 60}m{total % 60}s'
    return f'{total}s'


class SpeakerBar(Horizontal):
    """Shows the session's speakers. A click opens a rename dialog."""

    DEFAULT_CSS = """
    SpeakerBar {
        height: auto;
        padding: 0 1;
    }
    SpeakerBar > .speaker-chip {
        width: auto;
        height: 1;
        margin: 0 1 0 0;
        padding: 0 1;
        background: $surface;
    }
    SpeakerBar > .speaker-chip:hover {
        background: $accent;
    }
    """

    def show(self, speakers: list[tuple[int, str, float]]) -> None:
        """Update the chips in place. Each entry is (speaker number, name, speech seconds).

        Buttons are reused rather than rebuilt: removing and re-adding the same
        widget id in one tick trips Textual's duplicate-id check.
        """
        wanted = {f'speaker-{speaker}': (speaker, name, seconds) for speaker, name, seconds in speakers}
        for child in list(self.children):
            if child.id not in wanted:
                child.remove()
        for speaker, name, seconds in speakers:
            widget_id = f'speaker-{speaker}'
            text = f'{name or f"S{speaker}"} {format_spoken_time(seconds)}'
            existing = next((child for child in self.children if child.id == widget_id), None)
            if isinstance(existing, Static):
                existing.update(text)
            else:
                self.mount(Static(text, id=widget_id, classes='speaker-chip'))

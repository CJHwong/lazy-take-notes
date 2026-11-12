"""Appearance page: theme and a live recording preview."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Rule, Static

from lazy_take_notes.l4_frameworks_and_drivers.apps.config.fields import SelectRow
from lazy_take_notes.l4_frameworks_and_drivers.apps.config.sections import section_title
from lazy_take_notes.l4_frameworks_and_drivers.widgets.digest_panel import DigestPanel
from lazy_take_notes.l4_frameworks_and_drivers.widgets.status_bar import StatusBar
from lazy_take_notes.l4_frameworks_and_drivers.widgets.transcript_panel import TranscriptPanel

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
        height: 1fr;
        margin: 0;
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
        height: 1fr;
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


class AppearanceTab(Vertical):
    """Theme and the recording preview, stacked in one column."""

    DEFAULT_CSS = """
    AppearanceTab {
        height: 1fr;
        padding: 1 2;
    }
    AppearanceTab .theme-row {
        max-width: 70;
        height: auto;
    }
    """

    def __init__(self, infra, **kwargs) -> None:
        super().__init__(**kwargs)
        self._infra = infra

    def compose(self) -> ComposeResult:
        yield section_title('Theme')
        yield Rule(classes='section-rule')
        with Vertical(classes='theme-row'):
            yield SelectRow(
                'Theme',
                'cfg-theme',
                list(self.app.available_themes.keys()),
                self._infra.theme,
            )
        yield RecordingPreview()

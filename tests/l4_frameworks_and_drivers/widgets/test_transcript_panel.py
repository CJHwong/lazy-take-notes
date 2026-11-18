"""TranscriptPanel shows every character of a wrapped line."""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult

from lazy_take_notes.l1_entities.transcript import TranscriptSegment
from lazy_take_notes.l4_frameworks_and_drivers.widgets.transcript_panel import TranscriptPanel

LONG = (
    "Let's move the billing migration to next sprint, because right now? The team is "
    'still on the resize work and we need it done by Thursday.'
)


class Host(App):
    def compose(self) -> ComposeResult:
        yield TranscriptPanel(id='transcript-panel')


class TestWrapping:
    @pytest.mark.asyncio
    async def test_a_narrow_panel_wraps_inside_its_width(self):
        app = Host()
        async with app.run_test(size=(60, 20)) as pilot:
            panel = app.query_one(TranscriptPanel)
            await pilot.pause()
            panel.append_segments([TranscriptSegment(text=LONG, wall_start=1.0, wall_end=2.0)])
            await pilot.pause()

            width = panel.scrollable_content_region.width
            assert all(line.cell_length <= width for line in panel.lines)
            shown = ' '.join(line.text.strip() for line in panel.lines)
            assert ' '.join(LONG.split()) in ' '.join(shown.split())

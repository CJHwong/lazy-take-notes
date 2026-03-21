"""Stable public API surface for lazy-take-notes plugins.

Plugin authors should import from ``lazy_take_notes.plugin_api`` (the
top-level re-export) instead of reaching into internal modules directly.
This insulates plugins from internal refactors.

Example::

    from lazy_take_notes.plugin_api import run_transcribe, TranscriptSegment

    @click.command('my-source')
    @click.argument('input_path')
    @click.pass_context
    def my_command(ctx, input_path):
        segments = fetch_and_parse(input_path)
        run_transcribe(ctx, subtitle_segments=segments, label='my session')
"""

from __future__ import annotations

from lazy_take_notes.l1_entities.transcript import TranscriptSegment
from lazy_take_notes.l4_frameworks_and_drivers.cli_helpers import run_transcribe

__all__ = [
    'run_transcribe',
    'TranscriptSegment',
]

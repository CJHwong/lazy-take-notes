"""Public plugin API — convenience re-export from the top-level package.

Plugin authors import from here::

    from lazy_take_notes.plugin_api import run_transcribe, TranscriptSegment

This is a thin re-export; the real implementation lives in
``l4_frameworks_and_drivers.plugin_api``.
"""

from lazy_take_notes.l4_frameworks_and_drivers.plugin_api import (  # noqa: F401 -- re-export
    TranscriptSegment,
    run_transcribe,
)

__all__ = [
    'run_transcribe',
    'TranscriptSegment',
]

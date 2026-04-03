"""Public plugin API -- convenience re-export from the top-level package.

Plugin authors import from here::

    from lazy_take_notes.plugin_api import run_record, run_transcribe, LLMClient

This is a thin re-export; the real implementation lives in
``l4_frameworks_and_drivers.plugin_api``.
"""

from lazy_take_notes.l4_frameworks_and_drivers.plugin_api import (  # noqa: F401 -- re-export
    AudioSource,
    ChatMessage,
    ChatResponse,
    InfraConfig,
    LLMClient,
    ModelResolver,
    Transcriber,
    TranscriptionBackend,
    TranscriptSegment,
    run_record,
    run_transcribe,
)

__all__ = [
    'AudioSource',
    'ChatMessage',
    'ChatResponse',
    'InfraConfig',
    'LLMClient',
    'ModelResolver',
    'Transcriber',
    'TranscriptionBackend',
    'TranscriptSegment',
    'run_record',
    'run_transcribe',
]

"""Stable public API surface for lazy-take-notes plugins.

Plugin authors should import from ``lazy_take_notes.plugin_api`` (the
top-level re-export) instead of reaching into internal modules directly.
This insulates plugins from internal refactors.

Example::

    from lazy_take_notes.plugin_api import (
        run_record, run_transcribe, LLMClient, TranscriptSegment,
    )

    class MyLLMClient:
        async def chat(self, model, messages): ...
        async def chat_single(self, model, prompt): ...
        def check_connectivity(self): ...
        def check_models(self, models): ...

    @click.command('my-record')
    @click.pass_context
    def my_command(ctx):
        run_record(ctx, llm_client=MyLLMClient())
"""

from __future__ import annotations

from lazy_take_notes.l1_entities.chat_message import ChatMessage
from lazy_take_notes.l1_entities.transcript import TranscriptSegment
from lazy_take_notes.l2_use_cases.ports.audio_source import AudioSource
from lazy_take_notes.l2_use_cases.ports.llm_client import ChatResponse, LLMClient
from lazy_take_notes.l2_use_cases.ports.model_resolver import ModelResolver
from lazy_take_notes.l2_use_cases.ports.transcriber import Transcriber
from lazy_take_notes.l4_frameworks_and_drivers.cli_helpers import run_record, run_transcribe
from lazy_take_notes.l4_frameworks_and_drivers.config import InfraConfig
from lazy_take_notes.l4_frameworks_and_drivers.container import TranscriptionBackend

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

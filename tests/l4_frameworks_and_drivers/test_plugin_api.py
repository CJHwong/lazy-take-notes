"""Tests for the public plugin API surface."""

from __future__ import annotations

import inspect


class TestPluginApiExports:
    """All expected symbols are importable from lazy_take_notes.plugin_api."""

    def test_run_transcribe_importable(self):
        from lazy_take_notes.plugin_api import run_transcribe

        assert callable(run_transcribe)

    def test_run_record_importable(self):
        from lazy_take_notes.plugin_api import run_record

        assert callable(run_record)

    def test_transcript_segment_importable(self):
        from lazy_take_notes.plugin_api import TranscriptSegment

        assert TranscriptSegment is not None

    def test_llm_client_protocol_importable(self):
        from lazy_take_notes.plugin_api import LLMClient

        assert LLMClient is not None

    def test_transcriber_protocol_importable(self):
        from lazy_take_notes.plugin_api import Transcriber

        assert Transcriber is not None

    def test_audio_source_protocol_importable(self):
        from lazy_take_notes.plugin_api import AudioSource

        assert AudioSource is not None

    def test_chat_response_importable(self):
        from lazy_take_notes.plugin_api import ChatResponse

        assert ChatResponse is not None

    def test_chat_message_importable(self):
        from lazy_take_notes.plugin_api import ChatMessage

        assert ChatMessage is not None


class TestPluginApiSignatures:
    """Override kwargs are accepted by run_transcribe and run_record."""

    def test_run_transcribe_accepts_llm_client_kwarg(self):
        from lazy_take_notes.plugin_api import run_transcribe

        sig = inspect.signature(run_transcribe)
        assert 'llm_client' in sig.parameters
        assert 'transcriber' in sig.parameters

    def test_run_record_accepts_override_kwargs(self):
        from lazy_take_notes.plugin_api import run_record

        sig = inspect.signature(run_record)
        assert 'llm_client' in sig.parameters
        assert 'transcriber' in sig.parameters
        assert 'audio_source' in sig.parameters

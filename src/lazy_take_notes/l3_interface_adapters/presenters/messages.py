"""Textual Message subclasses — contracts between controller/workers and the App."""

from __future__ import annotations

from textual.message import Message

from lazy_take_notes.l1_entities.transcript import TranscriptSegment


class TranscriptChunk(Message):
    """Posted by audio worker when new transcript segments are available."""

    def __init__(self, segments: list[TranscriptSegment]) -> None:
        super().__init__()
        self.segments = segments


class AudioWorkerStatus(Message):
    """Posted by audio worker for status updates (model loaded, error, etc.)."""

    def __init__(self, status: str, error: str = '') -> None:
        super().__init__()
        self.status = status
        self.error = error


class DigestReady(Message):
    """Posted by controller when a digest cycle completes successfully."""

    def __init__(self, markdown: str, digest_number: int, is_final: bool = False) -> None:
        super().__init__()
        self.markdown = markdown
        self.digest_number = digest_number
        self.is_final = is_final


class DigestError(Message):
    """Posted by controller when a digest cycle fails."""

    def __init__(self, error: str, consecutive_failures: int) -> None:
        super().__init__()
        self.error = error
        self.consecutive_failures = consecutive_failures


class ModelDownloadProgress(Message):
    """Posted during model download to report progress percentage."""

    def __init__(self, percent: int, model_name: str) -> None:
        super().__init__()
        self.percent = percent
        self.model_name = model_name


class QueryResult(Message):
    """Posted by controller when a quick-action query completes."""

    def __init__(self, result: str, action_label: str) -> None:
        super().__init__()
        self.result = result
        self.action_label = action_label


class AudioLevel(Message):
    """Posted by audio worker with current mic RMS level (~10 Hz during recording)."""

    def __init__(self, rms: float) -> None:
        super().__init__()
        self.rms = rms

"""Configuration Pydantic models — pure schema, no infrastructure defaults."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TranscriptionConfig(BaseModel):
    model: str
    models: dict[str, str] = Field(default_factory=dict)
    input_device: str | None = None
    chunk_duration: float
    overlap: float
    silence_threshold: float
    pause_duration: float

    def model_for_locale(self, locale: str) -> str:
        """Resolve model name for a locale. Checks full key, then primary subtag, then default."""
        key = locale.lower()
        if key in self.models:
            return self.models[key]
        prefix = key.split('-')[0]
        if prefix in self.models:
            return self.models[prefix]
        return self.model


class DigestConfig(BaseModel):
    model: str
    min_lines: int
    min_interval: float
    compact_token_threshold: int
    max_lines: int | None = None  # force-trigger threshold; None = 2×min_lines


class InteractiveConfig(BaseModel):
    model: str


class OutputConfig(BaseModel):
    directory: str
    save_audio: bool
    save_notes_history: bool
    save_context: bool
    save_debug_log: bool
    auto_label: bool


class DiarizationConfig(BaseModel):
    enabled: bool = False
    binary: str = ''
    model: str = ''
    device: str = 'auto'
    interval_s: float = 60.0


class AppConfig(BaseModel):
    transcription: TranscriptionConfig
    digest: DigestConfig
    interactive: InteractiveConfig
    output: OutputConfig
    diarization: DiarizationConfig = Field(default_factory=DiarizationConfig)
    recognition_hints: list[str] = Field(default_factory=list)

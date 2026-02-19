"""Gateway: file-based persistence — implements PersistenceGateway port."""

from __future__ import annotations

from pathlib import Path

from lazy_take_notes.l1_entities.transcript import TranscriptSegment, format_wall_time


class FilePersistenceGateway:
    """Persists transcripts, digests, and history to the filesystem."""

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    def save_transcript_lines(self, segments: list[TranscriptSegment], *, append: bool = True) -> Path:
        path = self._output_dir / 'transcript_raw.txt'
        lines = [f'[{format_wall_time(seg.wall_start)}] {seg.text}' for seg in segments]
        mode = 'a' if append else 'w'
        with path.open(mode, encoding='utf-8') as f:
            for line in lines:
                f.write(line + '\n')
        return path

    def save_digest_md(self, markdown: str, digest_number: int) -> Path:
        content = f'# Digest #{digest_number}\n\n{markdown}\n'
        path = self._output_dir / 'digest.md'
        path.write_text(content, encoding='utf-8')
        return path

    def save_history(self, markdown: str, digest_number: int, *, is_final: bool = False) -> Path:
        history_dir = self._output_dir / 'history'
        history_dir.mkdir(parents=True, exist_ok=True)
        suffix = '_final' if is_final else ''
        path = history_dir / f'digest_{digest_number:03d}{suffix}.md'
        content = f'# Digest #{digest_number}\n\n{markdown}\n'
        path.write_text(content, encoding='utf-8')
        return path

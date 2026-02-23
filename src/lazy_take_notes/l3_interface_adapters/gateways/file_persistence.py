"""Gateway: file-based persistence — implements PersistenceGateway port."""

from __future__ import annotations

import logging
from pathlib import Path

from lazy_take_notes.l1_entities.transcript import TranscriptSegment, format_wall_time

log = logging.getLogger('ltn.persist')


class FilePersistenceGateway:
    """Persists transcripts, digests, and history to the filesystem."""

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    def relocate(self, new_dir: Path) -> None:
        """Point subsequent writes at *new_dir* (caller already renamed the directory)."""
        self._output_dir = new_dir

    def save_transcript_lines(self, segments: list[TranscriptSegment], *, append: bool = True) -> Path:
        path = self._output_dir / 'transcript_raw.txt'
        lines = [f'[{format_wall_time(seg.wall_start)}] {seg.text}' for seg in segments]
        mode = 'a' if append else 'w'
        with path.open(mode, encoding='utf-8') as f:
            for line in lines:
                f.write(line + '\n')
        last_ts = format_wall_time(segments[-1].wall_start) if segments else '?'
        log.debug(
            'Wrote %d segments to %s (last_ts=%s, mode=%s)',
            len(segments),
            path.name,
            last_ts,
            mode,
        )
        return path

    def save_digest_md(self, markdown: str, digest_number: int) -> Path:
        content = f'# Digest #{digest_number}\n\n{markdown}\n'
        path = self._output_dir / 'digest.md'
        path.write_text(content, encoding='utf-8')
        return path

    def save_session_context(self, context: str) -> Path:
        path = self._output_dir / 'session_context.txt'
        path.write_text(context, encoding='utf-8')
        return path

    def save_history(self, markdown: str, digest_number: int, *, is_final: bool = False) -> Path:
        history_dir = self._output_dir / 'history'
        history_dir.mkdir(parents=True, exist_ok=True)
        suffix = '_final' if is_final else ''
        path = history_dir / f'digest_{digest_number:03d}{suffix}.md'
        content = f'# Digest #{digest_number}\n\n{markdown}\n'
        path.write_text(content, encoding='utf-8')
        return path

"""Digest page: summary triggers and token handling."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.widgets import Rule, Static

from lazy_take_notes.l4_frameworks_and_drivers.apps.config.fields import FieldRow, format_number
from lazy_take_notes.l4_frameworks_and_drivers.apps.config.sections import SettingsSection, section_title


class DigestTab(SettingsSection):
    """Summary frequency and advanced token handling."""

    def compose(self) -> ComposeResult:
        digest = self._raw.get('digest', {})

        yield section_title('When to summarize')
        yield Rule(classes='section-rule')
        yield FieldRow(
            'After lines',
            'cfg-digest-min-lines',
            str(digest.get('min_lines', '')),
            placeholder='15',
        )
        yield FieldRow(
            'Wait seconds',
            'cfg-digest-min-interval',
            format_number(digest.get('min_interval', '')),
            placeholder='60',
        )
        yield FieldRow(
            'Force after',
            'cfg-digest-max-lines',
            '' if digest.get('max_lines') is None else str(digest.get('max_lines')),
            placeholder='30',
        )
        yield Static('Force after is optional. Blank means twice the wait threshold.', classes='section-desc')

        yield section_title('Advanced')
        yield Rule(classes='section-rule')
        yield FieldRow(
            'Compaction threshold',
            'cfg-digest-compact',
            str(digest.get('compact_token_threshold', '')),
            placeholder='100000',
        )
        yield Static(
            'Compress history when it exceeds this token count. Change it only if you hit context-length errors.',
            classes='section-desc',
        )

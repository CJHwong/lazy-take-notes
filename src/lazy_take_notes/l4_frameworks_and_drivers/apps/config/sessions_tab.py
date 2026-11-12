"""Sessions page: storage, save options, vocabulary, and templates."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.widgets import Rule, Static, TextArea

from lazy_take_notes.l4_frameworks_and_drivers.apps.config.fields import FieldRow, SwitchRow
from lazy_take_notes.l4_frameworks_and_drivers.apps.config.sections import SettingsSection, section_title


class SessionsTab(SettingsSection):
    """Where sessions go and what they keep."""

    def __init__(self, raw: dict, infra, **kwargs) -> None:
        super().__init__(raw, **kwargs)
        self._infra = infra

    def compose(self) -> ComposeResult:
        output = self._raw.get('output', {})

        yield section_title('Session storage')
        yield Rule(classes='section-rule')
        yield FieldRow(
            'Output folder',
            'cfg-output-dir',
            str(output.get('directory', '')),
            placeholder='./output',
        )
        yield SwitchRow('Save audio', 'cfg-output-save-audio', output.get('save_audio', True))
        yield SwitchRow(
            'Keep notes history',
            'cfg-output-save-notes-history',
            output.get('save_notes_history', True),
        )
        yield SwitchRow('Save session context', 'cfg-output-save-context', output.get('save_context', True))
        yield SwitchRow('Save debug log', 'cfg-output-save-debug-log', output.get('save_debug_log', False))
        yield SwitchRow('Auto-label sessions', 'cfg-output-auto-label', output.get('auto_label', True))

        yield section_title('Custom vocabulary')
        yield Rule(classes='section-rule')
        yield Static(
            'Names, acronyms, or jargon the recognizer should watch for. One per line.',
            classes='section-desc',
        )
        yield TextArea('\n'.join(self._raw.get('recognition_hints', [])), id='cfg-recognition-hints')

        yield section_title('Templates')
        yield Rule(classes='section-rule')
        yield SwitchRow(
            'Show built-in templates',
            'cfg-show-builtin-templates',
            self._infra.show_builtin_templates,
        )
        yield Static(
            'Bundled examples appear in the picker. Turn it off to see only your own.',
            classes='section-desc',
        )

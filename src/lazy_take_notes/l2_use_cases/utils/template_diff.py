"""List what an edit changed in a template.

The LLM returns the whole template on every edit. This list shows the user each
field that changed, including any change they did not ask for.
"""

from __future__ import annotations

from lazy_take_notes.l1_entities.template import QuickAction, SessionTemplate

_TEXT_FIELDS = (
    ('name', lambda t: t.metadata.name),
    ('description', lambda t: t.metadata.description),
    ('locale', lambda t: t.metadata.locale),
    ('system prompt', lambda t: t.system_prompt),
    ('digest prompt', lambda t: t.digest_user_template),
    ('final summary prompt', lambda t: t.final_user_template),
)


def diff_templates(old: SessionTemplate, new: SessionTemplate) -> list[str]:
    """Name each changed field. The metadata key is set by the loader, so it is ignored."""
    changes = [label for label, read in _TEXT_FIELDS if read(old) != read(new)]
    hints = _hint_change(old.recognition_hints, new.recognition_hints)
    if hints:
        changes.append(hints)
    changes.extend(_quick_action_changes(old.quick_actions, new.quick_actions))
    return changes


def _hint_change(old: list[str], new: list[str]) -> str:
    added = [f'+{hint!r}' for hint in new if hint not in old]
    removed = [f'-{hint!r}' for hint in old if hint not in new]
    if not added and not removed:
        return ''
    return f'recognition hints ({", ".join(added + removed)})'


def _quick_action_changes(old: list[QuickAction], new: list[QuickAction]) -> list[str]:
    old_by_label = {action.label: action for action in old}
    new_by_label = {action.label: action for action in new}
    changes = [f'quick action {label!r} removed' for label in old_by_label if label not in new_by_label]
    for action in new:
        previous = old_by_label.get(action.label)
        if previous is None:
            changes.append(f'quick action {action.label!r} added')
        elif previous != action:
            changes.append(f'quick action {action.label!r} changed')
    kept_old = [label for label in old_by_label if label in new_by_label]
    kept_new = [label for label in new_by_label if label in old_by_label]
    if kept_old != kept_new:
        changes.append('quick action order (keys 1-5 follow it)')
    return changes

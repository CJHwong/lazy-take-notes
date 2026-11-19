//! Names each field an edit changed. The LLM returns the whole template on every
//! edit, so this shows the user every change, including one they did not ask for.

use super::pyformat::py_repr;
use crate::domain::template::{QuickAction, SessionTemplate};

type Read = fn(&SessionTemplate) -> &str;

const TEXT_FIELDS: [(&str, Read); 6] = [
    ("name", |t| &t.metadata.name),
    ("description", |t| &t.metadata.description),
    ("locale", |t| &t.metadata.locale),
    ("system prompt", |t| &t.system_prompt),
    ("digest prompt", |t| &t.digest_user_template),
    ("final summary prompt", |t| &t.final_user_template),
];

/// The metadata key comes from the loader, so a change to it is not listed.
pub fn diff_templates(old: &SessionTemplate, new: &SessionTemplate) -> Vec<String> {
    let mut changes: Vec<String> = TEXT_FIELDS
        .iter()
        .filter(|(_, read)| read(old) != read(new))
        .map(|(label, _)| label.to_string())
        .collect();
    if let Some(hints) = hint_change(&old.recognition_hints, &new.recognition_hints) {
        changes.push(hints);
    }
    changes.extend(quick_action_changes(&old.quick_actions, &new.quick_actions));
    changes
}

fn hint_change(old: &[String], new: &[String]) -> Option<String> {
    let added = new
        .iter()
        .filter(|hint| !old.contains(hint))
        .map(|hint| format!("+{}", py_repr(hint)));
    let removed = old
        .iter()
        .filter(|hint| !new.contains(hint))
        .map(|hint| format!("-{}", py_repr(hint)));
    let all: Vec<String> = added.chain(removed).collect();
    (!all.is_empty()).then(|| format!("recognition hints ({})", all.join(", ")))
}

fn quick_action_changes(old: &[QuickAction], new: &[QuickAction]) -> Vec<String> {
    let old_labels = unique_labels(old);
    let new_labels = unique_labels(new);
    let find = |actions: &[QuickAction], label: &str| {
        actions.iter().rev().find(|a| a.label == label).cloned()
    };
    let mut changes: Vec<String> = old_labels
        .iter()
        .filter(|label| !new_labels.contains(label))
        .map(|label| format!("quick action {} removed", py_repr(label)))
        .collect();
    for action in new {
        match find(old, &action.label) {
            None => changes.push(format!("quick action {} added", py_repr(&action.label))),
            Some(previous) if previous != *action => {
                changes.push(format!("quick action {} changed", py_repr(&action.label)));
            }
            Some(_) => {}
        }
    }
    let kept_old: Vec<&String> = old_labels
        .iter()
        .filter(|label| new_labels.contains(label))
        .collect();
    let kept_new: Vec<&String> = new_labels
        .iter()
        .filter(|label| old_labels.contains(label))
        .collect();
    if kept_old != kept_new {
        changes.push("quick action order (keys 1-5 follow it)".into());
    }
    changes
}

/// Labels in first-seen order, like the keys of a Python dict built from the list.
fn unique_labels(actions: &[QuickAction]) -> Vec<String> {
    let mut labels: Vec<String> = Vec::new();
    for action in actions {
        if !labels.contains(&action.label) {
            labels.push(action.label.clone());
        }
    }
    labels
}

//! Each Rust key table against the Python one in tests/fixtures/bindings.json.

use lazy_take_notes::tui::keys::Binding;
use lazy_take_notes::tui::screens::picker::PICKER_BINDINGS;
use lazy_take_notes::tui::screens::view::{
    DIGEST_PANEL_BINDINGS, TRANSCRIPT_PANEL_BINDINGS, VIEW_BINDINGS,
};
use serde_json::Value;

fn python_table(class: &str) -> Vec<(String, String, String, bool, bool)> {
    let all: Value = serde_json::from_str(include_str!("fixtures/bindings.json")).unwrap();
    let rows = all[class]
        .as_array()
        .unwrap_or_else(|| panic!("no Python table for {class}"));
    rows.iter()
        .map(|row| {
            (
                row["key"].as_str().unwrap().to_string(),
                row["action"].as_str().unwrap().to_string(),
                row["description"].as_str().unwrap().to_string(),
                row["show"].as_bool().unwrap(),
                row["priority"].as_bool().unwrap(),
            )
        })
        .collect()
}

fn assert_table(class: &str, rust: &[Binding]) {
    let rust: Vec<_> = rust
        .iter()
        .map(|b| {
            (
                b.key.to_string(),
                b.action.to_string(),
                b.description.to_string(),
                b.show,
                b.priority,
            )
        })
        .collect();
    assert_eq!(rust, python_table(class), "{class}");
}

#[test]
fn view_screen_and_panels() {
    assert_table("apps.view.ViewApp", &VIEW_BINDINGS);
    assert_table(
        "widgets.transcript_panel.TranscriptPanel",
        &TRANSCRIPT_PANEL_BINDINGS,
    );
    assert_table("widgets.digest_panel.DigestPanel", &DIGEST_PANEL_BINDINGS);
}

#[test]
fn searchable_pickers() {
    assert_table("pickers.base.SearchablePicker", &PICKER_BINDINGS);
}

#[test]
fn file_and_template_pickers_and_their_modals() {
    use lazy_take_notes::tui::screens::file_picker::FILE_PICKER_BINDINGS;
    use lazy_take_notes::tui::screens::modals::NOTICE_BINDINGS;
    use lazy_take_notes::tui::screens::template_picker::CONFIRM_DELETE_BINDINGS;
    assert_table("pickers.file_picker.FilePicker", &FILE_PICKER_BINDINGS);
    assert_table(
        "pickers.template_picker._ConfirmDeleteScreen",
        &CONFIRM_DELETE_BINDINGS,
    );
    assert_table(
        "widgets.builtin_templates_notice.BuiltinTemplatesNotice",
        &NOTICE_BINDINGS,
    );
}

#[test]
fn transcribe_screen_and_its_modals() {
    use lazy_take_notes::tui::screens::modals::{HELP_BINDINGS, LABEL_BINDINGS, QUERY_BINDINGS};
    use lazy_take_notes::tui::screens::transcribe::TRANSCRIBE_BINDINGS;
    assert_table("apps.transcribe.TranscribeApp", &TRANSCRIBE_BINDINGS);
    assert_table("widgets.help_modal.HelpModal", &HELP_BINDINGS);
    assert_table("widgets.query_modal.QueryModal", &QUERY_BINDINGS);
    assert_table("widgets.label_modal.LabelModal", &LABEL_BINDINGS);
}

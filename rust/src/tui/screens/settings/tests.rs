//! The settings form against Python's: the data at open and the file after Save.
//! `scripts/gen_settings_fixtures.py` records the Python side.

use serde_json::Value as Json;

use super::models::LanguageEditor;
use super::*;
use crate::tui::host::Host;

const FIXTURES: &str = include_str!("../../../../tests/fixtures/settings.json");
/// The fixture script creates this file; the custom path case needs it to exist.
const CUSTOM_MODEL: &str = "/tmp/ltn-settings-fixture/model.bin";

fn to_json(value: &Value) -> Json {
    serde_json::to_value(value).expect("YAML data converts to JSON")
}

fn host() -> (Host, tokio::runtime::Runtime) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let host = Host::new("textual-dark", runtime.handle().clone());
    (host, runtime)
}

fn screen_for(root: &Path, config: Option<&str>) -> SettingsScreen {
    let paths = Paths::under(root);
    if let Some(text) = config {
        std::fs::create_dir_all(&paths.config_dir).unwrap();
        std::fs::write(paths.config_dir.join("config.yaml"), text).unwrap();
    }
    let store = Arc::new(CachedModelStore::new(&paths));
    SettingsScreen::with(paths, None, store, &[])
}

fn apply(screen: &mut SettingsScreen, edit: &[Json], ctx: &mut Ctx) {
    let kind = edit[0].as_str().unwrap();
    let id = edit[1].as_str().unwrap();
    let page = screen
        .pages
        .iter()
        .position(|page| page.item(id).is_some() || page.items.iter().any(|item| matches!(&item.kind, Kind::Buttons { buttons, .. } if buttons.iter().any(|b| b.id == id))))
        .unwrap_or(0);
    let page = &mut screen.pages[page];
    match kind {
        "input" => page.set_input(id, edit[2].as_str().unwrap()),
        "switch" => page.set_switch(id, edit[2].as_bool().unwrap()),
        "select" => page
            .select_mut(id)
            .unwrap()
            .set_value(edit[2].as_str().unwrap()),
        "textarea" => page
            .text_area_mut(id)
            .unwrap()
            .set_text(edit[2].as_str().unwrap()),
        "press" => {
            screen.pressed(id, ctx);
        }
        "remove_language" => screen.pages[0]
            .custom::<LanguageEditor>("models-language")
            .unwrap()
            .remove(id),
        other => panic!("unknown edit {other}"),
    }
}

#[test]
fn the_form_reads_and_saves_config_yaml_as_python_does() {
    std::fs::create_dir_all(Path::new(CUSTOM_MODEL).parent().unwrap()).unwrap();
    std::fs::write(CUSTOM_MODEL, b"").unwrap();
    let cases: Vec<Json> = serde_json::from_str(FIXTURES).unwrap();
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let root = tempfile::tempdir().unwrap();
        let (mut host, _runtime) = host();
        let ctx = host.ctx();
        let mut screen = screen_for(root.path(), case["config"].as_str());
        screen.mounted(ctx);
        let opened = to_json(&screen.collect());
        assert_eq!(opened, case["opened"], "{name}: the data at open");
        for edit in case["edits"].as_array().unwrap() {
            apply(&mut screen, edit.as_array().unwrap(), ctx);
        }
        assert_eq!(
            screen.has_unsaved_changes(),
            case["unsaved_before_save"].as_bool().unwrap(),
            "{name}: unsaved changes"
        );
        screen.save(ctx);
        let path = root.path().join("config").join("config.yaml");
        let saved = std::fs::read_to_string(&path).ok();
        match case["saved_file"].as_str() {
            None => assert!(
                saved.is_none() || case["config"].is_string(),
                "{name}: nothing is saved"
            ),
            Some(expected) if name == "unreadable_yaml" => {
                // Save refuses a file it cannot read, so the file stays as it was.
                assert_eq!(
                    saved.as_deref(),
                    Some(expected),
                    "{name}: the file is untouched"
                );
            }
            Some(expected) => {
                let saved = saved.unwrap_or_else(|| panic!("{name}: no file saved"));
                let ours: Value = yaml_serde::from_str(&saved).unwrap();
                let theirs: Value = yaml_serde::from_str(expected).unwrap();
                assert_eq!(to_json(&ours), to_json(&theirs), "{name}: the saved file");
                // The key order too: Python writes the form's order.
                let keys = |value: &Value| -> Vec<String> {
                    value
                        .as_mapping()
                        .unwrap()
                        .keys()
                        .map(|k| k.as_str().unwrap().to_string())
                        .collect()
                };
                assert_eq!(keys(&ours), keys(&theirs), "{name}: the key order");
            }
        }
        // The toasts, with the temporary path out of the way.
        let ours: Vec<String> = ctx
            .toast_texts()
            .into_iter()
            .map(|text| text.replace(&path.display().to_string(), "<path>"))
            .collect();
        let theirs: Vec<String> = case["toasts"]
            .as_array()
            .unwrap()
            .iter()
            .map(|text| {
                let text = text.as_str().unwrap();
                match text.strip_prefix("Saved to ") {
                    Some(_) => "Saved to <path>".to_string(),
                    None => text.to_string(),
                }
            })
            .collect();
        if name == "unreadable_yaml" {
            // The parser's own words differ; the sentences around them do not.
            assert!(
                ours[0].starts_with("config.yaml cannot be read ("),
                "{ours:?}"
            );
            assert!(ours[0].ends_with("). Showing defaults. Press [e] to fix the file."));
            assert!(ours[1].starts_with("Cannot save config: "), "{ours:?}");
        } else {
            assert_eq!(ours, theirs, "{name}: the toasts");
        }
    }
}

//! The view over a config in a temp folder: it never reads or writes the
//! user's config, and it opens no audio device.

use std::path::Path;
use std::sync::Arc;

use gpui_kit::{
    AnyWindowHandle, AppContext as _, Context, Entity, Focusable as _, TestAppContext, Window,
    WindowOptions,
};

use super::SettingsView;
use crate::adapters::assets::store::CachedModelStore;
use crate::adapters::paths::Paths;

const CONFIG: &str = "llm_provider: ollama\ndigest:\n  model: llama3\ntheme: nord\n";

struct Open {
    window: AnyWindowHandle,
    view: Entity<SettingsView>,
}

impl Open {
    fn run<R>(
        &self,
        cx: &mut TestAppContext,
        f: impl FnOnce(&mut SettingsView, &mut Window, &mut Context<SettingsView>) -> R,
    ) -> R {
        cx.update_window(self.window, |_, window, cx| {
            self.view.update(cx, |view, cx| f(view, window, cx))
        })
        .unwrap()
    }

    fn set(&self, id: &'static str, text: &str, cx: &mut TestAppContext) {
        let text = text.to_string();
        self.run(cx, move |view, window, cx| {
            let input = view.inputs[id].clone();
            input.update(cx, |input, cx| input.set_value(text, window, cx));
        });
    }
}

fn open(root: &Path, cx: &mut TestAppContext) -> Open {
    open_with(root, CONFIG, cx)
}

fn open_with(root: &Path, text: &str, cx: &mut TestAppContext) -> Open {
    cx.update(gpui_kit::init);
    let paths = Paths::under(root);
    std::fs::create_dir_all(&paths.config_dir).unwrap();
    let config = paths.config_dir.join("config.yaml");
    std::fs::write(&config, text).unwrap();
    let store = Arc::new(CachedModelStore::new(&paths));
    let (window, view) = cx.update(|cx| {
        gpui_kit::open_window(WindowOptions::default(), cx, |window, cx| {
            cx.new(|cx| {
                SettingsView::with(
                    paths,
                    Some(config),
                    store,
                    vec!["Desk Mic".into()],
                    window,
                    cx,
                )
            })
        })
        .unwrap()
    });
    Open { window, view }
}

#[gpui_kit::test]
fn the_form_opens_with_the_file_and_no_changes(cx: &mut TestAppContext) {
    let root = tempfile::tempdir().unwrap();
    let form = open(root.path(), cx);
    form.run(cx, |view, _, cx| {
        assert!(!view.has_unsaved_changes(cx));
        let data = view.runtime.collect(&mut view.snapshot(cx));
        assert_eq!(data["llm_provider"].as_str(), Some("ollama"));
        assert_eq!(data["theme"].as_str(), Some("nord"));
        assert_eq!(data["digest"]["model"].as_str(), Some("llama3"));
    });
}

#[gpui_kit::test]
fn save_writes_the_temp_file_and_revert_reads_it_back(cx: &mut TestAppContext) {
    let root = tempfile::tempdir().unwrap();
    let form = open(root.path(), cx);
    form.set("cfg-digest-model", "qwen3", cx);
    form.run(cx, |view, _, cx| assert!(view.has_unsaved_changes(cx)));
    form.run(cx, |view, window, cx| view.save(window, cx));
    let config = Paths::under(root.path()).config_dir.join("config.yaml");
    let text = std::fs::read_to_string(&config).unwrap();
    assert!(text.contains("qwen3"), "{text}");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        let mode = std::fs::metadata(&config).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600);
    }
    form.run(cx, |view, _, cx| assert!(!view.has_unsaved_changes(cx)));
    form.set("cfg-digest-model", "other", cx);
    form.run(cx, |view, _, cx| assert!(view.has_unsaved_changes(cx)));
    form.run(cx, |view, window, cx| view.revert(window, cx));
    form.run(cx, |view, _, cx| {
        assert!(!view.has_unsaved_changes(cx));
        assert_eq!(view.inputs["cfg-digest-model"].read(cx).value(), "qwen3");
    });
}

#[gpui_kit::test]
fn marking_a_model_row_clears_the_custom_path(cx: &mut TestAppContext) {
    let root = tempfile::tempdir().unwrap();
    let form = open(root.path(), cx);
    form.set("cfg-trans-custom-path", "/models/own.bin", cx);
    form.run(cx, |view, window, cx| {
        // A row other than the config's own, so the choice is a change.
        let id = view
            .assets
            .iter()
            .map(|asset| asset.id)
            .find(|id| Some(*id) != view.loaded_choice)
            .unwrap();
        view.choose_model(id, window, cx);
        assert_eq!(view.inputs["cfg-trans-custom-path"].read(cx).value(), "");
        assert!(!view.snapshot(cx).chosen.is_empty());
    });
}

#[gpui_kit::test]
fn esc_in_a_field_goes_back_to_the_pane_and_esc_there_leaves(cx: &mut TestAppContext) {
    let root = tempfile::tempdir().unwrap();
    let form = open(root.path(), cx);
    let closed = std::rc::Rc::new(std::cell::Cell::new(0));
    let count = closed.clone();
    let _watch = cx.update(|cx| {
        cx.subscribe(&form.view, move |_, event: &super::SettingsEvent, _| {
            if let super::SettingsEvent::Closed = event {
                count.set(count.get() + 1);
            }
        })
    });
    form.run(cx, |view, window, cx| {
        let focus = view.inputs["cfg-digest-model"].read(cx).focus_handle(cx);
        focus.focus(window, cx);
    });
    cx.run_until_parked();
    cx.simulate_keystrokes(form.window, "escape");
    assert_eq!(closed.get(), 0);
    form.run(cx, |view, window, _| assert!(view.focus.is_focused(window)));
    cx.simulate_keystrokes(form.window, "escape");
    assert_eq!(closed.get(), 1);
}

#[gpui_kit::test]
fn a_short_model_name_marks_its_row_and_saves_unchanged(cx: &mut TestAppContext) {
    let root = tempfile::tempdir().unwrap();
    let text = format!("{CONFIG}transcription:\n  model: large-v3-turbo-q8_0\n");
    let form = open_with(root.path(), &text, cx);
    form.run(cx, |view, _, cx| {
        assert!(view.chosen.is_some(), "the short name marks its row");
        assert!(!view.has_unsaved_changes(cx));
    });
    form.set("cfg-digest-model", "qwen3", cx);
    form.run(cx, |view, window, cx| view.save(window, cx));
    let config = Paths::under(root.path()).config_dir.join("config.yaml");
    let saved = std::fs::read_to_string(&config).unwrap();
    assert!(saved.contains("model: large-v3-turbo-q8_0"), "{saved}");
}

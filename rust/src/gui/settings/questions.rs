//! The settings questions: leave with changes not saved, and remove a model.

use std::cell::Cell;
use std::rc::Rc;

use gpui_kit::component::WindowExt as _;
use gpui_kit::component::checkbox::Checkbox;
use gpui_kit::component::dialog::DialogFooter;
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::*;

use super::SettingsView;
use crate::app::settings::rows::{
    DISCARD_BODY, DISCARD_CANCEL, DISCARD_CONFIRM, DISCARD_TITLE, remove_question,
};
use crate::domain::model_asset::RemovalEstimate;
use crate::gui::design::{size, space};
use crate::gui::dialogs::{Kind, button, title_text};

/// "Unsaved changes": `on_discard` runs only for Discard and exit.
pub fn ask_discard(
    window: &mut Window,
    cx: &mut App,
    on_discard: impl Fn(&mut Window, &mut App) + 'static,
) {
    let on_discard = Rc::new(on_discard);
    window.open_dialog(cx, move |dialog, _, _| {
        let on_discard = on_discard.clone();
        let discard = on_discard.clone();
        dialog
            .w(size::DIALOG)
            .title(title_text(DISCARD_TITLE))
            .child(div().py(space::SM).child(DISCARD_BODY))
            .footer(
                DialogFooter::new()
                    .justify_end()
                    .child(
                        button("discard-cancel", DISCARD_CANCEL, Kind::Plain)
                            .on_click(|_, window, cx| window.close_dialog(cx)),
                    )
                    .child(button("discard", DISCARD_CONFIRM, Kind::Danger).on_click(
                        move |_, window, cx| {
                            discard(window, cx);
                            window.close_dialog(cx);
                        },
                    )),
            )
            .on_ok(move |_, window, cx| {
                on_discard(window, cx);
                true
            })
    });
}

/// "Remove <label>?", with the delete-legacy choice when an older copy exists.
/// Every way out answers the runtime, so it never keeps a stale question.
pub fn ask_remove(
    label: &str,
    estimate: &RemovalEstimate,
    view: WeakEntity<SettingsView>,
    window: &mut Window,
    cx: &mut App,
) {
    let question = remove_question(label, estimate);
    let checked = Rc::new(Cell::new(false));
    let answered = Rc::new(Cell::new(false));
    window.open_dialog(cx, move |dialog, _, _| {
        let checkbox = question.checkbox.clone().map(|text| {
            let checked = checked.clone();
            Checkbox::new("delete-legacy")
                .label(text)
                .checked(checked.get())
                .on_click(move |on, window, _| {
                    checked.set(*on);
                    window.refresh();
                })
        });
        // Every way out answers once: Remove, Cancel, the close box, Escape.
        let answer = {
            let (view, checked, answered) = (view.clone(), checked.clone(), answered.clone());
            Rc::new(move |remove: bool, window: &mut Window, cx: &mut App| {
                if answered.replace(true) {
                    return;
                }
                let legacy = remove && checked.get();
                let _ = view.update(cx, |view, cx| {
                    view.answer_remove(remove, legacy, window, cx)
                });
            })
        };
        let (on_ok, on_close, on_cancel, on_remove) = (
            answer.clone(),
            answer.clone(),
            answer.clone(),
            answer.clone(),
        );
        dialog
            .w(size::DIALOG)
            .title(title_text(question.title.clone()))
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap(space::MD)
                    .py(space::SM)
                    .child(question.body.clone())
                    .when_some(checkbox, |body, checkbox| body.child(checkbox)),
            )
            .footer(
                DialogFooter::new()
                    .justify_end()
                    .child(button("remove-cancel", "Cancel", Kind::Plain).on_click(
                        move |_, window, cx| {
                            on_cancel(false, window, cx);
                            window.close_dialog(cx);
                        },
                    ))
                    .child(button("remove", "Remove", Kind::Danger).on_click(
                        move |_, window, cx| {
                            on_remove(true, window, cx);
                            window.close_dialog(cx);
                        },
                    )),
            )
            .on_ok(move |_, window, cx| {
                on_ok(true, window, cx);
                true
            })
            .on_close(move |_, window, cx| on_close(false, window, cx))
    });
}

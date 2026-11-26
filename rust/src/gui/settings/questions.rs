//! The settings questions: leave with changes not saved, and remove a model.

use std::cell::Cell;
use std::rc::Rc;

use gpui_kit::component::WindowExt as _;
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::checkbox::Checkbox;
use gpui_kit::component::dialog::{DialogAction, DialogClose, DialogFooter};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::*;

use super::SettingsView;
use crate::app::settings::rows::{
    DISCARD_BODY, DISCARD_CANCEL, DISCARD_CONFIRM, DISCARD_TITLE, remove_question,
};
use crate::domain::model_asset::RemovalEstimate;

fn title(text: impl Into<SharedString>) -> Div {
    div().font_weight(FontWeight::BOLD).child(text.into())
}

/// "Unsaved changes": `on_discard` runs only for Discard and exit.
pub fn ask_discard(
    window: &mut Window,
    cx: &mut App,
    on_discard: impl Fn(&mut Window, &mut App) + 'static,
) {
    let on_discard = Rc::new(on_discard);
    window.open_dialog(cx, move |dialog, _, _| {
        let on_discard = on_discard.clone();
        dialog
            .w(px(440.))
            .title(title(DISCARD_TITLE))
            .child(div().py_2().child(DISCARD_BODY))
            .footer(
                DialogFooter::new()
                    .justify_end()
                    .child(DialogClose::new().trigger(|button| button.label(DISCARD_CANCEL)))
                    .child(
                        DialogAction::new()
                            .child(Button::new("discard").danger().label(DISCARD_CONFIRM)),
                    ),
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
        let (on_ok, on_close) = (view.clone(), view.clone());
        let (ok_checked, ok_answered, close_answered) =
            (checked.clone(), answered.clone(), answered.clone());
        dialog
            .w(px(480.))
            .title(title(question.title.clone()))
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap_3()
                    .py_2()
                    .child(question.body.clone())
                    .when_some(checkbox, |body, checkbox| body.child(checkbox)),
            )
            .footer(
                DialogFooter::new()
                    .justify_end()
                    .child(DialogClose::new().trigger(|button| button.label("Cancel")))
                    .child(
                        DialogAction::new().child(Button::new("remove").danger().label("Remove")),
                    ),
            )
            .on_ok(move |_, window, cx| {
                ok_answered.set(true);
                let legacy = ok_checked.get();
                let _ = on_ok.update(cx, |view, cx| view.answer_remove(true, legacy, window, cx));
                true
            })
            .on_close(move |_, window, cx| {
                if !close_answered.replace(true) {
                    let _ = on_close
                        .update(cx, |view, cx| view.answer_remove(false, false, window, cx));
                }
            })
    });
}

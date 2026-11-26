//! The modals: a Markdown text with a close button, and a one-line prompt.
//! They replace the TUI's MarkdownModal and LabelModal.

use std::rc::Rc;

use gpui_kit::component::ActiveTheme as _;
use gpui_kit::component::WindowExt as _;
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::dialog::{DialogAction, DialogClose, DialogFooter};
use gpui_kit::component::input::{Input, InputEvent, InputState};
use gpui_kit::component::text::{TextView, TextViewState};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::*;

/// Something a dialog runs on the main thread.
pub type Callback = Rc<dyn Fn(&mut Window, &mut App)>;

/// A second button on a Markdown dialog, and what it does before the dialog closes.
pub struct Extra {
    pub label: &'static str,
    pub run: Callback,
}

/// Markdown in a dialog. `accent` is the Textual variable of its title, as
/// the TUI colors a modal's border. `on_close` runs however it closes.
pub fn markdown(
    window: &mut Window,
    cx: &mut App,
    title: impl Into<SharedString>,
    body: &str,
    accent: &'static str,
    extra: Option<Extra>,
    on_close: Option<Callback>,
) {
    let text = cx.new(|cx| TextViewState::markdown(body, cx));
    let title: SharedString = title.into();
    let extra = extra.map(Rc::new);
    window.open_dialog(cx, move |dialog, _, cx| {
        let footer = DialogFooter::new()
            .justify_end()
            .when_some(extra.clone(), |footer, extra| {
                footer.child(DialogAction::new().child(Button::new("extra").label(extra.label)))
            })
            .child(DialogClose::new().trigger(|button| button.primary().label("Close")));
        let extra = extra.clone();
        let on_close = on_close.clone();
        dialog
            .w(px(640.))
            .title(
                div()
                    .text_color(title_color(cx, accent))
                    .font_weight(FontWeight::BOLD)
                    .child(title.clone()),
            )
            .child(
                div()
                    .id("dialog-body")
                    .max_h(px(480.))
                    .overflow_y_scroll()
                    .child(TextView::new(&text).selectable(true)),
            )
            .footer(footer)
            .on_ok(move |_, window, cx| {
                if let Some(extra) = &extra {
                    (extra.run)(window, cx);
                }
                true
            })
            .on_close(move |_, window, cx| {
                if let Some(on_close) = &on_close {
                    on_close(window, cx);
                }
            })
    });
}

/// A one-line prompt. Enter or Save hands the trimmed text to `on_submit`;
/// an empty text keeps the old value, as the TUI's LabelModal does.
pub fn prompt(
    window: &mut Window,
    cx: &mut App,
    title: impl Into<SharedString>,
    current: &str,
    placeholder: &'static str,
    on_submit: impl Fn(String, &mut Window, &mut App) + 'static,
) {
    let input = cx.new(|cx| {
        InputState::new(window, cx)
            .placeholder(placeholder)
            .default_value(current.to_string())
    });
    let on_submit = Rc::new(on_submit);
    let submit = {
        let (input, on_submit) = (input.clone(), on_submit.clone());
        move |window: &mut Window, cx: &mut App| {
            let text = input.read(cx).value().trim().to_string();
            if !text.is_empty() {
                on_submit(text, window, cx);
            }
        }
    };
    let handle = window.window_handle();
    let on_enter = submit.clone();
    cx.subscribe(&input, move |_, event: &InputEvent, cx| {
        if let InputEvent::PressEnter { .. } = event {
            let _ = handle.update(cx, |_, window, cx| {
                on_enter(window, cx);
                window.close_dialog(cx);
            });
        }
    })
    .detach();
    let title: SharedString = title.into();
    let first = std::cell::Cell::new(true);
    window.open_dialog(cx, move |dialog, window, cx| {
        if first.replace(false) {
            let focus = input.read(cx).focus_handle(cx);
            focus.focus(window, cx);
        }
        let submit = submit.clone();
        dialog
            .w(px(460.))
            .title(
                div()
                    .text_color(cx.theme().foreground)
                    .font_weight(FontWeight::BOLD)
                    .child(title.clone()),
            )
            .child(Input::new(&input))
            .footer(
                DialogFooter::new()
                    .justify_end()
                    .child(DialogClose::new().trigger(|button| button.label("Cancel")))
                    .child(DialogAction::new().child(Button::new("save").primary().label("Save"))),
            )
            .on_ok(move |_, window, cx| {
                submit(window, cx);
                true
            })
    });
}

/// A yes/no question. `on_yes` runs only for Yes; `danger` paints Yes in the error color.
pub fn confirm(
    window: &mut Window,
    cx: &mut App,
    message: impl Into<SharedString>,
    yes: &'static str,
    danger: bool,
    on_yes: impl Fn(&mut Window, &mut App) + 'static,
) {
    let message: SharedString = message.into();
    let on_yes = Rc::new(on_yes);
    window.open_dialog(cx, move |dialog, _, _| {
        let on_yes = on_yes.clone();
        let yes_button = Button::new("yes").label(yes);
        let yes_button = if danger {
            yes_button.danger()
        } else {
            yes_button.primary()
        };
        dialog
            .w(px(440.))
            .child(div().py_2().child(message.clone()))
            .footer(
                DialogFooter::new()
                    .justify_end()
                    .child(DialogClose::new().trigger(|button| button.label("No")))
                    .child(DialogAction::new().child(yes_button)),
            )
            .on_ok(move |_, window, cx| {
                on_yes(window, cx);
                true
            })
    });
}

/// A dialog title is plain text, or the warning color for a warning.
fn title_color(cx: &App, accent: &str) -> Hsla {
    match accent {
        "warning" => cx.theme().warning,
        _ => cx.theme().foreground,
    }
}

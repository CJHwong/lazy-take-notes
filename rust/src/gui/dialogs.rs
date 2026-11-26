//! The modals: a Markdown text with a close button, and a one-line prompt.
//! They replace the TUI's MarkdownModal and LabelModal.

use std::rc::Rc;

use gpui_kit::component::ActiveTheme as _;
use gpui_kit::component::Sizable as _;
use gpui_kit::component::WindowExt as _;
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::dialog::DialogFooter;
use gpui_kit::component::input::{Input, InputEvent, InputState};
use gpui_kit::component::text::{TextView, TextViewState};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::*;

use crate::gui::design::{control, size, space, text};

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
        let finish = {
            let on_close = on_close.clone();
            move |window: &mut Window, cx: &mut App| {
                if let Some(on_close) = &on_close {
                    on_close(window, cx);
                }
                window.close_dialog(cx);
            }
        };
        let close_finish = finish.clone();
        let footer = DialogFooter::new()
            .justify_end()
            .when_some(extra.clone(), |footer, extra| {
                let finish = finish.clone();
                footer.child(
                    Button::new("dialog-extra")
                        .with_size(control::CONTROL)
                        .label(extra.label)
                        .on_click(move |_, window, cx| {
                            (extra.run)(window, cx);
                            finish(window, cx);
                        }),
                )
            })
            .child(
                Button::new("dialog-close")
                    .primary()
                    .with_size(control::CONTROL)
                    .label("Close")
                    .on_click(move |_, window, cx| close_finish(window, cx)),
            );
        let extra = extra.clone();
        let on_close = on_close.clone();
        dialog
            .w(size::DIALOG_WIDE)
            .title(title_text(title.clone()).text_color(title_color(cx, accent)))
            .child(
                div()
                    .id("dialog-body")
                    .max_h(size::DIALOG_BODY)
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
        let save = submit.clone();
        dialog
            .w(size::DIALOG)
            .title(title_text(title.clone()))
            .child(Input::new(&input).with_size(control::CONTROL))
            .footer(buttons("Save", false, move |window, cx| {
                save(window, cx);
                true
            }))
            .on_ok(move |_, window, cx| {
                submit(window, cx);
                true
            })
    });
}

/// A question with Cancel and a verb. `on_yes` runs only for the verb;
/// `danger` paints it in the solid red of a destructive action.
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
        let act = on_yes.clone();
        dialog
            .w(size::DIALOG)
            .child(div().py(space::SM).child(message.clone()))
            .footer(buttons(yes, danger, move |window, cx| {
                act(window, cx);
                true
            }))
            .on_ok(move |_, window, cx| {
                on_yes(window, cx);
                true
            })
    });
}

/// A dialog's title, one step over the body, as a macOS alert sets it.
pub fn title_text(title: impl Into<SharedString>) -> Div {
    div()
        .text_size(text::TITLE)
        .font_weight(text::STRONG)
        .child(title.into())
}

/// Cancel, then the action at the right end, each as wide as its label.
/// `on_action` returns false to keep the dialog open, as for a bad name.
/// The kit's own footer buttons fill the footer's width, so these are plain
/// buttons that close the dialog themselves.
pub fn buttons(
    action: &'static str,
    danger: bool,
    on_action: impl Fn(&mut Window, &mut App) -> bool + 'static,
) -> DialogFooter {
    let button = Button::new("dialog-action")
        .with_size(control::CONTROL)
        .label(action)
        .on_click(move |_, window, cx| {
            if on_action(window, cx) {
                window.close_dialog(cx);
            }
        });
    let button = match danger {
        true => button.danger(),
        false => button.primary(),
    };
    DialogFooter::new()
        .justify_end()
        .child(
            Button::new("dialog-cancel")
                .with_size(control::CONTROL)
                .label("Cancel")
                .on_click(|_, window, cx| window.close_dialog(cx)),
        )
        .child(button)
}

/// A dialog title is plain text, or the warning color for a warning.
fn title_color(cx: &App, accent: &str) -> Hsla {
    match accent {
        "warning" => cx.theme().warning,
        _ => cx.theme().foreground,
    }
}

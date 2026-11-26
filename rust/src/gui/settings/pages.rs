//! What the settings pane draws: the page list, the five pages as labeled
//! rows, and the bar with Save and Revert.

use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::input::{Input, Textarea};
use gpui_kit::component::progress::Progress;
use gpui_kit::component::radio::Radio;
use gpui_kit::component::select::Select;
use gpui_kit::component::switch::Switch;
use gpui_kit::component::{ActiveTheme as _, Disableable as _, IconName, Sizable as _};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::*;

use super::{API_KEY, CUSTOM_PATH, DEVICE, PAGES, Page, SettingsView};
use crate::app::download_progress::megabytes;
use crate::app::session::status_line::{LEVEL_TOP, level_step};
use crate::app::settings::rows::{RowAction, row_actions, row_detail, row_status, status_color};
use crate::app::settings::{Meter, SPEAKER, open_privacy};
use crate::domain::model_asset::AssetStatus;
use crate::gui::design::size;

const NAV_WIDTH: Pixels = px(180.);
const LABEL_WIDTH: Pixels = px(190.);
const CONTROL_WIDTH: Pixels = px(360.);

/// A theme color by the name the runtime gives it.
fn tint(name: Option<&str>, cx: &App) -> Hsla {
    let theme = cx.theme();
    match name {
        Some("success") => theme.success,
        Some("warning") => theme.warning,
        Some("error") => theme.danger,
        Some("accent") => theme.primary,
        _ => theme.foreground,
    }
}

fn heading(text: &'static str) -> Div {
    div()
        .pt_5()
        .pb_1()
        .font_weight(FontWeight::SEMIBOLD)
        .child(text)
}

fn note(text: impl Into<SharedString>, cx: &App) -> Div {
    div()
        .text_sm()
        .text_color(cx.theme().muted_foreground)
        .child(text.into())
}

/// A form row: the label at the left, the control and its help at the right.
fn row(label: &'static str, control: impl IntoElement, help: &str, cx: &App) -> Div {
    div()
        .flex()
        .gap_4()
        .py_2()
        .border_b_1()
        .border_color(cx.theme().border.opacity(0.5))
        .child(div().flex_none().w(LABEL_WIDTH).pt_1().child(label))
        .child(
            div()
                .flex_1()
                .min_w_0()
                .flex()
                .flex_col()
                .gap_1()
                .child(div().max_w(CONTROL_WIDTH).child(control))
                .when(!help.is_empty(), |column| {
                    column.child(note(help.to_string(), cx))
                }),
        )
}

fn buttons() -> Div {
    div().flex().flex_wrap().items_center().gap_2().py_2()
}

impl SettingsView {
    fn input(&self, id: &'static str) -> Input {
        Input::new(&self.inputs[id])
    }

    fn field(&self, id: &'static str, label: &'static str, help: &str, cx: &App) -> Div {
        row(label, self.input(id), help, cx)
    }

    fn select(&self, id: &'static str, label: &'static str, help: &str, cx: &App) -> Div {
        row(label, Select::new(&self.selects[id]), help, cx)
    }

    fn switch(
        &self,
        id: &'static str,
        label: &'static str,
        help: &str,
        cx: &mut Context<Self>,
    ) -> Div {
        let on = self.switches.get(id).copied().unwrap_or(false);
        let view = cx.entity().downgrade();
        let control = Switch::new(id).checked(on).on_click(move |on, _, cx| {
            let on = *on;
            let _ = view.update(cx, |view, cx| {
                view.switches.insert(id, on);
                cx.notify();
            });
        });
        row(label, control, help, cx)
    }

    fn button(
        id: &'static str,
        label: &'static str,
        cx: &mut Context<Self>,
        run: impl Fn(&mut Self, &mut Window, &mut Context<Self>) + 'static,
    ) -> Button {
        Button::new(id)
            .small()
            .label(label)
            .on_click(cx.listener(move |view, _, window, cx| run(view, window, cx)))
    }

    // --- Models ---------------------------------------------------------

    fn models_page(&self, cx: &mut Context<Self>) -> Vec<AnyElement> {
        let models_text = self
            .runtime
            .models_text
            .clone()
            .unwrap_or_else(|| "Use Refresh to list the models on the server.".into());
        let mut items: Vec<AnyElement> = vec![
            heading("Text generation").into_any_element(),
            self.select(
                "cfg-llm-provider",
                "Provider",
                "Ollama runs a model here, or a cloud model after ollama signin. OpenAI needs an API key.",
                cx,
            )
            .into_any_element(),
            self.field("cfg-ollama-host", "Server", "", cx).into_any_element(),
            self.field("cfg-openai-base-url", "API base", "", cx)
                .into_any_element(),
        ];
        let key = div()
            .flex()
            .flex_col()
            .gap_2()
            .child(self.input(API_KEY))
            .when(self.key_stored, |column| {
                column.child(div().child(Self::button(
                    "clear-key",
                    "Clear Stored Key",
                    cx,
                    |view, window, cx| view.clear_key(window, cx),
                )))
            });
        items.push(row("API key", key, "", cx).into_any_element());
        items.push(
            self.field("cfg-digest-model", "Summary model", "", cx)
                .into_any_element(),
        );
        items.push(
            self.field("cfg-interactive-model", "Quick action model", "", cx)
                .into_any_element(),
        );
        let tools = buttons()
            .child(Self::button("ai-test", "Test", cx, |view, window, cx| {
                view.call(window, cx, |runtime, form, io| runtime.test_ai(form, io));
            }))
            .child(Self::button(
                "ai-refresh",
                "Refresh",
                cx,
                |view, window, cx| {
                    view.call(window, cx, |runtime, form, io| {
                        runtime.refresh_models(form, io)
                    });
                },
            ))
            .child(Self::button(
                "pull-digest",
                "Pull Summary Model",
                cx,
                |view, window, cx| {
                    view.call(window, cx, |runtime, form, io| {
                        runtime.pull_model("cfg-digest-model", form, io)
                    });
                },
            ))
            .child(Self::button(
                "pull-interactive",
                "Pull Quick Action Model",
                cx,
                |view, window, cx| {
                    view.call(window, cx, |runtime, form, io| {
                        runtime.pull_model("cfg-interactive-model", form, io)
                    });
                },
            ));
        items.push(row("Installed models", note(models_text, cx), "", cx).into_any_element());
        items.push(tools.into_any_element());
        items.push(heading("Speech-to-Text").into_any_element());
        items.push(
            self.select(
                "cfg-trans-provider",
                "Engine",
                "whisper-cpp runs locally. Plugin engines appear when installed.",
                cx,
            )
            .into_any_element(),
        );
        items.push(note("Default model", cx).pt_2().into_any_element());
        for index in 0..self.assets.len() {
            items.push(self.model_row(index, cx).into_any_element());
        }
        items.push(
            buttons()
                .child(Self::button(
                    "download-all",
                    "Download All Missing",
                    cx,
                    |view, window, cx| {
                        view.call(window, cx, |runtime, _, io| {
                            runtime.download_all_missing(io)
                        });
                    },
                ))
                .into_any_element(),
        );
        items.push(heading("Language models").into_any_element());
        items.push(
            note(
                "Use a different model for a locale. The default applies otherwise.",
                cx,
            )
            .into_any_element(),
        );
        for index in 0..self.languages.len() {
            items.push(self.language_line(index, cx).into_any_element());
        }
        items.push(
            buttons()
                .child(
                    Self::button("add-language", "Add Language", cx, |view, window, cx| {
                        view.add_language(window, cx)
                    })
                    .icon(IconName::Plus),
                )
                .into_any_element(),
        );
        items.push(heading("Speaker assets").into_any_element());
        items.push(self.speaker_row(cx).into_any_element());
        items.push(heading("Advanced").into_any_element());
        items.push(
            self.field(
                CUSTOM_PATH,
                "Custom model file",
                "A custom file wins over the marked model.",
                cx,
            )
            .into_any_element(),
        );
        let cache = self.runtime.store().cache_root().display().to_string();
        let open = Self::button("open-cache", "Open Cache Folder", cx, |view, _, cx| {
            view.open_cache(cx)
        })
        .icon(IconName::FolderOpen);
        items.push(
            row(
                "Cache",
                div()
                    .flex()
                    .flex_col()
                    .gap_2()
                    .child(note(cache, cx))
                    .child(div().child(open)),
                "",
                cx,
            )
            .into_any_element(),
        );
        items
    }

    /// One speech model: the mark and its name, size, status, buttons, then
    /// the progress or the reason it waits.
    fn model_row(&self, index: usize, cx: &mut Context<Self>) -> Div {
        let asset = &self.assets[index];
        let id = asset.id;
        let Some(state) = self.runtime.rows.get(id).cloned() else {
            return div();
        };
        let theme = cx.theme();
        let (muted, border) = (theme.muted_foreground, theme.border.opacity(0.5));
        let view = cx.entity().downgrade();
        let mark = Radio::new(("model", index))
            .label(asset.label)
            .checked(self.chosen == Some(id))
            .on_click(move |_, window, cx| {
                let _ = view.update(cx, |view, cx| view.choose_model(id, window, cx));
            });
        let actions = row_actions(state.status).into_iter().map(|action| {
            Button::new((action.id(), index))
                .small()
                .when(action == RowAction::Remove, |button| button.ghost())
                .label(action.label())
                .on_click(cx.listener(move |view, _, window, cx| match action {
                    RowAction::Download => {
                        view.call(window, cx, |runtime, _, io| runtime.start_download(id, io));
                    }
                    RowAction::Cancel => {
                        view.runtime.cancel_download(id);
                        cx.notify();
                    }
                    RowAction::Remove => view.ask_remove(id, window, cx),
                }))
        });
        let detail = row_detail(&state);
        let downloading = state.status == AssetStatus::Downloading;
        let percent = state
            .progress
            .map(|p| p.downloaded as f32 * 100. / p.total.max(1) as f32)
            .unwrap_or(0.);
        div()
            .flex()
            .flex_col()
            .gap_1()
            .py_2()
            .border_b_1()
            .border_color(border)
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_3()
                    .child(div().flex_1().min_w_0().child(mark))
                    .child(
                        div()
                            .w(px(80.))
                            .text_sm()
                            .text_color(muted)
                            .child(megabytes(asset.size)),
                    )
                    .child(
                        div()
                            .w(px(100.))
                            .text_sm()
                            .text_color(tint(status_color(state.status), cx))
                            .child(row_status(&state)),
                    )
                    .child(
                        div()
                            .flex()
                            .gap_1()
                            .w(px(170.))
                            .justify_end()
                            .children(actions),
                    ),
            )
            .when(downloading, |column| {
                column.child(
                    div()
                        .pl_6()
                        .max_w(CONTROL_WIDTH)
                        .child(Progress::new(("model-progress", index)).value(percent)),
                )
            })
            .when(!detail.is_empty(), |column| {
                column.child(note(detail, cx).pl_6())
            })
    }

    fn language_line(&self, index: usize, cx: &mut Context<Self>) -> Div {
        let row = &self.languages[index];
        div()
            .flex()
            .items_center()
            .gap_2()
            .py_1()
            .child(div().w(px(90.)).child(Input::new(&row.locale)))
            .child(
                div()
                    .flex_1()
                    .max_w(CONTROL_WIDTH)
                    .child(Select::new(&row.model)),
            )
            .child(
                Button::new(("remove-language", index))
                    .small()
                    .ghost()
                    .label("Remove")
                    .on_click(cx.listener(move |view, _, _, cx| view.remove_language(index, cx))),
            )
    }

    fn speaker_row(&self, cx: &mut Context<Self>) -> Div {
        let speaker = self.runtime.speaker.clone();
        let status = match speaker.status.is_empty() {
            true => "Checking...".to_string(),
            false => speaker.status.clone(),
        };
        let controls = buttons()
            .when(speaker.install, |bar| {
                bar.child(Self::button(
                    "speaker-install",
                    "Install",
                    cx,
                    |view, window, cx| {
                        view.call(window, cx, |runtime, _, io| runtime.install_speaker(io));
                    },
                ))
            })
            .when(speaker.cancel, |bar| {
                bar.child(Self::button(
                    "speaker-cancel",
                    "Cancel",
                    cx,
                    |view, _, cx| {
                        view.runtime.cancel_speaker();
                        cx.notify();
                    },
                ))
            })
            .when(speaker.remove, |bar| {
                bar.child(
                    Self::button("speaker-remove", "Remove", cx, |view, window, cx| {
                        view.ask_remove(SPEAKER, window, cx)
                    })
                    .ghost(),
                )
            });
        row(
            "Runtime and model",
            div()
                .flex()
                .flex_col()
                .child(div().pt_1().child(status))
                .child(controls),
            "Speaker labels need these. They download once.",
            cx,
        )
    }

    // --- Digest ---------------------------------------------------------

    fn digest_page(&self, cx: &mut Context<Self>) -> Vec<AnyElement> {
        vec![
            heading("When to summarize").into_any_element(),
            self.field("cfg-digest-min-lines", "After lines", "", cx)
                .into_any_element(),
            self.field("cfg-digest-min-interval", "Wait seconds", "", cx)
                .into_any_element(),
            self.field(
                "cfg-digest-max-lines",
                "Force after",
                "Force after is optional. Blank means twice After lines.",
                cx,
            )
            .into_any_element(),
            heading("Advanced").into_any_element(),
            self.field(
                "cfg-digest-compact",
                "Compaction threshold",
                "Compress history when it exceeds this token count. Change it only if you hit context-length errors.",
                cx,
            )
            .into_any_element(),
        ]
    }

    // --- Audio ----------------------------------------------------------

    /// A source's level bars against the silence threshold, and its permission.
    fn meter(&self, label: &'static str, meter: &Meter, cx: &App) -> Div {
        let theme = cx.theme();
        let (quiet, loud) = (theme.muted_foreground.opacity(0.4), theme.primary);
        let silence = self.runtime.silence_threshold();
        let bars = meter.levels.iter().map(|level| {
            let step = level_step(*level, silence);
            let height = 3. + 17. * step as f32 / LEVEL_TOP as f32;
            div()
                .w(px(5.))
                .h(px(height))
                .rounded(px(1.))
                .bg(if step == 0 { quiet } else { loud })
        });
        let (text, color) = match meter.permission {
            Some(permission) => (permission.text, permission.color),
            None => ("Not tested", None),
        };
        let color = match color {
            Some(_) => tint(color, cx),
            None => theme.muted_foreground,
        };
        row(
            label,
            div()
                .flex()
                .items_center()
                .gap_4()
                .child(
                    div()
                        .flex()
                        .items_end()
                        .gap(px(2.))
                        .h(px(20.))
                        .children(bars),
                )
                .child(div().text_sm().text_color(color).child(text)),
            "",
            cx,
        )
    }

    fn audio_page(&self, cx: &mut Context<Self>) -> Vec<AnyElement> {
        let running = self.runtime.probe_running();
        let probe_buttons = buttons()
            .child(
                Self::button("audio-start", "Start Test", cx, |view, window, cx| {
                    view.call(window, cx, |runtime, form, io| {
                        runtime.start_probe(form, io)
                    });
                })
                .icon(IconName::Play)
                .disabled(running),
            )
            .child(
                Self::button("audio-stop", "Stop", cx, |view, _, cx| {
                    view.runtime.stop_probe();
                    cx.notify();
                })
                .icon(IconName::Square)
                .disabled(!running),
            );
        let grant = buttons()
            .child(Self::button(
                "grant-system",
                "Grant System Audio",
                cx,
                |_, _, _| open_privacy("Privacy_ScreenCapture"),
            ))
            .child(Self::button(
                "grant-mic",
                "Open Microphone Settings",
                cx,
                |_, _, _| open_privacy("Privacy_Microphone"),
            ));
        let meters = &self.runtime.meters;
        vec![
            heading("Input check").into_any_element(),
            note(
                "Start a 15 second test and make a sound. Nothing is saved.",
                cx,
            )
            .into_any_element(),
            probe_buttons.into_any_element(),
            self.meter("Microphone", &meters[0], cx).into_any_element(),
            self.meter("System audio", &meters[1], cx)
                .into_any_element(),
            grant.into_any_element(),
            heading("Microphone").into_any_element(),
            self.select(
                DEVICE,
                "Input device",
                "A removed device falls back to the system default with a warning.",
                cx,
            )
            .into_any_element(),
            heading("Audio processing").into_any_element(),
            self.field("cfg-trans-chunk", "Chunk duration", "", cx)
                .into_any_element(),
            self.field("cfg-trans-overlap", "Overlap", "", cx)
                .into_any_element(),
            self.field("cfg-trans-silence", "Silence threshold", "", cx)
                .into_any_element(),
            self.field("cfg-trans-pause", "Pause duration", "", cx)
                .into_any_element(),
            heading("Speaker labels").into_any_element(),
            self.switch("cfg-diarization-enabled", "Label speakers", "", cx)
                .into_any_element(),
            self.select("cfg-diarization-device", "Device", "", cx)
                .into_any_element(),
            self.field("cfg-diarization-interval", "Update every", "", cx)
                .into_any_element(),
        ]
    }

    // --- Sessions and Appearance ----------------------------------------

    fn sessions_page(&self, cx: &mut Context<Self>) -> Vec<AnyElement> {
        vec![
            heading("Session storage").into_any_element(),
            self.field("cfg-output-dir", "Output folder", "", cx)
                .into_any_element(),
            self.switch("cfg-output-save-audio", "Save audio", "", cx)
                .into_any_element(),
            self.switch(
                "cfg-output-save-notes-history",
                "Keep notes history",
                "",
                cx,
            )
            .into_any_element(),
            self.switch("cfg-output-save-context", "Save session context", "", cx)
                .into_any_element(),
            self.switch("cfg-output-save-debug-log", "Save debug log", "", cx)
                .into_any_element(),
            self.switch("cfg-output-auto-label", "Auto-label sessions", "", cx)
                .into_any_element(),
            heading("Custom vocabulary").into_any_element(),
            row(
                "Words to watch for",
                Textarea::new(&self.hints),
                "Names, acronyms, or jargon the recognizer should watch for. One per line.",
                cx,
            )
            .into_any_element(),
            heading("Templates").into_any_element(),
            self.switch(
                "cfg-show-builtin-templates",
                "Show built-in templates",
                "Bundled examples appear in the picker. Turn it off to see only your own.",
                cx,
            )
            .into_any_element(),
        ]
    }

    fn appearance_page(&self, cx: &mut Context<Self>) -> Vec<AnyElement> {
        vec![
            heading("Theme").into_any_element(),
            self.select(
                "cfg-theme",
                "Terminal theme",
                "This theme colors the terminal app only. This window follows the macOS appearance.",
                cx,
            )
            .into_any_element(),
        ]
    }

    // --- The frame ------------------------------------------------------

    fn nav(&self, cx: &mut Context<Self>) -> Div {
        let theme = cx.theme();
        let (active, hover) = (theme.list_active, theme.foreground.opacity(0.05));
        let rows = PAGES.iter().map(|(page, label)| {
            let (page, current) = (*page, *page == self.page);
            div()
                .id(*label)
                .px_3()
                .py(px(6.))
                .rounded(px(6.))
                .text_sm()
                .when(current, |row| row.bg(active))
                .when(!current, |row| row.hover(|row| row.bg(hover)))
                .child(*label)
                .on_click(cx.listener(move |view, _, _, cx| {
                    view.page = page;
                    cx.notify();
                }))
        });
        div()
            .flex_none()
            .w(NAV_WIDTH)
            .h_full()
            .flex()
            .flex_col()
            .gap_1()
            .p_2()
            .border_r_1()
            .border_color(theme.border)
            .children(rows)
    }

    fn bar(&self, cx: &mut Context<Self>) -> Div {
        let path = self.runtime.config_file().display().to_string();
        let muted = cx.theme().muted_foreground;
        div()
            .flex()
            .flex_none()
            .items_center()
            .gap_2()
            .h(size::ACTION_BAR)
            .px_4()
            .border_t_1()
            .border_color(cx.theme().border)
            .child(
                Button::new("open-config")
                    .ghost()
                    .icon(IconName::FileText)
                    .label("Open Config File")
                    .tooltip(path.clone())
                    .on_click(cx.listener(|view, _, window, cx| view.open_config(window, cx))),
            )
            .child(
                div()
                    .flex_1()
                    .min_w_0()
                    .truncate()
                    .text_sm()
                    .text_color(muted)
                    .child(path),
            )
            .child(
                Button::new("revert")
                    .icon(IconName::Undo2)
                    .label("Revert")
                    .on_click(cx.listener(|view, _, window, cx| view.revert(window, cx))),
            )
            .child(
                Button::new("save")
                    .primary()
                    .label("Save")
                    .on_click(cx.listener(|view, _, window, cx| view.save(window, cx))),
            )
    }
}

impl Render for SettingsView {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let items = match self.page {
            Page::Models => self.models_page(cx),
            Page::Digest => self.digest_page(cx),
            Page::Audio => self.audio_page(cx),
            Page::Sessions => self.sessions_page(cx),
            Page::Appearance => self.appearance_page(cx),
        };
        div()
            .key_context("Settings")
            .track_focus(&self.focus)
            .on_key_down(cx.listener(|view, event: &KeyDownEvent, window, cx| {
                if event.keystroke.key == "escape" {
                    view.back(window, cx);
                }
            }))
            .size_full()
            .flex()
            .flex_col()
            .child(
                div().flex().flex_1().min_h_0().child(self.nav(cx)).child(
                    div()
                        .id("settings-page")
                        .flex_1()
                        .min_w_0()
                        .px_6()
                        .pb_6()
                        .overflow_y_scroll()
                        .children(items),
                ),
            )
            .child(self.bar(cx))
    }
}

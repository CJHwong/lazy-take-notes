//! The native window (GPUI Kit): the front end of the macOS .app. Each screen
//! is a view over the same runtime the TUI drives, so this module holds no
//! session logic of its own.

mod bridge;
mod builder;
mod env;
mod preflight;
mod session;
mod session_picker;
mod settings;
mod shell;
mod template_picker;
mod theme;
mod view;
mod welcome;
mod widgets;

use std::sync::OnceLock;

use gpui_kit::*;

use crate::gui::shell::Shell;

actions!(ltn, [Quit]);

/// Where sessions go when LTN_OUTPUT_DIR does not say.
const DEFAULT_OUTPUT_DIR: &str = "~/Documents/LazyTakeNotes";

static RUNTIME: OnceLock<tokio::runtime::Runtime> = OnceLock::new();

/// The tokio runtime every background task of the window runs on.
pub(crate) fn runtime() -> tokio::runtime::Handle {
    RUNTIME
        .get_or_init(|| {
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .thread_name("ltn-gui-worker")
                .build()
                .expect("the tokio runtime starts")
        })
        .handle()
        .clone()
}

/// Open the window and run until it closes. Returns the exit code.
pub fn run() -> i32 {
    env::import_login_environment();
    // As the Trolley app set them: a Finder start runs in `/`, so the default
    // `./output` cannot work, and the kiosk flag lets the system audio helper
    // ask for its permission.
    // SAFETY: no other thread runs yet.
    unsafe {
        if std::env::var_os("LTN_OUTPUT_DIR").is_none() {
            std::env::set_var("LTN_OUTPUT_DIR", DEFAULT_OUTPUT_DIR);
        }
        std::env::set_var("LTN_KIOSK", "1");
    }
    let _ = runtime();
    application()
        .with_assets(assets::Assets)
        .run(|cx: &mut App| {
            init(cx);
            theme::apply(&crate::app::theme::saved_theme(), cx);
            cx.bind_keys([KeyBinding::new("cmd-q", Quit, None)]);
            cx.on_action(|_: &Quit, cx: &mut App| shell::request_quit(cx));
            cx.set_menus([Menu::new("Lazy Take Notes").items([MenuItem::action("Quit", Quit)])]);
            let options = WindowOptions {
                titlebar: Some(TitlebarOptions {
                    title: Some("Lazy Take Notes".into()),
                    ..Default::default()
                }),
                window_bounds: Some(WindowBounds::centered(None, size(px(1200.), px(800.)), cx)),
                window_min_size: Some(size(px(760.), px(520.))),
                ..Default::default()
            };
            let opened = open_window(options, cx, |window, cx| {
                cx.new(|cx| Shell::new(window, cx))
            });
            match opened {
                Ok((handle, shell)) => shell::watch_close(handle, shell, cx),
                Err(e) => {
                    eprintln!("Could not open the window: {e}");
                    cx.quit();
                }
            }
        });
    crate::app::logging::close_file_logging();
    0
}

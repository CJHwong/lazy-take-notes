//! The native window (GPUI Kit): the front end of the macOS .app. Each screen
//! is a view over the same runtime the TUI drives, so this module holds no
//! session logic of its own.

mod bridge;
mod builder;
mod design;
mod dialogs;
mod env;
pub(crate) mod icons;
mod preflight;
mod session;
mod settings;
mod shell;
mod template_picker;
mod theme;
mod view;
mod widgets;

use std::sync::OnceLock;

use gpui_kit::*;

use crate::gui::shell::Shell;

actions!(
    ltn,
    [
        Quit,
        CloseWindow,
        NewSession,
        OpenSettings,
        ShowHelp,
        TogglePause,
        StopSession,
        FinishSession,
        DigestNow,
        ToggleMute,
        QuickAction1,
        QuickAction2,
        QuickAction3,
        QuickAction4,
        QuickAction5,
        CopyTranscript,
        CopyDigest,
        RenameSession,
        ShowInFinder
    ]
);

/// The key context of the live session pane: its commands work only there.
pub(crate) const SESSION_CONTEXT: &str = "Session";

/// The quick action at `index` (from 0), as a menu or key command.
pub(crate) fn quick_action(index: usize) -> Option<Box<dyn Action>> {
    let action: Box<dyn Action> = match index {
        0 => Box::new(QuickAction1),
        1 => Box::new(QuickAction2),
        2 => Box::new(QuickAction3),
        3 => Box::new(QuickAction4),
        4 => Box::new(QuickAction5),
        _ => return None,
    };
    Some(action)
}

fn session_menu() -> Menu {
    Menu::new("Session").items([
        MenuItem::action("Pause or Resume Recording", TogglePause),
        MenuItem::action("Stop", StopSession),
        MenuItem::action("Finish Session", FinishSession),
        MenuItem::separator(),
        MenuItem::action("Digest Now", DigestNow),
        MenuItem::action("Mute Microphone", ToggleMute),
        MenuItem::action("Quick Action 1", QuickAction1),
        MenuItem::action("Quick Action 2", QuickAction2),
        MenuItem::action("Quick Action 3", QuickAction3),
        MenuItem::action("Quick Action 4", QuickAction4),
        MenuItem::action("Quick Action 5", QuickAction5),
        MenuItem::separator(),
        MenuItem::action("Copy Transcript", CopyTranscript),
        MenuItem::action("Copy Digest", CopyDigest),
        MenuItem::separator(),
        MenuItem::action("Rename…", RenameSession),
        MenuItem::action("Show in Finder", ShowInFinder),
    ])
}

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
    // A Finder start runs in `/`, so the default `./output` cannot work, and
    // the kiosk flag lets the system audio helper ask for its permission.
    // SAFETY: no other thread runs yet.
    unsafe {
        if std::env::var_os("LTN_OUTPUT_DIR").is_none() {
            std::env::set_var("LTN_OUTPUT_DIR", DEFAULT_OUTPUT_DIR);
        }
        std::env::set_var("LTN_KIOSK", "1");
    }
    let _ = runtime();
    application()
        .with_assets(icons::AppAssets)
        .run(|cx: &mut App| {
            init(cx);
            cx.bind_keys([
                KeyBinding::new("cmd-q", Quit, None),
                KeyBinding::new("cmd-w", CloseWindow, None),
                KeyBinding::new("cmd-n", NewSession, None),
                KeyBinding::new("cmd-,", OpenSettings, None),
            ]);
            let session = Some(SESSION_CONTEXT);
            cx.bind_keys([
                KeyBinding::new("cmd-.", StopSession, session),
                KeyBinding::new("cmd-enter", FinishSession, session),
                KeyBinding::new("cmd-d", DigestNow, session),
                KeyBinding::new("shift-cmd-m", ToggleMute, session),
                KeyBinding::new("cmd-1", QuickAction1, session),
                KeyBinding::new("cmd-2", QuickAction2, session),
                KeyBinding::new("cmd-3", QuickAction3, session),
                KeyBinding::new("cmd-4", QuickAction4, session),
                KeyBinding::new("cmd-5", QuickAction5, session),
                KeyBinding::new("shift-cmd-c", CopyTranscript, session),
                KeyBinding::new("alt-cmd-c", CopyDigest, session),
                KeyBinding::new("cmd-l", RenameSession, session),
                KeyBinding::new("shift-cmd-r", ShowInFinder, session),
            ]);
            // A key reaches these while the window dispatches it, and the
            // window cannot be updated then: each runs once the dispatch ends.
            cx.on_action(|_: &Quit, cx: &mut App| cx.defer(shell::request_quit));
            cx.on_action(|_: &CloseWindow, cx: &mut App| cx.defer(shell::request_quit));
            cx.on_action(|_: &NewSession, cx: &mut App| cx.defer(shell::new_session));
            cx.on_action(|_: &OpenSettings, cx: &mut App| cx.defer(shell::open_settings));
            // The live session pane shows its own help; this one serves the rest.
            cx.on_action(|_: &ShowHelp, cx: &mut App| cx.defer(shell::show_help));
            cx.set_menus([
                Menu::new("Lazy Take Notes").items([
                    MenuItem::action("Settings…", OpenSettings),
                    MenuItem::separator(),
                    MenuItem::action("Quit Lazy Take Notes", Quit),
                ]),
                Menu::new("File").items([
                    MenuItem::action("New Session", NewSession),
                    MenuItem::separator(),
                    MenuItem::action("Close Window", CloseWindow),
                ]),
                session_menu(),
                Menu::new("Help").items([MenuItem::action("Lazy Take Notes Help", ShowHelp)]),
            ]);
            let options = WindowOptions {
                titlebar: Some(TitlebarOptions {
                    title: Some("Lazy Take Notes".into()),
                    appears_transparent: true,
                    traffic_light_position: Some(shell::TRAFFIC_LIGHTS),
                }),
                // The panes draw the title bar row and move the window themselves.
                app_owns_titlebar_drag: true,
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

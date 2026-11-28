//! The native window (GPUI Kit): the front end of the macOS .app. Each screen
//! is a view over the same runtime the TUI drives, so this module holds no
//! session logic of its own.

mod bridge;
mod builder;
mod design;
mod dialogs;
#[cfg(target_os = "macos")]
mod dock_icon;
mod env;
pub(crate) mod icons;
mod keyscroll;
mod preflight;
mod session;
mod settings;
mod shell;
mod template_picker;
mod theme;
mod view;
mod widgets;
#[cfg(target_os = "macos")]
mod window_move;

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
        ShowInFinder,
        ScrollPageUp,
        ScrollPageDown,
        ScrollTop,
        ScrollBottom,
        HideApp,
        HideOthers,
        ShowAll,
        Minimize,
        Zoom
    ]
);

/// Run `act` on the window that has the keys, once the key dispatch ends.
fn on_active_window(cx: &mut App, act: fn(&Window)) {
    cx.defer(move |cx| {
        if let Some(handle) = cx.active_window() {
            let _ = handle.update(cx, |_, window, _| act(window));
        }
    });
}

/// The standard text commands. A text field handles them; elsewhere the
/// items are off.
fn edit_menu() -> Menu {
    use gpui_kit::component::input;
    Menu::new("Edit").items([
        MenuItem::os_action("Undo", input::Undo, OsAction::Undo),
        MenuItem::os_action("Redo", input::Redo, OsAction::Redo),
        MenuItem::separator(),
        MenuItem::os_action("Cut", input::Cut, OsAction::Cut),
        MenuItem::os_action("Copy", input::Copy, OsAction::Copy),
        MenuItem::os_action("Paste", input::Paste, OsAction::Paste),
        MenuItem::os_action("Select All", input::SelectAll, OsAction::SelectAll),
    ])
}

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
        MenuItem::action("Update Notes", DigestNow),
        MenuItem::action("Mute Microphone", ToggleMute),
        MenuItem::action("Ask AI 1", QuickAction1),
        MenuItem::action("Ask AI 2", QuickAction2),
        MenuItem::action("Ask AI 3", QuickAction3),
        MenuItem::action("Ask AI 4", QuickAction4),
        MenuItem::action("Ask AI 5", QuickAction5),
        MenuItem::separator(),
        MenuItem::action("Copy Transcript", CopyTranscript),
        MenuItem::action("Copy Notes", CopyDigest),
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
            #[cfg(target_os = "macos")]
            dock_icon::set_outside_bundle();
            cx.bind_keys([
                KeyBinding::new("cmd-q", Quit, None),
                KeyBinding::new("cmd-w", CloseWindow, None),
                KeyBinding::new("cmd-n", NewSession, None),
                KeyBinding::new("cmd-,", OpenSettings, None),
                KeyBinding::new("cmd-h", HideApp, None),
                KeyBinding::new("alt-cmd-h", HideOthers, None),
                KeyBinding::new("cmd-m", Minimize, None),
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
                // A text field binds these keys too, and its binding wins
                // while it has the focus.
                KeyBinding::new("pageup", ScrollPageUp, session),
                KeyBinding::new("pagedown", ScrollPageDown, session),
                KeyBinding::new("home", ScrollTop, session),
                KeyBinding::new("end", ScrollBottom, session),
            ]);
            // A key reaches these while the window dispatches it, and the
            // window cannot be updated then: each runs once the dispatch ends.
            cx.on_action(|_: &Quit, cx: &mut App| cx.defer(shell::request_quit));
            cx.on_action(|_: &CloseWindow, cx: &mut App| cx.defer(shell::request_quit));
            cx.on_action(|_: &NewSession, cx: &mut App| cx.defer(shell::new_session));
            cx.on_action(|_: &OpenSettings, cx: &mut App| cx.defer(shell::open_settings));
            // The live session pane shows its own help; this one serves the rest.
            cx.on_action(|_: &ShowHelp, cx: &mut App| cx.defer(shell::show_help));
            cx.on_action(|_: &HideApp, cx: &mut App| cx.hide());
            cx.on_action(|_: &HideOthers, cx: &mut App| cx.hide_other_apps());
            cx.on_action(|_: &ShowAll, cx: &mut App| cx.unhide_other_apps());
            cx.on_action(|_: &Minimize, cx: &mut App| {
                on_active_window(cx, Window::minimize_window)
            });
            cx.on_action(|_: &Zoom, cx: &mut App| on_active_window(cx, Window::zoom_window));
            cx.set_menus([
                Menu::new("Lazy Take Notes").items([
                    MenuItem::action("Settings…", OpenSettings),
                    MenuItem::separator(),
                    MenuItem::os_submenu("Services", SystemMenuType::Services),
                    MenuItem::separator(),
                    MenuItem::action("Hide Lazy Take Notes", HideApp),
                    MenuItem::action("Hide Others", HideOthers),
                    MenuItem::action("Show All", ShowAll),
                    MenuItem::separator(),
                    MenuItem::action("Quit Lazy Take Notes", Quit),
                ]),
                Menu::new("File").items([
                    MenuItem::action("New Session", NewSession),
                    MenuItem::separator(),
                    MenuItem::action("Close Window", CloseWindow),
                ]),
                edit_menu(),
                session_menu(),
                // AppKit adds full screen, tiling and the window list to the menu
                // named Window.
                Menu::new("Window").items([
                    MenuItem::action("Minimize", Minimize),
                    MenuItem::action("Zoom", Zoom),
                ]),
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
                window_bounds: Some(WindowBounds::centered(
                    None,
                    size(design::size::WINDOW_W, design::size::WINDOW_H),
                    cx,
                )),
                window_min_size: Some(size(design::size::WINDOW_MIN_W, design::size::WINDOW_MIN_H)),
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

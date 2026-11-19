//! Raw mode and the alternate screen, restored on every way out.
//!
//! A normal exit, a panic and the exit watchdog all call `restore`. While the
//! TUI runs, a panic on another thread, or one a dispatch catches, goes only to
//! `debug.log`: printed to stderr it would draw over the screen.

use std::cell::Cell;
use std::io::{self, Stdout, Write};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Once;
use std::sync::atomic::{AtomicBool, Ordering};

use crossterm::event::{DisableBracketedPaste, EnableBracketedPaste};
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use crossterm::{cursor, execute};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;

use crate::app::logging::{install_logging, log_panic};

pub type Term = Terminal<CrosstermBackend<Stdout>>;

static ACTIVE: AtomicBool = AtomicBool::new(false);
static HOOK: Once = Once::new();
/// During an error loop the host logs a count; a backtrace per panic would fill the disk.
static QUIET_PANICS: AtomicBool = AtomicBool::new(false);

thread_local! {
    static CATCHING: Cell<bool> = const { Cell::new(false) };
}

fn install_hook() {
    HOOK.call_once(|| {
        install_logging();
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |panic| {
            let quiet = ACTIVE.load(Ordering::SeqCst)
                && (CATCHING.get() || std::thread::current().name() != Some("main"));
            if quiet {
                if !QUIET_PANICS.load(Ordering::SeqCst) {
                    log_panic(panic);
                }
                return;
            }
            restore();
            previous(panic);
        }));
    });
}

pub fn set_quiet_panics(quiet: bool) {
    QUIET_PANICS.store(quiet, Ordering::SeqCst);
}

/// Enter raw mode and the alternate screen.
pub fn enter() -> io::Result<Term> {
    install_hook();
    enable_raw_mode()?;
    ACTIVE.store(true, Ordering::SeqCst);
    let mut stdout = io::stdout();
    if let Err(e) = execute!(
        stdout,
        EnterAlternateScreen,
        EnableBracketedPaste,
        cursor::Hide
    ) {
        restore();
        return Err(e);
    }
    let terminal = Terminal::new(CrosstermBackend::new(stdout));
    if terminal.is_err() {
        restore();
    }
    terminal
}

/// Leave the alternate screen and raw mode. Safe to call twice, from any thread.
pub fn restore() {
    if !ACTIVE.swap(false, Ordering::SeqCst) {
        return;
    }
    let mut stdout = io::stdout();
    let _ = execute!(
        stdout,
        LeaveAlternateScreen,
        DisableBracketedPaste,
        cursor::Show
    );
    let _ = disable_raw_mode();
    let _ = stdout.flush();
}

/// Run `body`, turning a panic into its message. The hook logs the backtrace.
pub fn catching<R>(body: impl FnOnce() -> R) -> Result<R, String> {
    CATCHING.set(true);
    let result = catch_unwind(AssertUnwindSafe(body));
    CATCHING.set(false);
    result.map_err(|payload| {
        payload
            .downcast_ref::<&str>()
            .map(|text| (*text).to_string())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "panic".into())
    })
}

//! Moving the window from the title bar row, on macOS.
//!
//! The window sets `app_owns_titlebar_drag`, so AppKit adds no double-click
//! delay to the buttons in that row. On macOS 27 that setting also makes
//! `performWindowDragWithEvent` do nothing, so gpui's `start_window_move`
//! cannot move the window. This module moves it by its frame instead: it
//! keeps the pointer and the window origin from the press, and on each move
//! sets the origin by the distance the pointer went.

// objc 0.2's msg_send! tests `feature = "cargo-clippy"`, which this crate
// does not declare. It is gpui's own version, so the warning stays here.
#![allow(unexpected_cfgs)]

use cocoa::base::{id, nil};
use cocoa::foundation::{NSPoint, NSRect};
use objc::{class, msg_send, sel, sel_impl};

/// A press that may move the window: where the pointer and the window were.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Grab {
    window: usize,
    pointer: (f64, f64),
    origin: (f64, f64),
}

/// The press now being handled. None when AppKit gives no event or window.
pub fn grab() -> Option<Grab> {
    // SAFETY: called from a gpui mouse-down handler, on the main thread,
    // while AppKit dispatches that mouse-down. Each object is checked for nil.
    unsafe {
        let app: id = msg_send![class!(NSApplication), sharedApplication];
        let event: id = msg_send![app, currentEvent];
        if event == nil {
            return None;
        }
        let window: id = msg_send![event, window];
        if window == nil {
            return None;
        }
        let frame: NSRect = msg_send![window, frame];
        let pointer: NSPoint = msg_send![class!(NSEvent), mouseLocation];
        Some(Grab {
            window: window as usize,
            pointer: (pointer.x, pointer.y),
            origin: (frame.origin.x, frame.origin.y),
        })
    }
}

/// Move the window so it keeps its place under the pointer.
pub fn follow(grab: Grab) {
    // SAFETY: the main thread. The window pointer came from AppKit at the
    // press, and the window is open: its own view sends the move.
    unsafe {
        let pointer: NSPoint = msg_send![class!(NSEvent), mouseLocation];
        let (x, y) = moved_origin(grab, (pointer.x, pointer.y));
        let window = grab.window as id;
        let _: () = msg_send![window, setFrameOrigin: NSPoint::new(x, y)];
    }
}

/// The window origin after the pointer went from the press to `pointer`.
fn moved_origin(grab: Grab, pointer: (f64, f64)) -> (f64, f64) {
    (
        grab.origin.0 + pointer.0 - grab.pointer.0,
        grab.origin.1 + pointer.1 - grab.pointer.1,
    )
}

#[cfg(test)]
mod tests {
    use super::{Grab, moved_origin};

    #[test]
    fn the_window_moves_as_far_as_the_pointer() {
        let grab = Grab {
            window: 0,
            pointer: (500.0, 300.0),
            origin: (100.0, 200.0),
        };
        assert_eq!(moved_origin(grab, (560.0, 260.0)), (160.0, 160.0));
        assert_eq!(moved_origin(grab, (500.0, 300.0)), (100.0, 200.0));
    }
}

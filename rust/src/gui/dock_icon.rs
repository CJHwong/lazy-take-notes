//! The Dock icon of a run outside the `.app`. macOS reads an app's icon from
//! its bundle, so `cargo run` and a bare binary otherwise show the generic
//! executable icon. The `.app` keeps the icon its bundle names.

// objc 0.2's msg_send! tests `feature = "cargo-clippy"`, which this crate
// does not declare. It is gpui's own version, so the warning stays here.
#![allow(unexpected_cfgs)]

use cocoa::base::{id, nil};
use cocoa::foundation::NSData;
use objc::{class, msg_send, sel, sel_impl};

/// The bundle folder an executable sits in when it runs from an `.app`.
const BUNDLE_EXECUTABLES: &str = ".app/Contents/MacOS/";

/// Set the Dock icon from `assets/icon.png` unless the app runs from its
/// bundle. Call it on the main thread once the application exists.
pub fn set_outside_bundle() {
    if runs_from_bundle() {
        return;
    }
    let png = include_bytes!("../../../assets/icon.png");
    // SAFETY: the main thread, after NSApplication exists. Each object comes
    // from AppKit and is checked for nil before use.
    unsafe {
        let data = NSData::dataWithBytes_length_(nil, png.as_ptr().cast(), png.len() as u64);
        let image: id = msg_send![class!(NSImage), alloc];
        let image: id = msg_send![image, initWithData: data];
        if image == nil {
            tracing::warn!(target: "ltn.gui", "The Dock icon did not decode; the default icon stays");
            return;
        }
        let app: id = msg_send![class!(NSApplication), sharedApplication];
        let _: () = msg_send![app, setApplicationIconImage: image];
    }
}

fn runs_from_bundle() -> bool {
    std::env::current_exe()
        .map(|path| path.to_string_lossy().contains(BUNDLE_EXECUTABLES))
        .unwrap_or(false)
}

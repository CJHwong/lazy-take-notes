//! Background work for a GPUI view. A worker or task result goes into a tokio
//! channel; the view awaits it in a foreground task and handles it on the main
//! thread, as the TUI host does with its message queue.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use gpui_kit::component::WindowExt as _;
use gpui_kit::component::notification::Notification;
use gpui_kit::*;
use tokio::sync::mpsc::{UnboundedReceiver, unbounded_channel};

use crate::app::events::Event;
use crate::app::session::{SessionIo, Severity, Text};
use crate::app::tasks::{Payload, Tasks};

/// What arrives for a view from its background work.
pub enum Incoming {
    Task(Payload),
    Worker(Event),
}

/// The `Tasks` a view hands to a runtime, and the channel its results come in on.
pub struct Bridge {
    pub tasks: Tasks,
    receiver: Option<UnboundedReceiver<Incoming>>,
}

impl Bridge {
    pub fn new() -> Self {
        let (sender, receiver) = unbounded_channel();
        let events = sender.clone();
        let tasks = Tasks::new(
            super::runtime(),
            Arc::new(move |payload| {
                let _ = sender.send(Incoming::Task(payload));
            }),
            Arc::new(move |event| events.send(Incoming::Worker(event)).is_ok()),
        );
        Self {
            tasks,
            receiver: Some(receiver),
        }
    }

    /// Hand each result to `handle` on the main thread until the view goes away.
    /// Keep the returned task: dropping it stops the pump.
    pub fn pump<V: 'static>(
        &mut self,
        window: &mut Window,
        cx: &mut Context<V>,
        handle: fn(&mut V, Incoming, &mut Window, &mut Context<V>),
    ) -> Task<()> {
        let Some(mut receiver) = self.receiver.take() else {
            return Task::ready(());
        };
        cx.spawn_in(window, async move |this, cx| {
            while let Some(incoming) = receiver.recv().await {
                let handled = this.update_in(cx, |view, window, cx| {
                    handle(view, incoming, window, cx);
                    cx.notify();
                });
                if handled.is_err() {
                    break;
                }
            }
        })
    }
}

/// The `SessionIo` a GUI view lends a runtime for one call.
pub struct GuiIo<'a> {
    pub window: &'a mut Window,
    pub cx: &'a mut App,
    pub tasks: &'a Tasks,
    /// The runtime asked the app to end.
    pub exited: bool,
}

impl<'a> GuiIo<'a> {
    pub fn new(window: &'a mut Window, cx: &'a mut App, tasks: &'a Tasks) -> Self {
        Self {
            window,
            cx,
            tasks,
            exited: false,
        }
    }
}

impl SessionIo for GuiIo<'_> {
    fn notify(&mut self, text: String, severity: Severity, timeout: Duration) {
        notify(self.window, self.cx, text, severity, timeout);
    }

    fn tasks(&self) -> Tasks {
        self.tasks.clone()
    }

    fn exit(&mut self) {
        self.exited = true;
    }

    /// The window names its buttons, not keys, and calls the digest the notes.
    fn text(&self, text: Text) -> &'static str {
        match text {
            Text::FinalReady => "Final notes ready.",
            Text::FinalFailed => "Final notes failed. Update Notes tries again.",
            Text::Digesting => "Updating notes…",
            Text::FinalDigesting => "Writing final notes…",
            Text::ManualHint => "Update Notes writes the notes now.",
            Text::AlreadyDigesting => "The notes are already updating.",
            Text::NothingToDigest => "No new lines for the notes yet.",
            Text::QuitAfterDigest => "Finishing when the notes are done.",
        }
    }
}

/// The id type of the notices this app shows.
struct Notice;

static NEXT_NOTICE: AtomicU64 = AtomicU64::new(1);

/// A notice that closes after `timeout`, as the TUI's toasts do.
pub fn notify(
    window: &mut Window,
    cx: &mut App,
    text: String,
    severity: Severity,
    timeout: Duration,
) {
    let id = NEXT_NOTICE.fetch_add(1, Ordering::Relaxed);
    let key = ElementId::Integer(id);
    let note = match severity {
        Severity::Information => Notification::info(text),
        Severity::Warning => Notification::warning(text),
        Severity::Error => Notification::error(text),
    };
    window.push_notification(note.id1::<Notice>(key.clone()).autohide(false), cx);
    let handle = window.window_handle();
    cx.spawn(async move |cx| {
        cx.background_executor().timer(timeout).await;
        let _ = cx.update_window(handle, |_, window, cx| {
            window.remove_notification1::<Notice>(key, cx);
        });
    })
    .detach();
}

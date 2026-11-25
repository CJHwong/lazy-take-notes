//! Background work for a front end: async tasks, threads and worker posters.
//! Each result goes back through the closures the front end gives, so this
//! module knows nothing about the screen that waits for it.

use std::any::Any;
use std::cell::Cell;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

use tokio::sync::mpsc::unbounded_channel;
use tracing::error;

use crate::app::events::{Event, Poster};
use crate::app::workers::audio::panic_text;

/// A task or thread result, typed by the code that started it.
pub type Payload = Box<dyn Any + Send>;

thread_local! {
    static CATCHING: Cell<bool> = const { Cell::new(false) };
}

/// True while `catching` runs on this thread. The panic hook then only logs.
pub fn is_catching() -> bool {
    CATCHING.get()
}

/// Run `body`, turning a panic into its message. The hook logs the backtrace.
pub fn catching<R>(body: impl FnOnce() -> R) -> Result<R, String> {
    CATCHING.set(true);
    let result = catch_unwind(AssertUnwindSafe(body));
    CATCHING.set(false);
    result.map_err(|payload| panic_text(&payload))
}

/// Starts work and hands each result to the owner's `on_result`, and each
/// worker event to its `on_event`.
#[derive(Clone)]
pub struct Tasks {
    pub runtime: tokio::runtime::Handle,
    on_result: Arc<dyn Fn(Payload) + Send + Sync>,
    on_event: Arc<dyn Fn(Event) -> bool + Send + Sync>,
}

impl Tasks {
    /// `on_event` returns false once nobody listens, which ends a poster's forwarding.
    pub fn new(
        runtime: tokio::runtime::Handle,
        on_result: Arc<dyn Fn(Payload) + Send + Sync>,
        on_event: Arc<dyn Fn(Event) -> bool + Send + Sync>,
    ) -> Self {
        Self {
            runtime,
            on_result,
            on_event,
        }
    }

    /// Run `future` on the runtime; its output goes to `on_result`.
    pub fn spawn<T, F>(&self, future: F) -> tokio::task::AbortHandle
    where
        T: Send + 'static,
        F: Future<Output = T> + Send + 'static,
    {
        let on_result = self.on_result.clone();
        self.runtime
            .spawn(async move { on_result(Box::new(future.await)) })
            .abort_handle()
    }

    /// Like `spawn`, but a panic inside `future` comes back as `on_panic(message)`,
    /// so the owner that waits for the result is never left waiting. A Textual
    /// worker that raised did the same through its error handler. The panic hook
    /// logs the stack.
    pub fn spawn_or<T, F, P>(&self, future: F, on_panic: P) -> tokio::task::AbortHandle
    where
        T: Send + 'static,
        F: Future<Output = T> + Send + 'static,
        P: FnOnce(String) -> T + Send + 'static,
    {
        /// Aborting the outer task drops this, which aborts the inner one.
        struct AbortOnDrop(tokio::task::AbortHandle);
        impl Drop for AbortOnDrop {
            fn drop(&mut self) {
                self.0.abort();
            }
        }
        let inner = self.runtime.spawn(future);
        let guard = AbortOnDrop(inner.abort_handle());
        let on_result = self.on_result.clone();
        self.runtime
            .spawn(async move {
                let _guard = guard;
                let output = match inner.await {
                    Ok(output) => output,
                    Err(e) if e.is_panic() => on_panic(panic_text(&e.into_panic())),
                    Err(_) => return,
                };
                on_result(Box::new(output));
            })
            .abort_handle()
    }

    /// Run `job` on its own thread; its output goes to `on_result`.
    pub fn spawn_thread<T, F>(&self, name: &str, job: F)
    where
        T: Send + 'static,
        F: FnOnce() -> T + Send + 'static,
    {
        let on_result = self.on_result.clone();
        let spawned = std::thread::Builder::new()
            .name(name.into())
            .spawn(move || on_result(Box::new(job())));
        if let Err(e) = spawned {
            error!(target: "ltn.app", "Cannot start thread {name}: {e}");
        }
    }

    /// Like `spawn_thread`, but a panic inside `job` comes back as
    /// `on_panic(message)`, as `spawn_or` does for a task. The panic hook logs
    /// the stack.
    pub fn spawn_thread_or<T, F, P>(&self, name: &str, job: F, on_panic: P)
    where
        T: Send + 'static,
        F: FnOnce() -> T + Send + 'static,
        P: FnOnce(String) -> T + Send + 'static,
    {
        self.spawn_thread(name, move || catching(job).unwrap_or_else(on_panic));
    }

    /// A poster whose events go to `on_event`.
    pub fn poster(&self) -> Poster {
        let (events, mut received) = unbounded_channel();
        let on_event = self.on_event.clone();
        self.runtime.spawn(async move {
            while let Some(event) = received.recv().await {
                if !on_event(event) {
                    break;
                }
            }
        });
        Poster::new(events)
    }
}

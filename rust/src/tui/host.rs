//! The screen stack and its event loop: the Rust form of Python's `SessionHostApp`.
//!
//! One loop merges terminal input, worker events, spawned task results and OS
//! signals. Keys go to the top screen; every other message goes down the stack
//! until a screen takes it. A panic in a handler goes to `debug.log` with a toast,
//! and the app keeps running: a meeting recording must not end because of a bug.

use std::any::Any;
use std::collections::VecDeque;
use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crossterm::event::{
    Event as TermEvent, EventStream, KeyCode, KeyEvent, KeyModifiers, MouseButton, MouseEvent,
    MouseEventKind,
};
use futures_util::StreamExt;
use ratatui::Frame;
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::Line;
use ratatui::widgets::{Block, Clear, Paragraph, Wrap};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender, unbounded_channel};
use tracing::{error, info, warn};

use crate::app::events::{Event, Poster};
use crate::app::logging::{critical, see_debug_log};
use crate::app::workers::audio::panic_text;
use crate::tui::keys::key_name;
use crate::tui::terminal::{self, Term, catching};
use crate::tui::theme::Theme;

/// How long a shutdown signal waits for the audio worker to flush its last chunk.
pub const SHUTDOWN_GRACE: Duration = Duration::from_secs(3);
/// Extra time for the app's own shutdown before the watchdog ends the process.
pub const FORCE_EXIT_AFTER: Duration = Duration::from_secs(5);
/// More unexpected errors than this inside the window means an error loop.
pub const ERROR_BURST_LIMIT: usize = 5;
pub const ERROR_BURST_WINDOW: Duration = Duration::from_secs(10);
/// During an error loop, log a count this often instead of each error.
pub const STORM_LOG_INTERVAL: Duration = Duration::from_secs(60);
const TICK: Duration = Duration::from_millis(250);
pub const NOTIFICATION_TIMEOUT: Duration = Duration::from_secs(5);

pub type Payload = Box<dyn Any + Send>;

/// Everything that reaches the loop besides a key.
pub enum Msg {
    Worker(Event),
    Task(Payload),
    Signal(i32),
    /// A worker or task result from the screen with this id. It is dropped
    /// when that screen is gone, as Textual drops a message to a removed screen.
    From(u64, Box<Msg>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Information,
    Warning,
    Error,
}

struct Toast {
    title: String,
    text: String,
    severity: Severity,
    until: Instant,
}

/// What a screen sees of the host.
pub struct Ctx {
    pub theme: Theme,
    toasts: Vec<Toast>,
    sender: UnboundedSender<Msg>,
    pub runtime: tokio::runtime::Handle,
    exit_at: Option<Instant>,
    exit_now: bool,
    suspend_job: Option<SuspendJob>,
    /// The id of the screen the host is calling now.
    current: u64,
    /// What each widget drew where in the last frame, for the mouse.
    hits: std::cell::RefCell<Vec<Hit>>,
    /// The id of the screen the host is drawing now.
    drawing: std::cell::Cell<u64>,
}

/// The widget a click or the wheel landed on: the key its screen gave it while
/// drawing, and a row or item index within it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Target {
    pub key: String,
    pub index: usize,
    /// Where the widget was drawn.
    pub rect: Rect,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MouseKind {
    Click,
    WheelUp,
    WheelDown,
}

/// A click or a wheel step, with the widget under the pointer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MouseInput {
    pub kind: MouseKind,
    pub column: u16,
    pub row: u16,
    pub target: Option<Target>,
}

impl MouseInput {
    /// The target's index when the pointer is on the widget named `key`.
    pub fn on(&self, key: &str) -> Option<usize> {
        self.target
            .as_ref()
            .filter(|target| target.key == key)
            .map(|target| target.index)
    }
}

#[derive(Debug, Clone)]
struct Hit {
    screen: u64,
    target: Target,
}

/// The screen id that marks the host's own toasts.
const TOAST_SCREEN: u64 = u64::MAX;

type SuspendJob = (Box<dyn FnOnce() -> Payload + Send>, u64);

impl Ctx {
    pub fn notify(&mut self, text: impl Into<String>) {
        self.notify_with(text, Severity::Information, NOTIFICATION_TIMEOUT, "");
    }

    pub fn warn(&mut self, text: impl Into<String>, timeout: Duration) {
        self.notify_with(text, Severity::Warning, timeout, "");
    }

    pub fn error(&mut self, text: impl Into<String>, timeout: Duration) {
        self.notify_with(text, Severity::Error, timeout, "");
    }

    pub fn notify_with(
        &mut self,
        text: impl Into<String>,
        severity: Severity,
        timeout: Duration,
        title: &str,
    ) {
        self.toasts.push(Toast {
            title: title.into(),
            text: text.into(),
            severity,
            until: Instant::now() + timeout,
        });
    }

    /// Texts of the toasts on screen, oldest first.
    /// Mark `rect` as the widget `key` (item `index`) for clicks and the wheel.
    /// A widget drawn later sits on top of one drawn before it.
    pub fn hit(&self, rect: Rect, key: &str, index: usize) {
        if rect.width == 0 || rect.height == 0 {
            return;
        }
        self.hits.borrow_mut().push(Hit {
            screen: self.drawing.get(),
            target: Target {
                key: key.into(),
                index,
                rect,
            },
        });
    }

    fn target_at(&self, screen: u64, column: u16, row: u16) -> Option<Target> {
        let point = ratatui::layout::Position::new(column, row);
        self.hits
            .borrow()
            .iter()
            .rev()
            .find(|hit| hit.screen == screen && hit.target.rect.contains(point))
            .map(|hit| hit.target.clone())
    }

    pub fn toast_texts(&self) -> Vec<String> {
        self.toasts.iter().map(|toast| toast.text.clone()).collect()
    }

    /// Run `future` on the runtime; its output comes back as `Msg::Task`.
    pub fn spawn<T, F>(&self, future: F) -> tokio::task::AbortHandle
    where
        T: Send + 'static,
        F: Future<Output = T> + Send + 'static,
    {
        let sender = self.sender.clone();
        let from = self.current;
        self.runtime
            .spawn(async move {
                let _ = sender.send(Msg::From(from, Box::new(Msg::Task(Box::new(future.await)))));
            })
            .abort_handle()
    }

    /// Like `spawn`, but a panic inside `future` comes back as `on_panic(message)`,
    /// so the screen that waits for the result is never left waiting. A Textual
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
        let sender = self.sender.clone();
        let from = self.current;
        self.runtime
            .spawn(async move {
                let _guard = guard;
                let output = match inner.await {
                    Ok(output) => output,
                    Err(e) if e.is_panic() => on_panic(panic_text(&e.into_panic())),
                    Err(_) => return,
                };
                let _ = sender.send(Msg::From(from, Box::new(Msg::Task(Box::new(output)))));
            })
            .abort_handle()
    }

    /// Run `job` on its own thread; its output comes back as `Msg::Task`.
    pub fn spawn_thread<T, F>(&self, name: &str, job: F)
    where
        T: Send + 'static,
        F: FnOnce() -> T + Send + 'static,
    {
        let sender = self.sender.clone();
        let from = self.current;
        let spawned = std::thread::Builder::new()
            .name(name.into())
            .spawn(move || {
                let _ = sender.send(Msg::From(from, Box::new(Msg::Task(Box::new(job())))));
            });
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
        self.spawn_thread(name, move || {
            crate::tui::terminal::catching(job).unwrap_or_else(on_panic)
        });
    }

    /// A sender a thread keeps to post several results, each as `Msg::Task`.
    pub fn task_sender(&self) -> TaskSender {
        TaskSender(self.sender.clone(), self.current)
    }

    /// A poster whose events arrive as `Msg::Worker`.
    pub fn poster(&self) -> Poster {
        let (events, mut received) = unbounded_channel();
        let sender = self.sender.clone();
        let from = self.current;
        self.runtime.spawn(async move {
            while let Some(event) = received.recv().await {
                if sender
                    .send(Msg::From(from, Box::new(Msg::Worker(event))))
                    .is_err()
                {
                    break;
                }
            }
        });
        Poster::new(events)
    }

    pub fn set_theme(&mut self, name: &str) {
        self.theme = Theme::load(name);
    }

    /// Exit the app after `delay`, unless something exits it first.
    pub fn exit_after(&mut self, delay: Duration) {
        self.exit_at = Some(Instant::now() + delay);
    }

    pub fn exit(&mut self) {
        self.exit_now = true;
    }

    /// Run `job` with the terminal given back (an editor), after this event.
    /// Its output comes back as `Msg::Task`.
    pub fn suspend<T, F>(&mut self, job: F)
    where
        T: Send + 'static,
        F: FnOnce() -> T + Send + 'static,
    {
        self.suspend_job = Some((Box::new(move || Box::new(job()) as Payload), self.current));
    }

    /// The job a screen asked to run suspended, for tests.
    pub fn take_suspend_job(&mut self) -> Option<Box<dyn FnOnce() -> Payload + Send>> {
        self.suspend_job.take().map(|(job, _)| job)
    }
}

/// Posts payloads to the loop from any thread. A post after the loop ended is dropped.
#[derive(Clone)]
pub struct TaskSender(UnboundedSender<Msg>, u64);

impl TaskSender {
    pub fn post<T: Send + 'static>(&self, payload: T) {
        let _ = self
            .0
            .send(Msg::From(self.1, Box::new(Msg::Task(Box::new(payload)))));
    }
}

/// What a screen asks the host to do after an event.
pub enum Action {
    None,
    Push(Box<dyn Screen>),
    /// Remove this screen and hand the result to the one below. From a screen
    /// under a modal, the modals above it close first, as Textual's `dismiss`
    /// after popping them.
    Pop(Option<Payload>),
    /// Remove this screen and put another in its place.
    Replace(Box<dyn Screen>),
}

pub trait Screen {
    fn render(&mut self, frame: &mut Frame, area: Rect, ctx: &Ctx);

    /// A key press, by Textual's name.
    fn key(&mut self, key: &KeyEvent, name: &str, ctx: &mut Ctx) -> Action;

    fn paste(&mut self, _text: &str, _ctx: &mut Ctx) -> Action {
        Action::None
    }

    /// A click or a wheel step. Unless a screen handles it, the wheel moves as
    /// up and down do, and a click does nothing.
    fn mouse(&mut self, input: &MouseInput, ctx: &mut Ctx) -> Action {
        let (code, name) = match input.kind {
            MouseKind::WheelUp => (KeyCode::Up, "up"),
            MouseKind::WheelDown => (KeyCode::Down, "down"),
            MouseKind::Click => return Action::None,
        };
        self.key(&KeyEvent::new(code, KeyModifiers::NONE), name, ctx)
    }

    /// A worker event or task result. Give it back when this screen does not own it.
    fn message(&mut self, msg: Msg, _ctx: &mut Ctx) -> (Option<Msg>, Action) {
        (Some(msg), Action::None)
    }

    fn tick(&mut self, _ctx: &mut Ctx) -> Action {
        Action::None
    }

    /// The screen this one pushed was popped with `result`.
    fn resumed(&mut self, _result: Option<Payload>, _ctx: &mut Ctx) -> Action {
        Action::None
    }

    /// Called once, when the screen is pushed.
    fn mounted(&mut self, _ctx: &mut Ctx) -> Action {
        Action::None
    }

    /// A modal draws over the dimmed screen below it.
    fn is_modal(&self) -> bool {
        false
    }

    /// Closes on any key, and Tab still moves the focus of the screen below.
    fn passes_focus_keys(&self) -> bool {
        false
    }

    /// A live session: an error toast says it continues.
    fn hosts_session(&self) -> bool {
        false
    }

    /// Stop recording for a shutdown. True when the app must wait for the audio flush.
    fn stop_for_shutdown(&mut self, _ctx: &mut Ctx) -> bool {
        false
    }

    /// What ctrl+c says instead of the generic quit hint.
    fn quit_hint(&self) -> Option<String> {
        None
    }

    /// The terminal window got the focus back (Textual's `AppFocus`).
    fn app_focus(&mut self, _ctx: &mut Ctx) -> Action {
        Action::None
    }
}

/// Counts errors to tell a single fault from an error loop.
#[derive(Default)]
struct ErrorTracker {
    times: VecDeque<Instant>,
    storm: bool,
    storm_errors: u32,
    storm_logged_at: Option<Instant>,
}

impl ErrorTracker {
    fn record(&mut self, message: &str, session: bool, ctx: &mut Ctx) {
        let now = Instant::now();
        if self.storm {
            self.count_storm_error(message, now);
            return;
        }
        critical(&format!(
            "Unexpected error; the session continues: {message}"
        ));
        self.times.push_back(now);
        if self.times.len() > ERROR_BURST_LIMIT + 1 {
            self.times.pop_front();
        }
        let first = self.times[0];
        let storm = self.times.len() > ERROR_BURST_LIMIT && now - first <= ERROR_BURST_WINDOW;
        if !storm {
            let tail = if session {
                " The session continues."
            } else {
                ""
            };
            ctx.error(
                format!("Unexpected error: {message}{}.{tail}", see_debug_log()),
                Duration::from_secs(15),
            );
            return;
        }
        self.storm = true;
        terminal::set_quiet_panics(true);
        self.storm_logged_at = Some(now);
        critical(&format!(
            "Errors repeat {} times in {:.0}s; the recording continues",
            self.times.len(),
            (now - first).as_secs_f64()
        ));
        ctx.error(
            format!(
                "Errors keep repeating{}. The audio recording continues; press s to stop and save.",
                see_debug_log()
            ),
            Duration::from_secs(3600),
        );
    }

    /// A traceback per error would fill the disk in a long meeting. Log a count instead.
    fn count_storm_error(&mut self, message: &str, now: Instant) {
        self.storm_errors += 1;
        let logged_at = self.storm_logged_at.unwrap_or(now);
        if now - logged_at < STORM_LOG_INTERVAL {
            return;
        }
        critical(&format!(
            "{} more errors in {:.0}s; the last: {message}",
            self.storm_errors,
            (now - logged_at).as_secs_f64()
        ));
        self.storm_errors = 0;
        self.storm_logged_at = Some(now);
    }
}

pub struct Host {
    stack: Vec<Box<dyn Screen>>,
    /// One id per screen in `stack`, never reused.
    ids: Vec<u64>,
    next_id: u64,
    /// The screen that started the suspended job.
    suspended_by: u64,
    ctx: Ctx,
    receiver: UnboundedReceiver<Msg>,
    errors: ErrorTracker,
    result: Option<Payload>,
    watchdog_armed: Arc<AtomicBool>,
}

impl Host {
    pub fn new(theme: &str, runtime: tokio::runtime::Handle) -> Self {
        let (sender, receiver) = unbounded_channel();
        Self {
            stack: Vec::new(),
            ids: Vec::new(),
            next_id: 1,
            suspended_by: 0,
            ctx: Ctx {
                theme: Theme::load(theme),
                toasts: Vec::new(),
                sender,
                runtime,
                exit_at: None,
                exit_now: false,
                suspend_job: None,
                current: 0,
                hits: std::cell::RefCell::new(Vec::new()),
                drawing: std::cell::Cell::new(0),
            },
            receiver,
            errors: ErrorTracker::default(),
            result: None,
            watchdog_armed: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn ctx(&mut self) -> &mut Ctx {
        &mut self.ctx
    }

    pub fn depth(&self) -> usize {
        self.stack.len()
    }

    pub fn finished(&self) -> bool {
        self.stack.is_empty()
            || self.ctx.exit_now
            || self.ctx.exit_at.is_some_and(|at| Instant::now() >= at)
    }

    /// Handle every message already queued. Returns how many there were.
    /// The run loop does this itself; tests call it to step the screens.
    pub fn pump(&mut self) -> usize {
        let mut handled = 0;
        while let Ok(msg) = self.receiver.try_recv() {
            self.message(msg);
            handled += 1;
        }
        handled
    }

    /// The value the last screen popped with.
    pub fn take_result(&mut self) -> Option<Payload> {
        self.result.take()
    }

    fn session_running(&self) -> bool {
        self.stack.iter().any(|screen| screen.hosts_session())
    }

    /// Run `body` against the top screen; a panic becomes an error toast.
    fn guarded(&mut self, body: impl FnOnce(&mut dyn Screen, &mut Ctx) -> Action) -> Action {
        let Some(screen) = self.stack.last_mut() else {
            return Action::None;
        };
        let ctx = &mut self.ctx;
        ctx.current = self.ids.last().copied().unwrap_or(0);
        match catching(|| body(screen.as_mut(), ctx)) {
            Ok(action) => action,
            Err(message) => {
                let session = self.session_running();
                self.errors.record(&message, session, &mut self.ctx);
                Action::None
            }
        }
    }

    pub fn push(&mut self, screen: Box<dyn Screen>) {
        self.stack.push(screen);
        self.ids.push(self.next_id);
        self.next_id += 1;
        let action = self.guarded(|screen, ctx| screen.mounted(ctx));
        self.apply(action);
    }

    fn apply(&mut self, action: Action) {
        match action {
            Action::None => {}
            Action::Push(screen) => self.push(screen),
            Action::Replace(screen) => {
                self.stack.pop();
                self.ids.pop();
                self.push(screen);
            }
            Action::Pop(result) => {
                self.stack.pop();
                self.ids.pop();
                if self.stack.is_empty() {
                    self.result = result;
                    return;
                }
                let action = self.guarded(|screen, ctx| screen.resumed(result, ctx));
                self.apply(action);
            }
        }
    }

    /// Apply an action from the screen at `index`, which a modal may cover.
    /// A push goes on top, as Textual's `push_screen` does from any screen.
    fn apply_from(&mut self, index: usize, action: Action) {
        if index >= self.stack.len() {
            return;
        }
        match action {
            Action::Pop(result) if index + 1 < self.stack.len() => {
                self.stack.truncate(index + 1);
                self.ids.truncate(index + 1);
                self.apply(Action::Pop(result));
            }
            Action::Replace(_) if index + 1 < self.stack.len() => {
                warn!(target: "ltn.app", "a covered screen asked to be replaced; ignored");
            }
            action => self.apply(action),
        }
    }

    /// Handle one key press.
    pub fn key(&mut self, key: &KeyEvent) {
        let Some(name) = key_name(key) else {
            return;
        };
        if name == "ctrl+q" {
            info!(target: "ltn.app", "ctrl+q; ending the session without a final digest");
            self.end_sessions();
            return;
        }
        if name == "ctrl+c" {
            self.help_quit();
            return;
        }
        // Tab is an app binding in Textual: a notice that closes on any key
        // lets it through, so the screen below moves its focus too.
        let passes = matches!(name.as_str(), "tab" | "shift+tab")
            && self
                .stack
                .last()
                .is_some_and(|screen| screen.passes_focus_keys());
        let action = self.guarded(|screen, ctx| screen.key(key, &name, ctx));
        self.apply(action);
        if passes {
            let action = self.guarded(|screen, ctx| screen.key(key, &name, ctx));
            self.apply(action);
        }
    }

    fn help_quit(&mut self) {
        let hint = self
            .stack
            .iter()
            .rev()
            .find_map(|screen| screen.quit_hint());
        let text = hint.unwrap_or_else(|| "Press ctrl+q to quit the app".into());
        self.ctx.notify_with(
            text,
            Severity::Information,
            NOTIFICATION_TIMEOUT,
            "Do you want to quit?",
        );
    }

    /// A mouse event from the terminal: a click closes the toast under it, or
    /// goes with the wheel to the top screen with the widget under the pointer.
    pub fn mouse(&mut self, event: &MouseEvent) {
        let kind = match event.kind {
            MouseEventKind::Down(MouseButton::Left) => MouseKind::Click,
            MouseEventKind::ScrollUp => MouseKind::WheelUp,
            MouseEventKind::ScrollDown => MouseKind::WheelDown,
            _ => return,
        };
        let (column, row) = (event.column, event.row);
        if kind == MouseKind::Click
            && let Some(toast) = self.ctx.target_at(TOAST_SCREEN, column, row)
        {
            if toast.index < self.ctx.toasts.len() {
                self.ctx.toasts.remove(toast.index);
            }
            return;
        }
        let top = self.ids.last().copied().unwrap_or(0);
        let input = MouseInput {
            kind,
            column,
            row,
            target: self.ctx.target_at(top, column, row),
        };
        let action = self.guarded(|screen, ctx| screen.mouse(&input, ctx));
        self.apply(action);
    }

    pub fn paste(&mut self, text: &str) {
        let action = self.guarded(|screen, ctx| screen.paste(text, ctx));
        self.apply(action);
    }

    /// Offer a message to each screen from the top until one takes it.
    pub fn message(&mut self, msg: Msg) {
        if let Msg::Signal(signum) = msg {
            self.on_signal(signum);
            return;
        }
        let msg = match msg {
            Msg::From(id, inner) if self.ids.contains(&id) => *inner,
            Msg::From(..) => return,
            other => other,
        };
        let mut pending = Some(msg);
        for index in (0..self.stack.len()).rev() {
            let Some(msg) = pending.take() else {
                break;
            };
            // A pop by a screen above can shrink the stack under this loop.
            let Some(screen) = self.stack.get_mut(index) else {
                pending = Some(msg);
                continue;
            };
            let ctx = &mut self.ctx;
            ctx.current = self.ids[index];
            let outcome = catching(|| screen.message(msg, ctx));
            let action = match outcome {
                Ok((back, action)) => {
                    pending = back;
                    action
                }
                Err(message) => {
                    let session = self.session_running();
                    self.errors.record(&message, session, &mut self.ctx);
                    Action::None
                }
            };
            self.apply_from(index, action);
        }
    }

    pub fn tick(&mut self) {
        let now = Instant::now();
        self.ctx.toasts.retain(|toast| toast.until > now);
        let mut index = 0;
        while index < self.stack.len() {
            let screen = &mut self.stack[index];
            let ctx = &mut self.ctx;
            ctx.current = self.ids[index];
            let outcome = catching(|| screen.tick(ctx));
            match outcome {
                Ok(action) => self.apply_from(index, action),
                Err(message) => {
                    let session = self.session_running();
                    self.errors.record(&message, session, &mut self.ctx);
                }
            }
            index += 1;
        }
    }

    fn on_signal(&mut self, signum: i32) {
        warn!(
            target: "ltn.app",
            "Received {}; ending the session without a final digest",
            signal_name(signum)
        );
        arm_watchdog_once(&self.watchdog_armed, signum);
        self.end_sessions();
    }

    /// Stop every recording (audio flush, context save), then exit.
    pub fn end_sessions(&mut self) {
        let mut waiting = false;
        for screen in &mut self.stack {
            let ctx = &mut self.ctx;
            if catching(|| screen.stop_for_shutdown(ctx)).unwrap_or(false) {
                waiting = true;
            }
        }
        if waiting {
            self.ctx.exit_after(SHUTDOWN_GRACE);
        } else {
            self.ctx.exit();
        }
    }

    /// Draw the stack: the top full screen, the modals over it, then the toasts.
    pub fn draw(&mut self, frame: &mut Frame) {
        let area = frame.area();
        let background = self.ctx.theme.color("background");
        let base_style = Style::default()
            .bg(background)
            .fg(self.ctx.theme.color("foreground"));
        frame.render_widget(Block::default().style(base_style), area);
        self.ctx.hits.borrow_mut().clear();
        let base = self
            .stack
            .iter()
            .rposition(|screen| !screen.is_modal())
            .unwrap_or(0);
        for index in base..self.stack.len() {
            if index > base {
                dim(frame.buffer_mut(), area, background);
            }
            let screen = &mut self.stack[index];
            let ctx = &self.ctx;
            ctx.drawing.set(self.ids[index]);
            if let Err(message) = catching(|| screen.render(frame, area, ctx)) {
                let session = self.session_running();
                self.errors.record(&message, session, &mut self.ctx);
            }
        }
        self.ctx.drawing.set(TOAST_SCREEN);
        render_toasts(frame, area, &self.ctx);
        resolve_dim(frame.buffer_mut(), area);
    }

    /// Run until the stack empties or a screen exits the app.
    pub fn run(&mut self, terminal: &mut Term) -> std::io::Result<()> {
        let signals = watch_shutdown_signals(self.ctx.sender.clone(), self.watchdog_armed.clone());
        let runtime = self.ctx.runtime.clone();
        let outcome = runtime.block_on(self.event_loop(terminal));
        if let Some(watch) = signals {
            watch.close();
        }
        outcome
    }

    async fn event_loop(&mut self, terminal: &mut Term) -> std::io::Result<()> {
        // None once the terminal is gone (a closed window): keys stop, the loop goes on.
        let mut input = Some(EventStream::new());
        let mut ticks = tokio::time::interval(TICK);
        while !self.finished() {
            if let Some((job, by)) = self.ctx.suspend_job.take() {
                self.suspended_by = by;
                // The input reader must stop first, or it takes the editor's keys.
                drop(input);
                let payload = terminal::suspended(job)?;
                input = Some(EventStream::new());
                // `Terminal::clear` asks the terminal where the cursor is, and that
                // read times out right after the editor gives the terminal back.
                // Clear without it, and blank both buffers so the next draw
                // repaints every cell.
                crossterm::execute!(
                    std::io::stdout(),
                    crossterm::terminal::Clear(crossterm::terminal::ClearType::All)
                )?;
                terminal.swap_buffers();
                terminal.swap_buffers();
                self.message(Msg::From(self.suspended_by, Box::new(Msg::Task(payload))));
                continue;
            }
            // A closed window fails every write. The session must still stop and save,
            // so the frame is dropped and the loop goes on, as Python's terminal writer does.
            if let Err(e) = terminal.draw(|frame| self.draw(frame)) {
                warn!(target: "ltn.app", "Terminal output failed ({e}); frame dropped");
            }
            let next_key = async {
                match input.as_mut() {
                    Some(stream) => stream.next().await,
                    None => std::future::pending().await,
                }
            };
            tokio::select! {
                event = next_key => match event {
                    Some(Ok(event)) => self.terminal_event(event),
                    Some(Err(e)) => {
                        warn!(target: "ltn.app", "Terminal input failed ({e}); keys stop");
                        input = None;
                    }
                    None => {
                        warn!(target: "ltn.app", "Terminal input closed; keys stop");
                        input = None;
                    }
                },
                msg = self.receiver.recv() => {
                    if let Some(msg) = msg {
                        self.message(msg);
                    }
                    // Take what else waits before the next draw.
                    while let Ok(msg) = self.receiver.try_recv() {
                        self.message(msg);
                    }
                }
                _ = ticks.tick() => self.tick(),
            }
        }
        Ok(())
    }

    fn terminal_event(&mut self, event: TermEvent) {
        match event {
            TermEvent::Key(key) => self.key(&key),
            TermEvent::Paste(text) => self.paste(&text),
            TermEvent::Mouse(event) => self.mouse(&event),
            TermEvent::FocusGained => {
                let action = self.guarded(|screen, ctx| screen.app_focus(ctx));
                self.apply(action);
            }
            _ => {}
        }
    }
}

/// Run one screen on its own host, as Python's `ShellScreen.run()` does. Returns its result.
pub fn run_screen(
    screen: Box<dyn Screen>,
    theme: &str,
    runtime: tokio::runtime::Handle,
) -> std::io::Result<Option<Payload>> {
    let mut host = Host::new(theme, runtime);
    let mut terminal = terminal::enter()?;
    host.push(screen);
    let outcome = host.run(&mut terminal);
    terminal::restore();
    // `restore` showed the cursor. Ratatui's own drop shows it again and prints any
    // error to stderr, which panics when the window is already closed.
    std::mem::forget(terminal);
    outcome.map(|()| host.take_result())
}

fn signal_name(signum: i32) -> &'static str {
    match signum {
        signal_hook::consts::SIGTERM => "SIGTERM",
        signal_hook::consts::SIGHUP => "SIGHUP",
        signal_hook::consts::SIGINT => "SIGINT",
        _ => "signal",
    }
}

/// SIGTERM and SIGHUP (the window closed) end the session. A plain thread waits
/// for them, so a hung event loop still arms the exit watchdog.
fn watch_shutdown_signals(
    sender: UnboundedSender<Msg>,
    armed: Arc<AtomicBool>,
) -> Option<SignalWatch> {
    use signal_hook::consts::{SIGHUP, SIGTERM};
    watch_signals(&[SIGTERM, SIGHUP], sender, armed)
}

/// No handler (a signal the OS refuses) logs a warning and returns None: the
/// session runs on, as Python's host does when `signal.signal` raises.
fn watch_signals(
    list: &[i32],
    sender: UnboundedSender<Msg>,
    armed: Arc<AtomicBool>,
) -> Option<SignalWatch> {
    keep_default_actions(list);
    let mut signals = match signal_hook::iterator::Signals::new(list) {
        Ok(signals) => signals,
        Err(e) => {
            warn!(target: "ltn.app", "No handler for SIGTERM and SIGHUP: {e}");
            return None;
        }
    };
    let handle = signals.handle();
    let spawned = std::thread::Builder::new()
        .name("ltn-signals".into())
        .spawn(move || {
            for signum in signals.forever() {
                arm_watchdog_once(&armed, signum);
                if sender.send(Msg::Signal(signum)).is_err() {
                    break;
                }
            }
        });
    if let Err(e) = spawned {
        warn!(target: "ltn.app", "No signal thread: {e}");
        return None;
    }
    NO_HOST_LISTENS.store(false, Ordering::SeqCst);
    Some(SignalWatch(handle))
}

/// The host's signal handlers. `close` gives each signal its old action back.
struct SignalWatch(signal_hook::iterator::Handle);

impl SignalWatch {
    fn close(self) {
        self.0.close();
        NO_HOST_LISTENS.store(true, Ordering::SeqCst);
    }
}

/// While true, a signal whose action was the default before any host takes that
/// action again: Python's host restores the handlers it replaced. signal-hook
/// keeps its own handler installed for the next host in the process, so the
/// default comes back through this flag instead.
static NO_HOST_LISTENS: std::sync::LazyLock<Arc<AtomicBool>> =
    std::sync::LazyLock::new(|| Arc::new(AtomicBool::new(true)));

/// Before the first host replaces a signal's handler, note whether its action
/// was the default. A signal that was ignored (SIGHUP under nohup) stays ignored.
fn keep_default_actions(list: &[i32]) {
    static SEEN: std::sync::Mutex<Vec<i32>> = std::sync::Mutex::new(Vec::new());
    let mut seen = SEEN
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    for &signum in list {
        if seen.contains(&signum) {
            continue;
        }
        seen.push(signum);
        if has_default_action(signum)
            && let Err(e) =
                signal_hook::flag::register_conditional_default(signum, NO_HOST_LISTENS.clone())
        {
            warn!(target: "ltn.app", "Cannot restore the default action of signal {signum}: {e}");
        }
    }
}

fn has_default_action(signum: i32) -> bool {
    // SAFETY: a null new action only reads the current one into `current`.
    unsafe {
        let mut current: libc::sigaction = std::mem::zeroed();
        libc::sigaction(signum, std::ptr::null(), &mut current) == 0
            && current.sa_sigaction == libc::SIG_DFL
    }
}

/// End the process if the clean shutdown hangs. By then the audio flush and the
/// context save have had their grace period, so a hard exit loses nothing.
fn arm_watchdog_once(armed: &AtomicBool, signum: i32) {
    if armed.swap(true, Ordering::SeqCst) {
        return;
    }
    let spawned = std::thread::Builder::new()
        .name("ltn-exit-watchdog".into())
        .spawn(move || {
            std::thread::sleep(SHUTDOWN_GRACE + FORCE_EXIT_AFTER);
            error!(
                target: "ltn.app",
                "Shutdown after {} did not finish; forcing exit",
                signal_name(signum)
            );
            terminal::restore();
            std::process::exit(128 + signum);
        });
    if let Err(e) = spawned {
        error!(target: "ltn.app", "No exit watchdog: {e}");
    }
}

/// Textual's modal background: `$background 60%` over the screen below.
fn dim(buffer: &mut Buffer, area: Rect, background: Color) {
    let toward = |color: Color| match (color, background) {
        (Color::Rgb(r, g, b), Color::Rgb(br, bg, bb)) => {
            let mix = |from: u8, to: u8| {
                (f64::from(from) + (f64::from(to) - f64::from(from)) * 0.4) as u8
            };
            Color::Rgb(mix(r, br), mix(g, bg), mix(b, bb))
        }
        (other, _) => other,
    };
    for y in area.top()..area.bottom() {
        for x in area.left()..area.right() {
            if let Some(cell) = buffer.cell_mut((x, y)) {
                cell.fg = toward(cell.fg);
                cell.bg = toward(cell.bg);
            }
        }
    }
}

/// Toasts stack at the bottom right, newest lowest, as Textual's toast rack does.
fn render_toasts(frame: &mut Frame, area: Rect, ctx: &Ctx) {
    let theme = &ctx.theme;
    // Newest at the bottom right, older ones stacked above it. A click closes one.
    let width = 50.min(area.width.saturating_sub(2) / 2).max(24).min(area.width);
    let mut bottom = area.bottom().saturating_sub(1);
    for (index, toast) in ctx.toasts.iter().enumerate().rev() {
        let edge = match toast.severity {
            Severity::Information => "success",
            Severity::Warning => "warning",
            Severity::Error => "error",
        };
        let background = theme.color("surface");
        let lines: Vec<Line> = toast
            .text
            .lines()
            .map(|line| Line::from(line.to_string()))
            .collect();
        let inner_width = width.saturating_sub(4).max(1);
        let text_rows = crate::tui::text::wrapped_height(&lines, inner_width);
        let height = (text_rows + 2).min(bottom.saturating_sub(area.top()));
        if height < 3 {
            break;
        }
        let rect = Rect::new(
            area.right().saturating_sub(width + 1),
            bottom - height,
            width,
            height,
        );
        frame.render_widget(Clear, rect);
        let inner = crate::tui::widgets::rounded(
            frame,
            rect,
            &toast.title,
            theme.color(edge),
            background,
            theme,
        );
        let style = Style::default()
            .bg(background)
            .fg(theme.on_color("foreground", background));
        frame.render_widget(
            Paragraph::new(lines).style(style).wrap(Wrap { trim: false }),
            Rect::new(inner.x + 1, inner.y, inner_width, inner.height),
        );
        ctx.hit(rect, "toast", index);
        bottom = rect.top();
    }
}

/// Textual's `DIM_FACTOR`: dim text is its color 66% of the way from the background.
const DIM_FACTOR: f64 = 0.66;

/// Turn the dim attribute into a color, as Textual's output filter does after
/// compositing. A terminal's own dim differs from one terminal to the next.
fn resolve_dim(buffer: &mut ratatui::buffer::Buffer, area: Rect) {
    for y in area.top()..area.bottom() {
        for x in area.left()..area.right() {
            let Some(cell) = buffer.cell_mut((x, y)) else {
                continue;
            };
            if !cell.modifier.contains(Modifier::DIM) {
                continue;
            }
            let (Color::Rgb(fr, fg, fb), Color::Rgb(br, bg, bb)) = (cell.fg, cell.bg) else {
                continue;
            };
            let mix = |back: u8, fore: u8| {
                (f64::from(back) + (f64::from(fore) - f64::from(back)) * DIM_FACTOR) as u8
            };
            cell.fg = Color::Rgb(mix(br, fr), mix(bg, fg), mix(bb, fb));
            cell.modifier.remove(Modifier::DIM);
        }
    }
}

#[cfg(test)]
#[path = "host_tests.rs"]
mod tests;

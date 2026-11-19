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

use crossterm::event::{Event as TermEvent, EventStream, KeyEvent};
use futures_util::StreamExt;
use ratatui::Frame;
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Clear, Paragraph, Widget, Wrap};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender, unbounded_channel};
use tracing::{error, info, warn};

use crate::app::events::{Event, Poster};
use crate::app::logging::{critical, see_debug_log};
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
}

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
    pub fn toast_texts(&self) -> Vec<String> {
        self.toasts.iter().map(|toast| toast.text.clone()).collect()
    }

    /// Run `future` on the runtime; its output comes back as `Msg::Task`.
    pub fn spawn<T, F>(&self, future: F)
    where
        T: Send + 'static,
        F: Future<Output = T> + Send + 'static,
    {
        let sender = self.sender.clone();
        self.runtime.spawn(async move {
            let _ = sender.send(Msg::Task(Box::new(future.await)));
        });
    }

    /// Run `job` on its own thread; its output comes back as `Msg::Task`.
    pub fn spawn_thread<T, F>(&self, name: &str, job: F)
    where
        T: Send + 'static,
        F: FnOnce() -> T + Send + 'static,
    {
        let sender = self.sender.clone();
        let spawned = std::thread::Builder::new()
            .name(name.into())
            .spawn(move || {
                let _ = sender.send(Msg::Task(Box::new(job())));
            });
        if let Err(e) = spawned {
            error!(target: "ltn.app", "Cannot start thread {name}: {e}");
        }
    }

    /// A poster whose events arrive as `Msg::Worker`.
    pub fn poster(&self) -> Poster {
        let (events, mut received) = unbounded_channel();
        let sender = self.sender.clone();
        self.runtime.spawn(async move {
            while let Some(event) = received.recv().await {
                if sender.send(Msg::Worker(event)).is_err() {
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
}

/// What a screen asks the host to do after an event.
pub enum Action {
    None,
    Push(Box<dyn Screen>),
    /// Remove this screen and hand the result to the one below.
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
            ctx: Ctx {
                theme: Theme::load(theme),
                toasts: Vec::new(),
                sender,
                runtime,
                exit_at: None,
                exit_now: false,
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
        let action = self.guarded(|screen, ctx| screen.mounted(ctx));
        self.apply(action);
    }

    fn apply(&mut self, action: Action) {
        match action {
            Action::None => {}
            Action::Push(screen) => self.push(screen),
            Action::Replace(screen) => {
                self.stack.pop();
                self.push(screen);
            }
            Action::Pop(result) => {
                self.stack.pop();
                if self.stack.is_empty() {
                    self.result = result;
                    return;
                }
                let action = self.guarded(|screen, ctx| screen.resumed(result, ctx));
                self.apply(action);
            }
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
        let action = self.guarded(|screen, ctx| screen.key(key, &name, ctx));
        self.apply(action);
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
        let mut pending = Some(msg);
        for index in (0..self.stack.len()).rev() {
            let Some(msg) = pending.take() else {
                break;
            };
            let screen = &mut self.stack[index];
            let ctx = &mut self.ctx;
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
            // A screen below the top only reacts; it does not push or pop over the top.
            if index + 1 == self.stack.len() {
                self.apply(action);
            } else if !matches!(action, Action::None) {
                warn!(target: "ltn.app", "a covered screen asked for a screen change; ignored");
            }
        }
    }

    pub fn tick(&mut self) {
        let now = Instant::now();
        self.ctx.toasts.retain(|toast| toast.until > now);
        for index in 0..self.stack.len() {
            let screen = &mut self.stack[index];
            let ctx = &mut self.ctx;
            let outcome = catching(|| screen.tick(ctx));
            match outcome {
                Ok(action) if index + 1 == self.stack.len() => self.apply(action),
                Ok(_) => {}
                Err(message) => {
                    let session = self.session_running();
                    self.errors.record(&message, session, &mut self.ctx);
                }
            }
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
            if let Err(message) = catching(|| screen.render(frame, area, ctx)) {
                let session = self.session_running();
                self.errors.record(&message, session, &mut self.ctx);
            }
        }
        render_toasts(frame, area, &self.ctx);
    }

    /// Run until the stack empties or a screen exits the app.
    pub fn run(&mut self, terminal: &mut Term) -> std::io::Result<()> {
        let signals = watch_shutdown_signals(self.ctx.sender.clone(), self.watchdog_armed.clone());
        let runtime = self.ctx.runtime.clone();
        let outcome = runtime.block_on(self.event_loop(terminal));
        if let Some(handle) = signals {
            handle.close();
        }
        outcome
    }

    async fn event_loop(&mut self, terminal: &mut Term) -> std::io::Result<()> {
        let mut input = EventStream::new();
        let mut ticks = tokio::time::interval(TICK);
        while !self.finished() {
            terminal.draw(|frame| self.draw(frame))?;
            tokio::select! {
                event = input.next() => match event {
                    Some(Ok(event)) => self.terminal_event(event),
                    Some(Err(e)) => return Err(e),
                    None => break,
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
) -> Option<signal_hook::iterator::Handle> {
    use signal_hook::consts::{SIGHUP, SIGTERM};
    let mut signals = match signal_hook::iterator::Signals::new([SIGTERM, SIGHUP]) {
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
    Some(handle)
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
                (f64::from(from) + (f64::from(to) - f64::from(from)) * 0.6) as u8
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
    let width = 60.min(area.width / 2).max(20).min(area.width);
    let mut bottom = area.bottom().saturating_sub(1);
    for toast in ctx.toasts.iter().rev() {
        let (edge, title_color) = match toast.severity {
            Severity::Information => ("success", "text-success"),
            Severity::Warning => ("warning", "text-warning"),
            Severity::Error => ("error", "text-error"),
        };
        let background = theme.on("panel-lighten-1", "background");
        let mut lines = Vec::new();
        if !toast.title.is_empty() {
            lines.push(Line::from(Span::styled(
                toast.title.clone(),
                Style::default()
                    .fg(theme.on(title_color, "panel-lighten-1"))
                    .add_modifier(Modifier::BOLD),
            )));
        }
        lines.extend(toast.text.lines().map(|line| Line::from(line.to_string())));
        let inner_width = width.saturating_sub(3).max(1);
        let text_rows = crate::tui::text::wrapped_height(&lines, inner_width);
        let height = (text_rows + 2).min(bottom.saturating_sub(area.top()));
        if height < 3 {
            break;
        }
        let rect = Rect::new(
            area.right().saturating_sub(width),
            bottom - height,
            width,
            height,
        );
        frame.render_widget(Clear, rect);
        let style = Style::default()
            .bg(background)
            .fg(theme.on("foreground", "panel-lighten-1"));
        frame.render_widget(Block::default().style(style), rect);
        let edge_style = Style::default().fg(theme.color(edge)).bg(background);
        for y in rect.top()..rect.bottom() {
            Span::styled("▌", edge_style).render(Rect::new(rect.x, y, 1, 1), frame.buffer_mut());
        }
        let text_area = Rect::new(rect.x + 2, rect.y + 1, inner_width, height - 2);
        frame.render_widget(
            Paragraph::new(lines)
                .style(style)
                .wrap(Wrap { trim: false }),
            text_area,
        );
        bottom = rect.top().saturating_sub(1);
    }
}

#[cfg(test)]
#[path = "host_tests.rs"]
mod tests;

//! The plugin host process (PLAN.md, "Plugins"): its launch through uvx, JSON-RPC
//! over its pipes, and each way it can fail, as text a user can act on.
//!
//! Threads, not tasks: the host outlives any one runtime (a command runs a TUI
//! session with its own runtime in the middle), and its pipes block.

use std::collections::{HashMap, VecDeque};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::oneshot;
use tracing::{info, warn};

/// The protocol this app speaks; the host says its own in `hello`.
pub const PROTOCOL: u64 = 1;
/// The first start installs Python and the plugins, which can take a while.
pub const START_LIMIT: Duration = Duration::from_secs(120);
/// The host stderr lines kept for an error message.
const TAIL_LINES: usize = 12;
/// The package that provides the host script.
pub const HOST_SPEC: &str = "lazy-take-notes @ git+https://github.com/CJHwong/lazy-take-notes.git";
pub const HOST_SCRIPT: &str = "lazy-take-notes-plugin-host";
pub const UV_INSTALL: &str = "curl -LsSf https://astral.sh/uv/install.sh | sh";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    /// uv is not installed.
    NoUv,
    /// No plugin is installed.
    NoPlugins,
    /// The host did not start or did not say hello.
    Start,
    /// The host died, or broke the protocol.
    Gone,
    /// The host did not answer in time.
    Timeout,
    /// The plugin raised: `kind` is its Python exception class.
    Raised,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PluginError {
    pub kind: ErrorKind,
    /// The Python exception class for `Raised`.
    pub python_type: String,
    pub message: String,
}

impl PluginError {
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            python_type: String::new(),
            message: message.into(),
        }
    }
}

impl std::fmt::Display for PluginError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.python_type.is_empty() {
            true => f.write_str(&self.message),
            false => write!(f, "{}: {}", self.python_type, self.message),
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct CommandInfo {
    pub name: String,
    #[serde(default)]
    pub help: String,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct LoadError {
    pub plugin: String,
    /// The package the entry point comes from, when the host knows it.
    #[serde(default)]
    pub dist: Option<String>,
    #[serde(default)]
    pub group: String,
    pub error: String,
}

/// What the host said it provides.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct Hello {
    pub protocol: u64,
    #[serde(default)]
    pub llm_providers: Vec<String>,
    #[serde(default)]
    pub commands: Vec<CommandInfo>,
    #[serde(default)]
    pub load_errors: Vec<LoadError>,
}

impl LoadError {
    /// Whether this error comes from package `name` (pip compares names with
    /// case, `-`, `_` and `.` folded).
    pub fn from_package(&self, name: &str) -> bool {
        let fold = |text: &str| text.to_lowercase().replace(['_', '.'], "-");
        self.dist
            .as_deref()
            .is_some_and(|dist| fold(dist) == fold(name))
    }
}

impl Hello {
    /// One line per plugin that failed to load, as Python warned.
    pub fn warnings(&self) -> Vec<String> {
        self.load_errors
            .iter()
            .map(|e| format!("Warning: plugin '{}' failed to load: {}", e.plugin, e.error))
            .collect()
    }
}

/// A request from the host to the app (`session.transcribe`).
#[derive(Debug)]
pub struct Incoming {
    pub id: Value,
    pub method: String,
    pub params: Value,
}

type Reply = Result<Value, PluginError>;

struct Inner {
    stdin: Mutex<Option<ChildStdin>>,
    child: Mutex<Child>,
    pending: Mutex<HashMap<u64, oneshot::Sender<Reply>>>,
    next_id: AtomicU64,
    gone: Mutex<Option<PluginError>>,
    tail: Mutex<VecDeque<String>>,
    /// Host stderr goes to the terminal; off while a TUI owns it.
    echo: AtomicBool,
    stderr_done: AtomicBool,
    incoming: Mutex<Option<mpsc::Sender<Incoming>>>,
}

impl Inner {
    fn tail_text(&self) -> String {
        let tail = self.tail.lock().unwrap_or_else(|e| e.into_inner());
        match tail.is_empty() {
            true => String::new(),
            false => format!(
                "\nLast lines from the plugin host:\n{}",
                tail.iter().cloned().collect::<Vec<_>>().join("\n")
            ),
        }
    }

    /// The host is unusable: every waiting call and every later one gets `error`.
    fn die(&self, error: PluginError) {
        {
            let mut gone = self.gone.lock().unwrap_or_else(|e| e.into_inner());
            if gone.is_none() {
                warn!(target: "ltn.plugins", "plugin host unusable: {error}");
                *gone = Some(error.clone());
            }
        }
        let waiting: Vec<_> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .drain()
            .collect();
        for (_, sender) in waiting {
            let _ = sender.send(Err(error.clone()));
        }
    }

    fn gone(&self) -> Option<PluginError> {
        self.gone.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    fn write(&self, message: &Value) -> Result<(), PluginError> {
        let mut line = message.to_string();
        line.push('\n');
        let mut stdin = self.stdin.lock().unwrap_or_else(|e| e.into_inner());
        let Some(pipe) = stdin.as_mut() else {
            return Err(self.gone().unwrap_or_else(|| {
                PluginError::new(ErrorKind::Gone, "The plugin host is closed.")
            }));
        };
        if let Err(e) = pipe.write_all(line.as_bytes()).and_then(|()| pipe.flush()) {
            let error = PluginError::new(
                ErrorKind::Gone,
                format!("Cannot write to the plugin host: {e}.{}", self.tail_text()),
            );
            drop(stdin);
            self.die(error.clone());
            return Err(error);
        }
        Ok(())
    }

    /// Wait up to `limit` for the stderr reader to drain, so an exit message has
    /// the host's last words.
    fn wait_stderr(&self, limit: Duration) {
        let end = Instant::now() + limit;
        while !self.stderr_done.load(Ordering::SeqCst) && Instant::now() < end {
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    fn exit_status(&self) -> String {
        let mut child = self.child.lock().unwrap_or_else(|e| e.into_inner());
        let end = Instant::now() + Duration::from_secs(1);
        loop {
            match child.try_wait() {
                Ok(Some(status)) => {
                    return match status.code() {
                        Some(code) => format!(" with exit code {code}"),
                        None => " on a signal".into(),
                    };
                }
                Ok(None) if Instant::now() < end => std::thread::sleep(Duration::from_millis(10)),
                _ => return String::new(),
            }
        }
    }
}

/// The last handle shuts the host down: its stdin closes, which the host takes
/// as the end, and after 2 s it is killed.
struct Handle(Arc<Inner>);

impl Drop for Handle {
    fn drop(&mut self) {
        self.0
            .stdin
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        let mut child = self.0.child.lock().unwrap_or_else(|e| e.into_inner());
        let end = Instant::now() + Duration::from_secs(2);
        while Instant::now() < end {
            if let Ok(Some(_)) = child.try_wait() {
                return;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        let _ = child.kill();
        let _ = child.wait();
    }
}

#[derive(Clone)]
pub struct PluginHost {
    handle: Arc<Handle>,
    pub hello: Arc<Hello>,
}

impl PluginHost {
    fn inner(&self) -> &Inner {
        &self.handle.0
    }

    /// Start the host for `specs` through uvx; on a failed start, once more from
    /// uv's cache (`--offline`), so a laptop without network still has its plugins.
    pub fn start(specs: &[String]) -> Result<Self, PluginError> {
        if specs.is_empty() {
            return Err(PluginError::new(
                ErrorKind::NoPlugins,
                "No plugins are installed. Add one with: lazy-take-notes plugin add <spec>",
            ));
        }
        let uvx = find_uvx().ok_or_else(|| {
            PluginError::new(
                ErrorKind::NoUv,
                format!(
                    "Plugins run on Python through uv, and uv is not installed.\nInstall it: {UV_INSTALL}"
                ),
            )
        })?;
        let source = host_source(std::env::var("LTN_PLUGIN_HOST_FROM").ok().as_deref());
        let overrides = overrides_file(&source)?;
        let overrides = overrides.as_ref().map(|file| file.path());
        let launch = |offline| launch_command(&uvx, specs, offline, &source, overrides);
        match Self::start_command(launch(false), START_LIMIT) {
            Err(error) if error.kind == ErrorKind::Start => {
                info!(target: "ltn.plugins", "plugin host start failed, trying uv's cache: {error}");
                Self::start_command(launch(true), START_LIMIT).map_err(|_| error)
            }
            other => other,
        }
    }

    /// Start `command` as a host and wait up to `limit` for its `hello`.
    pub fn start_command(mut command: Command, limit: Duration) -> Result<Self, PluginError> {
        let program = command.get_program().to_string_lossy().into_owned();
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut child = command.spawn().map_err(|e| {
            PluginError::new(
                ErrorKind::Start,
                format!("Cannot start the plugin host ({program}): {e}"),
            )
        })?;
        let (Some(stdin), Some(stdout), Some(stderr)) =
            (child.stdin.take(), child.stdout.take(), child.stderr.take())
        else {
            let _ = child.kill();
            return Err(PluginError::new(
                ErrorKind::Start,
                "Cannot open the plugin host's pipes.",
            ));
        };
        let inner = Arc::new(Inner {
            stdin: Mutex::new(Some(stdin)),
            child: Mutex::new(child),
            pending: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(1),
            gone: Mutex::new(None),
            tail: Mutex::new(VecDeque::new()),
            echo: AtomicBool::new(false),
            stderr_done: AtomicBool::new(false),
            incoming: Mutex::new(None),
        });
        let handle = Arc::new(Handle(inner.clone()));
        spawn_stderr_reader(inner.clone(), stderr);
        let (hello_tx, hello_rx) = mpsc::channel();
        spawn_stdout_reader(inner.clone(), stdout, hello_tx);
        let hello = match hello_rx.recv_timeout(limit) {
            Ok(Ok(hello)) => hello,
            Ok(Err(error)) => return Err(PluginError::new(ErrorKind::Start, error.message)),
            Err(_) => {
                let error = PluginError::new(
                    ErrorKind::Start,
                    format!(
                        "The plugin host did not start within {} s.{}",
                        limit.as_secs(),
                        inner.tail_text()
                    ),
                );
                inner.die(error.clone());
                return Err(error);
            }
        };
        if hello.protocol != PROTOCOL {
            return Err(PluginError::new(
                ErrorKind::Start,
                format!(
                    "The plugin host speaks protocol {}, and this app speaks {PROTOCOL}. Update lazy-take-notes and the plugins to matching versions.",
                    hello.protocol
                ),
            ));
        }
        for line in hello.warnings() {
            warn!(target: "ltn.plugins", "{line}");
        }
        Ok(Self {
            handle,
            hello: Arc::new(hello),
        })
    }

    pub fn is_alive(&self) -> bool {
        self.inner().gone().is_none()
    }

    /// Show the host's stderr on the terminal (true), or only in the log while a
    /// TUI owns the screen (false, the default).
    pub fn echo_stderr(&self, on: bool) {
        self.inner().echo.store(on, Ordering::SeqCst);
    }

    /// Requests the host sends to the app arrive on the returned channel. Without
    /// one, the app answers them "unknown method".
    pub fn take_incoming(&self) -> mpsc::Receiver<Incoming> {
        let (sender, receiver) = mpsc::channel();
        *self
            .inner()
            .incoming
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = Some(sender);
        receiver
    }

    fn send(
        &self,
        method: &str,
        params: Value,
    ) -> Result<(u64, oneshot::Receiver<Reply>), PluginError> {
        let inner = self.inner();
        if let Some(error) = inner.gone() {
            return Err(error);
        }
        let id = inner.next_id.fetch_add(1, Ordering::SeqCst);
        let (sender, receiver) = oneshot::channel();
        inner
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(id, sender);
        let message = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
        if let Err(error) = inner.write(&message) {
            inner
                .pending
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .remove(&id);
            return Err(error);
        }
        Ok((id, receiver))
    }

    fn lost(&self) -> PluginError {
        self.inner().gone().unwrap_or_else(|| {
            PluginError::new(ErrorKind::Gone, "The plugin host stopped answering.")
        })
    }

    /// Call `method` and wait up to `limit` for the answer.
    pub async fn call(&self, method: &str, params: Value, limit: Duration) -> Reply {
        let (id, receiver) = self.send(method, params)?;
        match tokio::time::timeout(limit, receiver).await {
            Ok(Ok(reply)) => reply,
            Ok(Err(_)) => Err(self.lost()),
            Err(_) => {
                self.inner()
                    .pending
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .remove(&id);
                Err(PluginError::new(
                    ErrorKind::Timeout,
                    format!(
                        "The plugin did not answer {method} within {} s.",
                        limit.as_secs()
                    ),
                ))
            }
        }
    }

    /// Call `method` from a plain thread and wait for the answer with no limit
    /// (a command runs as long as it runs). Requests from the host arrive on
    /// `incoming` meanwhile and go to `serve`.
    pub fn call_serving(
        &self,
        method: &str,
        params: Value,
        incoming: &mpsc::Receiver<Incoming>,
        mut serve: impl FnMut(Incoming) -> Result<Value, (i64, String)>,
    ) -> Reply {
        let (_, mut receiver) = self.send(method, params)?;
        loop {
            match receiver.try_recv() {
                Ok(reply) => return reply,
                Err(oneshot::error::TryRecvError::Closed) => return Err(self.lost()),
                Err(oneshot::error::TryRecvError::Empty) => {}
            }
            if let Ok(request) = incoming.recv_timeout(Duration::from_millis(50)) {
                let id = request.id.clone();
                let answer = serve(request);
                self.respond(id, answer);
            }
        }
    }

    /// Answer a request the host sent.
    pub fn respond(&self, id: Value, answer: Result<Value, (i64, String)>) {
        let message = match answer {
            Ok(result) => json!({"jsonrpc": "2.0", "id": id, "result": result}),
            Err((code, text)) => {
                json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": text}})
            }
        };
        if let Err(error) = self.inner().write(&message) {
            warn!(target: "ltn.plugins", "cannot answer the plugin host: {error}");
        }
    }
}

fn spawn_stderr_reader(inner: Arc<Inner>, stderr: std::process::ChildStderr) {
    let spawned = std::thread::Builder::new()
        .name("plugin-stderr".into())
        .spawn(move || {
            for line in BufReader::new(stderr).lines() {
                let Ok(line) = line else { break };
                if inner.echo.load(Ordering::SeqCst) {
                    eprintln!("{line}");
                } else {
                    info!(target: "ltn.plugins", "host: {line}");
                }
                let mut tail = inner.tail.lock().unwrap_or_else(|e| e.into_inner());
                if tail.len() == TAIL_LINES {
                    tail.pop_front();
                }
                tail.push_back(line);
            }
            inner.stderr_done.store(true, Ordering::SeqCst);
        });
    if spawned.is_err() {
        warn!(target: "ltn.plugins", "cannot start the plugin stderr reader");
    }
}

fn spawn_stdout_reader(
    inner: Arc<Inner>,
    stdout: std::process::ChildStdout,
    hello: mpsc::Sender<Result<Hello, PluginError>>,
) {
    let fallback = inner.clone();
    let spawned = std::thread::Builder::new()
        .name("plugin-stdout".into())
        .spawn(move || read_stdout(&inner, stdout, &hello));
    if let Err(e) = spawned {
        fallback.die(PluginError::new(
            ErrorKind::Start,
            format!("Cannot start the plugin reader: {e}"),
        ));
    }
}

fn read_stdout(
    inner: &Inner,
    stdout: std::process::ChildStdout,
    hello: &mpsc::Sender<Result<Hello, PluginError>>,
) {
    for line in BufReader::new(stdout).lines() {
        let line = match line {
            Ok(line) => line,
            Err(e) => {
                let error = PluginError::new(
                    ErrorKind::Gone,
                    format!(
                        "Cannot read from the plugin host: {e}.{}",
                        inner.tail_text()
                    ),
                );
                let _ = hello.send(Err(error.clone()));
                inner.die(error);
                return;
            }
        };
        if line.trim().is_empty() {
            continue;
        }
        if let Err(error) = route(inner, &line, hello) {
            let _ = hello.send(Err(error.clone()));
            inner.die(error);
            let _ = inner.child.lock().unwrap_or_else(|e| e.into_inner()).kill();
            return;
        }
    }
    // End of output: the host exited, or closed its stdout.
    inner.wait_stderr(Duration::from_secs(1));
    let error = PluginError::new(
        ErrorKind::Gone,
        format!(
            "The plugin host exited{}.{}",
            inner.exit_status(),
            inner.tail_text()
        ),
    );
    let _ = hello.send(Err(error.clone()));
    inner.die(error);
}

/// One line from the host: a reply, a request, or `hello`. An error ends the host.
fn route(
    inner: &Inner,
    line: &str,
    hello: &mpsc::Sender<Result<Hello, PluginError>>,
) -> Result<(), PluginError> {
    let message: Value = serde_json::from_str(line).map_err(|_| {
        let shown: String = line.chars().take(200).collect();
        PluginError::new(
            ErrorKind::Gone,
            format!("The plugin host sent a line that is not JSON: {shown}"),
        )
    })?;
    if let Some(method) = message.get("method").and_then(Value::as_str) {
        let params = message.get("params").cloned().unwrap_or(Value::Null);
        let Some(id) = message.get("id").cloned() else {
            if method == "hello" {
                let parsed: Hello = serde_json::from_value(params).map_err(|e| {
                    PluginError::new(
                        ErrorKind::Start,
                        format!("The plugin host's hello is malformed: {e}"),
                    )
                })?;
                let _ = hello.send(Ok(parsed));
            }
            return Ok(());
        };
        let request = Incoming {
            id: id.clone(),
            method: method.to_string(),
            params,
        };
        let sender = inner
            .incoming
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let delivered = sender.is_some_and(|sender| sender.send(request).is_ok());
        if !delivered {
            inner.write(&json!({
                "jsonrpc": "2.0",
                "id": id,
                "error": {"code": -32601, "message": format!("The app does not take {method} now.")}
            }))?;
        }
        return Ok(());
    }
    let Some(id) = message.get("id").and_then(Value::as_u64) else {
        return Ok(());
    };
    let reply = match message.get("error") {
        Some(error) => Err(PluginError {
            kind: ErrorKind::Raised,
            python_type: error
                .pointer("/data/type")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
            message: error
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("The plugin failed without a message.")
                .to_string(),
        }),
        None => Ok(message.get("result").cloned().unwrap_or(Value::Null)),
    };
    let sender = inner
        .pending
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .remove(&id);
    if let Some(sender) = sender {
        let _ = sender.send(reply);
    }
    Ok(())
}

/// `uvx` on PATH, or where the uv installer and Homebrew put it: an app started
/// from a launcher may have a shorter PATH than the user's shell.
pub fn find_uvx() -> Option<PathBuf> {
    let mut places: Vec<PathBuf> = std::env::var_os("PATH")
        .map(|path| std::env::split_paths(&path).collect())
        .unwrap_or_default();
    if let Some(home) = dirs::home_dir() {
        places.push(home.join(".local/bin"));
        places.push(home.join(".cargo/bin"));
    }
    places.push(PathBuf::from("/opt/homebrew/bin"));
    places.push(PathBuf::from("/usr/local/bin"));
    places
        .into_iter()
        .map(|dir| dir.join("uvx"))
        .find(|path| is_executable(path))
}

fn is_executable(path: &Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::metadata(path)
            .is_ok_and(|meta| meta.is_file() && meta.permissions().mode() & 0o111 != 0)
    }
    #[cfg(windows)]
    {
        path.is_file()
    }
}

/// Where the host comes from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostSource {
    /// The `--from` argument.
    pub from: String,
    /// The plugins pin `HOST_SPEC`; any other source must override that pin,
    /// or uv refuses the two URLs for one package.
    pub pin: Option<String>,
    /// uv caches a local folder's build by its pyproject.toml only, so a
    /// source edit would run old code: build it fresh each start.
    pub no_cache: bool,
}

/// `LTN_PLUGIN_HOST_FROM` (a local checkout, or a full spec such as another git
/// ref) while developing; else the released package.
pub fn host_source(value: Option<&str>) -> HostSource {
    let Some(value) = value.map(str::trim).filter(|v| !v.is_empty()) else {
        return HostSource {
            from: HOST_SPEC.to_string(),
            pin: None,
            no_cache: false,
        };
    };
    let folder = Path::new(value);
    if folder.is_dir() {
        let folder = folder
            .canonicalize()
            .unwrap_or_else(|_| folder.to_path_buf());
        return HostSource {
            from: folder.display().to_string(),
            pin: Some(format!("lazy-take-notes @ file://{}", folder.display())),
            no_cache: true,
        };
    }
    HostSource {
        from: value.to_string(),
        pin: Some(value.to_string()),
        no_cache: false,
    }
}

/// `overrides`: a file holding `source.pin`, when it has one.
pub fn launch_command(
    uvx: &Path,
    specs: &[String],
    offline: bool,
    source: &HostSource,
    overrides: Option<&Path>,
) -> Command {
    let mut command = Command::new(uvx);
    if offline {
        command.arg("--offline");
    }
    if source.no_cache && !offline {
        command.arg("--no-cache");
    }
    if let Some(file) = overrides {
        command.arg("--overrides").arg(file);
    }
    command.arg("--from").arg(&source.from);
    for spec in specs {
        command.arg("--with").arg(spec);
    }
    command.arg(HOST_SCRIPT);
    command
}

/// The overrides file for `source`, alive until the host has started.
fn overrides_file(source: &HostSource) -> Result<Option<tempfile::NamedTempFile>, PluginError> {
    let Some(pin) = &source.pin else {
        return Ok(None);
    };
    let fail = |e: std::io::Error| {
        PluginError::new(
            ErrorKind::Start,
            format!("Cannot write the uv overrides file: {e}"),
        )
    };
    let mut file = tempfile::Builder::new()
        .prefix("ltn-plugin-overrides-")
        .suffix(".txt")
        .tempfile()
        .map_err(fail)?;
    writeln!(file, "{pin}").map_err(fail)?;
    Ok(Some(file))
}

#[cfg(test)]
#[path = "host_tests.rs"]
mod tests;

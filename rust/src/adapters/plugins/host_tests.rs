//! The host's failure paths against fake hosts written in `sh`: each fake says
//! `hello` (or does not), then answers the app's requests in order, ids 1, 2, ...

use super::*;

const HELLO: &str = r#"{"jsonrpc":"2.0","method":"hello","params":{"protocol":1,"llm_providers":["fake"],"commands":[{"name":"echo","help":"Echo."}],"load_errors":[{"plugin":"broken","group":"lazy_take_notes.plugins","error":"No module named x"}]}}"#;

fn fake(script: &str) -> Command {
    let mut command = Command::new("/bin/sh");
    command
        .arg("-c")
        .arg(script.replace("HELLO", &format!("echo '{HELLO}'")));
    command
}

fn start(script: &str) -> Result<PluginHost, PluginError> {
    PluginHost::start_command(fake(script), Duration::from_secs(5))
}

fn block_on<T>(future: impl std::future::Future<Output = T>) -> T {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(future)
}

fn call(host: &PluginHost, method: &str) -> Reply {
    block_on(host.call(method, json!({}), Duration::from_secs(5)))
}

#[test]
fn hello_lists_the_providers_and_the_plugins_that_failed_to_load() {
    let host = start("HELLO; read line").unwrap();
    assert_eq!(host.hello.llm_providers, ["fake"]);
    assert_eq!(host.hello.commands[0].name, "echo");
    assert_eq!(
        host.hello.warnings(),
        ["Warning: plugin 'broken' failed to load: No module named x"]
    );
}

#[test]
fn a_host_that_cannot_run_says_so() {
    let error = PluginHost::start_command(Command::new("/no/such/uvx"), Duration::from_secs(1))
        .err()
        .unwrap();
    assert_eq!(error.kind, ErrorKind::Start);
    assert!(
        error
            .message
            .starts_with("Cannot start the plugin host (/no/such/uvx)"),
        "{error}"
    );
}

#[test]
fn a_host_that_exits_before_hello_shows_its_last_words() {
    let error = start("echo 'error: No solution found for ltn-nope' >&2; exit 2")
        .err()
        .unwrap();
    assert_eq!(error.kind, ErrorKind::Start);
    assert!(error.message.contains("exited with exit code 2"), "{error}");
    assert!(
        error.message.contains("No solution found for ltn-nope"),
        "{error}"
    );
}

#[test]
fn a_host_that_never_says_hello_times_out() {
    let error = PluginHost::start_command(fake("sleep 5"), Duration::from_millis(300))
        .err()
        .unwrap();
    assert_eq!(error.kind, ErrorKind::Start);
    assert!(error.message.contains("did not start within"), "{error}");
}

#[test]
fn another_protocol_asks_for_matching_versions() {
    let error =
        start(r#"echo '{"jsonrpc":"2.0","method":"hello","params":{"protocol":9}}'; read line"#)
            .err()
            .unwrap();
    assert!(error.message.contains("protocol 9"), "{error}");
}

#[test]
fn a_reply_and_a_plugin_exception_come_back_to_the_caller() {
    let host = start(concat!(
        "HELLO; read line; ",
        r#"echo '{"jsonrpc":"2.0","id":1,"result":{"content":"hi"}}'; read line; "#,
        r#"echo '{"jsonrpc":"2.0","id":2,"error":{"code":-32000,"message":"bad key","data":{"type":"ValueError"}}}'; read line"#,
    ))
    .unwrap();
    assert_eq!(call(&host, "llm.chat").unwrap(), json!({"content": "hi"}));
    let error = call(&host, "llm.chat").unwrap_err();
    assert_eq!(error.kind, ErrorKind::Raised);
    assert_eq!(error.to_string(), "ValueError: bad key");
    assert!(host.is_alive());
}

#[test]
fn a_line_that_is_not_json_ends_the_host_for_every_later_call() {
    let host = start("HELLO; read line; echo 'Downloading torch...'; sleep 5").unwrap();
    let error = call(&host, "llm.chat").unwrap_err();
    assert_eq!(error.kind, ErrorKind::Gone);
    assert!(
        error.message.contains("not JSON: Downloading torch..."),
        "{error}"
    );
    assert_eq!(call(&host, "llm.chat").unwrap_err(), error);
    assert!(!host.is_alive());
}

#[test]
fn a_host_that_dies_during_a_call_fails_that_call_with_its_last_words() {
    let host = start("HELLO; read line; echo 'Traceback: boom' >&2; exit 3").unwrap();
    let error = call(&host, "llm.chat").unwrap_err();
    assert_eq!(error.kind, ErrorKind::Gone);
    assert!(error.message.contains("exit code 3"), "{error}");
    assert!(error.message.contains("Traceback: boom"), "{error}");
}

#[test]
fn a_call_with_no_answer_times_out_and_the_host_stays() {
    let host = start("HELLO; read line; sleep 5").unwrap();
    let error = block_on(host.call("llm.chat", json!({}), Duration::from_millis(200))).unwrap_err();
    assert_eq!(error.kind, ErrorKind::Timeout);
    assert!(error.message.contains("llm.chat"), "{error}");
    assert!(host.is_alive());
}

#[test]
fn a_command_gets_the_answer_to_its_request_to_the_app() {
    let host = start(concat!(
        "HELLO; read line; ",
        r#"echo '{"jsonrpc":"2.0","id":"h1","method":"session.transcribe","params":{"label":"talk"}}'; "#,
        r#"read answer; printf '{"jsonrpc":"2.0","id":1,"result":{"answer":%s}}\n' "$answer"; read line"#,
    ))
    .unwrap();
    let incoming = host.take_incoming();
    let mut seen = Vec::new();
    let reply = host
        .call_serving("command.run", json!({}), &incoming, |request| {
            seen.push((request.method.clone(), request.params.clone()));
            Ok(json!({"session_dir": "/tmp/s"}))
        })
        .unwrap();
    assert_eq!(
        seen,
        [("session.transcribe".to_string(), json!({"label": "talk"}))]
    );
    assert_eq!(
        reply["answer"],
        json!({"jsonrpc": "2.0", "id": "h1", "result": {"session_dir": "/tmp/s"}})
    );
}

#[test]
fn a_request_with_nobody_to_take_it_is_refused() {
    let host = start(concat!(
        "HELLO; read line; ",
        r#"echo '{"jsonrpc":"2.0","id":"h1","method":"session.record","params":{}}'; "#,
        r#"read answer; printf '{"jsonrpc":"2.0","id":1,"result":%s}\n' "$answer"; read line"#,
    ))
    .unwrap();
    let reply = call(&host, "command.run").unwrap();
    assert_eq!(reply["error"]["code"], -32601);
}

#[test]
fn no_plugins_means_no_host() {
    let error = PluginHost::start(&[]).err().unwrap();
    assert_eq!(error.kind, ErrorKind::NoPlugins);
    assert!(
        error.message.contains("lazy-take-notes plugin add"),
        "{error}"
    );
}

#[test]
fn the_launch_passes_each_plugin_and_the_offline_retry_uses_the_cache() {
    let specs = [
        "ltn-a".to_string(),
        "ltn-b @ git+https://x/b.git".to_string(),
    ];
    let released = host_source(None);
    let args = |offline| -> Vec<String> {
        launch_command(Path::new("/bin/uvx"), &specs, offline, &released, None)
            .get_args()
            .map(|a| a.to_string_lossy().into_owned())
            .collect()
    };
    assert_eq!(
        args(false),
        [
            "--from",
            HOST_SPEC,
            "--with",
            "ltn-a",
            "--with",
            "ltn-b @ git+https://x/b.git",
            HOST_SCRIPT
        ]
    );
    assert_eq!(args(true)[0], "--offline");
}

#[test]
fn another_host_source_overrides_the_plugins_pin() {
    let folder = tempfile::tempdir().unwrap();
    let local = host_source(folder.path().to_str());
    assert!(local.no_cache);
    let real = folder.path().canonicalize().unwrap();
    assert_eq!(
        local.pin.unwrap(),
        format!("lazy-take-notes @ file://{}", real.display())
    );
    let git = "lazy-take-notes @ git+https://github.com/CJHwong/lazy-take-notes.git@rust-rewrite";
    let other = host_source(Some(git));
    assert_eq!(
        (other.from.as_str(), other.pin.as_deref(), other.no_cache),
        (git, Some(git), false)
    );
    let args: Vec<String> = launch_command(
        Path::new("/bin/uvx"),
        &[],
        false,
        &other,
        Some(Path::new("/tmp/o.txt")),
    )
    .get_args()
    .map(|a| a.to_string_lossy().into_owned())
    .collect();
    assert_eq!(
        args,
        ["--overrides", "/tmp/o.txt", "--from", git, HOST_SCRIPT]
    );
}

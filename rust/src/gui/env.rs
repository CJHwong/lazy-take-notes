//! A Finder start gets launchd's environment: no shell PATH, so no Homebrew
//! ffmpeg, and none of the API keys the shell profile exports. Import the
//! login shell's environment once, before anything reads it.

use std::collections::HashMap;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use tracing::warn;

const SHELL_TIMEOUT: Duration = Duration::from_secs(5);
/// Where Homebrew and uv put programs, when the shell cannot tell.
const FALLBACK_PATHS: [&str; 3] = ["/opt/homebrew/bin", "/usr/local/bin", "~/.local/bin"];

/// Set each variable the login shell has and this process lacks. PATH is
/// always the shell's, so programs resolve as they do in a terminal.
pub fn import_login_environment() {
    match login_environment() {
        Ok(shell_env) => apply(shell_env),
        Err(e) => {
            warn!(target: "ltn.gui", "Could not read the login shell environment: {e}");
            add_fallback_paths();
        }
    }
}

fn apply(shell_env: HashMap<String, String>) {
    for (name, value) in shell_env {
        let keep_ours = name != "PATH" && std::env::var_os(&name).is_some();
        if keep_ours || name.is_empty() || name.contains('=') || name.starts_with("__CF") {
            continue;
        }
        // SAFETY: this runs on the main thread before any other thread starts.
        unsafe { std::env::set_var(name, value) };
    }
}

fn add_fallback_paths() {
    let home = std::env::var("HOME").unwrap_or_default();
    let mut paths: Vec<String> = std::env::var("PATH")
        .unwrap_or_default()
        .split(':')
        .filter(|part| !part.is_empty())
        .map(str::to_string)
        .collect();
    for fallback in FALLBACK_PATHS {
        let fallback = fallback.replacen('~', &home, 1);
        if !paths.contains(&fallback) {
            paths.push(fallback);
        }
    }
    // SAFETY: this runs on the main thread before any other thread starts.
    unsafe { std::env::set_var("PATH", paths.join(":")) };
}

/// `env -0` from an interactive login shell, so both the profile and the rc
/// file run, as in a new terminal.
fn login_environment() -> Result<HashMap<String, String>, String> {
    let shell = std::env::var("SHELL").unwrap_or_else(|_| "/bin/zsh".into());
    let mut child = Command::new(&shell)
        .args(["-l", "-i", "-c", "/usr/bin/env -0"])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("cannot run {shell}: {e}"))?;
    let started = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if started.elapsed() > SHELL_TIMEOUT => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!("{shell} took more than {SHELL_TIMEOUT:?}"));
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(20)),
            Err(e) => return Err(e.to_string()),
        }
    }
    let output = child.wait_with_output().map_err(|e| e.to_string())?;
    if !output.status.success() {
        return Err(format!("{shell} exited with {}", output.status));
    }
    Ok(parse_env(&output.stdout))
}

/// `NAME=value` entries split by NUL. An rc file that prints text first leaves
/// it before the first entry; that text has no `=` before a NUL and is skipped
/// only when it does not parse.
fn parse_env(raw: &[u8]) -> HashMap<String, String> {
    String::from_utf8_lossy(raw)
        .split('\0')
        .filter_map(|entry| {
            let (name, value) = entry.split_once('=')?;
            let name = name.rsplit('\n').next().unwrap_or(name);
            Some((name.to_string(), value.to_string()))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::parse_env;

    #[test]
    fn entries_split_on_nul_and_rc_output_is_dropped() {
        let raw = b"welcome banner\nPATH=/a:/b\0KEY=x=y\0\0";
        let env = parse_env(raw);
        assert_eq!(env["PATH"], "/a:/b");
        assert_eq!(env["KEY"], "x=y");
        assert_eq!(env.len(), 2);
    }
}

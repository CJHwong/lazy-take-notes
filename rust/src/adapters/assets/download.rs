//! HTTP download and archive extraction for the model cache.
//!
//! Resumes a `.part` file, reports progress, and checks the size and SHA-256 of
//! every asset before it lands in the cache.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use flate2::read::GzDecoder;
use sha2::{Digest, Sha256};

use crate::domain::model_asset::AssetProgress;

const CHUNK: usize = 1024 * 1024;
const REPORT_INTERVAL: Duration = Duration::from_millis(250);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(15);
const READ_TIMEOUT: Duration = Duration::from_secs(60);

#[derive(Debug, thiserror::Error)]
pub enum DownloadError {
    /// The caller cancelled. The partial file is kept for the next try.
    #[error("{0}")]
    Cancelled(String),
    #[error("{0}")]
    Failed(String),
    #[error(transparent)]
    Io(#[from] io::Error),
}

fn failed(message: impl Into<String>) -> DownloadError {
    DownloadError::Failed(message.into())
}

fn check_cancel(cancel: Option<&AtomicBool>, what: &str) -> Result<(), DownloadError> {
    if cancel.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
        return Err(DownloadError::Cancelled(what.into()));
    }
    Ok(())
}

/// A client whose reads time out one at a time: a stalled server fails the download,
/// but a slow 1.6 GB download does not.
fn http_client() -> Result<reqwest::Client, DownloadError> {
    reqwest::Client::builder()
        .connect_timeout(CONNECT_TIMEOUT)
        .read_timeout(READ_TIMEOUT)
        // Over HTTP/2 a 5 MB GitHub release download stalled for 16 s at a time (40 s in all);
        // over HTTP/1.1 it took 0.65 s, like curl. Python's requests also speaks HTTP/1.1 only.
        .http1_only()
        .build()
        .map_err(|e| failed(format!("could not start the HTTP client: {e}")))
}

/// SHA-256 of a file, as lowercase hex.
pub fn sha256_file(path: &Path, cancel: Option<&AtomicBool>) -> Result<String, DownloadError> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0; CHUNK];
    loop {
        check_cancel(cancel, "cancelled while verifying")?;
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

/// What a download must match. Zero size or an empty digest skips that check.
#[derive(Debug, Clone, Copy, Default)]
pub struct Expected<'a> {
    pub size: u64,
    pub sha256: &'a str,
}

/// Download `url` to `dest`, resuming `dest.part` when the server allows it.
pub fn download_file(
    url: &str,
    dest: &Path,
    expected: Expected<'_>,
    on_progress: &mut dyn FnMut(AssetProgress),
    cancel: Option<&AtomicBool>,
) -> Result<PathBuf, DownloadError> {
    if dest.is_file() && matches(dest, expected, cancel)? {
        return Ok(dest.to_path_buf());
    }
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)?;
    }
    let part = PathBuf::from(format!("{}.part", dest.display()));
    let existing = fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
    // Downloads run on a worker thread. The connection belongs to one runtime for its whole life.
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let _context = runtime.enter();
    let mut request = http_client()?.get(url);
    if existing > 0 {
        request = request.header(reqwest::header::RANGE, format!("bytes={existing}-"));
    }
    let response = runtime
        .block_on(request.send())
        .map_err(|e| failed(format!("could not download {url}: {e}")))?;
    let status = response.status().as_u16();
    if status == 416 {
        if expected.size > 0 && existing == expected.size {
            verify(&part, expected, cancel)?;
            fs::rename(&part, dest)?;
            return Ok(dest.to_path_buf());
        }
        let _ = fs::remove_file(&part);
        return Err(failed(format!("stale partial file for {url}")));
    }
    let (append, start) = match status {
        206 => (true, existing),
        200 => (false, 0),
        _ => return Err(failed(format!("unexpected HTTP {status} from {url}"))),
    };
    let total = response.content_length().map_or(0, |length| start + length);
    let started = Instant::now();
    let body = Body {
        runtime: &runtime,
        response,
        start,
        total,
    };
    let downloaded = body.write_to(&part, append, on_progress, cancel)?;
    verify(
        &part,
        Expected {
            size: if expected.size > 0 {
                expected.size
            } else {
                total
            },
            ..expected
        },
        cancel,
    )?;
    fs::rename(&part, dest)?;
    let elapsed = started.elapsed().as_secs_f64().max(0.001);
    on_progress(AssetProgress {
        downloaded,
        total: if total > 0 { total } else { downloaded },
        bytes_per_second: (downloaded - start) as f64 / elapsed,
    });
    Ok(dest.to_path_buf())
}

fn matches(
    path: &Path,
    expected: Expected<'_>,
    cancel: Option<&AtomicBool>,
) -> Result<bool, DownloadError> {
    if expected.size > 0 && fs::metadata(path)?.len() != expected.size {
        return Ok(false);
    }
    Ok(expected.sha256.is_empty() || sha256_file(path, cancel)? == expected.sha256)
}

struct Body<'a> {
    runtime: &'a tokio::runtime::Runtime,
    response: reqwest::Response,
    start: u64,
    total: u64,
}

impl Body<'_> {
    fn write_to(
        mut self,
        part: &Path,
        append: bool,
        on_progress: &mut dyn FnMut(AssetProgress),
        cancel: Option<&AtomicBool>,
    ) -> Result<u64, DownloadError> {
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .append(append)
            .truncate(!append)
            .open(part)?;
        let (start, total) = (self.start, self.total);
        let mut downloaded = start;
        let started = Instant::now();
        let mut last_report: Option<Instant> = None;
        loop {
            check_cancel(cancel, "download cancelled")?;
            let chunk = self.runtime.block_on(self.response.chunk());
            let Some(chunk) = chunk.map_err(|e| failed(format!("download interrupted: {e}")))?
            else {
                return Ok(downloaded);
            };
            file.write_all(&chunk)?;
            downloaded += chunk.len() as u64;
            if total > 0 && downloaded > total {
                return Err(failed(format!("download exceeded {total} bytes")));
            }
            if last_report.is_none_or(|at| at.elapsed() >= REPORT_INTERVAL) {
                let speed =
                    (downloaded - start) as f64 / started.elapsed().as_secs_f64().max(0.001);
                on_progress(AssetProgress {
                    downloaded,
                    total,
                    bytes_per_second: speed,
                });
                last_report = Some(Instant::now());
            }
        }
    }
}

/// Check a finished `.part` file. A wrong one is deleted: resuming it can never succeed.
fn verify(
    part: &Path,
    expected: Expected<'_>,
    cancel: Option<&AtomicBool>,
) -> Result<(), DownloadError> {
    let size = fs::metadata(part)?.len();
    if expected.size > 0 && size != expected.size {
        let _ = fs::remove_file(part);
        return Err(failed(format!(
            "size mismatch: expected {}, got {size}",
            expected.size
        )));
    }
    if !expected.sha256.is_empty() {
        let actual = sha256_file(part, cancel)?;
        if actual != expected.sha256 {
            let _ = fs::remove_file(part);
            return Err(failed(format!(
                "digest mismatch: expected {}, got {actual}",
                expected.sha256
            )));
        }
    }
    Ok(())
}

/// Extract a `.tar.gz` into `target` and return `target/entry`.
///
/// The archive holds one top-level folder, named like the archive without
/// `.tar.gz`. No member may land outside the target, by path or by link.
pub fn extract_archive(
    archive: &Path,
    target: &Path,
    entry: &str,
    expected_entry_sha256: &str,
    cancel: Option<&AtomicBool>,
) -> Result<PathBuf, DownloadError> {
    let entry_path = target.join(entry);
    if entry_path.is_file() {
        if !expected_entry_sha256.is_empty()
            && sha256_file(&entry_path, cancel)? != expected_entry_sha256
        {
            return Err(failed(format!(
                "entry digest mismatch in {}",
                target.display()
            )));
        }
        return Ok(entry_path);
    }
    let parent = target
        .parent()
        .ok_or_else(|| failed("extract target has no parent"))?;
    fs::create_dir_all(parent)?;
    let workdir = tempfile::Builder::new()
        .prefix(".extract-")
        .tempdir_in(parent)?;
    let source = extract_tree(archive, workdir.path(), cancel)?;
    let candidate = source.join(entry);
    let archive_name = archive.file_name().unwrap_or_default().to_string_lossy();
    if !candidate.is_file() {
        return Err(failed(format!("archive {archive_name} has no {entry}")));
    }
    if !expected_entry_sha256.is_empty()
        && sha256_file(&candidate, cancel)? != expected_entry_sha256
    {
        return Err(failed(format!("entry digest mismatch in {archive_name}")));
    }
    fs::rename(&source, target)?;
    Ok(entry_path)
}

fn extract_tree(
    archive: &Path,
    workdir: &Path,
    cancel: Option<&AtomicBool>,
) -> Result<PathBuf, DownloadError> {
    let name = archive
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned();
    // Check every member first, so a bad archive writes nothing.
    let mut tar = tar::Archive::new(GzDecoder::new(File::open(archive)?));
    for member in tar.entries()? {
        let member = member?;
        check_member(
            &member.path()?,
            member.link_name()?.as_deref(),
            member.header().entry_type(),
        )?;
    }
    let mut tar = tar::Archive::new(GzDecoder::new(File::open(archive)?));
    tar.set_preserve_permissions(true);
    for member in tar.entries()? {
        check_cancel(cancel, "cancelled while extracting")?;
        member?.unpack_in(workdir)?;
    }
    let expected = name.strip_suffix(".tar.gz").unwrap_or(&name);
    let source = workdir.join(expected);
    if source.is_dir() {
        return Ok(source);
    }
    let folders: Vec<PathBuf> = fs::read_dir(workdir)?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    match folders.as_slice() {
        [only] => Ok(only.clone()),
        _ => Err(failed(format!(
            "archive {name} does not contain {expected}"
        ))),
    }
}

/// A member path and a link target must stay inside the extraction root.
fn check_member(
    path: &Path,
    link: Option<&Path>,
    kind: tar::EntryType,
) -> Result<(), DownloadError> {
    if !stays_inside(Path::new(""), path) {
        return Err(failed(format!(
            "archive member escapes the target: {}",
            path.display()
        )));
    }
    if (kind.is_symlink() || kind.is_hard_link())
        && let Some(link) = link
    {
        // A symlink resolves from its own folder; a hard link names a path from the root.
        let base = if kind.is_symlink() {
            path.parent().unwrap_or(Path::new(""))
        } else {
            Path::new("")
        };
        if !stays_inside(base, link) {
            return Err(failed(format!(
                "archive link escapes the target: {}",
                path.display()
            )));
        }
    }
    Ok(())
}

/// True when `relative`, taken from folder `base` under the root, never climbs above the root.
fn stays_inside(base: &Path, relative: &Path) -> bool {
    let mut depth: isize = base
        .components()
        .filter(|c| matches!(c, Component::Normal(_)))
        .count() as isize;
    for component in relative.components() {
        match component {
            Component::Normal(_) => depth += 1,
            Component::ParentDir => depth -= 1,
            Component::CurDir => {}
            Component::RootDir | Component::Prefix(_) => return false,
        }
        if depth < 0 {
            return false;
        }
    }
    true
}

#[cfg(test)]
pub(crate) mod test_server {
    //! A one-file HTTP server that honors Range requests, for download tests.

    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};
    use std::thread;

    pub struct Server {
        pub url: String,
        pub requests: Arc<Mutex<Vec<String>>>,
    }

    pub fn serve(body: Vec<u8>) -> Server {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/file.bin", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let seen = requests.clone();
        thread::spawn(move || {
            for stream in listener.incoming() {
                let mut stream = stream.unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut range_start = None;
                loop {
                    let mut line = String::new();
                    reader.read_line(&mut line).unwrap();
                    if line.trim().is_empty() {
                        break;
                    }
                    if let Some(value) = line.to_lowercase().strip_prefix("range: bytes=") {
                        range_start = value.trim().trim_end_matches('-').parse::<usize>().ok();
                    }
                }
                seen.lock().unwrap().push(format!("range={range_start:?}"));
                let response = match range_start {
                    Some(start) if start >= body.len() => b"HTTP/1.1 416 Range Not Satisfiable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_vec(),
                    Some(start) => {
                        let rest = &body[start..];
                        let mut out = format!("HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", rest.len()).into_bytes();
                        out.extend_from_slice(rest);
                        out
                    }
                    None => {
                        let mut out = format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len()).into_bytes();
                        out.extend_from_slice(&body);
                        out
                    }
                };
                let _ = stream.write_all(&response);
            }
        });
        Server { url, requests }
    }
}

#[cfg(test)]
mod tests {
    use super::test_server::serve;
    use super::*;

    fn sha(bytes: &[u8]) -> String {
        Sha256::digest(bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }

    fn body() -> Vec<u8> {
        (0..3_000_000u32).map(|i| (i % 251) as u8).collect()
    }

    #[test]
    fn downloads_verifies_and_reports_progress() {
        let body = body();
        let server = serve(body.clone());
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("m/model.bin");
        let mut reports = Vec::new();
        let expected = Expected {
            size: body.len() as u64,
            sha256: &sha(&body),
        };
        download_file(&server.url, &dest, expected, &mut |p| reports.push(p), None).unwrap();
        assert_eq!(fs::read(&dest).unwrap(), body);
        assert_eq!(reports.last().unwrap().downloaded, body.len() as u64);
        // A second call finds the verified file and sends no request.
        download_file(&server.url, &dest, expected, &mut |_| {}, None).unwrap();
        assert_eq!(server.requests.lock().unwrap().len(), 1);
    }

    #[test]
    fn resumes_a_partial_file() {
        let body = body();
        let server = serve(body.clone());
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("model.bin");
        fs::write(dir.path().join("model.bin.part"), &body[..1000]).unwrap();
        let expected = Expected {
            size: body.len() as u64,
            sha256: &sha(&body),
        };
        download_file(&server.url, &dest, expected, &mut |_| {}, None).unwrap();
        assert_eq!(fs::read(&dest).unwrap(), body);
        assert_eq!(server.requests.lock().unwrap()[0], "range=Some(1000)");
    }

    #[test]
    fn a_complete_part_file_is_finished_without_a_body() {
        let body = b"complete".to_vec();
        let server = serve(body.clone());
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("m.bin.part"), &body).unwrap();
        let expected = Expected {
            size: body.len() as u64,
            sha256: &sha(&body),
        };
        let dest = dir.path().join("m.bin");
        download_file(&server.url, &dest, expected, &mut |_| {}, None).unwrap();
        assert_eq!(fs::read(dest).unwrap(), body);
    }

    #[test]
    fn a_wrong_digest_fails_and_deletes_the_part() {
        let server = serve(b"tampered".to_vec());
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("m.bin");
        let expected = Expected {
            size: 0,
            sha256: &sha(b"original"),
        };
        let error = download_file(&server.url, &dest, expected, &mut |_| {}, None).unwrap_err();
        assert!(error.to_string().starts_with("digest mismatch"));
        assert!(!dir.path().join("m.bin.part").exists() && !dest.exists());
    }

    #[test]
    fn a_cancel_keeps_the_part_file() {
        let server = serve(body());
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("m.bin");
        let cancel = AtomicBool::new(true);
        let error = download_file(
            &server.url,
            &dest,
            Expected::default(),
            &mut |_| {},
            Some(&cancel),
        )
        .unwrap_err();
        assert!(matches!(error, DownloadError::Cancelled(_)));
        assert!(dir.path().join("m.bin.part").exists());
    }

    fn archive(
        dir: &Path,
        name: &str,
        members: &[(&str, &[u8])],
        link: Option<(&str, &str)>,
    ) -> PathBuf {
        let path = dir.join(name);
        let mut builder = tar::Builder::new(flate2::write::GzEncoder::new(
            File::create(&path).unwrap(),
            flate2::Compression::fast(),
        ));
        for (member, bytes) in members {
            let mut header = tar::Header::new_gnu();
            header.set_size(bytes.len() as u64);
            header.set_mode(0o755);
            header.set_cksum();
            builder.append_data(&mut header, member, *bytes).unwrap();
        }
        if let Some((member, target)) = link {
            let mut header = tar::Header::new_gnu();
            header.set_entry_type(tar::EntryType::Symlink);
            header.set_size(0);
            builder.append_link(&mut header, member, target).unwrap();
        }
        builder.into_inner().unwrap().finish().unwrap();
        path
    }

    #[test]
    fn extracts_the_top_folder_and_checks_the_entry() {
        let dir = tempfile::tempdir().unwrap();
        let path = archive(
            dir.path(),
            "rt-1.tar.gz",
            &[("rt-1/bin/tool", b"#!x"), ("rt-1/lib/a.so", b"lib")],
            Some(("rt-1/lib/b.so", "a.so")),
        );
        let target = dir.path().join("assets/rt-1");
        let entry = extract_archive(&path, &target, "bin/tool", &sha(b"#!x"), None).unwrap();
        assert_eq!(fs::read(entry).unwrap(), b"#!x");
        assert!(
            fs::symlink_metadata(target.join("lib/b.so"))
                .unwrap()
                .file_type()
                .is_symlink()
        );
        assert!(extract_archive(&path, &target, "bin/tool", &sha(b"other"), None).is_err());
    }

    #[test]
    fn rejects_members_and_links_that_escape() {
        assert!(check_member(Path::new("rt/../../etc/x"), None, tar::EntryType::Regular).is_err());
        assert!(check_member(Path::new("/etc/x"), None, tar::EntryType::Regular).is_err());
        assert!(
            check_member(
                Path::new("rt/lib/b.so"),
                Some(Path::new("../../../x")),
                tar::EntryType::Symlink
            )
            .is_err()
        );
        assert!(
            check_member(
                Path::new("rt/lib/b.so"),
                Some(Path::new("/abs")),
                tar::EntryType::Symlink
            )
            .is_err()
        );
        assert!(
            check_member(
                Path::new("rt/lib/b.so"),
                Some(Path::new("../bin/tool")),
                tar::EntryType::Symlink
            )
            .is_ok()
        );
        let dir = tempfile::tempdir().unwrap();
        let path = archive(
            dir.path(),
            "bad.tar.gz",
            &[("bad/ok", b"x")],
            Some(("bad/evil", "../../../outside")),
        );
        let error = extract_archive(&path, &dir.path().join("t/bad"), "ok", "", None).unwrap_err();
        assert!(error.to_string().contains("escapes"));
        assert!(!dir.path().join("t").join("bad").exists());
    }
}

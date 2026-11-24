//! The session WAV: processed 16 kHz int16 mono, kept going through a failing disk.

use std::collections::VecDeque;
use std::fs::File;
use std::io::{self, Seek, SeekFrom, Write};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{self, Sender};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use tracing::{debug, error, info, warn};

use super::session_folder::SessionFolder;
pub use crate::adapters::os_error::OsError;

/// A failing disk: retry this often, and hold at most this much audio in memory (about 19 MB).
pub const WAV_RETRY: Duration = Duration::from_secs(5);
pub const WAV_BACKLOG_S: f64 = 600.0;

const ENOENT: i32 = 2;

/// The size of the header `StreamingWav` writes: RIFF, one `fmt ` chunk, then `data`.
const HEADER_BYTES: u64 = 44;

/// A WAV writer that keeps the RIFF sizes current after every chunk.
///
/// The diarizer reads the file while the session records, and a reader that
/// trusts the header fails on sizes left at zero. Rewriting 8 header bytes per
/// chunk keeps the sizes close. Between the data write and the header write the
/// file holds more data than its header says, and nemo-speech reads that tail as
/// a broken chunk: a live reader takes a `snapshot` instead.
struct StreamingWav {
    path: PathBuf,
    file: File,
    bytes: u64,
}

impl StreamingWav {
    fn create(path: PathBuf, sample_rate: u32) -> Result<Self, OsError> {
        let mut file = File::create(&path).map_err(|e| OsError::from_io(&e, Some(&path), None))?;
        let mut header = Vec::with_capacity(44);
        header.extend_from_slice(b"RIFF");
        header.extend_from_slice(&36u32.to_le_bytes());
        header.extend_from_slice(b"WAVEfmt ");
        header.extend_from_slice(&16u32.to_le_bytes());
        header.extend_from_slice(&1u16.to_le_bytes()); // PCM
        header.extend_from_slice(&1u16.to_le_bytes()); // mono
        header.extend_from_slice(&sample_rate.to_le_bytes());
        header.extend_from_slice(&(sample_rate * 2).to_le_bytes());
        header.extend_from_slice(&2u16.to_le_bytes());
        header.extend_from_slice(&16u16.to_le_bytes());
        header.extend_from_slice(b"data");
        header.extend_from_slice(&0u32.to_le_bytes());
        file.write_all(&header)
            .map_err(|e| OsError::from_io(&e, None, None))?;
        Ok(Self {
            path,
            file,
            bytes: 0,
        })
    }

    fn write(&mut self, pcm: &[u8]) -> Result<(), OsError> {
        let total = self.bytes + pcm.len() as u64;
        let size = |n: u64| u32::try_from(n).unwrap_or(u32::MAX).to_le_bytes();
        let result: io::Result<()> = (|| {
            self.file.write_all(pcm)?;
            self.file.seek(SeekFrom::Start(4))?;
            self.file.write_all(&size(36 + total))?;
            self.file.seek(SeekFrom::Start(40))?;
            self.file.write_all(&size(total))?;
            self.file.seek(SeekFrom::End(0))?;
            Ok(())
        })();
        result.map_err(|e| OsError::from_io(&e, None, None))?;
        // Counted only once on disk, so a retry that wrote nothing can be deleted and its name reused.
        self.bytes = total;
        if !self.still_at_path()? {
            // Writes to a removed or moved file (Finder's Move to Trash) still succeed.
            return Err(OsError::new(
                ENOENT,
                "the recording file was moved or removed",
                Some(&self.path),
            ));
        }
        Ok(())
    }

    fn still_at_path(&self) -> Result<bool, OsError> {
        let on_disk = match std::fs::metadata(&self.path) {
            Ok(meta) => meta,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(false),
            Err(e) => return Err(OsError::from_io(&e, Some(&self.path), None)),
        };
        let opened = self
            .file
            .metadata()
            .map_err(|e| OsError::from_io(&e, None, None))?;
        Ok((on_disk.dev(), on_disk.ino()) == (opened.dev(), opened.ino()))
    }
}

pub type Notice = Box<dyn Fn(String) + Send>;
pub type KeepFiles = Box<dyn Fn() -> bool + Send>;
/// Copy a WAV that `StreamingWav` may still write, with sizes that match the copy.
///
/// The copy takes the whole frames on disk at the start. Audio written later
/// is not in it. A file without the `StreamingWav` header layout is an error.
pub fn snapshot(source: &Path, target: &Path) -> io::Result<()> {
    let mut input = File::open(source)?;
    let mut header = [0u8; HEADER_BYTES as usize];
    io::Read::read_exact(&mut input, &mut header)?;
    if &header[0..4] != b"RIFF" || &header[8..16] != b"WAVEfmt " || &header[36..40] != b"data" {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{} is not a recording WAV", source.display()),
        ));
    }
    let on_disk = input.metadata()?.len().saturating_sub(HEADER_BYTES);
    // Whole 16-bit frames only, and a RIFF size that still fits in 32 bits.
    let data = (on_disk.min(u64::from(u32::MAX - 36)) & !1) as u32;
    header[4..8].copy_from_slice(&(36 + data).to_le_bytes());
    header[40..44].copy_from_slice(&data.to_le_bytes());
    let mut output = File::create(target)?;
    output.write_all(&header)?;
    let copied = io::copy(&mut io::Read::take(input, u64::from(data)), &mut output)?;
    if copied != u64::from(data) {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            format!("{} got shorter during the copy", source.display()),
        ));
    }
    Ok(())
}

/// Send `Some(chunk)` to record it, `None` to flush and close.
pub type RecorderQueue = Sender<Option<Vec<f32>>>;

/// The session WAV, kept going through a failing disk.
///
/// A failed open or write (full disk, removed folder) keeps the audio in memory,
/// up to `WAV_BACKLOG_S`, and retries every `retry`. Each retry writes a new
/// file (recording_2.wav, ...), so a file that was valid up to the failure stays
/// valid. The meeting can always be recovered from the audio, so the audio must
/// not stop.
pub struct RecoveringWav {
    folder: Arc<SessionFolder>,
    keep_files: Option<KeepFiles>,
    names: Vec<String>,
    sample_rate: u32,
    notice: Option<Notice>,
    wav: Option<StreamingWav>,
    backlog: VecDeque<Vec<u8>>,
    backlog_bytes: usize,
    backlog_limit: usize,
    retry: Duration,
    retry_at: Option<Instant>,
    failing: bool,
    dropping: bool,
    /// Test-only fault injection: return an error to fail the next chunk write.
    #[cfg(test)]
    write_fault: Option<Box<dyn FnMut() -> Option<OsError> + Send>>,
}

impl RecoveringWav {
    pub fn new(
        folder: Arc<SessionFolder>,
        sample_rate: u32,
        notice: Option<Notice>,
        keep_files: Option<KeepFiles>,
    ) -> Self {
        Self {
            folder,
            keep_files,
            names: Vec::new(),
            sample_rate,
            notice,
            wav: None,
            backlog: VecDeque::new(),
            backlog_bytes: 0,
            backlog_limit: (WAV_BACKLOG_S * f64::from(sample_rate) * 2.0) as usize,
            retry: WAV_RETRY,
            retry_at: None,
            failing: false,
            dropping: false,
            #[cfg(test)]
            write_fault: None,
        }
    }

    #[cfg(test)]
    fn with_limits(mut self, retry: Duration, backlog_limit: usize) -> Self {
        self.retry = retry;
        self.backlog_limit = backlog_limit;
        self
    }

    pub fn write(&mut self, chunk: &[f32]) {
        let pcm: Vec<u8> = chunk
            .iter()
            .flat_map(|s| ((s.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes())
            .collect();
        self.hold(pcm);
        if self.wav.is_none() && self.retry_at.is_some_and(|at| Instant::now() < at) {
            return;
        }
        self.flush();
    }

    pub fn close(&mut self) {
        self.retry_at = None;
        self.flush();
        {
            let folder = self.folder.clone();
            let _guard = folder.hold();
            self.close_wav();
            if !self.keep_files_at_end() {
                self.delete_files();
            }
        }
        if !self.backlog.is_empty() {
            let lost = self.backlog_bytes as f64 / f64::from(self.sample_rate * 2);
            error!(target: "ltn.audio", "The disk still fails at the end of the session; {lost:.0} s of audio was not saved");
        }
    }

    fn keep_files_at_end(&self) -> bool {
        self.keep_files.as_ref().is_none_or(|keep| keep())
    }

    fn delete_files(&self) {
        let dir = self.folder.path();
        for name in &self.names {
            let path = dir.join(name);
            if let Err(e) = std::fs::remove_file(&path)
                && e.kind() != io::ErrorKind::NotFound
            {
                warn!(target: "ltn.audio", "Could not delete {}: {e}", path.display());
            }
        }
        info!(target: "ltn.audio", "save_audio is off and nothing failed; deleted the session audio");
    }

    /// The lock covers one chunk at a time, so a rename never waits for a long backlog.
    fn flush(&mut self) {
        let mut written_to: PathBuf;
        let folder = self.folder.clone();
        loop {
            let guard = folder.hold();
            let dir = guard.path();
            if self.wav.is_none() {
                match self.open_next(&dir) {
                    Ok(wav) => self.wav = Some(wav),
                    Err(e) => {
                        drop(guard);
                        return self.fail(&e);
                    }
                }
            }
            let wav = self.wav.as_mut().expect("opened above");
            // Follows a rename through SessionFolder.
            wav.path = dir.join(wav.path.file_name().expect("a WAV path has a name"));
            written_to = wav.path.clone();
            let Some(pcm) = self.backlog.front() else {
                break;
            };
            #[cfg(test)]
            if let Some(e) = self.write_fault.as_mut().and_then(|fault| fault()) {
                drop(guard);
                return self.fail(&e);
            }
            if let Err(e) = wav.write(pcm) {
                drop(guard);
                return self.fail(&e);
            }
            let written = self.backlog.pop_front().expect("front exists");
            self.backlog_bytes -= written.len();
        }
        if self.failing {
            self.failing = false;
            self.dropping = false;
            let path = written_to;
            warn!(target: "ltn.audio", "Recording continues in {}", path.display());
            let name = path
                .file_name()
                .expect("a WAV path has a name")
                .to_string_lossy()
                .into_owned();
            self.tell(format!("Audio recording continues in {name}."));
        }
    }

    fn open_next(&mut self, dir: &Path) -> Result<StreamingWav, OsError> {
        // The session folder may have been removed.
        std::fs::create_dir_all(dir).map_err(|e| OsError::from_io(&e, Some(dir), None))?;
        let mut path = dir.join("recording.wav");
        let mut number = 2;
        while path.exists() {
            path = dir.join(format!("recording_{number}.wav"));
            number += 1;
        }
        let wav = StreamingWav::create(path, self.sample_rate)?;
        self.names.push(
            wav.path
                .file_name()
                .expect("named above")
                .to_string_lossy()
                .into_owned(),
        );
        Ok(wav)
    }

    fn hold(&mut self, pcm: Vec<u8>) {
        self.backlog_bytes += pcm.len();
        self.backlog.push_back(pcm);
        while self.backlog_bytes > self.backlog_limit && self.backlog.len() > 1 {
            let dropped = self.backlog.pop_front().expect("more than one block");
            self.backlog_bytes -= dropped.len();
            if !self.dropping {
                self.dropping = true;
                error!(target: "ltn.audio", "Audio held in memory passed {WAV_BACKLOG_S:.0} s; dropping the oldest audio");
            }
        }
    }

    fn fail(&mut self, failure: &OsError) {
        let already_failing = self.failing;
        self.failing = true;
        {
            let folder = self.folder.clone();
            let _guard = folder.hold();
            self.close_wav();
        }
        self.retry_at = Some(Instant::now() + self.retry);
        if already_failing {
            debug!(target: "ltn.audio", "Recording retry failed: {failure}");
            return;
        }
        error!(target: "ltn.audio", "Recording to {} paused: {failure}", self.folder.path().display());
        self.tell(format!("Audio recording paused: {failure}. Audio is kept in memory and saved when the disk works again."));
    }

    fn close_wav(&mut self) {
        let Some(wav) = self.wav.take() else { return };
        // A failed retry holds no audio; the next one reuses the name.
        if self.failing && wav.bytes == 0 {
            let path = wav.path.clone();
            drop(wav);
            if let Err(e) = std::fs::remove_file(&path) {
                debug!(target: "ltn.audio", "closing a failed WAV file: {e}");
            }
        }
    }

    fn tell(&self, message: String) {
        if let Some(notice) = &self.notice {
            notice(message);
        }
    }
}

/// Write processed float32 chunks to the session WAV on a thread.
///
/// Send None to flush and close. The writer always drains the channel, so it
/// cannot grow while the disk fails. `notice` hears when the recording pauses
/// and resumes. When `keep_files` returns false at the end, the writer deletes
/// the files it wrote.
pub fn start_processed_recorder(
    folder: Arc<SessionFolder>,
    sample_rate: u32,
    notice: Option<Notice>,
    keep_files: Option<KeepFiles>,
) -> io::Result<(RecorderQueue, JoinHandle<()>)> {
    let (sender, receiver) = mpsc::channel::<Option<Vec<f32>>>();
    let mut recording = RecoveringWav::new(folder, sample_rate, notice, keep_files);
    let writer = thread::Builder::new()
        .name("ltn-wav".into())
        .spawn(move || {
            // A dropped sender ends the session the same way as None.
            while let Ok(Some(chunk)) = receiver.recv() {
                recording.write(&chunk);
            }
            recording.close();
        })?;
    Ok((sender, writer))
}

#[cfg(test)]
#[path = "wav_tests.rs"]
mod tests;

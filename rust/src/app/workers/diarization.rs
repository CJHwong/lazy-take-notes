//! Re-run diarization over the recording as it grows.

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use tracing::{info, warn};

use crate::app::events::{Cancel, Event, Poster};
use crate::ports::diarizer::Diarizer;

/// New audio below this size does not start a pass.
const MIN_GROWTH_BYTES: u64 = 1024;

pub struct DiarizationTiming {
    pub interval: Duration,
    pub timeout: Duration,
}

impl Default for DiarizationTiming {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(60),
            timeout: Duration::from_secs(300),
        }
    }
}

/// One full pass per interval, over the audio recorded so far.
///
/// Speaker numbers follow arrival order, so labels already on screen stay valid
/// when a new voice joins. `audio_path` is asked each pass, so a renamed session
/// folder is followed. A failed pass is skipped: one bad pass must not end the session.
///
/// `finish` says the recording is complete: one last pass covers whatever the
/// earlier passes did not, then `DiarizationFinished` is posted. `cancel` stops
/// at once, with no last pass (a shutdown).
pub fn run_diarization_worker(
    poster: &Poster,
    cancel: &Cancel,
    finish: &Cancel,
    audio_path: &dyn Fn() -> PathBuf,
    diarizer: &dyn Diarizer,
    timing: &DiarizationTiming,
) {
    let mut processed: Option<u64> = None;
    let size_of = |path: &PathBuf| {
        std::fs::metadata(path)
            .ok()
            .filter(std::fs::Metadata::is_file)
            .map(|meta| meta.len())
    };
    while !cancel.is_cancelled() {
        let last = finish.is_cancelled();
        let path = audio_path();
        // The last pass takes any new audio; a running one waits for a real gain.
        let growth = if last { 0 } else { MIN_GROWTH_BYTES };
        if let Some(size) =
            size_of(&path).filter(|&size| processed.is_none_or(|done| size > done + growth))
        {
            if last {
                info!(target: "ltn.diarize", "last diarization pass over the whole recording");
            }
            run_pass(poster, diarizer, &path, timing.timeout);
            processed = Some(size);
        }
        if last {
            if !cancel.is_cancelled() {
                poster.post(Event::DiarizationFinished);
            }
            return;
        }
        let deadline = Instant::now() + timing.interval;
        while !cancel.is_cancelled() && !finish.is_cancelled() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(200));
        }
    }
}

fn run_pass(poster: &Poster, diarizer: &dyn Diarizer, path: &std::path::Path, timeout: Duration) {
    match diarizer.diarize(path, timeout) {
        Err(e) => warn!(target: "ltn.diarize", "diarization pass failed: {e}"),
        Ok(turns) if turns.is_empty() => {}
        Ok(turns) => {
            let speakers: BTreeSet<u32> = turns.iter().map(|t| t.speaker).collect();
            info!(target: "ltn.diarize", "diarization pass: {} turns, {} speakers", turns.len(), speakers.len());
            poster.post(Event::SpeakerTurns(turns));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::speaker_turn::SpeakerTurn;
    use std::path::Path;
    use std::sync::Mutex;

    struct Scripted {
        replies: Mutex<Vec<Result<Vec<SpeakerTurn>, String>>>,
        calls: Mutex<u32>,
        cancel: Cancel,
    }

    impl Diarizer for Scripted {
        fn diarize(&self, _path: &Path, _timeout: Duration) -> Result<Vec<SpeakerTurn>, String> {
            *self.calls.lock().unwrap() += 1;
            let mut replies = self.replies.lock().unwrap();
            let reply = replies.remove(0);
            if replies.is_empty() {
                self.cancel.cancel();
            }
            reply
        }
    }

    fn turn(speaker: u32) -> SpeakerTurn {
        SpeakerTurn {
            start: 0.0,
            end: 1.0,
            speaker,
        }
    }

    #[test]
    fn a_failed_pass_is_skipped_and_the_next_one_posts() {
        let dir = tempfile::tempdir().unwrap();
        let wav = dir.path().join("recording.wav");
        let cancel = Cancel::default();
        let diarizer = Scripted {
            replies: Mutex::new(vec![Err("bad json".into()), Ok(vec![turn(0), turn(1)])]),
            calls: Mutex::new(0),
            cancel: cancel.clone(),
        };
        let (sender, mut events) = tokio::sync::mpsc::unbounded_channel();
        let grows = Mutex::new(0usize);
        let path = || {
            let mut size = grows.lock().unwrap();
            *size += 2048;
            std::fs::write(&wav, vec![0u8; *size]).unwrap();
            wav.clone()
        };
        let timing = DiarizationTiming {
            interval: Duration::ZERO,
            timeout: Duration::from_secs(1),
        };
        let finish = Cancel::default();
        run_diarization_worker(
            &Poster::new(sender),
            &cancel,
            &finish,
            &path,
            &diarizer,
            &timing,
        );
        assert_eq!(*diarizer.calls.lock().unwrap(), 2);
        assert_eq!(
            events.try_recv().unwrap(),
            Event::SpeakerTurns(vec![turn(0), turn(1)])
        );
        assert!(events.try_recv().is_err());
    }

    #[test]
    fn no_pass_runs_until_the_recording_grows() {
        let dir = tempfile::tempdir().unwrap();
        let wav = dir.path().join("recording.wav");
        std::fs::write(&wav, vec![0u8; 4096]).unwrap();
        let cancel = Cancel::default();
        let diarizer = Scripted {
            replies: Mutex::new(vec![Ok(vec![turn(0)]), Ok(vec![turn(0)])]),
            calls: Mutex::new(0),
            cancel: cancel.clone(),
        };
        let (sender, _events) = tokio::sync::mpsc::unbounded_channel();
        let passes = Mutex::new(0);
        let path = || {
            let mut count = passes.lock().unwrap();
            *count += 1;
            if *count == 4 {
                cancel.cancel();
            }
            wav.clone()
        };
        let timing = DiarizationTiming {
            interval: Duration::ZERO,
            timeout: Duration::from_secs(1),
        };
        let finish = Cancel::default();
        run_diarization_worker(
            &Poster::new(sender),
            &cancel,
            &finish,
            &path,
            &diarizer,
            &timing,
        );
        assert_eq!(
            *diarizer.calls.lock().unwrap(),
            1,
            "the same size runs once"
        );
    }

    /// A diarizer that answers every call and counts them.
    struct Steady(Mutex<u32>);

    impl Diarizer for Steady {
        fn diarize(&self, _path: &Path, _timeout: Duration) -> Result<Vec<SpeakerTurn>, String> {
            *self.0.lock().unwrap() += 1;
            Ok(vec![turn(0)])
        }
    }

    #[test]
    fn finish_runs_one_last_pass_over_new_audio_then_says_so() {
        let dir = tempfile::tempdir().unwrap();
        let wav = dir.path().join("recording.wav");
        std::fs::write(&wav, vec![0u8; 4096]).unwrap();
        let (cancel, finish) = (Cancel::default(), Cancel::default());
        let diarizer = Steady(Mutex::new(0));
        let (sender, mut events) = tokio::sync::mpsc::unbounded_channel();
        let calls = Mutex::new(0);
        let path = || {
            let mut count = calls.lock().unwrap();
            *count += 1;
            if *count == 2 {
                // A tail smaller than a running pass waits for, then the stop.
                std::fs::write(&wav, vec![0u8; 4096 + 100]).unwrap();
                finish.cancel();
            }
            wav.clone()
        };
        let timing = DiarizationTiming {
            interval: Duration::ZERO,
            timeout: Duration::from_secs(1),
        };
        run_diarization_worker(
            &Poster::new(sender),
            &cancel,
            &finish,
            &path,
            &diarizer,
            &timing,
        );
        assert_eq!(*diarizer.0.lock().unwrap(), 2, "the tail got its own pass");
        assert_eq!(
            events.try_recv().unwrap(),
            Event::SpeakerTurns(vec![turn(0)])
        );
        assert_eq!(
            events.try_recv().unwrap(),
            Event::SpeakerTurns(vec![turn(0)])
        );
        assert_eq!(events.try_recv().unwrap(), Event::DiarizationFinished);
    }

    #[test]
    fn finish_with_nothing_new_still_says_it_is_done_and_cancel_says_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let wav = dir.path().join("recording.wav");
        let (cancel, finish) = (Cancel::default(), Cancel::default());
        finish.cancel();
        let diarizer = Steady(Mutex::new(0));
        let (sender, mut events) = tokio::sync::mpsc::unbounded_channel();
        let path = || wav.clone();
        let timing = DiarizationTiming::default();
        run_diarization_worker(
            &Poster::new(sender.clone()),
            &cancel,
            &finish,
            &path,
            &diarizer,
            &timing,
        );
        assert_eq!(*diarizer.0.lock().unwrap(), 0, "no recording, no pass");
        assert_eq!(events.try_recv().unwrap(), Event::DiarizationFinished);
        cancel.cancel();
        run_diarization_worker(
            &Poster::new(sender),
            &cancel,
            &finish,
            &path,
            &diarizer,
            &timing,
        );
        assert!(events.try_recv().is_err());
    }
}

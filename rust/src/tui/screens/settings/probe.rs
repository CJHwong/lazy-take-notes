//! Probe both audio sources for the Audio page.
//!
//! The probe runs on a thread. It reports a level per source and a permission
//! state. A missing microphone grant returns exact zeros with no error, so
//! `no_signal` is the only evidence the page can show.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::ports::audio_source::AudioSource;

const PROBE_SECONDS: f64 = 15.0;
const PERMISSION_INTERVAL: f64 = 0.5;
/// Bounds one pass, so a source that never runs dry still lets the other source and Stop through.
const MAX_CHUNKS_PER_PASS: usize = 100;
const SAMPLE_RATE: u32 = 16_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    Mic,
    System,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ProbeEvent {
    Level(Source, f64),
    Permission(Source, &'static str),
    Error(String),
    Done,
}

/// Open both sources and report levels until `stop` is set or the time is up.
pub fn run_audio_probe(
    mic: Arc<dyn AudioSource>,
    system: Arc<dyn AudioSource>,
    post: &dyn Fn(ProbeEvent),
    stop: &AtomicBool,
    duration: f64,
) {
    let opened = mic
        .open(SAMPLE_RATE, 1)
        .and_then(|()| system.open(SAMPLE_RATE, 1));
    if let Err(e) = opened {
        post(ProbeEvent::Error(e.to_string()));
        close(&*mic, &*system);
        post(ProbeEvent::Done);
        return;
    }
    let mut mic_seen = false;
    let started = Instant::now();
    let mut last_permission: Option<Instant> = None;
    let sources: [(Source, &Arc<dyn AudioSource>); 2] =
        [(Source::Mic, &mic), (Source::System, &system)];
    let result = (|| -> Result<(), String> {
        while !stop.load(Ordering::SeqCst) && started.elapsed().as_secs_f64() < duration {
            for (name, source) in sources {
                let Some(samples) = read_queued(&**source)? else {
                    continue;
                };
                let rms = match samples.is_empty() {
                    true => 0.0,
                    false => {
                        let sum: f64 = samples.iter().map(|s| f64::from(*s).powi(2)).sum();
                        (sum / samples.len() as f64).sqrt()
                    }
                };
                if name == Source::Mic && samples.iter().any(|s| *s != 0.0) {
                    mic_seen = true;
                }
                post(ProbeEvent::Level(name, rms));
            }
            let due =
                last_permission.is_none_or(|at| at.elapsed().as_secs_f64() >= PERMISSION_INTERVAL);
            if due {
                last_permission = Some(Instant::now());
                post_permissions(post, mic_seen, &*system);
            }
        }
        post_permissions(post, mic_seen, &*system);
        Ok(())
    })();
    if let Err(e) = result {
        post(ProbeEvent::Error(e));
    }
    close(&*mic, &*system);
    post(ProbeEvent::Done);
}

pub fn probe_seconds() -> f64 {
    PROBE_SECONDS
}

fn post_permissions(post: &dyn Fn(ProbeEvent), mic_seen: bool, system: &dyn AudioSource) {
    let mic = if mic_seen { "allowed" } else { "no_signal" };
    post(ProbeEvent::Permission(Source::Mic, mic));
    post(ProbeEvent::Permission(Source::System, system_state(system)));
}

/// Every chunk the source has queued, as one block. The loop waits up to 50 ms
/// on each source; a mic queues a chunk about every 10 ms.
fn read_queued(source: &dyn AudioSource) -> Result<Option<Vec<f32>>, String> {
    let Some(mut samples) = source
        .read(Duration::from_millis(50))
        .map_err(|e| e.to_string())?
    else {
        return Ok(None);
    };
    for _ in 1..MAX_CHUNKS_PER_PASS {
        match source.read(Duration::ZERO).map_err(|e| e.to_string())? {
            Some(chunk) => samples.extend(chunk),
            None => break,
        }
    }
    Ok(Some(samples))
}

fn system_state(source: &dyn AudioSource) -> &'static str {
    if source.awaiting_permission() {
        "waiting"
    } else if source.system_signal_seen() {
        "allowed"
    } else {
        "no_signal"
    }
}

fn close(mic: &dyn AudioSource, system: &dyn AudioSource) {
    // Teardown must not mask the probe result.
    let _ = mic.close();
    let _ = system.close();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    use crate::ports::audio_source::AudioError;

    struct Fake {
        blocks: Mutex<Vec<Vec<f32>>>,
        fail_open: bool,
    }

    impl AudioSource for Fake {
        fn open(&self, _rate: u32, _channels: u16) -> Result<(), AudioError> {
            match self.fail_open {
                true => Err(AudioError("no device".into())),
                false => Ok(()),
            }
        }
        fn read(&self, _timeout: Duration) -> Result<Option<Vec<f32>>, AudioError> {
            Ok(self.blocks.lock().unwrap().pop())
        }
        fn close(&self) -> Result<(), AudioError> {
            Ok(())
        }
    }

    fn fake(blocks: Vec<Vec<f32>>, fail_open: bool) -> Arc<dyn AudioSource> {
        Arc::new(Fake {
            blocks: Mutex::new(blocks),
            fail_open,
        })
    }

    fn run(mic: Arc<dyn AudioSource>, system: Arc<dyn AudioSource>) -> Vec<ProbeEvent> {
        let events = Mutex::new(Vec::new());
        let stop = AtomicBool::new(false);
        run_audio_probe(
            mic,
            system,
            &|event| events.lock().unwrap().push(event),
            &stop,
            0.05,
        );
        events.into_inner().unwrap()
    }

    #[test]
    fn a_mic_with_sound_reads_allowed_and_reports_its_level() {
        let events = run(fake(vec![vec![0.5; 4]], false), fake(vec![], false));
        assert!(events.contains(&ProbeEvent::Level(Source::Mic, 0.5)));
        assert!(events.contains(&ProbeEvent::Permission(Source::Mic, "allowed")));
        assert_eq!(events.last(), Some(&ProbeEvent::Done));
    }

    #[test]
    fn a_mic_of_exact_zeros_reads_no_signal() {
        let events = run(fake(vec![vec![0.0; 4]], false), fake(vec![], false));
        assert!(events.contains(&ProbeEvent::Permission(Source::Mic, "no_signal")));
    }

    #[test]
    fn a_failed_open_reports_the_error_then_done() {
        let events = run(fake(vec![], true), fake(vec![], false));
        assert!(matches!(events[0], ProbeEvent::Error(_)));
        assert_eq!(events[1], ProbeEvent::Done);
    }
}

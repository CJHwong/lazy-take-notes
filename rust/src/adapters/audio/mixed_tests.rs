//! Ported from tests/l3_interface_adapters/gateways/test_mixed_audio_source.py.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use super::mixed::{MixTiming, MixedAudioSource, SourceHealth, TestTone};
use crate::ports::audio_source::{AudioError, AudioSource};

type ReadScript = Box<dyn Fn(&FakeSource) -> Result<Option<Vec<f32>>, AudioError> + Send + Sync>;

/// A source whose reads follow a script. It counts opens, reads and closes.
struct FakeSource {
    script: ReadScript,
    opens: AtomicUsize,
    reads: AtomicUsize,
    closes: AtomicUsize,
    failing_opens: usize,
    continuous: bool,
    queued: Mutex<Vec<Vec<f32>>>,
    tone_pending: AtomicBool,
    alive: AtomicBool,
}

impl FakeSource {
    fn scripted(
        script: impl Fn(&FakeSource) -> Result<Option<Vec<f32>>, AudioError> + Send + Sync + 'static,
    ) -> Self {
        Self {
            script: Box::new(script),
            opens: AtomicUsize::new(0),
            reads: AtomicUsize::new(0),
            closes: AtomicUsize::new(0),
            failing_opens: 0,
            continuous: false,
            queued: Mutex::default(),
            tone_pending: AtomicBool::new(false),
            alive: AtomicBool::new(true),
        }
    }

    /// Delivers these blocks once, then nothing.
    fn blocks(blocks: Vec<Vec<f32>>) -> Self {
        let source = Self::scripted(|source| {
            let mut queued = source.queued.lock().unwrap();
            if queued.is_empty() {
                drop(queued);
                thread::sleep(Duration::from_millis(5));
                return Ok(None);
            }
            Ok(Some(queued.remove(0)))
        });
        *source.queued.lock().unwrap() = blocks;
        source
    }

    fn idle() -> Self {
        Self::blocks(Vec::new())
    }

    /// Fails every read until it was opened more than `fail_opens` times, then sends `block` forever.
    fn flaky(fail_opens: usize, block: Vec<f32>) -> Self {
        Self::scripted(move |source| {
            if source.opens() <= fail_opens {
                return Err(AudioError("device gone".into()));
            }
            thread::sleep(Duration::from_millis(2));
            Ok(Some(block.clone()))
        })
    }

    /// A system tap that sends zeros, and returns the test tone only while alive.
    fn tone_loop() -> Self {
        Self::scripted(|source| {
            thread::sleep(Duration::from_millis(2));
            if source.alive.load(Ordering::SeqCst)
                && source.tone_pending.swap(false, Ordering::SeqCst)
            {
                return Ok(Some(vec![0.0005; 320]));
            }
            Ok(Some(vec![0.0; 320]))
        })
    }

    fn opens(&self) -> usize {
        self.opens.load(Ordering::SeqCst)
    }
}

impl AudioSource for FakeSource {
    fn open(&self, _sample_rate: u32, _channels: u16) -> Result<(), AudioError> {
        let count = self.opens.fetch_add(1, Ordering::SeqCst) + 1;
        if count <= self.failing_opens {
            return Err(AudioError("microphone busy".into()));
        }
        Ok(())
    }

    fn read(&self, _timeout: Duration) -> Result<Option<Vec<f32>>, AudioError> {
        self.reads.fetch_add(1, Ordering::SeqCst);
        (self.script)(self)
    }

    fn close(&self) -> Result<(), AudioError> {
        self.closes.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn continuous_capture(&self) -> bool {
        self.continuous
    }
}

fn fast() -> MixTiming {
    MixTiming {
        restart_delays: vec![Duration::from_millis(1)],
        ..MixTiming::default()
    }
}

fn mixer(
    mic: &Arc<FakeSource>,
    system: &Arc<FakeSource>,
    tone: Option<TestTone>,
    timing: MixTiming,
) -> MixedAudioSource {
    MixedAudioSource::new(mic.clone(), system.clone(), 0.01, 10.0, tone, timing)
}

fn wait_until(predicate: impl Fn() -> bool) -> bool {
    let deadline = Instant::now() + Duration::from_secs(2);
    while Instant::now() < deadline {
        if predicate() {
            return true;
        }
        thread::sleep(Duration::from_millis(5));
    }
    predicate()
}

fn read_after_readers(source: &MixedAudioSource) -> Option<Vec<f32>> {
    thread::sleep(Duration::from_millis(100));
    source.read(Duration::from_millis(500)).unwrap()
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{actual:?} vs {expected:?}");
    for (a, e) in actual.iter().zip(expected) {
        assert!((a - e).abs() < 1e-5, "{actual:?} vs {expected:?}");
    }
}

#[test]
fn both_sources_mix_under_the_peak_limit() {
    let mic = Arc::new(FakeSource::blocks(vec![vec![0.6, 0.7]]));
    let system = Arc::new(FakeSource::blocks(vec![vec![0.6, 0.7]]));
    let source = mixer(&mic, &system, None, fast());
    source.open(16_000, 1).unwrap();
    let result = read_after_readers(&source).unwrap();
    source.close().unwrap();
    assert_close(&result, &[1.2 * (0.99 / 1.4), 0.99]);
}

#[test]
fn only_the_mic_has_data() {
    let mic = Arc::new(FakeSource::blocks(vec![vec![0.3, 0.4]]));
    let source = mixer(&mic, &Arc::new(FakeSource::idle()), None, fast());
    source.open(16_000, 1).unwrap();
    let result = read_after_readers(&source).unwrap();
    source.close().unwrap();
    assert_close(&result, &[0.3, 0.4]);
}

#[test]
fn no_data_reads_none_and_close_closes_both() {
    let (mic, system) = (Arc::new(FakeSource::idle()), Arc::new(FakeSource::idle()));
    let source = mixer(&mic, &system, None, fast());
    source.open(16_000, 1).unwrap();
    assert_eq!(source.read(Duration::from_millis(50)).unwrap(), None);
    source.close().unwrap();
    assert_eq!(
        (
            mic.closes.load(Ordering::SeqCst),
            system.closes.load(Ordering::SeqCst)
        ),
        (1, 1)
    );
}

#[test]
fn a_muted_mic_leaves_only_system_audio() {
    let mic = Arc::new(FakeSource::blocks(vec![vec![0.6, 0.7]]));
    let system = Arc::new(FakeSource::blocks(vec![vec![0.4, 0.5]]));
    let source = mixer(&mic, &system, None, fast());
    source.set_mic_muted(true);
    source.open(16_000, 1).unwrap();
    let result = read_after_readers(&source).unwrap();
    source.close().unwrap();
    assert_close(&result, &[0.4, 0.5]);
}

#[test]
fn shorter_system_audio_is_padded_to_the_mic_length() {
    let mic = Arc::new(FakeSource::blocks(vec![vec![0.1, 0.2, 0.3]]));
    let system = Arc::new(FakeSource::blocks(vec![vec![0.1, 0.2]]));
    let source = mixer(&mic, &system, None, fast());
    source.open(16_000, 1).unwrap();
    let result = read_after_readers(&source).unwrap();
    source.close().unwrap();
    assert_close(&result, &[0.2, 0.4, 0.3]);
}

#[test]
fn a_source_that_stays_away_is_retried_until_close() {
    let gone = || {
        Arc::new(FakeSource::scripted(|_| {
            Err(AudioError("unplugged".into()))
        }))
    };
    let (mic, system) = (gone(), gone());
    let source = mixer(&mic, &system, None, fast());
    source.open(16_000, 1).unwrap();
    assert!(wait_until(|| mic.opens() > 10 && system.opens() > 10));
    assert!(!source.exhausted());
    let health = source.health();
    assert_eq!(
        (health.mic, health.system),
        (SourceHealth::Lost, SourceHealth::Lost)
    );
    source.close().unwrap();
}

#[test]
fn a_source_that_returns_after_many_failures_is_back_in_the_mix() {
    let mic = Arc::new(FakeSource::flaky(8, vec![0.1; 512]));
    let source = mixer(&mic, &Arc::new(FakeSource::idle()), None, fast());
    source.open(16_000, 1).unwrap();
    assert!(wait_until(
        || source.health().mic == SourceHealth::Ok && mic.opens() > 8
    ));
    assert!(source.source_errors().is_empty());
    assert!(wait_until(|| source
        .read(Duration::from_millis(50))
        .unwrap()
        .is_some_and(|b| b.iter().any(|&s| s != 0.0))));
    source.close().unwrap();
}

#[test]
fn a_source_that_fails_to_open_is_retried_while_the_other_records() {
    let mut late = FakeSource::flaky(0, vec![0.4; 160]);
    late.failing_opens = 1;
    let mic = Arc::new(late);
    let system = Arc::new(FakeSource::blocks(vec![vec![0.2; 160]]));
    let source = mixer(&mic, &system, None, fast());
    source.open(16_000, 1).unwrap();
    assert!(
        source.read(Duration::from_millis(500)).unwrap().is_some(),
        "system audio records at once"
    );
    assert!(wait_until(|| mic.opens() >= 2));
    assert_eq!(system.opens(), 1);
    source.close().unwrap();
}

#[test]
fn errors_list_in_the_order_the_failures_began() {
    let mic_down = Arc::new(AtomicBool::new(false));
    let flag = mic_down.clone();
    let mic = Arc::new(FakeSource::scripted(move |_| {
        if flag.load(Ordering::SeqCst) {
            return Err(AudioError("mic gone".into()));
        }
        thread::sleep(Duration::from_millis(2));
        Ok(Some(vec![0.1; 160]))
    }));
    let system = Arc::new(FakeSource::scripted(|_| Err(AudioError("tap gone".into()))));
    let timing = MixTiming {
        restart_delays: vec![Duration::from_millis(50)],
        ..MixTiming::default()
    };
    let source = mixer(&mic, &system, None, timing);
    source.open(16_000, 1).unwrap();
    assert!(wait_until(|| source.source_errors().len() == 1));
    mic_down.store(true, Ordering::SeqCst);
    assert!(wait_until(|| source.source_errors().len() == 2));
    let errors = source.source_errors();
    assert!(errors[0].starts_with("System audio: "), "{errors:?}");
    assert!(errors[1].starts_with("Microphone: "), "{errors:?}");
    source.close().unwrap();
}

#[test]
fn retries_back_off_to_the_last_delay() {
    let delays: Vec<f64> = MixTiming::default()
        .restart_delays
        .iter()
        .map(Duration::as_secs_f64)
        .collect();
    assert_eq!(delays, [0.3, 1.0, 2.0, 5.0]);
    assert_eq!(MixTiming::default().dead_audio, Duration::from_secs(3));
}

#[test]
fn a_mic_that_sends_exact_zeros_is_reported_silent_and_not_restarted() {
    let mic = Arc::new(FakeSource::flaky(0, vec![0.0; 512]));
    let timing = MixTiming {
        zero_alarm: Duration::from_millis(50),
        ..fast()
    };
    let source = mixer(&mic, &Arc::new(FakeSource::idle()), None, timing);
    source.open(16_000, 1).unwrap();
    assert!(wait_until(|| source.health().mic == SourceHealth::Silent));
    assert_eq!(mic.opens(), 1);
    source.close().unwrap();
}

#[test]
fn a_quiet_live_mic_and_a_muted_mic_are_not_silent() {
    for (level, muted) in [(1e-6, false), (0.1, true)] {
        let mic = Arc::new(FakeSource::flaky(0, vec![level; 512]));
        let timing = MixTiming {
            zero_alarm: Duration::from_millis(50),
            ..fast()
        };
        let source = mixer(&mic, &Arc::new(FakeSource::idle()), None, timing);
        source.set_mic_muted(muted);
        source.open(16_000, 1).unwrap();
        thread::sleep(Duration::from_millis(200));
        assert_eq!(
            source.health().mic,
            SourceHealth::Ok,
            "level {level}, muted {muted}"
        );
        source.close().unwrap();
    }
}

fn tone_timing(wait_ms: u64) -> MixTiming {
    MixTiming {
        tone_after: Duration::from_millis(50),
        tone_wait: Duration::from_millis(wait_ms),
        ..fast()
    }
}

fn tone_player(system: &Arc<FakeSource>, played: &Arc<AtomicUsize>) -> TestTone {
    let (system, played) = (system.clone(), played.clone());
    Box::new(move || {
        played.fetch_add(1, Ordering::SeqCst);
        system.tone_pending.store(true, Ordering::SeqCst);
        true
    })
}

#[test]
fn a_live_tap_returns_the_tone_and_stays_ok() {
    let system = Arc::new(FakeSource::tone_loop());
    let played = Arc::new(AtomicUsize::new(0));
    let source = mixer(
        &Arc::new(FakeSource::idle()),
        &system,
        Some(tone_player(&system, &played)),
        tone_timing(100),
    );
    source.open(16_000, 1).unwrap();
    assert!(wait_until(|| played.load(Ordering::SeqCst) >= 2));
    assert_eq!(source.health().system, SourceHealth::Ok);
    assert_eq!(system.opens(), 1);
    source.close().unwrap();
}

#[test]
fn a_deaf_tap_is_restarted_and_stays_lost_until_it_hears_again() {
    let system = Arc::new(FakeSource::tone_loop());
    system.alive.store(false, Ordering::SeqCst);
    let played = Arc::new(AtomicUsize::new(0));
    let source = mixer(
        &Arc::new(FakeSource::idle()),
        &system,
        Some(tone_player(&system, &played)),
        tone_timing(50),
    );
    source.open(16_000, 1).unwrap();
    assert!(wait_until(|| system.opens() >= 2));
    thread::sleep(Duration::from_millis(30));
    assert_eq!(
        source.health().system,
        SourceHealth::Lost,
        "a restarted tap that sends zeros is not proven"
    );
    system.alive.store(true, Ordering::SeqCst);
    assert!(wait_until(|| source.health().system == SourceHealth::Ok));
    source.close().unwrap();
}

#[test]
fn a_tone_that_cannot_play_or_no_player_never_flags_the_tap() {
    for tone in [Some(Box::new(|| false) as TestTone), None] {
        let system = Arc::new(FakeSource::tone_loop());
        system.alive.store(false, Ordering::SeqCst);
        let source = mixer(
            &Arc::new(FakeSource::idle()),
            &system,
            tone,
            tone_timing(50),
        );
        source.open(16_000, 1).unwrap();
        thread::sleep(Duration::from_millis(300));
        assert_eq!(system.opens(), 1);
        assert_eq!(source.health().system, SourceHealth::Ok);
        source.close().unwrap();
    }
}

#[test]
fn a_hung_reader_is_reported_lost_and_close_does_not_wait_for_it() {
    let release = Arc::new(AtomicBool::new(false));
    let hold = release.clone();
    let mut hanging = FakeSource::scripted(move |source| {
        if source.reads.load(Ordering::SeqCst) > 2 {
            while !hold.load(Ordering::SeqCst) {
                thread::sleep(Duration::from_millis(5));
            }
            return Ok(None);
        }
        thread::sleep(Duration::from_millis(2));
        Ok(Some(vec![0.1; 320]))
    });
    hanging.continuous = true;
    let mic = Arc::new(hanging);
    let timing = MixTiming {
        dead_audio: Duration::from_millis(100),
        join_wait: Duration::from_millis(100),
        ..fast()
    };
    let source = mixer(&mic, &Arc::new(FakeSource::idle()), None, timing);
    source.open(16_000, 1).unwrap();
    assert!(wait_until(|| source.health().mic == SourceHealth::Lost));
    let started = Instant::now();
    assert_eq!(
        source.close(),
        Err(AudioError("Audio reader did not stop".into()))
    );
    assert!(started.elapsed() < Duration::from_secs(1));
    release.store(true, Ordering::SeqCst);
}

#[test]
fn only_a_continuous_tap_with_a_tone_verifies_itself() {
    let mut tap = FakeSource::idle();
    tap.continuous = true;
    let tap = Arc::new(tap);
    let idle = || Arc::new(FakeSource::idle());
    assert!(mixer(&idle(), &tap, Some(Box::new(|| true)), fast()).verifies_system_audio());
    assert!(!mixer(&idle(), &tap, None, fast()).verifies_system_audio());
    assert!(!mixer(&idle(), &idle(), Some(Box::new(|| true)), fast()).verifies_system_audio());
}

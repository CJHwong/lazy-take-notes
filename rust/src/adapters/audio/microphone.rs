//! The microphone through cpal, delivered at 16 kHz in 512-sample blocks.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, ErrorKind, SampleRate, Stream, StreamConfig};
use serde_json::Value;
use tracing::{info, warn};

use super::buffer::AudioBuffer;
use super::rate::RateConverter;
use crate::adapters::pyrepr::py_repr_json;
use crate::ports::audio_source::{AudioError, AudioSource};

/// -66 dBFS: far below hearing and below the transcription silence threshold.
pub const TEST_TONE_AMPLITUDE: f32 = 0.0005;
const TONE_HZ: f64 = 997.0;
const TONE_SECONDS: f64 = 0.2;

fn device_name(device: &Device) -> Option<String> {
    device
        .description()
        .ok()
        .map(|description| description.name().to_string())
}

/// The name of a real input, or None for a PulseAudio monitor of an output.
fn input_entry(id: Option<&str>, name: &str) -> Option<String> {
    match id {
        Some(id) if id.ends_with(".monitor") => None,
        _ => Some(name.to_string()),
    }
}

/// The names of devices with an input, in system order, without duplicates,
/// and without the PulseAudio monitors of outputs.
pub fn list_input_devices() -> Vec<String> {
    let Ok(devices) = cpal::default_host().input_devices() else {
        return Vec::new();
    };
    let mut names: Vec<String> = Vec::new();
    for device in devices {
        let Some(name) = device_name(&device) else {
            continue;
        };
        let id = device.id().ok();
        let Some(name) = input_entry(id.as_ref().map(|id| id.id()), &name) else {
            continue;
        };
        if !names.contains(&name) {
            names.push(name);
        }
    }
    names
}

/// The device a configured name means: an exact name, else the one name that
/// holds every word of it (case-insensitive), as sounddevice matches.
pub fn match_device(names: &[String], wanted: &str) -> Option<usize> {
    if let Some(index) = names.iter().position(|name| name == wanted) {
        return Some(index);
    }
    let words: Vec<String> = wanted
        .to_lowercase()
        .split_whitespace()
        .map(str::to_string)
        .collect();
    if words.is_empty() {
        return None;
    }
    let mut matches = names.iter().enumerate().filter(|(_, name)| {
        let name = name.to_lowercase();
        words.iter().all(|word| name.contains(word.as_str()))
    });
    let first = matches.next()?.0;
    matches.next().is_none().then_some(first)
}

fn panic_text(panic: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(text) = panic.downcast_ref::<&str>() {
        return (*text).to_string();
    }
    panic
        .downcast_ref::<String>()
        .cloned()
        .unwrap_or_else(|| "panic".into())
}

pub(crate) struct Run {
    queue: AudioBuffer,
    error: Mutex<Option<String>>,
    exhausted: AtomicBool,
    closing: AtomicBool,
}

impl Run {
    pub(crate) fn new(sample_rate: u32) -> Self {
        Self {
            queue: AudioBuffer::new(sample_rate as usize, sample_rate, "microphone"),
            error: Mutex::new(None),
            exhausted: AtomicBool::new(false),
            closing: AtomicBool::new(false),
        }
    }

    /// Run callback work. A panic must not cross into Core Audio's C callback,
    /// where it aborts the process and ends the recording: it fails the stream
    /// instead, and the mixer reopens the mic.
    fn guarded(&self, work: impl FnOnce()) {
        if self.exhausted.load(Ordering::Relaxed) {
            return;
        }
        if let Err(panic) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(work)) {
            let message = format!("Microphone callback failed: {}", panic_text(&panic));
            warn!(target: "ltn.audio.sounddevice", "{message}");
            *self
                .error
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(message);
            self.exhausted.store(true, Ordering::SeqCst);
        }
    }

    pub(crate) fn read(&self, timeout: Duration) -> Result<Option<Vec<f32>>, AudioError> {
        if let Some(block) = self.queue.get(timeout) {
            return Ok(Some(block));
        }
        match self.error.lock().expect("input error lock").clone() {
            Some(message) => Err(AudioError(message)),
            None => Ok(None),
        }
    }

    pub(crate) fn drain(&self) {
        self.queue.clear();
    }

    #[cfg(target_os = "windows")]
    pub(crate) fn put(&self, block: &[f32]) {
        self.queue.put(block);
    }

    pub(crate) fn start_close(&self) {
        self.closing.store(true, Ordering::SeqCst);
    }

    pub(crate) fn exhausted(&self) -> bool {
        self.exhausted.load(Ordering::SeqCst)
    }

    fn stream_failed(&self, failure: &cpal::Error) {
        match failure.kind() {
            // An overload drops a few samples; a default-device reroute keeps the stream.
            ErrorKind::Xrun | ErrorKind::DeviceChanged => {
                warn!(target: "ltn.audio.sounddevice", "audio status: {failure}")
            }
            _ => {
                if self.closing.load(Ordering::SeqCst) {
                    return;
                }
                warn!(target: "ltn.audio.sounddevice", "microphone stream error: {failure}");
                self.error
                    .lock()
                    .expect("mic error lock")
                    .get_or_insert_with(|| "Microphone stream stopped unexpectedly".into());
                self.exhausted.store(true, Ordering::SeqCst);
            }
        }
    }
}

/// Build and start an input stream on `device` that feeds `run` at `sample_rate`.
pub(crate) fn open_stream(
    device: &Device,
    sample_rate: u32,
    channels: u16,
    run: Arc<Run>,
    label: &str,
) -> Result<Stream, AudioError> {
    let native: SampleRate = device
        .default_input_config()
        .map_err(|e| AudioError(e.to_string()))?
        .sample_rate();
    let mut converter = RateConverter::new(native, sample_rate);
    info!(
        target: "ltn.audio.sounddevice",
        "opening {label}: device={}, native_sr={native}, conversion={}, channels={channels}",
        device_name(device).unwrap_or_else(|| "?".into()),
        converter.describe(),
    );
    let config = StreamConfig {
        channels,
        sample_rate: native,
        buffer_size: cpal::BufferSize::Default,
    };
    let (data_run, error_run) = (run.clone(), run.clone());
    let stream = device
        .build_input_stream::<f32, _, _>(
            config,
            move |samples: &[f32], _| {
                if data_run.closing.load(Ordering::Relaxed) {
                    return;
                }
                data_run.guarded(|| {
                    for block in converter.push(samples) {
                        data_run.queue.put(&block);
                    }
                });
            },
            move |failure| error_run.stream_failed(&failure),
            None,
        )
        .map_err(|e| AudioError(e.to_string()))?;
    stream.play().map_err(|e| AudioError(e.to_string()))?;
    Ok(stream)
}

pub struct MicrophoneSource {
    configured: Option<String>,
    notice: Mutex<String>,
    run: Mutex<Arc<Run>>,
    stream: Mutex<Option<Stream>>,
}

impl MicrophoneSource {
    pub fn new(device: Option<String>) -> Self {
        Self {
            configured: device.filter(|name| !name.is_empty()),
            notice: Mutex::default(),
            run: Mutex::new(Arc::new(Run::new(16_000))),
            stream: Mutex::default(),
        }
    }

    fn run(&self) -> Arc<Run> {
        self.run.lock().expect("mic run lock").clone()
    }

    /// The configured device, else the default input with a notice. Core Audio is
    /// asked afresh each time, so a reconnected headset shows up on a reopen.
    fn input_device(&self) -> Result<Device, AudioError> {
        *self.notice.lock().expect("mic notice lock") = String::new();
        let host = cpal::default_host();
        if let Some(wanted) = &self.configured {
            let devices: Vec<Device> = host
                .input_devices()
                .map(Iterator::collect)
                .unwrap_or_default();
            let names: Vec<String> = devices
                .iter()
                .map(|d| device_name(d).unwrap_or_default())
                .collect();
            if let Some(index) = match_device(&names, wanted) {
                return Ok(devices[index].clone());
            }
            let quoted = py_repr_json(&Value::String(wanted.clone()));
            warn!(target: "ltn.audio.sounddevice", "input device {quoted} is unavailable, using the default");
            let fallback = host
                .default_input_device()
                .ok_or_else(|| AudioError("No input device available".into()))?;
            let name = device_name(&fallback).unwrap_or_else(|| "the default input".into());
            *self.notice.lock().expect("mic notice lock") =
                format!("Microphone {quoted} is unavailable; using {name}");
            return Ok(fallback);
        }
        host.default_input_device()
            .ok_or_else(|| AudioError("No input device available".into()))
    }
}

impl AudioSource for MicrophoneSource {
    fn open(&self, sample_rate: u32, channels: u16) -> Result<(), AudioError> {
        self.close()?;
        let run = Arc::new(Run::new(sample_rate));
        *self.run.lock().expect("mic run lock") = run.clone();
        let device = self.input_device()?;
        let stream = open_stream(&device, sample_rate, channels, run, "microphone")?;
        *self.stream.lock().expect("mic stream lock") = Some(stream);
        Ok(())
    }

    fn read(&self, timeout: Duration) -> Result<Option<Vec<f32>>, AudioError> {
        self.run().read(timeout)
    }

    fn drain(&self) {
        self.run().drain();
    }

    fn close(&self) -> Result<(), AudioError> {
        let run = self.run();
        run.start_close();
        // Dropping the stream stops the device and waits for its callback.
        drop(self.stream.lock().expect("mic stream lock").take());
        run.drain();
        Ok(())
    }

    fn exhausted(&self) -> bool {
        self.run().exhausted()
    }

    fn continuous_capture(&self) -> bool {
        true
    }

    fn device_notice(&self) -> String {
        self.notice.lock().expect("mic notice lock").clone()
    }

    fn dropped_frames(&self) -> u64 {
        self.run().queue.dropped_frames()
    }
}

/// Play an inaudible tone on the default output. False when nothing can play it.
///
/// The tone plays at the output's own rate, so no resampler sits between it and the tap.
pub fn play_test_tone() -> bool {
    match play_tone() {
        Ok(()) => true,
        Err(failure) => {
            warn!(target: "ltn.audio.sounddevice", "test tone could not play: {failure}");
            false
        }
    }
}

fn play_tone() -> Result<(), String> {
    let device = cpal::default_host()
        .default_output_device()
        .ok_or("no output device")?;
    let config = device.default_output_config().map_err(|e| e.to_string())?;
    let (rate, channels) = (config.sample_rate(), usize::from(config.channels()));
    let length = (f64::from(rate) * TONE_SECONDS) as usize;
    let mut frame = 0usize;
    let failed = Arc::new(AtomicBool::new(false));
    let failed_in_callback = failed.clone();
    let stream = device
        .build_output_stream::<f32, _, _>(
            config.config(),
            move |out: &mut [f32], _| {
                let fill = std::panic::AssertUnwindSafe(|| {
                    for slot in out.chunks_mut(channels) {
                        let value = if frame < length {
                            let phase =
                                std::f64::consts::TAU * TONE_HZ * frame as f64 / f64::from(rate);
                            TEST_TONE_AMPLITUDE * phase.sin() as f32
                        } else {
                            0.0
                        };
                        slot.fill(value);
                        frame += 1;
                    }
                });
                // A panic must not cross into Core Audio's C callback; play silence instead.
                if std::panic::catch_unwind(fill).is_err() {
                    failed_in_callback.store(true, Ordering::SeqCst);
                    out.fill(0.0);
                }
            },
            |failure| warn!(target: "ltn.audio.sounddevice", "test tone stream error: {failure}"),
            None,
        )
        .map_err(|e| e.to_string())?;
    stream.play().map_err(|e| e.to_string())?;
    // Sleep instead of waiting on the device: a dead output must not hang the caller.
    std::thread::sleep(Duration::from_secs_f64(TONE_SECONDS + 0.1));
    if failed.load(Ordering::SeqCst) {
        return Err("the tone callback failed".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_panic_in_the_callback_fails_the_stream_instead_of_the_process() {
        let run = Run::new(16_000);
        run.guarded(|| panic!("converter broke"));
        assert!(run.exhausted.load(Ordering::SeqCst));
        assert_eq!(
            run.error.lock().unwrap().as_deref(),
            Some("Microphone callback failed: converter broke")
        );
        let mut ran = false;
        run.guarded(|| ran = true);
        assert!(!ran, "a failed stream does no more work");
    }

    #[test]
    fn a_device_matches_by_exact_name_then_by_every_word() {
        let names = ["MacBook Pro Microphone", "BlackHole 2ch", "BlackHole 16ch"].map(String::from);
        assert_eq!(match_device(&names, "BlackHole 2ch"), Some(1));
        assert_eq!(match_device(&names, "macbook mic"), Some(0));
        assert_eq!(match_device(&names, "BlackHole"), None, "two devices match");
        assert_eq!(match_device(&names, "AirPods"), None);
    }

    #[test]
    fn a_pulse_monitor_is_not_an_input() {
        assert_eq!(
            input_entry(
                Some("alsa_input.pci-0000_00_1f.3.analog-stereo"),
                "Built-in Audio"
            ),
            Some("Built-in Audio".into())
        );
        assert_eq!(
            input_entry(
                Some("alsa_output.pci-0000_00_1f.3.analog-stereo.monitor"),
                "Monitor of Built-in Audio"
            ),
            None
        );
        assert_eq!(
            input_entry(None, "Built-in Audio"),
            Some("Built-in Audio".into()),
            "a device with no id stays in the list"
        );
    }
}

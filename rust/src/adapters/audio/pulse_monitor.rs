//! The system audio on Linux: the PulseAudio monitor of the default output.
//!
//! A monitor is an input source that records what an output plays. cpal's ALSA
//! fallback has no such source, so this asks for the PulseAudio host and fails
//! with one clear line when it is missing. PipeWire desktops run pipewire-pulse,
//! which provides the same protocol.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait};
use cpal::{Device, Stream};

use super::microphone::{Run, open_stream};
use crate::ports::audio_source::{AudioError, AudioSource};

/// The PulseAudio source name that records what an output plays.
pub(super) fn monitor_id(output: &str) -> String {
    format!("{output}.monitor")
}

/// The index of the monitor of `output` among `ids`.
pub(super) fn pick_monitor(ids: &[String], output: &str) -> Option<usize> {
    let wanted = monitor_id(output);
    ids.iter().position(|id| *id == wanted)
}

fn no_monitor() -> AudioError {
    AudioError("No PulseAudio or PipeWire monitor for the default output".into())
}

fn monitor_device() -> Result<Device, AudioError> {
    let host = cpal::default_host();
    if host.id() != cpal::HostId::PulseAudio {
        return Err(no_monitor());
    }
    let output = host.default_output_device().ok_or_else(no_monitor)?;
    let output_id = output.id().map_err(|_| no_monitor())?;
    let devices: Vec<Device> = host
        .input_devices()
        .map(Iterator::collect)
        .unwrap_or_default();
    let ids: Vec<String> = devices
        .iter()
        .filter_map(|device| device.id().ok().map(|id| id.id().to_string()))
        .collect();
    let index = pick_monitor(&ids, output_id.id()).ok_or_else(no_monitor)?;
    Ok(devices[index].clone())
}

pub struct PulseMonitorSource {
    run: Mutex<Arc<Run>>,
    stream: Mutex<Option<Stream>>,
}

impl PulseMonitorSource {
    pub fn new() -> Self {
        Self {
            run: Mutex::new(Arc::new(Run::new(16_000))),
            stream: Mutex::default(),
        }
    }

    fn run(&self) -> Arc<Run> {
        self.run.lock().expect("monitor run lock").clone()
    }
}

impl Default for PulseMonitorSource {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioSource for PulseMonitorSource {
    fn open(&self, sample_rate: u32, channels: u16) -> Result<(), AudioError> {
        self.close()?;
        let device = monitor_device()?;
        let run = Arc::new(Run::new(sample_rate));
        *self.run.lock().expect("monitor run lock") = run.clone();
        let stream = open_stream(&device, sample_rate, channels, run, "system monitor")?;
        *self.stream.lock().expect("monitor stream lock") = Some(stream);
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
        drop(self.stream.lock().expect("monitor stream lock").take());
        run.drain();
        Ok(())
    }

    fn exhausted(&self) -> bool {
        self.run().exhausted()
    }

    /// A monitor delivers silence all the time, so a gap means the stream is stuck.
    fn continuous_capture(&self) -> bool {
        true
    }
}

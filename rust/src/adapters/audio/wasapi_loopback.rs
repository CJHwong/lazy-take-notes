//! The system audio on Windows: WASAPI loopback of the default output.
//!
//! The render endpoint is opened as a shared capture stream, which sets
//! `AUDCLNT_STREAMFLAGS_LOOPBACK`. The engine converts the mix to mono 48 kHz
//! floats, and the source converts that to the 16 kHz blocks the session uses.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::thread::JoinHandle;
use std::time::Duration;

use wasapi::{DeviceEnumerator, Direction, SampleType, StreamMode, WaveFormat, initialize_mta};

use super::microphone::Run;
use super::rate::RateConverter;
use crate::ports::audio_source::{AudioError, AudioSource};

pub struct WasapiLoopbackSource {
    run: Mutex<Arc<Run>>,
    stop: Mutex<Arc<AtomicBool>>,
    thread: Mutex<Option<JoinHandle<()>>>,
}

impl WasapiLoopbackSource {
    pub fn new() -> Self {
        Self {
            run: Mutex::new(Arc::new(Run::new(16_000))),
            stop: Mutex::new(Arc::new(AtomicBool::new(false))),
            thread: Mutex::default(),
        }
    }

    fn run(&self) -> Arc<Run> {
        self.run.lock().expect("loopback run lock").clone()
    }
}

impl Default for WasapiLoopbackSource {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioSource for WasapiLoopbackSource {
    fn open(&self, sample_rate: u32, _channels: u16) -> Result<(), AudioError> {
        self.close()?;
        let run = Arc::new(Run::new(sample_rate));
        *self.run.lock().expect("loopback run lock") = run.clone();
        let stop = Arc::new(AtomicBool::new(false));
        *self.stop.lock().expect("loopback stop lock") = stop.clone();
        let (ready_tx, ready_rx) = mpsc::channel();
        let handle = std::thread::Builder::new()
            .name("ltn-wasapi-loopback".into())
            .spawn(move || loopback_loop(run, stop, sample_rate, ready_tx))
            .map_err(|e| AudioError(e.to_string()))?;
        *self.thread.lock().expect("loopback thread lock") = Some(handle);
        match ready_rx.recv_timeout(Duration::from_secs(5)) {
            Ok(Ok(())) => Ok(()),
            Ok(Err(error)) => Err(AudioError(error)),
            Err(_) => Err(AudioError("WASAPI loopback did not start".into())),
        }
    }

    fn read(&self, timeout: Duration) -> Result<Option<Vec<f32>>, AudioError> {
        self.run().read(timeout)
    }

    fn drain(&self) {
        self.run().drain();
    }

    fn close(&self) -> Result<(), AudioError> {
        self.stop
            .lock()
            .expect("loopback stop lock")
            .store(true, Ordering::SeqCst);
        if let Some(handle) = self.thread.lock().expect("loopback thread lock").take() {
            let _ = handle.join();
        }
        let run = self.run();
        run.start_close();
        run.drain();
        Ok(())
    }

    fn exhausted(&self) -> bool {
        self.run().exhausted()
    }

    /// An idle loopback endpoint delivers nothing, so silence is not a stall.
    /// The preflight plays a tone to prove the capture still works.
    fn continuous_capture(&self) -> bool {
        false
    }
}

fn loopback_loop(
    run: Arc<Run>,
    stop: Arc<AtomicBool>,
    sample_rate: u32,
    ready: mpsc::Sender<Result<(), String>>,
) {
    let result = (|| -> Result<(), String> {
        initialize_mta().ok().map_err(|e| e.to_string())?;
        let enumerator = DeviceEnumerator::new().map_err(|e| e.to_string())?;
        let device = enumerator
            .get_default_device(Direction::Render)
            .map_err(|e| e.to_string())?;
        let mut client = device.get_iaudioclient().map_err(|e| e.to_string())?;
        let format = WaveFormat::new(32, 32, SampleType::Float, 48_000, 1, None);
        let (_default, minimum) = client.get_device_period().map_err(|e| e.to_string())?;
        let mode = StreamMode::EventsShared {
            autoconvert: true,
            buffer_duration_hns: minimum,
        };
        // A render endpoint opened for capture is a loopback stream.
        client
            .initialize_client(&format, Direction::Capture, mode, None)
            .map_err(|e| e.to_string())?;
        let event = client.set_get_eventhandle().map_err(|e| e.to_string())?;
        let capture = client.get_audiocaptureclient().map_err(|e| e.to_string())?;
        client.start_stream().map_err(|e| e.to_string())?;
        ready.send(Ok(())).ok();
        let mut converter = RateConverter::new(48_000, sample_rate);
        let mut queue: VecDeque<u8> = VecDeque::new();
        while !stop.load(Ordering::SeqCst) {
            // A shared event-driven stream can have nothing ready; wait below.
            let _ = capture.read_from_device_to_deque(&mut queue);
            while queue.len() >= 4 {
                let mut samples = Vec::with_capacity(queue.len() / 4);
                while queue.len() >= 4 {
                    let bytes = [
                        queue.pop_front().unwrap_or(0),
                        queue.pop_front().unwrap_or(0),
                        queue.pop_front().unwrap_or(0),
                        queue.pop_front().unwrap_or(0),
                    ];
                    samples.push(f32::from_le_bytes(bytes));
                }
                for block in converter.push(&samples) {
                    run.put(&block);
                }
            }
            if event.wait_for_event(10).is_err() && stop.load(Ordering::SeqCst) {
                break;
            }
        }
        let _ = client.stop_stream();
        Ok(())
    })();
    if let Err(error) = result {
        ready.send(Err(error)).ok();
    }
}

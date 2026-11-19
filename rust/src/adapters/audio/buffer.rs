//! A queue of audio blocks bounded by duration, so a slow reader cannot hold a session's audio.

use std::collections::VecDeque;
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

use tracing::warn;

#[derive(Default)]
struct State {
    blocks: VecDeque<Vec<f32>>,
    frames: usize,
    dropped: u64,
    reported: u64,
    last_report: Option<Instant>,
}

pub struct AudioBuffer {
    label: &'static str,
    max_frames: usize,
    sample_rate: u32,
    state: Mutex<State>,
    ready: Condvar,
}

impl AudioBuffer {
    pub fn new(max_frames: usize, sample_rate: u32, label: &'static str) -> Self {
        assert!(
            max_frames > 0 && sample_rate > 0,
            "Audio buffer frame limit and sample rate must be positive"
        );
        Self {
            label,
            max_frames,
            sample_rate,
            state: Mutex::default(),
            ready: Condvar::new(),
        }
    }

    /// Add a block. The oldest blocks go first when the total passes the limit.
    pub fn put(&self, samples: &[f32]) {
        if samples.is_empty() {
            return;
        }
        let trimmed = samples.len().saturating_sub(self.max_frames);
        let samples = samples[trimmed..].to_vec();
        let mut state = self.state.lock().expect("audio buffer lock");
        let mut dropped = trimmed;
        while state.frames + samples.len() > self.max_frames {
            let Some(removed) = state.blocks.pop_front() else {
                break;
            };
            state.frames -= removed.len();
            dropped += removed.len();
        }
        state.frames += samples.len();
        state.blocks.push_back(samples);
        state.dropped += dropped as u64;
        self.ready.notify_one();
    }

    /// The oldest block, waiting up to `timeout` for one.
    pub fn get(&self, timeout: Duration) -> Option<Vec<f32>> {
        let state = self.state.lock().expect("audio buffer lock");
        let (mut state, _) = self
            .ready
            .wait_timeout_while(state, timeout, |s| s.blocks.is_empty())
            .expect("audio buffer lock");
        let block = state.blocks.pop_front()?;
        state.frames -= block.len();
        self.report_drop(&mut state);
        Some(block)
    }

    /// Every block now queued, without waiting.
    pub fn take_all(&self) -> Vec<Vec<f32>> {
        let mut state = self.state.lock().expect("audio buffer lock");
        state.frames = 0;
        state.blocks.drain(..).collect()
    }

    pub fn clear(&self) {
        self.take_all();
    }

    /// Empty the queue and zero the drop count, as a new buffer would be.
    pub fn reset(&self) {
        *self.state.lock().expect("audio buffer lock") = State::default();
    }

    pub fn is_empty(&self) -> bool {
        self.state
            .lock()
            .expect("audio buffer lock")
            .blocks
            .is_empty()
    }

    pub fn dropped_frames(&self) -> u64 {
        self.state.lock().expect("audio buffer lock").dropped
    }

    /// At most one warning a second, so a stalled reader does not flood the log.
    fn report_drop(&self, state: &mut State) {
        if state.dropped == state.reported
            || state
                .last_report
                .is_some_and(|at| at.elapsed() < Duration::from_secs(1))
        {
            return;
        }
        state.last_report = Some(Instant::now());
        state.reported = state.dropped;
        let seconds = state.dropped as f64 / f64::from(self.sample_rate);
        warn!(target: "ltn.audio.buffer", "{} buffer dropped {} frames ({seconds:.3}s total) to keep recent audio", self.label, state.dropped);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_the_newest_audio_within_the_limit() {
        let buffer = AudioBuffer::new(10, 16_000, "test");
        buffer.put(&[1.0; 4]);
        buffer.put(&[2.0; 4]);
        buffer.put(&[3.0; 4]);
        assert_eq!(buffer.dropped_frames(), 4);
        assert_eq!(buffer.get(Duration::ZERO), Some(vec![2.0; 4]));
        buffer.put(&[4.0; 25]);
        assert_eq!(
            buffer.get(Duration::ZERO),
            Some(vec![4.0; 10]),
            "an oversized block keeps its tail"
        );
        assert_eq!(buffer.dropped_frames(), 4 + 4 + 15);
        assert_eq!(buffer.get(Duration::from_millis(10)), None);
    }

    #[test]
    fn a_reader_wakes_when_a_block_arrives() {
        let buffer = std::sync::Arc::new(AudioBuffer::new(100, 16_000, "test"));
        let writer = buffer.clone();
        let thread = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(50));
            writer.put(&[0.5; 3]);
        });
        assert_eq!(buffer.get(Duration::from_secs(2)), Some(vec![0.5; 3]));
        thread.join().unwrap();
    }
}

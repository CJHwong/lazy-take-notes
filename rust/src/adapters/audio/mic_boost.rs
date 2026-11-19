//! Gain for a quiet microphone that is mixed with system audio.
//!
//! Whisper writes the louder voice in the mix, so a quiet mic loses every turn
//! to the call audio. Two sounds must not be boosted: room noise, and the call
//! itself when the speakers play it into the mic, which would put the remote
//! voice in the mix twice. The gain therefore follows the level of the user's
//! speech only, and holds at 1x while the mic's loudness follows the call's
//! loudness, and while a call turn is too new to judge.

use std::collections::VecDeque;
use std::time::Instant;

use tracing::info;

pub const FRAME_SAMPLES: usize = 256; // 16 ms at 16 kHz; a 512-sample block holds exactly two
const HISTORY_S: f64 = 4.0; // shorter windows match two unrelated voices by chance
const MAX_LAG_S: f64 = 0.3; // speaker-to-mic delay, including the tap's own latency
const JUDGE_S: f64 = 1.0; // call audio the window needs before the match means anything
const QUIET_CALL_S: f64 = 0.25; // no call sound for this long: nothing can bleed
const TURN_GAP_S: f64 = 2.0; // a call pause this long starts a new turn; older audio leaves the match
const SPEECH_OVER_FLOOR: f64 = 3.0;
const MIN_SPEECH_RMS: f64 = 0.001;
const MIN_SPEECH_S: f64 = 0.5; // one click or cough must not set the gain
const LEVEL_UP_TAU_S: f64 = 0.2;
const LEVEL_DOWN_TAU_S: f64 = 2.0;
const ATTACK_TAU_S: f64 = 0.3;
const RELEASE_TAU_S: f64 = 2.0; // slow, so the gain holds through the gaps between words
const LOUDER_TAU_S: f64 = 0.05; // fast: a louder voice must not stay boosted and clip
const BLEED_ON: f64 = 0.7;
const BLEED_OFF: f64 = 0.5;
const SYSTEM_ACTIVE_RMS: f32 = 0.003;
const LOG_COOLDOWN_S: f64 = 10.0;

/// The floor doubles every 2 s, so a steady sound stops counting as speech.
fn floor_rise_per_s() -> f64 {
    std::f64::consts::LN_2 / 2.0
}

/// Python takes `np.dot` in float32, and its order depends on the BLAS numpy was
/// built with (Accelerate, OpenBLAS), so no fixed order matches it. A sum in f64,
/// rounded to f32 once, lands closest to it.
fn rms(samples: &[f32]) -> f64 {
    let dot: f64 = samples.iter().map(|&s| f64::from(s) * f64::from(s)).sum();
    (f64::from(dot as f32) / samples.len() as f64).sqrt()
}

/// RMS of each whole 256-sample frame. A partial frame at the end is dropped.
fn frame_rms(samples: &[f32]) -> Vec<f32> {
    samples
        .chunks_exact(FRAME_SAMPLES)
        .map(|frame| (frame.iter().map(|s| s * s).sum::<f32>() / FRAME_SAMPLES as f32).sqrt())
        .collect()
}

fn follow(value: f64, target: f64, seconds: f64, tau: f64) -> f64 {
    value + (target - value) * (1.0 - (-seconds / tau).exp())
}

fn pearson(a: &[f32], b: &[f32]) -> Option<f64> {
    let n = a.len() as f64;
    let mean_a = a.iter().map(|&x| f64::from(x)).sum::<f64>() / n;
    let mean_b = b.iter().map(|&x| f64::from(x)).sum::<f64>() / n;
    let (mut cov, mut var_a, mut var_b) = (0.0, 0.0, 0.0);
    for (&x, &y) in a.iter().zip(b) {
        let (dx, dy) = (f64::from(x) - mean_a, f64::from(y) - mean_b);
        cov += dx * dy;
        var_a += dx * dx;
        var_b += dy * dy;
    }
    (var_a > 0.0 && var_b > 0.0).then(|| (cov / (var_a * var_b).sqrt()).clamp(-1.0, 1.0))
}

/// Highest Pearson correlation of the two loudness curves, with the mic up to `max_lag` frames late.
fn best_correlation(mic_env: &[f32], sys_env: &[f32], max_lag: usize) -> f64 {
    let mut best = 0.0f64;
    let lags = max_lag.min(mic_env.len().saturating_sub(2));
    if mic_env.len() < 2 {
        return best;
    }
    for lag in 0..=lags {
        let later_mic = &mic_env[lag..];
        let earlier_sys = &sys_env[..sys_env.len() - lag];
        if let Some(correlation) = pearson(later_mic, earlier_sys) {
            best = best.max(correlation);
        }
    }
    best
}

/// Smart gain for the mic side of the mix. One reader calls `apply`.
pub struct MicBoost {
    target_rms: f64,
    max_gain: f64,
    pub sample_rate: u32,
    floor: f64,
    speech_level: f64,
    speech_seconds: f64,
    gain: f64,
    bleeding: bool,
    judging: bool,
    turn_frames: usize,
    quiet_frames: usize,
    mic_env: VecDeque<f32>,
    sys_env: VecDeque<f32>,
    boost_logged: bool,
    last_bleed_log: Option<Instant>,
}

impl MicBoost {
    pub fn new(target_rms: f64, max_gain: f64, sample_rate: u32) -> Self {
        let mut boost = Self {
            target_rms,
            max_gain,
            sample_rate,
            floor: 0.0,
            speech_level: 0.0,
            speech_seconds: 0.0,
            gain: 1.0,
            bleeding: false,
            judging: false,
            turn_frames: 0,
            quiet_frames: 0,
            mic_env: VecDeque::new(),
            sys_env: VecDeque::new(),
            boost_logged: false,
            last_bleed_log: None,
        };
        boost.reset();
        boost
    }

    pub fn reset(&mut self) {
        self.floor = 0.0;
        self.speech_level = 0.0;
        self.speech_seconds = 0.0;
        self.gain = 1.0;
        self.bleeding = false;
        self.judging = false;
        self.turn_frames = 0;
        self.quiet_frames = 0;
        self.mic_env.clear();
        self.sys_env.clear();
        self.boost_logged = false;
        self.last_bleed_log = None;
    }

    pub fn gain(&self) -> f64 {
        self.gain
    }

    pub fn bleeding(&self) -> bool {
        self.bleeding
    }

    pub fn judging(&self) -> bool {
        self.judging
    }

    fn holding(&self) -> bool {
        self.bleeding || self.judging
    }

    fn frames_per_s(&self) -> f64 {
        f64::from(self.sample_rate) / FRAME_SAMPLES as f64
    }

    /// The mic block with the gain applied. `system` covers the same time window.
    pub fn apply(&mut self, mic: &[f32], system: &[f32]) -> Vec<f32> {
        if mic.is_empty() {
            return Vec::new();
        }
        let seconds = mic.len() as f64 / f64::from(self.sample_rate);
        let speaking = self.track_level(rms(mic), seconds);
        self.remember(mic, system);
        self.update_bleed();
        let start = self.gain;
        let wanted = self.wanted_gain(speaking);
        self.gain = self.next_gain(wanted, speaking, seconds);
        self.log_boost();
        if start == 1.0 && self.gain == 1.0 {
            return mic.to_vec();
        }
        // A ramp from the old gain to the new one, so the change makes no click.
        let steps = (mic.len() - 1).max(1) as f64;
        mic.iter()
            .enumerate()
            .map(|(i, sample)| {
                let ramp = if i == mic.len() - 1 {
                    self.gain
                } else {
                    start + (self.gain - start) * i as f64 / steps
                };
                sample * ramp as f32
            })
            .collect()
    }

    /// Update the noise floor and the speech level. Return whether this block is speech.
    fn track_level(&mut self, level: f64, seconds: f64) -> bool {
        if self.floor == 0.0 || level < self.floor {
            self.floor = level;
        } else {
            self.floor *= (floor_rise_per_s() * seconds).exp();
        }
        let speaking = level >= MIN_SPEECH_RMS && level > SPEECH_OVER_FLOOR * self.floor;
        if !speaking {
            return false;
        }
        let tau = if level > self.speech_level {
            LEVEL_UP_TAU_S
        } else {
            LEVEL_DOWN_TAU_S
        };
        self.speech_level = follow(self.speech_level, level, seconds, tau);
        self.speech_seconds += seconds;
        true
    }

    fn remember(&mut self, mic: &[f32], system: &[f32]) {
        let keep = (HISTORY_S * self.frames_per_s()) as usize;
        let push = |history: &mut VecDeque<f32>, frames: &[f32]| {
            history.extend(frames);
            while history.len() > keep {
                history.pop_front();
            }
        };
        push(&mut self.mic_env, &frame_rms(mic));
        let system_env = frame_rms(system);
        push(&mut self.sys_env, &system_env);
        let turn_gap = TURN_GAP_S * self.frames_per_s();
        for level in system_env {
            self.quiet_frames = if level >= SYSTEM_ACTIVE_RMS {
                0
            } else {
                self.quiet_frames + 1
            };
            self.turn_frames = if self.quiet_frames as f64 >= turn_gap {
                0
            } else {
                self.turn_frames + 1
            };
        }
    }

    fn update_bleed(&mut self) {
        let frames_per_s = self.frames_per_s();
        self.judging = false;
        if self.quiet_frames as f64 >= QUIET_CALL_S * frames_per_s {
            self.set_bleeding(false, 0.0);
            return;
        }
        if (self.turn_frames as f64) < JUDGE_S * frames_per_s {
            self.judging = true;
            return;
        }
        let turn = self.turn_frames.min(self.sys_env.len());
        let mic: Vec<f32> = self
            .mic_env
            .iter()
            .skip(self.mic_env.len().saturating_sub(turn))
            .copied()
            .collect();
        let sys: Vec<f32> = self
            .sys_env
            .iter()
            .skip(self.sys_env.len() - turn)
            .copied()
            .collect();
        let correlation = best_correlation(&mic, &sys, (MAX_LAG_S * frames_per_s) as usize);
        let threshold = if self.bleeding { BLEED_OFF } else { BLEED_ON };
        self.set_bleeding(correlation >= threshold, correlation);
    }

    fn set_bleeding(&mut self, bleeding: bool, correlation: f64) {
        if bleeding == self.bleeding {
            return;
        }
        self.bleeding = bleeding;
        if self
            .last_bleed_log
            .is_some_and(|at| at.elapsed().as_secs_f64() < LOG_COOLDOWN_S)
        {
            return;
        }
        self.last_bleed_log = Some(Instant::now());
        if bleeding {
            info!(target: "ltn.audio", "mic boost held: the mic hears the speakers (correlation {correlation:.2})");
        } else {
            info!(target: "ltn.audio", "mic boost resumed: the mic no longer follows the speakers");
        }
    }

    fn wanted_gain(&self, speaking: bool) -> f64 {
        if !speaking || self.holding() || self.speech_seconds < MIN_SPEECH_S {
            return 1.0;
        }
        (self.target_rms / self.speech_level)
            .max(1.0)
            .min(self.max_gain)
    }

    fn next_gain(&self, wanted: f64, speaking: bool, seconds: f64) -> f64 {
        if self.holding() {
            return 1.0; // within this block: the bleed reaches the mic about 0.1 s after the tap
        }
        let tau = if wanted > self.gain {
            ATTACK_TAU_S
        } else if speaking {
            LOUDER_TAU_S
        } else {
            RELEASE_TAU_S
        };
        let gain = follow(self.gain, wanted, seconds, tau);
        if wanted == 1.0 && gain < 1.01 {
            1.0
        } else {
            gain
        }
    }

    fn log_boost(&mut self) {
        if self.boost_logged || self.gain < 1.5 {
            return;
        }
        self.boost_logged = true;
        info!(target: "ltn.audio", "mic boost on: speech level {:.4}, gain {:.1}x", self.speech_level, self.gain);
    }
}

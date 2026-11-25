//! Gain for a quiet microphone that is mixed with system audio.
//!
//! Whisper writes the louder voice in the mix, so a quiet mic loses every turn
//! to the call audio. Two sounds must not be raised much: room noise, and the
//! call itself when the speakers play it into the mic, which would put the
//! remote voice in the mix twice. So the gain follows the level of the user's
//! speech, and two limits cap it:
//!
//! - Room noise: up to `max_gain` always, and more only while the boosted noise
//!   floor stays under the silence gate (a mic that a call turned far down).
//! - Echo: while the call sounds, the boosted echo stays under `ECHO_SHARE` of
//!   the call audio. The echo is measured as the mic's pickup of the speakers
//!   while the mic's loudness follows the call's. Headphones measure near zero,
//!   so they keep the full gain; laptop speakers get what their echo allows.
//!   Until a pickup is measured, a new call turn holds 1x for `JUDGE_S`.
//!
//! The gain aims the user's speech at the call's own speech level, so both
//! voices end up equally loud in the mix. The aim never drops under
//! `target_rms`, and with no call heard it is `target_rms`.

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
// A call's processing can turn a mic down about 30x, to near 0.0016 for speech.
const MIN_SPEECH_RMS: f64 = 0.0004;
const MIN_SPEECH_S: f64 = 0.5; // one click or cough must not set the gain
const LEVEL_UP_TAU_S: f64 = 0.2;
const LEVEL_DOWN_TAU_S: f64 = 2.0;
const ATTACK_TAU_S: f64 = 0.3;
const RELEASE_TAU_S: f64 = 2.0; // slow, so the gain holds through the gaps between words
const LOUDER_TAU_S: f64 = 0.05; // fast: a louder voice must not stay boosted and clip
const BLEED_ON: f64 = 0.7;
const BLEED_OFF: f64 = 0.5;
/// Call audio counts as sound above this many times its own floor, so steady
/// comfort noise or a remote room's hum is not a call turn that never ends.
const CALL_OVER_FLOOR: f32 = 3.0;
const CALL_MIN_RMS: f32 = 0.0005; // below this the call is silent, whatever its floor
/// A mic block this many times louder than the predicted echo is the user's voice.
const ECHO_MARGIN: f64 = 2.0;
/// The boosted echo stays at most this share of the call audio (-12 dB).
const ECHO_SHARE: f64 = 0.25;
/// The ceiling for a mic that a call turned far down, where the floor allows it.
const HARD_MAX_GAIN: f64 = 50.0;
/// `target_rms` is this many times the silence gate (`silence_threshold * 5`).
const TARGET_OVER_GATE: f64 = 5.0;
/// A measured pickup fades this slowly while the mic stops following the call.
const PICKUP_DOWN_TAU_S: f64 = 60.0;
/// The call's speech level averages this long, so one loud laugh does not move the aim.
const CALL_LEVEL_TAU_S: f64 = 5.0;
const LOG_COOLDOWN_S: f64 = 10.0;
const STATS_S: f64 = 30.0;

/// The floor doubles every 2 s, so a steady sound stops counting as speech.
fn floor_rise_per_s() -> f64 {
    std::f64::consts::LN_2 / 2.0
}

/// A sum in f64, rounded to f32 once.
fn rms(samples: &[f32]) -> f64 {
    let dot: f64 = samples.iter().map(|&s| f64::from(s) * f64::from(s)).sum();
    (f64::from(dot as f32) / samples.len() as f64).sqrt()
}

/// RMS of each whole 256-sample frame. A partial frame at the end is dropped.
fn frame_rms(samples: &[f32]) -> Vec<f32> {
    samples
        .as_chunks::<FRAME_SAMPLES>()
        .0
        .iter()
        .map(|frame| (frame.iter().map(|s| s * s).sum::<f32>() / FRAME_SAMPLES as f32).sqrt())
        .collect()
}

fn follow(value: f64, target: f64, seconds: f64, tau: f64) -> f64 {
    value + (target - value) * (1.0 - (-seconds / tau).exp())
}

/// A floor rises from here, so one that sat on digital silence can still reach
/// a steady sound that starts later (12 doublings, about 24 s, to 0.004).
const FLOOR_RISE_FROM: f64 = 1e-6;

/// A floor that drops to any lower level at once and rises slowly. It starts at
/// infinity, so the first level sets it.
fn track_floor(floor: f64, level: f64, seconds: f64) -> f64 {
    if level < floor {
        level
    } else {
        floor.max(FLOOR_RISE_FROM) * (floor_rise_per_s() * seconds).exp()
    }
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

/// The best match of the two loudness curves, with the mic up to `max_lag` frames late.
struct Match {
    correlation: f64,
    /// Mic loudness over call loudness at that lag (least squares): the echo's share.
    pickup: f64,
}

fn best_match(mic_env: &[f32], sys_env: &[f32], max_lag: usize) -> Match {
    let mut best = Match {
        correlation: 0.0,
        pickup: 0.0,
    };
    if mic_env.len() < 2 {
        return best;
    }
    for lag in 0..=max_lag.min(mic_env.len().saturating_sub(2)) {
        let later_mic = &mic_env[lag..];
        let earlier_sys = &sys_env[..sys_env.len() - lag];
        let Some(correlation) = pearson(later_mic, earlier_sys) else {
            continue;
        };
        if correlation > best.correlation {
            let (mut both, mut call) = (0.0, 0.0);
            for (&m, &s) in later_mic.iter().zip(earlier_sys) {
                both += f64::from(m) * f64::from(s);
                call += f64::from(s) * f64::from(s);
            }
            let pickup = if call > 0.0 { both / call } else { 0.0 };
            best = Match {
                correlation,
                pickup,
            };
        }
    }
    best
}

/// Sums for the stats line, over `STATS_S` of audio.
#[derive(Default)]
struct Stats {
    seconds: f64,
    total_seconds: f64,
    blocks: u32,
    mic: f64,
    call: f64,
    gain: f64,
    call_blocks: u32,
    speech_blocks: u32,
    boosted_blocks: u32,
    judging_blocks: u32,
    bleed_blocks: u32,
}

/// Smart gain for the mic side of the mix. One reader calls `apply`.
pub struct MicBoost {
    target_rms: f64,
    max_gain: f64,
    pub sample_rate: u32,
    floor: f64,
    call_floor: f64,
    /// The call's speech level, from blocks where it sounds. Kept while it is quiet.
    call_level: f64,
    speech_level: f64,
    speech_seconds: f64,
    gain: f64,
    bleeding: bool,
    judging: bool,
    /// The echo's share of the call, once measured. None: not known yet.
    pickup: Option<f64>,
    /// Whether `pickup` came from real bleed, or only from an upper bound.
    pickup_measured: bool,
    turn_frames: usize,
    quiet_frames: usize,
    mic_env: VecDeque<f32>,
    sys_env: VecDeque<f32>,
    boost_logged: bool,
    last_bleed_log: Option<Instant>,
    stats: Stats,
}

impl MicBoost {
    /// `max_gain`: the gain always allowed. More is allowed, up to `HARD_MAX_GAIN`,
    /// only while the boosted noise floor stays under the silence gate.
    pub fn new(target_rms: f64, max_gain: f64, sample_rate: u32) -> Self {
        let mut boost = Self {
            target_rms,
            max_gain,
            sample_rate,
            floor: 0.0,
            call_floor: 0.0,
            call_level: 0.0,
            speech_level: 0.0,
            speech_seconds: 0.0,
            gain: 1.0,
            bleeding: false,
            judging: false,
            pickup: None,
            pickup_measured: false,
            turn_frames: 0,
            quiet_frames: 0,
            mic_env: VecDeque::new(),
            sys_env: VecDeque::new(),
            boost_logged: false,
            last_bleed_log: None,
            stats: Stats::default(),
        };
        boost.reset();
        boost
    }

    pub fn reset(&mut self) {
        self.floor = f64::INFINITY;
        // Low, so the first call sound counts at once: a first frame taken as the
        // floor would hide the call until its first pause.
        self.call_floor = f64::from(CALL_MIN_RMS / CALL_OVER_FLOOR);
        self.call_level = 0.0;
        self.speech_level = 0.0;
        self.speech_seconds = 0.0;
        self.gain = 1.0;
        self.bleeding = false;
        self.judging = false;
        self.pickup = None;
        self.pickup_measured = false;
        self.turn_frames = 0;
        // No call heard yet: the call starts quiet.
        self.quiet_frames = usize::MAX / 2;
        self.mic_env.clear();
        self.sys_env.clear();
        self.boost_logged = false;
        self.last_bleed_log = None;
        self.stats = Stats::default();
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

    /// The measured echo share of the call audio, once known.
    pub fn pickup(&self) -> Option<f64> {
        self.pickup
    }

    fn frames_per_s(&self) -> f64 {
        f64::from(self.sample_rate) / FRAME_SAMPLES as f64
    }

    /// Call sound within the last `QUIET_CALL_S`: the mic may be hearing it.
    fn call_active(&self) -> bool {
        (self.quiet_frames as f64) < QUIET_CALL_S * self.frames_per_s()
    }

    /// The mic block with the gain applied. `system` covers the same time window.
    pub fn apply(&mut self, mic: &[f32], system: &[f32]) -> Vec<f32> {
        if mic.is_empty() {
            return Vec::new();
        }
        let seconds = mic.len() as f64 / f64::from(self.sample_rate);
        let level = rms(mic);
        self.floor = track_floor(self.floor, level, seconds);
        self.remember(mic, system, seconds);
        self.update_bleed(seconds);
        let speaking = self.track_speech(level, seconds, self.louder_than_echo(level));
        let start = self.gain;
        let cap = self.gain_cap();
        let wanted = self.wanted_gain(speaking, cap);
        self.gain = self.next_gain(wanted, speaking, seconds, cap);
        self.log_boost();
        self.count(level, system, speaking, seconds);
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

    /// Echo is not the user's voice, so it must not set the speech level. While
    /// the call sounds, a block counts as the user only when it is clearly louder
    /// than the echo the measured pickup predicts. With no pickup yet, it does not.
    fn louder_than_echo(&self, level: f64) -> bool {
        if !self.call_active() {
            return true;
        }
        let lag = (MAX_LAG_S * self.frames_per_s()) as usize + 2;
        let call = self
            .sys_env
            .iter()
            .rev()
            .take(lag)
            .copied()
            .fold(0.0f32, f32::max);
        self.pickup
            .is_some_and(|pickup| level > ECHO_MARGIN * pickup * f64::from(call))
    }

    /// Whether this block is speech. `learn`: let it move the speech level.
    fn track_speech(&mut self, level: f64, seconds: f64, learn: bool) -> bool {
        let speaking = level >= MIN_SPEECH_RMS && level > SPEECH_OVER_FLOOR * self.floor;
        if !speaking || !learn {
            return speaking;
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

    fn remember(&mut self, mic: &[f32], system: &[f32], seconds: f64) {
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
        self.track_call_level(system, seconds);
        let turn_gap = TURN_GAP_S * self.frames_per_s();
        let frame_s = seconds / system_env.len().max(1) as f64;
        for level in system_env {
            self.call_floor = track_floor(self.call_floor, f64::from(level), frame_s);
            let sounding = level >= CALL_MIN_RMS.max(CALL_OVER_FLOOR * self.call_floor as f32);
            self.quiet_frames = if sounding {
                0
            } else {
                self.quiet_frames.saturating_add(1)
            };
            self.turn_frames = if self.quiet_frames as f64 >= turn_gap {
                0
            } else {
                self.turn_frames + 1
            };
        }
    }

    /// Follow the call's level over the blocks where it sounds, against the floor
    /// from before this block.
    fn track_call_level(&mut self, system: &[f32], seconds: f64) {
        if system.is_empty() {
            return;
        }
        let level = rms(system);
        let sounding =
            level >= f64::from(CALL_MIN_RMS).max(f64::from(CALL_OVER_FLOOR) * self.call_floor);
        if !sounding {
            return;
        }
        self.call_level = match self.call_level == 0.0 {
            true => level,
            false => follow(self.call_level, level, seconds, CALL_LEVEL_TAU_S),
        };
    }

    /// The level the user's speech is raised to: the call's, at least `target_rms`.
    fn aim(&self) -> f64 {
        self.target_rms.max(self.call_level)
    }

    fn update_bleed(&mut self, seconds: f64) {
        let frames_per_s = self.frames_per_s();
        self.judging = false;
        if !self.call_active() {
            self.set_bleeding(false, 0.0);
            return;
        }
        if (self.turn_frames as f64) < JUDGE_S * frames_per_s {
            // A known pickup already limits the gain; only an unknown one holds 1x.
            self.judging = self.pickup.is_none();
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
        let found = best_match(&mic, &sys, (MAX_LAG_S * frames_per_s) as usize);
        let threshold = if self.bleeding { BLEED_OFF } else { BLEED_ON };
        let bleeding = found.correlation >= threshold;
        self.update_pickup(bleeding, found.pickup, &sys, seconds);
        self.set_bleeding(bleeding, found.correlation);
    }

    /// While the mic follows the call, its pickup is measured; a higher one counts
    /// at once. Otherwise any echo hides under the mic's floor, which bounds the
    /// pickup. A lower bound replaces a pickup that was only ever bounded: a short
    /// sound in a noisy room proves little, and must not hold down a later call.
    /// A measured pickup fades toward the bound, because talking over the call
    /// also stops the mic following it while the echo is still there.
    fn update_pickup(&mut self, bleeding: bool, measured: f64, sys: &[f32], seconds: f64) {
        let call = sys.iter().map(|&s| f64::from(s)).sum::<f64>() / sys.len().max(1) as f64;
        let bound = if call > 0.0 { self.floor / call } else { 0.0 };
        self.pickup_measured |= bleeding;
        self.pickup = Some(match (self.pickup, bleeding) {
            (None, true) => measured,
            (None, false) => bound,
            (Some(known), true) if measured > known => measured,
            (Some(known), true) => follow(known, measured, seconds, PICKUP_DOWN_TAU_S),
            (Some(known), false) if !self.pickup_measured => known.min(bound),
            (Some(known), false) => follow(known, known.min(bound), seconds, PICKUP_DOWN_TAU_S),
        });
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
            let pickup = self.pickup.unwrap_or(0.0);
            info!(target: "ltn.audio", "mic boost limited: the mic hears the speakers (correlation {correlation:.2}, pickup {pickup:.3}, gain up to {:.1}x)", self.gain_cap());
        } else {
            info!(target: "ltn.audio", "mic boost resumed: the mic no longer follows the speakers");
        }
    }

    /// The most gain this block allows: the room's noise, then the call's echo.
    fn gain_cap(&self) -> f64 {
        let gate = self.target_rms / TARGET_OVER_GATE;
        let quiet_room = match self.floor > 0.0 {
            true => gate / self.floor,
            false => HARD_MAX_GAIN,
        };
        let mut cap = quiet_room.min(HARD_MAX_GAIN).max(self.max_gain).max(1.0);
        if self.call_active()
            && let Some(pickup) = self.pickup.filter(|&pickup| pickup > 0.0)
        {
            cap = cap.min(ECHO_SHARE / pickup);
        }
        cap.max(1.0)
    }

    fn wanted_gain(&self, speaking: bool, cap: f64) -> f64 {
        if !speaking || self.judging || self.speech_seconds < MIN_SPEECH_S {
            return 1.0;
        }
        (self.aim() / self.speech_level).max(1.0).min(cap)
    }

    fn next_gain(&self, wanted: f64, speaking: bool, seconds: f64, cap: f64) -> f64 {
        if self.judging {
            return 1.0; // within this block: the bleed reaches the mic about 0.1 s after the tap
        }
        let tau = if wanted > self.gain {
            ATTACK_TAU_S
        } else if speaking {
            LOUDER_TAU_S
        } else {
            RELEASE_TAU_S
        };
        // The echo cap applies within the block, for the same reason as the hold.
        let gain = follow(self.gain, wanted, seconds, tau).min(cap);
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

    /// Add this block to the stats, and log them every `STATS_S` of audio.
    fn count(&mut self, level: f64, system: &[f32], speaking: bool, seconds: f64) {
        let call_active = self.call_active();
        let stats = &mut self.stats;
        stats.seconds += seconds;
        stats.total_seconds += seconds;
        stats.blocks += 1;
        stats.mic += level;
        stats.call += if system.is_empty() { 0.0 } else { rms(system) };
        stats.gain += self.gain;
        stats.call_blocks += u32::from(call_active);
        stats.speech_blocks += u32::from(speaking);
        stats.boosted_blocks += u32::from(speaking && self.gain >= 1.5);
        stats.judging_blocks += u32::from(self.judging);
        stats.bleed_blocks += u32::from(self.bleeding && call_active);
        if stats.seconds >= STATS_S {
            info!(target: "ltn.audio", "{}", self.stats_line());
            let total = self.stats.total_seconds;
            self.stats = Stats {
                total_seconds: total,
                ..Stats::default()
            };
        }
    }

    /// One line per `STATS_S`: what each source sent, and what the boost did.
    pub fn stats_line(&self) -> String {
        let stats = &self.stats;
        let blocks = f64::from(stats.blocks.max(1));
        let share = |count: u32| f64::from(count) * 100.0 / blocks;
        let speech = f64::from(stats.speech_blocks.max(1));
        let total = stats.total_seconds.round() as u64;
        let pickup = match self.pickup {
            Some(pickup) => format!("{pickup:.3}"),
            None => "unknown".into(),
        };
        format!(
            "Mic boost [{}:{:02}]: mic {:.4} (floor {:.4}), call {:.4} (floor {:.4}, sounding {:.0}%, speech {:.4}), speech level {:.4}, aim {:.4}, gain {:.1}x avg, speech boosted {:.0}%, echo pickup {pickup}, judging {:.0}%, bleed {:.0}%",
            total / 60,
            total % 60,
            stats.mic / blocks,
            self.floor,
            stats.call / blocks,
            self.call_floor,
            share(stats.call_blocks),
            self.call_level,
            self.speech_level,
            self.aim(),
            stats.gain / blocks,
            f64::from(stats.boosted_blocks) * 100.0 / speech,
            share(stats.judging_blocks),
            share(stats.bleed_blocks),
        )
    }
}

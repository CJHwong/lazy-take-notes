"""Gain for a quiet microphone that is mixed with system audio.

Whisper writes the louder voice in the mix, so a quiet mic loses every turn to
the call audio. Two sounds must not be boosted: room noise, and the call itself
when the speakers play it into the mic, which would put the remote voice in the
mix twice. The gain therefore follows the level of the user's speech only, and
holds at 1x while the mic's loudness follows the call's loudness, and while
a call turn is too new to judge.
"""

from __future__ import annotations

import logging
import math
import time

import numpy as np

log = logging.getLogger('ltn.audio')

FRAME_SAMPLES = 256  # 16 ms at 16 kHz; a 512-sample block holds exactly two
HISTORY_S = 4.0  # shorter windows match two unrelated voices by chance
MAX_LAG_S = 0.3  # speaker-to-mic delay, including the tap's own latency
JUDGE_S = 1.0  # call audio the window needs before the match means anything
QUIET_CALL_S = 0.25  # no call sound for this long: nothing can bleed
TURN_GAP_S = 2.0  # a call pause this long starts a new turn; older audio leaves the match
SPEECH_OVER_FLOOR = 3.0
MIN_SPEECH_RMS = 0.001
MIN_SPEECH_S = 0.5  # one click or cough must not set the gain
FLOOR_RISE_PER_S = math.log(2) / 2  # the floor doubles every 2 s, so a steady sound stops counting as speech
LEVEL_UP_TAU_S = 0.2
LEVEL_DOWN_TAU_S = 2.0
ATTACK_TAU_S = 0.3
RELEASE_TAU_S = 2.0  # slow, so the gain holds through the gaps between words
LOUDER_TAU_S = 0.05  # fast: a louder voice must not stay boosted and clip
BLEED_ON = 0.7
BLEED_OFF = 0.5
SYSTEM_ACTIVE_RMS = 0.003
LOG_COOLDOWN_S = 10.0


def _rms(samples: np.ndarray) -> float:
    return math.sqrt(float(np.dot(samples, samples)) / samples.size)


def _frame_rms(samples: np.ndarray) -> np.ndarray:
    count = samples.size // FRAME_SAMPLES
    frames = samples[: count * FRAME_SAMPLES].reshape(count, FRAME_SAMPLES)
    return np.sqrt((frames * frames).mean(axis=1))


def _follow(value: float, target: float, seconds: float, tau: float) -> float:
    return value + (target - value) * (1.0 - math.exp(-seconds / tau))


def _best_correlation(mic_env: np.ndarray, sys_env: np.ndarray, max_lag: int) -> float:
    """Highest Pearson correlation of the two loudness curves, with the mic up to max_lag frames late."""
    best = 0.0
    for lag in range(min(max_lag, mic_env.size - 2) + 1):
        later_mic = mic_env[lag:]
        earlier_sys = sys_env[: sys_env.size - lag]
        if later_mic.std() == 0.0 or earlier_sys.std() == 0.0:
            continue
        best = max(best, float(np.corrcoef(later_mic, earlier_sys)[0, 1]))
    return best


class MicBoost:
    """Smart gain for the mic side of the mix. Not thread-safe: one reader calls apply()."""

    def __init__(self, *, target_rms: float, max_gain: float, sample_rate: int = 16_000) -> None:
        self._target_rms = target_rms
        self._max_gain = max_gain
        self.sample_rate = sample_rate
        self.reset()

    def reset(self) -> None:
        self._floor = 0.0
        self._speech_level = 0.0
        self._speech_seconds = 0.0
        self._gain = 1.0
        self._bleeding = False
        self._judging = False
        self._turn_frames = 0
        self._quiet_frames = 0
        self._mic_env = np.zeros(0, dtype=np.float32)
        self._sys_env = np.zeros(0, dtype=np.float32)
        self._boost_logged = False
        self._last_bleed_log = -math.inf

    @property
    def gain(self) -> float:
        return self._gain

    @property
    def bleeding(self) -> bool:
        return self._bleeding

    @property
    def _holding(self) -> bool:
        return self._bleeding or self._judging

    def apply(self, mic: np.ndarray, system: np.ndarray) -> np.ndarray:
        """Return the mic chunk with the gain applied. system covers the same time window."""
        if mic.size == 0:
            return mic
        seconds = mic.size / self.sample_rate
        speaking = self._track_level(_rms(mic), seconds)
        self._remember(mic, system)
        self._update_bleed()
        start = self._gain
        self._gain = self._next_gain(self._wanted_gain(speaking), speaking, seconds)
        self._log_boost()
        if start == 1.0 and self._gain == 1.0:
            return mic
        return mic * np.linspace(start, self._gain, mic.size, dtype=np.float32)

    def _track_level(self, level: float, seconds: float) -> bool:
        """Update the noise floor and the speech level. Return whether this chunk is speech."""
        if self._floor == 0.0 or level < self._floor:
            self._floor = level
        else:
            self._floor *= math.exp(FLOOR_RISE_PER_S * seconds)
        speaking = level >= MIN_SPEECH_RMS and level > SPEECH_OVER_FLOOR * self._floor
        if not speaking:
            return False
        tau = LEVEL_UP_TAU_S if level > self._speech_level else LEVEL_DOWN_TAU_S
        self._speech_level = _follow(self._speech_level, level, seconds, tau)
        self._speech_seconds += seconds
        return True

    def _remember(self, mic: np.ndarray, system: np.ndarray) -> None:
        keep = int(HISTORY_S * self.sample_rate / FRAME_SAMPLES)
        self._mic_env = np.concatenate([self._mic_env, _frame_rms(mic)])[-keep:]
        system_env = _frame_rms(system)
        self._sys_env = np.concatenate([self._sys_env, system_env])[-keep:]
        turn_gap = TURN_GAP_S * self.sample_rate / FRAME_SAMPLES
        for level in system_env:
            self._quiet_frames = 0 if level >= SYSTEM_ACTIVE_RMS else self._quiet_frames + 1
            self._turn_frames = 0 if self._quiet_frames >= turn_gap else self._turn_frames + 1

    def _update_bleed(self) -> None:
        frames_per_s = self.sample_rate / FRAME_SAMPLES
        self._judging = False
        if self._quiet_frames >= QUIET_CALL_S * frames_per_s:
            self._set_bleeding(False, 0.0)
            return
        if self._turn_frames < JUDGE_S * frames_per_s:
            self._judging = True
            return
        turn = min(self._turn_frames, self._sys_env.size)
        correlation = _best_correlation(self._mic_env[-turn:], self._sys_env[-turn:], int(MAX_LAG_S * frames_per_s))
        if self._bleeding:
            self._set_bleeding(correlation >= BLEED_OFF, correlation)
        else:
            self._set_bleeding(correlation >= BLEED_ON, correlation)

    def _set_bleeding(self, bleeding: bool, correlation: float) -> None:
        if bleeding == self._bleeding:
            return
        self._bleeding = bleeding
        now = time.monotonic()
        if now - self._last_bleed_log < LOG_COOLDOWN_S:
            return
        self._last_bleed_log = now
        if bleeding:
            log.info('mic boost held: the mic hears the speakers (correlation %.2f)', correlation)
        else:
            log.info('mic boost resumed: the mic no longer follows the speakers')

    def _wanted_gain(self, speaking: bool) -> float:
        if not speaking or self._holding or self._speech_seconds < MIN_SPEECH_S:
            return 1.0
        return min(max(self._target_rms / self._speech_level, 1.0), self._max_gain)

    def _next_gain(self, wanted: float, speaking: bool, seconds: float) -> float:
        if self._holding:
            return 1.0  # within this block: the bleed reaches the mic about 0.1 s after the tap
        if wanted > self._gain:
            tau = ATTACK_TAU_S
        elif speaking:
            tau = LOUDER_TAU_S
        else:
            tau = RELEASE_TAU_S
        gain = _follow(self._gain, wanted, seconds, tau)
        if wanted == 1.0 and gain < 1.01:
            return 1.0
        return gain

    def _log_boost(self) -> None:
        if self._boost_logged or self._gain < 1.5:
            return
        self._boost_logged = True
        log.info('mic boost on: speech level %.4f, gain %.1fx', self._speech_level, self._gain)

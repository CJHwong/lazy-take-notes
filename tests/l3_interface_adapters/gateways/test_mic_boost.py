"""Tests for MicBoost, the gain for a quiet microphone in the mix."""

from __future__ import annotations

import numpy as np
import pytest

from lazy_take_notes.l3_interface_adapters.gateways.mic_boost import MicBoost

RATE = 16_000
CHUNK = 512  # one sounddevice block at 16 kHz (32 ms)
ROOM = 0.0005


def syllables(seconds: float, seed: int) -> np.ndarray:
    """A speech-like envelope: random bursts of 100-400 ms with gaps between them."""
    rng = np.random.default_rng(seed)
    envelope = np.zeros(int(seconds * RATE), dtype=np.float32)
    position = 0
    while position < envelope.size:
        burst = int(rng.uniform(0.1, 0.4) * RATE)
        gap = int(rng.uniform(0.05, 0.25) * RATE)
        envelope[position : position + burst] = 1.0
        position += burst + gap
    return envelope


def voice(seconds: float, level: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 1000)
    carrier = rng.standard_normal(int(seconds * RATE)).astype(np.float32)
    return level * carrier * syllables(seconds, seed)


def room(seconds: float, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (ROOM * rng.standard_normal(int(seconds * RATE))).astype(np.float32)


def run(boost: MicBoost, mic: np.ndarray, system: np.ndarray) -> np.ndarray:
    out = [boost.apply(mic[i : i + CHUNK], system[i : i + CHUNK]) for i in range(0, mic.size, CHUNK)]
    return np.concatenate(out)


def make(max_gain: float = 10.0) -> MicBoost:
    return MicBoost(target_rms=0.05, max_gain=max_gain, sample_rate=RATE)


class TestBoostsAQuietVoice:
    def test_quiet_voice_with_a_silent_call_is_boosted(self):
        boost = make()
        mic = room(6) + voice(6, 0.008, seed=1)

        run(boost, mic, np.zeros_like(mic))

        assert boost.gain > 4.0

    def test_quiet_voice_over_a_loud_call_is_boosted(self):
        """The headset case: the call plays loud, the mic hears only the user."""
        boost = make()
        mic = room(6) + voice(6, 0.008, seed=1)
        call = voice(6, 0.08, seed=2)

        run(boost, mic, call)

        assert boost.gain > 4.0
        assert not boost.bleeding

    def test_gain_is_capped(self):
        boost = make(max_gain=3.0)
        mic = room(6) + voice(6, 0.004, seed=1)

        gains = []
        for i in range(0, mic.size, CHUNK):
            boost.apply(mic[i : i + CHUNK], np.zeros(CHUNK, dtype=np.float32))
            gains.append(boost.gain)

        assert max(gains) <= 3.0
        assert max(gains) > 2.9


class TestLeavesOtherSoundsAlone:
    def test_speaker_bleed_is_not_boosted(self):
        """The speakers case: the mic hears a delayed, quieter copy of the call."""
        boost = make()
        call = voice(6, 0.08, seed=2)
        delay = int(0.1 * RATE)
        bleed = np.concatenate([np.zeros(delay, dtype=np.float32), call[:-delay]]) * 0.1
        mic = room(6) + bleed

        run(boost, mic, call)

        assert boost.bleeding
        assert boost.gain < 1.3

    def test_bleed_at_the_start_of_a_call_turn_is_not_boosted(self):
        """The user talks, so the gain is up. Then the call starts on speakers: the
        window holds too little call audio to judge yet, and the gain must not stay up."""
        boost = make()
        user = room(5) + voice(5, 0.008, seed=1)
        run(boost, user, np.zeros_like(user))
        assert boost.gain > 4.0
        call = voice(3, 0.08, seed=2)
        delay = int(0.1 * RATE)
        bleed = room(3, seed=8) + np.concatenate([np.zeros(delay, dtype=np.float32), call[:-delay]]) * 0.1

        gains = []
        for i in range(0, bleed.size, CHUNK):
            boost.apply(bleed[i : i + CHUNK], call[i : i + CHUNK])
            gains.append(boost.gain)

        first_bleed = int((0.1 + 0.3) * RATE / CHUNK)  # the bleed's delay, plus 0.3 s
        assert max(gains[first_bleed:]) < 1.3

    def test_room_noise_is_not_boosted(self):
        boost = make()
        mic = room(6)

        out = run(boost, mic, np.zeros_like(mic))

        assert boost.gain == 1.0
        np.testing.assert_array_equal(out, mic)

    def test_a_loud_voice_passes_unchanged(self):
        boost = make()
        mic = room(6) + voice(6, 0.1, seed=1)

        out = run(boost, mic, np.zeros_like(mic))

        np.testing.assert_array_equal(out, mic)

    def test_gain_returns_to_one_after_the_voice_stops(self):
        boost = make()
        speech = room(6) + voice(6, 0.008, seed=1)
        run(boost, speech, np.zeros_like(speech))
        quiet = room(10, seed=9)

        run(boost, quiet, np.zeros_like(quiet))

        assert boost.gain < 1.1


class TestSmoothing:
    def test_gain_changes_without_a_step_inside_a_chunk(self):
        boost = make()
        mic = room(6) + voice(6, 0.008, seed=1)
        run(boost, mic, np.zeros_like(mic))
        before = boost.gain
        ones = np.ones(CHUNK, dtype=np.float32)

        out = boost.apply(ones * 0.0001, np.zeros(CHUNK, dtype=np.float32))

        ramp = out / 0.0001
        assert ramp[0] == pytest.approx(before, rel=0.02)
        assert np.all(np.abs(np.diff(ramp)) < 0.01)

    def test_gain_drops_fast_when_a_louder_voice_starts(self):
        """Measured on a real mic: a quiet sound set 4.8x, then normal speech stayed
        boosted for about 2 s and clipped. A louder voice must pull the gain down at once."""
        boost = make()
        quiet = room(4) + voice(4, 0.008, seed=1)
        run(boost, quiet, np.zeros_like(quiet))
        assert boost.gain > 4.0
        loud = np.full(int(0.3 * RATE), 0.06, dtype=np.float32) * np.sign(room(0.3, seed=3))

        run(boost, loud, np.zeros_like(loud))

        assert boost.gain < 1.3

    def test_reset_clears_the_state(self):
        boost = make()
        mic = room(6) + voice(6, 0.008, seed=1)
        run(boost, mic, np.zeros_like(mic))

        boost.reset()

        assert boost.gain == 1.0
        assert not boost.bleeding

    def test_empty_chunk_is_returned_as_is(self):
        boost = make()
        empty = np.zeros(0, dtype=np.float32)

        assert boost.apply(empty, empty).size == 0

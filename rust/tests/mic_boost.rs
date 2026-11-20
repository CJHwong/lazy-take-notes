//! The mic boost against generated mic and call audio.
//!
//! Each scenario checks the gain the boost reaches and the limits it keeps:
//! room noise is not raised past the gate, and the echo of the call stays
//! well under the call audio itself.

mod common;

use common::lcg_audio;
use lazy_take_notes::adapters::audio::mic_boost::MicBoost;

const RATE: f64 = 16_000.0;
const BLOCK: usize = 512;
/// The boosted echo must stay under this share of the call (the boost aims at 0.25).
const ECHO_LIMIT: f64 = 0.3;

/// One sound: (seed, amplitude, syllables, start s, length s).
type Part = (u64, f64, bool, f64, f64);

fn track(parts: &[Part], seconds: f64) -> Vec<f32> {
    let mut audio = vec![0.0f32; (seconds * RATE) as usize];
    for &(seed, amplitude, syllables, start, length) in parts {
        let clip = lcg_audio(seed, length, amplitude, syllables);
        let begin = (start * RATE) as usize;
        for (slot, value) in audio[begin..].iter_mut().zip(&clip) {
            *slot += value;
        }
    }
    audio
}

/// The speakers play `call` into the mic, `scale` as loud and `delay` s late.
fn add_echo(mic: &mut [f32], call: &[f32], scale: f32, delay: f64) {
    let shift = (delay * RATE) as usize;
    for index in (shift..mic.len()).rev() {
        mic[index] += call[index - shift] * scale;
    }
}

/// The gain after each 512-sample block.
fn gains(mic: &[f32], call: &[f32]) -> Vec<f64> {
    let mut boost = MicBoost::new(0.05, 10.0, 16_000);
    mic.chunks_exact(BLOCK)
        .zip(call.chunks_exact(BLOCK))
        .map(|(mic, call)| {
            boost.apply(mic, call);
            boost.gain()
        })
        .collect()
}

fn during(gains: &[f64], from: f64, to: f64) -> &[f64] {
    let block = |seconds: f64| (seconds * RATE / BLOCK as f64) as usize;
    &gains[block(from)..block(to).min(gains.len())]
}

fn lowest(gains: &[f64]) -> f64 {
    gains.iter().copied().fold(f64::INFINITY, f64::min)
}

fn highest(gains: &[f64]) -> f64 {
    gains.iter().copied().fold(0.0, f64::max)
}

const SILENT: &[Part] = &[];

#[test]
fn a_quiet_mic_with_no_call_is_raised_past_the_old_10x_cap() {
    // Speech near 0.0023 RMS needs about 21x to reach 0.05.
    let mic = track(
        &[(11, 0.004, true, 0.5, 7.0), (12, 0.0003, false, 0.0, 8.0)],
        8.0,
    );
    let gains = gains(&mic, &track(SILENT, 8.0));
    let end = *gains.last().unwrap();
    assert!((15.0..30.0).contains(&end), "end gain {end}");
}

#[test]
fn a_mic_a_call_turned_30x_down_reaches_the_target() {
    // Speech at 0.0016 RMS over a floor of 0.00007: about 31x.
    let mic = track(
        &[
            (31, 0.0028, true, 0.5, 9.5),
            (32, 0.00012, false, 0.0, 10.0),
        ],
        10.0,
    );
    let gains = gains(&mic, &track(SILENT, 10.0));
    let end = *gains.last().unwrap();
    assert!((25.0..40.0).contains(&end), "end gain {end}");
}

#[test]
fn room_noise_keeps_the_old_cap() {
    // A floor near 0.0015: past 10x the noise would pass the gate, so 10x stays the cap.
    let mic = track(
        &[(41, 0.008, true, 0.5, 7.5), (42, 0.0026, false, 0.0, 8.0)],
        8.0,
    );
    let gains = gains(&mic, &track(SILENT, 8.0));
    assert!(
        highest(&gains) <= 10.0 + 1e-9,
        "max gain {}",
        highest(&gains)
    );
}

#[test]
fn steady_noise_is_never_boosted() {
    let mic = track(&[(22, 0.01, false, 0.0, 10.0)], 10.0);
    assert_eq!(highest(&gains(&mic, &track(SILENT, 10.0))), 1.0);
}

#[test]
fn a_loud_voice_drops_the_gain_at_once() {
    let mic = track(
        &[
            (20, 0.004, true, 0.0, 5.0),
            (21, 0.2, true, 5.0, 3.0),
            (12, 0.0003, false, 0.0, 8.0),
        ],
        8.0,
    );
    let gains = gains(&mic, &track(SILENT, 8.0));
    assert!(highest(during(&gains, 0.0, 5.0)) > 10.0);
    assert!(
        highest(during(&gains, 5.5, 8.0)) < 1.1,
        "{}",
        highest(&gains[90..])
    );
}

#[test]
fn headphones_keep_the_full_gain_over_an_unrelated_call() {
    let mic = track(
        &[(13, 0.004, true, 0.0, 8.0), (12, 0.0003, false, 0.0, 8.0)],
        8.0,
    );
    let call = track(&[(14, 0.05, false, 0.0, 8.0)], 8.0);
    let end = *gains(&mic, &call).last().unwrap();
    assert!(end > 15.0, "end gain {end}");
}

#[test]
fn headphones_with_turn_taking_boost_each_of_the_users_turns() {
    let call = track(
        &[(51, 0.08, true, 0.0, 4.0), (52, 0.08, true, 8.0, 4.0)],
        16.0,
    );
    let mic = track(
        &[
            (53, 0.004, true, 4.0, 4.0),
            (54, 0.004, true, 12.0, 4.0),
            (55, 0.0003, false, 0.0, 16.0),
        ],
        16.0,
    );
    let gains = gains(&mic, &call);
    assert!(lowest(during(&gains, 5.5, 8.0)) > 15.0);
    assert!(lowest(during(&gains, 13.5, 16.0)) > 15.0);
}

#[test]
fn a_call_with_constant_comfort_noise_still_lets_the_boost_work() {
    // The call never goes under the old fixed 0.003, but it is only noise. The
    // call's floor climbs to it in about 10 s; then it no longer counts as a call.
    let call = track(&[(61, 0.007, false, 0.0, 16.0)], 16.0);
    let mic = track(
        &[(62, 0.004, true, 0.0, 16.0), (63, 0.0003, false, 0.0, 16.0)],
        16.0,
    );
    let gains = gains(&mic, &call);
    let late = lowest(during(&gains, 13.0, 16.0));
    assert!(late > 15.0, "gain {late}");
}

#[test]
fn loud_laptop_speakers_get_no_boost() {
    let call = track(&[(15, 0.05, true, 0.0, 8.0)], 8.0);
    let mut mic = track(&[(16, 0.0003, false, 0.0, 8.0)], 8.0);
    add_echo(&mut mic, &call, 0.3, 0.1);
    assert!(highest(&gains(&mic, &call)) <= 1.0 + 1e-9);
}

#[test]
fn quiet_laptop_speakers_boost_the_users_turn_and_keep_the_echo_small() {
    // A call turned everything 30x down: the user's speech and the echo alike.
    let call = track(&[(71, 0.08, true, 0.0, 5.0)], 12.0);
    let mut mic = track(
        &[
            (72, 0.0028, true, 6.0, 6.0),
            (73, 0.00012, false, 0.0, 12.0),
        ],
        12.0,
    );
    let pickup = 0.02f32;
    add_echo(&mut mic, &call, pickup, 0.1);
    let gains = gains(&mic, &call);
    let remote_turn = highest(during(&gains, 0.0, 5.0));
    assert!(
        remote_turn * f64::from(pickup) <= ECHO_LIMIT,
        "gain {remote_turn} while the call talks"
    );
    assert!(lowest(during(&gains, 9.0, 12.0)) > 20.0, "the user's turn");
}

#[test]
fn talking_over_the_call_on_laptop_speakers_keeps_the_echo_small() {
    // The remote talks alone first, which measures the echo, then both talk.
    let call = track(&[(81, 0.08, true, 0.0, 10.0)], 10.0);
    let mut mic = track(
        &[(82, 0.004, true, 3.0, 7.0), (83, 0.0003, false, 0.0, 10.0)],
        10.0,
    );
    let pickup = 0.1f32;
    add_echo(&mut mic, &call, pickup, 0.1);
    let gains = gains(&mic, &call);
    let most = highest(&gains);
    assert!(most * f64::from(pickup) <= ECHO_LIMIT, "max gain {most}");
}

#[test]
fn a_pause_in_the_call_lets_the_user_be_boosted_again() {
    let mic = track(
        &[(17, 0.004, true, 0.0, 10.0), (12, 0.0003, false, 0.0, 10.0)],
        10.0,
    );
    let call = track(
        &[(18, 0.05, false, 0.0, 4.0), (19, 0.05, false, 7.0, 3.0)],
        10.0,
    );
    let gains = gains(&mic, &call);
    assert!(lowest(during(&gains, 5.5, 7.0)) > 15.0);
    assert!(*gains.last().unwrap() > 15.0);
}

#[test]
fn the_stats_line_says_what_each_source_sent() {
    let mut boost = MicBoost::new(0.05, 10.0, 16_000);
    let mic = track(&[(11, 0.004, true, 0.0, 2.0)], 2.0);
    let call = track(SILENT, 2.0);
    for (mic, call) in mic.chunks_exact(BLOCK).zip(call.chunks_exact(BLOCK)) {
        boost.apply(mic, call);
    }
    let line = boost.stats_line();
    assert!(line.starts_with("Mic boost [0:02]: mic 0.0"), "{line}");
    assert!(line.contains("echo pickup unknown"), "{line}");
}

#[test]
fn echo_louder_than_the_user_does_not_set_the_users_level() {
    // The remote's echo (0.0046 RMS) is louder than the user's speech (0.0016):
    // learnt as speech, it would hold the gain near 11x into the user's turn.
    let call = track(&[(91, 0.08, true, 0.0, 5.0)], 9.0);
    let mut mic = track(
        &[(92, 0.0028, true, 5.5, 3.5), (93, 0.00012, false, 0.0, 9.0)],
        9.0,
    );
    add_echo(&mut mic, &call, 0.1, 0.1);
    let gains = gains(&mic, &call);
    let early = lowest(during(&gains, 6.8, 7.5));
    assert!(early > 22.0, "gain {early} early in the user's turn");
}

#[test]
fn the_remote_answering_right_after_the_user_does_not_get_the_users_gain() {
    // Remote, then the user (boosted), then the remote again at once: the
    // user's gain would release over seconds, so the echo cap must cut it.
    let call = track(
        &[(101, 0.08, true, 0.0, 3.0), (102, 0.08, true, 7.0, 3.0)],
        10.0,
    );
    let mut mic = track(
        &[
            (103, 0.0028, true, 3.5, 3.5),
            (104, 0.00012, false, 0.0, 10.0),
        ],
        10.0,
    );
    let pickup = 0.1f32;
    add_echo(&mut mic, &call, pickup, 0.1);
    let gains = gains(&mic, &call);
    assert!(lowest(during(&gains, 5.5, 7.0)) > 20.0, "the user's turn");
    let answer = highest(during(&gains, 7.05, 10.0));
    assert!(
        answer * f64::from(pickup) <= ECHO_LIMIT,
        "gain {answer} while the remote answers"
    );
}

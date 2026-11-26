mod common;

use common::{FakeTranscriber, segment, speech_like};
use lazy_take_notes::domain::SAMPLE_RATE as SR;
use lazy_take_notes::domain::transcript::TranscriptSegment;
use lazy_take_notes::usecases::transcribe_audio::{
    SpeechGate, TranscribeAudio, TranscribeSettings, voiced_seconds,
};

/// Plumbing tests feed constant blocks, which the real gate calls silence.
fn plumbing(chunk: f64, overlap: f64) -> TranscribeAudio {
    let settings = TranscribeSettings {
        chunk_duration: chunk,
        overlap,
        gate: SpeechGate::LevelOnly,
        ..Default::default()
    };
    TranscribeAudio::new("zh", settings, Vec::new())
}

fn real(chunk: f64, overlap: f64) -> TranscribeAudio {
    TranscribeAudio::new(
        "en",
        TranscribeSettings {
            chunk_duration: chunk,
            overlap,
            ..Default::default()
        },
        Vec::new(),
    )
}

fn speech(seconds: f64) -> Vec<f32> {
    speech_like((seconds * SR as f64) as usize, 0.1)
}

fn texts(lines: &[TranscriptSegment]) -> Vec<&str> {
    lines.iter().map(|line| line.text.as_str()).collect()
}

fn approx(a: f64, b: f64, tolerance: f64) -> bool {
    (a - b).abs() <= tolerance
}

mod plumbing {
    use super::*;

    #[test]
    fn triggers_at_the_chunk_size_only() {
        let mut uc = plumbing(1.0, 1.0);
        uc.feed_audio(&speech(0.5));
        assert!(!uc.should_trigger());
        uc.feed_audio(&speech(0.5));
        assert!(uc.should_trigger());
        uc.reset_buffer();
        assert!(!uc.should_trigger());
    }

    #[test]
    fn process_buffer_returns_the_segments() {
        let mut fake = FakeTranscriber::returning(vec![
            segment("Hello", 0.0, 2.5),
            segment("World", 2.5, 5.0),
        ]);
        let mut uc = plumbing(2.0, 0.0);
        uc.set_session_offset(1.0);
        uc.feed_audio(&speech(1.0));
        assert_eq!(
            texts(&uc.process_buffer(&mut fake, false).unwrap()),
            ["Hello", "World"]
        );
    }

    #[test]
    fn silence_is_never_sent() {
        let mut fake = FakeTranscriber::default();
        let mut uc = plumbing(1.0, 1.0);
        uc.feed_audio(&vec![0.0; SR]);
        assert!(uc.process_buffer(&mut fake, false).unwrap().is_empty());
        uc.feed_audio(&vec![0.0; SR * 3]);
        assert!(uc.flush(&mut fake).unwrap().is_empty());
        assert!(fake.calls.is_empty());
    }

    #[test]
    fn the_tail_survives_a_prepare_and_new_audio_appends_to_it() {
        let mut uc = plumbing(2.0, 0.5);
        uc.feed_audio(&vec![0.5; SR * 2]);
        assert!(uc.prepare_buffer().is_some());
        assert_eq!(uc.buffered(), vec![0.5; SR / 2].as_slice());
        uc.feed_audio(&vec![0.7; SR]);
        assert_eq!(uc.buffered().len(), SR / 2 + SR);
        assert!(uc.buffered()[SR / 2..].iter().all(|s| *s == 0.7));
    }

    #[test]
    fn flush_needs_two_seconds() {
        let mut fake = FakeTranscriber::returning(vec![segment("Flush", 0.0, 2.0)]);
        let mut uc = plumbing(10.0, 0.0);
        uc.feed_audio(&speech(0.5));
        assert!(uc.flush(&mut fake).unwrap().is_empty());
        uc.feed_audio(&speech(2.5));
        uc.set_session_offset(3.0);
        assert_eq!(texts(&uc.flush(&mut fake).unwrap()), ["Flush"]);
    }

    #[test]
    fn a_segment_that_ends_after_the_overlap_is_kept() {
        // A segment starting just before the 1 s overlap still holds new speech.
        let mut fake = FakeTranscriber::queue(vec![
            vec![segment("First speech", 0.0, 3.0)],
            vec![segment("Second speech", 0.95, 4.5)],
        ]);
        let mut uc = plumbing(1.0, 1.0);
        uc.set_session_offset(1.0);
        uc.feed_audio(&speech(1.0));
        assert_eq!(
            texts(&uc.process_buffer(&mut fake, false).unwrap()),
            ["First speech"]
        );
        uc.feed_audio(&speech(1.0));
        uc.set_session_offset(2.0);
        assert_eq!(
            texts(&uc.process_buffer(&mut fake, false).unwrap()),
            ["Second speech"]
        );
    }

    #[test]
    fn prepare_hands_out_the_buffer_and_keeps_the_overlap() {
        let mut uc = plumbing(1.0, 1.0);
        uc.set_session_offset(1.0);
        uc.feed_audio(&speech(1.0));
        let chunk = uc.prepare_buffer().unwrap();
        assert_eq!(chunk.audio.len(), SR);
        assert!(chunk.starts_fresh);
        assert!(approx(chunk.wall_start, 0.0, 1e-9));
        assert_eq!(uc.buffered().len(), SR);
        assert!(!uc.prepare_buffer().unwrap().starts_fresh);
    }

    #[test]
    fn apply_drops_the_overlap_and_moves_times_to_the_session_clock() {
        let mut uc = plumbing(1.0, 1.0);
        let lines = uc.apply_result(
            vec![segment("in overlap", 0.0, 0.5), segment("kept", 0.9, 2.0)],
            5.0,
            false,
            false,
        );
        assert_eq!(texts(&lines), ["kept"]);
        assert!(approx(lines[0].wall_start, 5.9, 1e-9) && approx(lines[0].wall_end, 7.0, 1e-9));
        let first = uc.apply_result(vec![segment("early", 0.0, 0.5)], 0.0, true, false);
        assert_eq!(texts(&first), ["early"]);
    }

    #[test]
    fn the_last_line_is_chained_into_the_next_prompt() {
        let mut uc = plumbing(25.0, 0.0);
        uc.apply_result(
            vec![segment("first", 0.0, 1.0), segment("last", 1.0, 2.0)],
            0.0,
            true,
            false,
        );
        assert_eq!(uc.current_hints(), ["last"]);
    }

    #[test]
    fn a_pause_after_speech_triggers() {
        let settings = TranscribeSettings {
            chunk_duration: 100.0,
            overlap: 0.0,
            gate: SpeechGate::LevelOnly,
            ..Default::default()
        };
        let mut uc = TranscribeAudio::new("zh", settings, Vec::new());
        uc.feed_audio(&speech(3.0));
        assert!(!uc.should_trigger());
        uc.feed_audio(&vec![0.0; SR * 3 / 2]);
        assert!(uc.should_trigger());
    }
}

mod speech_gate {
    use super::*;

    #[test]
    fn speech_over_a_quiet_floor_counts() {
        assert!(approx(voiced_seconds(&speech(5.0), 0.01), 4.0, 0.2));
    }

    /// 10 s of a tone that grows from silence to 0.05, as rumble from wind or traffic does.
    fn swelling_tone(hz: f64) -> Vec<f32> {
        (0..SR * 10)
            .map(|index| {
                let t = index as f64 / SR as f64;
                (0.05 * (t / 10.0) * (2.0 * std::f64::consts::PI * hz * t).sin()) as f32
            })
            .collect()
    }

    #[test]
    fn rumble_below_the_speech_band_is_not_speech() {
        assert_eq!(voiced_seconds(&swelling_tone(30.0), 0.01), 0.0);
    }

    #[test]
    fn a_tone_in_the_speech_band_at_the_same_level_counts() {
        assert!(voiced_seconds(&swelling_tone(1000.0), 0.01) > 5.0);
    }

    #[test]
    fn a_click_in_silence_is_not_sent() {
        let mut audio = vec![0.0; SR * 10];
        audio[SR * 5..SR * 5 + 4000].fill(0.5);
        let mut fake = FakeTranscriber::returning(vec![segment("Thank you.", 0.0, 1.0)]);
        let mut uc = real(25.0, 0.0);
        uc.feed_audio(&audio);
        assert!(uc.process_buffer(&mut fake, false).unwrap().is_empty());
        assert!(fake.calls.is_empty());
    }

    #[test]
    fn a_short_sentence_in_a_quiet_room_is_sent() {
        let mut audio = vec![0.0; SR * 20];
        audio[SR * 5..SR * 7].copy_from_slice(&speech_like(SR * 2, 0.2));
        let mut fake =
            FakeTranscriber::returning(vec![segment("Yes, ship it on Thursday.", 0.0, 1.0)]);
        let mut uc = real(25.0, 0.0);
        uc.feed_audio(&audio);
        assert_eq!(
            texts(&uc.process_buffer(&mut fake, false).unwrap()),
            ["Yes, ship it on Thursday."]
        );
    }

    #[test]
    fn speech_after_a_long_quiet_lead_is_sent_and_starts_at_the_sound() {
        let mut audio = vec![0.0; SR * 25];
        audio[SR * 20..].copy_from_slice(&speech_like(SR * 5, 0.05));
        let mut fake =
            FakeTranscriber::returning(vec![segment("Good morning everyone.", 0.0, 24.0)]);
        let mut uc = real(25.0, 0.0);
        uc.set_session_offset(25.0);
        uc.feed_audio(&audio);
        let lines = uc.process_buffer(&mut fake, true).unwrap();
        assert!(
            approx(lines[0].wall_start, 20.0, 0.1),
            "{}",
            lines[0].wall_start
        );
        assert!(approx(lines[0].wall_end, 24.0, 1e-9));
    }

    #[test]
    fn a_segment_with_no_sound_keeps_its_start() {
        let mut audio = vec![0.0; SR * 10];
        audio[SR * 6..].copy_from_slice(&speech_like(SR * 4, 0.05));
        let mut fake = FakeTranscriber::returning(vec![
            segment("Quiet words.", 1.0, 3.0),
            segment("Loud words.", 4.0, 9.0),
        ]);
        let mut uc = real(25.0, 0.0);
        uc.set_session_offset(10.0);
        uc.feed_audio(&audio);
        let lines = uc.process_buffer(&mut fake, true).unwrap();
        assert_eq!(lines[0].wall_start, 1.0);
        assert!(approx(lines[1].wall_start, 6.0, 0.1));
    }
}

mod cleaning {
    use super::*;

    fn run(segments: Vec<TranscriptSegment>) -> Vec<String> {
        let mut fake = FakeTranscriber::returning(segments);
        let mut uc = real(25.0, 0.0);
        uc.feed_audio(&speech(3.0));
        uc.process_buffer(&mut fake, false)
            .unwrap()
            .into_iter()
            .map(|line| line.text)
            .collect()
    }

    #[test]
    fn annotations_and_punctuation_are_dropped() {
        for text in [
            "*clap*",
            "*sad music*",
            "[BLANK_AUDIO]",
            "(laughs)",
            "♪",
            ".",
            "...",
            " - ",
        ] {
            assert_eq!(
                run(vec![
                    segment(text, 0.0, 1.0),
                    segment("Real words.", 1.0, 2.0)
                ]),
                ["Real words."],
                "{text}"
            );
        }
    }

    #[test]
    fn a_dropped_line_is_not_chained() {
        let mut fake = FakeTranscriber::returning(vec![
            segment("Real words.", 0.0, 1.0),
            segment("*clap*", 1.0, 2.0),
        ]);
        let mut uc = real(25.0, 0.0);
        for _ in 0..2 {
            uc.feed_audio(&speech(3.0));
            uc.process_buffer(&mut fake, false).unwrap();
        }
        assert_eq!(fake.calls[1].2, ["Real words."]);
    }

    #[test]
    fn a_long_loop_collapses() {
        let looped = format!(
            "We agreed. {}",
            "And you can write the search plan by next Friday. ".repeat(51)
        );
        assert_eq!(
            run(vec![segment(&looped, 0.0, 1.0)]),
            ["We agreed. And you can write the search plan by next Friday."]
        );
    }

    #[test]
    fn a_quoted_echo_across_lines_is_dropped() {
        let line = "\"We're going to show you a couple of things.\"";
        let twice = format!("{line} {line}");
        let lines = run(vec![
            segment(line, 0.0, 1.0),
            segment(&twice, 1.0, 2.0),
            segment(line, 2.0, 3.0),
            segment("Back to the agenda.", 3.0, 4.0),
        ]);
        assert_eq!(lines, [line, "Back to the agenda."]);
    }

    #[test]
    fn a_sentence_said_twice_is_kept() {
        assert_eq!(
            run(vec![segment("No. No. I mean Friday.", 0.0, 1.0)]),
            ["No. No. I mean Friday."]
        );
    }

    #[test]
    fn a_repeat_of_the_previous_chunk_line_is_dropped() {
        let mut fake = FakeTranscriber::returning(vec![segment(
            "We are going to show you a couple of things.",
            0.0,
            1.0,
        )]);
        let mut uc = real(25.0, 0.0);
        uc.feed_audio(&speech(3.0));
        assert_eq!(uc.process_buffer(&mut fake, false).unwrap().len(), 1);
        uc.feed_audio(&speech(3.0));
        assert!(uc.process_buffer(&mut fake, false).unwrap().is_empty());
    }
}

mod cut_sentence {
    use super::*;

    #[test]
    fn the_last_segment_waits_for_the_next_chunk() {
        let mut fake = FakeTranscriber::queue(vec![
            vec![
                segment("First sentence.", 0.0, 1.0),
                segment("I want to review the budget for the new", 7.0, 8.0),
            ],
            vec![segment(
                "I want to review the budget for the new servers before Friday.",
                0.0,
                1.0,
            )],
        ]);
        let mut uc = real(10.0, 1.0);
        uc.feed_audio(&speech(10.0));
        uc.set_session_offset(10.0);
        assert_eq!(
            texts(&uc.process_buffer(&mut fake, false).unwrap()),
            ["First sentence."]
        );
        assert_eq!(uc.buffered().len(), SR * 3);
        uc.feed_audio(&speech(2.0));
        uc.set_session_offset(12.0);
        let second = uc.flush(&mut fake).unwrap();
        assert_eq!(
            texts(&second),
            ["I want to review the budget for the new servers before Friday."]
        );
        assert!(approx(second[0].wall_start, 7.0, 1e-9));
        assert_eq!(fake.calls[1].2, ["First sentence."]);
    }

    #[test]
    fn a_chunk_that_ends_at_a_pause_or_a_flush_keeps_its_last_segment() {
        let mut fake = FakeTranscriber::returning(vec![
            segment("First.", 0.0, 1.0),
            segment("Second.", 2.0, 3.0),
        ]);
        let mut uc = real(10.0, 1.0);
        uc.feed_audio(&speech(5.0));
        assert_eq!(
            texts(&uc.process_buffer(&mut fake, false).unwrap()),
            ["First.", "Second."]
        );

        let mut fake = FakeTranscriber::returning(vec![
            segment("First.", 0.0, 1.0),
            segment("Second.", 8.0, 9.0),
        ]);
        let mut uc = real(10.0, 1.0);
        uc.feed_audio(&speech(10.0));
        assert_eq!(texts(&uc.flush(&mut fake).unwrap()), ["First.", "Second."]);
    }

    #[test]
    fn the_live_path_holds_the_cut_segment_too() {
        let mut uc = real(10.0, 1.0);
        uc.feed_audio(&speech(10.0));
        uc.set_session_offset(10.0);
        let chunk = uc.prepare_buffer().unwrap();
        uc.feed_audio(&vec![0.25; SR * 2]); // arrives during the transcription
        let segments = vec![
            segment("First sentence.", 0.0, 1.0),
            segment("I want to review the", 7.0, 8.0),
        ];
        let kept = uc.apply_result(segments, chunk.wall_start, chunk.starts_fresh, false);
        assert_eq!(texts(&kept), ["First sentence."]);
        assert_eq!(&uc.buffered()[..SR * 3], &chunk.audio[SR * 7..]);
        assert!(uc.buffered()[SR * 3..].iter().all(|s| *s == 0.25));
        assert_eq!(uc.buffered().len(), SR * 5);
    }

    #[test]
    fn a_final_apply_keeps_its_last_segment() {
        let mut uc = real(10.0, 1.0);
        uc.feed_audio(&speech(10.0));
        let chunk = uc.prepare_buffer().unwrap();
        let segments = vec![segment("First.", 0.0, 1.0), segment("Second.", 7.0, 8.0)];
        assert_eq!(
            texts(&uc.apply_result(segments, chunk.wall_start, chunk.starts_fresh, true)),
            ["First.", "Second."]
        );
    }
}

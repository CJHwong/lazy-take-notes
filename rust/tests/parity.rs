//! The Rust port must produce what the Python version produces, byte for byte.
//! Fixtures come from `scripts/gen_fixtures.py`, run against the Python code.

use lazy_take_notes::domain::template::SessionTemplate;
use lazy_take_notes::usecases::label::sanitize_label;
use lazy_take_notes::usecases::prompt_builder::{
    build_compact_user_message, build_digest_prompt, build_label_prompt, build_quick_action_prompt,
};
use lazy_take_notes::usecases::template_builder::TemplateBuilder;
use lazy_take_notes::usecases::template_diff::diff_templates;
use lazy_take_notes::usecases::template_validator::validate_template;
use lazy_take_notes::usecases::transcribe_audio::{
    collapse_loops, is_not_speech, speech_start, voiced_frames, voiced_seconds,
};
use serde_json::Value;

fn fixture(name: &str) -> Value {
    let path = format!("{}/tests/fixtures/{name}.json", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(path).expect("fixture exists"))
        .expect("fixture is JSON")
}

fn text(value: &Value) -> &str {
    value.as_str().expect("a string")
}

fn template(value: &Value) -> SessionTemplate {
    serde_json::from_value(value.clone()).expect("a template")
}

fn strings(value: &Value) -> Vec<String> {
    value
        .as_array()
        .expect("an array")
        .iter()
        .map(|item| text(item).to_string())
        .collect()
}

/// The same LCG noise as the fixture script, so no audio arrays are committed.
fn lcg_audio(seed: u64, seconds: f64, amplitude: f64, syllables: bool) -> Vec<f32> {
    let count = (seconds * 16000.0) as usize;
    let mut state = seed;
    (0..count)
        .map(|index| {
            state = (1_664_525 * state + 1_013_904_223) % (1 << 32);
            let mut value = state as f64 / (1u64 << 32) as f64 * 2.0 - 1.0;
            if syllables && (index as f64 / 16000.0) % 0.5 >= 0.4 {
                value = 0.0;
            }
            (value * amplitude) as f32
        })
        .collect()
}

#[test]
fn digest_prompts_match_for_every_built_in_template() {
    let templates = fixture("templates");
    let cases = fixture("prompts");
    let cases = cases.as_array().unwrap();
    assert!(cases.len() >= 120);
    for case in cases {
        let tmpl = template(&templates[text(&case["template"])]);
        let out = build_digest_prompt(
            &tmpl,
            &strings(&case["buffer"]),
            case["is_final"].as_bool().unwrap(),
            text(&case["full_transcript"]),
            text(&case["user_context"]),
        )
        .unwrap();
        assert_eq!(out, text(&case["out"]), "{}", case["template"]);
    }
}

#[test]
fn quick_action_prompts_match() {
    for case in fixture("quick_actions").as_array().unwrap() {
        let out = build_quick_action_prompt(
            text(&case["prompt_template"]),
            text(&case["digest"]),
            text(&case["recent"]),
            text(&case["user_context"]),
        )
        .unwrap();
        assert_eq!(out, text(&case["out"]));
    }
}

#[test]
fn validation_errors_match() {
    for case in fixture("validation").as_array().unwrap() {
        let result = validate_template(&template(&case["template"]));
        assert_eq!(
            result.errors,
            strings(&case["errors"]),
            "{}",
            case["template"]
        );
        assert_eq!(result.to_string(), text(&case["text"]));
    }
}

#[test]
fn template_diffs_match() {
    for case in fixture("diffs").as_array().unwrap() {
        let out = diff_templates(&template(&case["old"]), &template(&case["new"]));
        assert_eq!(out, strings(&case["out"]));
    }
}

#[test]
fn labels_match() {
    for case in fixture("labels").as_array().unwrap() {
        assert_eq!(
            sanitize_label(text(&case["raw"])),
            text(&case["out"]),
            "{}",
            case["raw"]
        );
    }
}

#[test]
fn label_and_compact_messages_match() {
    let misc = fixture("misc");
    for case in misc["label_prompt"].as_array().unwrap() {
        let out = build_label_prompt(
            text(&case["name"]),
            text(&case["description"]),
            text(&case["digest"]),
        );
        assert_eq!(out, text(&case["out"]));
    }
    let compact = &misc["compact"];
    assert_eq!(
        build_compact_user_message(text(&compact["markdown"])),
        text(&compact["out"])
    );
}

#[test]
fn builder_prompts_match() {
    let builder = fixture("builder");
    let example = template(&builder["example"]);
    let current = template(&builder["current"]);
    assert_eq!(
        TemplateBuilder::new(&example, None).prompt(),
        text(&builder["create_prompt"])
    );
    assert_eq!(
        TemplateBuilder::new(&example, Some(current)).prompt(),
        text(&builder["edit_prompt"])
    );
}

#[test]
fn loop_collapse_and_not_speech_match() {
    let cleaning = fixture("cleaning");
    for case in cleaning["collapse"].as_array().unwrap() {
        assert_eq!(
            collapse_loops(text(&case["text"]).trim()),
            text(&case["out"])
        );
    }
    for case in cleaning["not_speech"].as_array().unwrap() {
        assert_eq!(
            is_not_speech(text(&case["text"]).trim()),
            case["out"].as_bool().unwrap(),
            "{}",
            case["text"]
        );
    }
}

#[test]
fn speech_gate_matches() {
    for case in fixture("gate").as_array().unwrap() {
        let speech = lcg_audio(
            case["seed"].as_u64().unwrap(),
            case["seconds"].as_f64().unwrap(),
            case["amplitude"].as_f64().unwrap(),
            case["syllables"].as_bool().unwrap(),
        );
        let lead = (case["lead_s"].as_f64().unwrap() * 16000.0) as usize;
        let mut audio = vec![0.0f32; lead];
        audio.extend(speech);
        let flags = voiced_frames(&audio, 0.01);
        let expected: Vec<bool> = case["voiced"]
            .as_array()
            .unwrap()
            .iter()
            .map(|f| f.as_bool().unwrap())
            .collect();
        assert_eq!(flags, expected, "seed {}", case["seed"]);
        assert!(
            (voiced_seconds(&audio, 0.01) - case["voiced_seconds"].as_f64().unwrap()).abs() < 1e-9
        );
        for span in case["speech_start"].as_array().unwrap() {
            let (start, end, out) = (
                span[0].as_f64().unwrap(),
                span[1].as_f64().unwrap(),
                span[2].as_f64().unwrap(),
            );
            assert!(
                (speech_start(&flags, start, end) - out).abs() < 1e-9,
                "{start}..{end}"
            );
        }
    }
}

fn boost_track(parts: &serde_json::Value, seconds: f64) -> Vec<f32> {
    let mut track = vec![0.0f32; (seconds * 16000.0) as usize];
    for part in parts.as_array().unwrap() {
        let clip = lcg_audio(
            part[0].as_u64().unwrap(),
            part[4].as_f64().unwrap(),
            part[1].as_f64().unwrap(),
            part[2].as_bool().unwrap(),
        );
        let begin = (part[3].as_f64().unwrap() * 16000.0) as usize;
        for (slot, value) in track[begin..].iter_mut().zip(&clip) {
            *slot += value;
        }
    }
    track
}

#[test]
fn mic_boost_follows_python_block_by_block() {
    use lazy_take_notes::adapters::audio::mic_boost::MicBoost;
    for case in fixture("mic_boost").as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let seconds = case["seconds"].as_f64().unwrap();
        let mut mic = boost_track(&case["mic"], seconds);
        let system = boost_track(&case["system"], seconds);
        if let Some(bleed) = case["bleed"].as_array() {
            let (scale, shift) = (
                bleed[0].as_f64().unwrap() as f32,
                (bleed[1].as_f64().unwrap() * 16000.0) as usize,
            );
            for index in (shift..mic.len()).rev() {
                mic[index] += system[index - shift] * scale;
            }
        }
        let mut boost = MicBoost::new(0.05, 10.0, 16_000);
        for (block, expected) in case["blocks"].as_array().unwrap().iter().enumerate() {
            let range = block * 512..block * 512 + 512;
            let out = boost.apply(&mic[range.clone()], &system[range]);
            let (gain, bleeding, rms) = (
                expected[0].as_f64().unwrap(),
                expected[1].as_bool().unwrap(),
                expected[2].as_f64().unwrap(),
            );
            let out_rms = (out
                .iter()
                .map(|&s| f64::from(s) * f64::from(s))
                .sum::<f64>()
                / out.len() as f64)
                .sqrt();
            assert_eq!(boost.bleeding(), bleeding, "{name} block {block}: bleeding");
            assert!(
                (boost.gain() - gain).abs() <= 1e-4 * gain,
                "{name} block {block}: gain {} vs {gain}",
                boost.gain()
            );
            assert!(
                (out_rms - rms).abs() <= 1e-4 * rms.max(1e-6),
                "{name} block {block}: rms {out_rms} vs {rms}"
            );
        }
    }
}

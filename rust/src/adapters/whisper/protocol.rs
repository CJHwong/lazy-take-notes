//! Messages between the transcriber and its child, over the child's stdin and stdout.
//!
//! A request is one JSON line, then `samples` little-endian f32 values.
//! A reply is one JSON line.

use std::io::{self, BufRead, Write};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Request {
    pub language: String,
    pub hints: Vec<String>,
    pub samples: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Segment {
    pub text: String,
    /// Seconds into the chunk.
    pub start: f64,
    pub end: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "lowercase")]
pub enum Reply {
    Ready,
    Ok { segments: Vec<Segment> },
    Error { error: String },
}

pub fn write_request(
    out: &mut impl Write,
    language: &str,
    hints: &[String],
    audio: &[f32],
) -> io::Result<()> {
    let header = Request {
        language: language.into(),
        hints: hints.to_vec(),
        samples: audio.len(),
    };
    let mut bytes = serde_json::to_vec(&header).map_err(io::Error::other)?;
    bytes.push(b'\n');
    bytes.reserve(audio.len() * 4);
    for sample in audio {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    out.write_all(&bytes)?;
    out.flush()
}

/// The next request, or None at end of input: the parent closed the pipe or exited.
pub fn read_request(input: &mut impl BufRead) -> io::Result<Option<(Request, Vec<f32>)>> {
    let mut line = String::new();
    if input.read_line(&mut line)? == 0 {
        return Ok(None);
    }
    let request: Request = serde_json::from_str(&line).map_err(io::Error::other)?;
    let mut bytes = vec![0; request.samples * 4];
    input.read_exact(&mut bytes)?;
    let audio = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    Ok(Some((request, audio)))
}

pub fn write_reply(out: &mut impl Write, reply: &Reply) -> io::Result<()> {
    let mut bytes = serde_json::to_vec(reply).map_err(io::Error::other)?;
    bytes.push(b'\n');
    out.write_all(&bytes)?;
    out.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_request_round_trips_with_its_samples() {
        let mut pipe = Vec::new();
        write_request(&mut pipe, "zh", &["會議".into()], &[0.5, -1.0, 0.25]).unwrap();
        write_request(&mut pipe, "en", &[], &[]).unwrap();
        let mut input = io::Cursor::new(pipe);
        let (first, audio) = read_request(&mut input).unwrap().unwrap();
        assert_eq!(
            (
                first.language.as_str(),
                first.hints.as_slice(),
                audio.as_slice()
            ),
            ("zh", &["會議".to_string()][..], &[0.5, -1.0, 0.25][..])
        );
        assert_eq!(
            read_request(&mut input).unwrap().unwrap().1,
            Vec::<f32>::new()
        );
        assert!(read_request(&mut input).unwrap().is_none());
    }

    #[test]
    fn replies_carry_a_status_tag() {
        let ok = Reply::Ok {
            segments: vec![Segment {
                text: "hi".into(),
                start: 0.0,
                end: 1.5,
            }],
        };
        assert_eq!(
            serde_json::to_string(&ok).unwrap(),
            r#"{"status":"ok","segments":[{"text":"hi","start":0.0,"end":1.5}]}"#
        );
        assert_eq!(
            serde_json::to_string(&Reply::Ready).unwrap(),
            r#"{"status":"ready"}"#
        );
    }
}

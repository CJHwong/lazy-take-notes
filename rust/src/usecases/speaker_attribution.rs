//! Labels timestamped transcript lines with anonymous speakers.

use crate::domain::speaker_turn::SpeakerTurn;

/// Seconds between a line and a turn that still count as the same speech: the usual
/// mismatch between whisper segment edges and diarization turn edges.
pub const SPAN_TOLERANCE: f64 = 1.5;

/// The speaker of a line, by its whole span rather than its start.
///
/// A line that starts in a short gap belongs to the speaker it overlaps most. A
/// line with no overlap takes the nearest turn within `tolerance`, else none.
pub fn speaker_for_span(
    turns: &[SpeakerTurn],
    start: f64,
    end: f64,
    tolerance: f64,
) -> Option<u32> {
    let end = if end <= start { start + 0.01 } else { end };
    let mut best: Option<&SpeakerTurn> = None;
    let mut best_overlap = 0.0;
    for turn in turns {
        let overlap = end.min(turn.end) - start.max(turn.start);
        if overlap > best_overlap {
            best_overlap = overlap;
            best = Some(turn);
        }
    }
    if let Some(turn) = best {
        return Some(turn.speaker);
    }
    let mut nearest: Option<&SpeakerTurn> = None;
    let mut distance = f64::INFINITY;
    for turn in turns {
        let gap = (turn.start - end).max(start - turn.end).max(0.0);
        if gap < distance {
            distance = gap;
            nearest = Some(turn);
        }
    }
    nearest
        .filter(|_| distance <= tolerance)
        .map(|turn| turn.speaker)
}

/// The speaker at one moment. Of overlapping turns, the one that started last wins:
/// an overlap is usually a backchannel from the previous speaker.
pub fn speaker_at(turns: &[SpeakerTurn], at: f64) -> Option<u32> {
    turns
        .iter()
        .filter(|turn| turn.start <= at && at < turn.end)
        .fold(None::<&SpeakerTurn>, |latest, turn| match latest {
            Some(current) if current.start >= turn.start => Some(current),
            _ => Some(turn),
        })
        .map(|turn| turn.speaker)
}

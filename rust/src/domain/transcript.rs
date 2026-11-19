use serde::{Deserialize, Serialize};

/// One transcribed speech segment. Times are seconds from the session start.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranscriptSegment {
    pub text: String,
    pub wall_start: f64,
    pub wall_end: f64,
}

impl TranscriptSegment {
    pub fn new(text: impl Into<String>, wall_start: f64, wall_end: f64) -> Self {
        Self {
            text: text.into(),
            wall_start,
            wall_end,
        }
    }
}

/// Seconds as HH:MM:SS.
pub fn format_wall_time(seconds: f64) -> String {
    let total = seconds.max(0.0) as u64;
    format!(
        "{:02}:{:02}:{:02}",
        total / 3600,
        (total % 3600) / 60,
        total % 60
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_hours_minutes_seconds() {
        assert_eq!(format_wall_time(0.0), "00:00:00");
        assert_eq!(format_wall_time(3725.9), "01:02:05");
    }
}

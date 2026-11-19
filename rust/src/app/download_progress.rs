//! Model download progress as people read it: size, speed, and time left.

use crate::domain::model_asset::AssetProgress;

const MB: f64 = 1024.0 * 1024.0;

pub fn megabytes(size: u64) -> String {
    let size = size as f64;
    if size >= 1024.0 * MB {
        return format!("{:.1} GB", size / (1024.0 * MB));
    }
    format!("{:.0} MB", size / MB)
}

/// "143 MB of 834 MB (17%), 2.3 MB/s, about 5 min left". Parts without data are left out.
pub fn describe(progress: &AssetProgress) -> String {
    let done = megabytes(progress.downloaded);
    let mut text = if progress.total > 0 {
        format!(
            "{done} of {} ({}%)",
            megabytes(progress.total),
            progress.downloaded * 100 / progress.total
        )
    } else {
        done
    };
    if progress.bytes_per_second <= 0.0 {
        return text;
    }
    text += &format!(", {:.1} MB/s", progress.bytes_per_second / MB);
    if progress.total > 0 {
        let left = progress.total.saturating_sub(progress.downloaded) as f64;
        text += &format!(", {}", time_left(left / progress.bytes_per_second));
    }
    text
}

fn time_left(seconds: f64) -> String {
    // Python's round() rounds a tie to the even number.
    let minutes = (seconds / 60.0).round_ties_even() as u64;
    if minutes < 1 {
        return "less than 1 min left".into();
    }
    if minutes < 60 {
        return format!("about {minutes} min left");
    }
    format!("about {} h {} min left", minutes / 60, minutes % 60)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_like_the_python_text() {
        let progress = AssetProgress {
            downloaded: 143 * 1024 * 1024,
            total: 834 * 1024 * 1024,
            bytes_per_second: 2.3 * MB,
        };
        assert_eq!(
            describe(&progress),
            "143 MB of 834 MB (17%), 2.3 MB/s, about 5 min left"
        );
        let unknown = AssetProgress {
            downloaded: 3 * 1024 * 1024 * 1024 / 2,
            total: 0,
            bytes_per_second: 0.0,
        };
        assert_eq!(describe(&unknown), "1.5 GB");
        assert_eq!(time_left(20.0), "less than 1 min left");
        assert_eq!(time_left(3.0 * 3600.0 + 120.0), "about 3 h 2 min left");
    }
}

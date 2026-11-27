//! File identity and timestamps in the form the store and the WAV writer need.

use std::fs;

/// A file's identity on its volume: device and inode on Unix, volume and file
/// index on Windows.
pub(crate) fn file_key(meta: &fs::Metadata) -> (u64, u64) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        (meta.dev(), meta.ino())
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        (
            u64::from(meta.volume_serial_number().unwrap_or(0)),
            meta.file_index().unwrap_or(0),
        )
    }
}

/// How many names the file has, as far as the platform reports.
pub(crate) fn link_count(meta: &fs::Metadata) -> u64 {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        meta.nlink()
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        u64::from(meta.number_of_links().unwrap_or(1))
    }
}

/// The file's modification time in nanoseconds since the Unix epoch.
pub(crate) fn mtime_ns(meta: &fs::Metadata) -> i64 {
    meta.modified()
        .ok()
        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
        .and_then(|since| i64::try_from(since.as_nanos()).ok())
        .unwrap_or(0)
}

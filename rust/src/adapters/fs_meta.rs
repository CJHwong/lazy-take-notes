//! File identity and timestamps in the form the store and the WAV writer need.

use std::fs::{self, File};
use std::io;
use std::path::Path;

/// A file's identity on its volume, from its path.
pub(crate) fn file_key(path: &Path) -> io::Result<(u64, u64)> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let meta = fs::metadata(path)?;
        Ok((meta.dev(), meta.ino()))
    }
    #[cfg(windows)]
    {
        let file = File::open(path)?;
        let (volume, index, _) = details(&file)?;
        Ok((volume, index))
    }
}

/// A file's identity on its volume, from an open file.
pub(crate) fn file_key_of(file: &File) -> io::Result<(u64, u64)> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let meta = file.metadata()?;
        Ok((meta.dev(), meta.ino()))
    }
    #[cfg(windows)]
    {
        let (volume, index, _) = details(file)?;
        Ok((volume, index))
    }
}

/// How many names the file has, as far as the platform reports.
pub(crate) fn link_count(path: &Path) -> io::Result<u64> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        Ok(fs::metadata(path)?.nlink())
    }
    #[cfg(windows)]
    {
        let file = File::open(path)?;
        Ok(details(&file)?.2)
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

#[cfg(windows)]
fn details(file: &File) -> io::Result<(u64, u64, u64)> {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Storage::FileSystem::{
        BY_HANDLE_FILE_INFORMATION, GetFileInformationByHandle,
    };
    let mut info: BY_HANDLE_FILE_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: the handle is open and `info` is writable.
    if unsafe { GetFileInformationByHandle(file.as_raw_handle() as _, &mut info) } == 0 {
        return Err(io::Error::last_os_error());
    }
    let index = (u64::from(info.nFileIndexHigh) << 32) | u64::from(info.nFileIndexLow);
    Ok((
        u64::from(info.dwVolumeSerialNumber),
        index,
        u64::from(info.nNumberOfLinks),
    ))
}

"""Gateway: download audio or subtitles from a YouTube URL."""

from __future__ import annotations

from pathlib import Path


def is_url(value: str) -> bool:
    """Return True if value is an http/https URL."""
    return value.startswith('http://') or value.startswith('https://')


def fetch_youtube_subtitles(url: str, dest_dir: Path) -> tuple[Path, str] | None:
    """Try to download YouTube subtitles (VTT), return (vtt_path, title) or None.

    Prefer manual subtitles, fall back to automatic ones. Any failure returns None.
    """
    import yt_dlp  # noqa: PLC0415 -- deferred: only loaded for YouTube operations

    ydl_opts = {
        'skip_download': True,
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitlesformat': 'vtt',
        'outtmpl': str(dest_dir / 'sub.%(ext)s'),
        'quiet': True,
        'no_warnings': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except Exception:  # noqa: BLE001 -- any yt_dlp failure → silent fallback to audio download
        return None

    if not info:
        return None

    title = info.get('title', '')
    vtt_files = list(dest_dir.glob('*.vtt'))
    if not vtt_files:
        return None

    return vtt_files[0], title


def download_youtube_audio(url: str, dest_dir: Path) -> tuple[Path, str]:
    """Download YouTube audio to dest_dir, return (audio_path, title).

    Raises RuntimeError on download failure.
    """
    import yt_dlp  # noqa: PLC0415 -- deferred: only loaded for YouTube downloads

    output_template = str(dest_dir / 'audio.%(ext)s')
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_template,
        'quiet': True,
        'no_warnings': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except Exception as exc:  # noqa: BLE001 -- yt_dlp raises various internal exceptions
        raise RuntimeError(f'yt-dlp download failed: {exc}') from exc

    if not info:
        raise RuntimeError(f'yt-dlp returned no video info: {url}')

    title = info.get('title', '')
    # Resolve the actual downloaded file path.
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        filename = Path(ydl.prepare_filename(info))

    if not filename.exists():
        raise RuntimeError(f'yt-dlp completed but file not found: {filename}')

    return filename, title

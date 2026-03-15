"""Gateway: download audio or subtitles from a YouTube URL."""

from __future__ import annotations

from pathlib import Path


def is_url(value: str) -> bool:
    """Return True if value is an http/https URL."""
    return value.startswith('http://') or value.startswith('https://')


def _pick_best_vtt(vtt_files: list[Path], manual_langs: set[str]) -> Path | None:
    """Return the best VTT file: manual subtitle preferred, auto-generated as fallback.

    Selection within each tier is alphabetical for determinism.
    Language code is the last dot-segment of the stem (e.g. 'sub.en.vtt' → 'en').
    """
    manual: list[Path] = []
    auto: list[Path] = []
    for vtt in vtt_files:
        lang = vtt.stem.rsplit('.', 1)[-1]
        if lang in manual_langs:
            manual.append(vtt)
        else:
            auto.append(vtt)
    if manual:
        return sorted(manual)[0]
    if auto:
        return sorted(auto)[0]
    return None


def fetch_youtube_subtitles(url: str, dest_dir: Path) -> tuple[Path, str] | None:
    """Try to download YouTube subtitles (VTT), return (vtt_path, title) or None.

    Prefer manual subtitles, fall back to automatic ones. Any failure returns None.
    """
    import yt_dlp  # noqa: PLC0415 -- deferred: only loaded for YouTube operations

    # Ensure destination directory exists so yt-dlp can write subtitle files.
    dest_dir.mkdir(parents=True, exist_ok=True)

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
    manual_langs = set(info.get('subtitles', {}).keys())
    vtt_files = list(dest_dir.glob('*.vtt'))
    best = _pick_best_vtt(vtt_files, manual_langs)
    if best is None:
        return None
    return best, title


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

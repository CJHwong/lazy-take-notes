"""Gateway: 從 YouTube URL 下載音訊。"""

from __future__ import annotations

from pathlib import Path


def is_url(value: str) -> bool:
    """判斷 value 是否為 http/https URL。"""
    return value.startswith('http://') or value.startswith('https://')


def download_youtube_audio(url: str, dest_dir: Path) -> tuple[Path, str]:
    """下載 YouTube 音訊到 dest_dir，回傳 (音訊路徑, 影片標題)。

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
        raise RuntimeError(f'yt-dlp 下載失敗：{exc}') from exc

    if not info:
        raise RuntimeError(f'yt-dlp 未回傳影片資訊：{url}')

    title = info.get('title', '')
    # 取得實際下載的檔案路徑
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        filename = Path(ydl.prepare_filename(info))

    if not filename.exists():
        raise RuntimeError(f'yt-dlp 完成但找不到檔案：{filename}')

    return filename, title

"""Model download progress as people read it: size, speed, and time left."""

from __future__ import annotations

from lazy_take_notes.l1_entities.model_asset import AssetProgress

_MB = 1024 * 1024


def megabytes(size: int) -> str:
    if size >= 1024 * _MB:
        return f'{size / (1024 * _MB):.1f} GB'
    return f'{size / _MB:.0f} MB'


def describe(progress: AssetProgress) -> str:
    """'143 MB of 834 MB (17%), 2.3 MB/s, about 5 min left'. Parts without data are left out."""
    done = megabytes(progress.downloaded)
    text = (
        f'{done} of {megabytes(progress.total)} ({progress.downloaded * 100 // progress.total}%)'
        if progress.total > 0
        else done
    )
    if progress.bytes_per_second <= 0:
        return text
    text += f', {progress.bytes_per_second / _MB:.1f} MB/s'
    if progress.total > 0:
        text += f', {_time_left((progress.total - progress.downloaded) / progress.bytes_per_second)}'
    return text


def _time_left(seconds: float) -> str:
    minutes = round(seconds / 60)
    if minutes < 1:
        return 'less than 1 min left'
    if minutes < 60:
        return f'about {minutes} min left'
    return f'about {minutes // 60} h {minutes % 60} min left'

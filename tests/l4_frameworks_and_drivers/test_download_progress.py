"""Download progress as people read it."""

from __future__ import annotations

import pytest

from lazy_take_notes.l1_entities.model_asset import AssetProgress
from lazy_take_notes.l4_frameworks_and_drivers.download_progress import describe, megabytes

MB = 1024 * 1024


@pytest.mark.parametrize(
    ('size', 'text'),
    [(0, '0 MB'), (13 * MB, '13 MB'), (1536 * MB, '1.5 GB')],
)
def test_megabytes(size, text):
    assert megabytes(size) == text


@pytest.mark.parametrize(
    ('progress', 'text'),
    [
        (AssetProgress(143 * MB, 834 * MB, 2.3 * MB), '143 MB of 834 MB (17%), 2.3 MB/s, about 5 min left'),
        (AssetProgress(830 * MB, 834 * MB, 2 * MB), '830 MB of 834 MB (99%), 2.0 MB/s, less than 1 min left'),
        (AssetProgress(10 * MB, 1536 * MB, 0.3 * MB), '10 MB of 1.5 GB (0%), 0.3 MB/s, about 1 h 25 min left'),
        (AssetProgress(10 * MB, 834 * MB, 0), '10 MB of 834 MB (1%)'),
        (AssetProgress(10 * MB, 0, 1 * MB), '10 MB, 1.0 MB/s'),
    ],
    ids=['minutes', 'almost-done', 'hours', 'no-rate-yet', 'no-total'],
)
def test_describe(progress, text):
    assert describe(progress) == text

"""Tests for youtube_audio_downloader gateway."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from lazy_take_notes.l3_interface_adapters.gateways.youtube_audio_downloader import (
    download_youtube_audio,
    fetch_youtube_subtitles,
    is_url,
)


def _make_ydl_mock(info=None, side_effect=None):
    mock = MagicMock()
    mock.__enter__ = MagicMock(return_value=mock)
    mock.__exit__ = MagicMock(return_value=False)
    if side_effect:
        mock.extract_info.side_effect = side_effect
    else:
        mock.extract_info.return_value = info
    return mock


def test_is_url_http_returns_true():
    assert is_url('http://example.com/video') is True


def test_is_url_https_returns_true():
    assert is_url('https://www.youtube.com/watch?v=abc123') is True


def test_is_url_local_path_returns_false():
    assert is_url('/home/user/audio.wav') is False


def test_is_url_relative_path_returns_false():
    assert is_url('audio.wav') is False


def test_download_returns_path_and_title(tmp_path):
    fake_info = {'title': 'My Video', 'ext': 'webm'}
    expected_file = tmp_path / 'audio.webm'
    expected_file.touch()

    mock_ydl = _make_ydl_mock(info=fake_info)
    mock_ydl.prepare_filename.return_value = str(expected_file)

    with patch('yt_dlp.YoutubeDL', return_value=mock_ydl):
        audio_path, title = download_youtube_audio('https://youtube.com/watch?v=x', tmp_path)

    assert audio_path == expected_file
    assert title == 'My Video'


def test_download_raises_on_yt_dlp_exception(tmp_path):
    mock_ydl = _make_ydl_mock(side_effect=Exception('network error'))

    with patch('yt_dlp.YoutubeDL', return_value=mock_ydl):
        with pytest.raises(RuntimeError, match='yt-dlp download failed'):
            download_youtube_audio('https://youtube.com/watch?v=x', tmp_path)


def test_download_raises_when_info_is_none(tmp_path):
    mock_ydl = _make_ydl_mock(info=None)

    with patch('yt_dlp.YoutubeDL', return_value=mock_ydl):
        with pytest.raises(RuntimeError, match='yt-dlp returned no video info'):
            download_youtube_audio('https://youtube.com/watch?v=x', tmp_path)


def test_download_raises_when_file_missing_after_download(tmp_path):
    fake_info = {'title': 'My Video', 'ext': 'webm'}
    missing_file = tmp_path / 'audio.webm'
    # Intentionally do not create the file to simulate a missing download output.

    mock_ydl = _make_ydl_mock(info=fake_info)
    mock_ydl.prepare_filename.return_value = str(missing_file)

    with patch('yt_dlp.YoutubeDL', return_value=mock_ydl):
        with pytest.raises(RuntimeError, match='yt-dlp completed but file not found'):
            download_youtube_audio('https://youtube.com/watch?v=x', tmp_path)


# --- fetch_youtube_subtitles ---


def test_fetch_subtitles_returns_vtt_path_and_title(tmp_path):
    vtt_file = tmp_path / 'sub.zh-TW.vtt'
    vtt_file.touch()
    fake_info = {'title': 'My Video'}
    mock_ydl = _make_ydl_mock(info=fake_info)

    with patch('yt_dlp.YoutubeDL', return_value=mock_ydl):
        result = fetch_youtube_subtitles('https://youtube.com/watch?v=x', tmp_path)

    assert result is not None
    vtt_path, title = result
    assert vtt_path == vtt_file
    assert title == 'My Video'


def test_fetch_subtitles_returns_none_when_no_vtt(tmp_path):
    fake_info = {'title': 'No Subs Video'}
    mock_ydl = _make_ydl_mock(info=fake_info)

    with patch('yt_dlp.YoutubeDL', return_value=mock_ydl):
        result = fetch_youtube_subtitles('https://youtube.com/watch?v=x', tmp_path)

    assert result is None


def test_fetch_subtitles_returns_none_on_yt_dlp_exception(tmp_path):
    mock_ydl = _make_ydl_mock(side_effect=Exception('private video'))

    with patch('yt_dlp.YoutubeDL', return_value=mock_ydl):
        result = fetch_youtube_subtitles('https://youtube.com/watch?v=x', tmp_path)

    assert result is None


def test_fetch_subtitles_returns_none_when_info_is_none(tmp_path):
    mock_ydl = _make_ydl_mock(info=None)

    with patch('yt_dlp.YoutubeDL', return_value=mock_ydl):
        result = fetch_youtube_subtitles('https://youtube.com/watch?v=x', tmp_path)

    assert result is None


# --- _pick_best_vtt unit tests ---


def test_pick_best_vtt_prefers_manual_over_auto(tmp_path):
    from lazy_take_notes.l3_interface_adapters.gateways.youtube_audio_downloader import _pick_best_vtt

    auto_file = tmp_path / 'sub.en.vtt'
    manual_file = tmp_path / 'sub.zh-TW.vtt'
    auto_file.touch()
    manual_file.touch()
    assert _pick_best_vtt([auto_file, manual_file], manual_langs={'zh-TW'}) == manual_file


def test_pick_best_vtt_auto_fallback_is_sorted(tmp_path):
    from lazy_take_notes.l3_interface_adapters.gateways.youtube_audio_downloader import _pick_best_vtt

    file_en = tmp_path / 'sub.en.vtt'
    file_ja = tmp_path / 'sub.ja.vtt'
    file_en.touch()
    file_ja.touch()
    assert _pick_best_vtt([file_ja, file_en], manual_langs=set()) == file_en


def test_pick_best_vtt_returns_none_when_empty(tmp_path):
    from lazy_take_notes.l3_interface_adapters.gateways.youtube_audio_downloader import _pick_best_vtt

    assert _pick_best_vtt([], manual_langs={'en'}) is None


# --- fetch_youtube_subtitles integration: subtitle selection ---


def test_fetch_subtitles_manual_wins_over_auto(tmp_path):
    auto_file = tmp_path / 'sub.en.vtt'
    manual_file = tmp_path / 'sub.zh-TW.vtt'
    auto_file.touch()
    manual_file.touch()
    fake_info = {'title': 'Bilingual Video', 'subtitles': {'zh-TW': []}}
    mock_ydl = _make_ydl_mock(info=fake_info)

    with patch('yt_dlp.YoutubeDL', return_value=mock_ydl):
        result = fetch_youtube_subtitles('https://youtube.com/watch?v=x', tmp_path)

    assert result == (manual_file, 'Bilingual Video')


def test_fetch_subtitles_auto_fallback_is_deterministic(tmp_path):
    file_en = tmp_path / 'sub.en.vtt'
    file_ja = tmp_path / 'sub.ja.vtt'
    file_en.touch()
    file_ja.touch()
    fake_info = {'title': 'Auto Only Video'}
    mock_ydl = _make_ydl_mock(info=fake_info)

    with patch('yt_dlp.YoutubeDL', return_value=mock_ydl):
        result = fetch_youtube_subtitles('https://youtube.com/watch?v=x', tmp_path)

    assert result == (file_en, 'Auto Only Video')

"""Tests for youtube_audio_downloader gateway."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from lazy_take_notes.l3_interface_adapters.gateways.youtube_audio_downloader import (
    download_youtube_audio,
    is_url,
)


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

    mock_ydl = MagicMock()
    mock_ydl.__enter__ = MagicMock(return_value=mock_ydl)
    mock_ydl.__exit__ = MagicMock(return_value=False)
    mock_ydl.extract_info.return_value = fake_info
    mock_ydl.prepare_filename.return_value = str(expected_file)

    with patch('yt_dlp.YoutubeDL', return_value=mock_ydl):
        audio_path, title = download_youtube_audio('https://youtube.com/watch?v=x', tmp_path)

    assert audio_path == expected_file
    assert title == 'My Video'


def test_download_raises_on_yt_dlp_exception(tmp_path):
    mock_ydl = MagicMock()
    mock_ydl.__enter__ = MagicMock(return_value=mock_ydl)
    mock_ydl.__exit__ = MagicMock(return_value=False)
    mock_ydl.extract_info.side_effect = Exception('network error')

    with patch('yt_dlp.YoutubeDL', return_value=mock_ydl):
        with pytest.raises(RuntimeError, match='yt-dlp 下載失敗'):
            download_youtube_audio('https://youtube.com/watch?v=x', tmp_path)


def test_download_raises_when_info_is_none(tmp_path):
    mock_ydl = MagicMock()
    mock_ydl.__enter__ = MagicMock(return_value=mock_ydl)
    mock_ydl.__exit__ = MagicMock(return_value=False)
    mock_ydl.extract_info.return_value = None

    with patch('yt_dlp.YoutubeDL', return_value=mock_ydl):
        with pytest.raises(RuntimeError, match='yt-dlp 未回傳影片資訊'):
            download_youtube_audio('https://youtube.com/watch?v=x', tmp_path)


def test_download_raises_when_file_missing_after_download(tmp_path):
    fake_info = {'title': 'My Video', 'ext': 'webm'}
    missing_file = tmp_path / 'audio.webm'
    # 故意不建立檔案，模擬下載後找不到的情況

    mock_ydl = MagicMock()
    mock_ydl.__enter__ = MagicMock(return_value=mock_ydl)
    mock_ydl.__exit__ = MagicMock(return_value=False)
    mock_ydl.extract_info.return_value = fake_info
    mock_ydl.prepare_filename.return_value = str(missing_file)

    with patch('yt_dlp.YoutubeDL', return_value=mock_ydl):
        with pytest.raises(RuntimeError, match='yt-dlp 完成但找不到檔案'):
            download_youtube_audio('https://youtube.com/watch?v=x', tmp_path)

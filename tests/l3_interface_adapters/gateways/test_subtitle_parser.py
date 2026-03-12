"""Tests for subtitle_parser gateway."""

from __future__ import annotations

from pathlib import Path

import pytest

from lazy_take_notes.l3_interface_adapters.gateways.subtitle_parser import parse_vtt_to_segments


def _write_vtt(tmp_path: Path, content: str) -> Path:
    vtt = tmp_path / 'sub.vtt'
    vtt.write_text(content, encoding='utf-8')
    return vtt


class TestParseVttToSegments:
    def test_empty_vtt_returns_empty_list(self, tmp_path):
        vtt = _write_vtt(tmp_path, 'WEBVTT\n\n')
        assert parse_vtt_to_segments(vtt) == []

    def test_basic_cue_parsed_correctly(self, tmp_path):
        vtt = _write_vtt(
            tmp_path,
            'WEBVTT\n\n00:00:01.000 --> 00:00:03.000\nHello world\n\n',
        )
        segments = parse_vtt_to_segments(vtt)
        assert len(segments) == 1
        assert segments[0].text == 'Hello world'
        assert segments[0].wall_start == pytest.approx(1.0)
        assert segments[0].wall_end == pytest.approx(3.0)

    def test_timestamp_with_hours_parsed(self, tmp_path):
        vtt = _write_vtt(
            tmp_path,
            'WEBVTT\n\n01:02:03.500 --> 01:02:05.000\nDeep into video\n\n',
        )
        segments = parse_vtt_to_segments(vtt)
        assert segments[0].wall_start == pytest.approx(3723.5)

    def test_cue_attributes_ignored(self, tmp_path):
        # yt-dlp auto captions include "align:start position:0%" after timestamp
        vtt = _write_vtt(
            tmp_path,
            'WEBVTT\n\n00:00:01.000 --> 00:00:03.000 align:start position:0%\nWith attributes\n\n',
        )
        segments = parse_vtt_to_segments(vtt)
        assert len(segments) == 1
        assert segments[0].text == 'With attributes'

    def test_inline_tags_stripped(self, tmp_path):
        vtt = _write_vtt(
            tmp_path,
            'WEBVTT\n\n00:00:01.000 --> 00:00:03.000\n<c>Hello</c> <00:00:01.500><c>world</c>\n\n',
        )
        segments = parse_vtt_to_segments(vtt)
        assert segments[0].text == 'Hello world'

    def test_autocaption_dedup_collapses_superset(self, tmp_path):
        # YouTube auto captions: each cue extends the previous
        vtt = _write_vtt(
            tmp_path,
            'WEBVTT\n\n'
            '00:00:01.000 --> 00:00:03.000\nHello\n\n'
            '00:00:02.000 --> 00:00:04.000\nHello world\n\n'
            '00:00:03.000 --> 00:00:05.000\nHello world today\n\n',
        )
        segments = parse_vtt_to_segments(vtt)
        assert len(segments) == 1
        assert segments[0].text == 'Hello world today'
        assert segments[0].wall_start == pytest.approx(1.0)
        assert segments[0].wall_end == pytest.approx(5.0)

    def test_distinct_cues_all_preserved(self, tmp_path):
        vtt = _write_vtt(
            tmp_path,
            'WEBVTT\n\n'
            '00:00:01.000 --> 00:00:03.000\nFirst sentence.\n\n'
            '00:00:03.000 --> 00:00:05.000\nSecond sentence.\n\n',
        )
        segments = parse_vtt_to_segments(vtt)
        assert len(segments) == 2
        assert segments[0].text == 'First sentence.'
        assert segments[1].text == 'Second sentence.'

    def test_identical_cues_collapsed(self, tmp_path):
        vtt = _write_vtt(
            tmp_path,
            'WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nSame text\n\n00:00:01.500 --> 00:00:03.000\nSame text\n\n',
        )
        segments = parse_vtt_to_segments(vtt)
        assert len(segments) == 1
        assert segments[0].wall_end == pytest.approx(3.0)

    def test_empty_text_cues_skipped(self, tmp_path):
        vtt = _write_vtt(
            tmp_path,
            'WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n<c></c>\n\n00:00:02.000 --> 00:00:03.000\nReal text\n\n',
        )
        segments = parse_vtt_to_segments(vtt)
        assert len(segments) == 1
        assert segments[0].text == 'Real text'

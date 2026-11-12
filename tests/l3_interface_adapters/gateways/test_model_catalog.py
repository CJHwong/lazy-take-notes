"""Tests for the pinned model catalog."""

from __future__ import annotations

import pytest

from lazy_take_notes.l3_interface_adapters.gateways.model_catalog import (
    NEMOTRON_ID,
    RUNTIME_ID,
    asset_for_model,
    catalog,
    expand_model_alias,
    parse_hf_uri,
    platform_key,
)


class TestPlatformKey:
    @pytest.mark.parametrize(
        ('system', 'machine', 'expected'),
        [
            ('darwin', 'arm64', 'darwin-arm64'),
            ('darwin', 'x86_64', 'darwin-x86_64'),
            ('linux', 'x86_64', 'linux-x86_64'),
            ('linux', 'aarch64', 'linux-arm64'),
            ('win32', 'AMD64', 'win32-amd64'),
        ],
    )
    def test_maps_platform(self, system, machine, expected):
        assert platform_key(system, machine) == expected


class TestCatalog:
    def test_current_platform_has_runtime(self):
        assets = catalog('darwin-arm64')
        assert RUNTIME_ID in assets
        assert assets[RUNTIME_ID].archive is True
        assert assets[RUNTIME_ID].entry == 'bin/nemo-speech'

    def test_unsupported_platform_has_no_runtime(self):
        assets = catalog('windows-x86_64')
        assert RUNTIME_ID not in assets

    def test_every_asset_has_a_digest_and_size(self):
        for asset in catalog('darwin-arm64').values():
            assert asset.size > 0
            assert len(asset.sha256) == 64  # noqa: PLR2004 -- SHA-256 hex length
            assert asset.url.startswith('https://')
            assert asset.entry

    def test_alias_resolves(self):
        asset = asset_for_model('large-v3-turbo-q8_0')
        assert asset is not None
        assert asset.id == 'whisper-large-v3-turbo-q8_0'

    def test_hf_uri_resolves(self):
        asset = asset_for_model('hf://ggerganov/whisper.cpp/ggml-large-v3-turbo-q8_0.bin')
        assert asset is not None
        assert asset.id == 'whisper-large-v3-turbo-q8_0'

    def test_diarization_model_resolves(self):
        asset = asset_for_model(NEMOTRON_ID)
        assert asset is not None
        assert asset.group == 'speakers'

    def test_unknown_name_returns_none(self):
        assert asset_for_model('my-custom-model') is None


class TestAliases:
    def test_breeze25_alias_expands(self):
        assert expand_model_alias('breeze-q8') == 'hf://alan314159/Breeze-ASR-25-whispercpp/ggml-model-q8_0.bin'

    def test_breeze26_alias_expands(self):
        assert expand_model_alias('breeze26-q5') == 'hf://phate334/Breeze-ASR-26-GGML/ggml-model-q5_0.bin'

    def test_whisper_alias_expands(self):
        assert expand_model_alias('large-v3-turbo-q8_0') == 'hf://ggerganov/whisper.cpp/ggml-large-v3-turbo-q8_0.bin'

    def test_unknown_passes_through(self):
        assert expand_model_alias('my-custom-model') == 'my-custom-model'


class TestParseHfUri:
    def test_valid(self):
        assert parse_hf_uri('hf://owner/repo/file.bin') == ('owner/repo', 'file.bin')

    def test_nested_file(self):
        assert parse_hf_uri('hf://owner/repo/sub/file.bin') == ('owner/repo', 'sub/file.bin')

    @pytest.mark.parametrize('uri', ['hf://', 'hf://owner', 'hf://owner/repo'])
    def test_malformed(self, uri):
        assert parse_hf_uri(uri) is None

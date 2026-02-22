"""Tests for shared path constants."""

from __future__ import annotations

from pathlib import Path

from lazy_take_notes.l3_interface_adapters.gateways.paths import (
    CONFIG_DIR,
    DEFAULT_CONFIG_PATHS,
    USER_TEMPLATES_DIR,
)


class TestPaths:
    def test_config_dir_is_path(self):
        assert isinstance(CONFIG_DIR, Path)

    def test_user_templates_dir_is_path(self):
        assert isinstance(USER_TEMPLATES_DIR, Path)

    def test_config_dir_name(self):
        assert CONFIG_DIR.name == 'lazy-take-notes'

    def test_user_templates_dir_under_config_dir(self):
        assert USER_TEMPLATES_DIR.parent == CONFIG_DIR
        assert USER_TEMPLATES_DIR.name == 'templates'

    def test_default_config_paths_has_two_entries(self):
        assert len(DEFAULT_CONFIG_PATHS) == 2

    def test_default_config_paths_are_under_config_dir(self):
        for p in DEFAULT_CONFIG_PATHS:
            assert p.parent == CONFIG_DIR

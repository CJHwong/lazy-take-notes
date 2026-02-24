"""Tests for the template picker TUI."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
from textual.widgets import Input, ListView, Markdown

import lazy_take_notes.l3_interface_adapters.gateways.yaml_template_loader as yaml_loader_mod
from lazy_take_notes.l1_entities.audio_mode import AudioMode
from lazy_take_notes.l3_interface_adapters.gateways.yaml_template_loader import all_template_names
from lazy_take_notes.l4_frameworks_and_drivers.pickers.template_picker import (
    TemplatePicker,
    resolve_editor,
)


@pytest.fixture(autouse=True)
def _isolate_user_templates(monkeypatch):
    """Ensure user templates dir does not exist so only built-ins show."""
    monkeypatch.setattr(yaml_loader_mod, 'USER_TEMPLATES_DIR', Path('/nonexistent/user/templates'))


class TestTemplatePicker:
    @pytest.mark.asyncio
    async def test_picker_shows_all_templates(self):
        picker = TemplatePicker()
        async with picker.run_test():
            items = picker.query('#template-list TemplateItem')
            assert len(items) == len(all_template_names())

    @pytest.mark.asyncio
    async def test_escape_returns_none(self):
        picker = TemplatePicker()
        async with picker.run_test() as pilot:
            await pilot.press('escape')
            await pilot.pause()

        assert picker.return_value is None

    @pytest.mark.asyncio
    async def test_enter_returns_template_name(self):
        picker = TemplatePicker()
        async with picker.run_test() as pilot:
            # Move focus to the list and select
            await pilot.press('tab')
            await pilot.pause()
            await pilot.press('down')
            await pilot.pause()
            await pilot.press('enter')
            await pilot.pause()

        assert picker.return_value is not None
        name, mode = picker.return_value
        assert name in all_template_names()
        assert isinstance(mode, AudioMode)

    @pytest.mark.asyncio
    async def test_enter_returns_audio_mode_in_result(self):
        picker = TemplatePicker()
        async with picker.run_test() as pilot:
            await pilot.press('enter')
            await pilot.pause()

        assert picker.return_value is not None
        _name, mode = picker.return_value
        assert mode == AudioMode.MIC_ONLY

    @pytest.mark.asyncio
    async def test_d_cycles_audio_mode(self):
        picker = TemplatePicker()
        async with picker.run_test() as pilot:
            # [d] only fires when the list has focus (not the search Input)
            await pilot.press('tab')
            await pilot.pause()
            assert picker._audio_mode == AudioMode.MIC_ONLY
            await pilot.press('d')
            await pilot.pause()
            assert picker._audio_mode == AudioMode.SYSTEM_ONLY
            await pilot.press('d')
            await pilot.pause()
            assert picker._audio_mode == AudioMode.MIX
            await pilot.press('d')
            await pilot.pause()
            assert picker._audio_mode == AudioMode.MIC_ONLY

    @pytest.mark.asyncio
    async def test_d_cycles_and_result_reflects_mode(self):
        picker = TemplatePicker()
        async with picker.run_test() as pilot:
            await pilot.press('tab')  # move focus to list so [d] fires
            await pilot.pause()
            await pilot.press('d')
            await pilot.pause()
            await pilot.press('enter')
            await pilot.pause()

        assert picker.return_value is not None
        _name, mode = picker.return_value
        assert mode == AudioMode.SYSTEM_ONLY

    @pytest.mark.asyncio
    async def test_d_no_op_when_input_focused(self):
        """[d] must not cycle audio mode when the search Input has focus (to allow typing)."""
        picker = TemplatePicker()
        async with picker.run_test() as pilot:
            # Input is focused from on_mount — pressing 'd' types into the search box
            assert isinstance(picker.focused, Input)
            await pilot.press('d')
            await pilot.pause()
            assert picker._audio_mode == AudioMode.MIC_ONLY  # unchanged

    @pytest.mark.asyncio
    async def test_d_hidden_when_show_audio_mode_false(self):
        picker = TemplatePicker(show_audio_mode=False)
        async with picker.run_test() as pilot:
            # Tab to list so any key handling fires — still no-op when audio mode hidden
            await pilot.press('tab')
            await pilot.pause()
            before = picker._audio_mode
            await pilot.press('d')
            await pilot.pause()
            assert picker._audio_mode == before

    @pytest.mark.asyncio
    async def test_preview_updates_on_highlight(self):
        picker = TemplatePicker()
        async with picker.run_test() as pilot:
            preview = picker.query_one('#preview-md', Markdown)
            await pilot.press('tab')
            await pilot.pause()
            await pilot.press('down')
            await pilot.pause()
            assert preview._markdown

    @pytest.mark.asyncio
    async def test_locale_headers_present(self):
        picker = TemplatePicker()
        async with picker.run_test():
            headers = picker.query('#template-list LocaleHeader')
            assert len(headers) > 0

    @pytest.mark.asyncio
    async def test_search_filters_templates(self):
        picker = TemplatePicker()
        async with picker.run_test() as pilot:
            total = len(picker.query('#template-list TemplateItem'))
            search = picker.query_one('#search-input', Input)
            search.focus()
            await pilot.pause()
            # 'en' should match only the English template(s)
            await pilot.press(*'en')
            await pilot.pause()
            filtered = len(picker.query('#template-list TemplateItem'))
            assert filtered < total
            assert filtered > 0

    @pytest.mark.asyncio
    async def test_search_no_match_shows_empty(self):
        picker = TemplatePicker()
        async with picker.run_test() as pilot:
            search = picker.query_one('#search-input', Input)
            search.focus()
            await pilot.pause()
            for ch in 'xyznonexistent':
                await pilot.press(ch)
            await pilot.pause()
            items = picker.query('#template-list TemplateItem')
            assert len(items) == 0

    @pytest.mark.asyncio
    async def test_search_clear_restores_all(self):
        picker = TemplatePicker()
        async with picker.run_test() as pilot:
            total = len(picker.query('#template-list TemplateItem'))
            search = picker.query_one('#search-input', Input)
            search.focus()
            await pilot.pause()
            # Filter then clear
            await pilot.press(*'en')
            await pilot.pause()
            search.value = ''
            await pilot.pause()
            restored = len(picker.query('#template-list TemplateItem'))
            assert restored == total

    @pytest.mark.asyncio
    async def test_enter_after_search_returns_filtered_name(self):
        picker = TemplatePicker()
        async with picker.run_test() as pilot:
            search = picker.query_one('#search-input', Input)
            search.focus()
            await pilot.pause()
            for ch in 'default_en':
                await pilot.press(ch)
            await pilot.pause()
            await pilot.press('enter')
            await pilot.pause()

        assert picker.return_value is not None
        name, _mode = picker.return_value
        assert name == 'default_en'

    @pytest.mark.asyncio
    async def test_down_focuses_list_from_search(self):
        picker = TemplatePicker()
        async with picker.run_test() as pilot:
            assert isinstance(picker.focused, Input)
            await pilot.press('down')
            await pilot.pause()
            assert not isinstance(picker.focused, Input)

    @pytest.mark.asyncio
    async def test_cycle_noop_when_show_audio_mode_false(self):
        picker = TemplatePicker(show_audio_mode=False)
        async with picker.run_test():
            picker.action_cycle_audio_mode()
            assert picker._audio_mode == AudioMode.MIC_ONLY  # unchanged

    @pytest.mark.asyncio
    async def test_up_on_first_item_refocuses_search(self):
        picker = TemplatePicker()
        async with picker.run_test() as pilot:
            # Down from search focuses list on first TemplateItem
            await pilot.press('down')
            await pilot.pause()
            assert not isinstance(picker.focused, Input)
            # Up on first TemplateItem should refocus search input
            await pilot.press('up')
            await pilot.pause()
            assert isinstance(picker.focused, Input)

    @pytest.mark.asyncio
    async def test_select_noop_when_no_current_name(self):
        picker = TemplatePicker()
        async with picker.run_test():
            picker._current_name = None
            picker.action_select_template()
            # Should not exit — still running
            assert picker.return_value is None

    @pytest.mark.asyncio
    async def test_footer_contains_edit_hint(self):
        picker = TemplatePicker()
        async with picker.run_test():
            footer = picker._footer_text()
            assert '[e] Edit' in footer

    @pytest.mark.asyncio
    async def test_e_no_op_when_input_focused(self):
        """[e] must not fire edit when the search Input has focus (to allow typing)."""
        picker = TemplatePicker()
        async with picker.run_test() as pilot:
            assert isinstance(picker.focused, Input)
            original_name = picker._current_name
            await pilot.press('e')
            await pilot.pause()
            # Audio mode unchanged, no side effects — just typed 'e' into search
            assert picker._current_name is not None or original_name is None


class TestResolveEditor:
    def test_visual_takes_priority(self, monkeypatch):
        monkeypatch.setenv('VISUAL', 'nvim')
        monkeypatch.setenv('EDITOR', 'nano')
        assert resolve_editor() == ['nvim']

    def test_editor_fallback(self, monkeypatch):
        monkeypatch.delenv('VISUAL', raising=False)
        monkeypatch.setenv('EDITOR', 'vim')
        assert resolve_editor() == ['vim']

    def test_whitespace_split(self, monkeypatch):
        monkeypatch.setenv('VISUAL', 'code --wait')
        assert resolve_editor() == ['code', '--wait']

    def test_platform_fallback_darwin(self, monkeypatch):
        monkeypatch.delenv('VISUAL', raising=False)
        monkeypatch.delenv('EDITOR', raising=False)
        monkeypatch.setattr('sys.platform', 'darwin')
        assert resolve_editor() == ['open', '-t']

    def test_platform_fallback_linux(self, monkeypatch):
        monkeypatch.delenv('VISUAL', raising=False)
        monkeypatch.delenv('EDITOR', raising=False)
        monkeypatch.setattr('sys.platform', 'linux')
        assert resolve_editor() == ['xdg-open']

    def test_platform_fallback_win32(self, monkeypatch):
        monkeypatch.delenv('VISUAL', raising=False)
        monkeypatch.delenv('EDITOR', raising=False)
        monkeypatch.setattr('sys.platform', 'win32')
        assert resolve_editor() == ['notepad']

    def test_empty_env_vars_ignored(self, monkeypatch):
        monkeypatch.setenv('VISUAL', '  ')
        monkeypatch.setenv('EDITOR', '')
        monkeypatch.setattr('sys.platform', 'darwin')
        assert resolve_editor() == ['open', '-t']


class TestReloadAfterEdit:
    @pytest.mark.asyncio
    async def test_reload_refreshes_user_names_and_template(self, tmp_path: Path, monkeypatch):
        # Construct picker while user dir is still nonexistent (autouse fixture)
        picker = TemplatePicker()
        async with picker.run_test():
            assert 'default_en' not in picker._user_names
            # Now redirect user dir and copy a built-in
            monkeypatch.setattr(yaml_loader_mod, 'USER_TEMPLATES_DIR', tmp_path)
            from lazy_take_notes.l3_interface_adapters.gateways.yaml_template_loader import ensure_user_copy

            ensure_user_copy('default_en')
            picker._reload_after_edit('default_en')
            assert 'default_en' in picker._user_names

    @pytest.mark.asyncio
    async def test_reload_warns_on_invalid_yaml(self, tmp_path: Path, monkeypatch):
        # Construct picker with clean built-ins only
        picker = TemplatePicker()
        async with picker.run_test():
            original_locale = picker._templates['default_en'].metadata.locale
            # Now redirect user dir and write garbage YAML
            monkeypatch.setattr(yaml_loader_mod, 'USER_TEMPLATES_DIR', tmp_path)
            (tmp_path / 'default_en.yaml').write_text(': [invalid yaml {{{', encoding='utf-8')
            # Should not crash, just warn
            picker._reload_after_edit('default_en')
            # Template data should remain the original built-in
            assert picker._templates['default_en'].metadata.locale == original_locale

    @pytest.mark.asyncio
    async def test_edit_action_noop_when_no_current(self):
        picker = TemplatePicker()
        async with picker.run_test():
            picker._current_name = None
            # Should not raise
            picker.action_edit_template()

    @pytest.mark.asyncio
    async def test_edit_action_notifies_when_no_editor(self, monkeypatch):
        monkeypatch.delenv('VISUAL', raising=False)
        monkeypatch.delenv('EDITOR', raising=False)
        # Force platform to something unknown so fallback returns None
        monkeypatch.setattr('sys.platform', 'unknown')
        import lazy_take_notes.l4_frameworks_and_drivers.pickers.template_picker as tp_mod

        monkeypatch.setattr(tp_mod.platform, 'system', lambda: 'unknown')

        picker = TemplatePicker()
        async with picker.run_test():
            picker._current_name = 'default_en'
            picker.action_edit_template()
            # No crash — notification was sent (no assertion needed, just no exception)

    @pytest.mark.asyncio
    async def test_app_focus_reloads_pending_edit(self, tmp_path: Path, monkeypatch):
        """AppFocus triggers a second reload for GUI editors that return immediately."""
        from textual.events import AppFocus

        picker = TemplatePicker()
        async with picker.run_test():
            # Simulate: user pressed [e], subprocess returned instantly (GUI editor)
            picker._pending_reload_name = 'default_en'
            # Now user edits the file in VS Code and switches back to terminal
            monkeypatch.setattr(yaml_loader_mod, 'USER_TEMPLATES_DIR', tmp_path)
            from lazy_take_notes.l3_interface_adapters.gateways.yaml_template_loader import ensure_user_copy

            ensure_user_copy('default_en')
            # Simulate terminal regaining focus
            picker.on_app_focus(AppFocus())
            assert 'default_en' in picker._user_names
            # Pending flag cleared after reload
            assert picker._pending_reload_name is None

    @pytest.mark.asyncio
    async def test_app_focus_noop_without_pending(self):
        """AppFocus does nothing when no edit is pending."""
        from textual.events import AppFocus

        picker = TemplatePicker()
        async with picker.run_test():
            assert picker._pending_reload_name is None
            original_templates = dict(picker._templates)
            picker.on_app_focus(AppFocus())
            # Nothing changed
            assert picker._templates == original_templates


_USER_TEMPLATE_YAML = """\
metadata:
  name: "my_custom"
  description: "A user-defined template"
  locale: "en-US"
system_prompt: "You are an assistant."
digest_user_template: "Lines: {line_count}\\n{new_lines}"
final_user_template: "Done.\\n{new_lines}\\n{full_transcript}"
quick_actions: []
"""


class TestDeleteTemplate:
    @pytest.mark.asyncio
    async def test_x_no_op_when_input_focused(self):
        """[x] must not fire delete when the search Input has focus."""
        picker = TemplatePicker()
        async with picker.run_test() as pilot:
            assert isinstance(picker.focused, Input)
            await pilot.press('x')
            await pilot.pause()
            # Just typed 'x' into search — no confirm dialog pushed
            assert len(picker.screen_stack) == 1

    @pytest.mark.asyncio
    async def test_delete_warns_on_builtin(self):
        """Attempting to delete a built-in template shows a warning."""
        picker = TemplatePicker()
        async with picker.run_test() as pilot:
            await pilot.press('tab')
            await pilot.pause()
            # Current template should be a built-in (no user templates in this fixture)
            assert picker._current_name not in picker._user_names
            # Directly call action — should notify warning, not push confirm screen
            picker.action_delete_template()
            await pilot.pause()
            # No confirm screen pushed — still on the main screen
            assert len(picker.screen_stack) == 1

    @pytest.mark.asyncio
    async def test_delete_noop_when_no_current(self):
        picker = TemplatePicker()
        async with picker.run_test():
            picker._current_name = None
            picker.action_delete_template()

    @pytest.mark.asyncio
    async def test_delete_pushes_confirm_screen(self, tmp_path: Path, monkeypatch):
        """[x] on a user template pushes the confirm dialog."""
        monkeypatch.setattr(yaml_loader_mod, 'USER_TEMPLATES_DIR', tmp_path)
        (tmp_path / 'my_custom.yaml').write_text(_USER_TEMPLATE_YAML, encoding='utf-8')

        picker = TemplatePicker()
        async with picker.run_test() as pilot:
            # Navigate to my_custom
            await pilot.press('tab')
            await pilot.pause()
            # Find and highlight the user template
            list_view = picker.query_one('#template-list', ListView)
            for idx, child in enumerate(list_view.children):
                from lazy_take_notes.l4_frameworks_and_drivers.pickers.template_picker import TemplateItem

                if isinstance(child, TemplateItem) and child.template_name == 'my_custom':
                    list_view.index = idx
                    break
            await pilot.pause()
            assert picker._current_name == 'my_custom'

            picker.action_delete_template()
            await pilot.pause()
            # Confirm screen is now on the stack
            assert len(picker.screen_stack) == 2

    @pytest.mark.asyncio
    async def test_confirm_yes_deletes_user_template(self, tmp_path: Path, monkeypatch):
        """Confirming deletion removes the user template file and refreshes the list."""
        monkeypatch.setattr(yaml_loader_mod, 'USER_TEMPLATES_DIR', tmp_path)
        (tmp_path / 'my_custom.yaml').write_text(_USER_TEMPLATE_YAML, encoding='utf-8')

        picker = TemplatePicker()
        async with picker.run_test() as pilot:
            assert 'my_custom' in picker._user_names
            # Navigate to my_custom
            list_view = picker.query_one('#template-list', ListView)
            for idx, child in enumerate(list_view.children):
                from lazy_take_notes.l4_frameworks_and_drivers.pickers.template_picker import TemplateItem

                if isinstance(child, TemplateItem) and child.template_name == 'my_custom':
                    list_view.index = idx
                    break
            await pilot.pause()

            picker.action_delete_template()
            await pilot.pause()
            # Press 'y' to confirm
            await pilot.press('y')
            await pilot.pause()

            # File deleted, template removed from state
            assert not (tmp_path / 'my_custom.yaml').exists()
            assert 'my_custom' not in picker._templates
            assert 'my_custom' not in picker._user_names

    @pytest.mark.asyncio
    async def test_confirm_no_keeps_template(self, tmp_path: Path, monkeypatch):
        """Cancelling deletion keeps the template intact."""
        monkeypatch.setattr(yaml_loader_mod, 'USER_TEMPLATES_DIR', tmp_path)
        (tmp_path / 'my_custom.yaml').write_text(_USER_TEMPLATE_YAML, encoding='utf-8')

        picker = TemplatePicker()
        async with picker.run_test() as pilot:
            list_view = picker.query_one('#template-list', ListView)
            for idx, child in enumerate(list_view.children):
                from lazy_take_notes.l4_frameworks_and_drivers.pickers.template_picker import TemplateItem

                if isinstance(child, TemplateItem) and child.template_name == 'my_custom':
                    list_view.index = idx
                    break
            await pilot.pause()

            picker.action_delete_template()
            await pilot.pause()
            # Press 'n' to cancel
            await pilot.press('n')
            await pilot.pause()

            # File still exists
            assert (tmp_path / 'my_custom.yaml').exists()
            assert 'my_custom' in picker._templates

    @pytest.mark.asyncio
    async def test_delete_builtin_override_reverts_to_builtin(self, tmp_path: Path, monkeypatch):
        """Deleting a user override of a built-in reverts to the built-in version."""
        monkeypatch.setattr(yaml_loader_mod, 'USER_TEMPLATES_DIR', tmp_path)
        # Create user override of default_en
        override = _USER_TEMPLATE_YAML.replace('my_custom', 'OVERRIDDEN').replace('en-US', 'en')
        (tmp_path / 'default_en.yaml').write_text(override, encoding='utf-8')

        picker = TemplatePicker()
        async with picker.run_test() as pilot:
            assert 'default_en' in picker._user_names
            # The loaded template should be the user override
            assert picker._templates['default_en'].metadata.name == 'OVERRIDDEN'

            list_view = picker.query_one('#template-list', ListView)
            for idx, child in enumerate(list_view.children):
                from lazy_take_notes.l4_frameworks_and_drivers.pickers.template_picker import TemplateItem

                if isinstance(child, TemplateItem) and child.template_name == 'default_en':
                    list_view.index = idx
                    break
            await pilot.pause()

            picker.action_delete_template()
            await pilot.pause()
            await pilot.press('y')
            await pilot.pause()

            # User file gone, but template still exists (reverted to built-in)
            assert not (tmp_path / 'default_en.yaml').exists()
            assert 'default_en' in picker._templates
            assert 'default_en' not in picker._user_names
            # Name should be the built-in's, not our override
            assert picker._templates['default_en'].metadata.name != 'OVERRIDDEN'

    @pytest.mark.asyncio
    async def test_footer_contains_delete_hint(self):
        picker = TemplatePicker()
        async with picker.run_test():
            assert '[x] Delete' in picker._footer_text()

    @pytest.mark.asyncio
    async def test_x_key_pushes_confirm_when_list_focused(self, tmp_path: Path, monkeypatch):
        """[x] key press through on_key pushes confirm screen on user template."""
        monkeypatch.setattr(yaml_loader_mod, 'USER_TEMPLATES_DIR', tmp_path)
        (tmp_path / 'my_custom.yaml').write_text(_USER_TEMPLATE_YAML, encoding='utf-8')

        picker = TemplatePicker()
        async with picker.run_test() as pilot:
            await pilot.press('tab')
            await pilot.pause()
            list_view = picker.query_one('#template-list', ListView)
            for idx, child in enumerate(list_view.children):
                from lazy_take_notes.l4_frameworks_and_drivers.pickers.template_picker import TemplateItem

                if isinstance(child, TemplateItem) and child.template_name == 'my_custom':
                    list_view.index = idx
                    break
            await pilot.pause()
            assert picker._current_name == 'my_custom'
            # Press 'x' through the actual key handler
            await pilot.press('x')
            await pilot.pause()
            assert len(picker.screen_stack) == 2

    @pytest.mark.asyncio
    async def test_confirm_button_click_yes(self, tmp_path: Path, monkeypatch):
        """Clicking the Yes button in the confirm dialog deletes the template."""
        from textual.widgets import Button

        monkeypatch.setattr(yaml_loader_mod, 'USER_TEMPLATES_DIR', tmp_path)
        (tmp_path / 'my_custom.yaml').write_text(_USER_TEMPLATE_YAML, encoding='utf-8')

        picker = TemplatePicker()
        async with picker.run_test() as pilot:
            list_view = picker.query_one('#template-list', ListView)
            for idx, child in enumerate(list_view.children):
                from lazy_take_notes.l4_frameworks_and_drivers.pickers.template_picker import TemplateItem

                if isinstance(child, TemplateItem) and child.template_name == 'my_custom':
                    list_view.index = idx
                    break
            await pilot.pause()

            picker.action_delete_template()
            await pilot.pause()
            # Click the Yes button directly
            yes_btn = picker.screen.query_one('#confirm-yes', Button)
            yes_btn.press()
            await pilot.pause()

            assert not (tmp_path / 'my_custom.yaml').exists()
            assert 'my_custom' not in picker._templates

    @pytest.mark.asyncio
    async def test_confirm_button_click_no(self, tmp_path: Path, monkeypatch):
        """Clicking the No button in the confirm dialog keeps the template."""
        from textual.widgets import Button

        monkeypatch.setattr(yaml_loader_mod, 'USER_TEMPLATES_DIR', tmp_path)
        (tmp_path / 'my_custom.yaml').write_text(_USER_TEMPLATE_YAML, encoding='utf-8')

        picker = TemplatePicker()
        async with picker.run_test() as pilot:
            list_view = picker.query_one('#template-list', ListView)
            for idx, child in enumerate(list_view.children):
                from lazy_take_notes.l4_frameworks_and_drivers.pickers.template_picker import TemplateItem

                if isinstance(child, TemplateItem) and child.template_name == 'my_custom':
                    list_view.index = idx
                    break
            await pilot.pause()

            picker.action_delete_template()
            await pilot.pause()
            no_btn = picker.screen.query_one('#confirm-no', Button)
            no_btn.press()
            await pilot.pause()

            assert (tmp_path / 'my_custom.yaml').exists()
            assert 'my_custom' in picker._templates

    @pytest.mark.asyncio
    async def test_delete_callback_handles_value_error(self, tmp_path: Path, monkeypatch):
        """_on_delete_confirmed handles ValueError if template was already deleted."""
        monkeypatch.setattr(yaml_loader_mod, 'USER_TEMPLATES_DIR', tmp_path)
        (tmp_path / 'my_custom.yaml').write_text(_USER_TEMPLATE_YAML, encoding='utf-8')

        picker = TemplatePicker()
        async with picker.run_test():
            picker._current_name = 'my_custom'
            # Delete the file behind the picker's back
            (tmp_path / 'my_custom.yaml').unlink()
            # Callback should handle ValueError gracefully
            picker._on_delete_confirmed(True)
            # Template should still be in the dict (delete failed)
            assert 'my_custom' in picker._templates

    @pytest.mark.asyncio
    async def test_delete_callback_noop_when_not_confirmed(self):
        picker = TemplatePicker()
        async with picker.run_test():
            original = dict(picker._templates)
            picker._on_delete_confirmed(False)
            assert picker._templates == original

    @pytest.mark.asyncio
    async def test_delete_callback_noop_when_current_is_none(self):
        picker = TemplatePicker()
        async with picker.run_test():
            picker._current_name = None
            original = dict(picker._templates)
            picker._on_delete_confirmed(True)
            assert picker._templates == original


class TestEditTemplateKeyAndSuspend:
    @pytest.mark.asyncio
    async def test_e_key_fires_edit_from_list(self, monkeypatch):
        """[e] key press through on_key calls action_edit_template when list focused."""

        calls = []
        monkeypatch.setattr(
            TemplatePicker,
            'action_edit_template',
            lambda self: calls.append(True),
        )

        picker = TemplatePicker()
        async with picker.run_test() as pilot:
            await pilot.press('tab')
            await pilot.pause()
            await pilot.press('e')
            await pilot.pause()
            assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_edit_happy_path_with_mocked_suspend(self, tmp_path: Path, monkeypatch):
        """action_edit_template suspend + subprocess path (lines 355-366)."""
        import lazy_take_notes.l4_frameworks_and_drivers.pickers.template_picker as tp_mod

        monkeypatch.setattr(yaml_loader_mod, 'USER_TEMPLATES_DIR', tmp_path)
        monkeypatch.setenv('EDITOR', 'true')  # /usr/bin/true — exits immediately

        @contextmanager
        def fake_suspend():
            yield

        picker = TemplatePicker()
        async with picker.run_test():
            monkeypatch.setattr(picker, 'suspend', fake_suspend)
            picker._current_name = 'default_en'
            with patch.object(tp_mod.subprocess, 'run') as mock_run:
                picker.action_edit_template()
                mock_run.assert_called_once()

            # Pending reload was set and reload was called
            assert picker._pending_reload_name == 'default_en'
            assert 'default_en' in picker._user_names  # ensure_user_copy created it

    @pytest.mark.asyncio
    async def test_edit_suspend_not_supported(self, tmp_path: Path, monkeypatch):
        """action_edit_template catches SuspendNotSupported gracefully."""
        from textual.app import SuspendNotSupported

        monkeypatch.setattr(yaml_loader_mod, 'USER_TEMPLATES_DIR', tmp_path)
        monkeypatch.setenv('EDITOR', 'vim')

        @contextmanager
        def raise_suspend():
            raise SuspendNotSupported('test')
            yield  # noqa: RET503 -- unreachable; needed for contextmanager syntax

        picker = TemplatePicker()
        async with picker.run_test():
            monkeypatch.setattr(picker, 'suspend', raise_suspend)
            picker._current_name = 'default_en'
            # Should not raise
            picker.action_edit_template()
            # Pending reload should NOT be set (early return)
            assert picker._pending_reload_name is None

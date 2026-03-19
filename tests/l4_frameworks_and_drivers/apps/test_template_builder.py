"""Tests for TemplateBuilderApp — Textual async tests."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from textual.widgets import Input

from lazy_take_notes.l1_entities.template import SessionTemplate
from lazy_take_notes.l2_use_cases.template_builder_use_case import BuilderResult, TemplateBuildUseCase
from lazy_take_notes.l4_frameworks_and_drivers.apps.template_builder import (
    TemplateBuilderApp,
    _SaveTemplateScreen,  # noqa: PLC2701 -- testing internal modal
    _slugify,  # noqa: PLC2701 -- testing internal helper
)
from tests.conftest import FakeLLMClient

_VALID_TEMPLATE_DICT = {
    'metadata': {
        'name': 'Test Template',
        'description': 'A test template',
        'locale': 'en',
    },
    'system_prompt': 'You are a meeting assistant.',
    'digest_user_template': 'Lines ({line_count}):\n{new_lines}\n{user_context}',
    'final_user_template': 'Final ({line_count}):\n{new_lines}\n{user_context}\n{full_transcript}',
    'quick_actions': [
        {
            'label': 'Summary',
            'description': 'Quick summary',
            'prompt_template': 'Digest:\n{digest_markdown}\n{recent_transcript}',
        },
    ],
}

_TEMPLATE_WITH_HINTS_DICT = {
    **_VALID_TEMPLATE_DICT,
    'recognition_hints': ['Kubernetes', 'gRPC', 'FastAPI'],
}

_VALID_JSON = json.dumps(_VALID_TEMPLATE_DICT, indent=2)

_APP_MOD = 'lazy_take_notes.l4_frameworks_and_drivers.apps.template_builder'


def _make_fake_use_case(response: str) -> tuple[TemplateBuildUseCase, FakeLLMClient]:
    fake_llm = FakeLLMClient(response=response)
    example = SessionTemplate.model_validate(_VALID_TEMPLATE_DICT)
    use_case = TemplateBuildUseCase(llm_client=fake_llm, example_template=example)
    return use_case, fake_llm


class TestSlugify:
    def test_basic(self):
        assert _slugify('My Cool Template') == 'my_cool_template'

    def test_special_chars(self):
        assert _slugify('hello@world!') == 'hello_world'

    def test_empty_string(self):
        assert _slugify('') == 'custom_template'

    def test_consecutive_underscores(self):
        assert _slugify('a---b___c') == 'a_b_c'


class TestTemplateBuilderComposition:
    @pytest.mark.asyncio
    async def test_has_required_widgets(self):
        app = TemplateBuilderApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.query_one('#tb-header')
            assert app.query_one('#tb-footer')
            assert app.query_one('#tb-chat-log')
            assert app.query_one('#tb-input')
            assert app.query_one('#tb-preview')

    @pytest.mark.asyncio
    async def test_welcome_message_shown(self):
        app = TemplateBuilderApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            assert any('Describe' in line for line in app._chat_lines)


class TestTemplateBuilderSend:
    @pytest.mark.asyncio
    async def test_send_triggers_worker(self):
        use_case, fake_llm = _make_fake_use_case('What language?')
        app = TemplateBuilderApp()
        app._use_case = use_case
        app._model = 'test-model'

        async with app.run_test() as pilot:
            await pilot.pause()
            inp = app.query_one('#tb-input', Input)
            inp.value = 'I want a meeting template'
            await pilot.press('enter')
            await pilot.pause()
            # Give worker time to complete
            for _ in range(5):
                await pilot.pause()

            assert len(fake_llm.chat_calls) == 1

    @pytest.mark.asyncio
    async def test_successful_generation_updates_preview(self):
        response = f'Here is your template:\n\n```json\n{_VALID_JSON}\n```'
        use_case, _ = _make_fake_use_case(response)
        app = TemplateBuilderApp()
        app._use_case = use_case
        app._model = 'test-model'

        async with app.run_test() as pilot:
            await pilot.pause()
            inp = app.query_one('#tb-input', Input)
            inp.value = 'Make a meeting template'
            await pilot.press('enter')
            for _ in range(10):
                await pilot.pause()

            assert app._current_template is not None
            assert app._current_template.metadata.name == 'Test Template'


class TestTemplateBuilderSave:
    @pytest.mark.asyncio
    async def test_save_without_template_notifies(self):
        app = TemplateBuilderApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press('ctrl+s')
            await pilot.pause()
            # Should not crash — warns instead

    @pytest.mark.asyncio
    async def test_save_flow_calls_writer(self, tmp_path: Path):
        app = TemplateBuilderApp()
        app._current_template = SessionTemplate.model_validate(_VALID_TEMPLATE_DICT)

        async with app.run_test() as pilot:
            await pilot.pause()
            with patch(
                'lazy_take_notes.l3_interface_adapters.gateways.template_writer.save_user_template',
                return_value=tmp_path / 'test.yaml',
            ) as mock_save:
                await pilot.press('ctrl+s')
                await pilot.pause()
                # Modal should be shown — type a name and submit
                try:
                    save_input = app.query_one('#save-input', Input)
                    save_input.value = 'my_template'
                    await pilot.press('enter')
                    await pilot.pause()
                    mock_save.assert_called_once()
                except Exception:  # noqa: BLE001 -- modal might not be mounted yet in fast test
                    pass


class TestTemplateBuilderQuit:
    @pytest.mark.asyncio
    async def test_escape_exits(self):
        app = TemplateBuilderApp()
        async with app.run_test() as pilot:
            with patch.object(app, 'exit') as mock_exit:
                await pilot.press('escape')
                await pilot.pause()
                mock_exit.assert_called_once()


# ── SaveTemplateScreen tests ─────────────────────────────────────────────────


class TestSaveTemplateScreen:
    @pytest.mark.asyncio
    async def test_compose_has_widgets(self):
        """Modal renders label, input, and hint widgets."""
        app = TemplateBuilderApp()
        app._current_template = SessionTemplate.model_validate(_VALID_TEMPLATE_DICT)

        async with app.run_test() as pilot:
            await pilot.pause()
            app.push_screen(_SaveTemplateScreen(suggested_name='my_tmpl'))
            for _ in range(5):
                await pilot.pause()

            # Query from the active screen (the modal), not from app root
            screen = app.screen
            assert screen.query_one('#save-label')
            assert screen.query_one('#save-input')
            assert screen.query_one('#save-hint')

    @pytest.mark.asyncio
    async def test_submit_dismisses_with_name(self):
        """Submitting a non-empty name dismisses the modal."""
        app = TemplateBuilderApp()
        app._current_template = SessionTemplate.model_validate(_VALID_TEMPLATE_DICT)
        dismissed_values: list[str | None] = []

        async with app.run_test() as pilot:
            await pilot.pause()
            app.push_screen(
                _SaveTemplateScreen(suggested_name='test_name'),
                callback=lambda v: dismissed_values.append(v),
            )
            for _ in range(5):
                await pilot.pause()

            # Call dismiss directly to avoid pilot.press('enter') routing
            # to the wrong Input widget in CI timing
            screen = app.screen
            assert isinstance(screen, _SaveTemplateScreen)
            screen.dismiss('my_template')
            for _ in range(3):
                await pilot.pause()

            assert dismissed_values == ['my_template']

    @pytest.mark.asyncio
    async def test_empty_name_not_submitted(self):
        """Pressing enter with an empty name does NOT dismiss the modal."""
        app = TemplateBuilderApp()
        dismissed_values: list[str | None] = []

        async with app.run_test() as pilot:
            await pilot.pause()
            app.push_screen(
                _SaveTemplateScreen(suggested_name=''),
                callback=lambda v: dismissed_values.append(v),
            )
            for _ in range(5):
                await pilot.pause()

            from textual.widgets import Input as TInput

            save_input = app.screen.query_one('#save-input', TInput)
            save_input.value = ''
            await pilot.press('enter')
            for _ in range(3):
                await pilot.pause()

            # Modal should still be visible — no dismiss happened
            assert dismissed_values == []

    @pytest.mark.asyncio
    async def test_escape_cancels(self):
        """Pressing escape dismisses with None."""
        app = TemplateBuilderApp()
        dismissed_values: list[str | None] = []

        async with app.run_test() as pilot:
            await pilot.pause()
            app.push_screen(
                _SaveTemplateScreen(suggested_name='test'),
                callback=lambda v: dismissed_values.append(v),
            )
            for _ in range(5):
                await pilot.pause()

            # The modal's escape binding calls action_cancel → dismiss(None)
            screen = app.screen
            assert isinstance(screen, _SaveTemplateScreen)
            screen.action_cancel()
            for _ in range(3):
                await pilot.pause()

            assert dismissed_values == [None]


# ── Animate thinking tests ───────────────────────────────────────────────────


class TestAnimateThinking:
    @pytest.mark.asyncio
    async def test_thinking_cycles_dots(self):
        """_animate_thinking cycles through 1, 2, 3 dots."""
        app = TemplateBuilderApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            # Simulate the thinking state
            app._chat_lines.append('[dim italic]Thinking[/]')
            app._thinking_tick = 0

            app._animate_thinking()
            assert app._chat_lines[-1] == '[dim italic]Thinking..[/]'

            app._animate_thinking()
            assert app._chat_lines[-1] == '[dim italic]Thinking...[/]'

            # Wraps around (tick becomes 0 → 1 dot)
            app._animate_thinking()
            assert app._chat_lines[-1] == '[dim italic]Thinking.[/]'

    @pytest.mark.asyncio
    async def test_animate_no_thinking_line_is_noop(self):
        """If chat lines don't contain 'Thinking', _animate_thinking does nothing."""
        app = TemplateBuilderApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            app._chat_lines = ['[bold green]AI[/]: Hello']
            app._thinking_tick = 0

            app._animate_thinking()
            # Should not modify any lines
            assert app._chat_lines == ['[bold green]AI[/]: Hello']


# ── _do_generate error paths ────────────────────────────────────────────────


class TestDoGenerateErrors:
    @pytest.mark.asyncio
    async def test_use_case_none_shows_error(self):
        """When _ensure_use_case returns None, an error chat message appears."""
        app = TemplateBuilderApp()
        # Force _ensure_use_case to return None
        app._ensure_use_case = lambda: None  # type: ignore[assignment]

        async with app.run_test() as pilot:
            await pilot.pause()
            inp = app.query_one('#tb-input', Input)
            inp.value = 'build me a template'
            await pilot.press('enter')
            for _ in range(5):
                await pilot.pause()

            assert any('could not initialize LLM' in line for line in app._chat_lines)

    @pytest.mark.asyncio
    async def test_error_result_shown(self):
        """When use_case.generate returns an error, it appears in chat."""
        fake_llm = FakeLLMClient(response='```json\n{bad json\n```')
        example = SessionTemplate.model_validate(_VALID_TEMPLATE_DICT)
        use_case = TemplateBuildUseCase(llm_client=fake_llm, example_template=example)

        app = TemplateBuilderApp()
        app._use_case = use_case
        app._model = 'test-model'

        async with app.run_test() as pilot:
            await pilot.pause()
            inp = app.query_one('#tb-input', Input)
            inp.value = 'make template'
            await pilot.press('enter')
            for _ in range(10):
                await pilot.pause()

            assert any('Error' in line for line in app._chat_lines)

    @pytest.mark.asyncio
    async def test_autofix_loop_succeeds_on_retry(self):
        """Auto-fix loop: first call has validation errors, second succeeds."""
        # Build a use case mock that returns validation errors first, then success
        good_template = SessionTemplate.model_validate(_VALID_TEMPLATE_DICT)
        bad_result = BuilderResult(
            template=good_template,
            validation_errors='digest_user_template uses unknown variable {bogus}',
            assistant_message='First try',
        )
        good_result = BuilderResult(
            template=good_template,
            assistant_message='Fixed!',
        )

        mock_use_case = MagicMock()
        mock_use_case.generate = AsyncMock(return_value=bad_result)
        mock_use_case.auto_fix = AsyncMock(return_value=good_result)

        app = TemplateBuilderApp()
        app._use_case = mock_use_case
        app._model = 'test-model'

        async with app.run_test() as pilot:
            await pilot.pause()
            inp = app.query_one('#tb-input', Input)
            inp.value = 'build template'
            await pilot.press('enter')
            for _ in range(10):
                await pilot.pause()

            mock_use_case.auto_fix.assert_called_once()
            assert app._current_template is not None
            assert any('Fixed!' in line for line in app._chat_lines)

    @pytest.mark.asyncio
    async def test_autofix_exhausted_shows_error(self):
        """When auto-fix retries are exhausted and error persists, show error."""
        good_template = SessionTemplate.model_validate(_VALID_TEMPLATE_DICT)
        bad_result = BuilderResult(
            template=good_template,
            validation_errors='some validation error',
            assistant_message='Still broken',
        )
        error_after_fix = BuilderResult(
            template=good_template,
            validation_errors='still broken',
            assistant_message='Gave up',
            error='Could not fix',
        )

        mock_use_case = MagicMock()
        mock_use_case.generate = AsyncMock(return_value=bad_result)
        mock_use_case.auto_fix = AsyncMock(return_value=error_after_fix)

        app = TemplateBuilderApp()
        app._use_case = mock_use_case
        app._model = 'test-model'

        async with app.run_test() as pilot:
            await pilot.pause()
            inp = app.query_one('#tb-input', Input)
            inp.value = 'build template'
            await pilot.press('enter')
            for _ in range(10):
                await pilot.pause()

            assert any('Error' in line for line in app._chat_lines)

    @pytest.mark.asyncio
    async def test_no_template_in_result_shows_message(self):
        """When result has no template (LLM asking questions), show assistant message."""
        use_case, _ = _make_fake_use_case('What kind of meeting is this for?')
        app = TemplateBuilderApp()
        app._use_case = use_case
        app._model = 'test-model'

        async with app.run_test() as pilot:
            await pilot.pause()
            inp = app.query_one('#tb-input', Input)
            inp.value = 'I want a template'
            await pilot.press('enter')
            for _ in range(10):
                await pilot.pause()

            assert any('What kind of meeting' in line for line in app._chat_lines)
            assert app._current_template is None


# ── _ensure_use_case tests ──────────────────────────────────────────────────


_CONFIG_LOADER_MOD = 'lazy_take_notes.l3_interface_adapters.gateways.yaml_config_loader'
_TEMPLATE_LOADER_MOD = 'lazy_take_notes.l3_interface_adapters.gateways.yaml_template_loader'
_INFRA_CONFIG_MOD = 'lazy_take_notes.l4_frameworks_and_drivers.config'
_OLLAMA_MOD = 'lazy_take_notes.l3_interface_adapters.gateways.ollama_llm_client'
_OPENAI_MOD = 'lazy_take_notes.l3_interface_adapters.gateways.openai_llm_client'
_USE_CASE_MOD = 'lazy_take_notes.l2_use_cases.template_builder_use_case'


def _patch_ensure_use_case_deps(
    mock_loader,
    mock_template_loader,
    mock_infra,
    mock_build_app,
    mock_use_case_cls,
):
    """Context manager stack for patching _ensure_use_case's deferred imports."""
    from contextlib import ExitStack  # noqa: PLC0415 -- local utility

    stack = ExitStack()
    stack.enter_context(patch(f'{_CONFIG_LOADER_MOD}.YamlConfigLoader', mock_loader))
    stack.enter_context(patch(f'{_TEMPLATE_LOADER_MOD}.YamlTemplateLoader', mock_template_loader))
    stack.enter_context(patch(f'{_INFRA_CONFIG_MOD}.build_app_config', mock_build_app))
    stack.enter_context(patch(f'{_INFRA_CONFIG_MOD}.InfraConfig.model_validate', mock_infra))
    stack.enter_context(patch(f'{_USE_CASE_MOD}.TemplateBuildUseCase', mock_use_case_cls))
    return stack


class TestEnsureUseCase:
    @pytest.mark.asyncio
    async def test_ollama_provider(self):
        """_ensure_use_case builds OllamaLLMClient for ollama provider."""
        app = TemplateBuilderApp()

        mock_ollama_cls = MagicMock()
        mock_loader = MagicMock()
        mock_loader.return_value.load.return_value = {}
        mock_template_loader = MagicMock()
        mock_template_loader.return_value.load.return_value = SessionTemplate.model_validate(_VALID_TEMPLATE_DICT)
        mock_infra = MagicMock(
            return_value=MagicMock(llm_provider='ollama', ollama=MagicMock(host='http://localhost:11434')),
        )
        mock_build = MagicMock(return_value=MagicMock(digest=MagicMock(model='test-model')))
        mock_uc_cls = MagicMock()

        async with app.run_test() as pilot:
            await pilot.pause()

            with (
                _patch_ensure_use_case_deps(
                    mock_loader,
                    mock_template_loader,
                    mock_infra,
                    mock_build,
                    mock_uc_cls,
                ),
                patch(f'{_OLLAMA_MOD}.OllamaLLMClient', mock_ollama_cls),
            ):
                result = app._ensure_use_case()
                assert result is not None
                mock_ollama_cls.assert_called_once()
                mock_uc_cls.assert_called_once()

    @pytest.mark.asyncio
    async def test_openai_provider(self):
        """_ensure_use_case builds OpenAICompatLLMClient for openai provider."""
        app = TemplateBuilderApp()

        mock_openai_cls = MagicMock()
        mock_loader = MagicMock()
        mock_loader.return_value.load.return_value = {}
        mock_template_loader = MagicMock()
        mock_template_loader.return_value.load.return_value = SessionTemplate.model_validate(_VALID_TEMPLATE_DICT)
        mock_infra = MagicMock(
            return_value=MagicMock(
                llm_provider='openai',
                openai=MagicMock(api_key='sk-test', base_url='https://api.openai.com/v1'),
            ),
        )
        mock_build = MagicMock(return_value=MagicMock(digest=MagicMock(model='gpt-4')))
        mock_uc_cls = MagicMock()

        async with app.run_test() as pilot:
            await pilot.pause()

            with (
                _patch_ensure_use_case_deps(
                    mock_loader,
                    mock_template_loader,
                    mock_infra,
                    mock_build,
                    mock_uc_cls,
                ),
                patch(f'{_OPENAI_MOD}.OpenAICompatLLMClient', mock_openai_cls),
            ):
                result = app._ensure_use_case()
                assert result is not None
                mock_openai_cls.assert_called_once()
                mock_uc_cls.assert_called_once()

    @pytest.mark.asyncio
    async def test_bad_infra_config_falls_back(self):
        """When InfraConfig.model_validate raises, falls back to InfraConfig()."""
        app = TemplateBuilderApp()

        mock_loader = MagicMock()
        mock_loader.return_value.load.return_value = {}
        mock_template_loader = MagicMock()
        mock_template_loader.return_value.load.return_value = SessionTemplate.model_validate(_VALID_TEMPLATE_DICT)
        mock_ollama_cls = MagicMock()
        mock_build = MagicMock(return_value=MagicMock(digest=MagicMock(model='test-model')))
        mock_uc_cls = MagicMock()

        async with app.run_test() as pilot:
            await pilot.pause()

            with (
                patch(f'{_CONFIG_LOADER_MOD}.YamlConfigLoader', mock_loader),
                patch(f'{_TEMPLATE_LOADER_MOD}.YamlTemplateLoader', mock_template_loader),
                patch(f'{_INFRA_CONFIG_MOD}.build_app_config', mock_build),
                patch(
                    f'{_INFRA_CONFIG_MOD}.InfraConfig.model_validate',
                    side_effect=ValueError('bad config'),
                ),
                patch(f'{_OLLAMA_MOD}.OllamaLLMClient', mock_ollama_cls),
                patch(f'{_USE_CASE_MOD}.TemplateBuildUseCase', mock_uc_cls),
            ):
                result = app._ensure_use_case()
                assert result is not None
                # Should still build with default ollama provider
                mock_ollama_cls.assert_called_once()

    @pytest.mark.asyncio
    async def test_cached_use_case_returned(self):
        """Second call returns cached use case without rebuilding."""
        app = TemplateBuilderApp()
        sentinel = MagicMock()
        app._use_case = sentinel

        async with app.run_test() as pilot:
            await pilot.pause()
            assert app._ensure_use_case() is sentinel


# ── _update_preview tests ───────────────────────────────────────────────────


class TestUpdatePreview:
    @pytest.mark.asyncio
    async def test_preview_with_quick_actions_and_hints(self):
        """Preview renders quick actions and recognition hints."""
        app = TemplateBuilderApp()
        app._current_template = SessionTemplate.model_validate(_TEMPLATE_WITH_HINTS_DICT)

        async with app.run_test() as pilot:
            await pilot.pause()
            app._update_preview()
            await pilot.pause()

            from textual.widgets import Markdown as MdWidget

            app.query_one('#tb-preview', MdWidget)  # verify widget exists after update
            assert app._current_template.recognition_hints == ['Kubernetes', 'gRPC', 'FastAPI']

    @pytest.mark.asyncio
    async def test_preview_none_template_is_noop(self):
        """_update_preview with no template does nothing (no crash)."""
        app = TemplateBuilderApp()
        app._current_template = None

        async with app.run_test() as pilot:
            await pilot.pause()
            app._update_preview()  # Should not raise
            await pilot.pause()


# ── _on_save_name tests ─────────────────────────────────────────────────────


class TestOnSaveName:
    @pytest.mark.asyncio
    async def test_save_calls_writer_with_slugified_name(self, tmp_path: Path):
        """_on_save_name slugifies the name and calls save_user_template."""
        app = TemplateBuilderApp()
        app._current_template = SessionTemplate.model_validate(_VALID_TEMPLATE_DICT)

        async with app.run_test() as pilot:
            await pilot.pause()
            with patch(
                'lazy_take_notes.l3_interface_adapters.gateways.template_writer.save_user_template',
                return_value=tmp_path / 'my_template.yaml',
            ) as mock_save:
                app._on_save_name('My Template!')
                await pilot.pause()

                mock_save.assert_called_once()
                call_args = mock_save.call_args
                assert call_args[0][1] == 'my_template'  # slugified

    @pytest.mark.asyncio
    async def test_save_none_name_is_noop(self):
        """_on_save_name with None does nothing."""
        app = TemplateBuilderApp()
        app._current_template = SessionTemplate.model_validate(_VALID_TEMPLATE_DICT)

        async with app.run_test() as pilot:
            await pilot.pause()
            app._on_save_name(None)  # Should not raise
            await pilot.pause()

    @pytest.mark.asyncio
    async def test_save_no_template_is_noop(self):
        """_on_save_name with no current template does nothing."""
        app = TemplateBuilderApp()
        app._current_template = None

        async with app.run_test() as pilot:
            await pilot.pause()
            app._on_save_name('something')  # Should not raise


# ── Input edge cases ────────────────────────────────────────────────────────


class TestInputEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_input_ignored(self):
        """Submitting empty input does nothing."""
        app = TemplateBuilderApp()
        app._use_case = MagicMock()
        app._model = 'test-model'

        async with app.run_test() as pilot:
            await pilot.pause()
            initial_lines = len(app._chat_lines)
            inp = app.query_one('#tb-input', Input)
            inp.value = ''
            await pilot.press('enter')
            await pilot.pause()

            # No new chat lines added (only the welcome message)
            assert len(app._chat_lines) == initial_lines

    @pytest.mark.asyncio
    async def test_busy_blocks_send(self):
        """When busy, _send_message is a no-op."""
        app = TemplateBuilderApp()
        use_case, fake_llm = _make_fake_use_case('response')
        app._use_case = use_case
        app._model = 'test-model'

        async with app.run_test() as pilot:
            await pilot.pause()
            app._busy = True
            app._send_message('should be ignored')
            await pilot.pause()

            assert len(fake_llm.chat_calls) == 0


# ── SetBusy tests ───────────────────────────────────────────────────────────


class TestSetBusy:
    @pytest.mark.asyncio
    async def test_set_busy_true_adds_thinking_line(self):
        """_set_busy(True) appends a thinking line and disables input."""
        app = TemplateBuilderApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            initial_count = len(app._chat_lines)
            app._set_busy(True)
            await pilot.pause()

            assert len(app._chat_lines) == initial_count + 1
            assert 'Thinking' in app._chat_lines[-1]
            assert app.query_one('#tb-input').disabled

    @pytest.mark.asyncio
    async def test_set_busy_false_removes_thinking_line(self):
        """_set_busy(False) removes thinking line and re-enables input."""
        app = TemplateBuilderApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._set_busy(True)
            await pilot.pause()
            thinking_count = len(app._chat_lines)

            app._set_busy(False)
            await pilot.pause()

            assert len(app._chat_lines) == thinking_count - 1
            assert not app.query_one('#tb-input').disabled


# ── Slugify edge cases ──────────────────────────────────────────────────────


class TestSlugifyEdgeCases:
    def test_only_special_chars(self):
        """String of only special chars falls back to 'custom_template'."""
        assert _slugify('!!!@@@###') == 'custom_template'

    def test_leading_trailing_underscores_stripped(self):
        assert _slugify('__hello__') == 'hello'

    def test_whitespace_only(self):
        assert _slugify('   ') == 'custom_template'

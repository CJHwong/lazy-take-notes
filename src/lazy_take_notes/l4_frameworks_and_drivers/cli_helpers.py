"""Shared CLI helpers — used by cli.py, plugin_api.py, and plugins.

Extracted to avoid circular imports when plugins import from plugin_api
while cli.py loads plugins at module level.
"""

from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from lazy_take_notes.l1_entities.transcript import TranscriptSegment
    from lazy_take_notes.l2_use_cases.ports.audio_source import AudioSource
    from lazy_take_notes.l2_use_cases.ports.llm_client import LLMClient
    from lazy_take_notes.l2_use_cases.ports.transcriber import Transcriber


def _clear_normal_screen() -> None:  # pragma: no cover -- terminal escape; no-op in test
    """Clear the normal screen buffer before launching Textual apps.

    Each Textual App enters/exits the alternate screen buffer independently.
    Between apps the terminal briefly restores the normal screen. Clearing it
    beforehand means the flash shows a blank screen instead of shell history.
    """
    sys.stdout.write('\033[2J\033[H')
    sys.stdout.flush()


def resolve_base_dir(output_dir: str | None, config) -> Path:
    """Resolve the base output directory, expanding ~ in paths."""
    return Path(output_dir or config.output.directory).expanduser()


def make_session_dir(base_dir: Path, label: str | None) -> Path:
    """Create a timestamped session subdirectory under base_dir."""
    stamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    if label:
        safe_label = re.sub(r'[^\w\-]', '_', label)
        name = f'{stamp}_{safe_label}'
    else:
        name = stamp
    session_dir = base_dir / name
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def load_config(config_path, output_dir):
    """Load configuration and infra settings. Returns (config, infra, template_loader) or exits."""
    from lazy_take_notes.l3_interface_adapters.gateways.yaml_config_loader import (  # noqa: PLC0415 -- deferred: yaml stack not loaded on --help
        YamlConfigLoader,
    )
    from lazy_take_notes.l3_interface_adapters.gateways.yaml_template_loader import (  # noqa: PLC0415 -- deferred: yaml stack not loaded on --help
        YamlTemplateLoader,
    )
    from lazy_take_notes.l4_frameworks_and_drivers.config import (  # noqa: PLC0415 -- deferred: not needed for --help
        InfraConfig,
        build_app_config,
    )

    config_loader = YamlConfigLoader()
    template_loader = YamlTemplateLoader()

    try:
        overrides: dict = {}
        if output_dir:
            overrides['output'] = {'directory': output_dir}
        raw = config_loader.load(config_path, overrides=overrides if overrides else None)
        config = build_app_config(raw)
        infra = InfraConfig.model_validate(raw)
    except FileNotFoundError as e:
        click.echo(f'Error: {e}', err=True)
        sys.exit(1)

    return config, infra, template_loader


def pick_template(template_loader):
    """Show the interactive template picker. Returns a SessionTemplate, or None if cancelled.

    Handles the create-template loop internally — if the user presses [n] to
    create a new template, the builder launches and the picker re-appears.
    """
    from lazy_take_notes.l4_frameworks_and_drivers.apps.template_builder import (  # noqa: PLC0415 -- deferred: only loaded when user creates a template
        TemplateBuilderApp,
    )
    from lazy_take_notes.l4_frameworks_and_drivers.pickers.template_picker import (  # noqa: PLC0415 -- deferred: Textual not loaded on --help
        TemplatePicker,
    )

    # FIXME: replace with a single ShellApp that uses Textual Screens for
    # pickers and main apps, eliminating inter-app terminal flicker entirely.
    _clear_normal_screen()
    while True:
        picker_result = TemplatePicker().run()
        if picker_result is None:
            return None
        if picker_result == '__create_template__':
            TemplateBuilderApp().run()
            continue
        break

    try:
        return template_loader.load(picker_result)
    except FileNotFoundError as e:
        click.echo(f'Error: {e}', err=True)
        sys.exit(1)


def preflight_llm(infra, config) -> tuple[list[str], list[str]]:
    """Check LLM connectivity and model availability. Returns (missing_digest, missing_interactive)."""
    from lazy_take_notes.l4_frameworks_and_drivers.container import (  # noqa: PLC0415 -- deferred: avoid circular import at module level
        DependencyContainer,
    )

    try:
        client = DependencyContainer.resolve_llm_client(infra)
    except ValueError as e:
        click.echo(f'Error: {e}', err=True)
        sys.exit(1)

    ok, err = client.check_connectivity()
    if not ok:
        click.echo(f'Warning: LLM provider not reachable ({err}). Digests will fail.', err=True)
        click.echo('Transcript-only mode: audio capture will still work.', err=True)
        return [], []

    unique_models = list(dict.fromkeys([config.digest.model, config.interactive.model]))
    missing = set(client.check_models(unique_models))
    missing_digest = [config.digest.model] if config.digest.model in missing else []
    missing_interactive = [config.interactive.model] if config.interactive.model in missing else []
    return missing_digest, missing_interactive


def run_transcribe(
    ctx: click.Context,
    *,
    audio_path: Path | None = None,
    subtitle_segments: list[TranscriptSegment] | None = None,
    label: str | None = None,
    llm_client: LLMClient | None = None,
    transcriber: Transcriber | None = None,
    source_url: str | None = None,
    source_title: str | None = None,
) -> None:
    """Run a complete transcription session — the high-level plugin entry point.

    Handles the entire flow: config loading → template picker → session
    directory → LLM preflight → dependency wiring → TranscribeApp launch.

    Provide *audio_path* for whisper transcription, *subtitle_segments* for
    subtitle replay, or both (subtitle_segments takes priority, audio_path
    as fallback when segments are empty).

    Returns normally if the user cancels the template picker.
    """
    from lazy_take_notes.l4_frameworks_and_drivers.apps.transcribe import (  # noqa: PLC0415 -- deferred: Textual TUI not loaded for --help
        TranscribeApp,
    )
    from lazy_take_notes.l4_frameworks_and_drivers.container import (  # noqa: PLC0415 -- deferred: container not loaded for --help
        DependencyContainer,
    )

    config_path = ctx.obj['config_path']
    output_dir = ctx.obj['output_dir']
    config, infra, template_loader = load_config(config_path, output_dir)

    template = pick_template(template_loader)
    if template is None:
        return

    base_dir = resolve_base_dir(output_dir, config)
    out_dir = make_session_dir(base_dir, label)

    missing_digest, missing_interactive = preflight_llm(infra, config)

    needs_transcriber = audio_path is not None and subtitle_segments is None
    container = DependencyContainer(
        config,
        template,
        out_dir,
        infra=infra,
        build_audio=False,
        llm_client=llm_client,
        transcriber=transcriber,
        source_url=source_url,
        source_title=source_title,
    )

    app = TranscribeApp(
        config=config,
        template=template,
        output_dir=out_dir,
        controller=container.controller,
        subtitle_segments=subtitle_segments,
        audio_path=audio_path,
        transcriber=container.transcriber if needs_transcriber else None,
        missing_digest_models=missing_digest,
        missing_interactive_models=missing_interactive,
        label=label or '',
    )
    app.run()


def preflight_microphone() -> None:
    """Warn if no input audio devices are found."""
    try:
        import sounddevice as sd  # noqa: PLC0415 -- deferred: not loaded on --help

        devices = sd.query_devices()
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        if not input_devices:
            click.echo('Warning: No input audio devices found.', err=True)
    except Exception as e:
        click.echo(f'Warning: Cannot query audio devices ({e}).', err=True)


def run_record(
    ctx: click.Context,
    *,
    label: str | None = None,
    llm_client: LLMClient | None = None,
    transcriber: Transcriber | None = None,
    audio_source: AudioSource | None = None,
) -> None:
    """Run a live recording session -- the high-level plugin entry point.

    Handles the entire flow: config loading -> template picker -> session
    directory -> LLM preflight -> microphone preflight -> dependency wiring
    -> RecordApp launch.

    Plugin-supplied *llm_client*, *transcriber*, or *audio_source* override
    the defaults built by DependencyContainer.
    """
    from lazy_take_notes.l4_frameworks_and_drivers.apps.record import (  # noqa: PLC0415 -- deferred: Textual TUI not loaded for --help
        RecordApp,
    )
    from lazy_take_notes.l4_frameworks_and_drivers.container import (  # noqa: PLC0415 -- deferred: container not loaded for --help
        DependencyContainer,
    )

    config_path = ctx.obj['config_path']
    output_dir = ctx.obj['output_dir']
    config, infra, template_loader = load_config(config_path, output_dir)

    template = pick_template(template_loader)
    if template is None:
        return

    base_dir = resolve_base_dir(output_dir, config)
    out_dir = make_session_dir(base_dir, label)

    missing_digest, missing_interactive = preflight_llm(infra, config)
    if audio_source is None:
        preflight_microphone()

    container = DependencyContainer(
        config,
        template,
        out_dir,
        infra=infra,
        llm_client=llm_client,
        transcriber=transcriber,
        audio_source=audio_source,
    )
    app = RecordApp(
        config=config,
        template=template,
        output_dir=out_dir,
        controller=container.controller,
        audio_source=container.audio_source,
        transcriber=container.transcriber,
        missing_digest_models=missing_digest,
        missing_interactive_models=missing_interactive,
        label=label or '',
    )
    app.run()

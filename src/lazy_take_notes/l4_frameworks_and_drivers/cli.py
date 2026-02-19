"""CLI entry point for lazy-take-notes."""

from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path

import click

from lazy_take_notes import __version__


def _make_session_dir(base_dir: Path, label: str | None) -> Path:
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


@click.command()
@click.option(
    '-c',
    '--config',
    'config_path',
    default=None,
    type=click.Path(exists=True),
    help='Path to YAML config file.',
)
@click.option(
    '-o',
    '--output-dir',
    default=None,
    type=click.Path(),
    help='Base output directory (session subfolder created automatically).',
)
@click.option(
    '-l',
    '--label',
    default=None,
    help="Session label appended to the timestamp folder (e.g. 'sprint-review').",
)
@click.option(
    '-f',
    '--audio-file',
    'audio_file',
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help='Transcribe an audio file and generate a single final digest (no TUI).',
)
@click.version_option(version=__version__)
def cli(config_path, output_dir, label, audio_file):
    """lazy-take-notes -- TUI for real-time transcription and AI-assisted note-taking."""
    from lazy_take_notes.l3_interface_adapters.gateways.yaml_config_loader import (  # noqa: PLC0415 -- deferred: yaml stack not loaded on --help
        YamlConfigLoader,
    )
    from lazy_take_notes.l3_interface_adapters.gateways.yaml_template_loader import (  # noqa: PLC0415 -- deferred: yaml stack not loaded on --help
        YamlTemplateLoader,
    )
    from lazy_take_notes.l4_frameworks_and_drivers.infra_config import (  # noqa: PLC0415 -- deferred: not needed for --help
        InfraConfig,
        build_app_config,
    )
    from lazy_take_notes.l4_frameworks_and_drivers.template_picker import (  # noqa: PLC0415 -- deferred: Textual not loaded on --help
        TemplatePicker,
    )

    config_loader = YamlConfigLoader()
    template_loader = YamlTemplateLoader()

    try:
        overrides: dict = {}
        if output_dir:
            overrides['output'] = {'directory': output_dir}
        raw = config_loader.load_raw(config_path, overrides=overrides if overrides else None)
        config = build_app_config(raw)
        infra = InfraConfig.model_validate(raw)
    except FileNotFoundError as e:
        click.echo(f'Error: {e}', err=True)
        sys.exit(1)

    picker = TemplatePicker(show_audio_mode=audio_file is None)
    picker_result = picker.run()
    if picker_result is None:
        return
    tmpl_ref, audio_mode = picker_result

    try:
        template = template_loader.load(tmpl_ref)
    except FileNotFoundError as e:
        click.echo(f'Error: {e}', err=True)
        sys.exit(1)

    base_dir = Path(output_dir or config.output.directory)
    out_dir = _make_session_dir(base_dir, label)

    missing_digest, missing_interactive = _preflight_ollama(infra, config)

    if audio_file:
        from lazy_take_notes.l4_frameworks_and_drivers.batch_runner import (  # noqa: PLC0415 -- deferred: batch mode only, not loaded for TUI path
            run_batch,
        )

        run_batch(
            audio_path=Path(audio_file),
            config=config,
            template=template,
            out_dir=out_dir,
            infra=infra,
        )
        return

    _preflight_microphone()

    from lazy_take_notes.l4_frameworks_and_drivers.app import (  # noqa: PLC0415 -- deferred: Textual TUI not loaded for --help or --list-templates
        App,
    )
    from lazy_take_notes.l4_frameworks_and_drivers.container import (  # noqa: PLC0415 -- deferred: Textual TUI not loaded for --help or --list-templates
        DependencyContainer,
    )

    # Pre-initialize the resource tracker before Textual replaces sys.stderr.
    # ctx.Process.start() (spawn context) calls resource_tracker.ensure_running(),
    # which spawns the tracker subprocess and includes sys.stderr.fileno() in
    # fds_to_pass. Textual replaces sys.stderr with a stream that returns fileno()
    # == -1, which causes spawnv_passfds to raise ValueError. Calling
    # ensure_running() here (while sys.stderr is still the real fd) starts the
    # tracker once; all subsequent calls inside the TUI are no-ops.
    try:
        import multiprocessing.resource_tracker as _rt  # noqa: PLC0415 -- pre-init before Textual

        _rt.ensure_running()
    except Exception:  # noqa: S110 — best-effort; tracker may not exist on all platforms
        pass

    container = DependencyContainer(config, template, out_dir, infra=infra, audio_mode=audio_mode)
    app = App(
        config=config,
        template=template,
        output_dir=out_dir,
        controller=container.controller,
        audio_source=container.audio_source,
        transcriber=container.transcriber,
        missing_digest_models=missing_digest,
        missing_interactive_models=missing_interactive,
    )
    app.run()


def _preflight_ollama(infra, config) -> tuple[list[str], list[str]]:
    from lazy_take_notes.l3_interface_adapters.gateways.ollama_llm_client import (  # noqa: PLC0415 -- deferred: preflight only runs when starting a session
        OllamaLLMClient,
    )

    client = OllamaLLMClient(host=infra.ollama.host)
    ok, err = client.check_connectivity()
    if not ok:
        click.echo(f'Warning: Ollama not reachable ({err}). Digests will fail.', err=True)
        click.echo('Transcript-only mode: audio capture will still work.', err=True)
        return [], []

    unique_models = list(dict.fromkeys([config.digest.model, config.interactive.model]))
    missing = set(client.check_models(unique_models))
    missing_digest = [config.digest.model] if config.digest.model in missing else []
    missing_interactive = [config.interactive.model] if config.interactive.model in missing else []
    return missing_digest, missing_interactive


def _preflight_microphone() -> None:
    try:
        import sounddevice as sd  # noqa: PLC0415 -- deferred: not loaded on --help

        devices = sd.query_devices()
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        if not input_devices:
            click.echo('Warning: No input audio devices found.', err=True)
    except Exception as e:
        click.echo(f'Warning: Cannot query audio devices ({e}).', err=True)

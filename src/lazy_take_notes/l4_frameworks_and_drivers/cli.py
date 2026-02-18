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
    '-t',
    '--template',
    'template_ref',
    default=None,
    help='Template name or path to YAML template file.',
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
    '--list-templates',
    'show_templates',
    is_flag=True,
    default=False,
    help='List available templates and exit.',
)
@click.version_option(version=__version__)
def cli(config_path, template_ref, output_dir, label, show_templates):
    """lazy-take-notes -- TUI for real-time transcription and AI-assisted note-taking."""
    from lazy_take_notes.l3_interface_adapters.gateways.yaml_config_loader import (
        YamlConfigLoader,
    )
    from lazy_take_notes.l3_interface_adapters.gateways.yaml_template_loader import (
        YamlTemplateLoader,
        user_template_names,
    )

    config_loader = YamlConfigLoader()
    template_loader = YamlTemplateLoader()

    if show_templates:
        user_names = user_template_names()
        templates = template_loader.list_templates()
        for t in templates:
            tag = ' [user]' if t.name in user_names else ''
            click.echo(f'{t.name:<25s} {t.description} [{t.locale}]{tag}')
        return

    from lazy_take_notes.l4_frameworks_and_drivers.infra_config import (
        InfraConfig,
        build_app_config,
    )

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

    if template_ref:
        tmpl_ref = template_ref
    else:
        from lazy_take_notes.l4_frameworks_and_drivers.template_picker import (
            TemplatePicker,
        )

        picker = TemplatePicker()
        tmpl_ref = picker.run()
        if tmpl_ref is None:
            return

    try:
        template = template_loader.load(tmpl_ref)
    except FileNotFoundError as e:
        click.echo(f'Error: {e}', err=True)
        sys.exit(1)

    base_dir = Path(output_dir or config.output.directory)
    out_dir = _make_session_dir(base_dir, label)

    _preflight_ollama(infra)
    _preflight_microphone()

    from lazy_take_notes.l4_frameworks_and_drivers.app import App
    from lazy_take_notes.l4_frameworks_and_drivers.container import DependencyContainer

    container = DependencyContainer(config, template, out_dir, infra=infra)
    app = App(config=config, template=template, output_dir=out_dir, controller=container.controller)
    app.run()


def _preflight_ollama(infra) -> None:
    from lazy_take_notes.l3_interface_adapters.gateways.ollama_llm_client import (
        OllamaLLMClient,
    )

    client = OllamaLLMClient(host=infra.ollama.host)
    ok, err = client.check_connectivity()
    if not ok:
        click.echo(f'Warning: Ollama not reachable ({err}). Digests will fail.', err=True)
        click.echo('Transcript-only mode: audio capture will still work.', err=True)


def _preflight_microphone() -> None:
    try:
        import sounddevice as sd

        devices = sd.query_devices()
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        if not input_devices:
            click.echo('Warning: No input audio devices found.', err=True)
    except Exception as e:
        click.echo(f'Warning: Cannot query audio devices ({e}).', err=True)

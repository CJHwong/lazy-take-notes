"""CLI entry point for lazy-take-notes."""

from __future__ import annotations

import sys
from importlib.metadata import entry_points
from pathlib import Path

import click

from lazy_take_notes import __version__
from lazy_take_notes.l4_frameworks_and_drivers.cli_helpers import (
    load_config as _load_config,
)
from lazy_take_notes.l4_frameworks_and_drivers.cli_helpers import (
    make_session_dir as _make_session_dir,
)
from lazy_take_notes.l4_frameworks_and_drivers.cli_helpers import (
    pick_template as _pick_template,
)
from lazy_take_notes.l4_frameworks_and_drivers.cli_helpers import (
    preflight_llm as _preflight_llm,
)
from lazy_take_notes.l4_frameworks_and_drivers.cli_helpers import (
    run_transcribe as _run_transcribe,
)


def _pre_init_resource_tracker() -> None:  # pragma: no cover -- best-effort platform guard
    """Pre-initialize the multiprocessing resource tracker before Textual replaces sys.stderr.

    ctx.Process.start() (spawn context) calls resource_tracker.ensure_running(),
    which spawns the tracker subprocess and includes sys.stderr.fileno() in
    fds_to_pass. Textual replaces sys.stderr with a stream that returns fileno()
    == -1, which causes spawnv_passfds to raise ValueError. Calling
    ensure_running() here (while sys.stderr is still the real fd) starts the
    tracker once; all subsequent calls inside the TUI are no-ops.
    """
    try:
        import multiprocessing.resource_tracker as _rt  # noqa: PLC0415 -- pre-init before Textual

        _rt.ensure_running()
    except Exception:  # noqa: S110 -- best-effort; tracker may not exist on all platforms
        pass


@click.group(invoke_without_command=True)
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
@click.version_option(version=__version__)
@click.pass_context
def cli(ctx, config_path, output_dir):
    """lazy-take-notes -- live transcription & AI summaries in your terminal."""
    ctx.ensure_object(dict)
    ctx.obj['config_path'] = config_path
    ctx.obj['output_dir'] = output_dir

    _pre_init_resource_tracker()

    if ctx.invoked_subcommand is not None:
        return

    from lazy_take_notes.l4_frameworks_and_drivers.pickers.welcome_picker import (  # noqa: PLC0415 -- deferred: Textual not loaded on --help
        WelcomePicker,
    )

    while True:
        mode = WelcomePicker().run()
        if mode == 'record':
            ctx.invoke(record)
            return
        elif mode == 'transcribe':
            ctx.invoke(transcribe)
            return
        elif mode == 'view':
            ctx.invoke(view)
        elif mode == 'create-template':
            ctx.invoke(create_template)
        elif mode == 'config':
            ctx.invoke(config)
        else:
            return


@cli.command()
@click.option(
    '-l',
    '--label',
    default=None,
    help="Session label appended to the timestamp folder (e.g. 'sprint-review').",
)
@click.pass_context
def record(ctx, label):
    """Start a live recording session with transcription and digest."""
    config_path = ctx.obj['config_path']
    output_dir = ctx.obj['output_dir']
    config, infra, template_loader = _load_config(config_path, output_dir)

    template = _pick_template(template_loader)
    if template is None:
        return

    base_dir = Path(output_dir or config.output.directory)
    out_dir = _make_session_dir(base_dir, label)

    missing_digest, missing_interactive = _preflight_llm(infra, config)
    _preflight_microphone()

    from lazy_take_notes.l4_frameworks_and_drivers.apps.record import (  # noqa: PLC0415 -- deferred: Textual TUI not loaded for --help
        RecordApp,
    )
    from lazy_take_notes.l4_frameworks_and_drivers.container import (  # noqa: PLC0415 -- deferred: Textual TUI not loaded for --help
        DependencyContainer,
    )

    container = DependencyContainer(config, template, out_dir, infra=infra)
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


@cli.command()
@click.argument('audio_file', type=click.Path(dir_okay=False), required=False, default=None)
@click.option(
    '-l',
    '--label',
    default=None,
    help="Session label appended to the timestamp folder (e.g. 'sprint-review').",
)
@click.pass_context
def transcribe(ctx, audio_file, label):
    """Transcribe an audio file with streaming TUI and generate a final digest."""
    if audio_file is None:
        from lazy_take_notes.l4_frameworks_and_drivers.pickers.file_picker import (  # noqa: PLC0415 -- deferred: Textual not loaded on --help
            FilePicker,
        )

        selected = FilePicker().run()
        if selected is None:
            return
        audio_file = str(selected)
    if not Path(audio_file).is_file():
        click.echo(f'Error: {audio_file!r} is not a valid file.', err=True)
        sys.exit(1)

    _run_transcribe(ctx, audio_path=Path(audio_file), label=label)


@cli.command()
@click.pass_context
def view(ctx):
    """Browse a previously saved session (transcript + digest, read-only)."""
    config_path = ctx.obj['config_path']
    output_dir = ctx.obj['output_dir']
    config, _infra, _template_loader = _load_config(config_path, output_dir)

    base_dir = Path(output_dir or config.output.directory)

    from lazy_take_notes.l4_frameworks_and_drivers.apps.view import (  # noqa: PLC0415 -- deferred: Textual TUI not loaded for --help
        ViewApp,
    )
    from lazy_take_notes.l4_frameworks_and_drivers.pickers.session_picker import (  # noqa: PLC0415 -- deferred: Textual not loaded on --help
        SessionPicker,
    )

    while True:
        picker = SessionPicker(sessions_dir=base_dir)
        session_dir = picker.run()
        if session_dir is None:
            return

        app = ViewApp(session_dir=session_dir)
        app.run()


@cli.command('create-template')
@click.pass_context
def create_template(ctx):
    """Build a custom template with AI assistance."""
    _launch_template_builder()


@cli.command()
@click.pass_context
def config(ctx):
    """Open the configuration editor."""
    from lazy_take_notes.l4_frameworks_and_drivers.apps.config import (  # noqa: PLC0415 -- deferred: Textual TUI not loaded for --help
        ConfigApp,
    )

    ConfigApp().run()


def _launch_template_builder() -> None:
    """Launch the TemplateBuilderApp."""
    from lazy_take_notes.l4_frameworks_and_drivers.apps.template_builder import (  # noqa: PLC0415 -- deferred: Textual TUI not loaded for --help
        TemplateBuilderApp,
    )

    TemplateBuilderApp().run()


def _load_plugins(group: click.Group) -> None:
    """Discover and register plugin subcommands via entry_points."""
    for ep in entry_points(group='lazy_take_notes.plugins'):
        try:
            command = ep.load()
            if isinstance(command, click.Command):
                group.add_command(command, ep.name)
            else:
                click.echo(
                    f'Warning: plugin {ep.name!r} is not a click command, skipping.',
                    err=True,
                )
        except Exception as exc:  # noqa: BLE001 -- plugin isolation: one broken plugin must not crash the CLI
            click.echo(f'Warning: plugin {ep.name!r} failed to load: {exc}', err=True)


_load_plugins(cli)


def _preflight_microphone() -> None:
    try:
        import sounddevice as sd  # noqa: PLC0415 -- deferred: not loaded on --help

        devices = sd.query_devices()
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        if not input_devices:
            click.echo('Warning: No input audio devices found.', err=True)
    except Exception as e:
        click.echo(f'Warning: Cannot query audio devices ({e}).', err=True)

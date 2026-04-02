"""CLI entry point for lazy-take-notes."""

from __future__ import annotations

import os
import sys
from importlib.metadata import entry_points
from pathlib import Path

import click

from lazy_take_notes import __version__
from lazy_take_notes.l4_frameworks_and_drivers.cli_helpers import (
    load_config as _load_config,
)
from lazy_take_notes.l4_frameworks_and_drivers.cli_helpers import (
    resolve_base_dir as _resolve_base_dir,
)
from lazy_take_notes.l4_frameworks_and_drivers.cli_helpers import (
    run_transcribe as _run_transcribe,
)


def _clear_normal_screen() -> None:  # pragma: no cover -- terminal escape; no-op in test
    """Clear the normal screen buffer before launching Textual apps.

    Each Textual App enters/exits the alternate screen buffer independently.
    Between apps the terminal briefly restores the normal screen. Clearing it
    beforehand means the flash shows a blank screen instead of shell history.
    """
    sys.stdout.write('\033[2J\033[H')
    sys.stdout.flush()


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
    envvar='LTN_OUTPUT_DIR',
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

    # FIXME: replace with a single ShellApp that uses Textual Screens for
    # pickers and main apps, eliminating inter-app terminal flicker entirely.
    kiosk = os.environ.get('LTN_KIOSK') == '1'
    _clear_normal_screen()
    while True:
        mode = WelcomePicker().run()
        if mode == 'record':
            ctx.invoke(record)
            if not kiosk:
                return
        elif mode == 'transcribe':
            ctx.invoke(transcribe)
            if not kiosk:
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
    from lazy_take_notes.l4_frameworks_and_drivers.cli_helpers import (  # noqa: PLC0415 -- deferred: not loaded on --help
        run_record as _run_record_impl,
    )

    _run_record_impl(ctx, label=label)


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

        _clear_normal_screen()
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

    base_dir = _resolve_base_dir(output_dir, config)

    from lazy_take_notes.l4_frameworks_and_drivers.apps.view import (  # noqa: PLC0415 -- deferred: Textual TUI not loaded for --help
        ViewApp,
    )
    from lazy_take_notes.l4_frameworks_and_drivers.pickers.session_picker import (  # noqa: PLC0415 -- deferred: Textual not loaded on --help
        SessionPicker,
    )

    _clear_normal_screen()
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


@cli.group('plugin')
def plugin_group():
    """Manage uvx plugins (add, remove, list)."""


@plugin_group.command('add')
@click.argument('spec')
def plugin_add(spec):
    """Add a plugin by pip/uvx spec (e.g. 'ltn-youtube @ git+https://...')."""
    from lazy_take_notes.l4_frameworks_and_drivers.plugin_manifest import (  # noqa: PLC0415 -- deferred
        add_plugin,
        parse_spec_name,
    )

    name = parse_spec_name(spec)
    click.echo(f'Validating {name}...', nl=False)
    err = add_plugin(spec)
    if err is not None:
        click.echo(f' failed\n{err}', err=True)
        raise SystemExit(1)
    click.echo(f' ok\nPlugin {name} added.')


@plugin_group.command('remove')
@click.argument('name')
def plugin_remove(name):
    """Remove a plugin by package name (e.g. 'ltn-youtube')."""
    from lazy_take_notes.l4_frameworks_and_drivers.plugin_manifest import (  # noqa: PLC0415 -- deferred
        remove_plugin,
    )

    removed = remove_plugin(name)
    if removed:
        click.echo(f'Plugin {name} removed.')
    else:
        click.echo(f'Plugin {name} not found.', err=True)


@plugin_group.command('list')
def plugin_list():
    """List installed plugins."""
    from lazy_take_notes.l4_frameworks_and_drivers.plugin_manifest import (  # noqa: PLC0415 -- deferred
        load_plugins,
        parse_spec_name,
    )

    specs = load_plugins()
    if not specs:
        click.echo('No plugins installed.')
        return
    for spec in specs:
        click.echo(f'  {parse_spec_name(spec)}  ({spec})')


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

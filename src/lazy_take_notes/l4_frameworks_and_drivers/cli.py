"""CLI entry point for lazy-take-notes."""

from __future__ import annotations

import os
import sys
from importlib.metadata import entry_points
from pathlib import Path

import click

from lazy_take_notes import __version__
from lazy_take_notes.l1_entities.session_status import LIVE_STATES
from lazy_take_notes.l3_interface_adapters.gateways.session_reader import (
    SessionInfo,
    latest_session,
    list_sessions,
    read_notes,
    read_transcript,
)
from lazy_take_notes.l3_interface_adapters.gateways.session_status import (
    STATUS_FILE,
    is_pid_alive,
    read_status,
)
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


def _check_template_language(template_name, language) -> None:
    """Reject the ambiguous combination of --template and --language."""
    if template_name and language:
        raise click.ClickException('Use --template or --language, not both.')


def _require_headless_template(template_name, language) -> None:
    """Headless has no picker, so the language must be explicit (no silent default)."""
    if not template_name and not language:
        raise click.ClickException('--headless requires --language (e.g. --language en) or --template.')


@cli.command()
@click.option(
    '-l',
    '--label',
    default=None,
    help="Session label appended to the timestamp folder (e.g. 'sprint-review').",
)
@click.option('--headless', is_flag=True, help='Run without the TUI; write files only. Track it with `status`.')
@click.option(
    '--template',
    'template_name',
    default=None,
    help='Template key, e.g. lecture_notes_en. Skips the picker.',
)
@click.option(
    '--language',
    default=None,
    help="Language/locale, e.g. en or zh-TW. Uses that locale's default template.",
)
@click.option('--mute-mic', is_flag=True, help='Start with the microphone muted (capture system audio only).')
@click.pass_context
def record(ctx, label, headless, template_name, language, mute_mic):
    """Start a live recording session with transcription and digest."""
    _check_template_language(template_name, language)
    if headless:
        _require_headless_template(template_name, language)
        from lazy_take_notes.l4_frameworks_and_drivers.headless import (  # noqa: PLC0415 -- deferred: not loaded on --help
            run_record_headless,
        )

        run_record_headless(ctx, template_name=template_name, language=language, label=label, mute_mic=mute_mic)
        return

    from lazy_take_notes.l4_frameworks_and_drivers.cli_helpers import (  # noqa: PLC0415 -- deferred: not loaded on --help
        run_record as _run_record_impl,
    )

    _run_record_impl(ctx, label=label, template_name=template_name, language=language, mute_mic=mute_mic)


@cli.command()
@click.argument('audio_file', type=click.Path(dir_okay=False), required=False, default=None)
@click.option(
    '-l',
    '--label',
    default=None,
    help="Session label appended to the timestamp folder (e.g. 'sprint-review').",
)
@click.option('--headless', is_flag=True, help='Run without the TUI; write files only. Track it with `status`.')
@click.option(
    '--template',
    'template_name',
    default=None,
    help='Template key, e.g. lecture_notes_en. Skips the picker.',
)
@click.option(
    '--language',
    default=None,
    help="Language/locale, e.g. en or zh-TW. Uses that locale's default template.",
)
@click.pass_context
def transcribe(ctx, audio_file, label, headless, template_name, language):
    """Transcribe an audio file with streaming TUI and generate a final digest."""
    _check_template_language(template_name, language)
    if headless:
        if audio_file is None:
            raise click.ClickException('--headless requires an audio file argument (no interactive picker).')
        _require_headless_template(template_name, language)
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

    if headless:
        from lazy_take_notes.l4_frameworks_and_drivers.headless import (  # noqa: PLC0415 -- deferred: not loaded on --help
            run_transcribe_headless,
        )

        run_transcribe_headless(
            ctx, audio_path=Path(audio_file), template_name=template_name, language=language, label=label
        )
        return

    _run_transcribe(ctx, audio_path=Path(audio_file), label=label, template_name=template_name, language=language)


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


_LS_DEFAULT_LIMIT = 20


def _read_base_dir(ctx) -> Path:
    """Resolve the base sessions directory from config + CLI overrides."""
    config_path = ctx.obj['config_path']
    output_dir = ctx.obj['output_dir']
    config, _infra, _template_loader = _load_config(config_path, output_dir)
    return _resolve_base_dir(output_dir, config)


def _match_by_name(items, name: str, noun: str):
    """Pick the item whose ``.name`` matches: exact first, then unique substring.

    Raises ClickException when nothing matches or the name is ambiguous. Items
    can be SessionInfo or Path — both expose ``.name``.
    """
    exact = [i for i in items if i.name == name]
    if exact:
        return exact[0]
    matches = [i for i in items if name in i.name]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise click.ClickException(f'No {noun} matching {name!r}.')
    raise click.ClickException(f'{name!r} matches multiple sessions: ' + ', '.join(i.name for i in matches))


def _resolve_session(base_dir: Path, name: str | None) -> SessionInfo:
    """Resolve a session by name, falling back to the newest one."""
    if name is None:
        latest = latest_session(base_dir)
        if latest is None:
            raise click.ClickException('No sessions found.')
        return latest
    return _match_by_name(list_sessions(base_dir), name, 'session')


def _emit_session_text(ctx, name: str | None, reader, kind: str) -> None:
    """Print a session artifact to stdout, or exit with an error."""
    info = _resolve_session(_read_base_dir(ctx), name)
    text = reader(info.dir)
    if text is None:
        raise click.ClickException(f'Session {info.name!r} has no {kind}.')
    click.echo(text, nl=False)


@cli.command('ls')
@click.option('--json', 'as_json', is_flag=True, help='Output as JSON for scripting (all sessions).')
@click.option(
    '-a',
    '--all',
    'show_all',
    is_flag=True,
    help=f'Show every session instead of the newest {_LS_DEFAULT_LIMIT}.',
)
@click.pass_context
def list_command(ctx, as_json, show_all):
    """List saved sessions, newest first (newest 20 by default)."""
    sessions = list_sessions(_read_base_dir(ctx))
    if as_json:
        import json  # noqa: PLC0415 -- deferred: only needed for --json

        payload = [{'name': s.name, 'dir': str(s.dir), 'has_notes': s.has_notes} for s in sessions]
        click.echo(json.dumps(payload))
        return
    if not sessions:
        click.echo('No sessions found.')
        return
    shown = sessions if show_all else sessions[:_LS_DEFAULT_LIMIT]
    for session in shown:
        marker = 'notes' if session.has_notes else '   -'
        click.echo(f'{marker}  {session.name}')
    hidden = len(sessions) - len(shown)
    if hidden:
        click.echo(f'... {hidden} more (use -a/--all to show every session)', err=True)


@cli.command()
@click.argument('session', required=False, default=None)
@click.pass_context
def transcript(ctx, session):
    """Print a session's transcript (newest session by default)."""
    _emit_session_text(ctx, session, read_transcript, 'transcript')


@cli.command()
@click.argument('session', required=False, default=None)
@click.pass_context
def notes(ctx, session):
    """Print a session's notes (newest session by default)."""
    _emit_session_text(ctx, session, read_notes, 'notes')


def _resolve_status_dir(base_dir: Path, name: str | None) -> Path:
    """Find a session directory that has a status file (newest by default)."""
    candidates = (
        [c for c in sorted(base_dir.iterdir(), reverse=True) if c.is_dir() and (c / STATUS_FILE).exists()]
        if base_dir.exists()
        else []
    )
    if not candidates:
        raise click.ClickException('No headless session found.')
    if name is None:
        return candidates[0]
    return _match_by_name(candidates, name, 'headless session')


@cli.command()
@click.argument('session', required=False, default=None)
@click.pass_context
def status(ctx, session):
    """Show a headless session's status (newest by default)."""
    session_dir = _resolve_status_dir(_read_base_dir(ctx), session)
    state = read_status(session_dir)
    if state is None:
        raise click.ClickException(f'Status file for {session_dir.name!r} is missing or unreadable.')

    display = state.state
    if state.state in LIVE_STATES and not is_pid_alive(state.pid):
        display = f'{state.state} (crashed — process {state.pid} not running)'

    click.echo(f'session:  {session_dir.name}')
    click.echo(f'state:    {display}')
    click.echo(f'pid:      {state.pid}')
    click.echo(f'segments: {state.segment_count}')
    click.echo(f'digests:  {state.digest_count}')
    click.echo(f'started:  {state.started_at}')
    click.echo(f'updated:  {state.updated_at}')
    if state.error:
        click.echo(f'error:    {state.error}')


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

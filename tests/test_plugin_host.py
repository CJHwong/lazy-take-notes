"""The plugin host, driven as the Rust app drives it: JSON lines over real pipes."""

from __future__ import annotations

import asyncio
import json
import os
import select
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import click
import pytest

import lazy_take_notes.plugin_api as public_api
from lazy_take_notes import plugin_host
from lazy_take_notes.l1_entities.transcript import TranscriptSegment
from lazy_take_notes.l2_use_cases.ports.llm_client import ChatResponse
from lazy_take_notes.l4_frameworks_and_drivers import plugin_api as internal_api

TIMEOUT_S = 10


@pytest.fixture(autouse=True)
def _restore_api():
    """The host replaces the API functions; put them back for the other tests."""
    saved = [(module, module.run_transcribe, module.run_record) for module in (public_api, internal_api)]
    yield
    for module, run_transcribe, run_record in saved:
        setattr(module, 'run_transcribe', run_transcribe)  # noqa: B010 -- the host replaced it by name
        setattr(module, 'run_record', run_record)  # noqa: B010 -- the host replaced it by name


class FakeDist:
    name = 'ltn-fake'


class FakeEntryPoint:
    def __init__(self, name, load, dist=None):
        self.name = name
        self._load = load
        self.dist = dist

    def load(self):
        return self._load()


class FakeClient:
    def __init__(self, infra):
        self.infra = infra

    async def chat(self, model, messages):
        if model == 'boom':
            raise LookupError('no such model')
        if model == 'slow':
            await asyncio.Event().wait()
        return ChatResponse(content=f'{model}:{messages[-1].role}:{messages[-1].content}', prompt_tokens=7)

    async def chat_single(self, model, prompt):
        return f'single {model} {prompt}'

    def check_connectivity(self):
        return True, f'reachable {self.infra.model_extra.get("fake_opts")}'

    def check_models(self, models):
        return [model for model in models if model != 'ok']


def broken():
    raise ImportError('missing dependency')


def command_plugin(seen):
    """Loads the way a real plugin does: the API name is bound at import time."""

    def load():
        from lazy_take_notes.plugin_api import run_record, run_transcribe

        @click.command()
        @click.argument('what')
        @click.pass_context
        def fetch(ctx, what):
            """Fetch something and transcribe it."""
            if what == 'exit2':
                sys.exit(2)
            if what == 'record':
                run_record(ctx)
            if what == 'override':
                run_transcribe(ctx, audio_path=Path('/a.wav'), llm_client=FakeClient(None))
            segments = [TranscriptSegment(text='hi', wall_start=0.0, wall_end=1.5)]
            seen.append(run_transcribe(ctx, subtitle_segments=segments, label='lbl'))

        return fetch

    return load


class Wire:
    """The app's side of the pipes, with the host serving on a thread."""

    def __init__(self, monkeypatch, groups):
        monkeypatch.setattr(plugin_host, 'entry_points', lambda group: groups.get(group, []))
        app_read, host_write = os.pipe()
        host_read, app_write = os.pipe()
        self.to_host = os.fdopen(app_write, 'wb')
        self.from_host = os.fdopen(app_read, 'rb')
        host_in = os.fdopen(host_read, 'rb')
        host_out = os.fdopen(host_write, 'wb')
        self.codes = []
        self.thread = threading.Thread(target=lambda: self.codes.append(plugin_host.serve(host_in, host_out)))
        self.thread.daemon = True
        self.thread.start()
        self.next_id = 0

    def send(self, message):
        self.to_host.write((json.dumps({'jsonrpc': '2.0', **message}) + '\n').encode())
        self.to_host.flush()

    def request(self, method, params=None):
        self.next_id += 1
        self.send({'id': self.next_id, 'method': method, 'params': params or {}})
        return self.next_id

    def read(self):
        ready, _, _ = select.select([self.from_host], [], [], TIMEOUT_S)
        assert ready, 'the host sent nothing'
        return json.loads(self.from_host.readline())

    def call(self, method, params=None):
        request_id = self.request(method, params)
        message = self.read()
        assert message['id'] == request_id, message
        return message

    def shutdown(self):
        assert self.call('shutdown') == {'jsonrpc': '2.0', 'id': self.next_id, 'result': {}}
        self.thread.join(TIMEOUT_S)
        assert self.codes == [0]
        self.to_host.close()


def llm_wire(monkeypatch):
    groups = {plugin_host.LLM_GROUP: [FakeEntryPoint('fake', lambda: FakeClient)]}
    wire = Wire(monkeypatch, groups)
    assert wire.read()['method'] == 'hello'
    return wire


def test_hello_lists_the_plugins_and_each_load_error(monkeypatch):
    groups = {
        plugin_host.LLM_GROUP: [
            FakeEntryPoint('fake', lambda: FakeClient),
            FakeEntryPoint('broken', broken, FakeDist()),
            FakeEntryPoint('constant', lambda: 3),
        ],
        plugin_host.COMMAND_GROUP: [
            FakeEntryPoint('fetch', command_plugin([])),
            FakeEntryPoint('not-click', lambda: print),
            FakeEntryPoint('broken-command', broken),
        ],
    }
    wire = Wire(monkeypatch, groups)
    assert wire.read() == {
        'jsonrpc': '2.0',
        'method': 'hello',
        'params': {
            'protocol': 1,
            'llm_providers': ['fake'],
            'commands': [{'name': 'fetch', 'help': 'Fetch something and transcribe it.'}],
            'load_errors': [
                {'plugin': 'broken', 'dist': 'ltn-fake', 'group': plugin_host.LLM_GROUP, 'error': 'missing dependency'},
                {
                    'plugin': 'constant',
                    'dist': None,
                    'group': plugin_host.LLM_GROUP,
                    'error': 'not a callable provider factory',
                },
                {
                    'plugin': 'not-click',
                    'dist': None,
                    'group': plugin_host.COMMAND_GROUP,
                    'error': 'not a click command',
                },
                {
                    'plugin': 'broken-command',
                    'dist': None,
                    'group': plugin_host.COMMAND_GROUP,
                    'error': 'missing dependency',
                },
            ],
        },
    }
    wire.shutdown()


def test_an_llm_provider_answers_every_call(monkeypatch):
    wire = llm_wire(monkeypatch)
    config = {'llm_provider': 'fake', 'ollama': {'host': 'http://h'}, 'fake_opts': {'k': 1}}
    created = wire.call('llm.create', {'provider': 'fake', 'config': config})
    assert created['result'] == {'client': 'c1'}
    messages = [{'role': 'system', 'content': 's'}, {'role': 'user', 'content': '你好'}]
    chat = wire.call('llm.chat', {'client': 'c1', 'model': 'm', 'messages': messages})
    assert chat['result'] == {'content': 'm:user:你好', 'prompt_tokens': 7}
    single = wire.call('llm.chat_single', {'client': 'c1', 'model': 'm', 'prompt': 'p'})
    assert single['result'] == {'content': 'single m p'}
    reach = wire.call('llm.check_connectivity', {'client': 'c1'})
    assert reach['result'] == {'ok': True, 'message': "reachable {'k': 1}"}
    models = wire.call('llm.check_models', {'client': 'c1', 'models': ['ok', 'gone']})
    assert models['result'] == {'missing': ['gone']}
    wire.shutdown()


def test_a_plugin_exception_carries_its_class(monkeypatch):
    wire = llm_wire(monkeypatch)
    wire.call('llm.create', {'provider': 'fake', 'config': {}})
    chat = wire.call('llm.chat', {'client': 'c1', 'model': 'boom', 'messages': []})
    assert chat['error'] == {'code': -32000, 'message': 'no such model', 'data': {'type': 'LookupError'}}
    unknown = wire.call('llm.create', {'provider': 'nope', 'config': {}})
    assert unknown['error'] == {'code': -32602, 'message': "Unknown provider 'nope'.", 'data': {'type': 'ValueError'}}
    for params in ({'provider': 'fake'}, {'provider': 'fake', 'config': {'ollama': 'not an object'}}):
        assert wire.call('llm.create', params)['error']['code'] == -32602, params
    wire.shutdown()


def test_bad_requests_get_protocol_errors(monkeypatch):
    wire = llm_wire(monkeypatch)
    assert wire.call('llm.dance')['error'] == {
        'code': -32601,
        'message': 'Unknown method: llm.dance',
        'data': {'type': 'MethodNotFound'},
    }
    assert wire.call('llm.chat', {'client': 'c9', 'model': 'm', 'messages': []})['error']['code'] == -32602
    wire.call('llm.create', {'provider': 'fake', 'config': {}})
    for params in (
        {'client': 'c1', 'model': 'm', 'messages': 'hi'},
        {'client': 'c1', 'model': 'm', 'messages': [{'role': 'robot', 'content': 'x'}]},
        {'client': 'c1', 'model': 3, 'messages': []},
    ):
        assert wire.call('llm.chat', params)['error']['code'] == -32602, params
    assert wire.call('llm.check_models', {'client': 'c1', 'models': [1]})['error']['code'] == -32602
    wire.next_id += 1
    wire.send({'id': wire.next_id, 'method': 'llm.chat', 'params': [1]})
    assert wire.read()['error']['code'] == -32602
    wire.to_host.write(b'not json\n')
    wire.to_host.flush()
    assert wire.read()['error']['code'] == -32700
    wire.to_host.write(b'[1]\n')
    wire.to_host.flush()
    assert wire.read()['error']['code'] == -32700
    # A blank line, a notification and an answer to nothing the host asked are ignored.
    wire.to_host.write(b'\r\n')
    wire.send({'method': 'note', 'params': {}})
    wire.send({'id': 99, 'result': {}})
    wire.shutdown()


def test_a_command_transcribes_through_the_app(monkeypatch):
    seen = []
    wire = Wire(monkeypatch, {plugin_host.COMMAND_GROUP: [FakeEntryPoint('fetch', command_plugin(seen))]})
    wire.read()
    run_id = wire.request('command.run', {'name': 'fetch', 'argv': ['it'], 'config_path': None, 'output_dir': '/o'})
    ask = wire.read()
    assert ask['method'] == 'session.transcribe'
    assert ask['params'] == {
        'audio_path': None,
        'segments': [{'text': 'hi', 'wall_start': 0.0, 'wall_end': 1.5}],
        'label': 'lbl',
        'template_name': None,
        'language': None,
    }
    wire.send({'id': ask['id'], 'result': {'session_dir': '/s/2026', 'notes_path': '/s/2026/notes.md'}})
    assert wire.read() == {'jsonrpc': '2.0', 'id': run_id, 'result': {'exit_code': 0}}
    assert seen[0].session_dir == Path('/s/2026')
    assert seen[0].notes_path == Path('/s/2026/notes.md')
    # A cancelled template picker is a null answer, and the plugin sees None.
    run_id = wire.request('command.run', {'name': 'fetch', 'argv': ['it']})
    ask = wire.read()
    wire.send({'id': ask['id'], 'result': None})
    assert wire.read()['result'] == {'exit_code': 0}
    assert seen[1] is None
    # An error answer raises in the plugin, which ends with exit 1.
    run_id = wire.request('command.run', {'name': 'fetch', 'argv': ['it']})
    ask = wire.read()
    wire.send({'id': ask['id'], 'error': {'code': -32000, 'message': 'no audio'}})
    assert wire.read()['result'] == {'exit_code': 1}
    wire.shutdown()


@pytest.mark.parametrize(
    ('argv', 'code'),
    [(['exit2'], 2), (['record'], 1), (['override'], 1), (['--help'], 0), ([], 2)],
)
def test_a_command_gives_its_exit_code(monkeypatch, argv, code):
    wire = Wire(monkeypatch, {plugin_host.COMMAND_GROUP: [FakeEntryPoint('fetch', command_plugin([]))]})
    wire.read()
    assert wire.call('command.run', {'name': 'fetch', 'argv': argv})['result'] == {'exit_code': code}
    wire.shutdown()


def test_a_command_request_needs_a_known_name_and_string_args(monkeypatch):
    wire = Wire(monkeypatch, {plugin_host.COMMAND_GROUP: [FakeEntryPoint('fetch', command_plugin([]))]})
    wire.read()
    assert wire.call('command.run', {'name': 'nope', 'argv': []})['error']['message'] == 'Unknown command: nope'
    assert wire.call('command.run', {'name': 'fetch', 'argv': [1]})['error']['code'] == -32602
    wire.shutdown()


def test_the_end_of_input_stops_the_host_and_fails_a_waiting_command(monkeypatch):
    wire = Wire(monkeypatch, {plugin_host.COMMAND_GROUP: [FakeEntryPoint('fetch', command_plugin([]))]})
    wire.read()
    wire.request('command.run', {'name': 'fetch', 'argv': ['it']})
    assert wire.read()['method'] == 'session.transcribe'
    wire.to_host.close()
    wire.thread.join(TIMEOUT_S)
    assert wire.codes == [0]


@pytest.mark.parametrize(('code', 'expected'), [(None, 0), (3, 3), ('bye', 1)])
def test_exit_codes_follow_python(code, expected, capsys):
    assert plugin_host.exit_code_of(SystemExit(code)) == expected
    assert capsys.readouterr().err == ('bye\n' if code == 'bye' else '')


def test_click_errors_become_exit_codes(capsys):
    @click.command()
    def raises_click():
        raise click.ClickException('bad input')

    @click.command()
    def aborts():
        raise click.Abort

    @click.command()
    def crashes():
        raise KeyError('k')

    assert plugin_host.run_click(raises_click, 'x', [], {}) == 1
    assert plugin_host.run_click(aborts, 'x', [], {}) == 1
    assert plugin_host.run_click(crashes, 'x', [], {}) == 1
    err = capsys.readouterr().err
    assert 'bad input' in err
    assert 'Aborted!' in err
    assert "KeyError: 'k'" in err


def test_the_real_host_keeps_a_plugin_print_off_the_protocol(tmp_path):
    """main() in its own process: a plugin's print and click.echo go to stderr."""
    (tmp_path / 'noisy_plugin.py').write_text(
        textwrap.dedent("""
            import click

            print('printed while loading')

            @click.command()
            def noisy():
                print('printed by the command')
                click.echo('echoed by the command')
        """),
        encoding='utf-8',
    )
    dist = tmp_path / 'noisy_plugin-0.1.dist-info'
    dist.mkdir()
    (dist / 'METADATA').write_text('Metadata-Version: 2.1\nName: noisy-plugin\nVersion: 0.1\n', encoding='utf-8')
    (dist / 'entry_points.txt').write_text(
        '[lazy_take_notes.plugins]\nnoisy = noisy_plugin:noisy\ngone = noisy_plugin:gone\n', encoding='utf-8'
    )
    env = {**os.environ, 'PYTHONPATH': str(tmp_path)}
    requests = [
        {'jsonrpc': '2.0', 'id': 1, 'method': 'command.run', 'params': {'name': 'noisy', 'argv': []}},
    ]
    process = subprocess.Popen(
        [sys.executable, '-m', 'lazy_take_notes.plugin_host'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    hello = json.loads(process.stdout.readline())
    assert hello['params']['commands'] == [{'name': 'noisy', 'help': ''}]
    [gone] = hello['params']['load_errors']
    assert (gone['plugin'], gone['dist']) == ('gone', 'noisy-plugin')
    process.stdin.write((json.dumps(requests[0]) + '\n').encode())
    process.stdin.flush()
    assert json.loads(process.stdout.readline()) == {'jsonrpc': '2.0', 'id': 1, 'result': {'exit_code': 0}}
    process.stdin.write(b'{"jsonrpc": "2.0", "id": 2, "method": "shutdown"}\n')
    process.stdin.flush()
    assert json.loads(process.stdout.readline()) == {'jsonrpc': '2.0', 'id': 2, 'result': {}}
    # The app may keep stdin open after shutdown; the host must still exit cleanly.
    assert process.wait(timeout=TIMEOUT_S) == 0
    process.stdin.close()
    assert process.stdout.read() == b''
    assert process.stderr is not None
    err = process.stderr.read().decode()
    for line in ('printed while loading', 'printed by the command', 'echoed by the command'):
        assert line in err


def test_a_write_to_a_closed_app_is_dropped():
    read_end, write_end = os.pipe()
    os.close(read_end)
    channel = plugin_host.Channel(os.fdopen(write_end, 'wb'))
    assert channel.send({'method': 'x'}) is False
    assert channel.send({'method': 'x'}) is False
    with pytest.raises(plugin_host.AppError, match='closed'):
        channel.request('session.transcribe', {})


def test_shutdown_cancels_a_call_still_running(monkeypatch):
    wire = llm_wire(monkeypatch)
    wire.call('llm.create', {'provider': 'fake', 'config': {}})
    wire.request('llm.chat', {'client': 'c1', 'model': 'slow', 'messages': []})
    wire.shutdown()


def test_lines_split_reads_and_keep_a_last_partial_line():
    read_end, write_end = os.pipe()
    os.write(write_end, b'{"a": 1}\n{"b"')
    os.write(write_end, b': 2}\n\ntail')
    os.close(write_end)
    assert list(plugin_host.lines(read_end)) == [b'{"a": 1}', b'{"b": 2}', b'', b'tail']
    os.close(read_end)

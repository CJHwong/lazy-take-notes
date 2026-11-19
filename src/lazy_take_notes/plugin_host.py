"""Run Python plugins for the Rust app.

The app starts this host as a child process and talks JSON-RPC 2.0 to it, one
object per line on stdin and stdout. The contract is in rust/PLAN.md, section
"Plugins". Plugins load as they do in the Python app, from the same entry
point groups, and call the same ``lazy_take_notes.plugin_api``: the host
replaces ``run_transcribe`` there with a call back to the app.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import traceback
from collections.abc import Callable, Coroutine, Iterator
from concurrent.futures import Future
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, BinaryIO

import click

import lazy_take_notes.plugin_api as public_api
from lazy_take_notes.l1_entities.chat_message import ChatMessage
from lazy_take_notes.l1_entities.session_files import SessionArtifacts
from lazy_take_notes.l4_frameworks_and_drivers import plugin_api as internal_api

PROTOCOL = 1
LLM_GROUP = 'lazy_take_notes.llm_providers'
COMMAND_GROUP = 'lazy_take_notes.plugins'

PARSE_ERROR = -32700
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
PLUGIN_RAISED = -32000

RUN_RECORD_MESSAGE = 'The Rust app does not run run_record from a plugin yet.'
OVERRIDE_MESSAGE = 'The Rust app does not take llm_client or transcriber from a plugin yet.'
CLOSED_MESSAGE = 'The app closed the connection.'


class AppError(RuntimeError):
    """The app answered a request from the host with an error."""


class ParamsError(ValueError):
    """A request from the app carried params the method cannot use."""


class UnknownProviderError(ValueError):
    """The app asked for a provider no plugin offers. Its type shows as ValueError."""


class Channel:
    """The JSON lines to the app, and the host's own requests waiting for an answer."""

    def __init__(self, writer: BinaryIO) -> None:
        self._writer = writer
        self._lock = threading.Lock()
        self._next_id = 0
        self._pending: dict[int, Future] = {}
        self._gone = False

    def send(self, message: dict[str, Any]) -> bool:
        """Write one line. False once the app has gone: there is no one left to tell."""
        line = json.dumps({'jsonrpc': '2.0', **message}, ensure_ascii=False) + '\n'
        with self._lock:
            if self._gone:
                return False
            try:
                self._writer.write(line.encode('utf-8'))
                self._writer.flush()
            except OSError:
                self._gone = True
                return False
        return True

    def notify(self, method: str, params: dict[str, Any]) -> None:
        self.send({'method': method, 'params': params})

    def respond(self, request_id: Any, result: Any) -> None:
        self.send({'id': request_id, 'result': result})

    def fail(self, request_id: Any, code: int, message: str, error_type: str) -> None:
        error = {'code': code, 'message': message, 'data': {'type': error_type}}
        self.send({'id': request_id, 'error': error})

    def request(self, method: str, params: dict[str, Any]) -> Any:
        """Ask the app and wait for its answer. Only a worker thread may call this:
        the answer arrives through the event loop."""
        future: Future = Future()
        with self._lock:
            self._next_id += 1
            request_id = self._next_id
            self._pending[request_id] = future
        if not self.send({'id': request_id, 'method': method, 'params': params}):
            with self._lock:
                self._pending.pop(request_id, None)
            raise AppError(CLOSED_MESSAGE)
        return future.result()

    def deliver(self, message: dict[str, Any]) -> None:
        """An answer from the app to one of the host's requests."""
        with self._lock:
            future = self._pending.pop(message.get('id'), None)
        if future is None:
            return
        error = message.get('error')
        if error is not None:
            text = error.get('message', 'The app failed.') if isinstance(error, dict) else str(error)
            future.set_exception(AppError(text))
        else:
            future.set_result(message.get('result'))

    def fail_pending(self, reason: str) -> None:
        with self._lock:
            pending = list(self._pending.values())
            self._pending.clear()
        for future in pending:
            future.set_exception(AppError(reason))


def install_api(channel: Channel) -> None:
    """Point the plugin API at the app, before any plugin imports it."""

    def run_transcribe(
        ctx: click.Context | None = None,
        *,
        audio_path: Path | str | None = None,
        subtitle_segments: list | None = None,
        label: str | None = None,
        llm_client: object | None = None,
        transcriber: object | None = None,
        template_name: str | None = None,
        language: str | None = None,
    ) -> SessionArtifacts | None:
        del ctx  # the app already knows the config path and the output folder
        if llm_client is not None or transcriber is not None:
            raise RuntimeError(OVERRIDE_MESSAGE)
        segments = None
        if subtitle_segments is not None:
            segments = [
                {'text': seg.text, 'wall_start': seg.wall_start, 'wall_end': seg.wall_end} for seg in subtitle_segments
            ]
        result = channel.request(
            'session.transcribe',
            {
                'audio_path': None if audio_path is None else str(audio_path),
                'segments': segments,
                'label': label,
                'template_name': template_name,
                'language': language,
            },
        )
        if result is None:
            return None
        notes = result.get('notes_path')
        return SessionArtifacts(session_dir=Path(result['session_dir']), notes_path=Path(notes) if notes else None)

    def run_record(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError(RUN_RECORD_MESSAGE)

    for module in (public_api, internal_api):
        setattr(module, 'run_transcribe', run_transcribe)  # noqa: B010 -- replacing a module function by name
        setattr(module, 'run_record', run_record)  # noqa: B010 -- replacing a module function by name


def _text(params: dict[str, Any], key: str, *, optional: bool = False) -> str | None:
    value = params.get(key)
    if value is None and optional:
        return None
    if not isinstance(value, str):
        raise ParamsError(f'{key} must be a string')
    return value


def _strings(params: dict[str, Any], key: str) -> list[str]:
    value = params.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ParamsError(f'{key} must be a list of strings')
    return value


def exit_code_of(error: SystemExit) -> int:
    """The exit code a ``sys.exit`` call asked for; a message goes to stderr, as Python prints it."""
    if error.code is None:
        return 0
    if isinstance(error.code, int):
        return error.code
    print(error.code, file=sys.stderr)
    return 1


def run_click(command: click.Command, name: str, argv: list[str], obj: dict[str, Any]) -> int:
    """Run a click command as the CLI would, and return its exit code."""
    try:
        returned = command.main(args=argv, prog_name=f'lazy-take-notes {name}', standalone_mode=False, obj=obj)
    except SystemExit as error:
        return exit_code_of(error)
    except click.ClickException as error:
        error.show()
        return error.exit_code
    except click.Abort:
        click.echo('Aborted!', err=True)
        return 1
    except Exception:  # noqa: BLE001 -- a plugin's crash becomes exit 1 with its traceback, as in Python
        traceback.print_exc()
        return 1
    # With standalone_mode=False, click returns the code of ctx.exit() (--help) as an int.
    return returned if isinstance(returned, int) else 0


class Host:
    """The loaded plugins, and the handlers for the app's requests."""

    def __init__(self, channel: Channel, loop: asyncio.AbstractEventLoop) -> None:
        self.channel = channel
        self.loop = loop
        self.providers: dict[str, Callable[[Any], Any]] = {}
        self.commands: dict[str, click.Command] = {}
        self.load_errors: list[dict[str, str | None]] = []
        self.clients: dict[str, Any] = {}
        self.handlers: dict[str, Callable[[dict[str, Any]], Coroutine[Any, Any, Any]]] = {
            'llm.create': self.llm_create,
            'llm.chat': self.llm_chat,
            'llm.chat_single': self.llm_chat_single,
            'llm.check_connectivity': self.llm_check_connectivity,
            'llm.check_models': self.llm_check_models,
        }

    # --- Loading ---

    def load(self) -> None:
        for ep in entry_points(group=LLM_GROUP):
            try:
                factory = ep.load()
            except Exception as exc:  # noqa: BLE001 -- plugin isolation: one broken plugin must not stop the host
                self._load_error(ep, LLM_GROUP, str(exc))
                continue
            if not callable(factory):
                self._load_error(ep, LLM_GROUP, 'not a callable provider factory')
                continue
            self.providers[ep.name] = factory
        for ep in entry_points(group=COMMAND_GROUP):
            try:
                command = ep.load()
            except Exception as exc:  # noqa: BLE001 -- plugin isolation: one broken plugin must not stop the host
                self._load_error(ep, COMMAND_GROUP, str(exc))
                continue
            if not isinstance(command, click.Command):
                self._load_error(ep, COMMAND_GROUP, 'not a click command')
                continue
            self.commands[ep.name] = command

    def _load_error(self, ep: Any, group: str, error: str) -> None:
        """One plugin that failed. ``dist`` names its package, so the app can tell which install broke."""
        dist = getattr(ep, 'dist', None)
        self.load_errors.append(
            {
                'plugin': ep.name,
                'dist': getattr(dist, 'name', None),
                'group': group,
                'error': error,
            }
        )

    def hello(self) -> dict[str, Any]:
        return {
            'protocol': PROTOCOL,
            'llm_providers': list(self.providers),
            'commands': [
                {'name': name, 'help': command.get_short_help_str(limit=300)} for name, command in self.commands.items()
            ],
            'load_errors': self.load_errors,
        }

    # --- Messages from the app ---

    def receive(self, raw: bytes) -> None:
        if not raw.strip():
            return
        try:
            message = json.loads(raw)
        except ValueError as exc:
            self.channel.fail(None, PARSE_ERROR, f'Not JSON: {exc}', 'ParseError')
            return
        if not isinstance(message, dict):
            self.channel.fail(None, PARSE_ERROR, 'A message must be a JSON object', 'ParseError')
            return
        if 'method' not in message:
            self.channel.deliver(message)
            return
        request_id = message.get('id')
        if request_id is None:
            return  # the app sends no notifications in protocol 1
        method = message['method']
        params = message.get('params') or {}
        if not isinstance(params, dict):
            self.channel.fail(request_id, INVALID_PARAMS, 'params must be an object', 'InvalidParams')
            return
        if method == 'shutdown':
            self.channel.respond(request_id, {})
            self.loop.stop()
            return
        if method == 'command.run':
            worker = threading.Thread(target=self.run_command, args=(request_id, params), daemon=True)
            worker.start()
            return
        handler = self.handlers.get(method)
        if handler is None:
            self.channel.fail(request_id, METHOD_NOT_FOUND, f'Unknown method: {method}', 'MethodNotFound')
            return
        self.loop.create_task(self.answer(request_id, handler, params))

    def closed(self) -> None:
        """The app closed stdin: nothing more will come, so nothing waits for it."""
        self.channel.fail_pending(CLOSED_MESSAGE)
        self.loop.stop()

    async def answer(
        self,
        request_id: Any,
        handler: Callable[[dict[str, Any]], Coroutine[Any, Any, Any]],
        params: dict[str, Any],
    ) -> None:
        try:
            result = await handler(params)
        except UnknownProviderError as exc:
            self.channel.fail(request_id, INVALID_PARAMS, str(exc), 'ValueError')
        except ParamsError as exc:
            self.channel.fail(request_id, INVALID_PARAMS, str(exc), 'InvalidParams')
        except Exception as exc:  # noqa: BLE001 -- a plugin's exception goes back to the app as its error
            self.channel.fail(request_id, PLUGIN_RAISED, str(exc), type(exc).__name__)
        else:
            self.channel.respond(request_id, result)

    # --- LLM providers ---

    async def llm_create(self, params: dict[str, Any]) -> dict[str, str]:
        from lazy_take_notes.l4_frameworks_and_drivers.config import (  # noqa: PLC0415 -- deferred: the config stack loads only when a provider is used
            InfraConfig,
        )

        provider = _text(params, 'provider') or ''
        config = params.get('config')
        if not isinstance(config, dict):
            raise ParamsError('config must be an object')
        factory = self.providers.get(provider)
        if factory is None:
            raise UnknownProviderError(f'Unknown provider {provider!r}.')
        try:
            infra = InfraConfig.model_validate(config)
        except ValueError as exc:
            raise ParamsError(f'Bad config: {exc}') from exc
        client = await self.loop.run_in_executor(None, factory, infra)
        client_id = f'c{len(self.clients) + 1}'
        self.clients[client_id] = client
        return {'client': client_id}

    def _client(self, params: dict[str, Any]) -> Any:
        client = self.clients.get(params.get('client'))
        if client is None:
            raise ParamsError(f'Unknown client: {params.get("client")!r}')
        return client

    async def llm_chat(self, params: dict[str, Any]) -> dict[str, Any]:
        client = self._client(params)
        model = _text(params, 'model')
        raw = params.get('messages')
        if not isinstance(raw, list):
            raise ParamsError('messages must be a list')
        try:
            messages = [ChatMessage(role=item['role'], content=item['content']) for item in raw]
        except (KeyError, TypeError, ValueError) as exc:
            raise ParamsError(f'Bad message: {exc}') from exc
        response = await client.chat(model, messages)
        return {'content': response.content, 'prompt_tokens': getattr(response, 'prompt_tokens', 0)}

    async def llm_chat_single(self, params: dict[str, Any]) -> dict[str, str]:
        client = self._client(params)
        content = await client.chat_single(_text(params, 'model'), _text(params, 'prompt'))
        return {'content': content}

    async def llm_check_connectivity(self, params: dict[str, Any]) -> dict[str, Any]:
        client = self._client(params)
        ok, message = await self.loop.run_in_executor(None, client.check_connectivity)
        return {'ok': bool(ok), 'message': str(message)}

    async def llm_check_models(self, params: dict[str, Any]) -> dict[str, list[str]]:
        client = self._client(params)
        models = _strings(params, 'models')
        missing = await self.loop.run_in_executor(None, client.check_models, models)
        return {'missing': list(missing)}

    # --- Commands ---

    def run_command(self, request_id: Any, params: dict[str, Any]) -> None:
        """Run a plugin command on its own thread: it may wait on the app."""
        try:
            name = _text(params, 'name') or ''
            argv = _strings(params, 'argv')
            obj = {
                'config_path': _text(params, 'config_path', optional=True),
                'output_dir': _text(params, 'output_dir', optional=True),
            }
        except ParamsError as exc:
            self.channel.fail(request_id, INVALID_PARAMS, str(exc), 'InvalidParams')
            return
        command = self.commands.get(name)
        if command is None:
            self.channel.fail(request_id, INVALID_PARAMS, f'Unknown command: {name}', 'InvalidParams')
            return
        self.channel.respond(request_id, {'exit_code': run_click(command, name, argv, obj)})


def lines(fd: int) -> Iterator[bytes]:
    """The lines read from *fd*, then a last one without a newline, if any.

    It reads the raw descriptor, not a buffered file: the reader thread is still
    blocked here when the host exits, and a buffered file's lock held at
    interpreter shutdown aborts the process.
    """
    pending = b''
    while chunk := os.read(fd, 65536):
        pending += chunk
        *complete, pending = pending.split(b'\n')
        yield from complete
    if pending:
        yield pending


def serve(reader: BinaryIO, writer: BinaryIO) -> int:
    """Load the plugins, say hello, and answer until shutdown or the end of input."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    channel = Channel(writer)
    install_api(channel)
    host = Host(channel, loop)
    host.load()
    channel.notify('hello', host.hello())

    def read() -> None:
        try:
            for raw in lines(reader.fileno()):
                loop.call_soon_threadsafe(host.receive, raw)
            loop.call_soon_threadsafe(host.closed)
        except RuntimeError:
            return  # the loop has closed after shutdown

    threading.Thread(target=read, name='plugin-host-reader', daemon=True).start()
    try:
        loop.run_forever()
    finally:
        tasks = asyncio.all_tasks(loop)
        for task in tasks:
            task.cancel()
        loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
        loop.close()
    return 0


def main() -> (
    None
):  # pragma: no cover -- fd wiring; test_the_real_host_keeps_a_plugin_print_off_the_protocol runs it in a child
    # The protocol owns the real stdout. Everything else that writes to stdout,
    # a plugin's print or a C library, lands on stderr instead.
    protocol = os.fdopen(os.dup(1), 'wb')
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    sys.exit(serve(sys.stdin.buffer, protocol))


if __name__ == '__main__':  # pragma: no cover -- the child process in the subprocess test
    main()

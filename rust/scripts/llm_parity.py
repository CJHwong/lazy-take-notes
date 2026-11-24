"""Check that the Rust LLM clients fail like the Python ones.

Drives both versions against one local server that answers with error codes,
and compares the text each shows and the number of requests each sends
(the openai SDK retries). Uses a fake key; no real provider is called.

Run from the repo root: uv run python rust/scripts/llm_parity.py
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess  # noqa: S404 -- fixed cargo command, no shell
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from lazy_take_notes.l3_interface_adapters.gateways.ollama_llm_client import OllamaLLMClient
from lazy_take_notes.l3_interface_adapters.gateways.openai_llm_client import OpenAICompatLLMClient

RUST_DIR = Path(__file__).resolve().parent.parent
EXAMPLE = RUST_DIR / 'target' / 'debug' / 'examples' / 'cross_check'
TEST_KEY = 'test-key'
STATE: dict = {'status': 500, 'body': b'{}', 'headers': {}, 'count': 0}

CASES = [
    # One body that both providers read as a success.
    (
        200,
        b'{"models":[],"data":[],"message":{"role":"assistant","content":"hi"},'
        b'"choices":[{"message":{"content":"hi"}}]}',
        {},
    ),
    (500, b'{"error":"boom"}', {}),
    (404, b'{"error":"model \'x\' not found"}', {}),
    (500, b'not json', {}),
    (401, b'{"error":{"message":"Incorrect API key","type":"invalid_request_error","param":null,"code":"bad"}}', {}),
    (429, b'{"error":{"message":"Rate limit reached","code":"rate_limit_exceeded"}}', {}),
    (503, b'{"error":"busy"}', {'x-should-retry': 'false'}),
    (400, b'plain text error', {}),
    (418, b'', {}),
]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 -- the stdlib parameter name
        pass

    def reply(self) -> None:
        self.rfile.read(int(self.headers.get('content-length') or 0))
        STATE['count'] += 1
        self.send_response(STATE['status'])
        for name, value in STATE['headers'].items():
            self.send_header(name, value)
        self.send_header('content-length', str(len(STATE['body'])))
        self.end_headers()
        self.wfile.write(STATE['body'])

    do_GET = do_POST = reply  # noqa: N815 -- the method names http.server dispatches to


def python_case(provider: str, operation: str, base: str) -> dict:
    if provider == 'ollama':
        client = OllamaLLMClient(host=base)
    else:
        client = OpenAICompatLLMClient(api_key=os.environ.get('LTN_TEST_KEY'), base_url=f'{base}/v1')
    try:
        if operation == 'chat':
            return {'ok': asyncio.run(client.chat_single('m', 'p'))}
        if operation == 'conn':
            ok, text = client.check_connectivity()
            return {'ok': None} if ok else {'err': text}
        return {'ok': client.check_models(['x'])}
    except Exception as error:  # noqa: BLE001 -- the error text is what is compared
        return {'err': f'{type(error).__name__}: {error}'}


def rust_case(provider: str, operation: str, base: str) -> dict:
    result = subprocess.run(  # noqa: S603 -- fixed argument list
        [str(EXAMPLE), 'llm', provider, operation, base],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def counted(run: Callable[[], dict]) -> tuple[dict, int]:
    STATE['count'] = 0
    return run(), STATE['count']


def compare(label: str, provider: str, operation: str, base: str) -> bool:
    python = counted(lambda: python_case(provider, operation, base))
    rust = counted(lambda: rust_case(provider, operation, base))
    same = python == rust
    print(f'{"PASS" if same else "FAIL"}  {label}')
    if not same:
        print(f'      python: {python}\n      rust:   {rust}')
    return same


def main() -> None:
    subprocess.run(['cargo', 'build', '-q', '--example', 'cross_check'], cwd=RUST_DIR, check=True)  # noqa: S607 -- cargo from PATH
    os.environ['LTN_TEST_KEY'] = TEST_KEY
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f'http://127.0.0.1:{server.server_address[1]}'
    results = []
    for status, body, headers in CASES:
        # A short retry-after keeps the openai retries fast; the delays have their own Rust test.
        STATE.update(status=status, body=body, headers={'retry-after-ms': '10', **headers})
        for provider in ('ollama', 'openai'):
            for operation in ('chat', 'conn', 'models'):
                results.append(compare(f'{provider} {operation} {status}', provider, operation, base))
    server.shutdown()
    server.server_close()
    for provider in ('ollama', 'openai'):
        for operation in ('chat', 'conn', 'models'):
            results.append(compare(f'{provider} {operation} refused', provider, operation, base))
    del os.environ['LTN_TEST_KEY']
    if 'OPENAI_API_KEY' not in os.environ:
        results.append(compare('openai conn without a key', 'openai', 'conn', base))
    raise SystemExit(0 if all(results) else 1)


if __name__ == '__main__':
    main()

"""Smoke test: verify SoundCardLoopbackSource works with real PulseAudio loopback.

Run inside a Docker container with PulseAudio + module-null-sink loaded.
The null-sink monitor provides a loopback device that produces silence — we
verify the full open/read/close cycle returns actual float32 data.
"""

from __future__ import annotations

import numpy as np

from lazy_take_notes.l3_interface_adapters.gateways.soundcard_loopback_source import SoundCardLoopbackSource


def main() -> None:
    print('--- SoundCardLoopbackSource smoke test ---')

    src = SoundCardLoopbackSource()

    print('[1/4] Opening loopback source (16kHz, mono)...')
    src.open(16000, 1)
    print('       OK — device found and recorder started')

    print('[2/4] Reading audio chunk (2s timeout)...')
    chunk = src.read(timeout=2.0)
    assert chunk is not None, 'FAIL: read() returned None — no data from loopback'
    assert isinstance(chunk, np.ndarray), f'FAIL: expected ndarray, got {type(chunk)}'
    assert chunk.dtype == np.float32, f'FAIL: expected float32, got {chunk.dtype}'
    print(f'       OK — got {len(chunk)} float32 samples')

    print('[3/4] Verifying chunk shape (should be 1-D mono)...')
    assert chunk.ndim == 1, f'FAIL: expected 1-D array, got {chunk.ndim}-D'
    print(f'       OK — shape={chunk.shape}')

    print('[4/4] Closing...')
    src.close()
    print('       OK — clean shutdown')

    print('\nSUCCESS: SoundCardLoopbackSource works on Linux with PulseAudio')


if __name__ == '__main__':
    main()

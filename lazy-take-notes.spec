# PyInstaller spec for lazy-take-notes
# Usage: pyinstaller lazy-take-notes.spec

import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None
src_root = Path('src/lazy_take_notes')

# Collect data files that must ship with the binary
datas = []

# YAML templates (loaded via importlib.resources)
for yaml_file in (src_root / 'templates').glob('*.yaml'):
    datas.append((str(yaml_file), 'lazy_take_notes/templates'))

# Textual CSS (resolved relative to the App class file)
datas.append((str(src_root / 'l4_frameworks_and_drivers/apps/app.tcss'), 'lazy_take_notes/l4_frameworks_and_drivers/apps'))

# Native coreaudio-tap binary (macOS only)
coreaudio_bin = src_root / '_native/bin/coreaudio-tap'
if coreaudio_bin.exists():
    datas.append((str(coreaudio_bin), 'lazy_take_notes/_native/bin'))

# Textual ships CSS alongside its Python files; PyInstaller won't grab them by default
datas += collect_data_files('textual')

a = Analysis(
    ['src/lazy_take_notes/__main__.py'],
    pathex=['src'],
    binaries=[],
    datas=datas,
    hiddenimports=[
        # Textual uses __getattr__ lazy imports everywhere; grab all submodules
        *collect_submodules('textual'),
        *collect_submodules('lazy_take_notes'),
        # C extension backends
        '_sounddevice_data',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['scripts/pyinstaller_runtime_hook.py'],
    excludes=[],
    noarchive=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    name='lazy-take-notes',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,  # TUI app needs a console/PTY
)

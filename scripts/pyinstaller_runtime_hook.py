"""PyInstaller runtime hook: frozen-build setup that must not live in core app code.

1. freeze_support() - intercepts multiprocessing child process re-execution
2. PATH restoration - Trolley bundles start with minimal PATH (/usr/bin:/bin)
3. atexit forced exit - multiprocessing threads can hang during interpreter shutdown
"""

import atexit
import multiprocessing
import os
import subprocess
import sys

# ── 1. Multiprocessing freeze support ───────────────────────────────
multiprocessing.freeze_support()


# ── 2. Restore full system PATH ─────────────────────────────────────
def _restore_path():
    if sys.platform == 'darwin':
        try:
            result = subprocess.run(
                ['/usr/libexec/path_helper', '-s'],
                capture_output=True,
                text=True,
                timeout=5,
            )
            for line in result.stdout.splitlines():
                if line.startswith('PATH="'):
                    os.environ['PATH'] = line.split('"')[1]
                    return
        except Exception:
            pass
    else:
        try:
            with open('/etc/environment') as fh:
                for line in fh:
                    if line.startswith('PATH='):
                        system_path = line.split('=', 1)[1].strip().strip('"')
                        current = os.environ.get('PATH', '')
                        current_dirs = set(current.split(os.pathsep))
                        new_dirs = [d for d in system_path.split(os.pathsep) if d not in current_dirs]
                        if new_dirs:
                            os.environ['PATH'] = current + os.pathsep + os.pathsep.join(new_dirs)
                        return
        except Exception:
            pass


_restore_path()


# ── 3. Force exit on shutdown ────────────────────────────────────────
# Multiprocessing resource tracker threads and ThreadPoolExecutor atexit
# handlers can hang during interpreter shutdown in frozen builds.
atexit.register(lambda: os._exit(0))

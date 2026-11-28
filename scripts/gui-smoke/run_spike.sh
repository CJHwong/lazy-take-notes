#!/bin/bash
# Phase 3 spike: the window stays up under Xvfb for 10 s with no panic.
set -euo pipefail

LOG=/tmp/gui-spike.log
timeout 10 xvfb-run -s "-screen 0 1280x800x24" ./target/release/lazy-take-notes-gui \
  >"$LOG" 2>&1 || code=$?
code="${code:-0}"

if [[ "$code" -ne 124 ]]; then
  echo "FAIL: the window exited with $code before the timeout" >&2
  cat "$LOG" >&2
  exit 1
fi
if grep -qi "panic" "$LOG"; then
  echo "FAIL: the window panicked" >&2
  cat "$LOG" >&2
  exit 1
fi
echo "PASS: the window stayed up for 10 s"

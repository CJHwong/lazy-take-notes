#!/usr/bin/env bash
# Run the Python and Rust TUIs side by side in tmux, send both the same keys,
# and diff what each screen shows.
#
# Usage, from the repo root:
#   rust/scripts/tui_compare.sh "ARGS" KEY... [-- KEY...]
# ARGS are the lazy-take-notes arguments, the same for both versions. Each KEY
# is a tmux send-keys key name (Enter, Escape, Down, q). "wait:N" waits N
# seconds; "snap" saves both screens and prints their diff.
# Env: COLS, ROWS (default 110x32), OUT (default /tmp/tui-compare), RUST_BIN.
set -u
COLS=${COLS:-110}
ROWS=${ROWS:-32}
OUT=${OUT:-/tmp/tui-compare}
RUST_BIN=${RUST_BIN:-rust/target/debug/lazy-take-notes}
args=$1
shift
mkdir -p "$OUT"
for side in py rs; do
  tmux kill-session -t "cmp-$side" 2>/dev/null
done
tmux new-session -d -s cmp-py -x "$COLS" -y "$ROWS" "uv run lazy-take-notes $args; sleep 600"
tmux new-session -d -s cmp-rs -x "$COLS" -y "$ROWS" "$RUST_BIN $args; sleep 600"
sleep 4
snap=0
for key in "$@"; do
  case $key in
    wait:*) sleep "${key#wait:}" ;;
    snap)
      snap=$((snap + 1))
      sleep 1
      for side in py rs; do
        tmux capture-pane -p -t "cmp-$side" > "$OUT/$side-$snap.txt"
      done
      echo "=== snap $snap: diff py rs"
      diff "$OUT/py-$snap.txt" "$OUT/rs-$snap.txt" && echo "(identical)"
      ;;
    *)
      for side in py rs; do
        tmux send-keys -t "cmp-$side" "$key"
      done
      sleep 0.5
      ;;
  esac
done
for side in py rs; do
  tmux kill-session -t "cmp-$side" 2>/dev/null
done

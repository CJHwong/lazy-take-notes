#!/usr/bin/env bash
# Run the Python and Rust TUIs side by side in tmux, send both the same keys,
# and diff what each screen shows.
#
# Usage, from the repo root:
#   rust/scripts/tui_compare.sh "ARGS" KEY... [-- KEY...]
# ARGS are the lazy-take-notes arguments, the same for both versions; "{side}"
# in ARGS becomes py or rs, so each side can write its own folder. Each KEY
# is a tmux send-keys key name (Enter, Escape, Down, q). "wait:N" waits N
# seconds; "snap" saves both screens (text, and .ansi with colors) and prints
# the text diff.
# Env: COLS, ROWS (default 110x32), OUT (default /tmp/tui-compare), RUST_BIN, PY_BIN.
# HOME_ROOT: when set, each side runs with HOME=$HOME_ROOT/py or $HOME_ROOT/rs,
# so config and template writes stay apart from each other and from the user's.
# The sessions run on their own tmux server (-L ltn-cmp) with focus events on.
set -u
COLS=${COLS:-110}
ROWS=${ROWS:-32}
OUT=${OUT:-/tmp/tui-compare}
RUST_BIN=${RUST_BIN:-rust/target/debug/lazy-take-notes}
PY_BIN=${PY_BIN:-uv run lazy-take-notes}
args=$1
shift
mkdir -p "$OUT"
tmx() { tmux -L ltn-cmp "$@"; }
env_for() {
  if [ -n "${HOME_ROOT:-}" ]; then
    echo "env HOME=$HOME_ROOT/$1 UV_CACHE_DIR=${UV_CACHE_DIR:-$HOME/.cache/uv}"
  fi
}
for side in py rs; do
  tmx kill-session -t "cmp-$side" 2>/dev/null
done
tmx new-session -d -s cmp-py -x "$COLS" -y "$ROWS" "$(env_for py) $PY_BIN ${args//\{side\}/py}; sleep 600"
tmx set-option -g focus-events on
tmx new-session -d -s cmp-rs -x "$COLS" -y "$ROWS" "$(env_for rs) $RUST_BIN ${args//\{side\}/rs}; sleep 600"
sleep 4
snap=0
for key in "$@"; do
  case $key in
    wait:*) sleep "${key#wait:}" ;;
    snap)
      snap=$((snap + 1))
      sleep 1
      for side in py rs; do
        tmx capture-pane -p -t "cmp-$side" > "$OUT/$side-$snap.txt"
        tmx capture-pane -p -e -t "cmp-$side" > "$OUT/$side-$snap.ansi"
      done
      echo "=== snap $snap: diff py rs"
      diff "$OUT/py-$snap.txt" "$OUT/rs-$snap.txt" && echo "(identical)"
      ;;
    *)
      for side in py rs; do
        tmx send-keys -t "cmp-$side" "$key"
      done
      sleep 0.5
      ;;
  esac
done
tmx kill-server 2>/dev/null

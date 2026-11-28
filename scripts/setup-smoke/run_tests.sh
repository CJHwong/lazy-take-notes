#!/bin/bash
# setup.sh smoke test: install from a served release tarball, check the links
# and the idempotent second run, replace an old wrapper, then record the null
# sink's monitor while the LibriVox sentence plays and check the transcript.
set -euo pipefail

TAG="${LTN_TAG:?}"
case "$(uname -m)" in
  x86_64) ARCH=x86_64 ;;
  aarch64 | arm64) ARCH=arm64 ;;
  *) echo "unsupported arch: $(uname -m)" >&2; exit 1 ;;
esac

GREEN='\033[0;32m'
RED='\033[0;31m'
BOLD='\033[1m'
RESET='\033[0m'
pass() { echo -e "${GREEN}${BOLD}PASS${RESET}: $1"; }
fail() { echo -e "${RED}${BOLD}FAIL${RESET}: $1"; exit 1; }

# Serve the release the way the setup-smoke plan describes: a local folder,
# not GitHub. setup.sh only needs LTN_RELEASE_URL.
mkdir -p "/srv/$TAG"
cp /artifacts/lazy-take-notes-$ARCH-linux.tar.gz* "/srv/$TAG/"
python3 -m http.server 8000 --directory /srv >/tmp/http.log 2>&1 &
HTTP_PID=$!
trap 'kill "$HTTP_PID" 2>/dev/null || true' EXIT
for _ in $(seq 1 20); do
  if curl -fs "http://127.0.0.1:8000/$TAG/lazy-take-notes-$ARCH-linux.tar.gz.sha256" -o /dev/null; then
    break
  fi
  sleep 0.2
done

run_setup() { # home
  env -u XDG_CONFIG_HOME HOME="$1" \
    LTN_VERSION="$TAG" \
    LTN_RELEASE_URL="http://127.0.0.1:8000" \
    LTN_PROVIDER=openai \
    LTN_OPENAI_KEY=sk-test-key-12345 \
    bash /src/setup.sh -y
}

# ═════════════════════════════════════════════════════════════════════════════
# Test 1: clean install
# ═════════════════════════════════════════════════════════════════════════════
FRESH="$(mktemp -d /home/testuser/fresh.XXXXXX)"
run_setup "$FRESH"
CONFIG="$FRESH/.config/lazy-take-notes/config.yaml"
[[ -f "$CONFIG" ]] || fail "config.yaml not created"
grep -q 'llm_provider: "openai"' "$CONFIG" || fail "config.yaml missing openai"
grep -q 'sk-test-key-12345' "$CONFIG" || fail "config.yaml missing the key"
BIN="$FRESH/.local/bin/lazy-take-notes"
NOTE="$FRESH/.local/bin/take-note"
[[ "$(readlink "$BIN")" == "$FRESH/.local/share/lazy-take-notes/$TAG/lazy-take-notes" ]] \
  || fail "lazy-take-notes link target is wrong"
[[ "$(readlink "$NOTE")" == "$FRESH/.local/share/lazy-take-notes/$TAG/lazy-take-notes" ]] \
  || fail "take-note link target is wrong"
"$NOTE" --version | grep -q "${TAG#v}" || fail "take-note --version does not print $TAG"
pass "clean install places both links and runs $TAG"

# A release from 0.6.0 on carries the window package. setup.sh installs it when
# its .sha256 exists in the served folder.
if [[ -f "/artifacts/lazy-take-notes-gui-$ARCH-linux.tar.gz" ]]; then
  [[ -L "$FRESH/.local/bin/lazy-take-notes-gui" ]] || fail "window link not created"
  [[ -f "$FRESH/.local/share/applications/lazy-take-notes.desktop" ]] || fail "desktop file not installed"
  [[ -f "$FRESH/.local/share/icons/hicolor/512x512/apps/lazy-take-notes.png" ]] || fail "icon not installed"
  pass "window package installed"
fi

# ═════════════════════════════════════════════════════════════════════════════
# Test 2: second run changes nothing
# ═════════════════════════════════════════════════════════════════════════════
BEFORE="$(readlink "$BIN")|$(readlink "$NOTE")"
SECOND="$(run_setup "$FRESH")"
echo "$SECOND" | grep -q "already installed" || fail "second run did not say already installed"
AFTER="$(readlink "$BIN")|$(readlink "$NOTE")"
[[ "$BEFORE" == "$AFTER" ]] || fail "second run changed the links"
pass "second run says already installed and changes nothing"

# ═════════════════════════════════════════════════════════════════════════════
# Test 3: an old uvx wrapper is replaced
# ═════════════════════════════════════════════════════════════════════════════
OLD_HOME="$(mktemp -d /home/testuser/old.XXXXXX)"
mkdir -p "$OLD_HOME/.local/bin"
cat >"$OLD_HOME/.local/bin/take-note" <<'WRAPPER'
#!/bin/bash
exec uvx --from git+https://github.com/CJHwong/lazy-take-notes.git lazy-take-notes "$@"
WRAPPER
chmod +x "$OLD_HOME/.local/bin/take-note"
THIRD="$(run_setup "$OLD_HOME")"
echo "$THIRD" | grep -q "Removed the old wrapper" || fail "old wrapper was not removed"
[[ "$(readlink "$OLD_HOME/.local/bin/take-note")" == "$OLD_HOME/.local/share/lazy-take-notes/$TAG/lazy-take-notes" ]] \
  || fail "take-note is not the new link"
pass "old wrapper removed and replaced with the release link"

# ═════════════════════════════════════════════════════════════════════════════
# Test 4: record through the installed command
# ═════════════════════════════════════════════════════════════════════════════
export XDG_RUNTIME_DIR
XDG_RUNTIME_DIR="$(mktemp -d)"
chmod 700 "$XDG_RUNTIME_DIR"
pulseaudio --start --exit-idle-time=-1
pactl load-module module-null-sink sink_name=test_loopback
pactl set-default-sink test_loopback
pactl set-sink-mute test_loopback 0
pactl set-sink-volume test_loopback 100%

RECORD_CONFIG="$FRESH/record.yaml"
cat >"$RECORD_CONFIG" <<'YAML'
transcription:
  model: /models/ggml-base.en.bin
llm_provider: ollama
ollama:
  host: http://127.0.0.1:9
output:
  save_debug_log: true
YAML
chmod 600 "$RECORD_CONFIG"

OUT="$(mktemp -d /tmp/ltn-setup-smoke.XXXXXX)"
# The record runs under the user's HOME, as a real install does. The install
# tests above keep their own fresh HOME.
env -u XDG_CONFIG_HOME HOME="/home/testuser" \
  "$NOTE" record --headless --mute-mic --config "$RECORD_CONFIG" \
  --template default_en --output-dir "$OUT" >"$OUT/record.log" 2>&1 &
record_pid=$!
for _ in $(seq 1 60); do
  if grep -q 'opening system monitor' "$OUT"/*/debug.log 2>/dev/null; then
    break
  fi
  if ! kill -0 "$record_pid" 2>/dev/null; then
    break
  fi
  sleep 1
done
sleep 1
paplay /src/speech.wav
sleep 5
kill -TERM "$record_pid" || true
code=0
wait "$record_pid" || code=$?
[[ "$code" -eq 0 ]] || fail "record exit code $code"
if ! python3 /src/words_check.py /src/speech.txt "$OUT"/*/transcript.txt; then
  echo "--- transcript" >&2
  cat "$OUT"/*/transcript.txt >&2 || true
  echo "--- record.log" >&2
  cat "$OUT/record.log" >&2 || true
  echo "--- debug.log" >&2
  tail -40 "$OUT"/*/debug.log >&2 || true
  fail "the transcript missed the spoken words"
fi
if grep -E 'ERROR.*system audio' "$OUT"/*/debug.log; then
  fail "the debug log has a system audio error"
fi
pass "record through the installed take-note transcribes the sentence"

echo ""
echo -e "${GREEN}${BOLD}All setup smoke tests passed!${RESET}"
echo ""

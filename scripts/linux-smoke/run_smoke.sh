#!/usr/bin/env bash
# Record the null sink's monitor while a LibriVox sentence plays, then check
# the transcript. The image builds and installs the binary; PulseAudio refuses
# to run as root, so the image runs this as testuser.

set -euo pipefail

# The pure Rust PulseAudio client finds the server only through these, and a
# container user has none of them.
export XDG_RUNTIME_DIR
XDG_RUNTIME_DIR="$(mktemp -d)"
chmod 700 "$XDG_RUNTIME_DIR"
pulseaudio --start --exit-idle-time=-1
pactl load-module module-null-sink sink_name=test_loopback
pactl set-default-sink test_loopback

CONFIG="$HOME/ltn.yaml"
cat >"$CONFIG" <<'YAML'
transcription:
  model: /models/ggml-base.en.bin
llm_provider: ollama
ollama:
  host: http://127.0.0.1:9
output:
  save_debug_log: true
YAML
chmod 600 "$CONFIG"

OUT="$(mktemp -d /tmp/ltn-smoke-out.XXXXXX)"
LOG="$OUT/record.log"
lazy-take-notes record --headless --mute-mic --config "$CONFIG" --template default_en --output-dir "$OUT" >"$LOG" 2>&1 &
record_pid=$!
# Play only once the session's monitor stream is open: the "recording" line
# comes before it, and the first words would be lost.
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
paplay /src/scripts/linux-smoke/speech.wav
sleep 5
kill -TERM "$record_pid" || true
code=0
wait "$record_pid" || code=$?

# The first run on Docker Desktop (2026-11-27) ended with this code. Change it
# only with the run that showed the new one.
EXPECTED_EXIT=0

echo "record exit code: $code"
if [ "$code" -ne "$EXPECTED_EXIT" ]; then
    echo "FAIL: expected exit $EXPECTED_EXIT" >&2
    exit 1
fi
python3 /src/scripts/linux-smoke/words_check.py \
    /src/scripts/linux-smoke/speech.txt "$OUT"/*/transcript.txt
if grep -E 'ERROR.*system audio' "$OUT"/*/debug.log; then
    echo 'FAIL: the debug log has a system audio error' >&2
    exit 1
fi
echo 'PASS'

#!/bin/bash
# Setup smoke test — validates setup.sh in a clean environment.
# Runs both provider paths (ollama, openai) non-interactively.
set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
BOLD='\033[1m'
RESET='\033[0m'

pass() { echo -e "${GREEN}${BOLD}PASS${RESET}: $1"; }
fail() { echo -e "${RED}${BOLD}FAIL${RESET}: $1"; exit 1; }

BREW_BIN="$(brew --prefix)/bin"
TAKE_NOTE_BIN="$BREW_BIN/take-note"
CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/lazy-take-notes"
CONFIG_FILE="$CONFIG_DIR/config.yaml"

cleanup() {
  rm -f "$TAKE_NOTE_BIN"
  rm -rf "$CONFIG_DIR"
}

# ═════════════════════════════════════════════════════════════════════════════
# Test 1: OpenAI provider path
# ═════════════════════════════════════════════════════════════════════════════

echo ""
echo -e "${BOLD}═══ Test 1: OpenAI provider ═══${RESET}"
cleanup

LTN_PROVIDER=openai LTN_OPENAI_KEY=sk-test-key-12345 LTN_ACCEPT_BREW=1 bash /home/testuser/setup.sh

# Verify config.yaml was created with openai settings
if [[ ! -f "$CONFIG_FILE" ]]; then
  fail "config.yaml not created"
fi
if ! grep -q 'llm_provider: "openai"' "$CONFIG_FILE"; then
  fail "config.yaml missing openai provider"
fi
if ! grep -q 'gpt-5.4-nano' "$CONFIG_FILE"; then
  fail "config.yaml missing gpt-5.4-nano model"
fi
if ! grep -q 'sk-test-key-12345' "$CONFIG_FILE"; then
  fail "config.yaml missing API key"
fi
pass "OpenAI config.yaml correct"

# Verify take-note command was created and works
if [[ ! -x "$TAKE_NOTE_BIN" ]]; then
  fail "take-note command not created"
fi
pass "take-note command exists and is executable"

# ═════════════════════════════════════════════════════════════════════════════
# Test 2: Ollama provider path (skip signin)
# ═════════════════════════════════════════════════════════════════════════════

echo ""
echo -e "${BOLD}═══ Test 2: Ollama provider ═══${RESET}"
cleanup

LTN_PROVIDER=ollama LTN_SKIP_SIGNIN=1 LTN_ACCEPT_BREW=1 bash /home/testuser/setup.sh

# Verify config.yaml was created with ollama settings
if [[ ! -f "$CONFIG_FILE" ]]; then
  fail "config.yaml not created"
fi
if ! grep -q 'llm_provider: "ollama"' "$CONFIG_FILE"; then
  fail "config.yaml missing ollama provider"
fi
if ! grep -q 'gpt-oss:120b-cloud' "$CONFIG_FILE"; then
  fail "config.yaml missing ollama model"
fi
pass "Ollama config.yaml correct"

# Verify take-note command exists
if [[ ! -x "$TAKE_NOTE_BIN" ]]; then
  fail "take-note command not created"
fi
pass "take-note command exists and is executable"

# ═════════════════════════════════════════════════════════════════════════════
# Test 3: Idempotency — running setup again doesn't break anything
# ═════════════════════════════════════════════════════════════════════════════

echo ""
echo -e "${BOLD}═══ Test 3: Idempotency ═══${RESET}"

LTN_PROVIDER=ollama LTN_SKIP_SIGNIN=1 LTN_ACCEPT_BREW=1 bash /home/testuser/setup.sh

# config.yaml should still exist and be unchanged
if ! grep -q 'llm_provider: "ollama"' "$CONFIG_FILE"; then
  fail "config.yaml corrupted on re-run"
fi
pass "Re-run is idempotent"

# ═════════════════════════════════════════════════════════════════════════════

echo ""
echo -e "${GREEN}${BOLD}All setup smoke tests passed!${RESET}"
echo ""

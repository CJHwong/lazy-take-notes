#!/bin/bash
set -e

# ── Colors & helpers ─────────────────────────────────────────────────────────

GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
DIM='\033[2m'
RESET='\033[0m'

ok()   { echo -e "  ${GREEN}✓${RESET} $1"; }
fail() { echo -e "  ${RED}✗${RESET} $1"; }
info() { echo -e "  ${BLUE}→${RESET} $1"; }
warn() { echo -e "  ${YELLOW}⚠${RESET}  $1"; }
section() { echo -e "\n${BOLD}$1${RESET}"; }

echo -e "${BOLD}"
echo "  ╔════════════════════════════╗"
echo "  ║    lazy-take-notes setup   ║"
echo "  ╚════════════════════════════╝"
echo -e "${RESET}"

# ── Non-interactive mode (for CI / Docker) ───────────────────────────────────
# Set LTN_PROVIDER=ollama or LTN_PROVIDER=openai to skip prompts.
# For openai, also set LTN_OPENAI_KEY=sk-...
# Set LTN_SKIP_SIGNIN=1 to skip the ollama signin step.

PROVIDER="${LTN_PROVIDER:-}"

# ── Platform detection ───────────────────────────────────────────────────────

if [[ "$(uname)" == "Darwin" ]]; then
  PLATFORM="macos"
  CONFIG_DIR="$HOME/Library/Application Support/lazy-take-notes"
else
  PLATFORM="linux"
  CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/lazy-take-notes"
fi

# ═════════════════════════════════════════════════════════════════════════════
# Step 1: Homebrew
# ═════════════════════════════════════════════════════════════════════════════

section "1 / 6  Homebrew"
if command -v brew &>/dev/null; then
  ok "Already installed"
elif [[ "$PLATFORM" == "linux" ]] && [[ "${LTN_ACCEPT_BREW:-}" != "1" ]]; then
  # Homebrew on Linux is not standard — warn before touching the system
  echo ""
  warn "Homebrew is not installed."
  echo -e "  This script uses Homebrew to install dependencies."
  echo -e "  On Linux, Homebrew (linuxbrew) will:"
  echo -e "    • Create ${BOLD}/home/linuxbrew/.linuxbrew${RESET} (~1 GB)"
  echo -e "    • Add entries to your shell profile"
  echo ""
  echo -e "  ${DIM}If you prefer your distro package manager, install these manually:${RESET}"
  echo -e "    ${BOLD}uv${RESET}      — https://docs.astral.sh/uv/getting-started/installation/"
  echo -e "    ${BOLD}ollama${RESET}   — https://ollama.com/download (only if using Ollama provider)"
  echo -e "  ${DIM}Then re-run this script — it will skip what's already installed.${RESET}"
  echo ""
  read -rp "  Install Homebrew on this system? (y/N): " brew_confirm
  if [[ "$brew_confirm" != [yY]* ]]; then
    info "Skipping Homebrew — checking for dependencies directly..."
    # Fall through to step 2; uv/ollama checks will use whatever is on PATH
    HAS_BREW=0
  fi
fi

if [[ "${HAS_BREW:-1}" == "1" ]] && ! command -v brew &>/dev/null; then
  info "Installing Homebrew..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  # Ensure brew is on PATH for the rest of this script
  if [[ "$PLATFORM" == "linux" ]] && [[ -f /home/linuxbrew/.linuxbrew/bin/brew ]]; then
    eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"
  fi
fi

# ═════════════════════════════════════════════════════════════════════════════
# Step 2: uv (package manager)
# ═════════════════════════════════════════════════════════════════════════════

section "2 / 6  uv (package manager)"
if command -v uv &>/dev/null; then
  ok "Already installed"
elif command -v brew &>/dev/null; then
  info "Installing uv via Homebrew..."
  brew install uv
else
  # No brew available (Linux user declined) — try the standalone installer
  info "Installing uv via standalone installer..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  if ! command -v uv &>/dev/null; then
    fail "Could not install uv. Install it manually: https://docs.astral.sh/uv/"
    exit 1
  fi
  ok "Installed uv"
fi

# ═════════════════════════════════════════════════════════════════════════════
# Step 3: Choose AI provider
# ═════════════════════════════════════════════════════════════════════════════

section "3 / 6  AI provider"

if [[ -z "$PROVIDER" ]]; then
  echo ""
  echo -e "  Which AI provider do you want to use?"
  echo ""
  echo -e "  ${BOLD}1${RESET}  Ollama  ${DIM}— runs on your computer (free, private, needs ~4 GB RAM)${RESET}"
  echo -e "  ${BOLD}2${RESET}  OpenAI  ${DIM}— cloud API (needs an API key from platform.openai.com)${RESET}"
  echo ""
  while true; do
    read -rp "  Enter 1 or 2: " choice
    case "$choice" in
      1) PROVIDER="ollama"; break ;;
      2) PROVIDER="openai"; break ;;
      *) echo -e "  ${YELLOW}Please enter 1 or 2${RESET}" ;;
    esac
  done
fi

ok "Selected: $PROVIDER"

# ═════════════════════════════════════════════════════════════════════════════
# Step 4: Provider-specific setup
# ═════════════════════════════════════════════════════════════════════════════

section "4 / 6  Provider setup ($PROVIDER)"

if [[ "$PROVIDER" == "ollama" ]]; then
  # ── Ollama path ──────────────────────────────────────────────────────────
  if command -v ollama &>/dev/null; then
    ok "Ollama already installed"
  elif command -v brew &>/dev/null; then
    info "Installing Ollama via Homebrew..."
    brew install ollama
  else
    warn "Ollama is not installed and Homebrew is not available."
    echo -e "  ${DIM}Install Ollama manually: https://ollama.com/download${RESET}"
    echo -e "  ${DIM}Then re-run this script.${RESET}"
    exit 1
  fi

elif [[ "$PROVIDER" == "openai" ]]; then
  # ── OpenAI path ──────────────────────────────────────────────────────────
  OPENAI_KEY="${LTN_OPENAI_KEY:-}"
  if [[ -z "$OPENAI_KEY" ]]; then
    echo ""
    echo -e "  Enter your OpenAI API key (starts with ${BOLD}sk-${RESET})."
    echo -e "  ${DIM}Get one at: https://platform.openai.com/api-keys${RESET}"
    echo ""
    read -rsp "  API key: " OPENAI_KEY
    echo ""
    if [[ -z "$OPENAI_KEY" ]]; then
      warn "No API key provided. You can add it later in Settings."
    fi
  fi
  ok "OpenAI provider configured"
else
  fail "Unknown provider: $PROVIDER"
  exit 1
fi

# ═════════════════════════════════════════════════════════════════════════════
# Step 5: take-note command
# ═════════════════════════════════════════════════════════════════════════════

section "5 / 6  take-note command"

BREW_BIN="$(brew --prefix)/bin"
TAKE_NOTE_BIN="$BREW_BIN/take-note"

# Clean up stale alias from previous setup versions
SHELL_RC="$HOME/.zshrc"
if grep -q "alias take-note=" "$SHELL_RC" 2>/dev/null; then
  sed -i '' '/# lazy-take-notes/d;/alias take-note=/d' "$SHELL_RC" 2>/dev/null || true
  info "Removed old alias from $SHELL_RC"
fi

UVX_PATH="$(command -v uvx 2>/dev/null)"
if [[ -z "$UVX_PATH" ]]; then
  warn "uvx not found in PATH — install uv first"
  exit 1
fi

if [[ -f "$TAKE_NOTE_BIN" ]]; then
  ok "Already exists at $TAKE_NOTE_BIN"
else
  if ! touch "$TAKE_NOTE_BIN" 2>/dev/null; then
    warn "Cannot write to $BREW_BIN — re-run with: sudo bash setup.sh"
    exit 1
  fi
  cat > "$TAKE_NOTE_BIN" << WRAPPER
#!/bin/bash
exec "$UVX_PATH" --from git+https://github.com/CJHwong/lazy-take-notes.git lazy-take-notes "\$@"
WRAPPER
  chmod +x "$TAKE_NOTE_BIN"
  ok "Created 'take-note' command at $TAKE_NOTE_BIN"
fi

# ═════════════════════════════════════════════════════════════════════════════
# Step 6: config.yaml
# ═════════════════════════════════════════════════════════════════════════════

section "6 / 6  Config"

CONFIG_FILE="$CONFIG_DIR/config.yaml"
mkdir -p "$CONFIG_DIR"

if [[ -f "$CONFIG_FILE" ]]; then
  ok "config.yaml already exists, skipping"
else
  if [[ "$PROVIDER" == "openai" ]]; then
    cat > "$CONFIG_FILE" <<YAMLEOF
llm_provider: "openai"
openai:
  base_url: "https://api.openai.com/v1"
  api_key: "${OPENAI_KEY:-}"
digest:
  model: "gpt-5.4-nano"
interactive:
  model: "gpt-5.4-nano"
YAMLEOF
  else
    cat > "$CONFIG_FILE" <<YAMLEOF
llm_provider: "ollama"
digest:
  model: "gpt-oss:120b-cloud"
interactive:
  model: "gpt-oss:20b-cloud"
YAMLEOF
  fi
  ok "config.yaml created at $CONFIG_FILE"
fi

# ── Ollama sign-in (only for ollama provider) ────────────────────────────────

if [[ "$PROVIDER" == "ollama" ]] && [[ "${LTN_SKIP_SIGNIN:-}" != "1" ]]; then
  echo -e "\n${YELLOW}${BOLD}Almost done!${RESET}"
  echo -e "Sign in to Ollama to enable cloud models:\n"
  ollama signin
fi

# ═════════════════════════════════════════════════════════════════════════════
# Validation: verify the installed command actually works
# ═════════════════════════════════════════════════════════════════════════════

echo ""
section "Validating installation..."

if "$TAKE_NOTE_BIN" --version &>/dev/null; then
  VERSION="$("$TAKE_NOTE_BIN" --version 2>&1)"
  ok "take-note works! ($VERSION)"
else
  fail "take-note command failed — check the output above for errors"
  echo -e "  ${DIM}Try running manually: $TAKE_NOTE_BIN --version${RESET}"
  exit 1
fi

echo -e "\n${GREEN}${BOLD}Setup complete!${RESET}"
echo -e "Run:\n"
echo -e "  ${BOLD}take-note${RESET}          ${DIM}— interactive mode selector${RESET}"
echo -e "  ${BOLD}take-note record${RESET}   ${DIM}— start a live recording session${RESET}"
echo -e "  ${BOLD}take-note config${RESET}   ${DIM}— change settings anytime${RESET}"
echo ""

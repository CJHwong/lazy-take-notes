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
# Step 1: uv (package manager)
# ═════════════════════════════════════════════════════════════════════════════

section "1 / 5  uv (package manager)"
if command -v uv &>/dev/null; then
  ok "Already installed"
else
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
# Step 2: Choose AI provider
# ═════════════════════════════════════════════════════════════════════════════

section "2 / 5  AI provider"

if [[ -z "$PROVIDER" ]]; then
  echo ""
  echo -e "  Which AI provider do you want to use?"
  echo ""
  echo -e "  ${BOLD}1${RESET}  Ollama  ${DIM}— runs on your computer (free, private, needs ~4 GB RAM)${RESET}"
  echo -e "  ${BOLD}2${RESET}  OpenAI  ${DIM}— cloud API (needs an API key from platform.openai.com)${RESET}"
  echo ""
  while true; do
    read -rp "  Enter 1 or 2: " choice </dev/tty
    case "$choice" in
      1) PROVIDER="ollama"; break ;;
      2) PROVIDER="openai"; break ;;
      *) echo -e "  ${YELLOW}Please enter 1 or 2${RESET}" ;;
    esac
  done
fi

ok "Selected: $PROVIDER"

# ═════════════════════════════════════════════════════════════════════════════
# Step 3: Provider-specific setup
# ═════════════════════════════════════════════════════════════════════════════

section "3 / 5  Provider setup ($PROVIDER)"

if [[ "$PROVIDER" == "ollama" ]]; then
  # ── Ollama path ──────────────────────────────────────────────────────────
  if command -v ollama &>/dev/null; then
    ok "Ollama already installed"
  elif command -v brew &>/dev/null; then
    info "Installing Ollama via Homebrew..."
    brew install ollama
  else
    # No brew available — on Linux, offer to install brew for ollama
    if [[ "$PLATFORM" == "linux" ]] && [[ "${LTN_ACCEPT_BREW:-}" != "1" ]]; then
      echo ""
      warn "Ollama is not installed and Homebrew is not available."
      echo -e "  Homebrew is needed to install Ollama."
      echo -e "  On Linux, Homebrew (linuxbrew) will:"
      echo -e "    • Create ${BOLD}/home/linuxbrew/.linuxbrew${RESET} (~1 GB)"
      echo -e "    • Add entries to your shell profile"
      echo ""
      echo -e "  ${DIM}Alternatively, install Ollama manually: https://ollama.com/download${RESET}"
      echo -e "  ${DIM}Then re-run this script.${RESET}"
      echo ""
      read -rp "  Install Homebrew to get Ollama? (y/N): " brew_confirm </dev/tty
      if [[ "$brew_confirm" != [yY]* ]]; then
        warn "Skipping. Install Ollama manually: https://ollama.com/download"
        echo -e "  ${DIM}Then re-run this script.${RESET}"
        exit 1
      fi
    fi
    # Install brew, then ollama
    info "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    if [[ "$PLATFORM" == "linux" ]] && [[ -f /home/linuxbrew/.linuxbrew/bin/brew ]]; then
      eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"
    fi
    info "Installing Ollama via Homebrew..."
    brew install ollama
  fi

elif [[ "$PROVIDER" == "openai" ]]; then
  # ── OpenAI path ──────────────────────────────────────────────────────────
  OPENAI_KEY="${LTN_OPENAI_KEY:-}"
  if [[ -z "$OPENAI_KEY" ]]; then
    echo ""
    echo -e "  Enter your OpenAI API key (starts with ${BOLD}sk-${RESET})."
    echo -e "  ${DIM}Get one at: https://platform.openai.com/api-keys${RESET}"
    echo ""
    read -rsp "  API key: " OPENAI_KEY </dev/tty
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
# Step 4: take-note command
# ═════════════════════════════════════════════════════════════════════════════

section "4 / 5  take-note command"

# ── Determine install directory ──────────────────────────────────────────────
if [[ "$PLATFORM" == "macos" ]]; then
  INSTALL_DIR="/usr/local/bin"
  if ! touch "$INSTALL_DIR/.ltn-write-test" 2>/dev/null; then
    INSTALL_DIR="$HOME/.local/bin"
  else
    rm -f "$INSTALL_DIR/.ltn-write-test"
  fi
else
  INSTALL_DIR="$HOME/.local/bin"
fi

mkdir -p "$INSTALL_DIR"
TAKE_NOTE_BIN="$INSTALL_DIR/take-note"

# ── Migrate: remove old brew-based wrapper if it exists ──────────────────────
if command -v brew &>/dev/null; then
  OLD_BREW_BIN="$(brew --prefix)/bin/take-note"
  if [[ -f "$OLD_BREW_BIN" ]] && [[ "$OLD_BREW_BIN" != "$TAKE_NOTE_BIN" ]]; then
    rm -f "$OLD_BREW_BIN"
    info "Removed old wrapper at $OLD_BREW_BIN"
  fi
fi

# ── Clean up stale alias from previous setup versions ────────────────────────
SHELL_RC="$HOME/.zshrc"
if grep -q "alias take-note=" "$SHELL_RC" 2>/dev/null; then
  sed -i '' '/# lazy-take-notes/d;/alias take-note=/d' "$SHELL_RC" 2>/dev/null || true
  info "Removed old alias from $SHELL_RC"
fi

# ── Ensure ~/.local/bin is on PATH if we're installing there ─────────────────
if [[ "$INSTALL_DIR" == "$HOME/.local/bin" ]]; then
  case ":$PATH:" in
    *":$HOME/.local/bin:"*) ;;
    *)
      # Detect shell rc file
      if [[ -n "${ZSH_VERSION:-}" ]] || [[ "$(basename "$SHELL")" == "zsh" ]]; then
        RC_FILE="$HOME/.zshrc"
      else
        RC_FILE="$HOME/.bashrc"
      fi
      EXPORT_LINE='export PATH="$HOME/.local/bin:$PATH"'
      if ! grep -qF '.local/bin' "$RC_FILE" 2>/dev/null; then
        echo "" >> "$RC_FILE"
        echo "# Added by lazy-take-notes setup" >> "$RC_FILE"
        echo "$EXPORT_LINE" >> "$RC_FILE"
        info "Added ~/.local/bin to PATH in $RC_FILE"
      fi
      export PATH="$HOME/.local/bin:$PATH"
      warn "Restart your shell or run: source $RC_FILE"
      ;;
  esac
fi

# ── Create wrapper script ───────────────────────────────────────────────────
UVX_PATH="$(command -v uvx 2>/dev/null)"
if [[ -z "$UVX_PATH" ]]; then
  warn "uvx not found in PATH — install uv first"
  exit 1
fi

cat > "$TAKE_NOTE_BIN" << WRAPPER
#!/bin/bash
# Auto-generated by lazy-take-notes setup — plugins managed via 'take-note plugin add/remove'
ARGS=()
PLUGIN_FILE="$CONFIG_DIR/plugins.txt"
if [[ -f "\$PLUGIN_FILE" ]]; then
  while IFS= read -r line; do
    [[ -n "\$line" ]] && ARGS+=(--with "\$line")
  done < "\$PLUGIN_FILE"
fi
exec "$UVX_PATH" --from git+https://github.com/CJHwong/lazy-take-notes.git "\${ARGS[@]}" lazy-take-notes "\$@"
WRAPPER
chmod +x "$TAKE_NOTE_BIN"
ok "Created 'take-note' command at $TAKE_NOTE_BIN"

# ═════════════════════════════════════════════════════════════════════════════
# Step 5: config.yaml
# ═════════════════════════════════════════════════════════════════════════════

section "5 / 5  Config"

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

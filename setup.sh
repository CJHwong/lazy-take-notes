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

# ── Risk prompt ──────────────────────────────────────────────────────────────
# A piped install runs whatever the server sends, with your permissions, so ask
# first. The answer comes from /dev/tty because stdin is the script itself.
# Pass -y to skip it (for agents and CI).

ASSUME_YES=0
for arg in "$@"; do
  case "$arg" in
    -y|--yes) ASSUME_YES=1 ;;
  esac
done

confirm_install() {
  cat >&2 <<'EOF'
WARNING: this installer downloads code from the internet and runs it as you.
It can read, change, or delete anything your user account can.
The server can send different code each time, so read the script first:
  https://github.com/CJHwong/lazy-take-notes/blob/main/setup.sh
Pass -y to skip this question (for agents and CI).
EOF
  if ! (: </dev/tty) 2>/dev/null; then
    echo "setup.sh: no terminal to ask on; re-run with -y to accept the risk" >&2
    exit 1
  fi
  printf 'Proceed? [Y/n] ' >&2
  local answer=""
  read -r answer </dev/tty || answer=""
  case "$answer" in
    n|N|no|No|NO) echo "setup.sh: aborted" >&2; exit 1 ;;
  esac
}
[[ "$ASSUME_YES" == 1 ]] || confirm_install

# ── Non-interactive mode (for CI / Docker) ───────────────────────────────────
# Pass -y (above), then set LTN_PROVIDER=ollama or LTN_PROVIDER=openai to skip prompts.
# For openai, also set LTN_OPENAI_KEY=sk-...
# Set LTN_SKIP_SIGNIN=1 to skip the ollama signin step.
# LTN_VERSION=vX.Y.Z selects a release; LTN_RELEASE_URL serves a local folder.

PROVIDER="${LTN_PROVIDER:-}"

# ── Platform detection ───────────────────────────────────────────────────────

case "$(uname -s)" in
  Darwin)
    PLATFORM="macos"
    CONFIG_DIR="$HOME/Library/Application Support/lazy-take-notes"
    ;;
  Linux)
    PLATFORM="linux"
    CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/lazy-take-notes"
    ;;
  *)
    fail "Unsupported system: $(uname -s). Supported: macOS and Linux."
    exit 1
    ;;
esac

case "$(uname -m)" in
  x86_64) ARCH="x86_64" ;;
  arm64 | aarch64) ARCH="arm64" ;;
  *)
    fail "Unsupported machine: $(uname -m). Supported: x86_64 and arm64."
    exit 1
    ;;
esac

ASSET="lazy-take-notes-$ARCH-$PLATFORM.tar.gz"
BIN_ROOT=""
URL_BASE=""
TMP_DIR=""

# Download a release file and its .sha256, then check the hash.
fetch() {
  local name="$1" expected actual
  curl -fL "$URL_BASE/$name" -o "$TMP_DIR/$name"
  curl -fL "$URL_BASE/$name.sha256" -o "$TMP_DIR/$name.sha256"
  expected="$(cut -d' ' -f1 "$TMP_DIR/$name.sha256")"
  if command -v shasum &>/dev/null; then
    actual="$(shasum -a 256 "$TMP_DIR/$name" | cut -d' ' -f1)"
  else
    actual="$(sha256sum "$TMP_DIR/$name" | cut -d' ' -f1)"
  fi
  if [[ "$expected" != "$actual" ]]; then
    fail "SHA-256 mismatch for $name"
    exit 1
  fi
}

# ═════════════════════════════════════════════════════════════════════════════
# Step 1: Terminal app
# ═════════════════════════════════════════════════════════════════════════════

section "1 / 5  Terminal app"

# Homebrew is the recommended way on a Mac. With no terminal, use the tarball.
if [[ "$PLATFORM" == macos ]] && command -v brew &>/dev/null; then
  info "The recommended install on macOS is:"
  echo -e "    ${BOLD}brew install --cask CJHwong/tap/lazy-take-notes${RESET}"
  ANSWER=""
  if (: </dev/tty) 2>/dev/null; then
    printf '  Install the terminal app from the tarball instead? (y/N) ' >&2
    read -r ANSWER </dev/tty || ANSWER=""
  fi
  case "$ANSWER" in
    n|N|no|No|NO)
      echo ""
      info "Run: brew install --cask CJHwong/tap/lazy-take-notes"
      exit 0
      ;;
  esac
fi

TAG="${LTN_VERSION:-}"
if [[ -z "$TAG" ]]; then
  TAG="$(curl -fsSL https://api.github.com/repos/CJHwong/lazy-take-notes/releases/latest 2>/dev/null \
    | sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p' | head -n 1)" || TAG=""
fi
if [[ -z "$TAG" ]]; then
  fail "Cannot read the newest release. Set LTN_VERSION=vX.Y.Z and run again."
  exit 1
fi

URL_BASE="${LTN_RELEASE_URL:-https://github.com/CJHwong/lazy-take-notes/releases/download}/$TAG"
BIN_ROOT="$HOME/.local/share/lazy-take-notes/$TAG"
BIN_LINK="$HOME/.local/bin/lazy-take-notes"
NOTE_LINK="$HOME/.local/bin/take-note"

if [[ -d "$BIN_ROOT" ]] && [[ -x "$BIN_ROOT/lazy-take-notes" ]] \
  && [[ "$(readlink "$BIN_LINK" 2>/dev/null)" == "$BIN_ROOT/lazy-take-notes" ]] \
  && [[ "$(readlink "$NOTE_LINK" 2>/dev/null)" == "$BIN_ROOT/lazy-take-notes" ]]; then
  ok "lazy-take-notes $TAG is already installed"
  exit 0
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

info "Downloading lazy-take-notes $TAG ($ASSET)"
fetch "$ASSET"
tar -xzf "$TMP_DIR/$ASSET" -C "$TMP_DIR"
mkdir -p "$BIN_ROOT"
cp -R "$TMP_DIR/lazy-take-notes-$ARCH-$PLATFORM/." "$BIN_ROOT/"
chmod +x "$BIN_ROOT/lazy-take-notes"
if [[ -f "$BIN_ROOT/coreaudio-tap" ]]; then
  chmod +x "$BIN_ROOT/coreaudio-tap"
fi
ok "Installed $TAG in $BIN_ROOT"

# ═════════════════════════════════════════════════════════════════════════════
# Step 2: Choose AI provider
# ═════════════════════════════════════════════════════════════════════════════

section "2 / 5  AI provider"

if [[ -z "$PROVIDER" ]]; then
  echo ""
  echo -e "  Which AI provider do you want to use?"
  echo ""
  echo -e "  ${BOLD}1${RESET}  Ollama  ${DIM}cloud models on Ollama's servers (needs a free Ollama sign-in)${RESET}"
  echo -e "  ${BOLD}2${RESET}  OpenAI  ${DIM}OpenAI API (needs an API key from platform.openai.com)${RESET}"
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
# Step 4: Commands
# ═════════════════════════════════════════════════════════════════════════════

section "4 / 5  Commands"

mkdir -p "$HOME/.local/bin"

# An old wrapper is a regular file that runs the git checkout. The Rust binary
# also contains the text "uvx", so match the repository URL, not uvx.
OLD_WRAPPERS=("/usr/local/bin/take-note" "$HOME/.local/bin/take-note")
if command -v brew &>/dev/null; then
  OLD_WRAPPERS+=("$(brew --prefix)/bin/take-note")
fi
for wrapper in "${OLD_WRAPPERS[@]}"; do
  [[ -f "$wrapper" ]] || continue
  if [[ -L "$wrapper" ]]; then
    continue
  fi
  if ! grep -q "lazy-take-notes.git" "$wrapper" 2>/dev/null; then
    continue
  fi
  if rm -f "$wrapper" 2>/dev/null; then
    info "Removed the old wrapper at $wrapper"
  elif (: </dev/tty) 2>/dev/null; then
    printf '  Remove the old wrapper at %s with sudo? (y/N) ' "$wrapper" >&2
    read -r answer </dev/tty || answer=""
    case "$answer" in
      y|Y|yes|Yes|YES)
        sudo rm -f "$wrapper"
        info "Removed the old wrapper at $wrapper"
        ;;
    esac
  fi
done

ln -sfn "$BIN_ROOT/lazy-take-notes" "$BIN_LINK"
ln -sfn "$BIN_ROOT/lazy-take-notes" "$NOTE_LINK"
ok "Linked lazy-take-notes and take-note in $HOME/.local/bin"

# The window app ships from 0.6.0 on Linux. Its absence is not an error.
if [[ "$PLATFORM" == linux ]]; then
  GUI_ASSET="lazy-take-notes-gui-$ARCH-linux.tar.gz"
  if curl -fsIL "$URL_BASE/$GUI_ASSET.sha256" >/dev/null 2>&1; then
    info "Downloading the window app ($GUI_ASSET)"
    fetch "$GUI_ASSET"
    tar -xzf "$TMP_DIR/$GUI_ASSET" -C "$TMP_DIR"
    GUI_ROOT="$TMP_DIR/lazy-take-notes-gui-$ARCH-linux"
    mkdir -p "$BIN_ROOT/gui" "$HOME/.local/share/applications" \
      "$HOME/.local/share/icons/hicolor/512x512/apps"
    cp "$GUI_ROOT/lazy-take-notes-gui" "$BIN_ROOT/gui/"
    chmod +x "$BIN_ROOT/gui/lazy-take-notes-gui"
    ln -sfn "$BIN_ROOT/gui/lazy-take-notes-gui" "$HOME/.local/bin/lazy-take-notes-gui"
    if [[ -f "$GUI_ROOT/lazy-take-notes.desktop" ]]; then
      cp "$GUI_ROOT/lazy-take-notes.desktop" "$HOME/.local/share/applications/"
    fi
    if [[ -f "$GUI_ROOT/lazy-take-notes.png" ]]; then
      cp "$GUI_ROOT/lazy-take-notes.png" "$HOME/.local/share/icons/hicolor/512x512/apps/"
    fi
    ok "Installed the window app"
  fi
fi

case ":$PATH:" in
  *":$HOME/.local/bin:"*) ;;
  *)
    warn "Add this line to your shell profile for the new commands:"
    echo -e "    ${BOLD}export PATH=\"\$HOME/.local/bin:\$PATH\"${RESET}"
    ;;
esac

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
  echo -e "Sign in to Ollama to use its cloud models:\n"
  ollama signin
fi

# ═════════════════════════════════════════════════════════════════════════════
# Validation: verify the installed command actually works
# ═════════════════════════════════════════════════════════════════════════════

echo ""
section "Validating installation..."

if "$NOTE_LINK" --version &>/dev/null; then
  VERSION="$("$NOTE_LINK" --version 2>&1)"
  ok "take-note works! ($VERSION)"
else
  fail "take-note command failed — check the output above for errors"
  echo -e "  ${DIM}Try running manually: $NOTE_LINK --version${RESET}"
  exit 1
fi

echo -e "\n${GREEN}${BOLD}Setup complete!${RESET}"
echo -e "Run:\n"
echo -e "  ${BOLD}lazy-take-notes${RESET}          ${DIM}— interactive mode selector${RESET}"
echo -e "  ${BOLD}lazy-take-notes record${RESET}   ${DIM}— start a live recording session${RESET}"
echo -e "  ${BOLD}lazy-take-notes config${RESET}   ${DIM}— change settings anytime${RESET}"
echo ""

#!/bin/bash
set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
RESET='\033[0m'

ok()   { echo -e "  ${GREEN}✓${RESET} $1"; }
info() { echo -e "  ${BLUE}→${RESET} $1"; }
section() { echo -e "\n${BOLD}$1${RESET}"; }

echo -e "${BOLD}"
echo "  ╔════════════════════════════╗"
echo "  ║    lazy-take-notes setup   ║"
echo "  ╚════════════════════════════╝"
echo -e "${RESET}"

# 1. Homebrew
section "1 / 5  Homebrew"
if ! command -v brew &>/dev/null; then
  info "Installing Homebrew..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
  ok "Already installed"
fi

# 2. uv
section "2 / 5  uv (package manager)"
if ! command -v uv &>/dev/null; then
  info "Installing uv..."
  brew install uv
else
  ok "Already installed"
fi

# 3. Ollama
section "3 / 5  Ollama"
if ! command -v ollama &>/dev/null; then
  info "Installing Ollama..."
  brew install ollama
else
  ok "Already installed"
fi

# 4. alias
section "4 / 5  Shell alias"
SHELL_RC="$HOME/.zshrc"
ALIAS_LINE='alias take-note="uvx --from git+https://github.com/CJHwong/lazy-meeting-note.git lazy-take-notes"'

if grep -q "alias take-note=" "$SHELL_RC" 2>/dev/null; then
  ok "Already set in $SHELL_RC"
else
  echo "" >> "$SHELL_RC"
  echo "# lazy-take-notes" >> "$SHELL_RC"
  echo "$ALIAS_LINE" >> "$SHELL_RC"
  ok "Added 'take-note' alias to $SHELL_RC"
fi

# 5. config.yaml
section "5 / 5  Config (Ollama cloud models)"
CONFIG_DIR="$HOME/Library/Application Support/lazy-take-notes"
CONFIG_FILE="$CONFIG_DIR/config.yaml"
mkdir -p "$CONFIG_DIR"

if [ -f "$CONFIG_FILE" ]; then
  ok "config.yaml already exists, skipping"
else
  cat > "$CONFIG_FILE" <<EOF
digest:
  model: "gpt-oss:120b-cloud"
interactive:
  model: "gpt-oss:20b-cloud"
EOF
  ok "config.yaml created"
fi

# 6. Ollama sign in
echo -e "\n${YELLOW}${BOLD}Almost done!${RESET}"
echo -e "Sign in to Ollama to enable cloud models:\n"
ollama signin

echo -e "\n${GREEN}${BOLD}Setup complete!${RESET}"
echo -e "Open a new terminal window and run:\n"
echo -e "  ${BOLD}take-note record${RESET}\n"

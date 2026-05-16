#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# LLM OptiCoach — one-liner installer for NVIDIA Jetson & DGX Spark
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/Caltongroup/llm-opticoach/main/install.sh | bash
#
# What it does:
#   1. Checks Python 3.10+ is available
#   2. Checks (and optionally installs) Ollama
#   3. Clones the repo (or updates if already present)
#   4. Creates a Python virtual environment and installs dependencies
#   5. Installs a systemd user service so OptiCoach starts on boot
#   6. Starts the service and prints the URL
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_URL="https://github.com/Caltongroup/llm-opticoach.git"
INSTALL_DIR="$HOME/llm_opti_coach"
SERVICE_NAME="opticoach"
PORT=8080

# ── Colors ──────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${CYAN}[info]${NC}  $*"; }
ok()    { echo -e "${GREEN}[  ok]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[warn]${NC}  $*"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $*"; exit 1; }

# ── 1. Python ───────────────────────────────────────────────────────
info "Checking Python..."
PYTHON=""
for cmd in python3.12 python3.11 python3.10 python3; do
    if command -v "$cmd" &>/dev/null; then
        ver=$("$cmd" -c 'import sys; print(sys.version_info[:2])')
        major=$("$cmd" -c 'import sys; print(sys.version_info[0])')
        minor=$("$cmd" -c 'import sys; print(sys.version_info[1])')
        if [[ "$major" -ge 3 && "$minor" -ge 10 ]]; then
            PYTHON="$cmd"
            break
        fi
    fi
done
[[ -z "$PYTHON" ]] && fail "Python 3.10+ not found. Install it with: sudo apt install python3 python3-venv"
ok "Found $PYTHON ($($PYTHON --version))"

# ── 2. Ollama ───────────────────────────────────────────────────────
info "Checking Ollama..."
if command -v ollama &>/dev/null; then
    ok "Ollama is installed ($(ollama --version 2>/dev/null || echo 'version unknown'))"
else
    warn "Ollama not found."
    echo ""
    echo "  Ollama is required to serve AI models locally."
    echo "  Install it now with the official one-liner?"
    echo ""
    read -r -p "  Install Ollama? [Y/n] " answer
    answer="${answer:-Y}"
    if [[ "$answer" =~ ^[Yy] ]]; then
        info "Installing Ollama..."
        curl -fsSL https://ollama.com/install.sh | sh
        ok "Ollama installed"
    else
        warn "Skipping Ollama install. You'll need it before using OptiCoach."
    fi
fi

# ── 3. Clone / Update ──────────────────────────────────────────────
info "Setting up OptiCoach in $INSTALL_DIR ..."
if [[ -d "$INSTALL_DIR/.git" ]]; then
    info "Existing installation found — pulling latest..."
    git -C "$INSTALL_DIR" pull --ff-only
    ok "Updated to latest"
else
    if [[ -d "$INSTALL_DIR" ]]; then
        # Directory exists but isn't a git repo (e.g. manual copy)
        warn "$INSTALL_DIR exists but is not a git repo. Using it as-is."
    else
        git clone "$REPO_URL" "$INSTALL_DIR"
        ok "Cloned to $INSTALL_DIR"
    fi
fi

# ── 4. Virtual environment + deps ──────────────────────────────────
info "Setting up Python virtual environment..."
if [[ ! -d "$INSTALL_DIR/.venv" ]]; then
    "$PYTHON" -m venv "$INSTALL_DIR/.venv"
fi
source "$INSTALL_DIR/.venv/bin/activate"
pip install --quiet --upgrade pip
pip install --quiet -r "$INSTALL_DIR/requirements.txt"
ok "Dependencies installed"

# ── 5. systemd user service ────────────────────────────────────────
info "Installing systemd user service..."
mkdir -p "$HOME/.config/systemd/user"

cat > "$HOME/.config/systemd/user/${SERVICE_NAME}.service" <<EOF
[Unit]
Description=LLM OptiCoach — local AI setup wizard
After=network.target ollama.service

[Service]
Type=simple
WorkingDirectory=${INSTALL_DIR}
ExecStart=${INSTALL_DIR}/.venv/bin/uvicorn main:app --host 0.0.0.0 --port ${PORT}
Restart=on-failure
RestartSec=5
Environment=PATH=${INSTALL_DIR}/.venv/bin:/usr/local/bin:/usr/bin:/bin

[Install]
WantedBy=default.target
EOF

# Enable lingering so the user service starts at boot (not just at login)
if command -v loginctl &>/dev/null; then
    loginctl enable-linger "$(whoami)" 2>/dev/null || true
fi

systemctl --user daemon-reload
systemctl --user enable "${SERVICE_NAME}.service"
systemctl --user restart "${SERVICE_NAME}.service"
ok "Service installed and started"

# ── 6. Done! ────────────────────────────────────────────────────────
HOSTNAME_OR_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
[[ -z "$HOSTNAME_OR_IP" ]] && HOSTNAME_OR_IP="localhost"

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  LLM OptiCoach is running!${NC}"
echo ""
echo -e "  Open in your browser:"
echo -e "    ${CYAN}http://${HOSTNAME_OR_IP}:${PORT}${NC}"
echo -e "    ${CYAN}http://localhost:${PORT}${NC}"
echo ""
echo -e "  Useful commands:"
echo -e "    Status:   systemctl --user status ${SERVICE_NAME}"
echo -e "    Logs:     journalctl --user -u ${SERVICE_NAME} -f"
echo -e "    Restart:  systemctl --user restart ${SERVICE_NAME}"
echo -e "    Update:   cd ${INSTALL_DIR} && git pull && systemctl --user restart ${SERVICE_NAME}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

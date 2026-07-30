#!/usr/bin/env bash
# =============================================================================
# Project — Linux / macOS launcher
# Installs uv (if missing), downloads Python, syncs dependencies,
# then starts the application.
# Usage:  ./setup.sh [app arguments...]
# =============================================================================
set -euo pipefail

UV_PYTHON="3.10"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1. Ensure uv is available
if ! command -v uv &>/dev/null; then
    echo "[project] uv not found — installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # The installer adds ~/.local/bin; source env so PATH is updated
    export PATH="$HOME/.local/bin:$PATH"
    if ! command -v uv &>/dev/null; then
        echo "[project] ERROR: uv installation failed. Please install manually:"
        echo "              https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    fi
    echo "[project] uv installed: $(uv --version)"
else
    echo "[project] uv found: $(uv --version)"
fi

# 2. Ensure Python is available
cd "$SCRIPT_DIR"
echo "[project] Checking Python ${UV_PYTHON}..."
# uv python install "$UV_PYTHON" --quiet
uv python install "$UV_PYTHON"

# 3. Sync dependencies (create / update .venv)
echo "[project] Syncing dependencies..."
# uv sync --quiet
uv sync

# 4. Run the example
# echo "[project] Starting example..."
# exec uv run python ./examples/cross_framework.py "$@"

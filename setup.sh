#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Candor Setup ==="
echo

if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Install Python 3.11+ first."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
REQUIRED_VERSION="3.11"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "ERROR: Python 3.11+ required (found $PYTHON_VERSION)"
    exit 1
fi

VENV_DIR="venv"

if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at $VENV_DIR"
    read -rp "Recreate? (y/N): " recreate
    if [[ "$recreate" =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_DIR"
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

echo "Installing package and dependencies..."
pip install -q --upgrade pip
pip install -q -e ".[dev]"

echo
echo "=== Setup Complete ==="
echo
echo "Run tests:"
echo "  source venv/bin/activate"
echo "  pytest"
echo
echo "Optional vector search backend:"
echo "  pip install -e \".[vector]\""
echo

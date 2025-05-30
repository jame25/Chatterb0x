#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# --- Config ---
# Set UV_APP_DRY=1 externally to skip torch install
# Example: export UV_APP_DRY=1; ./run.sh
REQUIREMENTS_INPUT_FILE="requirements.in"
REQUIREMENTS_LOCK_FILE="requirements.lock.txt"


# --- CONFIG ---
VENV_DIR=".venv"
SCRIPT_NAME="main.py"
WINDOW_TITLE="Chatterb0x Clipboard Reader "

# Handle dry-run mode
UV_APP_DRY=${UV_APP_DRY:-0}
echo "Dry Run Mode: $UV_APP_DRY"

# --- Core Setup ---
echo
echo "=== Starting Chatterb0x Clipboard Reader Installer ==="
echo

# Check Python
if ! command -v python3.11 &> /dev/null; then
    echo "ERROR: Python 3.11 not found!"
    echo "Please install Python 3.11 or higher from: https://www.python.org/downloads/"
    echo "After installing Python, re-run this script."
    exit 1
fi

# Check UV
if ! command -v uv &> /dev/null; then
    echo "ERROR: uv is not installed."
    echo "Official install: https://github.com/astral-sh/uv#getting-started/installation"
    echo "Easiest way: pip3 install uv"
    echo "(Or: pip install uv, if your pip points to Python 3.11)"
    echo "If you still see 'uv not installed' after this, try closing and reopening your terminal window."
    read -p "Do you want to auto-install uv with pip now? (Y/N): " userchoice
    if [[ "$userchoice" =~ ^[Yy]$ ]]; then
        pip3 install uv || { echo "Failed to install uv! Please install it manually."; exit 1; }
        echo
        echo "Done installing uv. Restarting the script..."
        exec "$0"
    else
        echo "Please install uv manually and re-run this script."
        exit 1
    fi
fi

# Create venv
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating venv..."
    uv venv "$VENV_DIR" --python python3.11 || { echo "Venv creation failed!"; exit 1; }
fi

# Activate
source "$VENV_DIR/bin/activate"

# Ensure lock file is up to date
echo "Ensuring lock file is up to date..."
uv pip compile "$REQUIREMENTS_INPUT_FILE" -o "$REQUIREMENTS_LOCK_FILE" || { echo "Lock file generation failed!"; exit 1; }

# Install from lock file
echo "Installing dependencies from lock file..."
uv pip sync "$REQUIREMENTS_LOCK_FILE" || { echo "Dependency installation failed!"; exit 1; }

# Conditional PyTorch install
if [ "$UV_APP_DRY" = "0" ]; then
    echo "Installing PyTorch..."
    python install_torch.py || echo "WARNING: PyTorch install failed. App may lack GPU support."
else
    echo "[Dry Run] Skipped PyTorch installation"
fi

# Launch app
echo "Starting application..."
python "$SCRIPT_NAME"
#!/usr/bin/env bash
set -e
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ ! -f ./venv/bin/activate ]]; then
    python_bin="${PYTHON:-python3}"
    if ! command -v "$python_bin" >/dev/null 2>&1; then
        echo "Python not found: $python_bin. Install Python 3 or set PYTHON to its executable." >&2
        exit 1
    fi
    echo "Creating Python environment in ./venv..."
    if ! "$python_bin" -m venv ./venv; then
        echo "Could not create ./venv. On Ubuntu/WSL, install the matching venv package (usually: sudo apt install python3-venv), then run this launcher again." >&2
        exit 1
    fi
fi
source ./venv/bin/activate
exec python entry_with_update.py "$@"

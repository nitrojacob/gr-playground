#!/usr/bin/env bash
# RTL-SDR Spectrum Scanner Launcher Shell Script
# Location: ./scripts/scanner.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="/usr/lib/python3/dist-packages:${PROJECT_ROOT}:${PYTHONPATH}"

# Check for python3
if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: 'python3' command not found in PATH." >&2
    echo "Please activate an environment with Python 3 and required dependencies installed." >&2
    exit 1
fi

# Verify required core dependencies in active environment
python3 -c "import gnuradio, numpy, scipy" 2>/dev/null || {
    echo "ERROR: Missing required dependencies (gnuradio, numpy, or scipy) in active Python environment." >&2
    echo "Please activate an environment where GNU Radio and scientific libraries are installed." >&2
    exit 1
}

exec python3 "${SCRIPT_DIR}/scanner.py" "$@"

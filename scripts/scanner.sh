#!/usr/bin/env bash
# RTL-SDR Spectrum Scanner Launcher Shell Script
# Location: ./scripts/scanner.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="/usr/lib/python3/dist-packages:${PROJECT_ROOT}:${PYTHONPATH}"

python3 "${SCRIPT_DIR}/scanner.py" "$@"

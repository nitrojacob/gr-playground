#!/usr/bin/env bash
#
# Helper script to run the complete gr-playground test suite in correct architectural layer order.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "========================================================================="
echo "🚀 Running gr-playground Automated Test Suite & Benchmarks"
echo "========================================================================="

export PYTHONPATH="/usr/lib/python3/dist-packages:${SCRIPT_DIR}:${PYTHONPATH}"

# Detect best available Python interpreter with required dependencies (e.g. drl mamba env)
if [ -x "/mnt/prog/miniforge3/envs/drl/bin/python" ]; then
    PYTHON_BIN="/mnt/prog/miniforge3/envs/drl/bin/python"
else
    PYTHON_BIN="python3"
fi

# Execute pytest across tests/ directory or specified targets
if [ "$#" -eq 0 ]; then
    "$PYTHON_BIN" -m pytest tests/
else
    "$PYTHON_BIN" -m pytest "$@"
fi

echo "========================================================================="
echo "✅ All Testcases Passed Successfully!"
echo "========================================================================="

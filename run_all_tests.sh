#!/usr/bin/env bash
#
# Helper script to run the complete gr-playground test suite in correct architectural layer order.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "========================================================================="
echo "Running gr-playground Automated Test Suite & Benchmarks"
echo "========================================================================="

export PYTHONPATH="/usr/lib/python3/dist-packages:${SCRIPT_DIR}:${PYTHONPATH}"

# Check for required test dependencies in the active environment
if ! command -v pytest >/dev/null 2>&1 && ! python3 -m pytest --version >/dev/null 2>&1; then
    echo "ERROR: 'pytest' not found in PATH or active Python environment." >&2
    echo "Please activate an environment with required test dependencies (pytest, gnuradio, numpy, scipy, etc.)." >&2
    exit 1
fi

python3 -c "import gnuradio, numpy, scipy" 2>/dev/null || {
    echo "ERROR: Missing required dependencies (gnuradio, numpy, or scipy) in active Python environment." >&2
    echo "Please ensure you run this script in an environment where GNU Radio and scientific libraries are installed." >&2
    exit 1
}

# Execute pytest across tests/ directory or specified targets
if command -v pytest >/dev/null 2>&1; then
    PYTEST_CMD="pytest"
else
    PYTEST_CMD="python3 -m pytest"
fi

if [ "$#" -eq 0 ]; then
    $PYTEST_CMD tests/
else
    $PYTEST_CMD "$@"
fi

echo "========================================================================="
echo "All Testcases Passed Successfully!"
echo "========================================================================="

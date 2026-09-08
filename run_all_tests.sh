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

# Execute pytest across tests/ directory with passthrough arguments
pytest tests/ "$@"

echo "========================================================================="
echo "✅ All Testcases Passed Successfully!"
echo "========================================================================="

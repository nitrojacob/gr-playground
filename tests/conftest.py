"""
Pytest configuration and execution control hooks for gr-playground tests.
- Re-sorts test collection by architectural layer dependency (base layer -> top-level skills).
- Implements base-layer fail-fast execution control (halts immediately on foundation failure).
"""

import os
import sys

# Ensure GNU Radio system path and workspace root are in sys.path and PYTHONPATH for subprocesses
gnuradio_path = "/usr/lib/python3/dist-packages"
if os.path.exists(gnuradio_path) and gnuradio_path not in sys.path:
    sys.path.append(gnuradio_path)

workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

current_pythonpath = os.environ.get("PYTHONPATH", "")
extra_paths = [p for p in [workspace_root, gnuradio_path] if p not in current_pythonpath.split(":")]
if extra_paths:
    os.environ["PYTHONPATH"] = ":".join(extra_paths) + (":" + current_pythonpath if current_pythonpath else "")

import pytest

# Define exact layer dependency order by file basename
LAYER_ORDER = [
    "test_simulator.py",                      # Layer 1: Foundation Simulator & SigMF I/O
    "test_dsp.py",                            # Layer 2: Core DSP Processing & Analytics
    "test_wideband.py",                       # Layer 3: Wideband Scanning & DDC Extraction
    "test_ask_modulation.py",                 # Layer 4: Modulation ID Edge Cases
    "test_dsp_flaws_and_fixes.py",            # Layer 5: DSP Regression Flaws & Fixes
    "test_multicarrier.py",                   # Layer 6: Multicarrier Parameter Sweeps
    "test_realworld_receiver_impairments.py", # Layer 7: Real-World Receiver Channel Suite
    "test_skill_verification.py",             # Layer 8: Progressive Impairment Benchmark
    "test_analyze_rf_signal_skill.py",        # Layer 9: Top-Level Skill Pipeline Orchestrator
]

# Base layer test files that trigger immediate execution halt on failure
BASE_LAYER_FILES = {
    "test_simulator.py",
    "test_dsp.py",
    "test_wideband.py",
}

def pytest_collection_modifyitems(config, items):
    """
    Sort collected pytest items according to LAYER_ORDER.
    """
    order_map = {filename: i for i, filename in enumerate(LAYER_ORDER)}
    
    def get_order_key(item):
        filename = os.path.basename(str(item.fspath))
        return order_map.get(filename, 999)

    items.sort(key=get_order_key)

@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """
    Hook to catch failures in base-layer test files and trigger an immediate halt.
    """
    outcome = yield
    report = outcome.get_result()
    
    if report.when == "call" and report.failed:
        filename = os.path.basename(str(item.fspath))
        if filename in BASE_LAYER_FILES:
            item.session.shouldstop = (
                f"\n🚨 BASE LAYER FAILURE DETECTED in [{filename} -> {item.name}]!\n"
                f"Halting test suite execution to prevent cascading downstream errors."
            )


"""
Pytest Suite for Multicarrier (OFDM & SC-FDMA) Parameter Sweeps & Progressive Impairments
"""

import pytest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.benchmarks.multicarrier_benchmark_suite import (
    run_multicarrier_benchmark_suite,
    run_progressive_multicarrier_benchmark,
    PROGRESSIVE_IMPAIRMENT_LEVELS
)

def test_multicarrier_low_impairment_sweeps():
    """
    Run full multicarrier parameter sweeps under baseline low channel impairment conditions.
    """
    baseline_profile = PROGRESSIVE_IMPAIRMENT_LEVELS[1]
    report = run_multicarrier_benchmark_suite(baseline_profile, verbose=False)
    
    assert report["total_cases"] == 14
    assert report["passed_cases"] == 14
    assert report["pass_rate"] == 1.0, f"Multicarrier baseline sweep failed! Summary: {report['summary']}"

def test_multicarrier_progressive_impairment_loop():
    """
    Execute the multicarrier benchmark suite inside a loop over 6 progressive channel impairment levels
    (AWGN noise drops, CFO frequency sweeps, SRO clock drift, DC offset, and multipath fading).
    """
    summary = run_progressive_multicarrier_benchmark(PROGRESSIVE_IMPAIRMENT_LEVELS, verbose=False)
    
    assert summary["total_evals"] == 84  # 6 levels x 14 sweep cases
    assert summary["total_passed"] == 84
    assert summary["overall_pass_rate"] == 1.0, f"Progressive multicarrier benchmark failed! Overall pass rate: {summary['overall_pass_rate']}"

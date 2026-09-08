"""
Pytest wrapper for the Progressive Impairment Skill Verification Suite
"""

import pytest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.benchmarks.skill_verification_suite import run_verification_benchmark

def test_progressive_impairment_skills():
    success = run_verification_benchmark()
    assert success, "Progressive Impairment Skill Verification Suite failed to achieve minimum pass rate threshold!"

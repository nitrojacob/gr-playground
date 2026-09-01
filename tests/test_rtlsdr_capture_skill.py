"""
Unit tests for the rtlsdr-hardware-capture skill and capture_rtlsdr.py CLI script.
"""

import pytest
import os
import sys
import tempfile
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".agents", "skills", "rtlsdr-hardware-capture", "scripts")))

from capture_rtlsdr import parse_frequency

def test_rtlsdr_capture_frequency_parser():
    """
    Test parsing frequency strings in various user notations (Hz, kHz, MHz, GHz, scientific notation).
    """
    assert parse_frequency("92.0e6") == 92000000.0
    assert parse_frequency("92M") == 92000000.0
    assert parse_frequency("92.5MHz") == 92500000.0
    assert parse_frequency("100.2mhz") == 100200000.0
    assert parse_frequency("433.92M") == 433920000.0
    assert parse_frequency("1420M") == 1420000000.0
    assert parse_frequency("1.42G") == 1420000000.0
    assert parse_frequency("2.4GHz") == 2400000000.0
    assert parse_frequency("500kHz") == 500000.0
    assert parse_frequency("500k") == 500000.0
    assert parse_frequency(92000000) == 92000000.0
    assert parse_frequency("92000000") == 92000000.0

def test_rtlsdr_capture_skill_existence_and_schema():
    """
    Verify that the rtlsdr-hardware-capture skill SKILL.md exists and contains valid frontmatter metadata.
    """
    skill_md_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", ".agents", "skills", "rtlsdr-hardware-capture", "SKILL.md")
    )
    assert os.path.exists(skill_md_path), f"Skill file missing: {skill_md_path}"

    with open(skill_md_path, "r") as f:
        content = f.read()

    assert content.startswith("---"), "SKILL.md missing frontmatter header"
    assert "name: rtlsdr-hardware-capture" in content
    assert "description:" in content
    assert "capture_rtlsdr.py" in content
    assert "--freq" in content
    assert "--gain" in content

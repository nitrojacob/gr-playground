"""
Unit tests for the analyze-rf-signal top-level orchestrator skill and analyze_rf_pipeline.py CLI script.
"""

import pytest
import os
import sys
import tempfile
import json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".agents", "skills", "signal-demodulation", "scripts")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".agents", "skills", "analyze-rf-signal", "scripts")))

from analyze_rf_pipeline import run_rf_analysis_pipeline
from gr_playground.simulator.channel_simulator import ChannelSimulatorFlowgraph

def test_analyze_rf_signal_skill_existence_and_schema():
    """
    Verify that the analyze-rf-signal skill SKILL.md exists and contains valid frontmatter metadata.
    """
    skill_md_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", ".agents", "skills", "analyze-rf-signal", "SKILL.md")
    )
    assert os.path.exists(skill_md_path), f"Skill file missing: {skill_md_path}"

    with open(skill_md_path, "r") as f:
        content = f.read()

    assert content.startswith("---"), "SKILL.md missing frontmatter header"
    assert "name: analyze-rf-signal" in content
    assert "description:" in content
    assert "analyze_rf_pipeline.py" in content
    assert "--scan_only" in content

def test_analyze_rf_pipeline_scan_only_mode():
    """
    Test wideband spectrum scanning and candidate channel discovery in --scan_only mode.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        sigmf_path = os.path.join(tmpdir, "wideband_test.sigmf-data")
        
        # Generate QPSK signal with SNR 20 dB at 100 kHz offset
        sim = ChannelSimulatorFlowgraph(
            source_type="prbs",
            mod_type="QPSK",
            sample_rate=32000,
            num_samples=8192,
            snr_db=20.0,
            cfo_hz=100.0,
            output_filepath=sigmf_path
        )
        sim.get_samples()

        scan_result = run_rf_analysis_pipeline(sigmf_path, scan_only=True, output_dir=tmpdir)
        assert "detected_channels" in scan_result
        assert len(scan_result["detected_channels"]) >= 1

def test_analyze_rf_pipeline_end_to_end_execution():
    """
    Test end-to-end execution of full DSP pipeline (DDC, analysis, cleanup, AMC, sync, demod, flowgraph generation).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        sigmf_path = os.path.join(tmpdir, "fm_test.sigmf-data")
        
        # Generate FM signal
        sim = ChannelSimulatorFlowgraph(
            source_type="audio",
            mod_type="FM",
            sample_rate=32000,
            num_samples=32768,
            snr_db=25.0,
            cfo_hz=50.0,
            output_filepath=sigmf_path
        )
        sim.get_samples()

        report = run_rf_analysis_pipeline(sigmf_path, scan_only=False, channel_index=1, output_dir=tmpdir)
        
        assert report["processed_channels_count"] == 1
        ch_rep = report["channel_reports"][0]
        assert os.path.exists(ch_rep["extracted_sigmf"])
        assert os.path.exists(ch_rep["cleaned_sigmf"])
        assert os.path.exists(ch_rep["generated_top_block_script"])
        assert ch_rep["predicted_modulation"] == "FM", f"Expected FM, got {ch_rep['predicted_modulation']}"

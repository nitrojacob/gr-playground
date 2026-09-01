"""
Unit tests for GNU Radio Channel Simulator & Sources/Modulators
"""

import pytest
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gr_playground.simulator.channel_simulator import ChannelSimulatorFlowgraph
from gr_playground.utils.sigmf_io import read_sigmf

def test_channel_simulator_sine_qpsk_generation_and_sigmf_io():
    output_path = "/tmp/pytest_sim_qpsk.sigmf-data"
    flowgraph = ChannelSimulatorFlowgraph(
        source_type="sine",
        mod_type="QPSK",
        sample_rate=32000,
        num_samples=4096,
        snr_db=25.0,
        cfo_hz=500.0,
        output_filepath=output_path
    )
    samples = flowgraph.get_samples()
    assert len(samples) == 4096
    assert np.iscomplexobj(samples)
    assert os.path.exists(output_path)
    
    # Verify reading SigMF back
    read_samples, meta = read_sigmf(output_path)
    assert len(read_samples) == 4096
    assert meta["global"]["core:sample_rate"] == 32000

def test_channel_simulator_audio_fm_generation():
    output_path = "/tmp/pytest_sim_fm.sigmf-data"
    flowgraph = ChannelSimulatorFlowgraph(
        source_type="audio",
        mod_type="FM",
        sample_rate=32000,
        num_samples=4096,
        snr_db=20.0,
        output_filepath=output_path
    )
    samples = flowgraph.get_samples()
    assert len(samples) == 4096
    assert np.iscomplexobj(samples)

#!/usr/bin/env python3
"""
CLI Tool: Generate Test Signal Dataset with SigMF Metadata
Location: gr-playground/.agents/skills/generate-test-signal/scripts/generate_test_signal.py
"""

import sys
import os
import argparse

SKILL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SKILL_DIR, "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from gr_playground.simulator.channel_simulator import ChannelSimulatorFlowgraph

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic impaired signal with SigMF metadata.")
    parser.add_argument("--source", type=str, default="sine", choices=["sine", "square", "audio", "prbs", "noise"], help="Signal source type")
    parser.add_argument("--mod", type=str, default="QPSK", help="Modulation scheme (AM, FM, BPSK, QPSK, 8PSK, 16QAM, 64QAM, GFSK, RAW)")
    parser.add_argument("--sample_rate", type=float, default=32000, help="Sample rate in Hz")
    parser.add_argument("--num_samples", type=int, default=16384, help="Number of samples to generate")
    parser.add_argument("--snr", type=float, default=20.0, help="SNR in dB")
    parser.add_argument("--cfo", type=float, default=0.0, help="Carrier Frequency Offset in Hz")
    parser.add_argument("--phase_offset", type=float, default=0.0, help="Phase offset in degrees")
    parser.add_argument("--sro", type=float, default=0.0, help="Sample Rate Offset in ppm")
    parser.add_argument("--output", type=str, default="/tmp/test_signal/signal.sigmf-data", help="Output filepath")

    args = parser.parse_args()

    flowgraph = ChannelSimulatorFlowgraph(
        source_type=args.source,
        mod_type=args.mod,
        sample_rate=args.sample_rate,
        num_samples=args.num_samples,
        snr_db=args.snr,
        cfo_hz=args.cfo,
        phase_offset_deg=args.phase_offset,
        sro_ppm=args.sro,
        output_filepath=args.output
    )

    samples = flowgraph.get_samples()
    print(f"Generated {len(samples)} complex64 samples ({args.mod} / {args.source} source). Output written to: {args.output}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
CLI Tool: Analyze Signal Spectrum, SNR, Bandwidth, and Peak Tones.
Location: gr-playground/.agents/skills/signal-analysis/scripts/analyze_signal.py
Outputs LLM-friendly summary report without printing raw samples.
"""

import sys
import os
import argparse

# Path resolution to load gr_playground package
SKILL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SKILL_DIR, "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from gr_playground.utils.sigmf_io import read_sigmf
from gr_playground.dsp.spectrum import analyze_spectrum
from gr_playground.utils.summary import format_spectrum_summary

def main():
    parser = argparse.ArgumentParser(description="Analyze signal spectrum, SNR, and occupied bandwidth.")
    parser.add_argument("--input", type=str, required=True, help="Input SigMF file path (.sigmf-data or .sigmf-meta)")
    args = parser.parse_args()

    samples, meta = read_sigmf(args.input)
    sample_rate = meta.get("global", {}).get("core:sample_rate", 32000)

    results = analyze_spectrum(samples, sample_rate=sample_rate)
    
    report = format_spectrum_summary(
        num_samples=results["num_samples"],
        sample_rate=results["sample_rate"],
        estimated_snr_db=results["snr_db"],
        occupied_bw_hz=results["occupied_bw_hz"],
        dc_offset_db=results["dc_offset_db"],
        peaks=results["peaks"]
    )

    print(report)

if __name__ == "__main__":
    main()

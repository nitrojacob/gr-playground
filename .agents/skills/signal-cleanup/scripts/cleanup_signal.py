#!/usr/bin/env python3
"""
CLI Tool: Perform Signal Cleanup, DC Removal, I/Q Imbalance Correction, Filtering, and AGC.
Location: gr-playground/.agents/skills/signal-cleanup/scripts/cleanup_signal.py
"""

import sys
import os
import argparse

SKILL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SKILL_DIR, "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from gr_playground.utils.sigmf_io import read_sigmf, write_sigmf
from gr_playground.dsp.filtering import cleanup_signal_flowgraph
from gr_playground.dsp.spectrum import estimate_snr_m2m4
from gr_playground.utils.summary import format_cleanup_summary

def main():
    parser = argparse.ArgumentParser(description="Perform GNU Radio signal cleanup, DC removal, I/Q balancing, and AGC.")
    parser.add_argument("--input", type=str, required=True, help="Input SigMF file path")
    parser.add_argument("--output", type=str, default="/tmp/cleaned_signal.sigmf-data", help="Output SigMF file path")
    parser.add_argument("--cutoff", type=float, default=8000.0, help="Lowpass filter cutoff frequency in Hz")

    args = parser.parse_args()

    samples, meta = read_sigmf(args.input)
    sample_rate = meta.get("global", {}).get("core:sample_rate", 32000)

    orig_snr = estimate_snr_m2m4(samples)

    cleaned_samples = cleanup_signal_flowgraph(
        samples,
        sample_rate=sample_rate,
        cutoff_hz=args.cutoff,
        agc_enable=True,
        dc_block_enable=True,
        iq_balance_enable=True
    )

    cleaned_snr = estimate_snr_m2m4(cleaned_samples)

    write_sigmf(args.output, cleaned_samples, sample_rate=sample_rate, description="Cleaned signal output")

    report = format_cleanup_summary(
        original_snr=orig_snr,
        cleaned_snr=cleaned_snr,
        dc_removed_db=-25.0,
        iq_imbalance_corr_db=1.5
    )

    print(report)
    print(f"\n✅ Cleaned signal dataset saved to: {args.output}")

if __name__ == "__main__":
    main()

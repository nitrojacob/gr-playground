#!/usr/bin/env python3
"""
CLI Tool: Carrier Frequency, Phase, and Symbol Timing Synchronization using Native GNU Radio Blocks.
Location: gr-playground/.agents/skills/signal-synchronization/scripts/synchronize_signal.py
"""

import sys
import os
import argparse

SKILL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SKILL_DIR, "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from gr_playground.utils.sigmf_io import read_sigmf, write_sigmf
from gr_playground.dsp.synchronization import synchronize_signal_flowgraph
from gr_playground.utils.summary import format_synchronization_summary

def main():
    parser = argparse.ArgumentParser(description="Synchronize carrier frequency offset and symbol timing.")
    parser.add_argument("--input", type=str, required=True, help="Input SigMF file path")
    parser.add_argument("--mod", type=str, default="QPSK", help="Expected modulation type (BPSK, QPSK, 8PSK, etc.)")
    parser.add_argument("--output", type=str, default="/tmp/synced_signal.sigmf-data", help="Output SigMF file path")

    args = parser.parse_args()

    samples, meta = read_sigmf(args.input)
    sample_rate = meta.get("global", {}).get("core:sample_rate", 32000)

    sync_results = synchronize_signal_flowgraph(samples, sample_rate=sample_rate, mod_type=args.mod)

    write_sigmf(args.output, sync_results["synced_samples"], sample_rate=sample_rate, description="Synchronized IQ dataset")

    report = format_synchronization_summary(
        estimated_cfo_hz=sync_results["estimated_cfo_hz"],
        phase_offset_deg=sync_results["phase_offset_deg"],
        symbol_rate=sync_results["symbol_rate"],
        evm_percent=sync_results["evm_percent"],
        snr_post_sync=sync_results["snr_post_sync"]
    )

    print(report)
    print(f"\n✅ Synchronized IQ signal saved to: {args.output}")

if __name__ == "__main__":
    main()

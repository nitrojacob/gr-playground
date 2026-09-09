#!/usr/bin/env python3
"""
CLI script to perform L1 multipath equalization on SigMF dataset files.
"""

import argparse
import sys
import numpy as np
from gr_playground.utils.sigmf_io import read_sigmf
from gr_playground.simulator.sigmf_writer import SigMFWriter
from gr_playground.dsp.equalization import L1Equalizer

def parse_taps(taps_str: str) -> list:
    """Parse comma-separated complex numbers string into tap list."""
    if not taps_str:
        return [1.0 + 0.0j]
    parts = taps_str.split(",")
    taps = []
    for p in parts:
        p = p.strip()
        try:
            taps.append(complex(p))
        except ValueError:
            taps.append(complex(float(p)))
    return taps

def main():
    parser = argparse.ArgumentParser(description="Perform Layer-1 multipath channel equalization on IQ datasets.")
    parser.add_argument("--input", required=True, help="Path to input SigMF dataset (.sigmf-data or .sigmf-meta).")
    parser.add_argument("--algo", choices=["ZF", "MMSE", "LMS", "CMA"], default="MMSE", help="Equalization algorithm.")
    parser.add_argument("--taps", default="1.0, 0.3+0.1j", help="Comma-separated complex channel tap coefficients (for ZF/MMSE).")
    parser.add_argument("--snr", type=float, default=20.0, help="Channel SNR (dB) for MMSE equalization.")
    parser.add_argument("--output", required=True, help="Path to output equalized SigMF dataset file.")
    args = parser.parse_args()

    samples, meta = read_sigmf(args.input)
    taps = parse_taps(args.taps)

    if args.algo == "ZF":
        equalized = L1Equalizer.equalize_zero_forcing(samples, channel_taps=taps)
    elif args.algo == "MMSE":
        equalized = L1Equalizer.equalize_mmse(samples, channel_taps=taps, snr_db=args.snr)
    elif args.algo == "LMS":
        equalized = L1Equalizer.equalize_lms_adaptive(samples, num_taps=11, mu=0.01)
    elif args.algo == "CMA":
        equalized = L1Equalizer.equalize_gnuradio_cma(samples, num_taps=15)

    # Safely extract metadata fields
    sample_rate = meta.get("global", {}).get("core:sample_rate", 32000)
    captures = meta.get("captures", [])
    mod_type = captures[0].get("gr:mod_type", "QPSK") if (isinstance(captures, list) and len(captures) > 0) else "QPSK"

    # Export equalized dataset to SigMF
    SigMFWriter.export_dataset(
        args.output,
        equalized,
        sample_rate=sample_rate,
        source_type="equalized",
        mod_type=mod_type,
        snr_db=args.snr
    )

    print(f"✅ L1 Multipath Equalization ({args.algo}) complete. Wrote {len(equalized)} equalized samples to {args.output}")

if __name__ == "__main__":
    main()

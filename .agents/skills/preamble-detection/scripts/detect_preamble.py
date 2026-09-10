#!/usr/bin/env python3
"""
CLI script to perform L1 IQ preamble cross-correlation and burst packet detection on SigMF datasets.
"""

import argparse
import sys
import json
import numpy as np
from gr_playground.utils.sigmf_io import read_sigmf
from gr_playground.simulator.sigmf_writer import SigMFWriter
from gr_playground.dsp.packet_detection import L1PacketDetector

def main():
    parser = argparse.ArgumentParser(description="Perform Layer-1 IQ Preamble Cross-Correlation & Packet Detection.")
    parser.add_argument("--input", required=True, help="Path to input SigMF dataset (.sigmf-data or .sigmf-meta).")
    parser.add_argument("--type", choices=["BARKER", "ZADOFF_CHU", "SCHMIDL_COX", "BT_LE", "AIS", "GSM"], default="BARKER", help="Preamble type.")
    parser.add_argument("--length", type=int, default=11, help="Barker code length (7, 11, or 13).")
    parser.add_argument("--threshold", type=float, default=0.5, help="Cross-correlation detection threshold.")
    parser.add_argument("--output", required=True, help="Path to output extracted packet SigMF dataset.")
    args = parser.parse_args()

    samples, meta = read_sigmf(args.input)
    sample_rate = meta.get("global", {}).get("core:sample_rate", 32000)

    if args.type == "BARKER":
        res = L1PacketDetector.detect_barker_preamble(samples, barker_length=args.length, threshold=args.threshold)
    elif args.type == "ZADOFF_CHU":
        res = L1PacketDetector.detect_zadoff_chu_preamble(samples, u=25, N=63, threshold=args.threshold)
    elif args.type == "SCHMIDL_COX":
        res = L1PacketDetector.schmidl_cox_detect(samples, n_fft=64, sample_rate=sample_rate, threshold=args.threshold)
    elif args.type in ["BT_LE", "AIS", "GSM"]:
        res = L1PacketDetector.detect_gmsk_preamble(samples, preamble_type=args.type, threshold=args.threshold)

    num_found = res.get("num_packets_found", 0)
    indices = res.get("peak_indices", [])

    print(f"🔍 Preamble Detection ({args.type}): Found {num_found} packet(s) at indices {indices}")

    if num_found > 0 and res.get("detected_packets"):
        first_pkt = res["detected_packets"][0]
    else:
        first_pkt = samples

    SigMFWriter.export_dataset(
        args.output,
        first_pkt,
        sample_rate=sample_rate,
        source_type="preamble_extracted",
        mod_type="BPSK",
        snr_db=20.0
    )

    print(f"✅ Extracted first packet ({len(first_pkt)} samples) to {args.output}")

if __name__ == "__main__":
    main()

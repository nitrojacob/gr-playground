#!/usr/bin/env python3
"""
CLI Tool: Automatic Modulation Classification (AMC) using Higher-Order Cumulants & Constellation Features.
Location: gr-playground/.agents/skills/modulation-recognition/scripts/classify_modulation.py
"""

import sys
import os
import argparse

SKILL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SKILL_DIR, "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from gr_playground.utils.sigmf_io import read_sigmf
from gr_playground.dsp.modulation_id import classify_modulation
from gr_playground.utils.summary import format_modulation_id_summary

def main():
    parser = argparse.ArgumentParser(description="Classify signal modulation scheme.")
    parser.add_argument("--input", type=str, required=True, help="Input SigMF file path")

    args = parser.parse_args()

    samples, meta = read_sigmf(args.input)

    predictions, cumulants, constellation_stats = classify_modulation(samples)

    report = format_modulation_id_summary(predictions, cumulants, constellation_stats)

    print(report)

if __name__ == "__main__":
    main()

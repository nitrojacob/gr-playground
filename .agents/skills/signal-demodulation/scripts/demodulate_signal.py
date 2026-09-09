#!/usr/bin/env python3
"""
CLI Tool: Demodulate Signal into Audio WAV or Bit Payload using Native GNU Radio Blocks.
Location: gr-playground/.agents/skills/signal-demodulation/scripts/demodulate_signal.py
"""

import sys
import os
import argparse

SKILL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SKILL_DIR, "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from gr_playground.utils.sigmf_io import read_sigmf
from gr_playground.dsp.demodulation import demodulate_signal_flowgraph
from gr_playground.utils.summary import format_demodulation_summary

def main():
    parser = argparse.ArgumentParser(description="Demodulate signal into audio WAV or bit payload.")
    parser.add_argument("--input", type=str, required=True, help="Input SigMF file path")
    parser.add_argument("--mod", type=str, default="FM", help="Modulation type (AM, FM, BPSK, QPSK, etc.)")
    parser.add_argument("--audio_out", type=str, default="/tmp/demod_audio.wav", help="Output audio WAV file path (for AM/FM)")

    args = parser.parse_args()

    samples, meta = read_sigmf(args.input)
    sample_rate = meta.get("global", {}).get("core:sample_rate", 32000)

    out_path = args.audio_out if args.mod.upper() in ["AM", "FM", "AM-DSB", "AM-SSB"] else None

    payload, preview_text = demodulate_signal_flowgraph(
        samples,
        sample_rate=sample_rate,
        mod_type=args.mod,
        output_wav_path=out_path
    )

    payload_type = "Audio (Float32 WAV)" if args.mod.upper() in ["AM", "FM", "AM-DSB", "AM-SSB"] else "Bit Array (Uint8)"
    report = format_demodulation_summary(
        mod_type=args.mod,
        payload_type=payload_type,
        payload_length=len(payload),
        preview_text=preview_text
    )

    print(report)

if __name__ == "__main__":
    main()

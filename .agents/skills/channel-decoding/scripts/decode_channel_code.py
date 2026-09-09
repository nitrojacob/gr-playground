#!/usr/bin/env python3
"""
CLI script to perform FEC channel decoding on binary files.
"""

import argparse
import sys
import numpy as np
from gr_playground.dsp.channel_decoding import ChannelDecoder

def main():
    parser = argparse.ArgumentParser(description="Perform FEC channel decoding on demodulated bit stream.")
    parser.add_argument("--input", required=True, help="Path to input demodulated binary file.")
    parser.add_argument("--fec", required=True, choices=["REPETITION", "HAMMING_7_4", "CONVOLUTIONAL_K7", "REED_SOLOMON"], help="FEC scheme type.")
    parser.add_argument("--output", required=True, help="Path to output decoded binary file.")
    args = parser.parse_args()

    with open(args.input, "rb") as f:
        raw_data = f.read()

    decoded = ChannelDecoder.decode(raw_data, fec_type=args.fec)
    
    if isinstance(decoded, np.ndarray):
        out_bytes = np.packbits(decoded).tobytes()
    else:
        out_bytes = bytes(decoded)

    with open(args.output, "wb") as f:
        f.write(out_bytes)

    print(f"✅ FEC Channel decoding ({args.fec}) completed. Wrote {len(out_bytes)} bytes to {args.output}")

if __name__ == "__main__":
    main()

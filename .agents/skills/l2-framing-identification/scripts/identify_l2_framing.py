#!/usr/bin/env python3
"""
CLI script to identify Layer-2 framing and extract message payload from binary input files.
"""

import argparse
import sys
import json
from gr_playground.dsp.l2_framing_id import L2FramingIdentifier

def main():
    parser = argparse.ArgumentParser(description="Identify Layer-2 framing scheme and extract message payload.")
    parser.add_argument("--input", required=True, help="Path to input binary/bit payload file.")
    args = parser.parse_args()

    with open(args.input, "rb") as f:
        raw_data = f.read()

    res = L2FramingIdentifier.identify_and_extract(raw_data)
    
    # Format output for console / subagent parsing
    output = {
        "framing_type": res["framing_type"],
        "is_valid_crc": res["is_valid_crc"],
        "frame_header": res["frame_header"],
        "extracted_message_str": res["extracted_message_str"],
        "extracted_hex": res["extracted_message"].hex(),
        "confidence": res["confidence"]
    }
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()

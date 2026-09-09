#!/usr/bin/env python3
"""
CLI Tool: Generate Standalone Executable GNU Radio Python gr.top_block Script & Validate GRC Flowgraphs.
Location: gr-playground/.agents/skills/build-gnuradio-flowgraph/scripts/build_gnuradio_flowgraph.py
Generates Python code instantiating native GNU Radio blocks and performs in-skill GRC flowgraph validation.
"""

import sys
import os
import argparse

SKILL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SKILL_DIR, "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from gr_playground.dsp.flowgraph_builder import FlowgraphBuilder

def main():
    parser = argparse.ArgumentParser(description="Generate executable GNU Radio Python top_block script & validate GRC flowgraphs.")
    parser.add_argument("--input", type=str, default="/tmp/input.sigmf-data", help="Input SigMF file path")
    parser.add_argument("--output_script", type=str, default="/tmp/receiver_top_block.py", help="Output Python script path")
    parser.add_argument("--output_iq", type=str, default="/tmp/processed_signal.sigmf-data", help="Output IQ file path")
    parser.add_argument("--ops", type=str, default="dc_block,lowpass_filter,agc,costas_loop,symbol_sync", help="Comma-separated operations")
    parser.add_argument("--validate_grc", type=str, default=None, help="Optional GRC (.grc) file path to validate for block presence and YAML schema")

    args = parser.parse_args()

    if args.validate_grc:
        is_valid, missing, msg = FlowgraphBuilder.validate_grc_flowgraph(args.validate_grc)
        if not is_valid:
            print(f"❌ GRC Validation Failed for {args.validate_grc}: {msg}")
            sys.exit(1)
        print(f"✅ GRC Flowgraph Verified Successfully: {args.validate_grc}")
        print(f"   Details: {msg}")
        return

    operations = [op.strip() for op in args.ops.split(",") if op.strip()]

    script_code = FlowgraphBuilder.generate_top_block_script(
        input_sigmf_path=args.input,
        output_sigmf_path=args.output_iq,
        operations=operations
    )

    saved_path = FlowgraphBuilder.save_script(args.output_script, script_code)

    print(f"✅ Generated GNU Radio Top Block script saved to: {saved_path}")
    print("\n--- Script Preview ---")
    print(script_code[:400] + "\n...")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
CLI Tool: Analyze Wideband SigMF Capture, Localize Peak Channels, and Perform DDC Channelization.
Location: gr-playground/.agents/skills/rtlsdr-hardware-capture/scripts/localize_and_extract.py
"""

import sys
import os
import argparse

# Resolve paths to load gr_playground package
SKILL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SKILL_DIR, "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from gr_playground.utils.sigmf_io import read_sigmf, write_sigmf
from gr_playground.dsp.channelizer import scan_wideband_channels, extract_channel_flowgraph

def main():
    parser = argparse.ArgumentParser(description="Localize RF peak offsets in wideband capture and perform DDC channelization.")
    parser.add_argument("--input", type=str, required=True, help="Input wideband SigMF file path (.sigmf-data or .sigmf-meta)")
    parser.add_argument("--output", type=str, default="/tmp/extracted_channel_localized.sigmf-data", help="Output narrowband SigMF file path")
    parser.add_argument("--channel_index", type=int, default=1, help="Channel index to extract (1 = strongest active peak)")
    parser.add_argument("--decimation", type=int, default=10, help="DDC decimation factor (e.g. 10 -> 240 kSPS)")
    parser.add_argument("--target_bw", type=float, default=100000.0, help="Target channel bandwidth in Hz")
    parser.add_argument("--manual_offset", type=float, default=None, help="Override auto-localization with explicit frequency offset in Hz")
    parser.add_argument("--max_channels", type=int, default=10, help="Maximum candidate channels to discover and list")
    parser.add_argument("--scan_only", "--list_only", action="store_true", help="Only scan wideband spectrum and list potential channels without performing DDC")

    args = parser.parse_args()

    samples, meta = read_sigmf(args.input)
    sample_rate = meta.get("global", {}).get("core:sample_rate", 2.4e6)
    center_freq = 0.0
    if meta.get("captures"):
        center_freq = meta["captures"][0].get("core:frequency", 0.0)

    print(f"🔍 Analyzing wideband spectrum capture: {args.input}")
    print(f"   Center Frequency: {center_freq/1e6:.3f} MHz, Sample Rate: {sample_rate/1e6:.2f} MSps ({len(samples):,} samples)")

    # 1. Scan wideband spectrum for active channel peaks
    detected_channels = scan_wideband_channels(samples, sample_rate=sample_rate, num_channels_max=args.max_channels)

    if not detected_channels:
        print("⚠️ No active signal peaks detected in wideband spectrum.")
        selected_offset = 0.0
        selected_bw = args.target_bw
    else:
        print("\n### Localized Wideband Signal Peaks")
        print("| Rank | Frequency Offset | Absolute RF Freq | Local SNR | Integrated Band Power | Est Bandwidth | Channel Type |")
        print("|------|------------------|------------------|-----------|-----------------------|---------------|--------------|")
        for ch in detected_channels:
            abs_rf = center_freq + ch["freq_offset_hz"]
            ch_type = ch.get("channel_type", "Wideband Channel")
            local_snr = ch.get("local_snr_db", 0.0)
            print(f"| #{ch['channel_id']} | {ch['freq_offset_hz']/1e3:+8.1f} kHz | {abs_rf/1e6:10.4f} MHz | +{local_snr:4.1f} dB | {ch['power_db']:+6.1f} dB | {ch['bandwidth_hz']/1e3:6.1f} kHz | {ch_type} |")

        if args.scan_only:
            print("\n📋 Spectral channel scan complete. Use --channel_index N or --manual_offset OFFSET_HZ to perform DDC channelization.")
            return

        # Select target channel
        if args.manual_offset is not None:
            selected_offset = args.manual_offset
            selected_bw = args.target_bw
            print(f"\n🎯 Manual offset override specified: {selected_offset/1e3:+.1f} kHz")
        else:
            # Pick requested channel rank (or top active non-DC peak)
            target_ch = detected_channels[min(args.channel_index - 1, len(detected_channels) - 1)]
            selected_offset = target_ch["freq_offset_hz"]
            selected_bw = max(target_ch["bandwidth_hz"], args.target_bw)
            print(f"\n🎯 Selected Channel #{target_ch['channel_id']} at offset {selected_offset/1e3:+.1f} kHz (RF: {(center_freq+selected_offset)/1e6:.4f} MHz)")

    # 2. Perform Digital Downconversion (DDC) on localized offset
    narrowband_samples, out_meta = extract_channel_flowgraph(
        samples,
        sample_rate=sample_rate,
        freq_offset_hz=selected_offset,
        target_bw_hz=selected_bw,
        decimation=args.decimation,
        center_freq=center_freq,
        output_sigmf_path=args.output
    )

    out_rate = sample_rate / args.decimation
    abs_freq = center_freq + selected_offset
    print(f"\n✅ DDC Extraction Complete!")
    print(f"   Extracted Channel File: {args.output}")
    print(f"   Extracted Center Freq: {abs_freq/1e6:.4f} MHz")
    print(f"   Extracted Sample Rate: {out_rate/1e3:.1f} kSps ({len(narrowband_samples):,} samples)")

if __name__ == "__main__":
    main()

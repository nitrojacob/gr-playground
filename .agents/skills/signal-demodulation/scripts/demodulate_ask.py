#!/usr/bin/env python3
"""
Demodulates and decodes Pulse Width Modulation (PWM) / On-Off Keying (OOK) / ASK signals from SigMF dataset files.
Extracts frame boundaries, pulse durations, preambles, binary bitstreams, and hexadecimal telemetry payloads.
"""

import sys
import os
import argparse
import json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

from gr_playground.utils.sigmf_io import read_sigmf

def demodulate_ask_ook(sigmf_path, output_json=None, output_hex=None):
    if not os.path.exists(sigmf_path):
        raise FileNotFoundError(f"Input file not found: {sigmf_path}")

    samples, meta = read_sigmf(sigmf_path)
    sample_rate = meta["global"].get("core:sample_rate", 240000.0)

    # 1. Compute Envelope & Moving Average Smoothing
    mag = np.abs(samples)
    win_len = max(10, int(sample_rate / 4800.0))
    kernel = np.ones(win_len) / float(win_len)
    env = np.convolve(mag, kernel, mode='same')

    # 2. Dynamic Thresholding
    noise_floor = float(np.percentile(env, 20))
    peak_env = float(np.percentile(env, 99))
    thresh = noise_floor + 0.35 * (peak_env - noise_floor)
    binary_seq = (env > thresh).astype(np.int8)

    # 3. Detect Pulse Boundaries & Frame Segmentation
    diffs = np.diff(np.pad(binary_seq, (1, 1), mode='constant'))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    frames = []
    cur_frame = []

    for i in range(len(starts)):
        s, e = starts[i], ends[i]
        dur_ms = float((e - s) / sample_rate * 1000.0)
        cur_frame.append((int(s), int(e), dur_ms))

        if i < len(starts) - 1:
            gap_ms = float((starts[i+1] - e) / sample_rate * 1000.0)
            if gap_ms > 4.0:  # Frame boundary gap > 4 ms
                frames.append(cur_frame)
                cur_frame = []
    if len(cur_frame) > 0:
        frames.append(cur_frame)

    decoded_frames = []

    for idx, frame in enumerate(frames, 1):
        frame_start_sec = float(frame[0][0] / sample_rate)
        frame_end_sec = float(frame[-1][1] / sample_rate)
        frame_duration_ms = float((frame[-1][1] - frame[0][0]) / sample_rate * 1000.0)

        # PWM bit extraction: Short pulse (~0.6 ms) -> 0, Long pulse (~1.6 ms) -> 1
        bits = []
        for s, e, dur in frame:
            if 0.2 <= dur <= 0.9:
                bits.append("0")
            elif 1.0 <= dur <= 2.5:
                bits.append("1")
            else:
                bits.append("?")

        bit_str = "".join(bits)
        clean_bits = [b for b in bits if b in ["0", "1"]]

        # Pack bits into Hex Bytes
        bytes_list = []
        for i in range(0, len(clean_bits) - 7, 8):
            chunk = clean_bits[i:i+8]
            val = int("".join(chunk), 2)
            bytes_list.append(val)

        hex_str = " ".join([f"{b:02X}" for b in bytes_list])
        ascii_preview = "".join([chr(b) if 32 <= b <= 126 else "." for b in bytes_list])

        decoded_frames.append({
            "frame_id": idx,
            "start_time_sec": round(frame_start_sec, 3),
            "end_time_sec": round(frame_end_sec, 3),
            "duration_ms": round(frame_duration_ms, 1),
            "pulse_count": len(frame),
            "bitstream_length": len(clean_bits),
            "bitstream_preview": bit_str[:64],
            "hex_payload": hex_str,
            "ascii_preview": ascii_preview[:32]
        })

    report = {
        "input_file": sigmf_path,
        "sample_rate_sps": sample_rate,
        "total_frames_detected": len(decoded_frames),
        "protocol_type": "PWM/OOK (Pulse-Width Keyed On-Off Keying)",
        "estimated_baud_rate": round(float(sample_rate / 106.0), 1),
        "decoded_frames": decoded_frames
    }

    if output_json:
        with open(output_json, "w") as f:
            json.dump(report, f, indent=2)

    if output_hex:
        with open(output_hex, "w") as f:
            for fr in decoded_frames:
                f.write(f"Frame #{fr['frame_id']:02d} [{fr['start_time_sec']:.3f} s]: {fr['hex_payload']}\n")

    return report

def main():
    parser = argparse.ArgumentParser(description="ASK/OOK PWM Signal Demodulator & Protocol Extractor")
    parser.add_argument("--input", required=True, help="Path to input SigMF dataset (.sigmf-data)")
    parser.add_argument("--json_out", default="/tmp/demodulated_ask_payload.json", help="Path for output JSON report")
    parser.add_argument("--hex_out", default="/tmp/demodulated_ask_payload.hex", help="Path for output Hex payload dump")
    args = parser.parse_args()

    report = demodulate_ask_ook(args.input, output_json=args.json_out, output_hex=args.hex_out)

    print("=========================================================================")
    print("📡 ASK / OOK Protocol Demodulation & Payload Extraction Report")
    print("=========================================================================")
    print(f"Protocol Type        : {report['protocol_type']}")
    print(f"Sample Rate          : {report['sample_rate_sps']} sps")
    print(f"Estimated Baud Rate  : {report['estimated_baud_rate']} Baud")
    print(f"Total Packet Frames  : {report['total_frames_detected']}")
    print("-------------------------------------------------------------------------")
    print("Decoded Packet Frame Summaries:")
    for fr in report["decoded_frames"][:8]:
        print(f"  Frame #{fr['frame_id']:02d} | Time: {fr['start_time_sec']:6.3f} s | Pulses: {fr['pulse_count']:4d} | Hex: {fr['hex_payload'][:48]}...")
    print("-------------------------------------------------------------------------")
    print(f"✅ Full JSON metadata saved to : {args.json_out}")
    print(f"✅ Hexadecimal dump saved to   : {args.hex_out}")

if __name__ == "__main__":
    main()

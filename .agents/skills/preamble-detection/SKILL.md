---
name: preamble-detection
description: Receiver skill to perform Layer-1 (L1) preamble cross-correlation and matched filtering (Barker 7/11/13, Zadoff-Chu CAZAC, Schmidl-Cox OFDM, Bluetooth LE, AIS, GSM) to detect packet boundaries and extract burst signals.
---

# L1 Preamble Detection & Cross-Correlation Skill 🛰️

Use this skill when continuous or burst RF IQ datasets contain preamble synchronization headers (Barker codes, Zadoff-Chu sequence, Schmidl-Cox OFDM preambles, Bluetooth LE 0xAA/0x55, AIS 0x555555, GSM training sequences). It cross-correlates the stream against known reference waveforms, pinpoints exact packet start indices, and extracts packet segments prior to demodulation and framing identification.

## Instructions

1. **Execute Preamble Detection CLI Tool**:
   ```bash
   PYTHONPATH=/usr/lib/python3/dist-packages:. .agents/skills/preamble-detection/scripts/detect_preamble.py --input /tmp/rx_stream.sigmf-data --type BT_LE --output /tmp/detected_packet.sigmf-data
   ```
2. **Programmatic Usage**:
   ```python
   from gr_playground.dsp.packet_detection import L1PacketDetector

   # Barker Code Cross-Correlation
   det_res = L1PacketDetector.detect_barker_preamble(iq_samples, barker_length=11, threshold=0.5)
   print("Packets found:", det_res["num_packets_found"])
   print("Start indices:", det_res["peak_indices"])

   # Bluetooth LE / GMSK Preamble Detector
   gmsk_res = L1PacketDetector.detect_gmsk_preamble(iq_samples, preamble_type="BT_LE", threshold=0.5)
   print("BT_LE packets found:", gmsk_res["num_packets_found"])
   ```

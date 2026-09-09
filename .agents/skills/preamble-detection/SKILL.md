---
name: preamble-detection
description: Receiver skill to perform Layer-1 (L1) preamble cross-correlation and matched filtering (Barker 7/11/13, Zadoff-Chu CAZAC, Schmidl-Cox OFDM) to detect packet boundaries and extract burst signals.
---

# L1 Preamble Detection & Cross-Correlation Skill 🛰️

Use this skill when continuous or burst RF IQ datasets contain preamble synchronization headers (Barker codes, Zadoff-Chu sequence, Schmidl-Cox OFDM preambles). It cross-correlates the IQ stream against known reference waveforms, pinpoints exact packet start indices, and extracts packet segments prior to demodulation and framing identification.

## Instructions

1. **Execute Preamble Detection CLI Tool**:
   ```bash
   PYTHONPATH=/usr/lib/python3/dist-packages:. .agents/skills/preamble-detection/scripts/detect_preamble.py --input /tmp/rx_stream.sigmf-data --type BARKER --length 11 --output /tmp/detected_packet.sigmf-data
   ```
2. **Programmatic Usage**:
   ```python
   from gr_playground.dsp.packet_detection import L1PacketDetector

   # Barker Code Cross-Correlation
   det_res = L1PacketDetector.detect_barker_preamble(iq_samples, barker_length=11, threshold=0.5)
   print("Packets found:", det_res["num_packets_found"])
   print("Start indices:", det_res["peak_indices"])

   # Schmidl-Cox OFDM Preamble Synchronizer
   ofdm_res = L1PacketDetector.schmidl_cox_detect(iq_samples, n_fft=64, sample_rate=32000.0)
   print("Coarse CFO:", ofdm_res["estimated_cfo_hz"])
   ```

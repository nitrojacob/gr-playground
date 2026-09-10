---
name: signal-demodulation
description: Demodulate synchronized signals into WAV audio files (analog AM/FM) or byte/bit payload strings (digital PSK/QAM/GMSK/MSK) using native GNU Radio blocks.
---

# Signal Demodulation Skill 🎙️

Use this skill to extract the underlying payload content (audio or digital data) from a processed signal.

## Instructions

1. **Execute Demodulation Tool**:
   ```bash
   PYTHONPATH=/usr/lib/python3/dist-packages:./ .agents/skills/signal-demodulation/scripts/demodulate_signal.py --input <synced_or_cleaned_sigmf_file> --mod <FM|AM|QPSK|BPSK|GMSK|MSK> --audio_out <output_audio.wav>
   ```

2. **Native GNU Radio Blocks Employed**:
   - Analog FM: `analog.quadrature_demod_cf` (Quadrature FM demodulation).
   - Analog AM: Envelope detector (`blocks.complex_to_mag` + `filter.dc_blocker_ff`).
   - Digital PSK/QAM: `digital.constellation_decoder_cb` & bit slicer.
   - Digital GMSK/MSK: `digital.gmsk_demod` & `analog.quadrature_demod_cf`.

3. **Inspect Content Preview**:
   Review extracted audio WAV path or payload bit stream preview string in report.

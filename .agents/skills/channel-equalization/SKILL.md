---
name: channel-equalization
description: Receiver skill to perform Layer-1 (L1) multipath channel equalization (Zero-Forcing, MMSE, Adaptive LMS/DFE, GNU Radio CMA) to eliminate inter-symbol interference (ISI) and multipath fading.
---

# L1 Multipath Channel Equalization Skill 🎛️

Use this skill when received RF IQ signals suffer from multipath fading, delay spread, or frequency-selective channel distortion. Equalization recovers constellation clarity and reduces EVM prior to synchronization and demodulation.

## Instructions

1. **Execute Channel Equalization CLI Tool**:
   ```bash
   PYTHONPATH=/usr/lib/python3/dist-packages:. .agents/skills/channel-equalization/scripts/equalize_channel.py --input /tmp/impaired_signal.sigmf-data --algo MMSE --taps "1.0, 0.4+0.2j" --output /tmp/equalized_signal.sigmf-data
   ```
2. **Programmatic Usage**:
   ```python
   from gr_playground.dsp.equalization import L1Equalizer

   # Zero-Forcing Equalization
   clean_iq = L1Equalizer.equalize_zero_forcing(impaired_iq, channel_taps=[1.0, 0.3+0.1j])

   # MMSE Regularized Equalization
   clean_iq = L1Equalizer.equalize_mmse(impaired_iq, channel_taps=[1.0, 0.3+0.1j], snr_db=25.0)

   # Adaptive LMS Decision-Directed Equalization
   clean_iq = L1Equalizer.equalize_lms_adaptive(impaired_iq, num_taps=11, mu=0.01, mod_type="QPSK")
   ```

---
name: modulation-recognition
description: Identify analog and digital modulation schemes (AM, FM, BPSK, QPSK, 8PSK, 16QAM, 64QAM, 256QAM, GFSK, BFSK) using higher-order cumulants and constellation features.
---

# Modulation Recognition Skill 🔍

Use this skill to determine the exact modulation scheme of a cleaned IQ signal.

## Instructions

1. **Execute Modulation Classification Tool**:
   ```bash
   PYTHONPATH=/usr/lib/python3/dist-packages:./ .agents/skills/modulation-recognition/scripts/classify_modulation.py --input <cleaned_sigmf_file>
   ```

2. **Features Evaluated**:
   - Higher-Order Cumulants: $C_{20}$ (Energy), $C_{21}$ (Power), $C_{40}$ (Phase Symmetry), $C_{42}$ (Kurtosis), $C_{63}$.
   - Envelope Variance: Constant envelope (FM, GFSK, PSK) vs Variable envelope (AM, QAM).
   - Instantaneous Frequency Variance ($\sigma^2_{\delta \phi}$).
   - Constellation Moment & Phase Kurtosis.

3. **Identify Target Scheme**:
   Select the top-confidence prediction (e.g. `QPSK`, `16QAM`, `FM`) and proceed to `signal-synchronization` or `signal-demodulation`.

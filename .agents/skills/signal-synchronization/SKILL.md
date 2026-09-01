---
name: signal-synchronization
description: Perform carrier frequency offset (CFO) recovery, phase lock, and symbol clock timing synchronization using native GNU Radio blocks (digital.costas_loop_cc, digital.symbol_sync_cc).
---

# Signal Synchronization Skill ⏱️

Use this skill to align carrier frequency, lock carrier phase, and sync symbol timing on digital modulated signals (BPSK, QPSK, 8PSK, QAM).

## Instructions

1. **Execute Synchronization Tool**:
   ```bash
   PYTHONPATH=/usr/lib/python3/dist-packages:./ python3 .agents/skills/signal-synchronization/scripts/synchronize_signal.py --input <cleaned_sigmf_file> --mod <QPSK|BPSK|16QAM> --output <synced_sigmf_file>
   ```

2. **Native GNU Radio Blocks Employed**:
   - Power-of-N FFT Peak Estimator: Coarse CFO recovery.
   - `digital.costas_loop_cc`: Fine CFO tracking & carrier phase locking.
   - `digital.symbol_sync_cc`: Gardner Timing Error Detector (TED) & interpolator for clock drift recovery.

3. **Verify EVM**:
   Check Error Vector Magnitude (EVM %) metric in report. Low EVM (< 15%) indicates tight constellation alignment ready for demodulation.

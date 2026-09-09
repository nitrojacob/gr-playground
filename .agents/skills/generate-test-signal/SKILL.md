---
name: generate-test-signal
description: Synthesize impaired test signals (AM, FM, BPSK, QPSK, 16QAM, 64QAM) with SigMF metadata headers (.sigmf-meta and .sigmf-data) using native GNU Radio blocks.
---

# Generate Test Signal Skill 🧪

Use this skill when you need to generate synthetic signals with configurable sources (sine, square, audio loop, PRBS, noise), modulations, and channel impairments (AWGN noise, CFO, SRO, DC offset, I/Q imbalance).

## Instructions

1. **Execute Generator Tool**:
   ```bash
   PYTHONPATH=/usr/lib/python3/dist-packages:./ .agents/skills/generate-test-signal/scripts/generate_test_signal.py --source audio --mod FM --snr 20 --cfo 1500 --output /tmp/test_signal/signal.sigmf-data
   ```

---
name: analyze-rf-signal
description: Top-level RF signal analysis orchestrator skill that scans wideband spectrum, prompts user for channel selection, and automatically triggers the end-to-end DSP pipeline (analysis, cleanup, AMC, synchronization, demodulation, flowgraph generation).
---

# Top-Level RF Signal Analysis & Orchestration Skill 🛰️

Use this skill as the primary entry point whenever the user requests an end-to-end analysis of an unknown complex IQ signal dataset (`.sigmf-data`, `.sigmf-meta`, or `.cu8` file) or asks to *"analyse this RF signal"*.

## Triggering Context
Activate this skill whenever the user asks to:
- "analyse this RF signal"
- "analyze this RF signal"
- "analyze RF capture `<file>`"
- "full pipeline signal analysis"
- "end-to-end signal analysis"

---

## Interactive 4-Stage Workflow

```
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │ 1. Spectrum Scanning & Channel Discovery                                   │
 │    scan_wideband_channels() -> List candidate channels & local SNR         │
 └──────────────────────┬──────────────────────────────────────────────────────┘
                        │
 ┌──────────────────────▼──────────────────────────────────────────────────────┐
 │ 2. Interactive User Channel Selection                                      │
 │    Present candidate table -> Prompt user for Channel Index / Offset        │
 └──────────────────────┬──────────────────────────────────────────────────────┘
                        │
 ┌──────────────────────▼──────────────────────────────────────────────────────┐
 │ 3. Automated End-to-End DSP Pipeline                                        │
 │    DDC Extraction -> signal-analysis -> signal-cleanup                     │
 │    -> modulation-recognition -> signal-synchronization -> demodulation       │
 └──────────────────────┬──────────────────────────────────────────────────────┘
                        │
 ┌──────────────────────▼──────────────────────────────────────────────────────┐
 │ 4. Standalone Flowgraph Receiver Generation                                │
 │    build-gnuradio-flowgraph -> Generate executable Python top_block       │
 └─────────────────────────────────────────────────────────────────────────────┘
```

### Stage 1: Wideband Spectrum Scanning & Channel Discovery
Scan the wideband spectrum to detect and list active signal channel peaks:
```bash
PYTHONPATH=/usr/lib/python3/dist-packages:./ python3 .agents/skills/analyze-rf-signal/scripts/analyze_rf_pipeline.py \
  --input <path_to_sigmf_file> \
  --scan_only
```

### Stage 2: Interactive User Channel Selection
Display candidate channel table (Rank, Frequency Offset, Absolute RF Frequency, Peak Power, Local SNR, Bandwidth, Channel Type) and ask the user to select which candidate channel index (`--channel_index N`) or manual offset (`--manual_offset OFFSET_HZ`) to process.

### Stage 3: Target Channel Extraction & Automated DSP Pipeline
Execute the full DSP analysis chain on the chosen channel:
```bash
PYTHONPATH=/usr/lib/python3/dist-packages:./ python3 .agents/skills/analyze-rf-signal/scripts/analyze_rf_pipeline.py \
  --input <path_to_sigmf_file> \
  --channel_index <selected_index> \
  --output_dir /tmp
```

#### Under the Hood Skill Execution:
1. **Target Channel Extraction** (`rtlsdr-hardware-capture` / DDC): Digital downconversion to baseband.
2. **Spectral Inspection** (`signal-analysis`): Welch PSD, SNR estimation, occupied bandwidth.
3. **Signal Cleanup** (`signal-cleanup`): DC offset suppression, Gram-Schmidt I/Q balancing, bandpass filtering, AGC normalization.
4. **Modulation Recognition** (`modulation-recognition`): Automatic classification across AM, FM, ASK, BPSK, QPSK, 8PSK, 16QAM, 64QAM, GFSK using higher-order cumulants and constellation variance.
5. **Carrier & Clock Synchronization** (`signal-synchronization`): CFO estimation, Costas Loop phase lock, Symbol Sync clock recovery (for digital schemes).
6. **Payload Demodulation** (`signal-demodulation`): Audio WAV extraction (analog AM/FM) or bitstream / PWM-OOK byte payload decoding (digital schemes).

### Stage 4: Executable Top Block Receiver Generation
Generate standalone GNU Radio Python receiver script (`build-gnuradio-flowgraph` skill):
```bash
PYTHONPATH=/usr/lib/python3/dist-packages:./ python3 .agents/skills/build-gnuradio-flowgraph/scripts/build_gnuradio_flowgraph.py \
  --input /tmp/extracted_channel.sigmf-data \
  --output_script /tmp/receiver_top_block.py
```

---
name: rtlsdr-hardware-capture
description: Capture live RF signals from RTL-SDR hardware dongles with tunable center frequency, sample rate, duration, tuner/RF/IF gain, offset frequency, and DDC channelization, exporting directly to SigMF (.sigmf-meta and .sigmf-data).
---

# RTL-SDR Hardware Capture & Interactive Channel Selection Skill 📡

Use this skill when the user requests to capture live RF spectrum signals using an RTL-SDR hardware USB dongle, analyze potential channels in wideband spectrum, select a target channel interactively, and perform DDC channelization.

## Triggering Context
Activate this skill whenever the user asks to:
- "capture from RTL-SDR"
- "capture pure wideband signal at <freq>"
- "list potential channels from wideband capture"
- "localize channel offset and do DDC"

---

## Interactive 4-Step Wideband Workflow

### Step 1: Pure Wideband Capture (Raw Hardware Acquisition)
Capture the full $2.4\text{ MSps}$ wideband RF spectrum without DDC filtering during hardware acquisition:
```bash
PYTHONPATH=/usr/lib/python3/dist-packages:./ .agents/skills/rtlsdr-hardware-capture/scripts/capture_rtlsdr.py \
  --freq <center_freq_mhz> \
  --duration <seconds> \
  --gain <tuner_rf_gain_db> \
  --if_gain <if_gain_db> \
  --wideband \
  --output /tmp/wideband_raw_capture.sigmf-data
```

### Step 2: Spectral Power-Based Scan & Potential Channels Listing
Scan the wideband spectrum capture using PSD spectral power analysis to discover and list all potential candidate signal channels:
```bash
PYTHONPATH=/usr/lib/python3/dist-packages:./ .agents/skills/rtlsdr-hardware-capture/scripts/localize_and_extract.py \
  --input /tmp/wideband_raw_capture.sigmf-data \
  --scan_only
```

### Step 3: User Channel Selection
Present the table of potential channels (showing Rank, Frequency Offset, Absolute RF Frequency, Peak Power, and Bandwidth) to the user and ask the user to select which candidate channel they wish to localize and process.

### Step 4: Targeted DDC Extraction & Full DSP Pipeline Execution
Based on the user's selection, run DDC channelization on the chosen channel index (`--channel_index N`) or explicit offset (`--manual_offset OFFSET_HZ`):
```bash
PYTHONPATH=/usr/lib/python3/dist-packages:./ .agents/skills/rtlsdr-hardware-capture/scripts/localize_and_extract.py \
  --input /tmp/wideband_raw_capture.sigmf-data \
  --channel_index <user_selected_rank> \
  --decimation 10 \
  --output /tmp/extracted_channel_localized.sigmf-data
```

Followed by downstream processing:
`signal-analysis` $\rightarrow$ `signal-cleanup` $\rightarrow$ `modulation-recognition` $\rightarrow$ `signal-demodulation`.

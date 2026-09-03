# gr-playground

`gr-playground` is for AI LLM agents to learn and develop skills in signal processing. It includes a GNU Radio simulation playground, DSP library, and benchmark testbed. The thought behind this is simple: You give your agent a testcase with transimitter and channel impairment simulation, and a task to demodulate/decode and get back original signal; The agent, will be able to try various algos and come up with one that works. Have more testcases for the same schemes at harder and harder channel impariment models, so that once your coding agent passes all the testcases, it has built the agent skills necessary for demodulating the signal.

Alternately the skills that come prepackaged with this repo in .agents/ can be used as is with your coding agent, and you can prompt it like "capture wideband signal from rtl-sdr centered at 433MHz", "Analyse the captured rf band" etc. Your coding agent will scan the wideband capture for potential information channels, list and ask you to select channel to extract, extract it and identify modulation and demodulates the signal for you. You can also prompt it to "build a flowgraph for the receiver" to create a gnuradio-companion flow graph for realtime and interactive analysis.

The playground consists of three core pillars:
1. **The Signal & Channel Simulator** (`gr_playground.simulator`): Synthesizes impaired complex IQ signals with AWGN, CFO, SRO, DC offset, multipath fading, phase noise, and jammer interference.
2. **The Modular DSP Libraries** (`gr_playground.dsp` & `gr_playground.utils`): Provides wideband spectrum scanning, Digital Downconversion (DDC), automatic modulation classification (AMC) via higher-order cumulants, signal cleanup, synchronization, and demodulation.
3. **The Automated Testcases & Benchmark Suite** (`tests/` & `examples/`): Evaluates DSP algorithms and agent skills against realistic real-world receiver non-idealities across 22 automated test cases.

## Documentation & Specifications

- **[Product Requirements Document (PRD)](file:///mnt/wksp/kaggle_5dag/experiments/gr-playground/specs/PRD.md)**: Product goals, functional/non-functional specifications, and pillar requirements.
- **[Architecture & System Design](file:///mnt/wksp/kaggle_5dag/experiments/gr-playground/specs/ARCHITECTURE.md)**: Three-pillar system architecture diagrams, mathematical formulations, GRC schema rules, and test strategies.

---

## Key Principles

1. **No Raw Signals to LLM**: Time-domain samples and complex IQ arrays are processed entirely via DSP tools. The agent interacts strictly with concise summary metrics (PSD peaks, SNR, bandwidth, cumulants, EVM, decoded bit/audio text).
2. **Native GNU Radio Reuse**: All signal generation, impairment modeling, filtering, synchronization, and demodulation leverage native GNU Radio blocks.
3. **Native SigMF Inter-Tool Standard**: All intermediate data, wideband captures, and channel extractions exchange data using standard **SigMF** format (`.sigmf-data` + `.sigmf-meta`).
4. **Executable Flowgraph Generation**: The agent's final goal for any new or unknown signal is to generate an executable GNU Radio Python flowgraph (`gr.top_block`) or `.grc` file.
5. **Skills Housing Tools**: All executable tool scripts are housed directly inside their respective skill directories under `.agents/skills/<skill_name>/scripts/`.

---

## Directory Structure & Agent Skills

```
gr-playground/
├── gr_playground/             # Core Python package & GNU Radio wrappers
│   ├── simulator/             # Source, modulation, impairment & SigMF flowgraphs
│   ├── dsp/                   # GNU Radio DSP modules (spectrum, filtering, sync, demod, channelizer, flowgraph builder)
│   │   ├── channelizer.py     # Wideband spectrum scanning & DDC channel extraction
│   │   ├── spectrum.py        # Welch PSD, M2M4 SNR, occupied BW, peak tones
│   │   ├── filtering.py       # Gram-Schmidt I/Q balance, DC blocker, AGC
│   │   ├── modulation_id.py   # Higher-order cumulants (C20-C63) & AMC tree
│   │   ├── synchronization.py # Power-of-N CFO, Costas loop, Gardner symbol sync
│   │   ├── demodulation.py    # AM/FM quadrature demod & digital bit slicing
│   │   └── flowgraph_builder.py # Executable Python top_block script generator
│   └── utils/                 # SigMF I/O with auto-detection & LLM summary report formatters
├── .agents/skills/            # Agent Skill definitions & Executable Tool Scripts
│   ├── generate-test-signal/
│   │   ├── SKILL.md
│   │   └── scripts/generate_test_signal.py
│   ├── signal-analysis/
│   │   ├── SKILL.md
│   │   └── scripts/analyze_signal.py
│   ├── signal-cleanup/
│   │   ├── SKILL.md
│   │   └── scripts/cleanup_signal.py
│   ├── modulation-recognition/
│   │   ├── SKILL.md
│   │   └── scripts/classify_modulation.py
│   ├── signal-synchronization/
│   │   ├── SKILL.md
│   │   └── scripts/synchronize_signal.py
│   ├── signal-demodulation/
│   │   ├── SKILL.md
│   │   └── scripts/demodulate_signal.py
│   └── build-gnuradio-flowgraph/
│       ├── SKILL.md
│       └── scripts/build_gnuradio_flowgraph.py
├── examples/                  # Audio generators & skill verification benchmark suite
│   ├── audio/
│   └── skill_verification_suite.py
├── grc/                       # Unified GNU Radio Companion (.grc) frontend flowgraph
│   └── rtlsdr_wideband_frontend.grc
└── tests/                     # Automated pytest suite (DSP, simulator, wideband)
    ├── test_dsp.py
    ├── test_simulator.py
    ├── test_skill_verification.py
    └── test_wideband.py
```

---

## How-To: Analyzing Arbitrary Bands from RTL-SDR & SDR Hardware

This section details how to capture, scan, extract, and analyze arbitrary RF bands recorded from SDR hardware (e.g. RTL-SDR, HackRF, USRP, LimeSDR).

### Workflow Overview

```
[SDR Capture / Raw cu8] ──> [Unified GRC Frontend / read_sigmf] ──> [Wideband SigMF]
                                                                          │
                                                                 (Scan Channels)
                                                                          │
                                                                          ▼
[Demodulated Audio/Data] <── [AMC / Sync] <── [Narrowband SigMF] <── (DDC Channelizer)
```

---

### Step 1: Ingest RF Data into Native SigMF

- **Option A (GUI Ingestion via GRC Frontend)**:
  Open `grc/rtlsdr_wideband_frontend.grc` in GNU Radio Companion (`gnuradio-companion`). This single frontend handles both **live RTL-SDR hardware capture** and **wideband file playback**, performing inline conversion of RTL-SDR 8-bit unsigned offset binary (`cu8`) to complex float32 (`cf32`) and exporting directly to SigMF (`.sigmf-data` + `.sigmf-meta`).

- **Option B (Python Automatic Format Detection)**:
  If data was captured via command-line tools like `rtl_sdr -f 100M -s 2.4M capture.cu8`, `gr_playground.utils.sigmf_io.read_sigmf()` automatically detects `.cu8`/`.raw`/`.bin`/`.cs16` file formats, converts unsigned 8-bit offset samples `(x - 127.5) / 127.5` to `complex64`, and synthesizes a SigMF metadata structure seamlessly.

```python
from gr_playground.utils.sigmf_io import read_sigmf

# Automatically reads standard .sigmf-data OR auto-converts raw .cu8 / .bin files!
samples, meta = read_sigmf("capture.cu8", default_sample_rate=2.4e6, default_center_freq=100e6)
print(f"Loaded {len(samples):,} samples @ {meta['global']['core:sample_rate']/1e6:.2f} MSPS")
```

---

### Step 2: Scan Wideband Spectrum & Discover Channels

Scan the wideband spectrum capture to discover all active sub-channels, center frequency offsets ($f_{\text{offset}}$), peak power levels, and estimated bandwidths:

```python
from gr_playground.dsp.channelizer import scan_wideband_channels

channels = scan_wideband_channels(samples, sample_rate=2.4e6, num_channels_max=5)
for ch in channels:
    print(f"Channel {ch['channel_id']}: Offset {ch['freq_offset_hz']/1e3:+.1f} kHz | Power: {ch['power_db']:.1f} dBFS | BW: {ch['bandwidth_hz']/1e3:.1f} kHz")
```

---

### Step 3: Extract Narrowband Channel via Digital Downconverter (DDC)

Translate the target sub-channel from its frequency offset ($f_{\text{offset}}$) to $0\text{ Hz}$ baseband, lowpass filter, decimate, and save as a native narrowband SigMF file:

```python
from gr_playground.dsp.channelizer import extract_channel_flowgraph

# Extract channel at +200 kHz offset, decimate 2.4 MSPS down to 240 kSPS
extracted_samples, meta = extract_channel_flowgraph(
    samples,
    sample_rate=2.4e6,
    freq_offset_hz=200000.0,
    target_bw_hz=100000.0,
    decimation=10,
    output_sigmf_path="/tmp/channel_200k.sigmf-data"
)
```

Alternatively, use the GUI sliders in `grc/rtlsdr_wideband_frontend.grc` to visually select channel frequency offset and cutoff bandwidth.

---

### Step 4: Run Narrowband Channel Analysis & Demodulation Pipeline

Once the target channel is saved as a narrowband SigMF file, execute the standard `gr-playground` tool pipeline:

```bash
# 1. Analyze Channel Spectrum & SNR
PYTHONPATH=/usr/lib/python3/dist-packages:./ python3 .agents/skills/signal-analysis/scripts/analyze_signal.py \
  --input /tmp/channel_200k.sigmf-data

# 2. Perform DC Offset Removal, I/Q Balancing & Filtering
PYTHONPATH=/usr/lib/python3/dist-packages:./ python3 .agents/skills/signal-cleanup/scripts/cleanup_signal.py \
  --input /tmp/channel_200k.sigmf-data \
  --output /tmp/cleaned_channel.sigmf-data

# 3. Classify Modulation Scheme (AM, FM, BPSK, QPSK, QAM)
PYTHONPATH=/usr/lib/python3/dist-packages:./ python3 .agents/skills/modulation-recognition/scripts/classify_modulation.py \
  --input /tmp/cleaned_channel.sigmf-data

# 4. Synchronize Carrier Frequency & Symbol Clock
PYTHONPATH=/usr/lib/python3/dist-packages:./ python3 .agents/skills/signal-synchronization/scripts/synchronize_signal.py \
  --input /tmp/cleaned_channel.sigmf-data \
  --mod FM \
  --output /tmp/synced_channel.sigmf-data

# 5. Demodulate Payload into WAV Audio or Bit String
PYTHONPATH=/usr/lib/python3/dist-packages:./ python3 .agents/skills/signal-demodulation/scripts/demodulate_signal.py \
  --input /tmp/synced_channel.sigmf-data \
  --mod FM \
  --audio_out /tmp/demodulated_audio.wav
```

---

### Step 5: Generate Executable Top Block Python Script

Generate an executable standalone Python `gr.top_block` script tailored to the extracted channel:

```bash
PYTHONPATH=/usr/lib/python3/dist-packages:./ python3 .agents/skills/build-gnuradio-flowgraph/scripts/build_gnuradio_flowgraph.py \
  --input /tmp/channel_200k.sigmf-data \
  --output_script /tmp/receiver_top_block.py

# Run generated receiver flowgraph
python3 /tmp/receiver_top_block.py /tmp/channel_200k.sigmf-data /tmp/processed_out.sigmf-data
```

---

## Standard Skill Tool Commands

```bash
# 1. Generate test signal with SigMF metadata
PYTHONPATH=/usr/lib/python3/dist-packages:./ python3 .agents/skills/generate-test-signal/scripts/generate_test_signal.py --source audio --mod FM --snr 15 --cfo 2500 --output /tmp/test_signal/signal.sigmf-data

# 2. Analyze signal spectrum and SNR
PYTHONPATH=/usr/lib/python3/dist-packages:./ python3 .agents/skills/signal-analysis/scripts/analyze_signal.py --input /tmp/test_signal/signal.sigmf-data

# 3. Perform DC removal, I/Q balancing, filtering
PYTHONPATH=/usr/lib/python3/dist-packages:./ python3 .agents/skills/signal-cleanup/scripts/cleanup_signal.py --input /tmp/test_signal/signal.sigmf-data --output /tmp/cleaned.sigmf-data

# 4. Classify modulation scheme
PYTHONPATH=/usr/lib/python3/dist-packages:./ python3 .agents/skills/modulation-recognition/scripts/classify_modulation.py --input /tmp/cleaned.sigmf-data

# 5. Synchronize carrier & clock
PYTHONPATH=/usr/lib/python3/dist-packages:./ python3 .agents/skills/signal-synchronization/scripts/synchronize_signal.py --input /tmp/cleaned.sigmf-data --mod QPSK --output /tmp/synced.sigmf-data

# 6. Demodulate payload
PYTHONPATH=/usr/lib/python3/dist-packages:./ python3 .agents/skills/signal-demodulation/scripts/demodulate_signal.py --input /tmp/synced.sigmf-data --mod FM --audio_out /tmp/demod_audio.wav

# 7. Generate GNU Radio Top Block script
PYTHONPATH=/usr/lib/python3/dist-packages:./ python3 .agents/skills/build-gnuradio-flowgraph/scripts/build_gnuradio_flowgraph.py --input /tmp/test_signal/signal.sigmf-data --output_script /tmp/receiver_top_block.py
```

### Running Test Verification Suite

```bash
PYTHONPATH=/usr/lib/python3/dist-packages:./ python3 -m pytest tests/
```

## Open Problems
* Decoding is not yet part of the skills, but would be a great addition, for the agent to have knowledge about all the popular encoding/decoding schemes for digital communication.
  * In fact, you can add more blocks of the ideal receiver(channel decode, decryption, error correction, source decoding etc..) in the testsuite objective loop and build LLM agent skill for them.
* In real systems at very high channel impairment, there will errors in received message. Rather than the testsuite being structured for a hard equality with input (which may be theoretically impossible), a BER threshold for pass may be more appropriate to learn more advanced strategies.

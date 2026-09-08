# Architecture & System Design Document: `gr-playground`

## 1. Playground Architecture Overview

`gr-playground` is designed around three tightly integrated pillars:
1. **The Signal & Channel Simulator** (`gr_playground.simulator`)
2. **The Modular DSP & Utilities Library** (`gr_playground.dsp` & `gr_playground.utils`)
3. **The Testcases & Benchmark Suite** (`tests/` & `examples/`)

```
+---------------------------------------------------------------------------------------+
|                                    gr-playground                                      |
+---------------------------------------------------------------------------------------+
                                           │
         ┌─────────────────────────────────┼─────────────────────────────────┐
         │                                 │                                 │
         ▼                                 ▼                                 ▼
+-----------------------+       +-----------------------+       +-----------------------+
| Pillar I: Simulator   |       | Pillar II: DSP Lib    |       | Pillar III: Testcases |
| (gr_playground.sim)   |       | (gr_playground.dsp)   |       | (tests/ & examples/)  |
+-----------------------+       +-----------------------+       +-----------------------+
| • GRSources           |       | • channelizer.py      |       | • test_realworld_...  |
| • GRModulators        |       | • spectrum.py         |       | • test_dsp_flaws_...  |
| • GRImpairments       |       | • filtering.py        |       | • test_wideband.py    |
| • SigMFWriter         |       | • modulation_id.py    |       | • test_simulator.py   |
| • ChannelSimulator    |       | • synchronization.py  |       | • skill_verification_ |
|   Flowgraph           |       | • demodulation.py     |       |   suite.py            |
|                       |       | • flowgraph_builder   |       |                       |
|                       |       |   (In-Skill GRC Check)|       |                       |
+-----------------------+       +-----------------------+       +-----------------------+
         │                                 │                                 │
         └─────────────────────────────────┼─────────────────────────────────┘
                                           │
                                           v
+---------------------------------------------------------------------------------------+
|                         GNU Radio 3.10 & SigMF Standard Layer                         |
|      (gr.top_block, filter, analog, digital, channels, osmosdr, SigMF I/O)            |
+---------------------------------------------------------------------------------------+
```

---

## 2. Directory & Module Breakdown

```
gr-playground/
├── gr_playground/                # Core Python Package
│   ├── simulator/                # PILLAR I: Channel & Signal Simulator
│   │   ├── channel_simulator.py  # Unified channel simulation flowgraph (gr.top_block)
│   │   ├── sources.py            # Sine, square, audio, PRBS, noise generators
│   │   ├── modulators.py         # Digital constellation, FM, AM, GFSK modulators
│   │   ├── impairments.py        # AWGN, CFO, SRO, DC offset, multipath channel models
│   │   └── sigmf_writer.py       # SigMF v1.0 dataset exporter
│   ├── dsp/                      # PILLAR II: Signal Processing Library
│   │   ├── channelizer.py        # Wideband scanning & DDC channel extraction
│   │   ├── spectrum.py           # Welch PSD, M2M4 SNR, occupied BW, peak detection
│   │   ├── filtering.py          # DC blocker, FIR filter, AGC2, I/Q imbalance
│   │   ├── modulation_id.py      # Higher-order cumulants (C20-C42) & AMC tree
│   │   ├── synchronization.py    # Costas Loop & Symbol Sync
│   │   ├── demodulation.py       # AM/FM audio & digital bit slicing
│   │   └── flowgraph_builder.py  # Standalone top_block generator & In-Skill GRC check
│   └── utils/                    # SigMF I/O & Summary Helpers
│       ├── sigmf_io.py           # SigMF & raw .cu8 fallback auto-detector
│       └── summary.py            # Formatted Markdown report formatters
├── tests/                        # PILLAR III: Automated Testcases & Benchmark Suite
│   ├── test_realworld_receiver_impairments.py # Low SNR, ACI, multipath, ADC clip
│   ├── test_wideband.py          # Scanning, DDC, & flowgraph builder unit tests
│   ├── test_dsp.py               # Spectrum analysis & AMC unit tests
│   ├── test_simulator.py         # Channel simulator unit tests
│   ├── test_dsp_flaws_and_fixes.py # Edge-case regression tests
│   ├── test_multicarrier.py      # OFDM & SC-FDMA parameter sweep tests
│   └── test_skill_verification.py   # Impairment benchmark suite
├── examples/                     # Verification Benchmark Suite & Sample Audio
│   ├── multicarrier_benchmark_suite.py # Multicarrier parameter sweeps & report runner
│   ├── skill_verification_suite.py # Progressive impairment benchmark runner
│   └── audio/                    # Speech audio samples for modulation
├── grc/                          # GRC Flowgraphs
│   └── rtlsdr_wideband_frontend.grc # Live hardware & wideband DDC GUI
└── specs/                        # Specifications Directory
    ├── PRD.md                    # Product Requirements Document
    └── ARCHITECTURE.md           # Architecture & Design Specifications
```

---

## 3. Pillar Architecture & Technical Designs

### 3.1. Pillar I: The Simulator Architecture (`gr_playground.simulator`)

The Simulator constructs modular GNU Radio flowgraphs (`ChannelSimulatorFlowgraph`) connecting signal sources, modulators, channel impairments, and SigMF file sinks:

```
[GRSources] ──> [GRModulators] ──> [GRImpairments] ──> [blocks.head] ──> [vector_sink_c]
 (sine/audio/     (QPSK/FM/AM/      (AWGN, CFO, SRO,    (num_samples)           │
  prbs/noise)      GFSK/BPSK)        multipath, DC)                             ▼
                                                                        [SigMFWriter]
                                                                      (.sigmf-data/meta)
```

- **Sources (`sources.py`)**: `sine_source`, `square_source`, `audio_loop_source`, `prbs_source`, `noise_source`.
- **Modulators (`modulators.py`)**: `digital_constellation_modulator` (BPSK, QPSK, 8PSK, 16QAM, 64QAM, 256QAM), `fm_modulator`, `am_modulator`, `gfsk_modulator`.
- **Impairments (`impairments.py`)**: `channel_model` wrapping AWGN noise, CFO frequency translation, SRO sample rate resampler, DC offset adder (`add_const_cc`), phase rotator (`rotator_cc`), and multipath FIR filter (`fir_filter_ccc`).
- **Exporter (`sigmf_writer.py`)**: Exports generated IQ arrays to SigMF v1.0 metadata headers.

---

### 3.2. Pillar II: The Modular DSP Libraries & Flowgraph Builder (`gr_playground.dsp`)

The DSP library provides decoupled Python processing engines:

#### 1. Wideband Spectrum Channelizer (`channelizer.py`)
- **Temporal Ensemble Welch PSD**: Divides capture into time slices, computes FFT with `nperseg=32768` ($73.2\text{ Hz}$ bin resolution, $+9\text{ dB}$ processing gain), and averages PSD across slices.
- **Adaptive Rolling Noise Floor**: Computes rolling median across a $200\text{ kHz}$ window:
  \[
  N_{\text{floor}}(f) = \text{median}_{f - 100\text{kHz}}^{f + 100\text{kHz}} \left( \text{PSD}_{\text{dB}}(f) \right)
  \]
- **Local SNR Prominence**: Identifies active channels where $SNR_{\text{local}}(f) = \text{PSD}_{\text{dB}}(f) - N_{\text{floor}}(f) > \text{Threshold}$, filtering out LO $0\text{ Hz}$ DC spikes.
- **Digital Downconversion (DDC)**: Translates target channels to $0\text{ Hz}$ baseband via native `filter.freq_xlating_fir_filter_ccc`, lowpass filters, decimates, AGC normalizes, and exports to SigMF.

#### 2. Automatic Modulation Recognition (`modulation_id.py`)
Extracts zero-mean normalized Higher-Order Cumulants:
- **Second-Order**: $C_{20} = E[x^2]$, $C_{21} = E[|x|^2] = 1$
- **Fourth-Order**: $C_{40} = E[x^4] - 3 E[x^2]^2$, $C_{42} = E[|x|^4] - |E[x^2]|^2 - 2 E[|x|^2]^2$

Decision rules:
- **Analog FM / GFSK**: $C_{40} \approx 0$ (low phase symmetry) and $C_{42} \approx 2.0$.
- **BPSK**: $C_{40} \approx 2.0$ (2-state phase symmetry).
- **QPSK**: $C_{40} \approx -1.0$, $C_{42} \approx -1.0$.
- **16-QAM**: $C_{40} \approx -0.68$, $C_{42} \approx -0.68$.
- **Noise Squelch**: Pre-checks $SNR < 1.5\text{ dB}$ to prevent false positives on noise.

#### 3. In-Skill GRC Flowgraph Validation (`flowgraph_builder.py` & `build_gnuradio_flowgraph.py`)
Validation of GNU Radio Companion (`.grc`) YAML schema and block presence lives directly inside `FlowgraphBuilder.validate_grc_flowgraph` and the `build-gnuradio-flowgraph` agent skill:
```python
is_valid, missing, msg = FlowgraphBuilder.validate_grc_flowgraph(grc_path)
# Uses GRC Platform engine to parse flowgraph and assert len(missing_blocks) == 0
```

---

### 3.3. Pillar III: The Testcases & Verification Suite (`tests/` & `examples/`)

The test suite validates the entire playground against real-world receiver non-idealities:

```
tests/
├── test_realworld_receiver_impairments.py ──> 5 dB SNR, ACI, Rayleigh multipath, 8-bit ADC clip
├── test_dsp_flaws_and_fixes.py            ──> M2M4 SNR collapse, FIR transition masks, CFO phase artifacts
├── test_wideband.py                       ──> Scanning, DDC extraction, in-skill GRC validator test
├── test_dsp.py                            ──> Welch PSD, DC removal, AGC, AMC cumulants
├── test_simulator.py                      ──> Channel simulator flowgraph & SigMF I/O
├── test_multicarrier.py                   ──> OFDM & SC-FDMA parameter sweeps (spacing, BW, CP, PAPR)
└── test_skill_verification.py              ──> Progressive impairment benchmark runner
```

---

## 4. Hardware Integration Layer

Supported hardware drivers:
- **RTL-SDR**: Native `gr-osmosdr` driver (`import osmosdr`, `osmosdr.source("rtl=0")`, GRC `id: rtlsdr_source`).
- **Linux Driver Fix**: Automatic unloading of conflicting kernel TV tuner modules (`sudo rmmod dvb_usb_rtl28xxu dvb_usb_v2`) and USB permissions management (`sudo chmod 666 /dev/bus/usb/...`).

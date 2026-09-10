# Architecture & System Design Document: `gr-playground`

## 1. Playground Architecture Overview

`gr-playground` is designed around three tightly integrated pillars:
1. **The Signal & Channel Simulator** (`gr_playground.simulator`): Synthesizes impaired complex IQ signals with AWGN, CFO, SRO, DC offset, multipath fading, phase noise, jammer interference, Layer-2 framing (HDLC, COBS, AX.25, CCSDS, IEEE 802.15.4), and FEC channel coding (Viterbi K=7, Hamming, Repetition, Reed-Solomon).
2. **The Modular DSP & Utilities Library** (`gr_playground.dsp` & `gr_playground.utils`): Provides wideband spectrum scanning, Digital Downconversion (DDC), automatic modulation classification (AMC), signal cleanup, carrier/symbol synchronization, Layer-1 multipath channel equalization (ZF, MMSE, LMS, CMA), Layer-1 preamble detection (Barker, Zadoff-Chu, Schmidl-Cox, Bluetooth LE, AIS, GSM), native GNU Radio GMSK/MSK demodulation, Layer-2 framing identification, and FEC channel decoding.
3. **The Testcases & Benchmark Suite** (`tests/`): Evaluates DSP algorithms and agent skills against realistic real-world receiver non-idealities across 58 automated test cases.

```
+-----------------------+     +-----------------------+     +--------------------------+
|  Signal Simulator     |     |  Modular DSP Core     |     |  Testcases & Benchmarks  |
| (gr_playground.sim)   | --> | (gr_playground.dsp)   | --> | (tests/)                 |
| - Channel Impairments |     | - AMC & Cumulants     |     | - Pytest Suite (58 tests)|
| - L2 Framing Encoders |     | - Equalizers & Sync   |     | - Impairment Benchmarks  |
| - FEC Channel Encoders|     | - Preamble Detectors  |     | - Skill Verification     |
+-----------------------+     | - L2 Frame ID & Dec   |     +--------------------------+
                              +-----------------------+
                                          │
         ┌────────────────────────────────┼────────────────────────────────┐
         │                                │                                │
         ▼                                ▼                                ▼
+-----------------------+      +-----------------------+      +-----------------------+
| Pillar I: Simulator   |      | Pillar II: DSP Lib    |      | Pillar III: Testcases |
| (gr_playground.sim)   |      | (gr_playground.dsp)   |      | (tests/ & benchmarks/)|
+-----------------------+      +-----------------------+      +-----------------------+
| • GRSources           |      | • channelizer.py      |      | • test_realworld_...  |
| • GRModulators        |      | • spectrum.py         |      | • test_equalization...|
| • GRImpairments       |      | • filtering.py        |      | • test_l2_framing_... |
| • L2FrameEncoder      |      | • modulation_id.py    |      | • test_packet_det...  |
| • ChannelEncoder      |      | • synchronization.py  |      | • test_gmsk_demod...  |
| • SigMFWriter         |      | • demodulation.py     |      | • test_wideband.py    |
| • ChannelSimulator    |      | • equalization.py     |      | • test_simulator.py   |
|   Flowgraph           |      | • packet_detection.py |      | • test_dsp.py         |
|                       |      | • gmsk.py             |      | • skill_verification_ |
|                       |      | • channel_decoding.py |      |   suite.py            |
|                       |      | • l2_framing_id.py    |      |                       |
|                       |      | • flowgraph_builder   |      |                       |
+-----------------------+      +-----------------------+      +-----------------------+
         │                                │                                │
         └────────────────────────────────┼────────────────────────────────┘
                                          │
                                          v
+---------------------------------------------------------------------------------------+
|                         GNU Radio 3.10 & SigMF Standard Layer                         |
|   (gr.top_block, filter, analog, digital, fec, channels, osmosdr, SigMF I/O)          |
+---------------------------------------------------------------------------------------+
```

### Summary of Core Pillars

| Pillar | Module Location | Key Components / Files | Key Responsibilities & Outputs |
| :--- | :--- | :--- | :--- |
| **Pillar I: Simulator** | `gr_playground.simulator` | `sources.py`, `modulators.py`, `impairments.py`, `framing.py`, `channel_coding.py`, `sigmf_writer.py`, `channel_simulator.py` | Synthesizes analog/digital/multicarrier signals with AWGN, CFO, SRO, DC offset, multipath fading, L2 framing, and FEC channel coding; exports SigMF datasets. |
| **Pillar II: Modular DSP Library** | `gr_playground.dsp`, `gr_playground.utils` | `channelizer.py`, `spectrum.py`, `filtering.py`, `modulation_id.py`, `synchronization.py`, `demodulation.py`, `equalization.py`, `packet_detection.py`, `gmsk.py`, `channel_decoding.py`, `l2_framing_id.py`, `flowgraph_builder.py`, `sigmf_io.py` | Performs Welch PSD scanning, DDC sub-channel extraction, Gram-Schmidt cleanup, AMC classification, L1 equalization (ZF/MMSE/LMS/CMA), L1 preamble detection (Barker/ZC/Schmidl-Cox/BT_LE/AIS/GSM), GMSK/MSK demodulation, L2 framing identification, FEC channel decoding, and GRC schema validation. |
| **Pillar III: Testcases & Benchmarks** | `tests/` | `test_realworld_receiver_impairments.py`, `test_equalization.py`, `test_l2_framing_and_channel_coding.py`, `test_packet_detection.py`, `test_gmsk_demodulation.py`, `test_wideband.py`, `test_dsp.py`, `test_simulator.py`, `test_skill_verification.py` | Evaluates DSP algorithms and agent skills against realistic real-world receiver non-idealities across 58 automated test cases. |

---

## 2. Key Architectural Principles

1. **No Raw Signals to LLM**: Time-domain samples and complex IQ arrays are processed entirely via DSP tools. The agent interacts strictly with concise summary metrics (PSD peaks, SNR, bandwidth, cumulants, EVM, decoded bit/audio text, CRC status).
2. **Native GNU Radio Reuse**: All signal generation, impairment modeling, filtering, synchronization, equalization, demodulation, preamble detection, and channel coding leverage native GNU Radio blocks (`gr.top_block`, `filter`, `analog`, `digital`, `fec`).
3. **Native SigMF Inter-Tool Standard**: All intermediate data, wideband captures, and channel extractions exchange data using standard **SigMF** format (`.sigmf-data` + `.sigmf-meta`).
4. **Executable Flowgraph Generation**: The agent's final goal for any new or unknown signal is to generate an executable GNU Radio Python flowgraph (`gr.top_block`) or `.grc` file.
5. **Skills Housing Tools**: All executable tool scripts are housed directly inside their respective skill directories under `.agents/skills/<skill_name>/scripts/`.

---

## 3. Directory Structure & Module Breakdown

```
gr-playground/
├── gr_playground/                 # Core Python Package & GNU Radio Wrappers
│   ├── simulator/                 # PILLAR I: Channel & Signal Simulator
│   │   ├── channel_simulator.py   # Unified channel simulation flowgraph (gr.top_block)
│   │   ├── sources.py             # Sine, square, audio, PRBS, noise generators
│   │   ├── modulators.py          # Digital constellation, FM, AM, GFSK, OFDM modulators
│   │   ├── impairments.py         # AWGN, CFO, SRO, DC offset, multipath channel models
│   │   ├── framing.py             # Layer-2 Framing Encoders (HDLC, COBS, AX.25, CCSDS, 802.15.4, Length-Prefixed)
│   │   ├── channel_coding.py      # FEC Encoders (Convolutional K=7, Hamming 7/4, Repetition, Reed-Solomon)
│   │   └── sigmf_writer.py        # SigMF v1.0 dataset exporter
│   ├── dsp/                       # PILLAR II: Modular Signal Processing Library
│   │   ├── channelizer.py         # Wideband spectrum scanning & DDC channel extraction
│   │   ├── spectrum.py            # Welch PSD, M2M4 SNR, occupied BW, peak tone detection
│   │   ├── filtering.py           # Gram-Schmidt I/Q balance, DC blocker, AGC2
│   │   ├── modulation_id.py       # Higher-order cumulants (C20-C63) & AMC decision tree
│   │   ├── synchronization.py     # Costas loop carrier phase lock & Gardner symbol sync
│   │   ├── demodulation.py        # AM/FM audio & digital PSK/QAM/GMSK bit slicing
│   │   ├── equalization.py        # Layer-1 Equalizers (Zero-Forcing, MMSE, Adaptive LMS, CMA)
│   │   ├── packet_detection.py    # Layer-1 Preamble Detectors (Barker 7/11/13, Zadoff-Chu, Schmidl-Cox)
│   │   ├── gmsk.py                # Native GNU Radio GMSK/MSK Demodulator & Preamble Detector (BT_LE, AIS, GSM)
│   │   ├── channel_decoding.py    # Layer-2 FEC Decoders (Viterbi K=7, Hamming 7/4, Repetition, RS(15,11))
│   │   ├── l2_framing_id.py       # Layer-2 Framing Identifier & Message Extractor (Bit shift & order search)
│   │   └── flowgraph_builder.py   # Standalone top_block generator & In-Skill GRC validator
│   └── utils/                     # SigMF I/O & Summary Helpers
│       ├── sigmf_io.py            # SigMF & raw .cu8 fallback auto-detector
│       └── summary.py             # Formatted Markdown report formatters
├── .agents/skills/                # Agent Skill Definitions & Executable Tool Scripts
│   ├── analyze-rf-signal/         # Top-level RF signal analysis orchestrator skill
│   │   ├── SKILL.md
│   │   └── scripts/analyze_rf_pipeline.py
│   ├── generate-test-signal/      # Test signal generator skill
│   │   ├── SKILL.md
│   │   └── scripts/generate_test_signal.py
│   ├── signal-analysis/           # Spectrum analysis sub-skill
│   │   ├── SKILL.md
│   │   └── scripts/analyze_signal.py
│   ├── signal-cleanup/            # DC blocker & IQ balancing sub-skill
│   │   ├── SKILL.md
│   │   └── scripts/cleanup_signal.py
│   ├── modulation-recognition/     # AMC higher-order cumulant classification sub-skill
│   │   ├── SKILL.md
│   │   └── scripts/classify_modulation.py
│   ├── signal-synchronization/    # CFO & symbol clock sync sub-skill
│   │   ├── SKILL.md
│   │   └── scripts/synchronize_signal.py
│   ├── signal-demodulation/       # Demodulation sub-skill (AM, FM, PSK, QAM, GMSK, MSK)
│   │   ├── SKILL.md
│   │   └── scripts/demodulate_signal.py
│   ├── channel-equalization/      # L1 Multipath Equalization receiver skill
│   │   ├── SKILL.md
│   │   └── scripts/equalize_channel.py
│   ├── preamble-detection/        # L1 Preamble Cross-Correlation receiver skill (Barker, ZC, Schmidl-Cox, BT_LE, AIS, GSM)
│   │   ├── SKILL.md
│   │   └── scripts/detect_preamble.py
│   ├── channel-decoding/          # L2 FEC Channel Decoding receiver skill
│   │   ├── SKILL.md
│   │   └── scripts/decode_channel_code.py
│   ├── l2_framing_identification/ # L2 Framing Identification & Payload Extraction receiver skill
│   │   ├── SKILL.md
│   │   └── scripts/identify_l2_framing.py
│   ├── build-gnuradio-flowgraph/  # Flowgraph script & GRC diagram builder skill
│   │   ├── SKILL.md
│   │   └── scripts/build_gnuradio_flowgraph.py
│   └── rtlsdr-hardware-capture/   # RTL-SDR dongle hardware capture skill
│       ├── SKILL.md
│       └── scripts/
│           ├── capture_rtlsdr.py
│           └── localize_and_extract.py
├── grc/                           # GRC Flowgraphs
│   └── rtlsdr_wideband_frontend.grc # Live hardware & wideband DDC GUI
└── tests/                         # PILLAR III: Automated Testcases & Benchmark Suite (58 Tests)
    ├── assets/audio/              # Speech audio reference generators & sample files
    ├── benchmarks/                # Verification Benchmark Suite
    │   ├── multicarrier_benchmark_suite.py # Multicarrier parameter sweeps & report runner
    │   └── skill_verification_suite.py     # Progressive impairment benchmark runner
    ├── conftest.py                # Pytest layer ordering & base-layer fail-fast hooks
    ├── test_simulator.py          # Layer 1: Foundation simulator & SigMF I/O
    ├── test_dsp.py                # Layer 2: Core DSP & analytics
    ├── test_wideband.py           # Layer 3: Wideband scanning & DDC
    ├── test_ask_modulation.py     # Layer 4: Modulation ID edge cases
    ├── test_dsp_flaws_and_fixes.py # Layer 5: DSP regression fixes
    ├── test_multicarrier.py       # Layer 6: Multicarrier parameter sweeps
    ├── test_realworld_receiver_impairments.py # Layer 7: Receiver channel suite
    ├── test_skill_verification.py # Layer 8: Impairment benchmark
    ├── test_analyze_rf_signal_skill.py # Layer 9: Top-level skill pipeline
    ├── test_equalization.py       # Layer 10: L1 Multipath Channel Equalization tests
    ├── test_l2_framing_and_channel_coding.py # Layer 11: L2 Framing & FEC Channel Coding tests
    ├── test_packet_detection.py   # Layer 12: L1 Preamble Cross-Correlation & Burst Detection tests
    └── test_gmsk_demodulation.py  # Layer 13: Native GMSK/MSK Demodulation & Protocol Preamble tests
```

---

## 4. Technical Specifications & Processing Workflows

### 4.1. Simulator Architecture (`gr_playground.simulator`)

```
[GRSources] ──> [GRModulators] ──> [GRImpairments] ──> [blocks.head] ──> [vector_sink_c]
 (sine/audio/    (QPSK/FM/AM/      (AWGN, CFO, SRO,    (num_samples)           │
  prbs/noise)     GFSK/GMSK/BPSK)   multipath, DC)                             ▼
                                                                        [SigMFWriter]
                                                                      (.sigmf-data/meta)
```

- **L2 Framing Encoders (`framing.py`)**: `encode_hdlc`, `encode_cobs`, `encode_ax25`, `encode_ccsds`, `encode_ieee802154`, `encode_generic_length_prefix`. Includes CRC-16 CCITT and CRC-32 IEEE 802.3 checksum computation.
- **FEC Channel Encoders (`channel_coding.py`)**: `encode_convolutional_k7` (Rate 1/2 Viterbi generator polynomials $G_1=121_8$, $G_2=117_8$), `encode_hamming_7_4`, `encode_repetition`, `encode_reed_solomon` (RS(15,11) Galois field $GF(2^4)$).

---

### 4.2. Modular DSP & Decoding Architecture (`gr_playground.dsp`)

#### 1. Layer-1 Equalization (`equalization.py`)
- **Zero-Forcing (ZF)**: Frequency-domain channel inversion $W(f) = 1 / H(f)$.
- **Minimum Mean Square Error (MMSE)**: Regularized channel inversion $W(f) = \frac{H^*(f)}{|H(f)|^2 + 1/\text{SNR}}$.
- **Adaptive LMS / DFE**: Decision-directed FIR filter with LMS gradient updates $w[n+1] = w[n] + \mu e[n] x^*[n]$.
- **GNU Radio CMA**: Constant Modulus Algorithm wrapper using `digital.cma_equalizer_cc`.

#### 2. Layer-1 Preamble Cross-Correlation (`packet_detection.py` & `gmsk.py`)
- **Barker Code Correlator**: Matched filter correlation for 7, 11, and 13-bit Barker codes ($R[n] = \frac{|\sum y s^*|^2}{\sum |y|^2 \sum |s|^2}$).
- **Zadoff-Chu (CAZAC)**: $x_u[n] = e^{-j \frac{\pi u n (n+1)}{N}}$ sequence cross-correlator.
- **Schmidl-Cox OFDM**: Sliding conjugate product metric $M(d) = \frac{|P(d)|^2}{R(d)^2}$ and coarse CFO estimator ($\Delta f = \frac{\text{angle}(P(d))}{\pi T_s}$).
- **GMSK Protocol Preambles**: Matched filter correlation for Bluetooth LE (`BT_LE` / `0xAA`/`0x55`), AIS Maritime (`AIS` / `0x555555`), and GSM training sequences.

#### 3. Layer-2 Framing Identification & FEC Decoding (`l2_framing_id.py` & `channel_decoding.py`)
- **Layer-2 Framing Identifier**: Bit-shift search (0–7 bits), polarity inversion tolerance, MSB (`big`) and LSB (`little`) bit orders, flag delimiter search, and FCS/CRC validation.
- **FEC Decoders**: `ChannelDecoder.decode` wrapping native GNU Radio `fec` flowgraphs for Viterbi $K=7$ Rate 1/2 decoding, Hamming(7,4) syndrome correction, Repetition majority voting, and Reed-Solomon RS(15,11) error correction.

---

### 4.3. Testcases & Verification Suite (`tests/`)

The test suite validates the playground across 13 layer-ordered test modules containing 58 automated test cases:

| Test Module | Scope & Objectives | Key Tests Included |
| :--- | :--- | :--- |
| `test_simulator.py` | Layer 1: Simulator Engine | `ChannelSimulatorFlowgraph` execution, SigMF v1.0 I/O writing/reading |
| `test_dsp.py` | Layer 2: Core DSP Primitives | Welch PSD, DC blocker, AGC2, Higher-Order Cumulants ($C_{20}\dots C_{63}$) |
| `test_wideband.py` | Layer 3: Wideband Scanning & DDC | Wideband channelization, FIR DDC sub-channel extraction, SigMF conversion |
| `test_ask_modulation.py` | Layer 4: Modulation Classification | AMC distinction between ASK, BPSK, QPSK, 8PSK, 16QAM, 64QAM, GFSK |
| `test_dsp_flaws_and_fixes.py` | Layer 5: Regression & Edge Cases | M2M4 SNR metric behavior, FIR filter bandwidth masks, CFO artifacts |
| `test_multicarrier.py` | Layer 6: Multicarrier Sweeps | OFDM & SC-FDMA parameter sweeps (FFT size, CP length, DFT precoding) |
| `test_realworld_receiver_impairments.py` | Layer 7: Receiver Channel Suite | $5\text{ dB}$ SNR, ACI, Rayleigh fading, 8-bit ADC quantization clipping |
| `test_skill_verification.py` | Layer 8: Benchmark Suite | Progressive impairment benchmark suite execution |
| `test_analyze_rf_signal_skill.py` | Layer 9: Top-Level Orchestrator | `analyze-rf-signal` skill end-to-end pipeline execution |
| `test_equalization.py` | Layer 10: L1 Multipath Equalization | ZF, MMSE, Adaptive LMS, GNU Radio CMA equalizers, end-to-end pipeline |
| `test_l2_framing_and_channel_coding.py` | Layer 11: L2 Framing & FEC Coding | HDLC, COBS, AX.25, CCSDS, 802.15.4 framing; Viterbi, Hamming, RS(15,11) decoding |
| `test_packet_detection.py` | Layer 12: L1 Preamble Cross-Correlation | Barker 11, Zadoff-Chu CAZAC, Schmidl-Cox OFDM timing & CFO estimation |
| `test_gmsk_demodulation.py` | Layer 13: GMSK/MSK Demod & Preambles | Native GNU Radio GMSK/MSK demodulation, BT_LE & AIS preamble detection |

---

## 5. Hardware Integration & GRC Schema Layer

- **Hardware Driver**: Native `gr-osmosdr` driver (`import osmosdr`, `osmosdr.source("rtl=0")`, GRC `id: rtlsdr_source`).
- **Linux Driver Management**: Automatic unloading of kernel TV tuner modules (`sudo rmmod dvb_usb_rtl28xxu dvb_usb_v2`) and USB permissions setup (`sudo chmod 666 /dev/bus/usb/...`).
- **GRC Schema Rules**: GRC 3.8/3.10 YAML format validation (`options.parameters`, `options.states`, `blocks.name`, `blocks.id`, `blocks.parameters`, `blocks.states`) enforced programmatically via `FlowgraphBuilder.validate_grc_flowgraph` with zero dummy blocks required.

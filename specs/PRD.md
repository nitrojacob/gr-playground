# Product Requirements Document (PRD): `gr-playground`

## 1. Product Overview & Vision

`gr-playground` is an open-source **GNU Radio Simulation Playground, DSP Library, and Testbed**. It is designed to provide autonomous LLM coding agents and SDR developers with an isolated, reproducible sandbox to:
1. **Simulate RF Channels & Impairments**: Synthesize complex IQ signals with precise physical layer impairments (AWGN, CFO, SRO, DC offset, multipath fading, phase noise, jammer interference).
2. **Execute Modular DSP Operations**: Perform wideband spectrum scanning, Digital Downconversion (DDC), spectral analysis, signal cleanup, automatic modulation classification (AMC), carrier/clock synchronization, and analog/digital demodulation.
3. **Run Verification Testcases & Benchmarks**: Evaluate DSP algorithms and agent skills against realistic real-world receiver non-idealities through automated test suites.

The **Playground** is built entirely on native GNU Radio blocks (`gnuradio.analog`, `gnuradio.digital`, `gnuradio.channels`, `gnuradio.filter`, `gnuradio.blocks`) and uses **SigMF** (`.sigmf-data` + `.sigmf-meta`) as the standard data inter-tool standard.

---

## 2. Core Pillars of the Playground

```
+---------------------------------------------------------------------------------------+
|                                    gr-playground                                      |
+---------------------------------------------------------------------------------------+
        │                                  │                                  │
        ▼                                  ▼                                  ▼
+-----------------------+      +-----------------------+      +-----------------------+
|  Pillar I: Simulator  |      | Pillar II: DSP Library|      | Pillar III: Testcases |
| (gr_playground.sim)   |      | (gr_playground.dsp)   |      | (tests/ & examples/)  |
+-----------------------+      +-----------------------+      +-----------------------+
| • Signal Sources      |      | • Spectrum Analysis   |      | • Real-World Receiver |
| • Modulators (Analog/ |      | • Wideband DDC        |      |   Impairments (5dB    |
|   Digital)            |      | • Signal Cleanup      |      |   SNR, ACI, Fading)   |
| • Impairment Chain    |      | • Modulation ID (C40) |      | • DSP Edge-Case Fixes |
| • SigMF Dataset Writer|      | • Sync & Demod        |      | • Skill Verification  |
|                       |      | • In-Skill GRC Check  |      |   Benchmark Suite     |
+-----------------------+      +-----------------------+      +-----------------------+
```

---

## 3. Target Personas & Use Cases

1. **AI Agent Developers**: Require a deterministic GNU Radio environment where agents can generate test datasets, analyze unknown spectrums, classify modulations, and generate executable Python `top_block` scripts.
2. **SDR & DSP Engineers**: Need modular Python DSP libraries for fast signal analysis, DDC channelization, and modulation recognition without manually building GRC flowgraphs.
3. **Algorithmic Researchers**: Need a reproducible simulation testbed to bench test receiver algorithms against progressive physical layer impairments.

---

## 4. Pillar Specifications & Requirements

### 4.1. Pillar I: The Channel & Signal Simulator (`gr_playground.simulator`)

The simulator provides automated signal generation and physical layer impairment modeling.

- **PRD-SIM-1 (Sources)**: Support continuous complex/float signal sources: `sine` (single-tone), `square` (harmonic-rich), `audio` (WAV speech/music loop), `prbs` (pseudo-random bit sequences), and `noise` (Gaussian noise).
- **PRD-SIM-2 (Modulation Schemes)**: Modulate input sources into analog and digital schemes:
  - Digital Single-Carrier Constellations: `BPSK`, `QPSK`, `8PSK`, `16QAM`, `64QAM`, `256QAM`.
  - Continuous Phase / Analog: `FM` (quadrature frequency modulation), `AM` (DSB/SSB amplitude modulation), `GFSK`, `BFSK`.
  - Multicarrier Schemes: `OFDM` (Orthogonal Frequency Division Multiplexing) and `SC-FDMA` (Single Carrier Frequency Division Multiple Access with $M$-point DFT precoding).
- **PRD-SIM-3 (Impairment Pipeline)**: Apply chainable physical channel impairments via native GNU Radio blocks:
  - Additive White Gaussian Noise (AWGN) specified in `snr_db` ($0\text{ to }30\text{ dB}$).
  - Carrier Frequency Offset (`cfo_hz`) & Doppler phase rotation.
  - Sampling Rate Offset (`sro_ppm`).
  - Direct Conversion DC Offset (`dc_offset` $I/Q$ tuple).
  - Quadrature I/Q Amplitude & Phase Imbalance (`mag_imbalance_db`, `phase_imbalance_deg`).
  - Frequency Selective Multipath Fading (`multipath_taps` array for Rayleigh/Rician channels).
- **PRD-SIM-4 (SigMF Dataset Export)**: Automatically export generated impaired signals to SigMF v1.0 specification files (`.sigmf-data` + `.sigmf-meta`).

### 4.2. Pillar II: The Modular DSP Libraries (`gr_playground.dsp` & `gr_playground.utils`)

The DSP library provides decoupled, reusable modules for inspecting and processing signals.

- **PRD-DSP-1 (Wideband Spectrum Scanning & DDC Channelizer)**:
  - Scan wideband spectrum captures ($2.4+\text{ MSPS}$) using Temporal Ensemble Welch PSD (`nperseg=32768`, $+9\text{ dB}$ processing gain).
  - Estimate adaptive rolling noise floor ($200\text{ kHz}$ median window) and local SNR prominence ($SNR_{\text{local}} = PSD_{\text{dB}} - NoiseFloor_{\text{dB}}$) to discover active sub-channels while rejecting $0\text{ Hz}$ LO leakage spikes.
  - Isolate sub-channels using native `filter.freq_xlating_fir_filter_ccc` (DDC), lowpass filter, decimate, AGC normalize, and export to SigMF.
- **PRD-DSP-2 (Spectrum Analysis & SNR Metrics)**:
  - Compute Welch Power Spectral Density (PSD), peak tone frequencies, occupied bandwidth (99% power), DC offset level, and overall SNR via M2M4 ratio.
- **PRD-DSP-3 (Signal Cleanup & Filtering)**:
  - Remove DC bias via `filter.dc_blocker_cc`, apply Gram-Schmidt I/Q imbalance compensation, and perform lowpass FIR filtering (`filter.fir_filter_ccc`).
- **PRD-DSP-4 (Automatic Modulation Classification)**:
  - Calculate Higher-Order Cumulants ($C_{20}, C_{21}, C_{40}, C_{42}$) and constellation statistics (phase variance, amplitude kurtosis) to classify modulations (`FM`, `AM`, `BPSK`, `QPSK`, `8PSK`, `16QAM`, `64QAM`, `256QAM`, `GFSK`, `OFDM`, `SC-FDMA`).
  - Squelch low SNR noise ($SNR < 1.5\text{ dB}$) to prevent false positive classifications on noise.
- **PRD-DSP-5 (Carrier & Clock Synchronization)**:
  - Perform Carrier Frequency Offset (CFO) recovery, Costas Loop phase locking (`digital.costas_loop_cc`), and Gardner symbol clock timing synchronization (`digital.symbol_sync_cc`).
- **PRD-DSP-6 (Demodulation)**:
  - Demodulate analog AM/FM into 16-bit PCM WAV audio files (`.wav`).
  - Slice digital modulations (BPSK, QPSK, GFSK) into bit/byte symbol strings.
- **PRD-DSP-7 (SigMF I/O & Fallback Auto-Detection)**:
  - Read standard SigMF files and provide fallback auto-conversion for raw 8-bit unsigned offset binary (`.cu8`) hardware captures `(x - 127.5)/127.5`.
- **PRD-DSP-8 (Flowgraph Builder & In-Skill GRC Validation)**:
  - Generate executable standalone GNU Radio Python top_block scripts (`gr.top_block`).
  - Include in-skill GRC YAML schema and block presence validation (`FlowgraphBuilder.validate_grc_flowgraph` / `build_gnuradio_flowgraph.py --validate_grc`) to programmatically verify that $0$ dummy or missing blocks exist upon creating or editing flowgraphs.

### 4.3. Pillar III: The Testcases & Benchmark Suite (`tests/` & `examples/`)

The test suite validates DSP algorithms and agent skills against real-world receiver non-idealities.

- **PRD-TEST-1 (Real-World Receiver Impairments Suite)**:
  - Test wideband scanning under $5\text{ dB}$ SNR, strong DC offset, and Adjacent Channel Interference (ACI).
  - Test DDC channel extraction under 3-path Rayleigh multipath fading and $+1.5\text{ kHz}$ CFO.
  - Test 8-bit RTL-SDR hardware ADC quantization noise and full-scale dynamic range clipping (`.cu8`).
- **PRD-TEST-2 (DSP Edge-Case Flaws & Fixes Suite)**:
  - Verify M2M4 SNR calculation behavior on constant-envelope signals and noise floor collapse.
  - Verify FIR filter transition band masks and uniform phase rotation CFO artifacts.
- **PRD-TEST-3 (Progressive Impairment Skill Verification Suite)**:
  - Run end-to-end benchmark suite across progressive SNR drops ($30\text{ dB} \to 0\text{ dB}$), CFO sweeps, and multipath channels to verify agent skill pass rates.
- **PRD-TEST-4 (Multicarrier Benchmark & Sub-Parameter Sweep Suite)**:
  - Perform sub-parameter sweeps over subcarrier spacing ($\Delta f$), subcarrier bandwidth ($N_{\text{used}}$), cyclic prefix ratios ($CP/N_{\text{fft}}$), and SC-FDMA DFT precoding PAPR reduction under configurable impairment profiles (`examples/multicarrier_benchmark_suite.py` & `tests/test_multicarrier.py`).
  - Modular runner design supports looping across progressively worse channel models.

---

## 5. Non-Functional Requirements

- **Performance**: Spectrum scanning of 72M samples (30s at 2.4 MSPS) must complete in under 5 seconds.
- **SigMF Compliance**: All datasets produced or consumed by the playground must comply with SigMF v1.0 specifications.
- **Zero Dummy Blocks**: In-skill GRC flowgraph validator must verify `b.is_dummy_block == False` for all blocks in generated flowgraphs.
- **Automated Test Pass Rate**: 100% pass rate across the automated test cases in `tests/`.

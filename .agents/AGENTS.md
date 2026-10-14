# Project Rules & Guidelines (`gr-playground`)

## GNU Radio Companion (.grc) Schema & Block Presence Guidelines
- When generating or editing GRC 3.8/3.10 `.grc` YAML files:
  - The `options:` section MUST contain both `parameters:` AND `states:` dictionaries.
  - Every block entry in `blocks:` MUST contain `name:` (instance variable name), `id:` (block definition key), `parameters:`, and `states:`.
  - Always validate GRC files programmatically using `FlowgraphBuilder.validate_grc_flowgraph` or `python3 .agents/skills/build-gnuradio-flowgraph/scripts/build_gnuradio_flowgraph.py --validate_grc <grc_file>` to verify that **all blocks are present** in the block library (`b.is_dummy_block == False`) before declaring a flowgraph ready.

## RTL-SDR & USB Hardware Access
- RTL-SDR hardware (`0bda:2838`) on Linux requires unloading the kernel DVB tuner module (`sudo rmmod dvb_usb_rtl28xxu dvb_usb_v2`) and setting USB permissions (`sudo chmod 666 /dev/bus/usb/...`).
- Inter-file exchange MUST remain strictly in native SigMF (`.sigmf-data` + `.sigmf-meta`). Raw `.cu8` / `.bin` data is handled via fallback auto-detection in `read_sigmf()`.

---

# Essential Project Context for AI Agents (`gr-playground`)

## 1. Mission & Concepts
`gr-playground` is an AI-first signal processing playground and benchmark testbed built on GNU Radio 3.10 and Python. Agents either iteratively develop DSP routines against simulated channel impairments or autonomously analyze live/recorded RF captures (scanning, channel extraction, AMC, synchronization, demodulation, framing/FEC decoding, and flowgraph synthesis).

See [README.md](../README.md) and [specs/PRD.md](../specs/PRD.md) for full project background and product requirements.

---

## 2. Key Architectural Rules for Agents
- **No Raw Signals to LLM**: Never dump raw IQ samples or time-domain arrays into context. Use DSP scripts and inspect extracted metrics (PSD peaks, SNR, EVM, CRC status, decoded payload).
- **Native GNU Radio Reuse**: Build signal paths using native GNU Radio 3.10 blocks (`gr.top_block`, `filter`, `analog`, `digital`, `fec`, `osmosdr`).
- **Strict SigMF Exchange**: All captures, intermediate signals, and channel extractions MUST use native SigMF (`.sigmf-data` + `.sigmf-meta`).
- **Target Outcome**: Synthesize an executable standalone GNU Radio script (`top_block.py`) or verified `.grc` flowgraph.
- **Skill Housing**: Runnable scripts for each agent capability live in `.agents/skills/<skill-name>/scripts/`.

---

## 3. Core Pillars & Documentation Pointers
Detailed specifications, mathematical formulations, and block diagrams are maintained in dedicated documents:

- **Pillar I: Signal & Channel Simulator (`gr_playground.simulator`)**: Synthesizes impaired IQ signals with AWGN, CFO, SRO, multipath fading, phase noise, L2 framing, and FEC coding.
  -> Reference: [specs/ARCHITECTURE.md (Simulator Architecture)](../specs/ARCHITECTURE.md#41-simulator-architecture)
- **Pillar II: Modular DSP Library (`gr_playground.dsp`, `gr_playground.utils`)**: Wideband scanning, DDC extraction, modular channel detection strategies, Gram-Schmidt cleanup, AMC cumulants & ML classifier, carrier/timing sync, L1 equalizers, preamble detectors, and L2 framing/FEC decoders.
  -> Reference: [specs/ARCHITECTURE.md (DSP Architecture)](../specs/ARCHITECTURE.md#42-modular-dsp--decoding-architecture), [specs/channel_identification.md](../specs/channel_identification.md), [specs/modulation_classifier.md](../specs/modulation_classifier.md)
- **Pillar III: Test & Verification Suite (`tests/`, `benchmarks/`)**: 13 layer-ordered test modules (58 pytest cases) and zero-shot transfer evaluation on RadioML 2018.01A.
  -> Reference: [specs/ARCHITECTURE.md (Verification Suite)](../specs/ARCHITECTURE.md#43-testcases--verification-suite)

---

## 4. Key Agent Commands
- **Run verification test suite**:
  `HOME=/tmp ./run_all_tests.sh` (or `HOME=/tmp ./run_all_tests.sh -v`)
- **Run end-to-end analysis pipeline**:
  `python3 .agents/skills/analyze-rf-signal/scripts/analyze_rf_pipeline.py <capture.sigmf-data>`
- **Validate a GRC flowgraph**:
  `python3 .agents/skills/build-gnuradio-flowgraph/scripts/build_gnuradio_flowgraph.py --validate_grc <flowgraph.grc>`
- **Individual sub-skills**:
  See [README.md (Step-by-Step Execution)](../README.md#step-by-step-execution-using-sub-skills-lower-level-components) for individual sub-skill CLI examples.


---
name: build-gnuradio-flowgraph
description: Compose and generate executable standalone GNU Radio Python top_block scripts (.py) or GRC block diagram files (.grc) using native GNU Radio DSP blocks, with built-in GRC schema and block presence validation.
---

# Build GNU Radio Flowgraph Skill 🛠️

Use this skill when you need to construct a complete, standalone GNU Radio Python flowgraph (`gr.top_block`) or `.grc` file to process, clean, synchronize, and demodulate a new or custom signal, or to validate GRC YAML schema and block presence.

## Instructions

1. **Execute Flowgraph Generator Tool**:
   ```bash
   PYTHONPATH=/usr/lib/python3/dist-packages:./ python3 .agents/skills/build-gnuradio-flowgraph/scripts/build_gnuradio_flowgraph.py --input <input_sigmf_file> --output_script <output_script.py> --ops "dc_block,lowpass_filter,agc,costas_loop,symbol_sync"
   ```

2. **In-Skill GRC Flowgraph Validation**:
   When creating or editing `.grc` flowgraph files, validate the GRC YAML schema and enforce $0$ missing/dummy blocks directly via the skill validator:
   ```bash
   PYTHONPATH=/usr/lib/python3/dist-packages:./ python3 .agents/skills/build-gnuradio-flowgraph/scripts/build_gnuradio_flowgraph.py --validate_grc <path_to_grc_file>
   ```

3. **Generated Python Top Block Architecture**:
   The output script imports `from gnuradio import gr, blocks, filter, analog, digital` and wires native blocks into an executable pipeline:
   - `blocks.file_source` $\rightarrow$ `filter.dc_blocker_cc` $\rightarrow$ `filter.fir_filter_ccc` $\rightarrow$ `analog.agc2_cc` $\rightarrow$ `digital.costas_loop_cc` $\rightarrow$ `digital.symbol_sync_cc` $\rightarrow$ `blocks.file_sink`.

4. **Execute Generated Script**:
   Run the generated GNU Radio top block script directly:
   ```bash
   PYTHONPATH=/usr/lib/python3/dist-packages python3 <output_script.py> <input_sigmf> <output_iq>
   ```

5. **GRC Schema & RTL-SDR Hardware Reference**:
   For detailed GRC 3.8/3.10 YAML schema rules, template block IDs, and RTL-SDR Linux hardware access setup, refer to [references/grc_and_rtlsdr_guide.md](file:///.agents/skills/build-gnuradio-flowgraph/references/grc_and_rtlsdr_guide.md).

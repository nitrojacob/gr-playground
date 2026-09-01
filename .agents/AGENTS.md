# Project Rules & Guidelines (`gr-playground`)

## GNU Radio Companion (.grc) Schema & Block Presence Guidelines
- When generating or editing GRC 3.8/3.10 `.grc` YAML files:
  - The `options:` section MUST contain both `parameters:` AND `states:` dictionaries.
  - Every block entry in `blocks:` MUST contain `name:` (instance variable name), `id:` (block definition key), `parameters:`, and `states:`.
  - Always validate GRC files programmatically using `test_grc_flowgraph_block_presence_and_schema` to verify that **all blocks are present** in the block library (`b.is_dummy_block == False`) before declaring a flowgraph ready.

## RTL-SDR & USB Hardware Access
- RTL-SDR hardware (`0bda:2838`) on Linux requires unloading the kernel DVB tuner module (`sudo rmmod dvb_usb_rtl28xxu dvb_usb_v2`) and setting USB permissions (`sudo chmod 666 /dev/bus/usb/...`).
- Inter-file exchange MUST remain strictly in native SigMF (`.sigmf-data` + `.sigmf-meta`). Raw `.cu8` / `.bin` data is handled via fallback auto-detection in `read_sigmf()`.

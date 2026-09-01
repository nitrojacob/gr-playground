# Reference: GNU Radio Companion (GRC) Schema & RTL-SDR Hardware Setup Guide 📡

This document captures essential knowledge for writing valid GNU Radio Companion (`.grc`) flowgraphs, programmatically validating block presence, troubleshooting RTL-SDR Linux hardware access, and executing wideband digital downconversion.

---

## 1. GNU Radio Companion (.grc) YAML Schema & Block Presence Rules 🛠️

GNU Radio 3.8 and 3.10 use YAML for `.grc` flowgraphs. To avoid `TypeError: Block.import_data() missing 1 required positional argument: 'states'` and missing block errors:

### Mandated YAML Hierarchy

1. **`options` Block**: Must include BOTH `parameters:` AND `states:` dictionaries:
   ```yaml
   options:
     parameters:
       id: rtlsdr_wideband_frontend
       title: RTL-SDR Wideband Capture & Channelizer Frontend
       generate_options: qt_gui
       output_language: python
     states:
       bus_sink: false
       bus_source: false
       bus_structure: null
       coordinate: [8, 8]
       rotation: 0
       state: enabled
   ```

2. **`blocks` List Items**: Every block item in `blocks:` MUST define `name:` (instance variable name), `id:` (GRC template block definition key), `parameters:`, and `states:`:
   ```yaml
   blocks:
   - name: samp_rate
     id: variable
     parameters:
       value: '2400000'
     states:
       bus_sink: false
       bus_source: false
       bus_structure: null
       coordinate: [180, 12]
       rotation: 0
       state: enabled
   ```

3. **Template Block IDs**:
   - Always use GRC's exact template block IDs (e.g. `freq_xlating_fir_filter_xxx` with `parameters: type: ccc`, `analog_agc2_xx` with `parameters: type: complex`).

---

## 2. Programmatic GRC Flowgraph Block Presence Validation 🐍

Before marking a `.grc` flowgraph ready, run this dynamic block presence test to guarantee that **every block in the flowgraph is recognized and present in the block library** (ensuring zero dummy/missing blocks):

```python
import os, gi
os.environ['GRC_BLOCKS_PATH'] = '/usr/share/gnuradio/grc/blocks'
gi.require_version('Gtk', '3.0')
from gnuradio.grc.core.platform import Platform

platform = Platform(version='3.10.9.2', name='GNU Radio Companion')
platform.build_library()

# Load and validate GRC file
fg_data = platform.parse_flow_graph("/path/to/flowgraph.grc")
fg = platform.make_flow_graph()
fg.import_data(fg_data)

# Dynamically assert that NO block in the flowgraph is a missing/dummy block
missing_blocks = [b.name for b in fg.blocks if b.is_dummy_block]
assert len(missing_blocks) == 0, f"Found missing/dummy blocks in flowgraph: {missing_blocks}"
print("GRC Flowgraph Validated: ALL blocks are present and ready!")
```

---

## 3. RTL-SDR Hardware Setup & Linux Driver Resolution 🔌

When an RTL-SDR USB dongle (USB Vendor/Product `0bda:2838`) is plugged into a Linux system, the kernel automatically claims the interface with its default DTV tuner module (`dvb_usb_rtl28xxu`). This causes userland SDR tools (`librtlsdr`, `SoapySDR`, GNU Radio) to fail with:
`[ERROR] rtlsdr_get_device_usb_strings(0) failed` / `No devices found!`

### Resolution Steps

1. **Unload Kernel TV Module**:
   ```bash
   sudo rmmod dvb_usb_rtl28xxu dvb_usb_v2
   ```
2. **Grant USB Permissions**:
   ```bash
   sudo chmod 666 /dev/bus/usb/<bus_id>/<device_id>
   ```
3. **Permanent Fix (Blacklist Module)**:
   ```bash
   echo 'blacklist dvb_usb_rtl28xxu' | sudo tee /etc/modprobe.d/blacklist-rtlsdr.conf
   ```

---

## 4. Wideband Channel Extraction & Native SigMF Workflow 📊

Wideband SDR captures ($2.4+\text{ MSPS}$) contain multiple sub-channels. Processing workflow:

1. **Format Ingestion**: Primary format is `.sigmf-data` (`cf32_le`). Fallback auto-detection converts raw command-line `.cu8` (8-bit unsigned offset binary: `(x - 127.5)/127.5`) on the fly in `read_sigmf()`.
2. **Spectrum Scanning**: Use `scan_wideband_channels(samples, sample_rate)` to discover channel center frequency offsets ($f_{\text{offset}}$).
3. **Digital Downconversion (DDC)**: Use `extract_channel_flowgraph(samples, sample_rate, freq_offset_hz, target_bw_hz, decimation)` with native `filter.freq_xlating_fir_filter_ccc` to shift target sub-channels to $0\text{ Hz}$ baseband, lowpass filter, decimate, and save as native SigMF.

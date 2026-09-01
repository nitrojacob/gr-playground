"""
SigMF (Signal Metadata Format) Header & Complex64 File I/O Utilities
"""

import os
import json
import numpy as np

def write_sigmf(filepath, samples, sample_rate=1e6, center_freq=0.0, description="gr-playground signal", annotations=None):
    """
    Write complex64 samples to .sigmf-data binary file and generate .sigmf-meta JSON header.
    """
    base, ext = os.path.splitext(filepath)
    data_path = base + ".sigmf-data" if ext != ".sigmf-data" else filepath
    meta_path = base + ".sigmf-meta" if ext != ".sigmf-data" else base + ".sigmf-meta"

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    # Write binary complex64 (interleaved float32 I and Q)
    samples = np.asarray(samples, dtype=np.complex64)
    samples.tofile(data_path)

    metadata = {
        "global": {
            "core:datatype": "cf32_le",
            "core:sample_rate": float(sample_rate),
            "core:version": "1.0.0",
            "core:description": description,
            "core:author": "gr-playground channel simulator",
        },
        "captures": [
            {
                "core:sample_start": 0,
                "core:frequency": float(center_freq),
                "core:datetime": "2026-08-31T00:00:00Z"
            }
        ],
        "annotations": annotations or []
    }

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return data_path, meta_path

def read_sigmf(filepath, default_sample_rate=2.4e6, default_center_freq=100.0e6):
    """
    Read SigMF dataset (.sigmf-data or .sigmf-meta path). Returns (complex64_array, meta_dict).
    Supports fallback auto-detection for raw RTL-SDR (.cu8, .raw, .bin, .cs16) files without SigMF headers.
    """
    base, ext = os.path.splitext(filepath)
    ext_lower = ext.lower()

    meta_path = base + ".sigmf-meta" if ext_lower in [".sigmf-data", ".sigmf-meta"] else filepath + ".sigmf-meta"
    data_path = filepath

    if ext_lower == ".sigmf-meta":
        data_path = base + ".sigmf-data"
    elif ext_lower == ".sigmf-data":
        data_path = filepath
    elif not os.path.exists(filepath) and os.path.exists(filepath + ".sigmf-data"):
        data_path = filepath + ".sigmf-data"

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"IQ data file not found: {data_path}")

    meta = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
        except Exception:
            meta = {}

    datatype = meta.get("global", {}).get("core:datatype", "").lower()

    # Fallback auto-detection for cu8 (RTL-SDR 8-bit unsigned offset binary)
    if ext_lower in [".cu8", ".raw", ".bin"] or datatype in ["cu8_le", "u8", "ri8"]:
        raw_bytes = np.fromfile(data_path, dtype=np.uint8)
        if len(raw_bytes) % 2 != 0:
            raw_bytes = raw_bytes[:len(raw_bytes) - 1]
        
        i_raw = raw_bytes[0::2].astype(np.float32)
        q_raw = raw_bytes[1::2].astype(np.float32)
        
        # Convert uint8 (0 to 255, center 127.5) to float32 [-1.0, 1.0]
        i_float = (i_raw - 127.5) / 127.5
        q_float = (q_raw - 127.5) / 127.5
        samples = (i_float + 1j * q_float).astype(np.complex64)

        if "global" not in meta:
            meta["global"] = {
                "core:datatype": "cf32_le",
                "core:sample_rate": float(default_sample_rate),
                "core:description": "Auto-converted RTL-SDR raw cu8 dataset",
            }
        if "captures" not in meta or not meta["captures"]:
            meta["captures"] = [{"core:sample_start": 0, "core:frequency": float(default_center_freq)}]

    elif ext_lower in [".cs16"] or datatype in ["cs16_le", "s16"]:
        raw_i16 = np.fromfile(data_path, dtype=np.int16)
        if len(raw_i16) % 2 != 0:
            raw_i16 = raw_i16[:len(raw_i16) - 1]
        i_float = raw_i16[0::2].astype(np.float32) / 32768.0
        q_float = raw_i16[1::2].astype(np.float32) / 32768.0
        samples = (i_float + 1j * q_float).astype(np.complex64)

        if "global" not in meta:
            meta["global"] = {
                "core:datatype": "cf32_le",
                "core:sample_rate": float(default_sample_rate),
                "core:description": "Auto-converted cs16 dataset",
            }
        if "captures" not in meta or not meta["captures"]:
            meta["captures"] = [{"core:sample_start": 0, "core:frequency": float(default_center_freq)}]

    else:
        # Standard complex64 (cf32_le) format
        samples = np.fromfile(data_path, dtype=np.complex64)

    return samples, meta

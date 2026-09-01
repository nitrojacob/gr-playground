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

def read_sigmf(filepath):
    """
    Read SigMF dataset (.sigmf-data or .sigmf-meta path). Returns (complex64_array, meta_dict).
    """
    base, ext = os.path.splitext(filepath)
    if ext == ".sigmf-meta":
        data_path = base + ".sigmf-data"
        meta_path = filepath
    elif ext == ".sigmf-data":
        data_path = filepath
        meta_path = base + ".sigmf-meta"
    else:
        data_path = filepath + ".sigmf-data" if os.path.exists(filepath + ".sigmf-data") else filepath
        meta_path = filepath + ".sigmf-meta" if os.path.exists(filepath + ".sigmf-meta") else base + ".sigmf-meta"

    # Read binary IQ
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"IQ data file not found: {data_path}")
    
    samples = np.fromfile(data_path, dtype=np.complex64)

    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)

    return samples, meta

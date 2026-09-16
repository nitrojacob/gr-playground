"""
Dataset Generation Script for Automatic Modulation Classification (AMC).
Synthesizes IQ frames across 16 modulation schemes under variable sample rates,
durations, SNRs, CFO, phase noise, and channel impairments.

Discards raw IQ immediately and persists compact feature matrices:
- `data/amc_stage1_features.npz` (Stage 1 frame features: [N_frames, 26])
- `data/amc_stage2_features.npz` (Stage 2 sequence features: [N_seqs, 21])
"""

import os
import argparse
import numpy as np
from scipy import signal

from gr_playground.dsp.amc.features import (
    extract_frame_features,
    extract_sequence_features
)
from gr_playground.dsp.amc.common import squelch_check

MODULATION_CLASSES = [
    "AM", "FM", "BPSK", "GFSK", "QPSK", "8PSK",
    "16QAM", "64QAM", "256QAM", "ASK", "16APSK", "32APSK",
    "OQPSK", "OFDM", "SC-FDMA", "Noise"
]

CLASS_MAP = {name: idx for idx, name in enumerate(MODULATION_CLASSES)}

def generate_raw_iq_frame(mod_type: str, num_samples: int = 2048, sps: int = 4,
                          snr_db: float = 20.0, cfo_rel: float = 0.0,
                          phase_noise_std: float = 0.0, mag_imbal_db: float = 0.0,
                          phase_imbal_deg: float = 0.0) -> np.ndarray:
    """
    Synthesize a single raw complex64 IQ frame (length num_samples) for a target modulation scheme.
    """
    num_symbols = int(np.ceil(num_samples / sps)) + 10

    if mod_type == "Noise":
        iq = (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)) / np.sqrt(2.0)
        return iq.astype(np.complex64)

    elif mod_type in ["AM", "FM"]:
        t = np.arange(num_samples)
        # Message signal (mix of single-tone sinusoidal FM and multi-tone audio)
        if np.random.rand() > 0.5:
            f_m = np.random.uniform(0.001, 0.015)
            audio = np.sin(2 * np.pi * f_m * t)
        else:
            audio = 0.6 * np.sin(2 * np.pi * 0.005 * t) + 0.4 * np.sin(2 * np.pi * 0.012 * t)

        if mod_type == "AM":
            # AM DSB-FC: s(t) = (1 + m*a(t)) * exp(j0)
            m_depth = np.random.uniform(0.5, 0.95)
            iq = (1.0 + m_depth * audio) + 1j * 0.0
        else: # FM
            # FM: s(t) = exp(j * 2pi * dev * int(a(t)))
            freq_dev = np.random.uniform(0.005, 0.25)
            phase = 2 * np.pi * freq_dev * np.cumsum(audio)
            iq = np.exp(1j * phase)

    elif mod_type == "ASK":
        bits = np.random.choice([0, 1], size=num_symbols)
        symbols = bits.astype(np.float32)
        symbols = np.repeat(symbols, sps)[:num_samples]
        iq = symbols + 1j * 0.0

    elif mod_type == "BPSK":
        bits = np.random.choice([0, 1], size=num_symbols)
        symbols = 2.0 * bits - 1.0
        iq = np.repeat(symbols, sps)[:num_samples] + 1j * 0.0

    elif mod_type == "QPSK":
        bits_i = np.random.choice([0, 1], size=num_symbols)
        bits_q = np.random.choice([0, 1], size=num_symbols)
        sym_i = (2.0 * bits_i - 1.0) / np.sqrt(2.0)
        sym_q = (2.0 * bits_q - 1.0) / np.sqrt(2.0)
        iq = np.repeat(sym_i + 1j * sym_q, sps)[:num_samples]

    elif mod_type == "OQPSK":
        bits_i = np.random.choice([0, 1], size=num_symbols)
        bits_q = np.random.choice([0, 1], size=num_symbols)
        sym_i = (2.0 * bits_i - 1.0) / np.sqrt(2.0)
        sym_q = (2.0 * bits_q - 1.0) / np.sqrt(2.0)
        iq_i = np.repeat(sym_i, sps)[:num_samples]
        iq_q = np.repeat(sym_q, sps)[:num_samples]
        # Shift Q channel by half symbol period (sps // 2)
        offset = max(1, sps // 2)
        iq_q = np.roll(iq_q, offset)
        iq = iq_i + 1j * iq_q

    elif mod_type == "8PSK":
        phases = (2 * np.pi / 8.0) * np.random.randint(0, 8, size=num_symbols)
        symbols = np.exp(1j * phases)
        iq = np.repeat(symbols, sps)[:num_samples]

    elif mod_type in ["16QAM", "64QAM", "256QAM"]:
        m_order = 16 if mod_type == "16QAM" else (64 if mod_type == "64QAM" else 256)
        side = int(np.sqrt(m_order))
        grid = np.linspace(-(side - 1), side - 1, side)
        i_syms = np.random.choice(grid, size=num_symbols)
        q_syms = np.random.choice(grid, size=num_symbols)
        symbols = i_syms + 1j * q_syms
        p_avg = np.mean(np.abs(symbols)**2)
        symbols = symbols / np.sqrt(p_avg)
        iq = np.repeat(symbols, sps)[:num_samples]

    elif mod_type == "16APSK":
        # 16APSK: Ring 1 (4 points r1), Ring 2 (12 points r2)
        r1, r2 = 1.0, 2.6
        n1, n2 = 4, 12
        p1 = np.exp(1j * (2 * np.pi / n1 * np.arange(n1) + np.pi/4)) * r1
        p2 = np.exp(1j * (2 * np.pi / n2 * np.arange(n2))) * r2
        constellation = np.concatenate([p1, p2])
        constellation /= np.sqrt(np.mean(np.abs(constellation)**2))
        syms = np.random.choice(constellation, size=num_symbols)
        iq = np.repeat(syms, sps)[:num_samples]

    elif mod_type == "32APSK":
        # 32APSK: Ring 1 (4 points r1), Ring 2 (12 points r2), Ring 3 (16 points r3)
        r1, r2, r3 = 1.0, 2.54, 4.33
        n1, n2, n3 = 4, 12, 16
        p1 = np.exp(1j * (2 * np.pi / n1 * np.arange(n1) + np.pi/4)) * r1
        p2 = np.exp(1j * (2 * np.pi / n2 * np.arange(n2))) * r2
        p3 = np.exp(1j * (2 * np.pi / n3 * np.arange(n3) + np.pi/16)) * r3
        constellation = np.concatenate([p1, p2, p3])
        constellation /= np.sqrt(np.mean(np.abs(constellation)**2))
        syms = np.random.choice(constellation, size=num_symbols)
        iq = np.repeat(syms, sps)[:num_samples]

    elif mod_type == "GFSK":
        bits = np.random.choice([0, 1], size=num_symbols)
        nrz = 2.0 * bits - 1.0
        upsampled = np.repeat(nrz, sps)
        # Gaussian pulse filter
        t_g = np.linspace(-2, 2, 4 * sps)
        h_g = np.exp(-t_g**2 / (2 * 0.5**2))
        h_g /= np.sum(h_g)
        filtered = np.convolve(upsampled, h_g, mode='same')[:num_samples]
        phase = np.pi * 0.5 * np.cumsum(filtered) / sps
        iq = np.exp(1j * phase)

    elif mod_type in ["OFDM", "SC-FDMA"]:
        n_fft = 64
        n_used = 48
        cp_len = 16
        n_syms = int(np.ceil(num_samples / (n_fft + cp_len)))
        ofdm_syms = []
        for _ in range(n_syms):
            subcarriers = np.zeros(n_fft, dtype=np.complex64)
            data_syms = (np.random.choice([-1, 1], size=n_used) + 1j * np.random.choice([-1, 1], size=n_used)) / np.sqrt(2)
            if mod_type == "SC-FDMA":
                # DFT precoding across used subcarriers
                data_syms = np.fft.fft(data_syms) / np.sqrt(n_used)
            subcarriers[1:n_used+1] = data_syms
            time_domain = np.fft.ifft(subcarriers) * np.sqrt(n_fft)
            cp = time_domain[-cp_len:]
            sym_with_cp = np.concatenate([cp, time_domain])
            ofdm_syms.append(sym_with_cp)
        iq = np.concatenate(ofdm_syms)[:num_samples]

    else:
        iq = (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)) / np.sqrt(2.0)

    # Apply Impairments
    # 1. Power normalization
    p_sig = np.mean(np.abs(iq)**2)
    if p_sig > 0:
        iq = iq / np.sqrt(p_sig)

    # 2. CFO
    if cfo_rel != 0.0:
        t = np.arange(num_samples)
        iq = iq * np.exp(1j * 2 * np.pi * cfo_rel * t)

    # 3. Phase Noise
    if phase_noise_std > 0.0:
        pn = np.random.normal(0.0, phase_noise_std, size=num_samples)
        iq = iq * np.exp(1j * pn)

    # 4. IQ Amplitude / Phase Imbalance
    if mag_imbal_db != 0.0 or phase_imbal_deg != 0.0:
        gain = 10.0**(mag_imbal_db / 20.0)
        phase_rad = np.radians(phase_imbal_deg)
        i_chan = iq.real * gain * np.cos(phase_rad / 2.0) - iq.imag * gain * np.sin(phase_rad / 2.0)
        q_chan = -iq.real * (1.0 / gain) * np.sin(phase_rad / 2.0) + iq.imag * (1.0 / gain) * np.cos(phase_rad / 2.0)
        iq = i_chan + 1j * q_chan

    # 5. AWGN Noise Addition
    if snr_db < 100.0:
        snr_lin = 10.0**(snr_db / 10.0)
        noise_pwr = 1.0 / max(snr_lin, 1e-4)
        noise = (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)) * np.sqrt(noise_pwr / 2.0)
        iq = iq + noise

    return iq.astype(np.complex64)

def build_dataset(num_stage1_frames: int = 80000, num_stage2_seqs: int = 8000,
                  output_dir: str = "data"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating Stage 1 Dataset ({num_stage1_frames} frame instances)...")

    num_classes = len(MODULATION_CLASSES)
    frames_per_class = num_stage1_frames // num_classes

    X1_list = []
    y1_list = []

    for cls_name in MODULATION_CLASSES:
        cls_idx = CLASS_MAP[cls_name]
        print(f"  Synthesizing {frames_per_class} frames for class {cls_name} (id {cls_idx})...")
        for _ in range(frames_per_class):
            # Sweep parameters
            snr_db = float(np.random.uniform(-10.0, 30.0))
            sps = int(np.random.choice([2, 4, 8, 16]))
            cfo_rel = float(np.random.uniform(-0.08, 0.08))
            phase_noise_std = float(np.random.uniform(0.0, 0.025))
            mag_imbal_db = float(np.random.uniform(0.0, 1.2))
            phase_imbal_deg = float(np.random.uniform(0.0, 8.0))

            frame = generate_raw_iq_frame(
                cls_name, num_samples=2048, sps=sps, snr_db=snr_db,
                cfo_rel=cfo_rel, phase_noise_std=phase_noise_std,
                mag_imbal_db=mag_imbal_db, phase_imbal_deg=phase_imbal_deg
            )

            feats = extract_frame_features(frame)
            X1_list.append(feats)
            y1_list.append(cls_idx)

    X1 = np.vstack(X1_list).astype(np.float32)
    y1 = np.array(y1_list, dtype=np.int64)

    s1_path = os.path.join(output_dir, "amc_stage1_features.npz")
    np.savez_compressed(s1_path, X=X1, y=y1)
    print(f"Stage 1 features saved to {s1_path} (Shape X: {X1.shape}, y: {y1.shape})")

    # Generate Stage 2 Sequence Dataset
    print(f"\nGenerating Stage 2 Sequence Dataset ({num_stage2_seqs} sequence instances)...")
    seqs_per_class = num_stage2_seqs // num_classes

    # Train a fast temporary Stage 1 classifier to produce realistic frame predictions for Stage 2
    from sklearn.ensemble import HistGradientBoostingClassifier
    print("Fitting temporary Stage 1 model to extract sequence frame probability distributions...")
    temp_stage1 = HistGradientBoostingClassifier(max_iter=50, random_state=42)
    temp_stage1.fit(X1, y1)

    X2_list = []
    y2_list = []

    for cls_name in MODULATION_CLASSES:
        cls_idx = CLASS_MAP[cls_name]
        print(f"  Synthesizing {seqs_per_class} sequence summaries for class {cls_name}...")
        for _ in range(seqs_per_class):
            num_frames = int(np.random.randint(4, 20))
            snr_db = float(np.random.uniform(-10.0, 30.0))

            frame_feats_seq = []
            snrs_seq = []
            for _ in range(num_frames):
                sps = int(np.random.choice([2, 4, 8, 16]))
                cfo_rel = float(np.random.uniform(-0.08, 0.08))
                phase_noise_std = float(np.random.uniform(0.0, 0.025))

                f = generate_raw_iq_frame(
                    cls_name, num_samples=2048, sps=sps, snr_db=snr_db,
                    cfo_rel=cfo_rel, phase_noise_std=phase_noise_std
                )
                ff = extract_frame_features(f)
                frame_feats_seq.append(ff)
                snrs_seq.append(ff[24])

            X_frames = np.vstack(frame_feats_seq)
            frame_probs = temp_stage1.predict_proba(X_frames)

            # Map predicted classes to 16 classes array
            classes_in_temp = getattr(temp_stage1, "classes_", list(range(16)))
            full_frame_probs = np.zeros((num_frames, 16), dtype=np.float32)
            for idx, c_idx in enumerate(classes_in_temp):
                if c_idx < 16:
                    full_frame_probs[:, c_idx] = frame_probs[:, idx]

            seq_feats = extract_sequence_features(full_frame_probs, snrs_list=snrs_seq)
            X2_list.append(seq_feats)
            y2_list.append(cls_idx)

    X2 = np.vstack(X2_list).astype(np.float32)
    y2 = np.array(y2_list, dtype=np.int64)

    s2_path = os.path.join(output_dir, "amc_stage2_features.npz")
    np.savez_compressed(s2_path, X=X2, y=y2)
    print(f"Stage 2 features saved to {s2_path} (Shape X: {X2.shape}, y: {y2.shape})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate AMC Feature Datasets")
    parser.add_argument("--stage1_frames", type=int, default=80000, help="Total Stage 1 frame instances")
    parser.add_argument("--stage2_seqs", type=int, default=8000, help="Total Stage 2 sequence instances")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory for feature files")
    args = parser.parse_args()

    build_dataset(
        num_stage1_frames=args.stage1_frames,
        num_stage2_seqs=args.stage2_seqs,
        output_dir=args.output_dir
    )

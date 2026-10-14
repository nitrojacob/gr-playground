#!/usr/bin/env python3
"""
Training Script for Deep Learning Automatic Modulation Classification (DL AMC).
Trains 1D ResNet directly on raw complex IQ data [Batch, 2, SequenceLength]
across all MODULATION_CLASSES (including all RadioML 2018.01A schemes).
Exports the trained model to:
- PyTorch weights: `gr_playground/dsp/models/amc_resnet1d.pt`
- Portable ONNX: `gr_playground/dsp/models/amc_resnet1d.onnx`

Preserves all ML AMC assets and training scripts.
"""

import os
import sys
import time
import argparse
from concurrent.futures import ProcessPoolExecutor

# Ensure GNU Radio system path and workspace root are in sys.path
gnuradio_path = "/usr/lib/python3/dist-packages"
if os.path.exists(gnuradio_path) and gnuradio_path not in sys.path:
    sys.path.append(gnuradio_path)

workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

from gr_playground.dsp.amc import MODULATION_CLASSES
from gr_playground.dsp.amc.dl import ResNet1D, DLAMCClassifier
from scripts.generate_amc_dataset import generate_raw_iq_frame


def _generate_iq_chunk(args):
    """
    Generate a chunk of normalized raw IQ frames [count, 2, frame_len] for a single modulation class.
    Channel 0: In-phase (Real)
    Channel 1: Quadrature (Imag)
    """
    mod_name, target_idx, count, frame_len = args
    X = np.zeros((count, 2, frame_len), dtype=np.float32)

    for i in range(count):
        tier = np.random.rand()
        if tier < 0.40:
            # Low SNR Tier (-6 dB to +6 dB)
            snr = float(np.random.uniform(-6.0, 6.0))
            cfo_rel = float(np.random.uniform(-0.04, 0.04))
            phase_noise = float(np.random.uniform(0.0, 0.025))
            mag_imbal = float(np.random.uniform(0.0, 0.8))
            phase_imbal = float(np.random.uniform(0.0, 8.0))
        elif tier < 0.75:
            # Moderate SNR Tier (+6 dB to +18 dB)
            snr = float(np.random.uniform(6.0, 18.0))
            cfo_rel = float(np.random.uniform(-0.03, 0.03))
            phase_noise = float(np.random.uniform(0.0, 0.015))
            mag_imbal = float(np.random.uniform(0.0, 0.5))
            phase_imbal = float(np.random.uniform(0.0, 4.0))
        else:
            # Clean / High SNR Tier (+18 dB to +30 dB)
            snr = float(np.random.uniform(18.0, 30.0))
            cfo_rel = float(np.random.uniform(-0.02, 0.02))
            phase_noise = float(np.random.uniform(0.0, 0.005))
            mag_imbal = float(np.random.uniform(0.0, 0.2))
            phase_imbal = float(np.random.uniform(0.0, 2.0))

        sps = int(np.random.choice([4, 8]))
        rrc_alpha = float(np.random.choice([0.25, 0.35, 0.45]))
        use_rrc = (np.random.rand() < 0.85)
        phase_offset_rad = float(np.random.uniform(-np.pi, np.pi))
        frac_delay = float(np.random.uniform(-sps / 2.0, sps / 2.0))

        raw_iq = generate_raw_iq_frame(
            mod_type=mod_name,
            num_samples=frame_len,
            sps=sps,
            snr_db=snr,
            cfo_rel=cfo_rel,
            phase_offset_rad=phase_offset_rad,
            fractional_delay=frac_delay,
            phase_noise_std=phase_noise,
            mag_imbal_db=mag_imbal,
            phase_imbal_deg=phase_imbal,
            use_rrc=use_rrc,
            rrc_alpha=rrc_alpha
        )

        # Normalize energy and extract 2-channel representation
        p = np.mean(np.abs(raw_iq)**2)
        if p > 1e-12:
            raw_iq = raw_iq / np.sqrt(p)

        X[i, 0, :] = raw_iq.real.astype(np.float32)
        X[i, 1, :] = raw_iq.imag.astype(np.float32)

    labels = np.full(count, target_idx, dtype=np.int64)
    return X, labels


def generate_dl_training_data(samples_per_class: int = 400, frame_len: int = 1024):
    """
    Parallel in-memory generation of raw IQ frames [Batch, 2, frame_len].
    """
    t0 = time.time()
    num_classes = len(MODULATION_CLASSES)
    total_samples = num_classes * samples_per_class
    print(f"Generating DL dataset: {num_classes} classes x {samples_per_class} = {total_samples} raw IQ frames (len={frame_len})...")

    class_to_idx = {c: i for i, c in enumerate(MODULATION_CLASSES)}
    tasks = [(mod_name, class_to_idx[mod_name], samples_per_class, frame_len) for mod_name in MODULATION_CLASSES]

    X_list = []
    y_list = []

    max_workers = min(os.cpu_count() or 4, 16)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(_generate_iq_chunk, tasks)
        for chunk_X, chunk_y in results:
            X_list.append(chunk_X)
            y_list.append(chunk_y)

    X = np.vstack(X_list).astype(np.float32)
    y = np.concatenate(y_list).astype(np.int64)
    X = np.nan_to_num(X, nan=0.0, posinf=5.0, neginf=-5.0)

    elapsed = time.time() - t0
    print(f"Dataset generated in {elapsed:.2f}s ({total_samples / max(elapsed, 0.001):.1f} frames/sec). X shape: {X.shape}")
    return X, y


def train_and_export_resnet1d(
    output_model_dir: str = "gr_playground/dsp/models",
    samples_per_class: int = 400,
    epochs: int = 35,
    batch_size: int = 64,
    frame_len: int = 1024
):
    os.makedirs(output_model_dir, exist_ok=True)
    pt_path = os.path.join(output_model_dir, "amc_resnet1d.pt")
    onnx_path = os.path.join(output_model_dir, "amc_resnet1d.onnx")

    num_threads = min(os.cpu_count() or 4, 8)
    torch.set_num_threads(num_threads)
    print(f"PyTorch CPU threads: {num_threads}")

    X, y = generate_dl_training_data(samples_per_class=samples_per_class, frame_len=frame_len)

    # 80:20 stratified split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Split: Train={X_train.shape[0]} frames, Val={X_val.shape[0]} frames")

    device = torch.device("cpu")
    model = ResNet1D(in_channels=2, num_classes=len(MODULATION_CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_x = torch.tensor(X_train, dtype=torch.float32)
    train_y = torch.tensor(y_train, dtype=torch.long)
    val_x = torch.tensor(X_val, dtype=torch.float32)
    val_y = torch.tensor(y_val, dtype=torch.long)

    train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    print(f"Training 1D ResNet ({epochs} epochs, batch_size={batch_size})...")
    best_val_acc = 0.0
    best_model_state = None

    t_start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct_train = 0
        total_train = 0

        for b_x, b_y in train_loader:
            optimizer.zero_grad()
            out = model(b_x)
            loss = criterion(out, b_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * b_x.size(0)
            preds = torch.argmax(out, dim=1)
            correct_train += (preds == b_y).sum().item()
            total_train += b_x.size(0)

        scheduler.step()
        train_loss = total_loss / max(total_train, 1)
        train_acc = correct_train / max(total_train, 1)

        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(val_x)
            val_loss = criterion(val_out, val_y).item()
            val_preds = torch.argmax(val_out, dim=1)
            val_acc = (val_preds == val_y).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

    elapsed = time.time() - t_start
    print(f"1D ResNet training completed in {elapsed:.2f}s. Best Val Acc: {best_val_acc*100:.2f}%")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # 1. Save PyTorch state dict temporarily
    torch.save(model.state_dict(), pt_path)
    print(f"Temporarily saved PyTorch weights: {pt_path}", flush=True)

    # 2. Export portable ONNX model with dynamic sequence length support
    model.eval()
    dummy_input = torch.randn(1, 2, frame_len, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 2: "sequence_length"},
            "output": {0: "batch_size"}
        }
    )
    print(f"Saved ONNX model: {onnx_path}", flush=True)

    # 3. Requirement: Once onnx is generated, remove the pt file
    if os.path.exists(pt_path):
        os.remove(pt_path)
        print(f"Removed PyTorch weights checkpoint ({pt_path}) after successful ONNX export.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 1D ResNet DL AMC on Raw IQ Data")
    parser.add_argument("--output_model_dir", type=str, default="gr_playground/dsp/models", help="Output directory")
    parser.add_argument("--samples_per_class", type=int, default=100, help="Synthetic frames per class")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--frame_len", type=int, default=1024, help="IQ frame length")
    args = parser.parse_args()

    train_and_export_resnet1d(
        output_model_dir=args.output_model_dir,
        samples_per_class=args.samples_per_class,
        epochs=args.epochs,
        batch_size=args.batch_size,
        frame_len=args.frame_len
    )

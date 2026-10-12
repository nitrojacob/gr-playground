"""
Training Script for Automatic Modulation Classification (AMC) Model.
Trains PyTorch AMCFeatureNet across all MODULATION_CLASSES (including all RadioML 2018.01A schemes)
with 80:20 train/validation split, multithreaded compute efficiency, and low-SNR impairment coverage,
exporting the trained model directly to portable ONNX format (`gr_playground/dsp/models/amc_model.onnx`).
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
from gr_playground.dsp.amc.features import extract_frame_features
from scripts.generate_amc_dataset import generate_raw_iq_frame


class AMCFeatureNet(nn.Module):
    """
    PyTorch Feature Classifier mapping 26 physical DSP features -> modulation logits.
    Exportable directly to portable ONNX format via torch.onnx.export.
    """
    def __init__(self, input_dim=26, num_classes=len(MODULATION_CLASSES)):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(input_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(self.input_bn(x))


def _generate_class_chunk(args):
    """
    Generate a chunk of synthetic feature vectors for a single modulation class in a tight loop.
    Avoids per-sample IPC serialization overhead across worker processes.
    """
    mod_name, target_idx, count = args
    feats = np.zeros((count, 26), dtype=np.float32)

    for i in range(count):
        tier = np.random.rand()
        if tier < 0.50:
            # Low & Very Low SNR Tier (-10.0 dB to +6.0 dB) -> 50% of dataset
            snr = float(np.random.uniform(-10.0, 6.0))
            cfo_rel = float(np.random.uniform(-0.025, 0.025))
            phase_noise = float(np.random.uniform(0.0, 0.03))
            mag_imbal = float(np.random.uniform(0.0, 1.0))
            phase_imbal = float(np.random.uniform(0.0, 10.0))
        elif tier < 0.80:
            # Moderate SNR Tier (+6.0 dB to +16.0 dB) -> 30% of dataset
            snr = float(np.random.uniform(6.0, 16.0))
            cfo_rel = float(np.random.uniform(-0.015, 0.015))
            phase_noise = float(np.random.uniform(0.0, 0.015))
            mag_imbal = float(np.random.uniform(0.0, 0.6))
            phase_imbal = float(np.random.uniform(0.0, 5.0))
        else:
            # Clean / High SNR Tier (+16.0 dB to +30.0 dB) -> 20% of dataset
            snr = float(np.random.uniform(16.0, 30.0))
            cfo_rel = float(np.random.uniform(-0.005, 0.005))
            phase_noise = float(np.random.uniform(0.0, 0.005))
            mag_imbal = float(np.random.uniform(0.0, 0.2))
            phase_imbal = float(np.random.uniform(0.0, 2.0))

        num_samples = int(np.random.choice([1024, 2048, 4096]))
        samples = generate_raw_iq_frame(
            mod_type=mod_name,
            num_samples=num_samples,
            snr_db=snr,
            cfo_rel=cfo_rel,
            phase_noise_std=phase_noise,
            mag_imbal_db=mag_imbal,
            phase_imbal_deg=phase_imbal
        )
        feats[i] = extract_frame_features(samples)

    labels = np.full(count, target_idx, dtype=np.int64)
    return feats, labels


def generate_training_data(samples_per_class=600):
    """
    Parallel in-memory generation of synthetic IQ frames and extracted features.
    Parallelized across classes to maximize CPU throughput and eliminate IPC bottlenecks.
    """
    t0 = time.time()
    num_classes = len(MODULATION_CLASSES)
    total_samples = num_classes * samples_per_class
    print(f"Generating dataset across {num_classes} classes ({samples_per_class} samples/class = {total_samples} total)...")

    class_to_idx = {c: i for i, c in enumerate(MODULATION_CLASSES)}
    tasks = [(mod_name, class_to_idx[mod_name], samples_per_class) for mod_name in MODULATION_CLASSES]

    X_list = []
    y_list = []

    max_workers = min(os.cpu_count() or 4, 16)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(_generate_class_chunk, tasks)
        for feats, labels in results:
            X_list.append(feats)
            y_list.append(labels)

    X = np.vstack(X_list).astype(np.float32)
    y = np.concatenate(y_list).astype(np.int64)
    X = np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0)

    elapsed = time.time() - t0
    print(f"Dataset generated in {elapsed:.2f}s ({total_samples / max(elapsed, 0.001):.1f} samples/sec). Shape: X={X.shape}, y={y.shape}")
    return X, y


def train_and_export_onnx(output_model_dir: str = "gr_playground/dsp/models", samples_per_class: int = 600, epochs: int = 40):
    os.makedirs(output_model_dir, exist_ok=True)
    onnx_path = os.path.join(output_model_dir, "amc_model.onnx")

    # Multithreading configuration for PyTorch CPU compute
    num_threads = min(os.cpu_count() or 4, 8)
    torch.set_num_threads(num_threads)
    print(f"Configured PyTorch CPU thread count: {num_threads}")

    X, y = generate_training_data(samples_per_class=samples_per_class)

    # Segregate training data into Train and Validation sets (80:20 stratified split)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Data Split (80:20): Train={X_train.shape[0]} samples, Val={X_val.shape[0]} samples")

    device = torch.device("cpu")
    model = AMCFeatureNet(input_dim=26, num_classes=len(MODULATION_CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_tensor_x = torch.tensor(X_train, dtype=torch.float32)
    train_tensor_y = torch.tensor(y_train, dtype=torch.long)
    val_tensor_x = torch.tensor(X_val, dtype=torch.float32)
    val_tensor_y = torch.tensor(y_val, dtype=torch.long)

    train_dataset = torch.utils.data.TensorDataset(train_tensor_x, train_tensor_y)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

    print(f"Starting PyTorch training ({epochs} epochs, batch_size=128)...")
    best_val_acc = 0.0
    best_model_state = None

    t_train_start = time.time()
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

        # Validation evaluation
        model.eval()
        with torch.no_grad():
            val_out = model(val_tensor_x)
            val_loss = criterion(val_out, val_tensor_y).item()
            val_preds = torch.argmax(val_out, dim=1)
            val_acc = (val_preds == val_tensor_y).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%")

    train_elapsed = time.time() - t_train_start
    print(f"Training completed in {train_elapsed:.2f}s. Best Validation Accuracy: {best_val_acc * 100:.2f}%")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Export ONNX Model Artifact directly
    model.eval()
    dummy_input = torch.randn(1, 26, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    print(f"Successfully exported ONNX model artifact to: {onnx_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AMC PyTorch Model & Export ONNX Binary")
    parser.add_argument("--output_model_dir", type=str, default="gr_playground/dsp/models", help="Directory to save amc_model.onnx")
    parser.add_argument("--samples_per_class", type=int, default=600, help="Number of synthetic frames per class")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    args = parser.parse_args()

    train_and_export_onnx(output_model_dir=args.output_model_dir, samples_per_class=args.samples_per_class, epochs=args.epochs)

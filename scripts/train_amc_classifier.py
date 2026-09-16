"""
Training Script for Automatic Modulation Classification (AMC) Models.
Trains Stage 1 Frame Classifier and Stage 2 Meta-Learner Classifier using HistGradientBoostingClassifier.
Saves model binaries to `gr_playground/dsp/models/amc_stage1_model.pkl` and `amc_stage2_model.pkl`.
"""

import os
import argparse
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

MODULATION_CLASSES = [
    "AM", "FM", "BPSK", "GFSK", "QPSK", "8PSK",
    "16QAM", "64QAM", "256QAM", "ASK", "16APSK", "32APSK",
    "OQPSK", "OFDM", "SC-FDMA", "Noise"
]

def train_models(data_dir: str = "data", output_model_dir: str = "gr_playground/dsp/models"):
    os.makedirs(output_model_dir, exist_ok=True)

    stage1_file = os.path.join(data_dir, "amc_stage1_features.npz")
    stage2_file = os.path.join(data_dir, "amc_stage2_features.npz")

    if not os.path.exists(stage1_file) or not os.path.exists(stage2_file):
        raise FileNotFoundError(
            f"Feature data files not found in {data_dir}. "
            "Run `python3 scripts/generate_amc_dataset.py` first."
        )

    # 1. Train Stage 1 Frame Classifier
    print("=" * 60)
    print("Training Stage 1 Frame Classifier...")
    data1 = np.load(stage1_file)
    X1, y1 = data1["X"], data1["y"]
    print(f"Loaded Stage 1 Feature Matrix: X shape {X1.shape}, y shape {y1.shape}")

    X1_train, X1_val, y1_train, y1_val = train_test_split(
        X1, y1, test_size=0.15, random_state=42, stratify=y1
    )

    stage1_model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.08,
        max_iter=250,
        l2_regularization=1.0,
        random_state=42
    )

    print("Fitting Stage 1 HistGradientBoostingClassifier...")
    stage1_model.fit(X1_train, y1_train)

    y1_pred = stage1_model.predict(X1_val)
    acc1 = accuracy_score(y1_val, y1_pred)
    print(f"\nStage 1 Validation Accuracy: {acc1 * 100:.2f}%\n")
    print(classification_report(y1_val, y1_pred, target_names=MODULATION_CLASSES, digits=3))

    s1_out = os.path.join(output_model_dir, "amc_stage1_model.pkl")
    joblib.dump(stage1_model, s1_out)
    print(f"Saved Stage 1 Model to {s1_out}")

    # 2. Train Stage 2 Sequence Classifier
    print("\n" + "=" * 60)
    print("Training Stage 2 Meta-Learner Classifier...")
    data2 = np.load(stage2_file)
    X2, y2 = data2["X"], data2["y"]
    print(f"Loaded Stage 2 Feature Matrix: X shape {X2.shape}, y shape {y2.shape}")

    X2_train, X2_val, y2_train, y2_val = train_test_split(
        X2, y2, test_size=0.15, random_state=42, stratify=y2
    )

    stage2_model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.08,
        max_iter=200,
        l2_regularization=1.0,
        random_state=42
    )

    print("Fitting Stage 2 HistGradientBoostingClassifier...")
    stage2_model.fit(X2_train, y2_train)

    y2_pred = stage2_model.predict(X2_val)
    acc2 = accuracy_score(y2_val, y2_pred)
    print(f"\nStage 2 Validation Accuracy: {acc2 * 100:.2f}%\n")
    print(classification_report(y2_val, y2_pred, target_names=MODULATION_CLASSES, digits=3))

    s2_out = os.path.join(output_model_dir, "amc_stage2_output.pkl") # also save as amc_stage2_model.pkl
    s2_model_path = os.path.join(output_model_dir, "amc_stage2_model.pkl")
    joblib.dump(stage2_model, s2_model_path)
    print(f"Saved Stage 2 Model to {s2_model_path}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AMC Classifiers")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing feature .npz files")
    parser.add_argument("--model_dir", type=str, default="gr_playground/dsp/models", help="Directory to save trained .pkl models")
    args = parser.parse_args()

    train_models(data_dir=args.data_dir, output_model_dir=args.model_dir)

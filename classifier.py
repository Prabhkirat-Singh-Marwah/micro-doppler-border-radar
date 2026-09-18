"""
classifier.py
=============
CNN-Based Micro-Doppler Target Classifier
------------------------------------------
Defines a lightweight 3-layer Convolutional Neural Network (CNN) for
classifying radar spectrograms into one of 5 target classes.

CNN ARCHITECTURE RATIONALE (for viva)
---------------------------------------
We use a CNN (not a traditional ML classifier like SVM/Random Forest)
because our input is a 2-D spectrogram IMAGE — CNNs excel at learning
spatial patterns like:
  - The sinusoidal band pattern of human limb micro-Doppler
  - The vertical flash stripes of drone rotor blades
  - The solid horizontal band of a vehicle's constant Doppler

We keep it shallow (3 conv layers) because:
  - Our training set is small (~3000 samples) — deep nets overfit easily
  - We're running on CPU — more layers = much slower training
  - Simple patterns: these 5 classes are visually very distinct

LAYER-BY-LAYER EXPLANATION
----------------------------
Conv2d(1→16, 3×3)  : Detects low-level features (edges, blobs) in the spectrogram
MaxPool(2×2)        : Halves spatial size, keeps dominant features
Conv2d(16→32, 3×3) : Combines low-level features into patterns (bands, stripes)
MaxPool(2×2)        : Further spatial reduction
Conv2d(32→64, 3×3) : High-level pattern detection (e.g., "periodic sinusoidal band")
MaxPool(2×2)        : Compact feature map
Flatten → Linear    : Collapses 2-D feature map into a class-score vector
Softmax (implicit)  : CrossEntropyLoss applies softmax internally during training
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Tuple, List, Optional

# Default path to save/load trained weights
DEFAULT_WEIGHTS_PATH = "classifier_weights.pth"

# Training hyper-parameters (tuned for CPU speed)
LEARNING_RATE = 1e-3
BATCH_SIZE    = 32
NUM_EPOCHS    = 15
NUM_CLASSES   = 5
CLASS_NAMES   = ["human_walk", "human_run", "animal", "drone", "vehicle"]


# ── Dataset wrapper ────────────────────────────────────────────────────────

class SpectrogramDataset(Dataset):
    """
    Simple PyTorch Dataset wrapping NumPy spectrogram arrays.

    Parameters
    ----------
    specs  : np.ndarray  shape (N, 1, H, W) — normalised spectrograms
    labels : np.ndarray  shape (N,)          — integer class labels
    """
    def __init__(self, specs: np.ndarray, labels: np.ndarray):
        self.specs  = torch.tensor(specs,  dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.specs[idx], self.labels[idx]


# ── CNN Model ─────────────────────────────────────────────────────────────

class MicroDopplerCNN(nn.Module):
    """
    3-Layer CNN for micro-Doppler spectrogram classification.

    Input  : (batch, 1, 64, 64) — grayscale spectrogram
    Output : (batch, 5)          — raw logits for 5 classes

    After 3 × (Conv + ReLU + MaxPool) the spatial size reduces:
      64 → 32 → 16 → 8   (each MaxPool halves width and height)
    With 64 output channels, the flattened vector has 64 × 8 × 8 = 4096 dims.
    A hidden Linear(4096→128) further compresses before the output layer.
    """
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        # ── Feature extractor (convolutional layers) ─────────────
        self.features = nn.Sequential(
            # Block 1: detect basic edges / blobs
            nn.Conv2d(1, 16, kernel_size=3, padding=1),   # (B,16,64,64)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                               # (B,16,32,32)

            # Block 2: combine into patterns (bands, stripes)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # (B,32,32,32)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                               # (B,32,16,16)

            # Block 3: high-level micro-Doppler signatures
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # (B,64,16,16)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                               # (B,64, 8, 8)
        )

        # ── Classifier head (fully connected layers) ─────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),                        # (B, 64*8*8) = (B, 4096)
            nn.Linear(64 * 8 * 8, 128),          # compress to 128 dims
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),                     # regularisation — reduces overfitting
            nn.Linear(128, num_classes),         # final class scores (logits)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x  # raw logits — CrossEntropyLoss applies softmax


# ── Training ──────────────────────────────────────────────────────────────

def train_model(
    specs  : np.ndarray,
    labels : np.ndarray,
    epochs : int = NUM_EPOCHS,
    weights_path: str = DEFAULT_WEIGHTS_PATH,
    progress_callback=None,
) -> Tuple["MicroDopplerCNN", List[dict]]:
    """
    Train the CNN on spectrograms. Saves weights to disk after training.

    Uses an 80/20 train/validation split.
    Prints per-epoch loss and accuracy to console.

    Parameters
    ----------
    specs            : np.ndarray (N, 1, H, W) — normalised spectrograms
    labels           : np.ndarray (N,)          — integer labels
    epochs           : int                      — training epochs
    weights_path     : str                      — file to save weights
    progress_callback: callable(epoch, total, metrics) — optional UI callback

    Returns
    -------
    model   : trained MicroDopplerCNN
    history : list of dicts with per-epoch metrics
    """
    device = torch.device("cpu")   # Always CPU — no GPU assumed
    model  = MicroDopplerCNN(num_classes=NUM_CLASSES).to(device)

    # 80/20 train/val split
    dataset   = SpectrogramDataset(specs, labels)
    n_train   = int(0.8 * len(dataset))
    n_val     = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()  # includes softmax internally

    # Optional: reduce LR if validation loss plateaus
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    history = []
    print(f"Training on {n_train} samples, validating on {n_val} samples")
    print(f"Device: CPU | Epochs: {epochs} | Batch size: {BATCH_SIZE}")
    print("-" * 55)

    for epoch in range(1, epochs + 1):
        # ── Training phase ────────────────────────────
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for specs_batch, labels_batch in train_loader:
            specs_batch  = specs_batch.to(device)
            labels_batch = labels_batch.to(device)

            optimizer.zero_grad()
            logits = model(specs_batch)
            loss   = criterion(logits, labels_batch)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item() * len(labels_batch)
            preds          = logits.argmax(dim=1)
            train_correct += (preds == labels_batch).sum().item()
            train_total   += len(labels_batch)

        scheduler.step()

        # ── Validation phase ──────────────────────────
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for specs_batch, labels_batch in val_loader:
                specs_batch  = specs_batch.to(device)
                labels_batch = labels_batch.to(device)
                logits       = model(specs_batch)
                loss         = criterion(logits, labels_batch)
                val_loss    += loss.item() * len(labels_batch)
                preds        = logits.argmax(dim=1)
                val_correct += (preds == labels_batch).sum().item()
                val_total   += len(labels_batch)

        metrics = {
            "epoch"    : epoch,
            "train_loss" : train_loss / train_total,
            "train_acc"  : train_correct / train_total,
            "val_loss"   : val_loss / val_total,
            "val_acc"    : val_correct / val_total,
        }
        history.append(metrics)
        print(f"Epoch {epoch:02d}/{epochs} | "
              f"Train loss={metrics['train_loss']:.4f} acc={metrics['train_acc']:.3f} | "
              f"Val loss={metrics['val_loss']:.4f} acc={metrics['val_acc']:.3f}")

        if progress_callback:
            progress_callback(epoch, epochs, metrics)

    # Save trained weights
    torch.save(model.state_dict(), weights_path)
    print(f"\nModel saved to: {weights_path}")
    return model, history


# ── Inference ─────────────────────────────────────────────────────────────

def load_model(weights_path: str = DEFAULT_WEIGHTS_PATH) -> "MicroDopplerCNN":
    """Load a trained model from saved weights."""
    model = MicroDopplerCNN(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def predict(
    model    : "MicroDopplerCNN",
    spec     : np.ndarray,   # shape (1, H, W) — single spectrogram
) -> Tuple[str, float, np.ndarray]:
    """
    Run inference on a single spectrogram.

    Returns
    -------
    class_name  : str    predicted class name
    confidence  : float  softmax probability of the top class (0–1)
    probs       : np.ndarray  full probability distribution over 5 classes
    """
    model.eval()
    with torch.no_grad():
        x      = torch.tensor(spec[np.newaxis], dtype=torch.float32)  # add batch dim
        logits = model(x)
        probs  = torch.softmax(logits, dim=1).squeeze().numpy()

    top_idx    = int(np.argmax(probs))
    confidence = float(probs[top_idx])
    class_name = CLASS_NAMES[top_idx]
    return class_name, confidence, probs


def is_trained(weights_path: str = DEFAULT_WEIGHTS_PATH) -> bool:
    """Check if trained weights already exist on disk."""
    return os.path.isfile(weights_path)


# ── CLI smoke test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data_generator import generate_dataset, PRF
    from preprocessor import batch_to_spectrograms

    print("=" * 55)
    print("Micro-Doppler CNN — Training Smoke Test")
    print("=" * 55)

    print("\n[1/3] Generating dataset...")
    X, y = generate_dataset(samples_per_class=200)   # small for quick test
    print(f"      {len(X)} signals generated")

    print("\n[2/3] Computing spectrograms...")
    specs = batch_to_spectrograms(X, fs=PRF)
    print(f"      Spectrogram array shape: {specs.shape}")

    print("\n[3/3] Training CNN (5 epochs for smoke test)...")
    model, history = train_model(specs, y, epochs=5)

    final = history[-1]
    print(f"\nFinal validation accuracy: {final['val_acc']:.1%}")

    # Test inference
    print("\nInference test on 5 random samples:")
    for i in range(5):
        idx = np.random.randint(len(specs))
        name, conf, _ = predict(model, specs[idx])
        true_name = CLASS_NAMES[y[idx]]
        match = "✓" if name == true_name else "✗"
        print(f"  {match} True: {true_name:12s}  Predicted: {name:12s}  Confidence: {conf:.3f}")

    print("\n✓ classifier.py OK")

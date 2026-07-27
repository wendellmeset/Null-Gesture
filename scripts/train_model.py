#!/usr/bin/env python3
"""Train GestureCNN on collected + augmented IMU gesture data.

Splits by ORIGINAL recording (not augmented variant) so validation is honest.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ══════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════

def load_collected(data_dir: Path, gestures: list[str]) -> dict[str, list[np.ndarray]]:
    """Load .npy recordings, grouped by gesture name."""
    samples: dict[str, list[np.ndarray]] = {g: [] for g in gestures}
    for fp in sorted(data_dir.glob("*.npy")):
        stem = fp.stem
        for g in gestures:
            if stem.startswith(g) and (len(stem) == len(g) or stem[len(g):].startswith("_")):
                arr = np.load(fp).astype(np.float32)
                if arr.ndim == 2 and arr.shape[1] == 6 and arr.shape[0] >= 10:
                    samples[g].append(arr)
                break
    return samples


# ══════════════════════════════════════════════════════════════════════
# Augmentation (stronger than before)
# ══════════════════════════════════════════════════════════════════════

def _smooth_warp(length: int, sigma: float, rng: np.random.Generator) -> np.ndarray:
    raw = rng.normal(0, 1, length)
    raw = np.convolve(raw, np.ones(5) / 5.0, mode="same")
    if raw.std() < 1e-9:
        return np.arange(length, dtype=np.float64)
    raw = raw / raw.std() * sigma
    warp = np.cumsum(1.0 + raw)
    return (warp - warp[0]) / (warp[-1] - warp[0]) * (length - 1)


def time_warp(signal: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    T, C = signal.shape
    warp = _smooth_warp(T, sigma, rng)
    out = np.zeros_like(signal)
    src = np.arange(T, dtype=np.float64)
    for c in range(C):
        out[:, c] = np.interp(src, warp, signal[:, c])
    return out.astype(np.float32)


def augment_one(sample: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate n synthetic variants of one real recording."""
    T, C = sample.shape
    ch_std = np.std(sample, axis=0, keepdims=True)
    ch_std = np.where(ch_std < 1e-9, 1e-9, ch_std)

    out = np.empty((n, T, C), dtype=np.float32)
    out[0] = sample.astype(np.float32)

    for i in range(1, n):
        v = sample.copy().astype(np.float32)

        # 1. Aggressive noise (0.5% to 12% of channel std)
        v += rng.normal(0, 1, (T, C)).astype(np.float32) * ch_std * rng.uniform(0.005, 0.12)

        # 2. Strong time warping (always applied, wider sigma range)
        v = time_warp(v, sigma=rng.uniform(0.04, 0.22), rng=rng)

        # 3. Wide magnitude scaling (0.5x to 1.6x)
        v *= rng.uniform(0.50, 1.60)

        # 4. Per-channel bias drift
        v += rng.normal(0, 1, (1, C)).astype(np.float32) * ch_std * rng.uniform(0.0, 0.08)

        # 5. Random time shift ±15 steps
        shift = int(rng.integers(-15, 16))
        if shift != 0:
            v = np.roll(v, shift, axis=0)
            if shift > 0:
                v[:shift] = v[shift:shift + 1]
            else:
                v[shift:] = v[shift - 1:shift]

        # 6. Random 3D rotation of accelerometer frame (small angles, ±20°)
        if rng.random() < 0.4:
            angle = rng.uniform(-0.35, 0.35)  # radians ≈ ±20°
            axis = rng.integers(0, 3)
            if axis == 0:  # rotate y-z plane
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                v[:, 1], v[:, 2] = cos_a * v[:, 1] - sin_a * v[:, 2], sin_a * v[:, 1] + cos_a * v[:, 2]
            elif axis == 1:  # rotate x-z plane
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                v[:, 0], v[:, 2] = cos_a * v[:, 0] + sin_a * v[:, 2], -sin_a * v[:, 0] + cos_a * v[:, 2]
            else:  # rotate x-y plane
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                v[:, 0], v[:, 1] = cos_a * v[:, 0] - sin_a * v[:, 1], sin_a * v[:, 0] + cos_a * v[:, 1]

        # 7. Channel dropout (random channel zeroed for 5–25% of window)
        if rng.random() < 0.08:
            ch = int(rng.integers(0, C))
            seg = int(rng.integers(T // 20, T // 4))
            start = int(rng.integers(0, max(1, T - seg)))
            v[start:start + seg, ch] = 0.0

        out[i] = v

    return out


def augment_dataset(
    samples: dict[str, list[np.ndarray]],
    n_per_sample: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Augment all recordings in a sample dict → (X, y)."""
    gesture_list = list(samples.keys())  # preserve order from caller (matches --gestures)
    label_map = {g: i for i, g in enumerate(gesture_list)}
    rng = np.random.default_rng(seed)

    xs, ys = [], []
    for g_idx, gesture in enumerate(gesture_list):
        for s_idx, sample in enumerate(samples[gesture]):
            sub_seed = seed + g_idx * 1000 + s_idx
            variants = augment_one(sample, n=n_per_sample, rng=np.random.default_rng(sub_seed))
            xs.append(variants)
            ys.append(np.full(n_per_sample, label_map[gesture], dtype=np.int64))

    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


# ══════════════════════════════════════════════════════════════════════
# Normalisation
# ══════════════════════════════════════════════════════════════════════

def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-channel z-score. Returns (X_scaled, mean, std) — mean/std are (C,)."""
    mean = X.mean(axis=(0, 1))  # (C,)
    std = X.std(axis=(0, 1)) + 1e-8
    return (X - mean) / std, mean, std


def apply_norm(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / (std + 1e-8)


# ══════════════════════════════════════════════════════════════════════
# Training loop helpers
# ══════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        correct += (model(xb).argmax(1) == yb).sum().item()
        total += xb.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item() * xb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        total += xb.size(0)
    return total_loss / total, correct / total


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train GestureCNN")
    parser.add_argument("--data", default="data/raw", help="Directory with .npy recordings")
    parser.add_argument("--gestures",
                        default="push,pull,left,right,clockwise,anti_clockwise,bye_bye,palm_up,palm_down,clapping,one_arm_boxing,t_arms,raise_arms",
                        help="Comma-separated gesture names")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--augment", type=int, default=300, help="Synthetic variants per TRAIN sample")
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--patience", type=int, default=40, help="Early stopping patience")
    parser.add_argument("--output", default="null_gesture/models/saved/gesture_cnn.pt")
    args = parser.parse_args(argv)

    PROJ = Path(__file__).resolve().parent.parent
    data_dir = (PROJ / args.data).resolve()
    out_path = (PROJ / args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gestures = [g.strip() for g in args.gestures.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load ──────────────────────────────────────────────────────
    print(f"\nLoading recordings from {data_dir} ...")
    collected = load_collected(data_dir, gestures)
    for g in gestures:
        print(f"  {g}: {len(collected[g])} recording(s)")
    empty = [g for g in gestures if not collected[g]]
    if empty:
        print(f"\n❌ No recordings found for: {', '.join(empty)}")
        return 1

    # Warn if only 1 recording per class
    min_rec = min(len(collected[g]) for g in gestures)
    if min_rec < 2:
        print("\n⚠️  Only 1 recording per gesture — can't do honest train/val split.")
        print("   Collect at least 2 samples per gesture for meaningful validation.")
        print("   Will train on all data but validation numbers are meaningless.\n")

    # ── Split by ORIGINAL recording ───────────────────────────────
    # Training: first recording of each gesture
    # Validation: remaining recording(s) of each gesture
    train_real: dict[str, list[np.ndarray]] = {}
    val_real: dict[str, list[np.ndarray]] = {}
    for g in gestures:
        recs = collected[g]
        if len(recs) >= 2:
            train_real[g] = [recs[0]]
            val_real[g] = recs[1:]  # all remaining
        else:
            train_real[g] = recs
            val_real[g] = []

    # ── Augment each split independently ──────────────────────────

    print(f"\nAugmenting train set: {args.augment} variants per recording ...")
    t0 = time.time()
    X_train, y_train = augment_dataset(train_real, n_per_sample=args.augment)
    print(f"  → {X_train.shape[0]} train samples in {time.time() - t0:.1f}s")

    has_val = any(len(v) > 0 for v in val_real.values())
    if has_val:
        val_aug = max(20, args.augment // 8)
        print(f"Augmenting val set: {val_aug} variants per recording ...")
        X_val, y_val = augment_dataset(val_real, n_per_sample=val_aug, seed=12345)
        print(f"  → {X_val.shape[0]} val samples (from HELD-OUT recordings)")
    else:
        # Fallback: split augmented train set (worse, but works)
        X_val, y_val = np.zeros((0, X_train.shape[1], X_train.shape[2]), dtype=np.float32), np.zeros(0, dtype=np.int64)

    # ── Standardise (fit on TRAIN only) ───────────────────────────
    X_tr_scaled, ch_mean, ch_std = standardize(X_train)
    X_tr_pt = torch.tensor(X_tr_scaled, dtype=torch.float32).permute(0, 2, 1)
    y_tr_pt = torch.tensor(y_train, dtype=torch.long)

    if has_val:
        X_val_scaled = apply_norm(X_val, ch_mean, ch_std)
        X_val_pt = torch.tensor(X_val_scaled, dtype=torch.float32).permute(0, 2, 1)
        y_val_pt = torch.tensor(y_val, dtype=torch.long)
    else:
        # Fallback: use 15% of augmented train as val
        n_total = len(X_tr_pt)
        perm = torch.randperm(n_total)
        n_train = int(n_total * 0.85)
        X_val_pt = X_tr_pt[perm[n_train:]]
        y_val_pt = y_tr_pt[perm[n_train:]]
        X_tr_pt = X_tr_pt[perm[:n_train]]
        y_tr_pt = y_tr_pt[perm[:n_train]]
        print("  (fallback: val split from augmented data — NOT honest)")

    print(f"  Train: {len(X_tr_pt)}  |  Val: {len(X_val_pt)}")

    tr_loader = DataLoader(TensorDataset(X_tr_pt, y_tr_pt), batch_size=args.batch, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(X_val_pt, y_val_pt), batch_size=args.batch * 2)

    # ── Smaller model ─────────────────────────────────────────────
    from null_gesture.models.nn_model import GestureCNN

    model = GestureCNN(n_classes=len(gestures), dropout=args.dropout).to(device)
    print(f"\nGestureCNN: {model.param_count:,} params")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=2e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── Train ─────────────────────────────────────────────────────
    best_acc = 0.0
    best_epoch = 0
    best_state = None
    patience_counter = 0

    print("\nTraining (val = held-out recordings — the honest number) ...\n")
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, tr_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        marker = ""
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            marker = " ★"
        else:
            patience_counter += 1

        print(f"  Epoch {epoch:3d} | tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}{marker}")

        if patience_counter >= args.patience:
            print(f"\nEarly stop at epoch {epoch} (best: {best_epoch}, val_acc={best_acc:.4f})")
            break

    if best_state is None:
        print("❌ Training failed.")
        return 1

    model.load_state_dict(best_state)

    # ── Per-class validation (on held-out recordings) ─────────────
    print("\nPer-class accuracy on HELD-OUT recordings:")
    model.eval()
    with torch.no_grad():
        all_logits = []
        for xb, _ in val_loader:
            all_logits.append(model(xb.to(device)).cpu())
        logits = torch.cat(all_logits)
        preds = logits.argmax(1)
        for i, g in enumerate(gestures):
            mask = y_val_pt == i
            if mask.sum() > 0:
                acc = (preds[mask] == i).float().mean().item()
                bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
                print(f"  {g:20s}  {acc:.3f}  {bar}")
            else:
                print(f"  {g:20s}  (no val data)")

    # ── Save ──────────────────────────────────────────────────────
    bundle = {
        "state_dict": best_state,
        "n_classes": len(gestures),
        "gestures": gestures,
        "ch_mean": ch_mean,
        "ch_std": ch_std,
        "val_acc": best_acc,
    }
    torch.save(bundle, out_path)
    print(f"\n✅ Saved: {out_path}  (honest val_acc={best_acc:.4f})")

    # ── Sanity: confusion matrix ──────────────────────────────────
    if has_val and len(gestures) <= 12:
        print("\nConfusion matrix (rows=true, cols=pred):")
        cm = np.zeros((len(gestures), len(gestures)), dtype=int)
        for i in range(len(y_val_pt)):
            cm[y_val_pt[i].item()][preds[i].item()] += 1
        header = "            " + "".join(f"{g[:4]:>5s}" for g in gestures)
        print(header)
        for i, g in enumerate(gestures):
            row = "".join(f"{cm[i][j]:5d}" for j in range(len(gestures)))
            print(f"  {g:10s} {row}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

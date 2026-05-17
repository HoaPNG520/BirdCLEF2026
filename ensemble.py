"""
ensemble.py
===========
Blends EfficientNet and Perch+MLP predictions into a final submission.

Two modes:
  1. OOF blending  — finds the best weight ratio using OOF predictions
  2. Test blending — applies the best weights to test predictions and writes submission.csv

Usage (in Kaggle notebook):
    from ensemble import find_best_weights, blend_submissions

    # Step 1: find optimal weights on OOF
    best_w = find_best_weights(
        oof_effnet_path  = "/kaggle/working/oof_effnet.npy",
        oof_perch_path   = "/kaggle/working/oof_perch.npy",
        oof_labels_path  = "/kaggle/working/oof_labels.npy",
    )

    # Step 2: blend test predictions
    blend_submissions(
        effnet_sub_path = "/kaggle/working/submission_effnet.csv",
        perch_sub_path  = "/kaggle/working/submission_perch.csv",
        effnet_weight   = best_w,
        output_path     = "/kaggle/working/submission_ensemble.csv",
    )
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import average_precision_score

# ── Metric ─────────────────────────────────────────────────────────────────────


def padded_cmap(y_true, y_pred, padding=5):
    """BirdCLEF padded mean AP."""
    scores = []
    for i in range(y_true.shape[1]):
        yt = (y_true[:, i] > 0).astype(int)
        yp = y_pred[:, i]
        yt = np.concatenate([yt, np.zeros(padding)])
        yp = np.concatenate([yp, np.zeros(padding)])
        if yt.sum() > 0:
            scores.append(average_precision_score(yt, yp))
    return float(np.mean(scores)) if scores else 0.0


# ── OOF weight search ──────────────────────────────────────────────────────────


def find_best_weights(
    oof_effnet_path,
    oof_perch_path,
    oof_labels_path,
    n_steps=21,
):
    """
    Grid-search over blend weights [0.0, 0.05, ..., 1.0] using OOF predictions.
    Returns the effnet weight that maximises padded cMAP.

    Args:
        oof_effnet_path  : path to .npy of shape (N, 234) — EfficientNet OOF probs
        oof_perch_path   : path to .npy of shape (N, 234) — Perch+MLP OOF probs
        oof_labels_path  : path to .npy of shape (N, 234) — ground-truth binary labels
        n_steps          : number of weight steps to try (default 21 → 0.00, 0.05, ..., 1.00)

    Returns:
        best_effnet_weight (float)
    """
    oof_effnet = np.load(oof_effnet_path).astype(np.float32)
    oof_perch = np.load(oof_perch_path).astype(np.float32)
    oof_labels = np.load(oof_labels_path).astype(np.float32)

    print(
        f"OOF shapes — EfficientNet: {oof_effnet.shape} | Perch: {oof_perch.shape} | Labels: {oof_labels.shape}"
    )

    weights = np.linspace(0.0, 1.0, n_steps)
    best_cmap = 0.0
    best_weight = 0.5
    results = []

    for w in weights:
        blended = w * oof_effnet + (1 - w) * oof_perch
        cmap = padded_cmap(oof_labels, blended)
        results.append((w, cmap))
        if cmap > best_cmap:
            best_cmap = cmap
            best_weight = w

    print("\nWeight search results:")
    print(f"{'EfficientNet w':>15} | {'Perch w':>8} | {'cMAP':>8}")
    print("-" * 38)
    for w, cmap in results:
        marker = " ◀ best" if w == best_weight else ""
        print(f"{w:>15.2f} | {1-w:>8.2f} | {cmap:>8.4f}{marker}")

    print(
        f"\nBest blend → EfficientNet: {best_weight:.2f} | Perch: {1-best_weight:.2f}"
    )
    print(f"Best OOF cMAP: {best_cmap:.4f}")

    return best_weight


# ── Test prediction blending ───────────────────────────────────────────────────


def blend_submissions(
    effnet_sub_path,
    perch_sub_path,
    effnet_weight=0.5,
    thresholds_path=None,
    output_path="/kaggle/working/submission_ensemble.csv",
):
    """
    Blend two submission CSVs and write the final submission.

    Args:
        effnet_sub_path  : path to EfficientNet submission CSV (raw probs, no threshold)
        perch_sub_path   : path to Perch+MLP submission CSV  (raw probs, no threshold)
        effnet_weight    : weight for EfficientNet (Perch gets 1 - effnet_weight)
        thresholds_path  : optional path to thresholds.json for per-class thresholding
        output_path      : where to save the blended submission
    """
    effnet_sub = pd.read_csv(effnet_sub_path)
    perch_sub = pd.read_csv(perch_sub_path)

    assert list(effnet_sub.columns) == list(
        perch_sub.columns
    ), "Submission columns don't match — check label2idx consistency"
    assert len(effnet_sub) == len(
        perch_sub
    ), "Submissions have different number of rows"

    species_cols = [c for c in effnet_sub.columns if c != "row_id"]

    # Blend
    blended = effnet_sub.copy()
    blended[species_cols] = (
        effnet_weight * effnet_sub[species_cols].values
        + (1 - effnet_weight) * perch_sub[species_cols].values
    )

    # Apply per-class thresholds
    if thresholds_path and Path(thresholds_path).exists():
        import json

        with open(thresholds_path) as f:
            thresholds = json.load(f)
        print(f"Applying {len(thresholds)} per-class thresholds...")
        for col in species_cols:
            thresh = thresholds.get(col, 0.5)
            blended[col] = blended[col].where(blended[col] >= thresh, 0.0)
    else:
        # Fallback: global 0.5 threshold
        print("No thresholds file found — applying global threshold 0.5")
        blended[species_cols] = blended[species_cols].where(
            blended[species_cols] >= 0.5, 0.0
        )

    blended.to_csv(output_path, index=False)
    print(f"\n✅ Ensemble submission saved → {output_path}")
    print(f"   Shape: {blended.shape}")
    print(
        f"   EfficientNet weight: {effnet_weight:.2f} | Perch weight: {1-effnet_weight:.2f}"
    )

    return blended


# ── Quick sanity check ─────────────────────────────────────────────────────────


def check_submission(sub_path):
    """Print basic stats about a submission file."""
    sub = pd.read_csv(sub_path)
    species_cols = [c for c in sub.columns if c != "row_id"]
    vals = sub[species_cols].values

    print(f"Submission: {sub_path}")
    print(f"  Rows         : {len(sub)}")
    print(f"  Species cols : {len(species_cols)}")
    print(f"  Non-zero preds: {(vals > 0).sum():,}")
    print(
        f"  Mean prob    : {vals[vals > 0].mean():.4f}"
        if (vals > 0).any()
        else "  All zeros!"
    )
    print(f"  Max prob     : {vals.max():.4f}")

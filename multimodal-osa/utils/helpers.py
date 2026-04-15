"""
Utility functions for training, evaluation, and cross-validation.
"""

import os
import json
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
)
from sklearn.model_selection import KFold


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_prob: np.ndarray = None) -> dict:
    """
    Compute evaluation metrics.

    Args:
        y_true: (N,) ground truth binary labels
        y_pred: (N,) predicted binary labels
        y_prob: (N,) continuous predicted probabilities (optional).
                If None, AUC is not computed (e.g. for hard classifiers
                like majority voting that output 0/1 only).

    Returns:
        dict of metric name → value
    """
    metrics = {
        "accuracy":    accuracy_score(y_true, y_pred) * 100,
        "recall":      recall_score(y_true, y_pred, zero_division=0) * 100,
        "specificity": recall_score(y_true, y_pred, pos_label=0, zero_division=0) * 100,
        "precision":   precision_score(y_true, y_pred, zero_division=0) * 100,
        "f1":          f1_score(y_true, y_pred, zero_division=0) * 100,
        "mcc":         matthews_corrcoef(y_true, y_pred),
    }

    # AUC requires a continuous ranking score and both classes present
    if y_prob is not None and len(np.unique(y_true)) > 1:
        metrics["auc"] = roc_auc_score(y_true, y_prob) * 100

    return metrics


def create_patient_folds(patient_ids: list, num_folds: int = 10, seed: int = 42) -> list:
    """
    Create patient-wise cross-validation folds.

    Args:
        patient_ids: list of patient identifiers
        num_folds: number of folds (default 10)
        seed: random seed

    Returns:
        list of (train_ids, val_ids, test_ids) tuples
    """
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    patient_ids = np.array(patient_ids)

    folds = []
    fold_indices = list(kf.split(patient_ids))

    for i, (train_val_idx, test_idx) in enumerate(fold_indices):
        test_ids = patient_ids[test_idx].tolist()

        # Use next fold as validation
        val_fold_idx = (i + 1) % num_folds

        # Use the next fold's test patients as validation (always a subset of train_val)
        val_idx = np.intersect1d(train_val_idx, fold_indices[val_fold_idx][1])
        train_idx = np.setdiff1d(train_val_idx, val_idx)

        train_ids = patient_ids[train_idx].tolist()
        val_ids = patient_ids[val_idx].tolist()

        folds.append((train_ids, val_ids, test_ids))

    return folds


class EarlyStopping:
    """Early stopping based on validation metric."""

    def __init__(self, patience: int = 50, min_delta: float = 1e-4, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_results(results: dict, output_path: str):
    """Save results to JSON file."""
    dirname = os.path.dirname(output_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    class _NumpyEncoder(json.JSONEncoder):
        """Recursively convert numpy scalars/arrays to Python-native types."""
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, cls=_NumpyEncoder)


def print_metrics(metrics: dict, prefix: str = ""):
    """Pretty-print metrics."""
    parts = []
    for key in ["accuracy", "recall", "specificity", "precision", "f1", "auc", "mcc"]:
        if key in metrics:
            if key == "mcc":
                parts.append(f"{key.upper()}: {metrics[key]:.4f}")
            else:
                parts.append(f"{key.capitalize()}: {metrics[key]:.2f}%")
    print(f"{prefix}{' | '.join(parts)}")

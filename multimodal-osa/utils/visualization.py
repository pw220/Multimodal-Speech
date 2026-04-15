"""
Visualization utilities for the multimodal OSA framework.

- t-SNE visualization of learned representations (Sec 5.7, Fig 6)
- Confusion matrices for aggregation strategies (Fig 5)
- Hyperparameter sensitivity plots (Fig 4)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix


def _makedirs_safe(path: str):
    """Create parent directories for a file path, safely handling bare filenames."""
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)


def plot_tsne(
    features_before: np.ndarray,
    features_after: np.ndarray,
    labels: np.ndarray,
    save_path: str = None,
):
    """
    Plot t-SNE visualization of embeddings before and after multimodal learning (Fig 6).

    Args:
        features_before: (N, D) raw acoustic features
        features_after: (N, D) features after proposed framework
        labels: (N,) binary severity labels
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, feats, title in zip(
        axes,
        [features_before, features_after],
        ["(a) Raw acoustic features", "(b) Features after proposed framework"],
    ):
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(feats) - 1))
        embedded = tsne.fit_transform(feats)

        for label, color, name in [(0, "#7FB3D8", "Non-severe"), (1, "#E8845C", "Severe")]:
            mask = labels == label
            ax.scatter(
                embedded[mask, 0], embedded[mask, 1],
                c=color, alpha=0.6, s=20, label=name, edgecolors="none",
            )

        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")
        ax.set_title(title)
        ax.legend(loc="lower right")

    plt.tight_layout()
    if save_path:
        _makedirs_safe(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved t-SNE plot to {save_path}")
    plt.close()


def plot_confusion_matrices(
    y_true_mv: np.ndarray,
    y_pred_mv: np.ndarray,
    y_true_seq: np.ndarray,
    y_pred_seq: np.ndarray,
    save_path: str = None,
):
    """
    Plot confusion matrices for majority voting and sequence aggregator (Fig 5).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, y_true, y_pred, title in zip(
        axes,
        [y_true_mv, y_true_seq],
        [y_pred_mv, y_pred_seq],
        ["(a) Majority voting", "(b) Sequence aggregator"],
    ):
        cm = confusion_matrix(y_true, y_pred)
        total = cm.sum()

        # Percentage annotations
        annot = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f"{cm[i, j]}\n{cm[i, j] / total * 100:.1f}%"

        sns.heatmap(
            cm, annot=annot, fmt="", cmap="YlOrBr",
            xticklabels=["Non-severe", "Severe"],
            yticklabels=["Non-severe", "Severe"],
            ax=ax, cbar_kws={"label": "Number of samples"},
        )
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(title)

    plt.tight_layout()
    if save_path:
        _makedirs_safe(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved confusion matrices to {save_path}")
    plt.close()


def plot_hyperparameter_sensitivity(
    results: dict,
    save_path: str = None,
):
    """
    Plot hyperparameter sensitivity analysis (Fig 4).

    Args:
        results: dict with keys 'temperature', 'lambda', 'segment_length', 'embedding_dim',
                 each mapping to {'values': [...], 'accuracy': [...], 'auc': [...]}
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    param_configs = [
        ("temperature", r"(a) Temperature $\tau$", r"$\tau$"),
        ("lambda", r"(b) Contrastive loss weight $\lambda$", r"$\lambda$"),
        ("segment_length", "(c) Segment length", "Length (s)"),
        ("embedding_dim", "(d) Embedding dimension $d$", "$d$"),
    ]

    for ax, (key, title, xlabel) in zip(axes.flat, param_configs):
        if key not in results:
            ax.set_visible(False)
            continue

        data = results[key]
        values = data["values"]
        accuracy = data["accuracy"]
        auc = data["auc"]

        ax.plot(values, accuracy, "s-", color="#4A90D9", label="Accuracy", linewidth=2, markersize=8)
        ax.plot(values, auc, "o-", color="#333333", label="AUC", linewidth=2, markersize=8)

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Performance (%)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        _makedirs_safe(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved sensitivity plot to {save_path}")
    plt.close()


def plot_backbone_comparison(
    backbone_results: dict,
    save_path: str = None,
):
    """
    Plot comparison of different speech foundation model backbones (Fig 3).

    Args:
        backbone_results: {backbone_name: {'auc': float, 'accuracy': float,
                          'recall': float, 'f1': float}}
    """
    backbones = list(backbone_results.keys())
    metrics = ["AUC", "Acc.", "Recall", "F1"]
    metric_keys = ["auc", "accuracy", "recall", "f1"]

    x = np.arange(len(backbones))
    width = 0.18

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ["#333333", "#4A90D9", "#E8845C", "#5CB85C"]

    for i, (metric, key, color) in enumerate(zip(metrics, metric_keys, colors)):
        values = [backbone_results[b][key] for b in backbones]
        bars = ax.bar(x + i * width, values, width, label=metric, color=color)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Speech Foundation Model")
    ax.set_ylabel("Performance (%)")
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(backbones)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Mark best
    best_backbone = max(backbones, key=lambda b: backbone_results[b]["auc"])
    ax.annotate("★ best", xy=(backbones.index(best_backbone), backbone_results[best_backbone]["auc"]),
                fontsize=10, ha="center")

    plt.tight_layout()
    if save_path:
        _makedirs_safe(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved backbone comparison to {save_path}")
    plt.close()

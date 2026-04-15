"""
End-to-end pipeline test using synthetic sample data and mock encoders.

Validates every component without downloading large pretrained models:
  - data generation (generate_demo_data)
  - contrastive losses (CrossModalContrastiveLoss, SeverityAwareContrastiveLoss)
  - CGF fusion (ClinicallyGuidedFusion)
  - full DownstreamModel forward + loss computation
  - train_one_epoch / evaluate
  - patient-level aggregation (MajorityVoting, StatisticalSequenceAggregator)
  - create_patient_folds
  - save_results (JSON serialisation of numpy scalars)
  - visualization helpers (save to temp files)

Run:
    python test_pipeline.py
"""

import os
import sys
import json
import shutil
import tempfile
import traceback

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader

# Insert project root (parent of tests/) so all packages resolve correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.generate_demo_data import generate_demo_data
from models.contrastive import CrossModalContrastiveLoss, SeverityAwareContrastiveLoss
from models.fusion import ClinicallyGuidedFusion
from models.framework import (
    TemporalAttentionAggregation,
    ProjectionHead,
    SupervisedContrastiveProjection,
    PredictionHead,
)
from models.aggregation import MajorityVoting, StatisticalSequenceAggregator
from utils.trainer import DownstreamModel, PrecomputedFeatureDataset, train_one_epoch, evaluate
from utils.helpers import (
    compute_metrics,
    create_patient_folds,
    EarlyStopping,
    set_seed,
    save_results,
)
from utils.visualization import (
    plot_tsne,
    plot_confusion_matrices,
    plot_hyperparameter_sensitivity,
)

# ---------------------------------------------------------------------------
# Tiny mock dimensions (no pretrained models needed)
# ---------------------------------------------------------------------------
SPEECH_HIDDEN = 64    # replaces 1024 (XLS-R)
TEXT_HIDDEN   = 48    # replaces 768  (ClinicalBERT)
EMBED_DIM     = 16
PRED_HIDDEN   = 8
SUP_DIM       = 8
SEQ_LEN       = 12   # temporal frames per segment
BATCH         = 8
N_PATIENTS    = 20

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def _result(ok: bool, name: str, detail: str = ""):
    tag = PASS if ok else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{tag}] {name}{suffix}")
    return ok


# ---------------------------------------------------------------------------
# 1. Demo data generation
# ---------------------------------------------------------------------------
def test_generate_demo_data(tmpdir):
    generate_demo_data(tmpdir, num_patients=N_PATIENTS, seed=0)
    csv_path = os.path.join(tmpdir, "clinical_profiles.csv")
    audio_dir = os.path.join(tmpdir, "audio")

    df = pd.read_csv(csv_path)
    ok_csv = len(df) == N_PATIENTS and set(df.columns) >= {
        "patient_id", "age", "gender", "bmi",
        "neck_circumference", "waist_circumference",
        "ess_score", "psqi_score", "ahi",
    }
    ok_audio = len(os.listdir(audio_dir)) == N_PATIENTS * 2  # pre + post
    return (
        _result(ok_csv,   "generate_demo_data: CSV columns & row count"),
        _result(ok_audio, "generate_demo_data: audio file count"),
        df,
        audio_dir,
    )


# ---------------------------------------------------------------------------
# 2. Contrastive losses
# ---------------------------------------------------------------------------
def test_contrastive_losses():
    set_seed(0)
    B, D = BATCH, EMBED_DIM
    z_a = torch.randn(B, D)
    z_a = nn.functional.normalize(z_a, dim=-1)
    z_t = torch.randn(B, D)
    z_t = nn.functional.normalize(z_t, dim=-1)

    # Four distinct patients, two segments each
    pids = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    labels = torch.tensor([1, 1, 0, 0, 1, 1, 0, 0])

    cml = CrossModalContrastiveLoss(temperature=0.1)
    loss_cml = cml(z_a, z_t, pids)
    ok_cml = loss_cml.item() > 0 and torch.isfinite(loss_cml)

    sup = SeverityAwareContrastiveLoss(temperature=0.1)
    r = nn.functional.normalize(torch.randn(B, D), dim=-1)
    loss_sup = sup(r, labels, pids)
    ok_sup = loss_sup.item() >= 0 and torch.isfinite(loss_sup)

    # Edge case: single patient in batch → CML should return 0
    pids_single = torch.zeros(4, dtype=torch.long)
    loss_single = cml(z_a[:4], z_t[:4], pids_single)
    ok_single = loss_single.item() == 0.0

    return (
        _result(ok_cml,    "CrossModalContrastiveLoss: positive scalar"),
        _result(ok_sup,    "SeverityAwareContrastiveLoss: non-negative scalar"),
        _result(ok_single, "CrossModalContrastiveLoss: single-patient batch → 0"),
    )


# ---------------------------------------------------------------------------
# 3. CGF fusion
# ---------------------------------------------------------------------------
def test_cgf():
    set_seed(0)
    B, D = BATCH, EMBED_DIM
    cgf = ClinicallyGuidedFusion(D)
    z_a = torch.randn(B, D)
    z_t = torch.randn(B, D)
    h, z_f = cgf(z_a, z_t)
    ok_shape_h  = tuple(h.shape)  == (B, 2 * D)
    ok_shape_zf = tuple(z_f.shape) == (B, D)
    return (
        _result(ok_shape_h,  f"CGF: h shape {tuple(h.shape)} == ({B}, {2*D})"),
        _result(ok_shape_zf, f"CGF: z_f shape {tuple(z_f.shape)} == ({B}, {D})"),
    )


# ---------------------------------------------------------------------------
# 4. DownstreamModel forward (train + eval)
# ---------------------------------------------------------------------------
def test_downstream_model():
    set_seed(0)
    model = DownstreamModel(
        speech_hidden_dim=SPEECH_HIDDEN,
        text_hidden_dim=TEXT_HIDDEN,
        embedding_dim=EMBED_DIM,
        prediction_hidden_dim=PRED_HIDDEN,
        sup_contrastive_dim=SUP_DIM,
    )

    speech_frames = torch.randn(BATCH, SEQ_LEN, SPEECH_HIDDEN)
    clinical_emb  = torch.randn(BATCH, TEXT_HIDDEN)
    # Two patients, four segments each
    labels      = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0])
    patient_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

    # Training forward
    out_train = model(speech_frames, clinical_emb, labels, patient_ids)
    ok_keys = {"y_hat", "loss_total", "loss_cls", "loss_cml", "loss_sup"}.issubset(out_train)
    ok_loss = torch.isfinite(out_train["loss_total"])
    ok_yhat = tuple(out_train["y_hat"].shape) == (BATCH,)

    # Backward pass
    try:
        out_train["loss_total"].backward()
        ok_backward = True
    except Exception as e:
        ok_backward = False

    # Inference forward (no labels)
    model.eval()
    with torch.no_grad():
        out_eval = model(speech_frames, clinical_emb)
    ok_eval = "y_hat" in out_eval and "loss_total" not in out_eval
    probs = out_eval["y_hat"].numpy()
    ok_range = ((probs >= 0) & (probs <= 1)).all()

    return (
        _result(ok_keys,     "DownstreamModel: output keys present"),
        _result(ok_loss,     f"DownstreamModel: finite total loss ({out_train['loss_total'].item():.4f})"),
        _result(ok_yhat,     f"DownstreamModel: y_hat shape ({BATCH},)"),
        _result(ok_backward, "DownstreamModel: backward pass"),
        _result(ok_eval,     "DownstreamModel: eval mode (no loss keys)"),
        _result(ok_range,    "DownstreamModel: probs in [0, 1]"),
    )


# ---------------------------------------------------------------------------
# 5. train_one_epoch + evaluate (with precomputed features)
# ---------------------------------------------------------------------------
def _make_features(n_samples: int, n_patients: int) -> list:
    """Build a list of synthetic precomputed feature dicts."""
    set_seed(1)
    feats = []
    for i in range(n_samples):
        pid = i % n_patients
        feats.append({
            "speech_frames": torch.randn(SEQ_LEN, SPEECH_HIDDEN),
            "clinical_emb":  torch.randn(TEXT_HIDDEN),
            "label":         pid % 2,       # alternating labels
            "patient_id":    pid,
            "patient_id_str": f"patient_{pid:03d}",
        })
    return feats


def test_trainer():
    set_seed(0)
    n_patients = 4
    features = _make_features(n_samples=40, n_patients=n_patients)

    loader = DataLoader(
        PrecomputedFeatureDataset(features),
        batch_size=BATCH, shuffle=True, drop_last=True,
    )
    small_loader = DataLoader(          # fewer samples than batch → drop_last empties it
        PrecomputedFeatureDataset(features[:3]),
        batch_size=BATCH, shuffle=False, drop_last=True,
    )

    model = DownstreamModel(
        speech_hidden_dim=SPEECH_HIDDEN,
        text_hidden_dim=TEXT_HIDDEN,
        embedding_dim=EMBED_DIM,
        prediction_hidden_dim=PRED_HIDDEN,
        sup_contrastive_dim=SUP_DIM,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    device = torch.device("cpu")

    losses = train_one_epoch(model, loader, opt, device)
    ok_train = all(k in losses for k in ("loss", "loss_cls", "loss_cml", "loss_sup"))
    ok_finite = all(np.isfinite(v) for v in losses.values())

    # Division-by-zero guard (drop_last empties the loader)
    try:
        losses_empty = train_one_epoch(model, small_loader, opt, device)
        ok_empty = losses_empty["loss"] == 0.0
    except ZeroDivisionError:
        ok_empty = False

    metrics, probs, labels, pids = evaluate(model, loader, device)
    ok_eval = set(metrics) >= {"accuracy", "recall", "auc", "mcc"}
    ok_probs = len(probs) == len(features) and ((probs >= 0) & (probs <= 1)).all()

    return (
        _result(ok_train,  "train_one_epoch: loss keys present"),
        _result(ok_finite, f"train_one_epoch: finite losses {losses}"),
        _result(ok_empty,  "train_one_epoch: drop_last empty loader → returns 0s (no ZeroDivisionError)"),
        _result(ok_eval,   "evaluate: metric keys present"),
        _result(ok_probs,  "evaluate: probs shape & range"),
    )


# ---------------------------------------------------------------------------
# 6. Patient-level aggregation
# ---------------------------------------------------------------------------
def test_aggregation():
    rng = np.random.default_rng(42)

    # --- MajorityVoting ---
    probs_severe = rng.uniform(0.6, 1.0, 20)   # mostly severe
    probs_mild   = rng.uniform(0.0, 0.4, 20)   # mostly non-severe
    mv = MajorityVoting()
    pred_s = mv(probs_severe)
    pred_m = mv(probs_mild)
    ok_mv_severe = pred_s == 1
    ok_mv_mild   = pred_m == 0

    # --- StatisticalSequenceAggregator ---
    n_patients = 30
    patient_probs = [rng.uniform(0.5 + 0.4 * (i % 2), 0.9 + 0.1 * (i % 2), 15)
                     for i in range(n_patients)]
    patient_labels = np.array([i % 2 for i in range(n_patients)])

    train_probs  = patient_probs[:20]
    train_labels = patient_labels[:20]
    test_probs   = patient_probs[20:]
    test_labels  = patient_labels[20:]

    agg = StatisticalSequenceAggregator()
    agg.fit(train_probs, train_labels)
    preds, probs_out = agg.predict_batch(test_probs)
    ok_agg_shape = len(preds) == len(test_labels)
    ok_agg_range = ((probs_out >= 0) & (probs_out <= 1)).all()
    ok_agg_binary = set(preds).issubset({0, 1})

    # extract_features shape
    feats = agg.extract_features(probs_severe)
    ok_feat_shape = feats.shape == (8,)

    return (
        _result(ok_mv_severe,  "MajorityVoting: severe probs → pred=1"),
        _result(ok_mv_mild,    "MajorityVoting: mild probs → pred=0"),
        _result(ok_agg_shape,  "StatisticalSequenceAggregator: predict_batch shape"),
        _result(ok_agg_range,  "StatisticalSequenceAggregator: probs in [0,1]"),
        _result(ok_agg_binary, "StatisticalSequenceAggregator: binary predictions"),
        _result(ok_feat_shape, "StatisticalSequenceAggregator: feature vector length=8"),
    )


# ---------------------------------------------------------------------------
# 7. create_patient_folds
# ---------------------------------------------------------------------------
def test_folds():
    n = 20
    pids = [f"p{i:02d}" for i in range(n)]
    folds = create_patient_folds(pids, num_folds=5, seed=0)

    ok_count = len(folds) == 5

    all_test_ids = []
    ok_disjoint = True
    ok_sizes = True
    for train_ids, val_ids, test_ids in folds:
        # Test set must not overlap with train or val
        overlap = set(test_ids) & (set(train_ids) | set(val_ids))
        if overlap:
            ok_disjoint = False
        if len(test_ids) == 0:
            ok_sizes = False
        all_test_ids.extend(test_ids)

    # Every patient appears in exactly one test fold
    ok_coverage = sorted(all_test_ids) == sorted(pids)

    return (
        _result(ok_count,    f"create_patient_folds: 5 folds created"),
        _result(ok_disjoint, "create_patient_folds: test ∩ train/val = ∅"),
        _result(ok_sizes,    "create_patient_folds: all test sets non-empty"),
        _result(ok_coverage, "create_patient_folds: every patient appears in exactly one test fold"),
    )


# ---------------------------------------------------------------------------
# 8. save_results (numpy scalar serialisation)
# ---------------------------------------------------------------------------
def test_save_results(tmpdir):
    results = {
        "config": {"lr": 1e-3, "epochs": 200},
        "fold_results": [
            {
                "segment_level":      {"accuracy": np.float64(75.3), "auc": np.float64(80.1), "mcc": np.float64(0.42)},
                "majority_voting":    {"accuracy": np.float64(80.0), "auc": np.float64(85.2), "mcc": np.float64(0.55)},
                "sequence_aggregator":{"accuracy": np.float64(82.5), "auc": np.float64(88.0), "mcc": np.float64(0.60)},
            }
        ],
        "num_folds_completed": np.int64(1),
    }

    out_path = os.path.join(tmpdir, "sub", "results.json")
    try:
        save_results(results, out_path)
        with open(out_path) as f:
            loaded = json.load(f)
        ok_saved  = loaded["num_folds_completed"] == 1
        ok_nested = abs(loaded["fold_results"][0]["segment_level"]["accuracy"] - 75.3) < 1e-6
        ok_mcc    = abs(loaded["fold_results"][0]["segment_level"]["mcc"] - 0.42) < 1e-6
        ok_dir    = True
    except Exception as e:
        ok_saved = ok_nested = ok_mcc = ok_dir = False
        print(f"    ERROR: {e}")

    # Bare filename (no directory component) must not crash
    bare_path = os.path.join(tmpdir, "bare.json")
    try:
        save_results({"x": np.float64(1.0)}, bare_path)
        ok_bare = True
    except Exception as e:
        ok_bare = False
        print(f"    ERROR (bare path): {e}")

    return (
        _result(ok_saved,  "save_results: file created and loadable"),
        _result(ok_nested, "save_results: nested np.float64 serialised correctly"),
        _result(ok_mcc,    "save_results: np.float64 mcc serialised correctly"),
        _result(ok_dir,    "save_results: parent directory auto-created"),
        _result(ok_bare,   "save_results: bare filename (no dir) does not crash"),
    )


# ---------------------------------------------------------------------------
# 9. Visualization (save to temp files, no display)
# ---------------------------------------------------------------------------
def test_visualization(tmpdir):
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend

    rng = np.random.default_rng(7)
    N = 40
    feats  = rng.standard_normal((N, 8))
    labels = (rng.random(N) > 0.5).astype(int)
    y_pred = (rng.random(N) > 0.5).astype(int)

    tsne_path = os.path.join(tmpdir, "plots", "tsne.png")
    cm_path   = os.path.join(tmpdir, "plots", "cm.png")
    sens_path = os.path.join(tmpdir, "plots", "sensitivity.png")
    bare_path = os.path.join(tmpdir, "bare_tsne.png")

    results_data = {
        "temperature": {"values": [0.01, 0.03, 0.1], "accuracy": [70., 75., 72.], "auc": [75., 80., 77.]},
        "lambda":      {"values": [0.3, 0.5, 0.7],   "accuracy": [71., 76., 74.], "auc": [76., 81., 78.]},
    }

    try:
        plot_tsne(feats, feats + 0.1, labels, save_path=tsne_path)
        ok_tsne = os.path.exists(tsne_path)
    except Exception as e:
        ok_tsne = False; print(f"    ERROR tsne: {e}")

    try:
        plot_confusion_matrices(labels, y_pred, labels, y_pred, save_path=cm_path)
        ok_cm = os.path.exists(cm_path)
    except Exception as e:
        ok_cm = False; print(f"    ERROR cm: {e}")

    try:
        plot_hyperparameter_sensitivity(results_data, save_path=sens_path)
        ok_sens = os.path.exists(sens_path)
    except Exception as e:
        ok_sens = False; print(f"    ERROR sensitivity: {e}")

    try:
        plot_tsne(feats, feats, labels, save_path=bare_path)
        ok_bare = os.path.exists(bare_path)
    except Exception as e:
        ok_bare = False; print(f"    ERROR bare: {e}")

    return (
        _result(ok_tsne, "plot_tsne: file written"),
        _result(ok_cm,   "plot_confusion_matrices: file written"),
        _result(ok_sens, "plot_hyperparameter_sensitivity: file written"),
        _result(ok_bare, "plot_tsne: bare filename (no dir) does not crash"),
    )


# ---------------------------------------------------------------------------
# 10. EarlyStopping
# ---------------------------------------------------------------------------
def test_early_stopping():
    es = EarlyStopping(patience=3, min_delta=0.01, mode="max")
    scores = [0.5, 0.55, 0.56, 0.56, 0.56, 0.56]  # stops after 3 non-improvements
    stopped_at = None
    for i, s in enumerate(scores):
        if es(s):
            stopped_at = i
            break
    ok_stopped = stopped_at == 4   # triggered on 5th call (index 4)

    es2 = EarlyStopping(patience=3, min_delta=0.01, mode="min")
    # Steps 0-2: improvements (1.0 → 0.9 → 0.8); steps 3-5: no improvement
    # patience counter hits 3 at step 5 → triggered at index 5
    scores2 = [1.0, 0.9, 0.8, 0.8, 0.8, 0.8]
    stopped_at2 = None
    for i, s in enumerate(scores2):
        if es2(s):
            stopped_at2 = i
            break
    ok_min = stopped_at2 == 5

    return (
        _result(ok_stopped, f"EarlyStopping (max mode): stopped at step {stopped_at}"),
        _result(ok_min,     f"EarlyStopping (min mode): stopped at step {stopped_at2} (expected 5)"),
    )


# ---------------------------------------------------------------------------
# 11. compute_metrics
# ---------------------------------------------------------------------------
def test_compute_metrics():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 100)
    y_pred = rng.integers(0, 2, 100)
    y_prob = rng.uniform(0, 1, 100)

    metrics = compute_metrics(y_true, y_pred, y_prob)
    expected_keys = {"accuracy", "recall", "specificity", "precision", "f1", "auc", "mcc"}
    ok_keys   = expected_keys.issubset(metrics)
    ok_range  = all(0 <= v <= 100 for k, v in metrics.items() if k != "mcc")
    ok_mcc    = -1 <= metrics["mcc"] <= 1
    ok_types  = all(isinstance(v, float) for v in metrics.values())

    # Single-class edge case: AUC should be absent (not computable) without crashing
    y_single = np.zeros(10, dtype=int)
    y_pred_s = np.zeros(10, dtype=int)
    y_prob_s = np.zeros(10)
    m_single = compute_metrics(y_single, y_pred_s, y_prob_s)
    ok_single = "auc" not in m_single  # single class → AUC skipped

    return (
        _result(ok_keys,   f"compute_metrics: all expected keys present"),
        _result(ok_range,  "compute_metrics: percentage metrics in [0,100]"),
        _result(ok_mcc,    f"compute_metrics: MCC in [-1,1] ({metrics['mcc']:.4f})"),
        _result(ok_types,  "compute_metrics: all values are Python floats"),
        _result(ok_single, "compute_metrics: single-class → AUC key absent (no crash)"),
    )


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------
def main():
    print("\n" + "=" * 62)
    print("  Multimodal OSA Pipeline Test Suite")
    print("=" * 62)

    set_seed(42)
    tmpdir = tempfile.mkdtemp(prefix="osa_test_")
    all_results = []

    sections = [
        ("Demo data generation",         lambda: test_generate_demo_data(tmpdir)[:2]),
        ("Contrastive losses",            test_contrastive_losses),
        ("Clinically Guided Fusion",      test_cgf),
        ("DownstreamModel forward/backward", test_downstream_model),
        ("Trainer (train_one_epoch + evaluate)", test_trainer),
        ("Patient-level aggregation",     test_aggregation),
        ("create_patient_folds",          test_folds),
        ("save_results (JSON serialisation)", lambda: test_save_results(tmpdir)),
        ("Visualization",                 lambda: test_visualization(tmpdir)),
        ("EarlyStopping",                 test_early_stopping),
        ("compute_metrics",               test_compute_metrics),
    ]

    for title, fn in sections:
        print(f"\n{title}")
        try:
            results = fn()
            all_results.extend(results)
        except Exception:
            print(f"  [{FAIL}] Section raised an unexpected exception:")
            traceback.print_exc()

    # Summary
    passed = sum(1 for r in all_results if r)
    total  = len(all_results)
    failed = total - passed

    print("\n" + "=" * 62)
    print(f"  Results: {passed}/{total} passed", end="")
    if failed:
        print(f"  ({failed} FAILED)")
    else:
        print("  — all green")
    print("=" * 62)

    # Write a machine-readable summary
    summary = {
        "total": int(total),
        "passed": int(passed),
        "failed": int(failed),
    }
    summary_path = os.path.join(tmpdir, "test_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Artefacts in: {tmpdir}")
    print(f"  Summary:      {summary_path}\n")

    shutil.rmtree(tmpdir, ignore_errors=True)
    return failed == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)

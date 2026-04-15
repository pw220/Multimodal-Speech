"""
Main training script for the multimodal OSA severity estimation framework.

Implements the full 10-fold patient-wise cross-validation pipeline (Sec 4.2):
1. Precompute frozen encoder features (speech + clinical)
2. Train downstream model with dual contrastive learning + CGF
3. Evaluate at segment-level
4. Aggregate to patient-level with majority voting + statistical sequence aggregation
5. Report metrics with mean ± std across folds

Usage:
    python main.py \
        --audio_dir data/demo/audio \
        --clinical_csv data/demo/clinical_profiles.csv \
        --output_dir experiments/demo \
        --num_folds 10 \
        --epochs 200 \
        --batch_size 64
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict

from configs.config import Config, ModelConfig, TrainingConfig, DataConfig
from models.encoders import SpeechEncoder, TextEncoder, build_clinical_prompt
from models.aggregation import MajorityVoting, StatisticalSequenceAggregator
from utils.helpers import (
    compute_metrics,
    create_patient_folds,
    EarlyStopping,
    set_seed,
    save_results,
    print_metrics,
)
from utils.trainer import (
    DownstreamModel,
    PrecomputedFeatureDataset,
    train_one_epoch,
    evaluate,
)


def load_and_segment_audio(
    patient_ids: list,
    audio_dir: str,
    clinical_df: pd.DataFrame,
    sample_rate: int = 16000,
    segment_length: float = 4.0,
    segment_overlap: float = 0.5,
    ahi_threshold: float = 30.0,
    conditions: list = None,
) -> list:
    """
    Load audio files and create segments for a set of patients.

    Returns list of dicts with waveform tensors and metadata.
    """
    import soundfile as sf
    import librosa

    if conditions is None:
        conditions = ["pre", "post"]

    segment_samples = int(segment_length * sample_rate)
    hop_samples = int(segment_samples * (1.0 - segment_overlap))

    segments = []
    patient_id_map = {}

    for pid_idx, pid in enumerate(patient_ids):
        patient_id_map[pid] = pid_idx
        row = clinical_df.loc[clinical_df["patient_id"] == pid].iloc[0]
        label = 1 if row["ahi"] >= ahi_threshold else 0

        for cond in conditions:
            audio_path = os.path.join(audio_dir, f"{pid}_{cond}.wav")
            if not os.path.exists(audio_path):
                continue

            # Load full audio
            waveform, orig_sr = sf.read(audio_path, dtype="float32")
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)

            # Resample to 16kHz
            if orig_sr != sample_rate:
                waveform = librosa.resample(waveform, orig_sr=orig_sr, target_sr=sample_rate)

            # Normalize to [-1, 1]
            max_val = np.abs(waveform).max()
            if max_val > 0:
                waveform = waveform / max_val

            total_samples = len(waveform)

            # Segment with sliding window
            start = 0
            seg_idx = 0
            while start + segment_samples <= total_samples:
                seg = waveform[start: start + segment_samples]
                segments.append({
                    "waveform": torch.from_numpy(seg).float(),
                    "patient_id_str": pid,
                    "patient_id": pid_idx,
                    "label": label,
                    "condition": cond,
                    "segment_idx": seg_idx,
                })
                start += hop_samples
                seg_idx += 1

    return segments, patient_id_map


def precompute_all_features(
    segments: list,
    clinical_df: pd.DataFrame,
    patient_ids: list,
    speech_encoder: SpeechEncoder,
    text_encoder: TextEncoder,
    device: torch.device,
    batch_size: int = 16,
) -> list:
    """Precompute frozen encoder outputs for all segments."""
    speech_encoder.eval()
    text_encoder.eval()

    # Precompute clinical embeddings
    clinical_embeddings = {}
    for pid in patient_ids:
        row = clinical_df.loc[clinical_df["patient_id"] == pid].iloc[0]
        prompt = build_clinical_prompt(
            age=row["age"],
            gender=row["gender"],
            bmi=row["bmi"],
            neck_circ=row["neck_circumference"],
            waist_circ=row["waist_circumference"],
            ess_score=row["ess_score"],
            psqi_score=row["psqi_score"],
        )
        tokens = text_encoder.tokenizer(
            prompt, padding="max_length", truncation=True,
            max_length=128, return_tensors="pt"
        )
        input_ids = tokens["input_ids"].to(device)
        attn_mask = tokens["attention_mask"].to(device)

        with torch.no_grad():
            emb = text_encoder(input_ids, attn_mask).squeeze(0).cpu()
        clinical_embeddings[pid] = emb

    # Precompute speech features in batches
    features_list = []
    for i in tqdm(range(0, len(segments), batch_size), desc="Precomputing features", leave=False):
        batch_segs = segments[i: i + batch_size]
        waveforms = torch.stack([s["waveform"] for s in batch_segs]).to(device)

        with torch.no_grad():
            H_a = speech_encoder(waveforms).cpu()

        for j, seg in enumerate(batch_segs):
            features_list.append({
                "speech_frames": H_a[j],
                "clinical_emb": clinical_embeddings[seg["patient_id_str"]],
                "label": seg["label"],
                "patient_id": seg["patient_id"],
                "patient_id_str": seg["patient_id_str"],
            })

    return features_list


def run_fold(
    fold_idx: int,
    train_features: list,
    val_features: list,
    test_features: list,
    test_patient_ids: list,
    train_patient_ids: list,
    clinical_df: pd.DataFrame,
    config: Config,
    device: torch.device,
) -> dict:
    """Run training and evaluation for a single fold."""
    print(f"\n{'=' * 60}")
    print(f"  Fold {fold_idx + 1}/{config.training.num_folds}")
    print(f"  Train: {len(train_features)} segments | Val: {len(val_features)} | Test: {len(test_features)}")
    print(f"{'=' * 60}")

    # Create data loaders
    train_dataset = PrecomputedFeatureDataset(train_features)
    val_dataset = PrecomputedFeatureDataset(val_features)
    test_dataset = PrecomputedFeatureDataset(test_features)

    train_loader = DataLoader(
        train_dataset, batch_size=config.training.batch_size,
        shuffle=True, num_workers=0, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.training.batch_size,
        shuffle=False, num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.training.batch_size,
        shuffle=False, num_workers=0,
    )

    # Initialize downstream model
    model = DownstreamModel(
        speech_hidden_dim=config.model.speech_hidden_dim,
        text_hidden_dim=config.model.text_hidden_dim,
        embedding_dim=config.model.embedding_dim,
        prediction_hidden_dim=config.model.prediction_hidden_dim,
        sup_contrastive_dim=config.model.sup_contrastive_dim,
        temperature=config.training.temperature,
        temperature_sup=config.training.temperature_sup,
        lambda_contrastive=config.training.lambda_contrastive,
        label_smoothing=config.training.label_smoothing,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.lr,
        betas=config.training.betas,
        weight_decay=config.training.weight_decay,
    )

    early_stopping = EarlyStopping(
        patience=config.training.patience,
        min_delta=config.training.min_delta,
        mode="max",
    )

    best_val_auc = 0
    best_model_state = None

    # Training loop
    for epoch in range(config.training.epochs):
        train_losses = train_one_epoch(model, train_loader, optimizer, device)

        # Validate
        val_metrics, _, _, _ = evaluate(model, val_loader, device)

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:4d} | Loss: {train_losses['loss']:.4f} "
                  f"(cls: {train_losses['loss_cls']:.4f}, "
                  f"cml: {train_losses['loss_cml']:.4f}, "
                  f"sup: {train_losses['loss_sup']:.4f}) | "
                  f"Val AUC: {val_metrics['auc']:.2f}%")

        if early_stopping(val_metrics["auc"]):
            print(f"  Early stopping at epoch {epoch + 1}")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    # Evaluate on test set
    test_metrics, test_probs, test_labels, test_pids = evaluate(model, test_loader, device)

    print(f"\n  Segment-level test results:")
    print_metrics(test_metrics, prefix="    ")

    # Patient-level aggregation
    # Group predictions by patient
    patient_probs = defaultdict(list)
    patient_labels_map = {}

    for prob, label, pid in zip(test_probs, test_labels, test_pids):
        # Map numeric pid back to string pid
        pid_str = None
        for seg in test_features:
            if seg["patient_id"] == pid:
                pid_str = seg["patient_id_str"]
                break
        if pid_str is None:
            continue
        patient_probs[pid_str].append(prob)
        patient_labels_map[pid_str] = label

    # Majority voting — hard 0/1 output, no AUC
    mv = MajorityVoting()
    mv_preds = []
    mv_labels = []

    for pid in test_patient_ids:
        if pid not in patient_probs:
            continue
        pred = mv(np.array(patient_probs[pid]))
        mv_preds.append(pred)
        mv_labels.append(patient_labels_map[pid])

    mv_preds = np.array(mv_preds)
    mv_labels = np.array(mv_labels)
    mv_metrics = compute_metrics(mv_labels, mv_preds)  # no y_prob → no AUC

    print(f"\n  Majority voting patient-level:")
    print_metrics(mv_metrics, prefix="    ")

    # Statistical sequence aggregator
    # Fit on training patient predictions
    # We need to get training predictions too
    train_metrics_out, train_probs_out, train_labels_out, train_pids_out = evaluate(
        model, train_loader, device
    )

    train_patient_probs = defaultdict(list)
    train_patient_labels_map = {}
    for prob, label, pid in zip(train_probs_out, train_labels_out, train_pids_out):
        pid_str = None
        for seg in train_features:
            if seg["patient_id"] == pid:
                pid_str = seg["patient_id_str"]
                break
        if pid_str is None:
            continue
        train_patient_probs[pid_str].append(prob)
        train_patient_labels_map[pid_str] = label

    # Prepare data for aggregator fitting
    agg_train_probs = []
    agg_train_labels = []
    for pid in train_patient_ids:
        if pid in train_patient_probs:
            agg_train_probs.append(np.array(train_patient_probs[pid]))
            agg_train_labels.append(train_patient_labels_map[pid])

    agg_train_labels = np.array(agg_train_labels)

    seq_agg = StatisticalSequenceAggregator()

    # Only fit if we have both classes
    if len(np.unique(agg_train_labels)) > 1 and len(agg_train_probs) > 1:
        seq_agg.fit(agg_train_probs, agg_train_labels)

        # Predict on test patients
        test_agg_probs = []
        test_agg_labels = []
        test_agg_patient_probs = []

        for pid in test_patient_ids:
            if pid not in patient_probs:
                continue
            test_agg_patient_probs.append(np.array(patient_probs[pid]))
            test_agg_labels.append(patient_labels_map[pid])

        test_agg_labels = np.array(test_agg_labels)
        seq_preds, seq_probs = seq_agg.predict_batch(test_agg_patient_probs)
        seq_metrics = compute_metrics(test_agg_labels, seq_preds, seq_probs)
    else:
        seq_metrics = mv_metrics.copy()

    print(f"\n  Sequence aggregator patient-level:")
    print_metrics(seq_metrics, prefix="    ")

    return {
        "segment_level": test_metrics,
        "majority_voting": mv_metrics,
        "sequence_aggregator": seq_metrics,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Multimodal contrastive learning for speech-based OSA severity estimation"
    )

    # Data
    parser.add_argument("--audio_dir", type=str, required=True, help="Directory containing audio files")
    parser.add_argument("--clinical_csv", type=str, required=True, help="Path to clinical profiles CSV")
    parser.add_argument("--output_dir", type=str, default="experiments/default", help="Output directory")
    parser.add_argument("--conditions", type=str, nargs="+", default=["pre", "post"],
                        help="Recording conditions to use")

    # Model
    parser.add_argument("--speech_model", type=str, default="facebook/wav2vec2-xls-r-300m")
    parser.add_argument("--text_model", type=str, default="emilyalsentzer/Bio_ClinicalBERT")
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--prediction_hidden_dim", type=int, default=64)
    parser.add_argument("--sup_contrastive_dim", type=int, default=64)

    # Training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--temperature", type=float, default=0.03)
    parser.add_argument("--temperature_sup", type=float, default=0.03)
    parser.add_argument("--lambda_contrastive", type=float, default=0.7)
    parser.add_argument("--label_smoothing", type=float, default=0.1)

    # Cross-validation
    parser.add_argument("--num_folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    # Audio
    parser.add_argument("--segment_length", type=float, default=4.0)
    parser.add_argument("--segment_overlap", type=float, default=0.5)
    parser.add_argument("--ahi_threshold", type=float, default=30.0)

    # Misc
    parser.add_argument("--precompute_batch_size", type=int, default=8)
    parser.add_argument("--patience", type=int, default=50)

    args = parser.parse_args()

    # Build config
    config = Config(
        model=ModelConfig(
            speech_model_name=args.speech_model,
            text_model_name=args.text_model,
            embedding_dim=args.embedding_dim,
            prediction_hidden_dim=args.prediction_hidden_dim,
            sup_contrastive_dim=args.sup_contrastive_dim,
        ),
        training=TrainingConfig(
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            batch_size=args.batch_size,
            temperature=args.temperature,
            temperature_sup=args.temperature_sup,
            lambda_contrastive=args.lambda_contrastive,
            label_smoothing=args.label_smoothing,
            num_folds=args.num_folds,
            seed=args.seed,
            patience=args.patience,
        ),
        data=DataConfig(
            audio_dir=args.audio_dir,
            clinical_csv=args.clinical_csv,
            output_dir=args.output_dir,
            segment_length=args.segment_length,
            segment_overlap=args.segment_overlap,
            ahi_threshold=args.ahi_threshold,
        ),
    )

    set_seed(config.training.seed)
    os.makedirs(config.data.output_dir, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load clinical data
    clinical_df = pd.read_csv(config.data.clinical_csv)
    all_patient_ids = clinical_df["patient_id"].tolist()
    print(f"Total patients: {len(all_patient_ids)}")

    # Severity distribution
    severe_count = (clinical_df["ahi"] >= config.data.ahi_threshold).sum()
    print(f"Severe OSA: {severe_count} ({severe_count / len(all_patient_ids) * 100:.1f}%)")
    print(f"Non-severe: {len(all_patient_ids) - severe_count} ({(len(all_patient_ids) - severe_count) / len(all_patient_ids) * 100:.1f}%)")

    # Initialize frozen encoders
    print(f"\nLoading speech encoder: {config.model.speech_model_name}")
    speech_encoder = SpeechEncoder(config.model.speech_model_name, freeze=True).to(device)
    config.model.speech_hidden_dim = speech_encoder.hidden_dim

    print(f"Loading text encoder: {config.model.text_model_name}")
    text_encoder = TextEncoder(config.model.text_model_name, freeze=True).to(device)
    config.model.text_hidden_dim = text_encoder.hidden_dim

    # Load and segment ALL audio
    print(f"\nLoading and segmenting audio files...")
    all_segments, patient_id_map = load_and_segment_audio(
        patient_ids=all_patient_ids,
        audio_dir=config.data.audio_dir,
        clinical_df=clinical_df,
        sample_rate=config.data.sample_rate,
        segment_length=config.data.segment_length,
        segment_overlap=config.data.segment_overlap,
        ahi_threshold=config.data.ahi_threshold,
        conditions=args.conditions,
    )
    print(f"Total segments: {len(all_segments)}")

    # Precompute ALL features once
    print(f"\nPrecomputing encoder features (this may take a while)...")
    all_features = precompute_all_features(
        segments=all_segments,
        clinical_df=clinical_df,
        patient_ids=all_patient_ids,
        speech_encoder=speech_encoder,
        text_encoder=text_encoder,
        device=device,
        batch_size=args.precompute_batch_size,
    )

    # Free encoder memory
    del speech_encoder, text_encoder
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Create patient-wise folds
    folds = create_patient_folds(all_patient_ids, config.training.num_folds, config.training.seed)

    # Index features by patient for fast splitting
    features_by_patient = defaultdict(list)
    for feat in all_features:
        features_by_patient[feat["patient_id_str"]].append(feat)

    # Run cross-validation
    all_fold_results = []

    for fold_idx, (train_ids, val_ids, test_ids) in enumerate(folds):
        # Split features
        train_features = []
        for pid in train_ids:
            train_features.extend(features_by_patient.get(pid, []))

        val_features = []
        for pid in val_ids:
            val_features.extend(features_by_patient.get(pid, []))

        test_features = []
        for pid in test_ids:
            test_features.extend(features_by_patient.get(pid, []))

        if len(train_features) == 0 or len(test_features) == 0:
            print(f"Skipping fold {fold_idx + 1}: insufficient data")
            continue

        fold_results = run_fold(
            fold_idx=fold_idx,
            train_features=train_features,
            val_features=val_features if len(val_features) > 0 else test_features,
            test_features=test_features,
            test_patient_ids=test_ids,
            train_patient_ids=train_ids,
            clinical_df=clinical_df,
            config=config,
            device=device,
        )

        all_fold_results.append(fold_results)

    # Aggregate results across folds
    print(f"\n{'=' * 70}")
    print(f"  FINAL RESULTS ({len(all_fold_results)}-fold cross-validation)")
    print(f"{'=' * 70}")

    for level in ["segment_level", "majority_voting", "sequence_aggregator"]:
        print(f"\n  {level.replace('_', ' ').title()}:")
        metrics_across_folds = defaultdict(list)

        for fold_res in all_fold_results:
            for metric, value in fold_res[level].items():
                metrics_across_folds[metric].append(value)

        summary = {}
        for metric, values in metrics_across_folds.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            summary[metric] = {"mean": mean_val, "std": std_val}

            if metric == "mcc":
                print(f"    {metric.upper():>12s}: {mean_val:.4f} ± {std_val:.4f}")
            else:
                print(f"    {metric.capitalize():>12s}: {mean_val:.2f}% ± {std_val:.2f}%")

    # Save results
    results_path = os.path.join(config.data.output_dir, "results.json")
    save_results({
        "config": {
            "model": vars(config.model),
            "training": {k: v for k, v in vars(config.training).items() if k != "betas"},
            "data": vars(config.data),
        },
        "fold_results": all_fold_results,
        "num_folds_completed": len(all_fold_results),
    }, results_path)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()

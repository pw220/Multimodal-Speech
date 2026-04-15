# Multimodal Contrastive Learning for Speech-Based OSA Severity Estimation

Official implementation of:

> **Dual contrastive learning with clinically guided multimodal fusion for enhancing speech-based sleep apnea severity estimation**  
> Wang et al., *Artificial Intelligence in Medicine*, 2026.

---

## Overview

A multimodal contrastive learning framework that jointly models wakeful speech recordings and structured clinical profiles for non-invasive obstructive sleep apnea (OSA) severity estimation.

```
Speech (wav) ──→ XLS-R 300M (frozen) ──→ Temporal Attention ──→ Acoustic Projection ──┐
                                                                                       ├──→ L_CML
Clinical Text ──→ ClinicalBERT (frozen) ──→ [CLS] ──→ Clinical Projection ────────────┘
                                                                                       │
                          ┌────────────────────────────────────────────────────────────┘
                          ▼
                 Clinically Guided Fusion (CGF)
                   channel modulation (γ, β) · gated residual
                          │
                          ▼
                 [z_f ; z̃_t] ──→ MLP ──→ ŷ  +  L_sup
                          │
                 L_total = L_cls + λ(L_CML + L_sup)
```

**Key components:**
- **SpeechEncoder** — XLS-R (300M) pretrained foundation model, frozen
- **TextEncoder** — Bio_ClinicalBERT, encodes structured clinical profiles as natural-language prompts, frozen
- **Cross-Modal Contrastive Loss (L_CML)** — patient-aware InfoNCE alignment of speech and clinical representations
- **Clinically Guided Fusion (CGF)** — asymmetric channel-wise modulation of acoustic features by clinical context
- **Severity-Aware Supervised Contrastive Loss (L_sup)** — label-consistent geometric structuring of the embedding space
- **Statistical Sequence Aggregator** — logistic regression over distributional statistics of segment-level predictions for robust patient-level inference

---

## Installation

```bash
git clone https://github.com/your-username/multimodal-osa.git
cd multimodal-osa
pip install -r requirements.txt
```

Or install as a package:

```bash
pip install -e .
```

---

## Data Format

### Directory structure

```
data/
├── audio/
│   ├── patient_001_pre.wav     # pre-sleep recording
│   ├── patient_001_post.wav    # post-sleep recording
│   └── ...
└── clinical_profiles.csv
```

### `clinical_profiles.csv`

| Column | Type | Description |
|--------|------|-------------|
| `patient_id` | str | Unique identifier, e.g. `patient_001` |
| `age` | float | Age in years |
| `gender` | str | `male` / `female` |
| `bmi` | float | Body mass index (kg/m²) |
| `neck_circumference` | float | Neck circumference (cm) |
| `waist_circumference` | float | Waist circumference (cm) |
| `ess_score` | int | Epworth Sleepiness Scale (0–24) |
| `psqi_score` | int | Pittsburgh Sleep Quality Index (0–21) |
| `ahi` | float | Apnea-Hypopnea Index (events/hr); label = 1 iff AHI ≥ 30 |

### Audio files

- Format: WAV, mono or stereo (stereo is averaged); any sample rate (resampled to 16 kHz)
- Naming: `{patient_id}_{condition}.wav` where `condition ∈ {pre, post}`
- Content: standardised reading passage recorded during wakefulness

---

## Quick Start

### 1. Generate synthetic demo data

```bash
python scripts/generate_demo_data.py --output_dir data/demo --num_patients 50
```

### 2. Train with 10-fold cross-validation

```bash
python main.py \
    --audio_dir data/demo/audio \
    --clinical_csv data/demo/clinical_profiles.csv \
    --output_dir experiments/demo \
    --num_folds 10 \
    --epochs 200 \
    --batch_size 64
```

### 3. Inference on a new patient

```bash
python scripts/inference.py \
    --audio_path path/to/recording.wav \
    --age 45 --gender male --bmi 28.5 \
    --neck_circumference 40.2 --waist_circumference 98.3 \
    --ess_score 12 --psqi_score 8 \
    --checkpoint_dir experiments/demo/fold_0
```

### 4. Ablation study

```bash
python scripts/ablation.py \
    --features_dir experiments/demo/features \
    --output_dir experiments/ablation
```

---

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--embedding_dim` | 128 | Shared projection dimension *d* |
| `--temperature` | 0.03 | Temperature τ for L_CML and L_sup |
| `--lambda_contrastive` | 0.7 | Loss weight λ: L = L_cls + λ(L_CML + L_sup) |
| `--label_smoothing` | 0.1 | Label smoothing ε |
| `--segment_length` | 4.0 s | Sliding-window segment duration |
| `--segment_overlap` | 0.5 | Fractional overlap between segments |
| `--ahi_threshold` | 30.0 | AHI threshold for severe OSA (label = 1) |
| `--lr` | 1e-3 | AdamW learning rate |
| `--weight_decay` | 5e-4 | AdamW weight decay |
| `--num_folds` | 10 | Patient-wise K-fold cross-validation |
| `--patience` | 50 | Early stopping patience (Val AUC) |

---

## Reported Metrics

| Level | Metrics |
|-------|---------|
| Segment-level | Accuracy, Recall, Specificity, Precision, F1, AUC, MCC |
| Patient-level (Majority Voting) | Accuracy, Recall, Specificity, Precision, F1, MCC |
| Patient-level (Sequence Aggregator) | Accuracy, Recall, Specificity, Precision, F1, AUC, MCC |

> **Note:** AUC is not reported for Majority Voting because it produces hard 0/1 decisions only.

---

## Testing

```bash
python -m pytest tests/
```

The test suite validates all components (contrastive losses, CGF, aggregation, cross-validation, serialisation) using synthetic data and small mock dimensions — no pretrained models are downloaded.

---

## Project Structure

```
multimodal-osa/
├── main.py                        # training + 10-fold CV entry point
├── configs/
│   └── config.py                  # ModelConfig, TrainingConfig, DataConfig
├── models/
│   ├── encoders.py                # SpeechEncoder (XLS-R), TextEncoder (ClinicalBERT)
│   ├── framework.py               # MultimodalOSAFramework (end-to-end)
│   ├── fusion.py                  # ClinicallyGuidedFusion (CGF)
│   ├── contrastive.py             # CrossModalContrastiveLoss, SeverityAwareContrastiveLoss
│   └── aggregation.py             # MajorityVoting, StatisticalSequenceAggregator
├── data/
│   └── dataset.py                 # OSADataset, PrecomputedOSADataset
├── utils/
│   ├── helpers.py                 # metrics, folds, early stopping, serialisation
│   ├── trainer.py                 # DownstreamModel, train_one_epoch, evaluate
│   └── visualization.py           # t-SNE, confusion matrices, sensitivity plots
├── scripts/
│   ├── generate_demo_data.py      # synthetic dataset generation
│   ├── inference.py               # single-patient inference
│   └── ablation.py                # ablation study variants
├── tests/
│   └── test_pipeline.py           # end-to-end test suite (44 assertions)
├── MODEL_INPUT_FORMAT.md          # complete tensor/data format specification
├── requirements.txt
└── setup.py
```

---


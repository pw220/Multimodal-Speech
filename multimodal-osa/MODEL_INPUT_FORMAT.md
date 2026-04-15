# Model Input Format

Complete specification of every tensor / data structure consumed by each
component of the multimodal OSA severity-estimation framework.

---

## 1. Raw Inputs (pipeline entry points)

### 1.1 Audio file
| Property | Value |
|----------|-------|
| Format | WAV (mono or stereo — stereo is averaged to mono) |
| Sample rate | Any (resampled to **16 000 Hz** internally) |
| Amplitude | Normalised to **[−1, 1]** per recording |
| Naming | `{patient_id}_{condition}.wav` where `condition ∈ {pre, post}` |

After loading, audio is cut into overlapping segments with a sliding window:

| Parameter | Default | CLI flag |
|-----------|---------|----------|
| Segment length | 4.0 s → **64 000 samples** at 16 kHz | `--segment_length` |
| Overlap | 50 % → hop = **32 000 samples** | `--segment_overlap` |

Each segment becomes a 1-D float32 tensor `waveform ∈ ℝ^N` where `N = segment_length × 16000`.

### 1.2 Clinical profiles CSV
File: `clinical_profiles.csv`

| Column | Type | Range / Example | Notes |
|--------|------|-----------------|-------|
| `patient_id` | str | `"patient_001"` | Unique identifier |
| `age` | float | 18 – 80 | Years |
| `gender` | str | `"male"` / `"female"` | Used in prompt template |
| `bmi` | float | 16 – 50 | kg/m² |
| `neck_circumference` | float | 25 – 55 | cm |
| `waist_circumference` | float | 60 – 140 | cm |
| `ess_score` | int | 0 – 24 | Epworth Sleepiness Scale |
| `psqi_score` | int | 0 – 21 | Pittsburgh Sleep Quality Index |
| `ahi` | float | 0 – 120 | Apnea-Hypopnea Index (events/hr); **label = 1 iff AHI ≥ 30** |

---

## 2. Encoder Inputs

### 2.1 SpeechEncoder (`models/encoders.py`)
Wraps **facebook/wav2vec2-xls-r-300m** (frozen).

| Tensor | Shape | dtype | Description |
|--------|-------|-------|-------------|
| `waveform` | `(B, N)` | float32 | Batch of raw waveform segments at 16 kHz |
| `attention_mask` | `(B, N)` | int64 | Optional; 1 = valid sample, 0 = padding |

Output: `H_a ∈ ℝ^{B × L × 1024}` — frame-level acoustic representations.
`L ≈ N / 320` (wav2vec2 stride ≈ 20 ms; for 4 s → L ≈ 199).

### 2.2 TextEncoder (`models/encoders.py`)
Wraps **emilyalsentzer/Bio_ClinicalBERT** (frozen).

| Tensor | Shape | dtype | Description |
|--------|-------|-------|-------------|
| `input_ids` | `(B, M)` | int64 | Tokenised clinical prompt, padded to `M = 128` |
| `attention_mask` | `(B, M)` | int64 | 1 = real token, 0 = pad |

The prompt is built by `build_clinical_prompt()`:
```
"The patient is a {age}-year-old {gender} with BMI {bmi:.1f}.
 Neck and waist circumferences are {neck_circ:.1f} cm and {waist_circ:.1f} cm, respectively.
 ESS and PSQI scores are {ess_score} and {psqi_score}, respectively."
```

Output: `t ∈ ℝ^{B × 768}` — [CLS] token embedding (global clinical representation).

---

## 3. DownstreamModel / MultimodalOSAFramework Inputs

This is the trainable part of the pipeline.  
It operates on **precomputed** frozen-encoder outputs.

| Argument | Shape | dtype | Description |
|----------|-------|-------|-------------|
| `speech_frames` | `(B, L, d_a)` | float32 | Frame-level acoustic features from SpeechEncoder. `d_a = 1024` (XLS-R 300M) |
| `clinical_emb` | `(B, d_t)` | float32 | [CLS] clinical embedding from TextEncoder. `d_t = 768` (ClinicalBERT) |
| `labels` *(train only)* | `(B,)` | int64 | Binary severity labels: 0 = non-severe, 1 = severe |
| `patient_ids` *(train only)* | `(B,)` | int64 | Integer patient index (used to mask intra-patient contrastive pairs) |

---

## 4. PrecomputedFeatureDataset (DataLoader items)

Each element returned by `__getitem__`:

| Key | Shape | dtype | Source |
|-----|-------|-------|--------|
| `speech_frames` | `(L, d_a)` | float32 | SpeechEncoder output, single segment |
| `clinical_emb` | `(d_t,)` | float32 | TextEncoder [CLS] output, per patient |
| `label` | scalar | int64 | Binary severity label |
| `patient_id` | scalar | int64 | Integer patient index |

After collation into a batch: shapes become `(B, L, d_a)`, `(B, d_t)`, `(B,)`, `(B,)`.

---

## 5. Internal Tensor Shapes (forward pass)

```
waveform (B, N=64000)
    │
    ▼ SpeechEncoder (XLS-R, frozen)
H_a (B, L≈199, d_a=1024)
    │
    ▼ TemporalAttentionAggregation
z_a (B, d_a=1024)
    │
    ▼ AcousticProjector  (d_a → d=128)
z̃_a (B, d=128)   ──── L2 norm ────► ẑ_a (B, d=128)  [contrastive path]
    │
    │           clinical prompt (B, M=128)
    │               │
    │           TextEncoder (ClinicalBERT, frozen)
    │           t (B, d_t=768)
    │               │
    │           TextualProjector (d_t → d=128)
    │           z̃_t (B, d=128)   ──── L2 norm ────► ẑ_t (B, d=128)  [contrastive path]
    │               │
    ▼               ▼
ClinicallyGuidedFusion (CGF)
    γ, β  ←  W_γ(z̃_t), W_β(z̃_t)         channel modulation
    Δz    =  γ ⊙ z̃_a + β
    α     =  σ(gate(z̃_t))
    z_f   =  z̃_a + α · Δz                 (B, d=128)
    h     =  [z_f ; z̃_t]                  (B, 2d=256)
    │
    ├──► SupervisedContrastiveProjection → r (B, d_s=64)  [L_sup path]
    │
    └──► PredictionHead → ŷ (B,)           sigmoid probability
```

---

## 6. Patient-Level Aggregation Inputs

### MajorityVoting
```python
segment_probs: np.ndarray  # shape (K,), float, range [0, 1]
```
Returns `(prediction: int, confidence: float)`.

### StatisticalSequenceAggregator
**Fit:**
```python
patient_segment_probs: list[np.ndarray]  # list of K_i-length arrays, one per patient
patient_labels:        np.ndarray        # shape (N,), binary int
```
**Predict:**
```python
segment_probs: np.ndarray  # shape (K,), float, range [0, 1]
```
Feature vector extracted: `g = [mean, std, max, min, median, q75, q25, mv_pred]` — shape `(8,)`.

---

## 7. Contrastive Loss Inputs

### CrossModalContrastiveLoss
| Argument | Shape | Notes |
|----------|-------|-------|
| `z_a` | `(B, d)` | **L2-normalised** acoustic projections |
| `z_t` | `(B, d)` | **L2-normalised** clinical projections |
| `patient_ids` | `(B,)` int64 | Used to exclude intra-patient pairs from negatives |

### SeverityAwareContrastiveLoss
| Argument | Shape | Notes |
|----------|-------|-------|
| `r` | `(B, d_s)` | **L2-normalised** supervised contrastive projections |
| `labels` | `(B,)` int64 | Binary severity labels (0 / 1) |
| `patient_ids` | `(B,)` int64 | Cross-patient masking for both positives and negatives |

---

## 8. Hyperparameter Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embedding_dim` d | 128 | Shared projection space dimension |
| `prediction_hidden_dim` | 64 | Hidden size of prediction MLP |
| `sup_contrastive_dim` | 64 | Supervised contrastive projection dimension |
| `temperature` τ | 0.03 | Temperature for L_CML |
| `temperature_sup` τ_s | 0.03 | Temperature for L_sup |
| `lambda_contrastive` λ | 0.7 | Weight: L_total = L_cls + λ(L_CML + L_sup) |
| `label_smoothing` ε | 0.1 | Soft labels: ỹ = (1−ε)y + ε/2 |
| `lr` | 1e-3 | AdamW learning rate |
| `weight_decay` | 5e-4 | AdamW weight decay |
| `batch_size` | 64 | Segments per batch (drop_last=True during training) |
| `segment_length` | 4.0 s | Sliding-window segment duration |
| `segment_overlap` | 0.5 | Fractional overlap between consecutive segments |
| `ahi_threshold` | 30.0 | AHI ≥ 30 → severe (label = 1) |
| `num_folds` | 10 | Patient-wise K-fold cross-validation |
| `patience` | 50 | Early stopping patience (epochs without AUC improvement) |

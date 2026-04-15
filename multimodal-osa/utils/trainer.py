"""
Training engine for the multimodal OSA framework.

Supports two modes:
1. End-to-end training (slower, uses raw audio)
2. Precomputed features (faster, precomputes frozen encoder outputs)

The precomputed mode is recommended since both speech and text encoders
are frozen during training (Sec 4.4).
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.framework import (
    TemporalAttentionAggregation,
    ProjectionHead,
    SupervisedContrastiveProjection,
    PredictionHead,
)
from models.fusion import ClinicallyGuidedFusion
from models.contrastive import CrossModalContrastiveLoss, SeverityAwareContrastiveLoss
from utils.helpers import compute_metrics, EarlyStopping


class PrecomputedFeatureDataset(Dataset):
    """Dataset wrapping precomputed features for efficient training."""

    def __init__(self, features_list: list):
        """
        Args:
            features_list: list of dicts, each with keys:
                - speech_frames: (L, d_a) frame-level speech features
                - clinical_emb: (d_t,) clinical embedding
                - label: int
                - patient_id: int
        """
        self.data = features_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "speech_frames": item["speech_frames"],
            "clinical_emb": item["clinical_emb"],
            "label": torch.tensor(item["label"], dtype=torch.long),
            "patient_id": torch.tensor(item["patient_id"], dtype=torch.long),
        }


class DownstreamModel(nn.Module):
    """
    Downstream model that operates on precomputed encoder features.

    Includes: temporal attention, projection heads, CGF, contrastive projector,
    and prediction head.
    """

    def __init__(
        self,
        speech_hidden_dim: int = 1024,
        text_hidden_dim: int = 768,
        embedding_dim: int = 128,
        prediction_hidden_dim: int = 64,
        sup_contrastive_dim: int = 64,
        temperature: float = 0.03,
        temperature_sup: float = 0.03,
        lambda_contrastive: float = 0.7,
        label_smoothing: float = 0.1,
    ):
        super().__init__()

        self.lambda_contrastive = lambda_contrastive
        self.label_smoothing = label_smoothing

        # Temporal attention (Eq. 7-9)
        self.temporal_attention = TemporalAttentionAggregation(speech_hidden_dim)

        # Projection heads (Eq. 10)
        self.acoustic_projector = ProjectionHead(speech_hidden_dim, embedding_dim)
        self.textual_projector = ProjectionHead(text_hidden_dim, embedding_dim)

        # Contrastive losses
        self.cml_loss_fn = CrossModalContrastiveLoss(temperature)
        self.sup_loss_fn = SeverityAwareContrastiveLoss(temperature_sup)

        # Clinically guided fusion (Sec 3.4.4)
        self.cgf = ClinicallyGuidedFusion(embedding_dim)

        # Supervised contrastive projector (Eq. 23)
        self.sup_projector = SupervisedContrastiveProjection(2 * embedding_dim, sup_contrastive_dim)

        # Prediction head (Eq. 21)
        self.prediction_head = PredictionHead(2 * embedding_dim, prediction_hidden_dim)

    def forward(self, speech_frames, clinical_emb, labels=None, patient_ids=None):
        """
        Args:
            speech_frames: (B, L, d_a) frame-level speech features
            clinical_emb: (B, d_t) clinical [CLS] embeddings
            labels: (B,) binary labels (optional)
            patient_ids: (B,) patient identifiers (optional)

        Returns:
            dict with predictions and losses
        """
        # Temporal attention aggregation
        z_a = self.temporal_attention(speech_frames)  # (B, d_a)

        # Projection with dual-path routing
        z_a_tilde = self.acoustic_projector(z_a)  # (B, d)
        z_t_tilde = self.textual_projector(clinical_emb)  # (B, d)

        # L2-normalized for contrastive path (Eq. 11)
        z_a_hat = F.normalize(z_a_tilde, p=2, dim=-1)
        z_t_hat = F.normalize(z_t_tilde, p=2, dim=-1)

        # Clinically guided fusion
        h, z_f = self.cgf(z_a_tilde, z_t_tilde)  # h: (B, 2d)

        # Prediction
        y_hat = self.prediction_head(h)  # (B,)

        result = {"y_hat": y_hat}

        if labels is not None and patient_ids is not None:
            # Classification loss with label smoothing (Eq. 27)
            eps = self.label_smoothing
            y_ls = (1 - eps) * labels.float() + eps / 2
            loss_cls = F.binary_cross_entropy(y_hat, y_ls, reduction="mean")

            # Cross-modal contrastive loss (Eq. 15)
            loss_cml = self.cml_loss_fn(z_a_hat, z_t_hat, patient_ids)

            # Supervised contrastive loss (Eq. 26)
            r = self.sup_projector(h)
            loss_sup = self.sup_loss_fn(r, labels, patient_ids)

            # Total loss (Eq. 28)
            loss_total = loss_cls + self.lambda_contrastive * (loss_cml + loss_sup)

            result.update({
                "loss_total": loss_total,
                "loss_cls": loss_cls,
                "loss_cml": loss_cml,
                "loss_sup": loss_sup,
            })

        return result


def precompute_features(
    audio_segments: list,
    clinical_data: dict,
    speech_encoder,
    text_encoder,
    device: torch.device,
    batch_size: int = 16,
) -> list:
    """
    Precompute frozen encoder outputs for all segments.

    Args:
        audio_segments: list of dicts with 'waveform', 'patient_id', 'label'
        clinical_data: {patient_id: {'input_ids': ..., 'attention_mask': ...}}
        speech_encoder: frozen SpeechEncoder
        text_encoder: frozen TextEncoder
        device: compute device
        batch_size: batch size for precomputation

    Returns:
        list of dicts with precomputed features
    """
    speech_encoder.eval()
    text_encoder.eval()

    # Precompute clinical embeddings (one per patient)
    clinical_embeddings = {}
    for pid, tokens in clinical_data.items():
        input_ids = tokens["input_ids"].unsqueeze(0).to(device)
        attn_mask = tokens["attention_mask"].unsqueeze(0).to(device)
        with torch.no_grad():
            emb = text_encoder(input_ids, attn_mask).squeeze(0).cpu()
        clinical_embeddings[pid] = emb

    # Precompute speech features in batches
    features_list = []
    for i in tqdm(range(0, len(audio_segments), batch_size), desc="Precomputing speech features"):
        batch = audio_segments[i: i + batch_size]
        waveforms = torch.stack([s["waveform"] for s in batch]).to(device)

        with torch.no_grad():
            H_a = speech_encoder(waveforms).cpu()  # (B, L, d_a)

        for j, seg in enumerate(batch):
            features_list.append({
                "speech_frames": H_a[j],  # (L, d_a)
                "clinical_emb": clinical_embeddings[seg["patient_id_str"]],
                "label": seg["label"],
                "patient_id": seg["patient_id"],
            })

    return features_list


def train_one_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_cls = 0
    total_cml = 0
    total_sup = 0
    num_batches = 0

    for batch in dataloader:
        speech_frames = batch["speech_frames"].to(device)
        clinical_emb = batch["clinical_emb"].to(device)
        labels = batch["label"].to(device)
        patient_ids = batch["patient_id"].to(device)

        optimizer.zero_grad()

        output = model(speech_frames, clinical_emb, labels, patient_ids)
        loss = output["loss_total"]

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_cls += output["loss_cls"].item()
        total_cml += output["loss_cml"].item()
        total_sup += output["loss_sup"].item()
        num_batches += 1

    if num_batches == 0:
        return {"loss": 0.0, "loss_cls": 0.0, "loss_cml": 0.0, "loss_sup": 0.0}

    return {
        "loss": total_loss / num_batches,
        "loss_cls": total_cls / num_batches,
        "loss_cml": total_cml / num_batches,
        "loss_sup": total_sup / num_batches,
    }


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate model and return segment-level predictions."""
    model.eval()
    all_probs = []
    all_labels = []
    all_patient_ids = []

    for batch in dataloader:
        speech_frames = batch["speech_frames"].to(device)
        clinical_emb = batch["clinical_emb"].to(device)
        labels = batch["label"]
        patient_ids = batch["patient_id"]

        output = model(speech_frames, clinical_emb)
        all_probs.append(output["y_hat"].cpu().numpy())
        all_labels.append(labels.numpy())
        all_patient_ids.append(patient_ids.numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    all_patient_ids = np.concatenate(all_patient_ids)

    # Segment-level metrics
    preds = (all_probs >= 0.5).astype(int)
    metrics = compute_metrics(all_labels, preds, all_probs)

    return metrics, all_probs, all_labels, all_patient_ids

"""
Ablation study script (Sec 5.3).

Evaluates the contribution of key components by selectively removing them:
1. Speech-only: acoustic features only
2. Text-only: clinical profile only
3. Simple concat: naive concatenation without CGF
4. w/o L_CML: remove cross-modal contrastive objective
5. w/o L_sup: remove severity-aware supervised contrastive objective
6. w/o contrastive: remove both contrastive objectives
7. Full model: all components enabled
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import defaultdict

from models.framework import (
    TemporalAttentionAggregation,
    ProjectionHead,
    SupervisedContrastiveProjection,
    PredictionHead,
)
from models.fusion import ClinicallyGuidedFusion
from models.contrastive import CrossModalContrastiveLoss, SeverityAwareContrastiveLoss
from utils.trainer import PrecomputedFeatureDataset, evaluate
from utils.helpers import compute_metrics, EarlyStopping, print_metrics


class AblatedModel(nn.Module):
    """
    Configurable model supporting various ablation settings.
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
        # Ablation flags
        use_speech: bool = True,
        use_text: bool = True,
        use_cgf: bool = True,
        use_cml: bool = True,
        use_sup: bool = True,
    ):
        super().__init__()
        self.use_speech = use_speech
        self.use_text = use_text
        self.use_cgf = use_cgf
        self.use_cml = use_cml
        self.use_sup = use_sup
        self.lambda_contrastive = lambda_contrastive
        self.label_smoothing = label_smoothing

        if use_speech:
            self.temporal_attention = TemporalAttentionAggregation(speech_hidden_dim)
            self.acoustic_projector = ProjectionHead(speech_hidden_dim, embedding_dim)

        if use_text:
            self.textual_projector = ProjectionHead(text_hidden_dim, embedding_dim)

        if use_cml and use_speech and use_text:
            self.cml_loss_fn = CrossModalContrastiveLoss(temperature)

        if use_cgf and use_speech and use_text:
            self.cgf = ClinicallyGuidedFusion(embedding_dim)

        # Determine prediction input dimension
        if use_speech and use_text:
            pred_input_dim = 2 * embedding_dim
        elif use_speech:
            pred_input_dim = embedding_dim
        else:
            pred_input_dim = embedding_dim

        if use_sup:
            self.sup_projector = SupervisedContrastiveProjection(pred_input_dim, sup_contrastive_dim)
            self.sup_loss_fn = SeverityAwareContrastiveLoss(temperature_sup)

        self.prediction_head = PredictionHead(pred_input_dim, prediction_hidden_dim)

    def forward(self, speech_frames, clinical_emb, labels=None, patient_ids=None):
        if self.use_speech:
            z_a = self.temporal_attention(speech_frames)
            z_a_tilde = self.acoustic_projector(z_a)
            z_a_hat = F.normalize(z_a_tilde, p=2, dim=-1)

        if self.use_text:
            z_t_tilde = self.textual_projector(clinical_emb)
            z_t_hat = F.normalize(z_t_tilde, p=2, dim=-1)

        # Build multimodal representation
        if self.use_speech and self.use_text:
            if self.use_cgf:
                h, z_f = self.cgf(z_a_tilde, z_t_tilde)
            else:
                # Simple concatenation
                h = torch.cat([z_a_tilde, z_t_tilde], dim=-1)
        elif self.use_speech:
            h = z_a_tilde
        else:
            h = z_t_tilde

        y_hat = self.prediction_head(h)
        result = {"y_hat": y_hat}

        if labels is not None and patient_ids is not None:
            eps = self.label_smoothing
            y_ls = (1 - eps) * labels.float() + eps / 2
            loss_cls = F.binary_cross_entropy(y_hat, y_ls, reduction="mean")

            loss_contrastive = torch.tensor(0.0, device=y_hat.device)

            if self.use_cml and self.use_speech and self.use_text:
                loss_cml = self.cml_loss_fn(z_a_hat, z_t_hat, patient_ids)
                loss_contrastive = loss_contrastive + loss_cml
                result["loss_cml"] = loss_cml

            if self.use_sup:
                r = self.sup_projector(h)
                loss_sup = self.sup_loss_fn(r, labels, patient_ids)
                loss_contrastive = loss_contrastive + loss_sup
                result["loss_sup"] = loss_sup

            loss_total = loss_cls + self.lambda_contrastive * loss_contrastive
            result["loss_total"] = loss_total
            result["loss_cls"] = loss_cls

        return result


def train_ablation_variant(
    variant_name: str,
    train_features: list,
    val_features: list,
    test_features: list,
    device: torch.device,
    ablation_flags: dict,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> dict:
    """Train and evaluate a single ablation variant."""
    print(f"\n  Training variant: {variant_name}")

    train_loader = DataLoader(
        PrecomputedFeatureDataset(train_features),
        batch_size=batch_size, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        PrecomputedFeatureDataset(val_features),
        batch_size=batch_size, shuffle=False,
    )
    test_loader = DataLoader(
        PrecomputedFeatureDataset(test_features),
        batch_size=batch_size, shuffle=False,
    )

    model = AblatedModel(**ablation_flags).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    early_stopping = EarlyStopping(patience=50, mode="max")

    best_val_auc = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            speech_frames = batch["speech_frames"].to(device)
            clinical_emb = batch["clinical_emb"].to(device)
            labels_batch = batch["label"].to(device)
            patient_ids_batch = batch["patient_id"].to(device)

            optimizer.zero_grad()
            output = model(speech_frames, clinical_emb, labels_batch, patient_ids_batch)
            output["loss_total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        val_metrics, _, _, _ = evaluate(model, val_loader, device)
        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if early_stopping(val_metrics["auc"]):
            break

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    test_metrics, _, _, _ = evaluate(model, test_loader, device)
    print(f"    ", end="")
    print_metrics(test_metrics)

    return test_metrics


# Ablation variant configurations (Table 6)
ABLATION_VARIANTS = {
    "Speech-only": dict(use_speech=True, use_text=False, use_cgf=False, use_cml=False, use_sup=False),
    "Text-only": dict(use_speech=False, use_text=True, use_cgf=False, use_cml=False, use_sup=False),
    "Concat (w/o CGF)": dict(use_speech=True, use_text=True, use_cgf=False, use_cml=True, use_sup=True),
    "w/o CML": dict(use_speech=True, use_text=True, use_cgf=True, use_cml=False, use_sup=True),
    "w/o sup": dict(use_speech=True, use_text=True, use_cgf=True, use_cml=True, use_sup=False),
    "w/o contrastive": dict(use_speech=True, use_text=True, use_cgf=True, use_cml=False, use_sup=False),
    "Full model": dict(use_speech=True, use_text=True, use_cgf=True, use_cml=True, use_sup=True),
}

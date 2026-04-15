"""
Complete multimodal contrastive learning framework for OSA severity estimation.

Integrates all components from Sec 3.4:
- Multimodal Feature Extractor (Sec 3.4.2)
- Cross-Modal Speech-Text Alignment (Sec 3.4.3)
- Clinically Guided Fusion (Sec 3.4.4)
- Severity-Aware Supervised Contrastive Learning (Sec 3.4.5)
- Multitask Learning Formulation (Sec 3.4.6)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoders import SpeechEncoder, TextEncoder
from models.fusion import ClinicallyGuidedFusion
from models.contrastive import CrossModalContrastiveLoss, SeverityAwareContrastiveLoss


class TemporalAttentionAggregation(nn.Module):
    """
    Temporal attention aggregation (Sec 3.4.3).

    Aggregates frame-level hidden states H_a into a single segment-level vector
    using a lightweight attention mechanism.

    u_t = tanh(W_u H_a[t] + b_u)                    (Eq. 7)
    α_t = exp(w^T u_t) / Σ_τ exp(w^T u_τ)           (Eq. 8)
    z_a = Σ_t α_t H_a[t]                             (Eq. 9)
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W_u = nn.Linear(hidden_dim, hidden_dim)
        self.w = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, H_a: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H_a: (B, L, d_a) frame-level acoustic representations

        Returns:
            z_a: (B, d_a) segment-level acoustic embedding
        """
        # Eq. 7
        u = torch.tanh(self.W_u(H_a))  # (B, L, d_a)

        # Eq. 8 - attention weights
        scores = torch.matmul(u, self.w)  # (B, L)
        alpha = F.softmax(scores, dim=-1)  # (B, L)

        # Eq. 9 - weighted sum
        z_a = torch.bmm(alpha.unsqueeze(1), H_a).squeeze(1)  # (B, d_a)
        return z_a


class ProjectionHead(nn.Module):
    """
    Modality-specific two-layer MLP projection head (Sec 3.4.3).

    Linear → ReLU → Linear (Eq. 10)
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SupervisedContrastiveProjection(nn.Module):
    """
    Lightweight two-layer MLP for supervised contrastive projection (Sec 3.4.5).

    r_i = q_s(h_i) / ||q_s(h_i)||_2   (Eq. 23)
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        projected = self.net(h)
        return F.normalize(projected, p=2, dim=-1)


class PredictionHead(nn.Module):
    """
    Two-layer MLP prediction head with GELU activation (Eq. 21).

    ŷ = σ(w_o^T GELU(W_h h + b_h) + b_o)
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (B, 2d) multimodal representation

        Returns:
            y_hat: (B,) predicted probability of severe OSA
        """
        return torch.sigmoid(self.net(h)).squeeze(-1)


class MultimodalOSAFramework(nn.Module):
    """
    Complete multimodal contrastive learning framework.

    Implements the full pipeline:
    1. Speech encoding via XLS-R → temporal attention → acoustic projection
    2. Clinical encoding via ClinicalBERT → clinical projection
    3. Dual-path routing: contrastive path (L2-normalized) + fusion path (unnormalized)
    4. Cross-modal contrastive alignment
    5. Clinically guided fusion
    6. Severity-aware supervised contrastive learning
    7. Binary prediction

    Total loss (Eq. 28):
        L_total = L_cls + λ(L_CML + L_sup)
    """

    def __init__(
        self,
        speech_model_name: str = "facebook/wav2vec2-xls-r-300m",
        text_model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        speech_hidden_dim: int = 1024,
        text_hidden_dim: int = 768,
        embedding_dim: int = 128,
        prediction_hidden_dim: int = 64,
        sup_contrastive_dim: int = 64,
        temperature: float = 0.03,
        temperature_sup: float = 0.03,
        lambda_contrastive: float = 0.7,
        label_smoothing: float = 0.1,
        freeze_speech: bool = True,
        freeze_text: bool = True,
    ):
        super().__init__()

        self.lambda_contrastive = lambda_contrastive
        self.label_smoothing = label_smoothing
        self.embedding_dim = embedding_dim

        # Stage 1: Multimodal Feature Extractors (Sec 3.4.2)
        self.speech_encoder = SpeechEncoder(speech_model_name, freeze=freeze_speech)
        self.text_encoder = TextEncoder(text_model_name, freeze=freeze_text)

        # Temporal attention aggregation (Sec 3.4.3, Eq. 7-9)
        self.temporal_attention = TemporalAttentionAggregation(speech_hidden_dim)

        # Modality-specific projection heads (Eq. 10)
        self.acoustic_projector = ProjectionHead(speech_hidden_dim, embedding_dim)
        self.textual_projector = ProjectionHead(text_hidden_dim, embedding_dim)

        # Stage 2: Cross-Modal Contrastive Learning
        self.cml_loss_fn = CrossModalContrastiveLoss(temperature)

        # Stage 3: Clinically Guided Fusion (Sec 3.4.4)
        self.cgf = ClinicallyGuidedFusion(embedding_dim)

        # Severity-Aware Supervised Contrastive Learning (Sec 3.4.5)
        self.sup_projector = SupervisedContrastiveProjection(2 * embedding_dim, sup_contrastive_dim)
        self.sup_loss_fn = SeverityAwareContrastiveLoss(temperature_sup)

        # Stage 4: Prediction Head (Eq. 21)
        self.prediction_head = PredictionHead(2 * embedding_dim, prediction_hidden_dim)

    def encode_speech(self, waveform: torch.Tensor) -> tuple:
        """
        Encode speech waveform through XLS-R + temporal attention + projection.

        Returns:
            z_a_tilde: (B, d) unnormalized acoustic projection (for fusion)
            z_a_hat: (B, d) L2-normalized acoustic projection (for contrastive)
        """
        H_a = self.speech_encoder(waveform)  # (B, L, d_a)
        z_a = self.temporal_attention(H_a)  # (B, d_a)
        z_a_tilde = self.acoustic_projector(z_a)  # (B, d)
        z_a_hat = F.normalize(z_a_tilde, p=2, dim=-1)  # (B, d) (Eq. 11)
        return z_a_tilde, z_a_hat

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple:
        """
        Encode clinical profile through ClinicalBERT + projection.

        Returns:
            z_t_tilde: (B, d) unnormalized clinical projection (for fusion)
            z_t_hat: (B, d) L2-normalized clinical projection (for contrastive)
        """
        t = self.text_encoder(input_ids, attention_mask)  # (B, d_t)
        z_t_tilde = self.textual_projector(t)  # (B, d)
        z_t_hat = F.normalize(z_t_tilde, p=2, dim=-1)  # (B, d) (Eq. 11)
        return z_t_tilde, z_t_hat

    def forward(
        self,
        waveform: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
        patient_ids: torch.Tensor = None,
    ) -> dict:
        """
        Full forward pass.

        Args:
            waveform: (B, N) raw waveform at 16kHz
            input_ids: (B, M) tokenized clinical prompts
            attention_mask: (B, M) attention mask for clinical prompts
            labels: (B,) binary severity labels (optional, for training)
            patient_ids: (B,) patient identifiers (optional, for training)

        Returns:
            dict with keys:
                - y_hat: (B,) predicted probabilities
                - loss_total: scalar total loss (if labels provided)
                - loss_cls: scalar classification loss
                - loss_cml: scalar cross-modal contrastive loss
                - loss_sup: scalar supervised contrastive loss
        """
        # Encode both modalities with dual-path routing
        z_a_tilde, z_a_hat = self.encode_speech(waveform)
        z_t_tilde, z_t_hat = self.encode_text(input_ids, attention_mask)

        # Clinically guided fusion (Sec 3.4.4)
        h, z_f = self.cgf(z_a_tilde, z_t_tilde)  # h: (B, 2d)

        # Prediction (Eq. 21)
        y_hat = self.prediction_head(h)  # (B,)

        result = {"y_hat": y_hat}

        if labels is not None and patient_ids is not None:
            # Classification loss with label smoothing (Eq. 27)
            epsilon = self.label_smoothing
            y_ls = (1 - epsilon) * labels.float() + epsilon / 2
            loss_cls = F.binary_cross_entropy(y_hat, y_ls, reduction="mean")

            # Cross-modal contrastive loss (Eq. 15)
            loss_cml = self.cml_loss_fn(z_a_hat, z_t_hat, patient_ids)

            # Severity-aware supervised contrastive loss (Eq. 26)
            r = self.sup_projector(h)  # (B, d_s)
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

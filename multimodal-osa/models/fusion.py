"""
Clinically Guided Fusion (CGF) module (Sec 3.4.4).

Uses the clinical embedding to generate channel-wise modulation parameters,
conditioning the acoustic representation on patient-level clinical semantics.
This is an asymmetric fusion where clinical context modulates speech features
rather than merging both modalities symmetrically.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClinicallyGuidedFusion(nn.Module):
    """
    Clinically Guided Fusion module.

    Operations:
    1. Channel modulation: γ and β derived from clinical projection (Eq. 16)
    2. Clinically conditioned correction: Δz = γ ⊙ z̃_a + β (Eq. 17)
    3. Gated residual integration: z_f = z̃_a + α · Δz (Eq. 18-19)
    4. Final representation: h = [z_f ; z̃_t] (Eq. 20)
    """

    def __init__(self, embedding_dim: int):
        """
        Args:
            embedding_dim: Shared embedding dimension d
        """
        super().__init__()
        d = embedding_dim

        # Channel modulation parameters (Eq. 16)
        # γ = 1 + tanh(W_γ z̃_t + b_γ)  →  γ ∈ (0, 2)
        self.W_gamma = nn.Linear(d, d)
        # β = tanh(W_β z̃_t + b_β)  →  β ∈ (-1, 1)
        self.W_beta = nn.Linear(d, d)

        # Gated residual integration (Eq. 18)
        # α = σ(w_g^T z̃_t + b_g)  →  α ∈ (0, 1)
        self.gate = nn.Linear(d, 1)

    def forward(
        self,
        z_a_tilde: torch.Tensor,
        z_t_tilde: torch.Tensor,
    ) -> tuple:
        """
        Args:
            z_a_tilde: (B, d) unnormalized acoustic projection
            z_t_tilde: (B, d) unnormalized clinical projection

        Returns:
            h: (B, 2d) final multimodal representation [z_f ; z̃_t]
            z_f: (B, d) modulated acoustic embedding
        """
        # Channel modulation (Eq. 16)
        gamma = 1.0 + torch.tanh(self.W_gamma(z_t_tilde))  # (B, d), range (0, 2)
        beta = torch.tanh(self.W_beta(z_t_tilde))  # (B, d), range (-1, 1)

        # Clinically conditioned correction (Eq. 17)
        delta_z = gamma * z_a_tilde + beta  # (B, d)

        # Gated residual integration (Eq. 18-19)
        alpha = torch.sigmoid(self.gate(z_t_tilde))  # (B, 1)
        z_f = z_a_tilde + alpha * delta_z  # (B, d)

        # Final multimodal representation (Eq. 20)
        h = torch.cat([z_f, z_t_tilde], dim=-1)  # (B, 2d)

        return h, z_f

"""
Contrastive learning objectives for the multimodal OSA framework.

Cross-Modal Contrastive Loss (Sec 3.4.3):
    Patient-aware InfoNCE-based alignment of speech and clinical representations.

Severity-Aware Supervised Contrastive Loss (Sec 3.4.5):
    Label-consistent geometric structuring of the embedding space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalContrastiveLoss(nn.Module):
    """
    Cross-modal speech-text alignment loss (Sec 3.4.3).

    Patient-aware contrastive objective that aligns each speech segment with
    its patient's clinical profile while separating from other patients.

    Negatives are restricted to cross-patient pairs to avoid false negatives
    from intra-patient segments sharing the same clinical embedding.

    L_CML = (1/2B) Σ_i (ℓ^(i)_{a→t} + ℓ^(i)_{t→a})   (Eq. 15)
    """

    def __init__(self, temperature: float = 0.03):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z_a: torch.Tensor,
        z_t: torch.Tensor,
        patient_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z_a: (B, d) L2-normalized acoustic projections (ẑ_a)
            z_t: (B, d) L2-normalized clinical projections (ẑ_t)
            patient_ids: (B,) integer patient identifiers

        Returns:
            L_CML: scalar cross-modal contrastive loss
        """
        B = z_a.shape[0]
        device = z_a.device

        # Compute pairwise similarity: s_ij = <ẑ_a^(i), ẑ_t^(j)>  (Eq. 12)
        sim_matrix = torch.matmul(z_a, z_t.T) / self.temperature  # (B, B)

        # Build cross-patient mask: N(i) = {j | pid(j) ≠ pid(i)}
        pid_i = patient_ids.unsqueeze(1)  # (B, 1)
        pid_j = patient_ids.unsqueeze(0)  # (1, B)
        cross_patient_mask = (pid_i != pid_j)  # (B, B), True = different patient

        # Diagonal entries are the positive pairs (same patient)
        # For audio→text direction (Eq. 13):
        # log( exp(s_ii/τ) / (exp(s_ii/τ) + Σ_{j∈N(i)} exp(s_ij/τ)) )
        pos_sim = torch.diag(sim_matrix)  # (B,)

        loss_a2t = self._compute_direction_loss(pos_sim, sim_matrix, cross_patient_mask, B, device)
        loss_t2a = self._compute_direction_loss(pos_sim, sim_matrix.T, cross_patient_mask.T, B, device)

        loss = (loss_a2t + loss_t2a) / 2.0
        return loss

    def _compute_direction_loss(
        self,
        pos_sim: torch.Tensor,
        sim_matrix: torch.Tensor,
        cross_patient_mask: torch.Tensor,
        B: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Compute contrastive loss for one direction."""
        loss = torch.tensor(0.0, device=device)
        valid_count = 0

        for i in range(B):
            # Cross-patient indices for anchor i
            neg_indices = cross_patient_mask[i]  # (B,) bool mask

            if neg_indices.sum() == 0:
                continue

            # Positive: sim with own clinical profile
            pos = pos_sim[i].unsqueeze(0)  # (1,)

            # Negatives: sim with other patients' clinical profiles
            neg = sim_matrix[i][neg_indices]  # (num_neg,)

            # log( exp(pos) / (exp(pos) + sum(exp(neg))) )
            logits = torch.cat([pos, neg], dim=0)  # (1 + num_neg,)
            labels = torch.zeros(1, dtype=torch.long, device=device)
            loss += F.cross_entropy(logits.unsqueeze(0), labels)
            valid_count += 1

        if valid_count > 0:
            loss = loss / valid_count

        return loss


class SeverityAwareContrastiveLoss(nn.Module):
    """
    Severity-aware supervised contrastive loss (Sec 3.4.5).

    Encourages samples with the same OSA severity label to cluster together
    while pushing apart samples from different severity groups.

    Both positive and negative sets are constructed exclusively from
    cross-patient pairs to prevent shortcut learning through patient identity.

    L_sup = (1/N) Σ_i (-1/|P+(i)|) Σ_{p∈P+(i)} log(
        exp(sim(r_i, r_p)/τ_s) / Σ_{k∈N(i)} exp(sim(r_i, r_k)/τ_s)
    )   (Eq. 26)
    """

    def __init__(self, temperature: float = 0.03):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        r: torch.Tensor,
        labels: torch.Tensor,
        patient_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            r: (B, d_s) L2-normalized projected embeddings
            labels: (B,) binary severity labels (0 or 1)
            patient_ids: (B,) integer patient identifiers

        Returns:
            L_sup: scalar supervised contrastive loss
        """
        B = r.shape[0]
        device = r.device

        # Cosine similarity matrix
        sim_matrix = torch.matmul(r, r.T) / self.temperature  # (B, B)

        # Cross-patient mask: N(i) = {j | pid(j) ≠ pid(i)}  (Eq. 25)
        pid_i = patient_ids.unsqueeze(1)
        pid_j = patient_ids.unsqueeze(0)
        cross_patient_mask = (pid_i != pid_j)  # (B, B)

        # Same-label mask
        label_i = labels.unsqueeze(1)
        label_j = labels.unsqueeze(0)
        same_label_mask = (label_i == label_j)  # (B, B)

        # Positive set: P+(i) = {j | y_j = y_i, pid(j) ≠ pid(i)}  (Eq. 24)
        positive_mask = same_label_mask & cross_patient_mask  # (B, B)

        # For numerical stability, subtract max
        logits_max, _ = sim_matrix.max(dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()

        # Denominator: sum over all cross-patient pairs (Eq. 25)
        # Mask out intra-patient entries
        exp_logits = torch.exp(logits) * cross_patient_mask.float()
        log_denom = torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        # Log probability for each pair
        log_prob = logits - log_denom  # (B, B)

        # Mean log-prob over positive pairs for each anchor
        num_positives = positive_mask.sum(dim=1)  # (B,)
        valid_anchors = num_positives > 0

        if valid_anchors.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Masked mean over positives
        masked_log_prob = (log_prob * positive_mask.float()).sum(dim=1)  # (B,)
        mean_log_prob = masked_log_prob[valid_anchors] / num_positives[valid_anchors]

        loss = -mean_log_prob.mean()
        return loss

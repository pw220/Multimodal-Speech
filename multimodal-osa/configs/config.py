"""Configuration for the multimodal OSA severity estimation framework."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    # Speech encoder
    speech_model_name: str = "facebook/wav2vec2-xls-r-300m"
    speech_hidden_dim: int = 1024  # XLS-R 300M hidden size

    # Text encoder
    text_model_name: str = "emilyalsentzer/Bio_ClinicalBERT"
    text_hidden_dim: int = 768  # ClinicalBERT hidden size

    # Shared embedding space
    embedding_dim: int = 128  # d in the paper

    # Prediction head
    prediction_hidden_dim: int = 64  # d_h in the paper

    # Supervised contrastive projection
    sup_contrastive_dim: int = 64  # d_s in the paper

    # Freeze encoders
    freeze_speech_encoder: bool = True
    freeze_text_encoder: bool = True


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Optimization
    lr: float = 1e-3
    weight_decay: float = 5e-4
    betas: tuple = (0.9, 0.999)
    epochs: int = 1000
    batch_size: int = 512

    # Loss weights and temperatures
    temperature: float = 0.03  # τ for cross-modal contrastive loss
    temperature_sup: float = 0.03  # τ_s for supervised contrastive loss
    lambda_contrastive: float = 0.7  # λ weight for contrastive losses
    label_smoothing: float = 0.1  # ε for label smoothing

    # Cross-validation
    num_folds: int = 10

    # Early stopping
    patience: int = 50
    min_delta: float = 1e-4

    # Reproducibility
    seed: int = 42


@dataclass
class DataConfig:
    """Data configuration."""

    # Paths
    audio_dir: str = "data/audio"
    clinical_csv: str = "data/clinical_profiles.csv"
    output_dir: str = "experiments/default"

    # Audio preprocessing
    sample_rate: int = 16000
    segment_length: float = 4.0  # T seconds
    segment_overlap: float = 0.5  # 50% overlap

    # AHI threshold for binary severity classification
    ahi_threshold: float = 30.0  # severe OSA ≥ 30

    # Number of dataloader workers
    num_workers: int = 4


@dataclass
class Config:
    """Full configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

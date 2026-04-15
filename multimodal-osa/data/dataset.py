"""
Dataset for the multimodal OSA framework.

Handles loading, segmenting, and pairing speech recordings with clinical profiles.
Each patient has pre-sleep and/or post-sleep recordings that are segmented into
fixed-length clips with 50% overlap (Sec 3.2).
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import soundfile as sf
import warnings

from models.encoders import build_clinical_prompt


class OSADataset(Dataset):
    """
    Dataset that pairs speech segments with clinical profile tokens.

    Each item is a (segment, clinical_tokens, label, patient_id) tuple.
    Segments are created by sliding a window of T seconds with 50% overlap
    over each recording.
    """

    def __init__(
        self,
        patient_ids: list,
        audio_dir: str,
        clinical_df: pd.DataFrame,
        tokenizer,
        sample_rate: int = 16000,
        segment_length: float = 4.0,
        segment_overlap: float = 0.5,
        ahi_threshold: float = 30.0,
        conditions: list = None,
        max_token_length: int = 128,
    ):
        """
        Args:
            patient_ids: List of patient IDs to include
            audio_dir: Directory containing audio files
            clinical_df: DataFrame with clinical profiles (indexed by patient_id)
            tokenizer: ClinicalBERT tokenizer
            sample_rate: Target sample rate (16kHz)
            segment_length: Segment length T in seconds
            segment_overlap: Overlap ratio (0.5 = 50%)
            ahi_threshold: AHI threshold for severe OSA (default 30)
            conditions: List of recording conditions to use ['pre', 'post']
            max_token_length: Maximum token length for clinical prompts
        """
        super().__init__()

        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.segment_samples = int(segment_length * sample_rate)
        self.hop_samples = int(self.segment_samples * (1 - segment_overlap))
        self.ahi_threshold = ahi_threshold
        self.max_token_length = max_token_length

        if conditions is None:
            conditions = ["pre", "post"]

        # Build segment index
        # Each entry: (patient_id, audio_path, start_sample_at_target_sr,
        #              orig_sr, label)
        # orig_sr is cached here so __getitem__ never calls sf.info() again.
        self.segments = []
        self.clinical_tokens = {}  # patient_id → {input_ids, attention_mask}
        self.patient_id_map = {}  # patient_id_str → int

        for pid_idx, pid in enumerate(patient_ids):
            self.patient_id_map[pid] = pid_idx

            row = clinical_df.loc[clinical_df["patient_id"] == pid].iloc[0]

            # Binary label: severe OSA (AHI ≥ 30) = 1, non-severe = 0
            label = 1 if row["ahi"] >= ahi_threshold else 0

            # Build and tokenize clinical prompt
            prompt = build_clinical_prompt(
                age=row["age"],
                gender=row["gender"],
                bmi=row["bmi"],
                neck_circ=row["neck_circumference"],
                waist_circ=row["waist_circumference"],
                ess_score=row["ess_score"],
                psqi_score=row["psqi_score"],
            )
            tokens = tokenizer(
                prompt,
                padding="max_length",
                truncation=True,
                max_length=max_token_length,
                return_tensors="pt",
            )
            self.clinical_tokens[pid] = {
                "input_ids": tokens["input_ids"].squeeze(0),
                "attention_mask": tokens["attention_mask"].squeeze(0),
            }

            # Process audio files for each condition
            for cond in conditions:
                audio_path = os.path.join(audio_dir, f"{pid}_{cond}.wav")
                if not os.path.exists(audio_path):
                    continue

                # Read file metadata once here; never call sf.info in __getitem__
                info = sf.info(audio_path)
                orig_sr = info.samplerate
                total_samples = int(info.duration * sample_rate)

                # Generate segments with sliding window
                start = 0
                while start + self.segment_samples <= total_samples:
                    self.segments.append((pid, audio_path, start, orig_sr, label))
                    start += self.hop_samples

        if len(self.segments) == 0:
            warnings.warn("No valid segments found! Check audio files and patient IDs.")

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        pid, audio_path, start_sample, orig_sr, label = self.segments[idx]

        # Convert start/stop from target-SR sample indices to original-SR indices
        scale = orig_sr / self.sample_rate
        start_orig = int(start_sample * scale)
        stop_orig  = int((start_sample + self.segment_samples) * scale)

        waveform, _ = sf.read(
            audio_path,
            start=start_orig,
            stop=stop_orig,
            dtype="float32",
        )

        # Handle stereo → mono
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)

        # Resample if needed (orig_sr cached from __init__, no extra I/O)
        if orig_sr != self.sample_rate:
            import librosa
            waveform = librosa.resample(waveform, orig_sr=orig_sr,
                                        target_sr=self.sample_rate)

        # Normalize to [-1, 1]
        max_val = np.abs(waveform).max()
        if max_val > 0:
            waveform = waveform / max_val

        # Pad if too short
        if len(waveform) < self.segment_samples:
            waveform = np.pad(waveform, (0, self.segment_samples - len(waveform)))
        else:
            waveform = waveform[: self.segment_samples]

        waveform_tensor = torch.from_numpy(waveform).float()

        # Clinical tokens
        tokens = self.clinical_tokens[pid]

        return {
            "waveform": waveform_tensor,
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "label": torch.tensor(label, dtype=torch.long),
            "patient_id": torch.tensor(self.patient_id_map[pid], dtype=torch.long),
            "patient_id_str": pid,
        }


class PrecomputedOSADataset(Dataset):
    """
    Dataset using precomputed speech encoder features for faster training.

    Since speech and text encoders are frozen, we can precompute their outputs
    and train only the downstream components.
    """

    def __init__(
        self,
        speech_features: dict,
        clinical_embeddings: dict,
        labels: dict,
        patient_id_map: dict,
    ):
        """
        Args:
            speech_features: {(patient_id, segment_idx): (L, d_a) tensor}
            clinical_embeddings: {patient_id: (d_t,) tensor}
            labels: {patient_id: int label}
            patient_id_map: {patient_id: int index}
        """
        self.items = []
        for (pid, seg_idx), feat in speech_features.items():
            self.items.append({
                "speech_features": feat,
                "clinical_embedding": clinical_embeddings[pid],
                "label": labels[pid],
                "patient_id": patient_id_map[pid],
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        return {
            "speech_features": item["speech_features"],
            "clinical_embedding": item["clinical_embedding"],
            "label": torch.tensor(item["label"], dtype=torch.long),
            "patient_id": torch.tensor(item["patient_id"], dtype=torch.long),
        }

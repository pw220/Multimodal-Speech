"""
Inference script for running OSA severity predictions on new patients.

Usage:
    python scripts/inference.py \
        --audio_path path/to/recording.wav \
        --age 45 --gender male --bmi 28.5 \
        --neck_circumference 40.2 --waist_circumference 98.3 \
        --ess_score 12 --psqi_score 8 \
        --checkpoint_dir experiments/demo/fold_0
"""

import argparse
import os
import numpy as np
import torch
import soundfile as sf

from models.encoders import SpeechEncoder, TextEncoder, build_clinical_prompt
from models.aggregation import MajorityVoting, StatisticalSequenceAggregator
from utils.trainer import DownstreamModel


def segment_audio(
    audio_path: str,
    sample_rate: int = 16000,
    segment_length: float = 4.0,
    segment_overlap: float = 0.5,
) -> list:
    """Load and segment an audio file."""
    import librosa

    waveform, orig_sr = sf.read(audio_path, dtype="float32")
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    if orig_sr != sample_rate:
        waveform = librosa.resample(waveform, orig_sr=orig_sr, target_sr=sample_rate)

    max_val = np.abs(waveform).max()
    if max_val > 0:
        waveform = waveform / max_val

    segment_samples = int(segment_length * sample_rate)
    hop_samples = int(segment_samples * (1 - segment_overlap))

    segments = []
    start = 0
    while start + segment_samples <= len(waveform):
        seg = waveform[start: start + segment_samples]
        segments.append(torch.from_numpy(seg).float())
        start += hop_samples

    return segments


def predict(
    audio_path: str,
    clinical_info: dict,
    speech_encoder: SpeechEncoder,
    text_encoder: TextEncoder,
    downstream_model: DownstreamModel,
    device: torch.device,
    segment_length: float = 4.0,
    segment_overlap: float = 0.5,
) -> dict:
    """
    Run inference for a single patient.

    Returns dict with segment-level and patient-level predictions.
    """
    # Segment audio
    segments = segment_audio(audio_path, segment_length=segment_length, segment_overlap=segment_overlap)

    if len(segments) == 0:
        raise ValueError(f"No valid segments from {audio_path}. Audio may be too short.")

    # Build clinical prompt
    prompt = build_clinical_prompt(**clinical_info)
    tokens = text_encoder.tokenizer(
        prompt, padding="max_length", truncation=True,
        max_length=128, return_tensors="pt",
    )

    # Encode clinical profile (once)
    input_ids = tokens["input_ids"].to(device)
    attn_mask = tokens["attention_mask"].to(device)

    with torch.no_grad():
        clinical_emb = text_encoder(input_ids, attn_mask).squeeze(0)  # (d_t,)

    # Process each speech segment
    segment_probs = []
    for seg_waveform in segments:
        with torch.no_grad():
            H_a = speech_encoder(seg_waveform.unsqueeze(0).to(device))  # (1, L, d_a)
            output = downstream_model(H_a, clinical_emb.unsqueeze(0))
            segment_probs.append(output["y_hat"].item())

    segment_probs = np.array(segment_probs)

    # Majority voting
    mv = MajorityVoting()
    mv_pred = mv(segment_probs)

    return {
        "num_segments": len(segments),
        "segment_probabilities": segment_probs.tolist(),
        "mean_probability": float(segment_probs.mean()),
        "majority_voting": {
            "prediction": "Severe OSA" if mv_pred == 1 else "Non-severe",
        },
        "clinical_prompt": prompt,
    }


def main():
    parser = argparse.ArgumentParser(description="Run OSA severity inference")
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--age", type=float, required=True)
    parser.add_argument("--gender", type=str, required=True, choices=["male", "female"])
    parser.add_argument("--bmi", type=float, required=True)
    parser.add_argument("--neck_circumference", type=float, required=True)
    parser.add_argument("--waist_circumference", type=float, required=True)
    parser.add_argument("--ess_score", type=float, required=True)
    parser.add_argument("--psqi_score", type=float, required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--speech_model", type=str, default="facebook/wav2vec2-xls-r-300m")
    parser.add_argument("--text_model", type=str, default="emilyalsentzer/Bio_ClinicalBERT")
    parser.add_argument("--segment_length", type=float, default=4.0)
    parser.add_argument("--segment_overlap", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    print("Loading speech encoder...")
    speech_encoder = SpeechEncoder(args.speech_model, freeze=True).to(device)

    print("Loading text encoder...")
    text_encoder = TextEncoder(args.text_model, freeze=True).to(device)

    print("Loading downstream model...")
    checkpoint_path = os.path.join(args.checkpoint_dir, "best_model.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    downstream_model = DownstreamModel(
        speech_hidden_dim=speech_encoder.hidden_dim,
        text_hidden_dim=text_encoder.hidden_dim,
        **checkpoint.get("model_config", {}),
    ).to(device)
    downstream_model.load_state_dict(checkpoint["model_state_dict"])
    downstream_model.eval()

    # Run inference
    clinical_info = {
        "age": args.age,
        "gender": args.gender,
        "bmi": args.bmi,
        "neck_circ": args.neck_circumference,
        "waist_circ": args.waist_circumference,
        "ess_score": args.ess_score,
        "psqi_score": args.psqi_score,
    }

    print(f"\nRunning inference on {args.audio_path}...")
    results = predict(
        audio_path=args.audio_path,
        clinical_info=clinical_info,
        speech_encoder=speech_encoder,
        text_encoder=text_encoder,
        downstream_model=downstream_model,
        device=device,
        segment_length=args.segment_length,
        segment_overlap=args.segment_overlap,
    )

    print(f"\n{'=' * 50}")
    print(f"  OSA Severity Prediction Results")
    print(f"{'=' * 50}")
    print(f"  Clinical profile: {results['clinical_prompt']}")
    print(f"  Number of segments: {results['num_segments']}")
    print(f"  Mean severity probability: {results['mean_probability']:.4f}")
    print(f"  Prediction: {results['majority_voting']['prediction']}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()

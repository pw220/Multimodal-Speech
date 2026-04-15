"""
Generate synthetic demo data for testing the multimodal OSA framework.

Creates:
- Synthetic audio files (random waveforms with basic structure)
- Clinical profiles CSV with realistic distributions

This is for testing/demo purposes only. Real clinical data should follow
the format described in Sec 3.2.
"""

import os
import argparse
import numpy as np
import pandas as pd
import soundfile as sf


def generate_demo_data(output_dir: str, num_patients: int = 50, seed: int = 42):
    """Generate synthetic demo dataset."""
    np.random.seed(seed)

    audio_dir = os.path.join(output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    sample_rate = 16000
    duration = 30.0  # seconds per recording

    records = []

    for i in range(num_patients):
        pid = f"patient_{i + 1:03d}"

        # Generate clinical profile with realistic distributions (Table 2)
        age = np.clip(np.random.normal(41.21, 13.45), 18, 80)
        gender = "male" if np.random.random() < 0.741 else "female"
        bmi = np.clip(np.random.normal(27.64, 5.82), 16, 50)
        neck_circ = np.clip(np.random.normal(39.0 if gender == "male" else 34.0, 3.0), 25, 55)
        waist_circ = np.clip(np.random.normal(95.0 if gender == "male" else 85.0, 12.0), 60, 140)
        ess_score = np.clip(np.random.normal(9, 5), 0, 24).astype(int)
        psqi_score = np.clip(np.random.normal(8, 4), 0, 21).astype(int)

        # AHI distribution matching Table 2 severity proportions
        # ~10% normal, ~24% mild, ~23% moderate, ~43% severe
        severity_draw = np.random.random()
        if severity_draw < 0.10:
            ahi = np.clip(np.random.exponential(2.0), 0, 4.9)
        elif severity_draw < 0.34:
            ahi = np.clip(np.random.uniform(5, 15), 5, 14.9)
        elif severity_draw < 0.57:
            ahi = np.clip(np.random.uniform(15, 30), 15, 29.9)
        else:
            ahi = np.clip(np.random.exponential(15) + 30, 30, 120)

        records.append({
            "patient_id": pid,
            "age": round(age, 1),
            "gender": gender,
            "bmi": round(bmi, 1),
            "neck_circumference": round(neck_circ, 1),
            "waist_circumference": round(waist_circ, 1),
            "ess_score": int(ess_score),
            "psqi_score": int(psqi_score),
            "ahi": round(ahi, 1),
        })

        # Generate synthetic audio for pre-sleep and post-sleep
        for cond in ["pre", "post"]:
            # Create a synthetic speech-like signal
            t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)

            # Base frequency varies with AHI (simulate OSA-related vocal changes)
            base_freq = 120 if gender == "male" else 200
            # Slight formant shift for severe OSA (lower formant frequencies)
            freq_shift = -5 * (ahi / 30.0)

            # Generate harmonics
            signal = np.zeros_like(t)
            for harmonic in range(1, 6):
                freq = (base_freq + freq_shift) * harmonic
                amplitude = 0.5 / harmonic
                signal += amplitude * np.sin(2 * np.pi * freq * t)

            # Add noise
            signal += 0.05 * np.random.randn(len(t))

            # Amplitude envelope (speech-like bursts)
            envelope = np.ones_like(t)
            num_pauses = np.random.randint(5, 15)
            for _ in range(num_pauses):
                pause_start = np.random.randint(0, len(t) - sample_rate)
                pause_len = np.random.randint(sample_rate // 4, sample_rate)
                envelope[pause_start: pause_start + pause_len] *= 0.01

            signal *= envelope

            # Normalize
            signal = signal / (np.abs(signal).max() + 1e-8)
            signal = (signal * 0.9).astype(np.float32)

            filepath = os.path.join(audio_dir, f"{pid}_{cond}.wav")
            sf.write(filepath, signal, sample_rate)

    # Save clinical profiles
    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "clinical_profiles.csv")
    df.to_csv(csv_path, index=False)

    # Print summary
    print(f"Generated {num_patients} patients")
    print(f"Audio files: {audio_dir}/ ({num_patients * 2} files)")
    print(f"Clinical profiles: {csv_path}")
    print(f"\nSeverity distribution:")
    print(f"  Normal (AHI < 5):    {(df['ahi'] < 5).sum()} ({(df['ahi'] < 5).mean() * 100:.1f}%)")
    print(f"  Mild (5-15):         {((df['ahi'] >= 5) & (df['ahi'] < 15)).sum()} ({((df['ahi'] >= 5) & (df['ahi'] < 15)).mean() * 100:.1f}%)")
    print(f"  Moderate (15-30):    {((df['ahi'] >= 15) & (df['ahi'] < 30)).sum()} ({((df['ahi'] >= 15) & (df['ahi'] < 30)).mean() * 100:.1f}%)")
    print(f"  Severe (≥ 30):       {(df['ahi'] >= 30).sum()} ({(df['ahi'] >= 30).mean() * 100:.1f}%)")
    print(f"\nBinary classification (threshold = 30):")
    print(f"  Non-severe: {(df['ahi'] < 30).sum()}")
    print(f"  Severe:     {(df['ahi'] >= 30).sum()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic demo data")
    parser.add_argument("--output_dir", type=str, default="data/demo")
    parser.add_argument("--num_patients", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_demo_data(args.output_dir, args.num_patients, args.seed)

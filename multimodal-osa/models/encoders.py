"""
Modality-specific encoders for the multimodal OSA framework.

Speech Encoder: XLS-R (300M) pretrained speech foundation model (Sec 3.4.2)
Text Encoder: ClinicalBERT for structured clinical profile encoding (Sec 3.4.2)
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, AutoModel, AutoTokenizer


class SpeechEncoder(nn.Module):
    """
    Speech encoder using a pretrained speech foundation model (XLS-R 300M).

    Given an input waveform x, the encoder produces frame-level representations:
        H_a = f_a(x),  H_a ∈ R^{L x d_a}
    where L is the number of temporal frames and d_a is the hidden dim. (Eq. 4)
    """

    def __init__(self, model_name: str = "facebook/wav2vec2-xls-r-300m", freeze: bool = True):
        super().__init__()
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.hidden_dim = self.model.config.hidden_size  # 1024 for XLS-R 300M

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

        self.freeze = freeze

    def forward(self, waveform: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            waveform: (B, N) raw waveform samples at 16kHz
            attention_mask: (B, N) optional mask for padded inputs

        Returns:
            H_a: (B, L, d_a) frame-level acoustic representations
        """
        if self.freeze:
            with torch.no_grad():
                outputs = self.model(waveform, attention_mask=attention_mask)
        else:
            outputs = self.model(waveform, attention_mask=attention_mask)

        return outputs.last_hidden_state  # (B, L, d_a)


class TextEncoder(nn.Module):
    """
    Text encoder using ClinicalBERT for clinical profile encoding.

    The clinical profile is formatted as a natural language prompt and encoded:
        H_t = f_t(P),  H_t ∈ R^{M x d_t}
    The [CLS] token hidden state serves as the global clinical embedding:
        t = H_t[0] ∈ R^{d_t}  (Eq. 5-6)
    """

    def __init__(self, model_name: str = "emilyalsentzer/Bio_ClinicalBERT", freeze: bool = True):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.hidden_dim = self.model.config.hidden_size  # 768 for ClinicalBERT

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

        self.freeze = freeze

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (B, M) tokenized clinical prompt
            attention_mask: (B, M) attention mask

        Returns:
            t: (B, d_t) [CLS] token embedding as global clinical representation
        """
        if self.freeze:
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Extract [CLS] token hidden state (Eq. 6)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # (B, d_t)
        return cls_embedding

    def tokenize_prompts(self, prompts: list, max_length: int = 128) -> dict:
        """Tokenize a list of clinical profile prompts."""
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return encoded


def build_clinical_prompt(age: float, gender: str, bmi: float,
                          neck_circ: float, waist_circ: float,
                          ess_score: float, psqi_score: float) -> str:
    """
    Construct the clinical profile prompt as described in Sec 3.4.2:

    "The patient is a {age}-year-old {gender} with BMI {b}. Neck and waist
     circumferences are {c_n} cm and {c_w} cm, respectively. ESS and PSQI
     scores are {s_ess} and {s_psqi}, respectively."
    """
    return (
        f"The patient is a {int(age)}-year-old {gender} with BMI {bmi:.1f}. "
        f"Neck and waist circumferences are {neck_circ:.1f} cm and {waist_circ:.1f} cm, respectively. "
        f"ESS and PSQI scores are {int(ess_score)} and {int(psqi_score)}, respectively."
    )

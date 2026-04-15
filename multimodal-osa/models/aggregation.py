"""
Patient-level aggregation strategies (Sec 3.5).

Convert segment-level predictions into robust patient-level severity decisions.

1. Majority Voting: Simple binary voting across segments (Eq. 29-30)
2. Statistical Sequence Aggregator: Logistic regression on distributional
   statistics of segment-level predictions (Eq. 31-33)
"""

import numpy as np
from sklearn.linear_model import LogisticRegression


class MajorityVoting:
    """
    Majority voting aggregation (Sec 3.5).

    Each segment probability is binarized at 0.5, then patient-level
    decision is the majority class.  Produces a hard 0/1 output only;
    AUC is not applicable and is not computed for this aggregator.

    ỹ_MV_i = I(Σ_k ỹ_{i,k} > K_i / 2)   (Eq. 30)
    """

    def __call__(self, segment_probs: np.ndarray) -> int:
        """
        Args:
            segment_probs: (K,) array of predicted probabilities for one patient

        Returns:
            prediction: binary patient-level prediction (0 or 1)
        """
        binary_preds = (segment_probs >= 0.5).astype(int)
        vote_count = binary_preds.sum()
        total = len(binary_preds)
        return int(vote_count > total / 2)


class StatisticalSequenceAggregator:
    """
    Statistical sequence aggregation (Sec 3.5).

    Summarizes segment-level predictions using descriptive statistics
    and performs patient-level classification through logistic regression.

    Feature vector (Eq. 32):
        g_i = [mean, std, max, min, median, q75, q25, ỹ_MV_i]^T

    Patient-level prediction (Eq. 33):
        ŷ_stat_i = σ(w^T g_i + b)
    """

    def __init__(self):
        self.model = LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
        )
        self.is_fitted = False

    def extract_features(self, segment_probs: np.ndarray) -> np.ndarray:
        """
        Compute statistical descriptors from segment-level predictions (Eq. 32).

        Args:
            segment_probs: (K,) array of predicted probabilities

        Returns:
            g: (8,) feature vector
        """
        mv = MajorityVoting()
        mv_pred = mv(segment_probs)

        features = np.array([
            np.mean(segment_probs),
            np.std(segment_probs),
            np.max(segment_probs),
            np.min(segment_probs),
            np.median(segment_probs),
            np.percentile(segment_probs, 75),
            np.percentile(segment_probs, 25),
            float(mv_pred),
        ])
        return features

    def fit(self, patient_segment_probs: list, patient_labels: np.ndarray):
        """
        Fit the logistic regression aggregator.

        Args:
            patient_segment_probs: list of (K_i,) arrays, one per patient
            patient_labels: (N,) binary severity labels
        """
        X = np.array([self.extract_features(probs) for probs in patient_segment_probs])
        self.model.fit(X, patient_labels)
        self.is_fitted = True

    def predict(self, segment_probs: np.ndarray) -> tuple:
        """
        Predict patient-level severity from segment predictions.

        Args:
            segment_probs: (K,) array of predicted probabilities

        Returns:
            prediction: binary prediction (0 or 1)
            probability: predicted probability of severe OSA
        """
        if not self.is_fitted:
            raise RuntimeError("Aggregator must be fitted before prediction.")

        features = self.extract_features(segment_probs).reshape(1, -1)
        prediction = int(self.model.predict(features)[0])
        probability = float(self.model.predict_proba(features)[0, 1])
        return prediction, probability

    def predict_batch(self, patient_segment_probs: list) -> tuple:
        """
        Predict for multiple patients.

        Args:
            patient_segment_probs: list of (K_i,) arrays

        Returns:
            predictions: (N,) binary predictions
            probabilities: (N,) predicted probabilities
        """
        X = np.array([self.extract_features(probs) for probs in patient_segment_probs])
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]
        return predictions, probabilities

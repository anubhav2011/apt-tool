"""
TVT-Lite: CPU-optimized Temporal Vision Transformer for behavior classification.
Uses landmark embeddings only (no raw frames). Pure NumPy for portability;
optional PyTorch can be added later for trained weights.
"""
import numpy as np
from typing import Dict, Optional, Any

# Behavior classes aligned with violation types
BEHAVIOR_CLASSES = [
    "normal",
    "left_cheating_glance",
    "right_cheating_glance",
    "phone_lookdown",
    "second_person",
    "face_spoof",
    "occluded_face_behavior",
]
NUM_CLASSES = len(BEHAVIOR_CLASSES)


class TVTLiteModel:
    """
    Lightweight temporal model for landmark sequences.
    Input: (T, D) array, T = time steps, D = 936 (468*2).
    Output: behavior_class, probability, temporal_confidence.
    """

    def __init__(
        self,
        input_dim: int = 936,
        embed_dim: int = 64,
        window_size: int = 24,
        num_classes: int = NUM_CLASSES,
        seed: int = 42,
    ) -> None:
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.num_classes = num_classes
        self._rng = np.random.default_rng(seed)

        # MLP: input_dim -> embed_dim (per-frame embedding)
        scale = 0.02
        self.W_embed = self._rng.standard_normal((input_dim, embed_dim)) * scale
        self.b_embed = np.zeros(embed_dim)

        # Simple temporal aggregation: mean over time then linear to logits
        # (avoids full attention for CPU speed; can be replaced with attention)
        self.W_out = self._rng.standard_normal((embed_dim, num_classes)) * scale
        self.b_out = np.zeros(num_classes)

    def _embed(self, x: np.ndarray) -> np.ndarray:
        """(T, D) -> (T, embed_dim)."""
        return x.astype(np.float32) @ self.W_embed + self.b_embed

    def _temporal_pool(self, emb: np.ndarray) -> np.ndarray:
        """(T, E) -> (E,) mean over time."""
        return np.mean(emb, axis=0)

    def predict(
        self, window: np.ndarray
    ) -> Dict[str, Any]:
        """
        Run forward pass on a single window.
        window: (T, D) float32 array.
        Returns dict with behavior_class, probability, temporal_confidence.
        """
        if window is None or window.size == 0:
            return self._default_prediction()

        window = np.asarray(window, dtype=np.float32)
        if window.ndim == 1:
            window = window.reshape(1, -1)
        t, d = window.shape
        if d != self.input_dim:
            return self._default_prediction()

        # Embed each frame
        emb = self._embed(window)  # (T, embed_dim)
        # Pool over time
        pooled = self._temporal_pool(emb)  # (embed_dim,)
        # Logits
        logits = pooled @ self.W_out + self.b_out  # (num_classes,)

        # Softmax
        logits = logits - logits.max()
        exp = np.exp(np.clip(logits, -20, 20))
        probs = exp / (exp.sum() + 1e-8)

        pred_idx = int(np.argmax(probs))
        probability = float(probs[pred_idx])
        behavior_class = BEHAVIOR_CLASSES[pred_idx]

        # Temporal confidence: entropy-based (low entropy = high confidence)
        entropy = -float(np.sum(probs * np.log(probs + 1e-8)))
        max_entropy = np.log(self.num_classes)
        temporal_confidence = 1.0 - min(entropy / max_entropy, 1.0)

        return {
            "behavior_class": behavior_class,
            "probability": probability,
            "temporal_confidence": temporal_confidence,
            "all_probs": probs.tolist(),
        }

    def _default_prediction(self) -> Dict[str, Any]:
        """Return when input invalid or model not ready (safe default)."""
        return {
            "behavior_class": "normal",
            "probability": 0.0,
            "temporal_confidence": 0.0,
            "all_probs": [1.0 / self.num_classes] * self.num_classes,
        }


def create_tvt_model(config: Any) -> Optional[TVTLiteModel]:
    """
    Create TVT-Lite model from config if ENABLE_TVT is True.
    Returns None if TVT disabled.
    """
    if not getattr(config, "ENABLE_TVT", False):
        return None
    window_size = getattr(config, "TVT_TEMPORAL_WINDOW", 24)
    return TVTLiteModel(
        input_dim=936,
        embed_dim=64,
        num_heads=2,
        num_layers=2,
        window_size=window_size,
        num_classes=NUM_CLASSES,
    )

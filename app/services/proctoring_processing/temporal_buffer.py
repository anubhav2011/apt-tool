"""
Rolling buffer for temporal landmark sequences.
Stores landmark vectors per frame for TVT (Temporal Vision Transformer) input.
"""
import numpy as np
from collections import deque
from typing import Optional, List


class TemporalBuffer:
    """
    Rolling buffer of landmark vectors for a fixed window size.
    Each frame adds one vector of shape (936,) i.e. [x1,y1, ..., x468,y468].
    """

    def __init__(self, window_size: int = 24, landmark_dim: int = 936):
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        self.window_size = window_size
        self.landmark_dim = landmark_dim
        self._buffer: deque = deque(maxlen=window_size)
        self._timestamps: deque = deque(maxlen=window_size)

    def push(self, vector: np.ndarray, timestamp: float = 0.0) -> None:
        """
        Append one frame's landmark vector.
        vector: shape (landmark_dim,) or (468, 2) flattened to (936,).
        """
        v = np.asarray(vector, dtype=np.float32)
        if v.size != self.landmark_dim:
            v = v.flatten()
        if v.size != self.landmark_dim:
            raise ValueError(
                f"Expected landmark_dim={self.landmark_dim}, got {v.size}"
            )
        self._buffer.append(v)
        self._timestamps.append(timestamp)

    def is_ready(self) -> bool:
        """True if buffer has enough frames for inference."""
        return len(self._buffer) >= self.window_size

    def get_window(self) -> Optional[np.ndarray]:
        """
        Return current window as (T, D) with T = window_size, D = landmark_dim.
        Returns None if not ready.
        """
        if not self.is_ready():
            return None
        return np.stack(list(self._buffer), axis=0)

    def get_timestamps(self) -> List[float]:
        """Return timestamps for the current window (oldest first)."""
        return list(self._timestamps)

    def clear(self) -> None:
        """Clear the buffer."""
        self._buffer.clear()
        self._timestamps.clear()

    def __len__(self) -> int:
        return len(self._buffer)

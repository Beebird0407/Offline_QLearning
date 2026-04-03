"""
Action Discretization, Tokenization, and Discretization

- Discretization: Split continuous hyperparameter range into M bins
- Tokenization: Convert bin index to 5-bit binary encoding
- Discretization (reverse): Map bin index back to parameter value
"""

import numpy as np
from typing import Tuple, List, Optional


class ActionDiscretizer:
    def __init__(self, M: int = 16):
        self.M = M
    def discretize(self, value: float, lo: float, hi: float) -> int:
        normalized = (value - lo) / (hi - lo + 1e-8)
        normalized = np.clip(normalized, 0, 1)
        return int(normalized * (self.M - 1))

    def undiscretize(self, bin_idx: int, lo: float, hi: float) -> float:
        bin_idx = int(np.clip(bin_idx, 0, self.M - 1))
        return lo + (bin_idx + 0.5) * (hi - lo) / self.M

    def get_bin_edges(self, lo: float, hi: float) -> np.ndarray:
        """Get bin edge values."""
        return np.linspace(lo, hi, self.M + 1)

    def get_bin_centers(self, lo: float, hi: float) -> np.ndarray:
        """Get bin center values."""
        edges = self.get_bin_edges(lo, hi)
        return (edges[:-1] + edges[1:]) / 2


class ActionTokenizer:
    TOKEN_DIM = 5  # 5-bit encoding

    def __init__(self, M: int = 16):
        self.M = M
        self._start_token_value = (1 << self.TOKEN_DIM) - 1  # 31 = 11111

    def tokenize(self, bin_idx: int) -> np.ndarray:
        bin_idx = int(np.clip(bin_idx, 0, self.M - 1))
        token = np.zeros(self.TOKEN_DIM, dtype=np.float32)
        for b in range(self.TOKEN_DIM):
            token[self.TOKEN_DIM - 1 - b] = (bin_idx >> b) & 1
        return token

    def tokenize_batch(self, bin_indices: np.ndarray) -> np.ndarray:
        original_shape = bin_indices.shape
        flat = bin_indices.flatten()
        tokens = np.zeros((len(flat), self.TOKEN_DIM), dtype=np.float32)
        for i, idx in enumerate(flat):
            tokens[i] = self.tokenize(idx)
        return tokens.reshape(*original_shape, self.TOKEN_DIM)

    def detokenize(self, token: np.ndarray) -> int:
        if hasattr(token, 'numpy'):
            token = token.numpy()
        bin_idx = 0
        for b in range(self.TOKEN_DIM):
            bin_idx |= (int(token[self.TOKEN_DIM - 1 - b]) << b)
        return min(bin_idx, self.M - 1)

    def get_start_token(self, batch_size: Optional[int] = None) -> np.ndarray:
        token = np.zeros(self.TOKEN_DIM, dtype=np.float32)
        token[:] = 1  # All ones = 11111
        if batch_size is not None:
            token = np.tile(token, (batch_size, 1))
        return token

    @property
    def start_token_value(self) -> int:
        """Get start token as integer value."""
        return self._start_token_value


class ActionSpace:

    def __init__(self, K: int, M: int = 16, param_ranges: Optional[List[Tuple[float, float]]] = None):
        self.K = K
        self.M = M
        self.param_ranges = param_ranges or [(0.0, 1.0)] * K

        self.discretizer = ActionDiscretizer(M)
        self.tokenizer = ActionTokenizer(M)

    def discretize_params(self, params: np.ndarray) -> np.ndarray:
        bins = np.zeros(self.K, dtype=np.int64)
        for i, (value, (lo, hi)) in enumerate(zip(params, self.param_ranges)):
            bins[i] = self.discretizer.discretize(value, lo, hi)
        return bins

    def undiscretize_bins(self, bins: np.ndarray) -> np.ndarray:
        params = np.zeros(self.K, dtype=np.float32)
        for i, (bin_idx, (lo, hi)) in enumerate(zip(bins, self.param_ranges)):
            params[i] = self.discretizer.undiscretize(bin_idx, lo, hi)
        return params

    def tokenize_bins(self, bins: np.ndarray) -> np.ndarray:
        return self.tokenizer.tokenize_batch(bins)

    def get_action_dim(self) -> int:
        """Get total action dimension (K * token_dim)."""
        return self.K * self.TOKEN_DIM

    @property
    def token_dim(self) -> int:
        """Get token dimension."""
        return self.TOKEN_DIM
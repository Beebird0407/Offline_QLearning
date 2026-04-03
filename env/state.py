"""
State Feature Extraction for Black-Box Optimization

calc_state(population, fitnesses, t, T, best_so_far) -> 9-dim vector

Based on paper Table 4:
1. Normalized mean fitness (fitness - mean) / std
2. Normalized best fitness
3. Normalized worst fitness
4. Population diversity (std of distances to centroid)
5. Population correlation (mean pairwise correlation)
6. Progress: t / T
7. Remaining: 1 - t/T
8. Log progress: log(t+1) / log(T+1)
9. Improvement: (best_so_far - current_best) / (|best_so_far| + eps)
"""

import numpy as np
from typing import Optional


def calc_state(
    population: np.ndarray,
    fitnesses: np.ndarray,
    t: int,
    T: int,
    best_so_far: Optional[float] = None
) -> np.ndarray:
    pop_size, dim = population.shape

    # 1. Normalized mean fitness
    f_mean = np.mean(fitnesses)
    f_std = np.std(fitnesses) + 1e-8
    norm_mean = (f_mean - f_mean) / f_std  # = 0

    # 2. Normalized best fitness
    f_best = np.min(fitnesses)
    norm_best = (f_best - f_mean) / f_std

    # 3. Normalized worst fitness
    f_worst = np.max(fitnesses)
    norm_worst = (f_worst - f_mean) / f_std

    # 4. Population diversity (std of distances to centroid)
    centroid = np.mean(population, axis=0)
    distances = np.linalg.norm(population - centroid, axis=1)
    diversity = np.std(distances) / (np.mean(distances) + 1e-8)

    # 5. Population correlation (mean pairwise correlation between dimensions)
    if dim > 1:
        pop_centered = population - centroid
        pop_std = np.std(pop_centered, axis=0)
        # Check for zero std (constant population)
        if pop_std.min() > 1e-10:
            try:
                corr_matrix = np.corrcoef(pop_centered.T)
                # Get upper triangle off-diagonal elements
                upper_tri_indices = np.triu_indices(dim, k=1)
                mean_corr = np.mean(corr_matrix[upper_tri_indices])
                if np.isnan(mean_corr):
                    mean_corr = 0.0
            except Exception:
                mean_corr = 0.0
        else:
            mean_corr = 0.0
    else:
        mean_corr = 0.0

    # 6. Progress: t / T
    progress = t / T if T > 0 else 0.0

    # 7. Remaining: 1 - t/T
    remaining = 1.0 - progress

    # 8. Log progress: log(t+1) / log(T+1)
    log_progress = np.log(t + 1) / np.log(T + 1) if T > 0 else 0.0

    # 9. Improvement from best_so_far
    if best_so_far is not None and best_so_far != float('inf'):
        improvement = (best_so_far - f_best) / (abs(best_so_far) + 1e-8)
    else:
        improvement = 0.0

    # Clip to reasonable ranges
    def clip(x, lo, hi):
        return float(np.clip(x, lo, hi))

    state = np.array([
        clip(norm_mean, -5, 5),      # 1. normalized mean fitness
        clip(norm_best, -5, 5),      # 2. normalized best fitness
        clip(norm_worst, -5, 5),     # 3. normalized worst fitness
        clip(diversity, 0, 10),       # 4. population diversity
        clip(mean_corr, -1, 1),      # 5. population correlation
        clip(progress, 0, 1),        # 6. progress
        clip(remaining, 0, 1),       # 7. remaining
        clip(log_progress, 0, 1),    # 8. log progress
        clip(improvement, -5, 5),    # 9. improvement
    ], dtype=np.float32)

    return state


class StateExtractor:
    """State extractor with history tracking."""

    def __init__(self):
        self.history_best = []
        self.prev_best = None

    def compute(self, population: np.ndarray, fitnesses: np.ndarray, t: int, T: int) -> np.ndarray:
        # Get best so far from history
        best_so_far = self.history_best[-1] if self.history_best else None

        state = calc_state(population, fitnesses, t, T, best_so_far)

        # Update history
        current_best = float(np.min(fitnesses))
        if self.prev_best is not None:
            if current_best < self.prev_best:  # Improvement (minimization)
                self.history_best.append(current_best)
                if len(self.history_best) > 100:
                    self.history_best.pop(0)
        else:
            self.history_best.append(current_best)

        self.prev_best = current_best

        return state

    def reset(self):
        """Reset history."""
        self.history_best = []
        self.prev_best = None

    @staticmethod
    def state_dim() -> int:
        """Return state dimension."""
        return 9
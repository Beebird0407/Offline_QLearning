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

    f_mean = np.mean(fitnesses)
    f_std = np.std(fitnesses) + 1e-8
    f_min = np.min(fitnesses)
    f_max = np.max(fitnesses)
    f_range = f_max - f_min + 1e-8
    norm_mean = (f_mean - f_min) / f_range

    f_best = np.min(fitnesses)
    norm_best = (f_best - f_mean) / f_std

    f_worst = np.max(fitnesses)
    norm_worst = (f_worst - f_mean) / f_std

    centroid = np.mean(population, axis=0)
    distances = np.linalg.norm(population - centroid, axis=1)
    diversity = np.std(distances) / (np.mean(distances) + 1e-8)

    if dim > 1:
        pop_centered = population - centroid
        pop_std = np.std(pop_centered, axis=0)
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

    progress = t / T if T > 0 else 0.0
    remaining = 1.0 - progress
    log_progress = np.log(t + 1) / np.log(T + 1) if T > 0 else 0.0

    if best_so_far is not None and best_so_far != float('inf'):
        improvement = (best_so_far - f_best) / (abs(best_so_far) + 1e-8)
    else:
        improvement = 0.0

    # Clip to reasonable ranges
    def clip(x, lo, hi):
        return float(np.clip(x, lo, hi))

    state = np.array([
        clip(norm_mean, -5, 5),
        clip(norm_best, -5, 5),
        clip(norm_worst, -5, 5),
        clip(diversity, 0, 10),
        clip(mean_corr, -1, 1),
        clip(progress, 0, 1),
        clip(remaining, 0, 1),
        clip(log_progress, 0, 1),
        clip(improvement, -5, 5),
    ], dtype=np.float32)

    return state


class StateExtractor:
    """State extractor with history tracking."""

    def __init__(self):
        self.history_best = []
        self.prev_best = None

    def compute(self, population: np.ndarray, fitnesses: np.ndarray, t: int, T: int) -> np.ndarray:
        best_so_far = self.history_best[-1] if self.history_best else None
        state = calc_state(population, fitnesses, t, T, best_so_far)

        current_best = float(np.min(fitnesses))
        if self.prev_best is not None:
            if current_best < self.prev_best:
                self.history_best.append(current_best)
                if len(self.history_best) > 100:
                    self.history_best.pop(0)
        else:
            self.history_best.append(current_best)
        self.prev_best = current_best
        return state

    def reset(self):
        self.history_best = []
        self.prev_best = None

    @staticmethod
    def state_dim() -> int:
        return 9
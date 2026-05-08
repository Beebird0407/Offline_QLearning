import numpy as np
from typing import Optional


def calc_state(
    population: np.ndarray,
    fitnesses: np.ndarray,
    t: int,
    T: int,
    gbest: float,
    gbest_solution: np.ndarray,
    init_max: float,
) -> np.ndarray:
    """Q-Mamba-main cal_feature: 9-dim state with log-scale fitness + FDC."""
    n, dim = population.shape
    cost = np.maximum(fitnesses, 1e-8)
    bounds_range = 10.0  # [-5, 5] → range = 10

    cbest = np.min(cost)
    cbest_idx = np.argmin(cost)
    cbest_solution = population[cbest_idx]

    # 1-4: log10 fitness features
    gbest_log = np.log10(max(1e-8, gbest))
    cbest_log = np.log10(max(1e-8, cbest))
    cost_log = np.log10(cost)
    init_max_log = np.log10(init_max)
    f1 = gbest_log / init_max_log
    f2 = cbest_log / init_max_log
    f3 = float(np.mean(cost_log / init_max_log))
    f4 = float(np.std(cost_log / init_max_log))

    # 5: max pairwise distance
    dist = np.sqrt(np.sum((population[None, :, :] - population[:, None, :]) ** 2, axis=-1))
    f5 = float(np.max(dist) / bounds_range / np.sqrt(dim))

    # 6: top-10% elite concentration
    top_n = max(1, int(0.1 * n))
    top_idx = np.argsort(cost)[:top_n]
    dist_top = np.sqrt(np.sum(
        (population[top_idx][None, :, :] - population[top_idx][:, None, :]) ** 2, axis=-1
    ))
    f6 = float((np.mean(dist_top) - np.mean(dist)) / bounds_range / np.sqrt(dim))

    # 7: FDC to global best
    d_gbest = np.sqrt(np.sum((population - gbest_solution) ** 2, axis=-1))
    c_gbest = cost - gbest
    nom = np.mean((c_gbest - np.mean(c_gbest)) * (d_gbest - np.mean(d_gbest)))
    den = np.std(c_gbest) * np.std(d_gbest) + 1e-8
    f7 = float(nom / den)

    # 8: FDC to current best
    d_cbest = np.sqrt(np.sum((population - cbest_solution) ** 2, axis=-1))
    c_cbest = cost - cbest
    nom = np.mean((c_cbest - np.mean(c_cbest)) * (d_cbest - np.mean(d_cbest)))
    den = np.std(c_cbest) * np.std(d_cbest) + 1e-8
    f8 = float(nom / den)

    # 9: remaining generations
    f9 = float((T - t) / T if T > 0 else 0.0)

    return np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9], dtype=np.float32)


class StateExtractor:
    def __init__(self):
        self.gbest = float('inf')
        self.gbest_solution = None
        self.init_max = None
        self.prev_best = float('inf')

    def compute(self, population: np.ndarray, fitnesses: np.ndarray, t: int, T: int) -> np.ndarray:
        cbest = float(np.min(fitnesses))
        cbest_idx = int(np.argmin(fitnesses))

        if self.init_max is None:
            self.init_max = float(np.max(fitnesses))
        if cbest < self.gbest:
            self.gbest = cbest
            self.gbest_solution = population[cbest_idx].copy()

        state = calc_state(population, fitnesses, t, T, self.gbest, self.gbest_solution, self.init_max)
        self.prev_best = cbest
        return state

    def reset(self):
        self.gbest = float('inf')
        self.gbest_solution = None
        self.init_max = None
        self.prev_best = float('inf')

    @staticmethod
    def state_dim() -> int:
        return 9

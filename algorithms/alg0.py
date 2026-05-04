import numpy as np
from typing import Tuple, Callable


class Alg0Optimizer:
    K = 3

    F1_range = (0.0, 1.0)
    F2_range = (0.0, 1.0)
    Cr_range = (0.0, 1.0)

    def __init__(
        self,
        dim: int,
        bounds: np.ndarray,
        pop_size: int = 100,
        seed: int = 42,
        use_lpsr: bool = True,
        min_pop_size: int = 4
    ):
        self.dim = dim
        self.bounds = bounds
        self.pop_size_init = pop_size
        self.pop_size = pop_size
        self.min_pop_size = min_pop_size
        self.use_lpsr = use_lpsr
        self.rng = np.random.RandomState(seed)
        self._step_count = 0
        self.best_x = None
        self.best_f = float('inf')

    def bin_to_F1(self, b: int, M: int = 16) -> float:
        return self.F1_range[0] + (b + 0.5) * (self.F1_range[1] - self.F1_range[0]) / M

    def bin_to_F2(self, b: int, M: int = 16) -> float:
        return self.F2_range[0] + (b + 0.5) * (self.F2_range[1] - self.F2_range[0]) / M

    def bin_to_Cr(self, b: int, M: int = 16) -> float:
        return self.Cr_range[0] + (b + 0.5) * (self.Cr_range[1] - self.Cr_range[0]) / M

    def bin_to_params(self, bins: Tuple[int, int, int], M: int = 16) -> Tuple[float, float, float]:
        return self.bin_to_F1(bins[0], M), self.bin_to_F2(bins[1], M), self.bin_to_Cr(bins[2], M)

    def initialize(self) -> np.ndarray:
        return np.column_stack([
            self.rng.uniform(lo, hi, self.pop_size)
            for lo, hi in self.bounds
        ])

    def _mutate_current_to_rand(
        self,
        target: np.ndarray,
        population: np.ndarray,
        F1: float,
        F2: float
    ) -> np.ndarray:
        n = len(population)
        indices = [i for i in range(n)]
        self.rng.shuffle(indices)
        r1, r2, r3, r4 = indices[:4]

        mutant = target.copy()
        mutant += F1 * (population[r1] - population[r2])
        mutant += F2 * (population[r3] - population[r4])
        return mutant

    def _crossover_exponential(
        self,
        target: np.ndarray,
        mutant: np.ndarray,
        Cr: float
    ) -> np.ndarray:
        dim = len(target)
        trial = target.copy()

        n = self.rng.randint(dim)
        for i in range(dim):
            idx = (n + i) % dim
            trial[idx] = mutant[idx]
            if self.rng.rand() >= Cr:
                break

        return trial

    def _bound(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, self.bounds[:, 0], self.bounds[:, 1])

    def _update_population_size(self, t: int, T: int):
        if not self.use_lpsr:
            return

        progress = t / T if T > 0 else 0
        new_size = max(
            self.min_pop_size,
            int(self.pop_size_init - (self.pop_size_init - self.min_pop_size) * progress)
        )
        if new_size < self.pop_size and new_size >= self.min_pop_size:
            self.pop_size = new_size

    def step(
        self,
        pop: np.ndarray,
        fitness: np.ndarray,
        action: Tuple[int, int, int],
        func: Callable,
        t: int = 0,
        T: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        F1_bin, F2_bin, Cr_bin = int(action[0]), int(action[1]), int(action[2])
        M = 16  # Default bin count
        F1 = self.bin_to_F1(F1_bin, M)
        F2 = self.bin_to_F2(F2_bin, M)
        Cr = self.bin_to_Cr(Cr_bin, M)

        self._update_population_size(t, T)

        current_pop_size = len(pop)
        new_pop = pop.copy()
        new_fit = fitness.copy()

        for i in range(current_pop_size):
            mutant = self._mutate_current_to_rand(pop[i], pop, F1, F2)

            mutant = self._bound(mutant)

            trial = self._crossover_exponential(pop[i], mutant, Cr)
            trial = self._bound(trial)

            f_trial = func(trial)
            if f_trial <= fitness[i]:
                new_pop[i] = trial
                new_fit[i] = f_trial

                # Update best
                if f_trial < self.best_f:
                    self.best_f = f_trial
                    self.best_x = trial.copy()

        if current_pop_size > self.pop_size:
            idx = np.argsort(new_fit)[:self.pop_size]
            new_pop = new_pop[idx]
            new_fit = new_fit[idx]

        self._step_count += 1
        return new_pop, new_fit
"""
Alg1: Hybrid GA + DE with 10 controllable parameters

GA Subgroup: MPX crossover, Gaussian mutation, Roulette selection
DE Subgroup: DE/best/2 mutation, Binomial crossover
Inter-subgroup information sharing (cm1, cm2)
"""

import numpy as np
from typing import Optional, Tuple, Callable, List


class Alg1Optimizer:
    """
    Hybrid GA-DE optimizer with two subgroups sharing information.

    Controllable parameters (10 total):
    - GA subgroup (5 params): mutation_sigma, crossover_prob, selection_pressure, ga_ratio, elite_ratio
    - DE subgroup (5 params): F1, F2, Cr, de_ratio, survivor_ratio
    - Shared (2 params): cm1, cm2 (communication rates)
    """

    K = 10  # Number of action parameters

    # GA parameter ranges
    sigma_range = (0.01, 0.5)      # Mutation strength
    ga_cr_range = (0.3, 1.0)      # GA crossover probability
    ga_ratio_range = (0.3, 0.7)   # GA subgroup ratio
    elite_range = (0.0, 0.2)       # Elite ratio

    # DE parameter ranges
    F1_range = (0.1, 1.0)
    F2_range = (0.1, 1.0)
    Cr_range = (0.0, 1.0)
    de_ratio_range = (0.3, 0.7)    # DE subgroup ratio

    # Communication parameter ranges
    cm1_range = (0.0, 0.3)        # GA -> DE communication rate
    cm2_range = (0.0, 0.3)        # DE -> GA communication rate

    def __init__(
        self,
        dim: int,
        bounds: np.ndarray,
        pop_size: int = 20,
        seed: int = 42
    ):
        self.dim = dim
        self.bounds = bounds
        self.pop_size = pop_size
        self.rng = np.random.RandomState(seed)

        # Subgroup tracking
        self.ga_indices: List[int] = []
        self.de_indices: List[int] = []

        # Best solution
        self.best_x: Optional[np.ndarray] = None
        self.best_f: float = float('inf')

    def bin_to_param(self, b: int, lo: float, hi: float, M: int = 16) -> float:
        """Convert bin index to parameter value."""
        return lo + (hi - lo) * b / (M - 1)

    def bin_to_params(self, bins: Tuple[int, ...], M: int = 16) -> dict:
        """Convert bin indices to all parameter values."""
        return {
            'sigma': self.bin_to_param(bins[0], *self.sigma_range, M),
            'ga_cr': self.bin_to_param(bins[1], *self.ga_cr_range, M),
            'ga_ratio': self.bin_to_param(bins[2], *self.ga_ratio_range, M),
            'elite_ratio': self.bin_to_param(bins[3], *self.elite_range, M),
            'F1': self.bin_to_param(bins[4], *self.F1_range, M),
            'F2': self.bin_to_param(bins[5], *self.F2_range, M),
            'Cr': self.bin_to_param(bins[6], *self.Cr_range, M),
            'de_ratio': self.bin_to_param(bins[7], *self.de_ratio_range, M),
            'cm1': self.bin_to_param(bins[8], *self.cm1_range, M),
            'cm2': self.bin_to_param(bins[9], *self.cm2_range, M),
        }

    def initialize(self) -> np.ndarray:
        """Initialize population."""
        return np.column_stack([
            self.rng.uniform(lo, hi, self.pop_size)
            for lo, hi in self.bounds
        ])

    def _partition_population(self, params: dict):
        """Partition population into GA and DE subgroups."""
        n_ga = max(1, int(self.pop_size * params['ga_ratio']))
        n_de = self.pop_size - n_ga

        indices = list(range(self.pop_size))
        self.rng.shuffle(indices)

        self.ga_indices = indices[:n_ga]
        self.de_indices = indices[n_ga:]

        return n_ga, n_de

    def _gaussian_mutation(self, x: np.ndarray, sigma: float) -> np.ndarray:
        """Gaussian mutation."""
        mutant = x + sigma * self.rng.randn(self.dim)
        return np.clip(mutant, self.bounds[:, 0], self.bounds[:, 1])

    def _mpx_crossover(self, p1: np.ndarray, p2: np.ndarray, cr: float) -> Tuple[np.ndarray, np.ndarray]:
        """Multiple point crossover."""
        mask = self.rng.rand(self.dim) < cr
        c1 = p1.copy()
        c2 = p2.copy()
        c1[mask] = p2[mask]
        c2[mask] = p1[mask]
        return c1, c2

    def _roulette_selection(
        self,
        pop: np.ndarray,
        fitness: np.ndarray,
        pressure: float = 2.0
    ) -> np.ndarray:
        """Roulette wheel selection (fitness-proportional)."""
        # Convert to maximization fitness (assuming minimization in input)
        max_f = fitness.max() + 1e-8
        probs = (max_f - fitness) / (max_f - fitness).sum()

        # Apply selection pressure
        probs = probs ** pressure
        probs /= probs.sum()

        idx = self.rng.choice(len(pop), p=probs)
        return pop[idx].copy()

    def _de_best2_mutation(
        self,
        target: np.ndarray,
        pop: np.ndarray,
        fitness: np.ndarray,
        F1: float,
        F2: float
    ) -> np.ndarray:
        """DE/best/2 mutation."""
        best_idx = np.argmin(fitness)
        candidates = [i for i in range(len(pop)) if i != best_idx]
        self.rng.shuffle(candidates)
        r1, r2, r3, r4 = candidates[:4]

        mutant = pop[best_idx].copy()
        mutant += F1 * (pop[r1] - pop[r2])
        mutant += F2 * (pop[r3] - pop[r4])
        return mutant

    def _binomial_crossover(
        self,
        target: np.ndarray,
        mutant: np.ndarray,
        Cr: float
    ) -> np.ndarray:
        """Binomial (uniform) crossover."""
        mask = self.rng.rand(self.dim) < Cr
        if not mask.any():
            mask[self.rng.randint(self.dim)] = True

        trial = target.copy()
        trial[mask] = mutant[mask]
        return np.clip(trial, self.bounds[:, 0], self.bounds[:, 1])

    def _share_information(
        self,
        pop: np.ndarray,
        fitness: np.ndarray,
        cm1: float,
        cm2: float
    ) -> np.ndarray:
        """Inter-subgroup information sharing via emigrated individuals."""
        new_pop = pop.copy()

        # GA -> DE: copy best GA individual to random DE locations
        if len(self.ga_indices) > 0 and cm1 > 0:
            n_share = max(1, int(len(self.de_indices) * cm1))
            ga_best_idx = self.ga_indices[np.argmin(fitness[self.ga_indices])]
            for idx in self.rng.choice(self.de_indices, n_share, replace=False):
                new_pop[idx] = pop[ga_best_idx].copy()

        # DE -> GA: copy best DE individual to random GA locations
        if len(self.de_indices) > 0 and cm2 > 0:
            n_share = max(1, int(len(self.ga_indices) * cm2))
            de_best_idx = self.de_indices[np.argmin(fitness[self.de_indices])]
            for idx in self.rng.choice(self.ga_indices, n_share, replace=False):
                new_pop[idx] = pop[de_best_idx].copy()

        return new_pop

    def step(
        self,
        pop: np.ndarray,
        fitness: np.ndarray,
        action: Tuple[int, ...],
        func: Callable,
        t: int = 0,
        T: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """One optimization step."""
        params = self.bin_to_params(action)
        n_ga, n_de = self._partition_population(params)

        new_pop = pop.copy()
        new_fit = fitness.copy()

        # === GA Subgroup Evolution ===
        for i in self.ga_indices:
            # Selection for mating
            p1 = self._roulette_selection(pop, fitness)
            p2 = self._roulette_selection(pop, fitness)

            # Crossover
            c1, _ = self._mpx_crossover(p1, p2, params['ga_cr'])

            # Mutation
            c1 = self._gaussian_mutation(c1, params['sigma'])

            # Evaluation
            f_c1 = func(c1)

            # Elite preservation
            if f_c1 <= fitness[i] or i in sorted(self.ga_indices, key=lambda x: fitness[x])[:max(1, int(n_ga * params['elite_ratio']))]:
                new_pop[i] = c1
                new_fit[i] = f_c1

        # === DE Subgroup Evolution ===
        for i in self.de_indices:
            # Mutation: DE/best/2
            mutant = self._de_best2_mutation(pop[i], pop, fitness, params['F1'], params['F2'])

            # Bounding
            mutant = np.clip(mutant, self.bounds[:, 0], self.bounds[:, 1])

            # Crossover: binomial
            trial = self._binomial_crossover(pop[i], mutant, params['Cr'])

            # Selection
            f_trial = func(trial)
            if f_trial <= fitness[i]:
                new_pop[i] = trial
                new_fit[i] = f_trial

        # === Inter-subgroup Information Sharing ===
        new_pop = self._share_information(new_pop, new_fit, params['cm1'], params['cm2'])

        # Update best
        best_idx = np.argmin(new_fit)
        if new_fit[best_idx] < self.best_f:
            self.best_f = new_fit[best_idx]
            self.best_x = new_pop[best_idx].copy()

        return new_pop, new_fit

    def reset(self):
        """Reset optimizer state."""
        self.ga_indices = []
        self.de_indices = []
        self.best_x = None
        self.best_f = float('inf')

    @property
    def name(self) -> str:
        return "Alg1_Hybrid_GA_DE"


Alg1 = Alg1Optimizer
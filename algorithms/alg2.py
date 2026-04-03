"""
Alg2: 4-Subgroup Heterogeneous Algorithm with 16 Controllable parameters

Subgroup 1: MPX + Polynomial mutation
Subgroup 2: SBX + Gaussian mutation + Tournament selection
Subgroup 3: DE/rand/2 + Exponential crossover
Subgroup 4: DE/current-to-best/1 + Binomial crossover
Inter-subgroup sharing mechanism (cm1, cm2, cm3, cm4)
"""

import numpy as np
from typing import Optional, Tuple, Callable, List, Dict


class Alg2Optimizer:
    """
    4-Subgroup heterogeneous evolutionary algorithm.

    Controllable parameters (16 total):
    - Subgroup 1 (4 params): sg1_cr, sg1_eta, sg1_ratio, sg1_cm
    - Subgroup 2 (4 params): sg2_cr, sg2_sigma, sg2_ratio, sg2_cm
    - Subgroup 3 (4 params): sg3_F1, sg3_F2, sg3_Cr, sg3_ratio
    - Subgroup 4 (4 params): sg4_F1, sg4_F2, sg4_Cr, sg4_ratio
    - Shared: communication rates (cm1~cm4)
    """

    K = 16  # Number of action parameters

    # Subgroup 1: MPX + Polynomial mutation
    sg1_cr_range = (0.3, 1.0)
    sg1_eta_range = (5.0, 50.0)     # Polynomial mutation index
    sg1_ratio_range = (0.1, 0.4)
    sg1_cm_range = (0.0, 0.3)

    # Subgroup 2: SBX + Gaussian + Tournament
    sg2_cr_range = (0.3, 1.0)
    sg2_sigma_range = (0.01, 0.5)
    sg2_ratio_range = (0.1, 0.4)
    sg2_cm_range = (0.0, 0.3)

    # Subgroup 3: DE/rand/2 + Exponential
    sg3_F1_range = (0.1, 1.0)
    sg3_F2_range = (0.1, 1.0)
    sg3_Cr_range = (0.0, 1.0)
    sg3_ratio_range = (0.1, 0.4)

    # Subgroup 4: DE/current-to-best/1 + Binomial
    sg4_F1_range = (0.1, 1.0)
    sg4_F2_range = (0.1, 1.0)
    sg4_Cr_range = (0.0, 1.0)
    sg4_ratio_range = (0.1, 0.4)

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

        # Subgroup indices
        self.sg_indices: Dict[int, List[int]] = {1: [], 2: [], 3: [], 4: []}

        # Best solution
        self.best_x: Optional[np.ndarray] = None
        self.best_f: float = float('inf')

    def bin_to_param(self, b: int, lo: float, hi: float, M: int = 16) -> float:
        """Convert bin index to parameter value."""
        return lo + (hi - lo) * b / (M - 1)

    def bin_to_params(self, bins: Tuple[int, ...], M: int = 16) -> dict:
        """Convert bin indices to all parameter values."""
        return {
            # Subgroup 1: MPX + Polynomial
            'sg1_cr': self.bin_to_param(bins[0], *self.sg1_cr_range, M),
            'sg1_eta': self.bin_to_param(bins[1], *self.sg1_eta_range, M),
            'sg1_ratio': self.bin_to_param(bins[2], *self.sg1_ratio_range, M),
            'sg1_cm': self.bin_to_param(bins[3], *self.sg1_cm_range, M),

            # Subgroup 2: SBX + Gaussian
            'sg2_cr': self.bin_to_param(bins[4], *self.sg2_cr_range, M),
            'sg2_sigma': self.bin_to_param(bins[5], *self.sg2_sigma_range, M),
            'sg2_ratio': self.bin_to_param(bins[6], *self.sg2_ratio_range, M),
            'sg2_cm': self.bin_to_param(bins[7], *self.sg2_cm_range, M),

            # Subgroup 3: DE/rand/2 + Exponential
            'sg3_F1': self.bin_to_param(bins[8], *self.sg3_F1_range, M),
            'sg3_F2': self.bin_to_param(bins[9], *self.sg3_F2_range, M),
            'sg3_Cr': self.bin_to_param(bins[10], *self.sg3_Cr_range, M),
            'sg3_ratio': self.bin_to_param(bins[11], *self.sg3_ratio_range, M),

            # Subgroup 4: DE/current-to-best/1 + Binomial
            'sg4_F1': self.bin_to_param(bins[12], *self.sg4_F1_range, M),
            'sg4_F2': self.bin_to_param(bins[13], *self.sg4_F2_range, M),
            'sg4_Cr': self.bin_to_param(bins[14], *self.sg4_Cr_range, M),
            'sg4_ratio': self.bin_to_param(bins[15], *self.sg4_ratio_range, M),
        }

    def initialize(self) -> np.ndarray:
        """Initialize population."""
        return np.column_stack([
            self.rng.uniform(lo, hi, self.pop_size)
            for lo, hi in self.bounds
        ])

    def _partition_population(self, params: dict) -> Dict[int, int]:
        """Partition population into 4 subgroups."""
        ratios = [
            params['sg1_ratio'],
            params['sg2_ratio'],
            params['sg3_ratio'],
            params['sg4_ratio']
        ]
        total = sum(ratios)
        ratios = [r / total for r in ratios]  # Normalize

        sizes = [max(1, int(self.pop_size * r)) for r in ratios]
        sizes[3] = self.pop_size - sum(sizes[:3])  # Ensure total = pop_size

        indices = list(range(self.pop_size))
        self.rng.shuffle(indices)

        start = 0
        for sg, size in enumerate(sizes, 1):
            self.sg_indices[sg] = indices[start:start + size]
            start += size

        return dict(zip([1, 2, 3, 4], sizes))

    # === Subgroup 1 Operators: MPX + Polynomial ===

    def _mpx_crossover(self, p1: np.ndarray, p2: np.ndarray, cr: float) -> Tuple[np.ndarray, np.ndarray]:
        """Multiple point crossover."""
        mask = self.rng.rand(self.dim) < cr
        c1 = p1.copy()
        c2 = p2.copy()
        c1[mask] = p2[mask]
        c2[mask] = p1[mask]
        return c1, c2

    def _polynomial_mutation(self, x: np.ndarray, eta: float) -> np.ndarray:
        """Polynomial mutation (Deb's method)."""
        mutant = x.copy()
        for i in range(self.dim):
            xl, xu = self.bounds[i]
            if self.rng.rand() < 1.0 / self.dim:
                u = self.rng.rand()
                if u < 0.5:
                    delta = (2 * u) ** (1.0 / (eta + 1)) - 1
                    mutant[i] += delta * (x[i] - xl)
                else:
                    delta = 1 - (2 * (1 - u)) ** (1.0 / (eta + 1))
                    mutant[i] += delta * (xu - x[i])
        return np.clip(mutant, self.bounds[:, 0], self.bounds[:, 1])

    def _evolve_sg1(self, pop: np.ndarray, fitness: np.ndarray, params: dict) -> Tuple[np.ndarray, np.ndarray]:
        """Evolve subgroup 1: MPX + Polynomial."""
        new_pop = pop.copy()
        new_fit = fitness.copy()

        for i in self.sg_indices[1]:
            # Select parents
            idxs = self.rng.choice(len(pop), 2, replace=False)
            p1, p2 = pop[idxs[0]], pop[idxs[1]]

            # Crossover
            c1, _ = self._mpx_crossover(p1, p2, params['sg1_cr'])

            # Mutation
            c1 = self._polynomial_mutation(c1, params['sg1_eta'])

            # Selection
            f_c1 = func(c1) if 'func' in dir() else None
            if f_c1 <= fitness[i]:
                new_pop[i] = c1
                new_fit[i] = f_c1

        return new_pop, new_fit

    # === Subgroup 2 Operators: SBX + Gaussian + Tournament ===

    def _sbx_crossover(self, p1: np.ndarray, p2: np.ndarray, bounds: np.ndarray, eta: float, cr: float) -> Tuple[np.ndarray, np.ndarray]:
        """Simulated Binary Crossover."""
        child1 = p1.copy()
        child2 = p2.copy()

        for i in range(self.dim):
            if self.rng.rand() < cr:
                if abs(p1[i] - p2[i]) > 1e-10:
                    xl, xu = bounds[i]
                    y1, y2 = min(p1[i], p2[i]), max(p1[i], p2[i])
                    y1 = max(xl, y1)
                    y2 = min(xu, y2)

                    u_val = self.rng.rand()
                    if u_val <= 0.5:
                        beta = (2 * u_val) ** (1.0 / (eta + 1))
                    else:
                        beta = (1.0 / (2 * (1 - u_val))) ** (1.0 / (eta + 1))

                    child1[i] = 0.5 * ((y1 + y2) - beta * (y2 - y1))
                    child2[i] = 0.5 * ((y1 + y2) + beta * (y2 - y1))

        return child1, child2

    def _gaussian_mutation(self, x: np.ndarray, sigma: float) -> np.ndarray:
        """Gaussian mutation."""
        mutant = x + sigma * self.rng.randn(self.dim)
        return np.clip(mutant, self.bounds[:, 0], self.bounds[:, 1])

    def _tournament_selection(self, pop: np.ndarray, fitness: np.ndarray, k: int = 2) -> np.ndarray:
        """Tournament selection."""
        idxs = self.rng.choice(len(pop), k, replace=False)
        best_idx = idxs[np.argmin(fitness[idxs])]
        return pop[best_idx].copy()

    def _evolve_sg2(self, pop: np.ndarray, fitness: np.ndarray, params: dict, func: Callable) -> Tuple[np.ndarray, np.ndarray]:
        """Evolve subgroup 2: SBX + Gaussian + Tournament."""
        new_pop = pop.copy()
        new_fit = fitness.copy()

        for i in self.sg_indices[2]:
            # Tournament selection
            p1 = self._tournament_selection(pop, fitness)
            p2 = self._tournament_selection(pop, fitness)

            # SBX crossover
            c1, c2 = self._sbx_crossover(p1, p2, self.bounds, eta=20.0, cr=params['sg2_cr'])

            # Gaussian mutation
            c1 = self._gaussian_mutation(c1, params['sg2_sigma'])
            c2 = self._gaussian_mutation(c2, params['sg2_sigma'])

            # Evaluate
            f_c1 = func(c1)
            f_c2 = func(c2)

            # Select best
            if f_c1 <= f_c2:
                c, f_c = c1, f_c1
            else:
                c, f_c = c2, f_c2

            # Replace if better
            if f_c <= fitness[i]:
                new_pop[i] = c
                new_fit[i] = f_c

        return new_pop, new_fit

    # === Subgroup 3 Operators: DE/rand/2 + Exponential ===

    def _de_rand2_mutation(self, target: np.ndarray, pop: np.ndarray, F1: float, F2: float) -> np.ndarray:
        """DE/rand/2 mutation."""
        indices = list(range(len(pop)))
        self.rng.shuffle(indices)
        r1, r2, r3, r4, r5 = indices[:5]

        mutant = pop[r1].copy()
        mutant += F1 * (pop[r2] - pop[r3])
        mutant += F2 * (pop[r4] - pop[r5])
        return mutant

    def _exponential_crossover(self, target: np.ndarray, mutant: np.ndarray, Cr: float) -> np.ndarray:
        """Exponential crossover."""
        trial = target.copy()
        n = self.rng.randint(self.dim)
        for i in range(self.dim):
            idx = (n + i) % self.dim
            trial[idx] = mutant[idx]
            if self.rng.rand() >= Cr:
                break
        return trial

    def _evolve_sg3(self, pop: np.ndarray, fitness: np.ndarray, params: dict, func: Callable) -> Tuple[np.ndarray, np.ndarray]:
        """Evolve subgroup 3: DE/rand/2 + Exponential."""
        new_pop = pop.copy()
        new_fit = fitness.copy()

        for i in self.sg_indices[3]:
            # Mutation
            mutant = self._de_rand2_mutation(pop[i], pop, params['sg3_F1'], params['sg3_F2'])
            mutant = np.clip(mutant, self.bounds[:, 0], self.bounds[:, 1])

            # Crossover
            trial = self._exponential_crossover(pop[i], mutant, params['sg3_Cr'])
            trial = np.clip(trial, self.bounds[:, 0], self.bounds[:, 1])

            # Selection
            f_trial = func(trial)
            if f_trial <= fitness[i]:
                new_pop[i] = trial
                new_fit[i] = f_trial

        return new_pop, new_fit

    # === Subgroup 4 Operators: DE/current-to-best/1 + Binomial ===

    def _de_current_to_best_mutation(self, target: np.ndarray, pop: np.ndarray, fitness: np.ndarray, F1: float, F2: float) -> np.ndarray:
        """DE/current-to-best/1 mutation."""
        best_idx = np.argmin(fitness)
        indices = [j for j in range(len(pop)) if j != best_idx]
        self.rng.shuffle(indices)
        r1, r2 = indices[:2]

        mutant = target.copy()
        mutant += F1 * (pop[best_idx] - target)
        mutant += F2 * (pop[r1] - pop[r2])
        return mutant

    def _binomial_crossover(self, target: np.ndarray, mutant: np.ndarray, Cr: float) -> np.ndarray:
        """Binomial crossover."""
        mask = self.rng.rand(self.dim) < Cr
        if not mask.any():
            mask[self.rng.randint(self.dim)] = True
        trial = target.copy()
        trial[mask] = mutant[mask]
        return np.clip(trial, self.bounds[:, 0], self.bounds[:, 1])

    def _evolve_sg4(self, pop: np.ndarray, fitness: np.ndarray, params: dict, func: Callable) -> Tuple[np.ndarray, np.ndarray]:
        """Evolve subgroup 4: DE/current-to-best/1 + Binomial."""
        new_pop = pop.copy()
        new_fit = fitness.copy()

        for i in self.sg_indices[4]:
            # Mutation
            mutant = self._de_current_to_best_mutation(pop[i], pop, fitness, params['sg4_F1'], params['sg4_F2'])
            mutant = np.clip(mutant, self.bounds[:, 0], self.bounds[:, 1])

            # Crossover
            trial = self._binomial_crossover(pop[i], mutant, params['sg4_Cr'])

            # Selection
            f_trial = func(trial)
            if f_trial <= fitness[i]:
                new_pop[i] = trial
                new_fit[i] = f_trial

        return new_pop, new_fit

    # === Inter-subgroup Sharing ===

    def _share_between_subgroups(self, pop: np.ndarray, fitness: np.ndarray, params: dict) -> np.ndarray:
        """Share best individuals between subgroups."""
        new_pop = pop.copy()
        cms = [params['sg1_cm'], params['sg2_cm'], 0.0, 0.0]  # SG3, SG4 use fixed sharing

        all_indices = list(range(self.pop_size))
        best_per_sg = {}
        for sg in [1, 2, 3, 4]:
            if self.sg_indices[sg]:
                best_idx = self.sg_indices[sg][np.argmin(fitness[self.sg_indices[sg]])]
                best_per_sg[sg] = best_idx

        # Share from best subgroups to others
        for sg_dest in [1, 2, 3, 4]:
            if not self.sg_indices[sg_dest]:
                continue
            cm = cms[sg_dest - 1]
            if cm <= 0:
                continue

            for sg_src in [1, 2, 3, 4]:
                if sg_src == sg_dest or sg_src not in best_per_sg:
                    continue
                n_share = max(1, int(len(self.sg_indices[sg_dest]) * cm))
                src_best = pop[best_per_sg[sg_src]]
                for idx in self.rng.choice(self.sg_indices[sg_dest], n_share, replace=False):
                    new_pop[idx] = src_best.copy()

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
        self._partition_population(params)

        new_pop, new_fit = pop.copy(), fitness.copy()

        # Evolve each subgroup
        new_pop, new_fit = self._evolve_sg1(new_pop, new_fit, params)
        new_pop, new_fit = self._evolve_sg2(new_pop, new_fit, params, func)
        new_pop, new_fit = self._evolve_sg3(new_pop, new_fit, params, func)
        new_pop, new_fit = self._evolve_sg4(new_pop, new_fit, params, func)

        # Inter-subgroup sharing
        new_pop = self._share_between_subgroups(new_pop, new_fit, params)

        # Update best
        best_idx = np.argmin(new_fit)
        if new_fit[best_idx] < self.best_f:
            self.best_f = new_fit[best_idx]
            self.best_x = new_pop[best_idx].copy()

        return new_pop, new_fit

    def reset(self):
        """Reset optimizer state."""
        self.sg_indices = {1: [], 2: [], 3: [], 4: []}
        self.best_x = None
        self.best_f = float('inf')

    @property
    def name(self) -> str:
        return "Alg2_4Subgroup_Heterogeneous"


Alg2 = Alg2Optimizer
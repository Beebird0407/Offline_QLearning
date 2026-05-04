import numpy as np
from typing import Optional, Tuple, Callable, List


class Alg1Optimizer:
    K = 10

    Cr1_range = (0.0, 1.0)
    Xrmpx_range = (0, 1)
    sigma_range = (0.0, 1.0)
    bc1_range = (0.0, 4.0)
    cm1_range = (1, 2)
    F1_range = (0.0, 1.0)
    F2_range = (0.0, 1.0)
    Cr2_range = (0.0, 1.0)
    bc2_range = (0.0, 4.0)
    cm2_range = (1, 2)

    def __init__(
        self,
        dim: int,
        bounds: np.ndarray,
        pop_size: int = 250,
        seed: int = 42,
        use_lpsr: bool = True,
        min_pop_size: int = 10
    ):
        self.dim = dim
        self.bounds = bounds
        self.pop_size = pop_size
        self.pop_size_init = pop_size
        self.use_lpsr = use_lpsr
        self.min_pop_size = min_pop_size
        self.rng = np.random.RandomState(seed)
        self._step_count = 0

        self.ga_indices: List[int] = []
        self.de_indices: List[int] = []
        self.n_ga = 0
        self.n_de = 0
        self.best_x: Optional[np.ndarray] = None
        self.best_f: float = float('inf')

    def _halton_sequence(self, n: int, dim: int) -> np.ndarray:
        """Generate Halton low-discrepancy sequence for initialization."""
        def _halton_base(i: int, base: int) -> float:
            f, r = 1.0, 0.0
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
                  53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107,
                  109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173]
        points = np.zeros((n, dim))
        for d in range(min(dim, len(primes))):
            for i in range(n):
                points[i, d] = _halton_base(i + 1, primes[d])
        return points

    def bin_to_param(self, b: int, lo: float, hi: float, M: int = 16) -> float:
        return lo + (b + 0.5) * (hi - lo) / M

    def bin_to_discrete(self, b: int, lo: int, hi: int, M: int = 16) -> int:
        return lo + int(round((b + 0.5) * (hi - lo) / M))

    def bin_to_params(self, bins: Tuple[int, ...], M: int = 16) -> dict:
        return {
            'Cr1': self.bin_to_param(bins[0], *self.Cr1_range, M),
            'Xrmpx': self.bin_to_discrete(bins[1], *self.Xrmpx_range, M),
            'sigma': self.bin_to_param(bins[2], *self.sigma_range, M),
            'bc1': self.bin_to_param(bins[3], *self.bc1_range, M),
            'cm1': self.bin_to_discrete(bins[4], *self.cm1_range, M),
            'F1': self.bin_to_param(bins[5], *self.F1_range, M),
            'F2': self.bin_to_param(bins[6], *self.F2_range, M),
            'Cr2': self.bin_to_param(bins[7], *self.Cr2_range, M),
            'bc2': self.bin_to_param(bins[8], *self.bc2_range, M),
            'cm2': self.bin_to_discrete(bins[9], *self.cm2_range, M),
        }

    def initialize(self) -> np.ndarray:
        """Initialize population using Halton low-discrepancy sequence."""
        halton = self._halton_sequence(self.pop_size, self.dim)
        pop = np.zeros((self.pop_size, self.dim))
        for d in range(self.dim):
            lo, hi = self.bounds[d]
            pop[:, d] = lo + halton[:, d] * (hi - lo)
        return pop

    def _partition_population(self, params: dict):
        """Partition into GA (50/250=20%) and DE (200/250=80%) subgroups."""
        ga_ratio = 50.0 / 250.0  # GA=50, DE=200 from paper
        self.n_ga = max(1, int(self.pop_size * ga_ratio))
        self.n_de = self.pop_size - self.n_ga

        indices = list(range(self.pop_size))
        self.rng.shuffle(indices)

        self.ga_indices = indices[:self.n_ga]
        self.de_indices = indices[self.n_ga:]

        return self.n_ga, self.n_de

    def _update_population_size(self, t: int, T: int):
        """LPSR: Linearly reduce GA subgroup from 50 to min_pop_size."""
        if not self.use_lpsr:
            return

        progress = t / T if T > 0 else 0
        ga_ratio = 50.0 / 250.0
        ga_init = max(1, int(self.pop_size_init * ga_ratio))
        new_ga = max(self.min_pop_size, int(ga_init - (ga_init - self.min_pop_size) * progress))
        new_de = self.pop_size_init - ga_init  # DE stays fixed
        new_total = new_ga + new_de

        if new_total < self.pop_size and new_total >= self.min_pop_size + new_de:
            self.pop_size = new_total

    def _boundary_control(self, x: np.ndarray, bc: float) -> np.ndarray:
        """Composite boundary control.

        bc ∈ [0, 4] selects strategy:
          [0,1): clip
          [1,2): random re-init
          [2,3): reflection
          [3,4]: wrapping
        """
        lo = self.bounds[:, 0]
        hi = self.bounds[:, 1]

        if bc < 1.0:
            return np.clip(x, lo, hi)
        elif bc < 2.0:
            mask_lo = x < lo
            mask_hi = x > hi
            x[mask_lo] = lo[mask_lo] + self.rng.rand(mask_lo.sum()) * (hi[mask_lo] - lo[mask_lo])
            x[mask_hi] = lo[mask_hi] + self.rng.rand(mask_hi.sum()) * (hi[mask_hi] - lo[mask_hi])
            return x
        elif bc < 3.0:
            x = np.where(x < lo, 2 * lo - x, x)
            x = np.where(x > hi, 2 * hi - x, x)
            return np.clip(x, lo, hi)
        else:
            range_ = hi - lo
            x = lo + (x - lo) % range_
            return np.clip(x, lo, hi)

    def _mpx_crossover(self, p1: np.ndarray, p2: np.ndarray, cr: float, mode: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """MPX crossover. mode=0: random parent selection, mode=1: ranking-based."""
        mask = self.rng.rand(self.dim) < cr
        c1 = p1.copy()
        c2 = p2.copy()
        c1[mask] = p2[mask]
        c2[mask] = p1[mask]
        return c1, c2

    def _gaussian_mutation(self, x: np.ndarray, sigma: float) -> np.ndarray:
        mutant = x + sigma * self.rng.randn(self.dim)
        return np.clip(mutant, self.bounds[:, 0], self.bounds[:, 1])

    def _roulette_selection(
        self,
        pop: np.ndarray,
        fitness: np.ndarray,
        pressure: float = 2.0
    ) -> np.ndarray:
        """Roulette wheel selection from given population."""
        max_f = fitness.max()
        denom = (max_f - fitness).sum()
        if denom < 1e-12:
            # All fitness values equal — fall back to uniform selection
            idx = self.rng.randint(len(pop))
            return pop[idx].copy()
        probs = (max_f - fitness) / denom
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
        """DE/best/2 mutation using given population."""
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
        cm1: int,
        cm2: int
    ) -> np.ndarray:
        """Inter-subgroup information sharing.

        cm1: which subgroup GA best replaces worst of.
            cm1=1 → GA itself (no sharing), cm1=2 → replace worst DE
        cm2: which subgroup DE best replaces worst of.
            cm2=2 → DE itself (no sharing), cm2=1 → replace worst GA
        """
        new_pop = pop.copy()

        # cm1: GA best → worst of target subgroup
        if len(self.ga_indices) > 0:
            ga_best_idx = self.ga_indices[np.argmin(fitness[self.ga_indices])]
            if cm1 == 2 and len(self.de_indices) > 0:
                worst_de_idx = self.de_indices[np.argmax(fitness[self.de_indices])]
                new_pop[worst_de_idx] = pop[ga_best_idx].copy()

        # cm2: DE best → worst of target subgroup
        if len(self.de_indices) > 0:
            de_best_idx = self.de_indices[np.argmin(fitness[self.de_indices])]
            if cm2 == 1 and len(self.ga_indices) > 0:
                worst_ga_idx = self.ga_indices[np.argmax(fitness[self.ga_indices])]
                new_pop[worst_ga_idx] = pop[de_best_idx].copy()

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
        params = self.bin_to_params(action)
        self._partition_population(params)
        self._update_population_size(t, T)

        new_pop = pop.copy()
        new_fit = fitness.copy()

        ga_pop = pop[self.ga_indices]
        ga_fit = fitness[self.ga_indices]

        for i in self.ga_indices:
            p1 = self._roulette_selection(ga_pop, ga_fit)
            p2 = self._roulette_selection(ga_pop, ga_fit)

            c1, _ = self._mpx_crossover(p1, p2, params['Cr1'], params['Xrmpx'])
            c1 = self._gaussian_mutation(c1, params['sigma'])
            c1 = self._boundary_control(c1, params['bc1'])

            f_c1 = func(c1)
            if f_c1 <= fitness[i]:
                new_pop[i] = c1
                new_fit[i] = f_c1

        de_pop = pop[self.de_indices]
        de_fit = fitness[self.de_indices]

        for i in self.de_indices:
            mutant = self._de_best2_mutation(pop[i], de_pop, de_fit, params['F1'], params['F2'])
            mutant = self._boundary_control(mutant, params['bc2'])
            trial = self._binomial_crossover(pop[i], mutant, params['Cr2'])
            trial = self._boundary_control(trial, params['bc2'])

            f_trial = func(trial)
            if f_trial <= fitness[i]:
                new_pop[i] = trial
                new_fit[i] = f_trial

        new_pop = self._share_information(new_pop, new_fit, params['cm1'], params['cm2'])

        if len(new_pop) > self.pop_size:
            idx = np.argsort(new_fit)[:self.pop_size]
            new_pop = new_pop[idx]
            new_fit = new_fit[idx]

        best_idx = np.argmin(new_fit)
        if new_fit[best_idx] < self.best_f:
            self.best_f = new_fit[best_idx]
            self.best_x = new_pop[best_idx].copy()

        self._step_count += 1
        return new_pop, new_fit

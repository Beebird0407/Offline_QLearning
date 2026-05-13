import numpy as np
from typing import Optional, Tuple, Callable, List, Dict


class Alg2Optimizer:
    K = 16

    Cr1_range = (0.0, 1.0)
    Xrmpx_range = (0, 1)
    eta_m_range = (1, 3)
    eta_c_range = (1, 3)
    Xrsbx_range = (0, 1)
    sigma_range = (0.0, 1.0)
    F13_range = (0.0, 1.0)
    F23_range = (0.0, 1.0)
    Cr3_range = (0.0, 1.0)
    F14_range = (0.0, 1.0)
    F24_range = (0.0, 1.0)
    Cr4_range = (0.0, 1.0)
    cm1_range = (1, 4)
    cm2_range = (1, 4)
    cm3_range = (1, 4)
    cm4_range = (1, 4)

    SG_POP_SIZES = [200, 100, 100, 100]

    def __init__(
        self,
        dim: int,
        bounds: np.ndarray,
        pop_size: int = 500,
        seed: int = 42,
        use_lpsr: bool = True,
        min_pop_size: int = 4
    ):
        self.dim = dim
        self.bounds = bounds
        self.pop_size = pop_size
        self.rng = np.random.RandomState(seed)
        self.sg_indices: Dict[int, List[int]] = {1: [], 2: [], 3: [], 4: []}
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
            # Shared GA parameters
            'Cr1': self.bin_to_param(bins[0], *self.Cr1_range, M),
            'Xrmpx': self.bin_to_discrete(bins[1], *self.Xrmpx_range, M),
            'eta_m': self.bin_to_discrete(bins[2], *self.eta_m_range, M),
            'eta_c': self.bin_to_discrete(bins[3], *self.eta_c_range, M),
            'Xrsbx': self.bin_to_discrete(bins[4], *self.Xrsbx_range, M),
            'sigma': self.bin_to_param(bins[5], *self.sigma_range, M),
            # Subgroup 3: DE/rand/2
            'F13': self.bin_to_param(bins[6], *self.F13_range, M),
            'F23': self.bin_to_param(bins[7], *self.F23_range, M),
            'Cr3': self.bin_to_param(bins[8], *self.Cr3_range, M),
            # Subgroup 4: DE/current-to-best/1
            'F14': self.bin_to_param(bins[9], *self.F14_range, M),
            'F24': self.bin_to_param(bins[10], *self.F24_range, M),
            'Cr4': self.bin_to_param(bins[11], *self.Cr4_range, M),
            # Communication parameters
            'cm1': self.bin_to_discrete(bins[12], *self.cm1_range, M),
            'cm2': self.bin_to_discrete(bins[13], *self.cm2_range, M),
            'cm3': self.bin_to_discrete(bins[14], *self.cm3_range, M),
            'cm4': self.bin_to_discrete(bins[15], *self.cm4_range, M),
        }

    def initialize(self) -> np.ndarray:
        """Initialize population using Halton low-discrepancy sequence."""
        halton = self._halton_sequence(self.pop_size, self.dim)
        pop = np.zeros((self.pop_size, self.dim))
        for d in range(self.dim):
            lo, hi = self.bounds[d]
            pop[:, d] = lo + halton[:, d] * (hi - lo)
        return pop

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
            idx = self.rng.randint(len(pop))
            return pop[idx].copy()
        probs = (max_f - fitness) / denom
        probs = probs ** pressure
        probs /= probs.sum()
        idx = self.rng.choice(len(pop), p=probs)
        return pop[idx].copy()

    def _partition_population(self, params: dict) -> Dict[int, int]:
        """Partition population into 4 subgroups using fixed ratios from paper."""
        # Fixed ratios: SG1=200, SG2=100, SG3=100, SG4=100 (total 500 from paper)
        # For smaller pop_size, scale proportionally
        total_fixed = 500
        ratios = [200/500, 100/500, 100/500, 100/500]

        sizes = [max(1, int(self.pop_size * r)) for r in ratios]
        sizes[3] = self.pop_size - sum(sizes[:3])

        indices = list(range(self.pop_size))
        self.rng.shuffle(indices)

        start = 0
        for sg, size in enumerate(sizes, 1):
            self.sg_indices[sg] = indices[start:start + size]
            start += size

        return dict(zip([1, 2, 3, 4], sizes))

    def _mpx_crossover(self, p1: np.ndarray, p2: np.ndarray, cr: float) -> Tuple[np.ndarray, np.ndarray]:
        mask = self.rng.rand(self.dim) < cr
        c1 = p1.copy()
        c2 = p2.copy()
        c1[mask] = p2[mask]
        c2[mask] = p1[mask]
        return c1, c2

    def _polynomial_mutation(self, x: np.ndarray, eta: float) -> np.ndarray:
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

    def _evolve_sg1(self, pop: np.ndarray, fitness: np.ndarray, params: dict, func: Callable) -> Tuple[np.ndarray, np.ndarray]:
        new_pop = pop.copy()
        new_fit = fitness.copy()

        # Use SG1 subpopulation for roulette selection
        sg1_idx = self.sg_indices[1]
        sg1_pop = pop[sg1_idx]
        sg1_fit = fitness[sg1_idx]

        for i in sg1_idx:
            p1 = self._roulette_selection(sg1_pop, sg1_fit)
            p2 = self._roulette_selection(sg1_pop, sg1_fit)
            c1, _ = self._mpx_crossover(p1, p2, params['Cr1'])
            c1 = self._polynomial_mutation(c1, float(params['eta_m']))
            f_c1 = func(c1)
            if f_c1 <= fitness[i]:
                new_pop[i] = c1
                new_fit[i] = f_c1
        return new_pop, new_fit

    def _sbx_crossover(self, p1: np.ndarray, p2: np.ndarray, eta: float, cr: float) -> Tuple[np.ndarray, np.ndarray]:
        child1 = p1.copy()
        child2 = p2.copy()
        for i in range(self.dim):
            if self.rng.rand() < cr:
                if abs(p1[i] - p2[i]) > 1e-10:
                    xl, xu = self.bounds[i]
                    y1, y2 = min(p1[i], p2[i]), max(p1[i], p2[i])
                    y1, y2 = max(xl, y1), min(xu, y2)
                    u_val = self.rng.rand()
                    if u_val <= 0.5:
                        beta = (2 * u_val) ** (1.0 / (eta + 1))
                    else:
                        beta = (1.0 / (2 * (1 - u_val))) ** (1.0 / (eta + 1))
                    child1[i] = 0.5 * ((y1 + y2) - beta * (y2 - y1))
                    child2[i] = 0.5 * ((y1 + y2) + beta * (y2 - y1))
        return child1, child2

    def _gaussian_mutation(self, x: np.ndarray, sigma: float) -> np.ndarray:
        mutant = x + sigma * self.rng.randn(self.dim)
        return np.clip(mutant, self.bounds[:, 0], self.bounds[:, 1])

    def _tournament_selection(self, pop: np.ndarray, fitness: np.ndarray, k: int = 2) -> np.ndarray:
        idxs = self.rng.choice(len(pop), k, replace=False)
        best_idx = idxs[np.argmin(fitness[idxs])]
        return pop[best_idx].copy()

    def _evolve_sg2(self, pop: np.ndarray, fitness: np.ndarray, params: dict, func: Callable) -> Tuple[np.ndarray, np.ndarray]:
        new_pop = pop.copy()
        new_fit = fitness.copy()
        cr_sg2 = (params['Cr1'] + params['Cr4']) / 2

        # Use SG2 subpopulation for tournament selection
        sg2_idx = self.sg_indices[2]
        sg2_pop = pop[sg2_idx]
        sg2_fit = fitness[sg2_idx]

        for i in sg2_idx:
            p1 = self._tournament_selection(sg2_pop, sg2_fit)
            p2 = self._tournament_selection(sg2_pop, sg2_fit)
            c1, c2 = self._sbx_crossover(p1, p2, float(params['eta_c']), cr_sg2)
            c1 = self._gaussian_mutation(c1, params['sigma'])
            c2 = self._gaussian_mutation(c2, params['sigma'])
            f_c1, f_c2 = func(c1), func(c2)
            if f_c1 <= f_c2:
                c, f_c = c1, f_c1
            else:
                c, f_c = c2, f_c2
            if f_c <= fitness[i]:
                new_pop[i] = c
                new_fit[i] = f_c
        return new_pop, new_fit

    def _de_rand2_mutation(self, pop: np.ndarray, F1: float, F2: float) -> np.ndarray:
        indices = list(range(len(pop)))
        self.rng.shuffle(indices)
        r1, r2, r3, r4, r5 = indices[:5]
        mutant = pop[r1].copy()
        mutant += F1 * (pop[r2] - pop[r3])
        mutant += F2 * (pop[r4] - pop[r5])
        return mutant

    def _exponential_crossover(self, target: np.ndarray, mutant: np.ndarray, Cr: float) -> np.ndarray:
        trial = target.copy()
        n = self.rng.randint(self.dim)
        for i in range(self.dim):
            idx = (n + i) % self.dim
            trial[idx] = mutant[idx]
            if self.rng.rand() >= Cr:
                break
        return trial

    def _evolve_sg3(self, pop: np.ndarray, fitness: np.ndarray, params: dict, func: Callable) -> Tuple[np.ndarray, np.ndarray]:
        new_pop = pop.copy()
        new_fit = fitness.copy()

        sg3_indices = self.sg_indices[3]
        if len(sg3_indices) < 5:
            # Not enough members for DE/rand/2, use global population
            sg3_pop = pop
        else:
            sg3_pop = pop[sg3_indices]

        for i in sg3_indices:
            mutant = self._de_rand2_mutation(sg3_pop, params['F13'], params['F23'])
            mutant = np.clip(mutant, self.bounds[:, 0], self.bounds[:, 1])
            trial = self._exponential_crossover(pop[i], mutant, params['Cr3'])
            trial = np.clip(trial, self.bounds[:, 0], self.bounds[:, 1])
            f_trial = func(trial)
            if f_trial <= fitness[i]:
                new_pop[i] = trial
                new_fit[i] = f_trial
        return new_pop, new_fit

    def _de_current_to_best_mutation(self, target: np.ndarray, pop: np.ndarray, fitness: np.ndarray, F1: float, F2: float) -> np.ndarray:
        best_idx = np.argmin(fitness)
        indices = [j for j in range(len(pop)) if j != best_idx]
        self.rng.shuffle(indices)
        r1, r2 = indices[:2]
        mutant = target.copy()
        mutant += F1 * (pop[best_idx] - target)
        mutant += F2 * (pop[r1] - pop[r2])
        return mutant

    def _binomial_crossover(self, target: np.ndarray, mutant: np.ndarray, Cr: float) -> np.ndarray:
        mask = self.rng.rand(self.dim) < Cr
        if not mask.any():
            mask[self.rng.randint(self.dim)] = True
        trial = target.copy()
        trial[mask] = mutant[mask]
        return np.clip(trial, self.bounds[:, 0], self.bounds[:, 1])

    def _evolve_sg4(self, pop: np.ndarray, fitness: np.ndarray, params: dict, func: Callable) -> Tuple[np.ndarray, np.ndarray]:
        new_pop = pop.copy()
        new_fit = fitness.copy()

        # Use SG4 subpopulation for DE/current-to-best/1
        sg4_idx = self.sg_indices[4]
        # Fall back to full population when subgroup is too small (<3 members)
        if len(sg4_idx) >= 3:
            sg4_pop = pop[sg4_idx]
            sg4_fit = fitness[sg4_idx]
        else:
            sg4_pop = pop
            sg4_fit = fitness

        for i in sg4_idx:
            mutant = self._de_current_to_best_mutation(pop[i], sg4_pop, sg4_fit, params['F14'], params['F24'])
            mutant = np.clip(mutant, self.bounds[:, 0], self.bounds[:, 1])
            trial = self._binomial_crossover(pop[i], mutant, params['Cr4'])
            f_trial = func(trial)
            if f_trial <= fitness[i]:
                new_pop[i] = trial
                new_fit[i] = f_trial
        return new_pop, new_fit

    def _share_between_subgroups(self, pop: np.ndarray, fitness: np.ndarray, params: dict) -> np.ndarray:
        new_pop = pop.copy()
        cms = [params['cm1'], params['cm2'], params['cm3'], params['cm4']]

        best_per_sg = {}
        for sg in [1, 2, 3, 4]:
            if self.sg_indices[sg]:
                best_idx = self.sg_indices[sg][np.argmin(fitness[self.sg_indices[sg]])]
                best_per_sg[sg] = best_idx

        for sg_dest in [1, 2, 3, 4]:
            if not self.sg_indices[sg_dest]:
                continue
            cm_target = cms[sg_dest - 1]
            if cm_target == sg_dest:
                continue
            if cm_target in best_per_sg:
                src_best = pop[best_per_sg[cm_target]]
                worst_idx = self.sg_indices[sg_dest][np.argmax(fitness[self.sg_indices[sg_dest]])]
                new_pop[worst_idx] = src_best.copy()

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

        new_pop, new_fit = pop.copy(), fitness.copy()

        new_pop, new_fit = self._evolve_sg1(new_pop, new_fit, params, func)
        new_pop, new_fit = self._evolve_sg2(new_pop, new_fit, params, func)
        new_pop, new_fit = self._evolve_sg3(new_pop, new_fit, params, func)
        new_pop, new_fit = self._evolve_sg4(new_pop, new_fit, params, func)

        new_pop = self._share_between_subgroups(new_pop, new_fit, params)

        best_idx = np.argmin(new_fit)
        if new_fit[best_idx] < self.best_f:
            self.best_f = new_fit[best_idx]
            self.best_x = new_pop[best_idx].copy()

        return new_pop, new_fit
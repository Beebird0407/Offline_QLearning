import os
import sys
import json
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

try:
    import nevergrad
    NEVERGRAD_AVAILABLE = True
except ImportError:
    NEVERGRAD_AVAILABLE = False

from model.qmamba import QMamba
from algorithms.alg0 import Alg0Optimizer
from algorithms.alg1 import Alg1Optimizer
from algorithms.alg2 import Alg2Optimizer
from env.state import StateExtractor
from data.bbob_suite import BBOBSuite


# ──────────────────────────── Data classes ────────────────────────────

@dataclass
class EvalResult:
    name: str
    test_type: str  # 'id' or 'ood'
    mean_performance: float  # Accumulated performance improvement (0~1)
    std_performance: float
    final_best: float  # Mean final best fitness
    total_time: float
    convergence_curve: List[float]
    per_function_performance: Optional[Dict[str, float]] = None
    per_function_bests: Optional[Dict[str, float]] = None


# ──────────────────────────── Nevergrad wrapper ────────────────────────────

class NevergradBBOBWrapper:
    """Nevergrad BBOB wrapper with F{id}_{instance} naming."""

    _NG_NAMES = {
        1: 'sphere',           2: 'ellipsoid',        3: 'rastrigin',
        4: 'bucherastrigin',   5: 'linearslope',      6: 'attractivesector',
        7: 'stepellipsoid',    8: 'cigar',            9: 'discus',
        10: 'bentcigar',       11: 'sharpridge',      12: 'sumppowers',
        13: 'schwefel',        14: 'gallagher',       15: 'gallagher',
        16: 'rosenbrock',      17: 'rosenbrock',      18: 'griewankrosenbrock',
        19: 'schaffers',       20: 'schaffers',       21: 'lunacek',
        22: 'katsuura',        23: 'deceptivemultimodal', 24: 'hm',
    }
    TRAIN_IDS = [1, 2, 4, 5, 6, 7, 10, 11, 12, 15, 16, 17, 19, 22, 23, 24]
    TEST_IDS  = [3, 8, 9, 13, 14, 18, 20, 21]

    def __init__(self, dim=5, n_train_instances=1, n_test_instances=1, **_kw):
        from nevergrad.functions import ArtificialFunction
        self.dim = dim
        self._functions = []
        for fid in self.TRAIN_IDS + self.TEST_IDS:
            ng_name = self._NG_NAMES.get(fid, 'sphere')
            n_inst = n_test_instances if fid in self.TEST_IDS else n_train_instances
            for inst in range(n_inst):
                func = ArtificialFunction(
                    name=ng_name, block_dimension=dim,
                    num_blocks=1, noise_level=0,
                    rotation=False, translation_factor=1.0,
                )
                self._functions.append((f"F{fid}_{inst}", func))

    def get_test_functions(self):
        return [(n, f) for n, f in self._functions
                if int(n.split('_')[0][1:]) in self.TEST_IDS]

    def get_bounds(self, fid=None):
        if isinstance(fid, str):
            fid = int(fid.lstrip('Ff').split('_')[0])
        if fid == 20:
            return np.array([[-500, 500]] * self.dim)
        return np.array([[-5, 5]] * self.dim)

    def __len__(self):
        return 24


# ──────────────────────────── OOD: Neural evolution environment ────────────────────────────

class NeuralEvolutionEnv:
    """
    Simulates a neural evolution task for OOD evaluation.

    Uses a high-dimensional, non-separable landscape that mimics
    the characteristics of evolving neural network weights:
    - High effective dimensionality
    - Non-separable interactions between parameters
    - Deceptive local optima

    When Mujoco is available, can optionally use real policy evaluation.
    """

    def __init__(
        self,
        dim: int = 50,
        pop_size: int = 10,
        seed: int = 42,
        env_name: str = 'Hopper-v4'
    ):
        self.dim = dim
        self.pop_size = pop_size
        self.seed = seed
        self.env_name = env_name
        self.rng = np.random.RandomState(seed)

        # Try to load Mujoco environment
        self._use_mujoco = False
        self._env = None
        self._policy_shape = None
        try:
            import gymnasium as gym
            self._env = gym.make(env_name)
            obs_dim = self._env.observation_space.shape[0]
            act_dim = self._env.action_space.shape[0]
            # Policy: linear layer weights + biases
            self._policy_shape = (obs_dim, act_dim)
            self._use_mujoco = True
            self._env.close()
            self._env = None
        except Exception:
            pass

    def _make_landscape_fn(self, seed_offset: int = 0):
        """Create a synthetic landscape that mimics neural evolution."""
        rng = np.random.RandomState(self.seed + seed_offset)
        n = self.dim

        # Random rotation matrix for non-separability
        H = rng.randn(n, n)
        Q, _ = np.linalg.qr(H)

        # Multi-modal with deceptive structure
        n_peaks = 5
        centers = rng.uniform(-3, 3, (n_peaks, n))
        widths = rng.uniform(0.5, 3.0, n_peaks)
        heights = rng.uniform(0.5, 2.0, n_peaks)

        def fn(x):
            z = Q.T @ x
            # Base: rotated ellipsoid
            coeffs = np.array([100 ** (i / (n - 1)) for i in range(n)])
            base = np.sum(coeffs * z ** 2) / n
            # Multi-modal component
            modal = 0.0
            for p in range(n_peaks):
                d = np.sqrt(np.sum(((z - centers[p]) / widths[p]) ** 2))
                modal += heights[p] * np.exp(-0.5 * d ** 2)
            return base - modal + 10.0

        return fn

    def get_test_problems(self, n_problems: int = 10) -> List[Tuple[str, callable]]:
        """Generate OOD test problems."""
        problems = []
        for i in range(n_problems):
            fn = self._make_landscape_fn(seed_offset=i * 100)
            problems.append((f"NeuralEvo_{i}", fn))
        return problems

    def get_bounds(self) -> np.ndarray:
        if self._use_mujoco:
            return np.array([[-1.0, 1.0]] * self.dim)
        return np.array([[-5.0, 5.0]] * self.dim)


# ──────────────────────────── Environment wrapper ────────────────────────────

class MambaEnvWrapper:
    """Wraps optimizer + problem into a step-based environment."""

    def __init__(self, optimizer, problem, dim, bounds, pop_size=20, seed=None, T=500):
        self.opt = optimizer
        self.problem = problem
        self.dim = dim
        self.bounds = bounds
        self.pop_size = pop_size
        self.rng = np.random.RandomState(seed)
        self.state_extractor = StateExtractor()
        self.K = getattr(optimizer, 'K', 3)
        self.T = T

        self.pop = self.opt.initialize()
        self.fitness = np.array([problem(x) for x in self.pop])
        self.best_fitness = float(self.fitness.min())
        self.t = 0

    def reset(self):
        self.pop = self.opt.initialize()
        self.fitness = np.array([self.problem(x) for x in self.pop])
        self.best_fitness = float(self.fitness.min())
        self.t = 0
        return self.state_extractor.compute(self.pop, self.fitness, 0, self.T)

    def step(self, actions_dict):
        action_names = self._get_action_names()
        action = tuple(actions_dict.get(name, 0) for name in action_names)

        self.pop, self.fitness = self.opt.step(
            self.pop, self.fitness, action, self.problem, self.t, self.T
        )

        curr_best = float(self.fitness.min())
        if curr_best < self.best_fitness:
            self.best_fitness = curr_best
        self.t += 1

        state = self.state_extractor.compute(self.pop, self.fitness, self.t, self.T)
        return state, curr_best, self.best_fitness, self.t >= self.T

    def _get_action_names(self):
        if self.K == 3:
            return ['F1', 'F2', 'Cr']
        elif self.K == 10:
            return ['Cr1', 'Xrmpx', 'sigma', 'bc1', 'cm1', 'F1', 'F2', 'Cr2', 'bc2', 'cm2']
        elif self.K == 16:
            return ['Cr1', 'Xrmpx', 'eta_m', 'eta_c', 'Xrsbx', 'sigma',
                    'F13', 'F23', 'Cr3', 'F14', 'F24', 'Cr4', 'cm1', 'cm2', 'cm3', 'cm4']
        return [f'p{i}' for i in range(self.K)]


# ──────────────────────────── Agents ────────────────────────────

class QMAgent:
    """Agent wrapping a trained QMamba model."""

    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.K = model.K
        self.M = model.M

    @classmethod
    def from_checkpoint(cls, path, device='cpu'):
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        cfg = ckpt.get('config', {})
        sd = {k.replace('module.', ''): v for k, v in ckpt['model_state_dict'].items()}

        # Detect backend and ensure architecture matches the checkpoint
        uses_mamba = any('in_proj' in k or 'A_log' in k for k in sd.keys())
        if uses_mamba:
            from model.qmamba import _MAMBA_AVAILABLE
            if not _MAMBA_AVAILABLE:
                raise RuntimeError(
                    "Checkpoint was saved with Mamba SSM backend, "
                    "but mamba-ssm is not installed in this environment.\n"
                    "Install it:  pip install mamba-ssm"
                )
            if device == 'cpu':
                if torch.cuda.is_available():
                    device = 'cuda'
                else:
                    raise RuntimeError(
                        "Checkpoint uses Mamba SSM backend which requires CUDA, "
                        "but no GPU is available."
                    )

        model = QMamba(
            state_dim=cfg.get('state_dim', 9),
            K=cfg.get('K', 3),
            M=cfg.get('M', 16),
            d_model=cfg.get('d_model', 14),
            d_state=cfg.get('d_state', 32),
            n_layers=cfg.get('n_layers', 1),
            num_hidden_mlp=cfg.get('num_hidden_mlp', 32),
            force_cpu=False,
        )
        model.load_state_dict(sd)
        return cls(model, device=device)

    def act(self, state_np):
        with torch.no_grad():
            s = torch.tensor(state_np, dtype=torch.float32, device=self.device)
            if s.dim() == 1:
                s = s.unsqueeze(0)
            acts, _, _ = self.model.act(s, deterministic=True)
            return acts.cpu().numpy()[0]


# ──────────────────────────── Core evaluation ────────────────────────────

def _get_action_names(K):
    if K == 3:
        return ['F1', 'F2', 'Cr']
    elif K == 10:
        return ['Cr1', 'Xrmpx', 'sigma', 'bc1', 'cm1', 'F1', 'F2', 'Cr2', 'bc2', 'cm2']
    elif K == 16:
        return ['Cr1', 'Xrmpx', 'eta_m', 'eta_c', 'Xrsbx', 'sigma',
                'F13', 'F23', 'Cr3', 'F14', 'F24', 'Cr4', 'cm1', 'cm2', 'cm3', 'cm4']
    return [f'p{i}' for i in range(K)]


def _compute_performance(fitness_history: List[float], f_star: float) -> float:
    """Compute accumulated performance improvement.

    Perf = Σ_t r_t
    r_t = (f_{*,t-1} - f_{*,t}) / (f_{*,0} - f*)

    Values closer to 1 mean the solution is closer to global optimum.
    """
    if len(fitness_history) < 2:
        return 0.0

    f0 = fitness_history[0]
    denom = f0 - f_star

    if abs(denom) < 1e-10:
        return 1.0 if fitness_history[-1] <= f_star + 1e-10 else 0.0

    perf = 0.0
    for t in range(1, len(fitness_history)):
        improvement = fitness_history[t - 1] - fitness_history[t]
        if improvement > 0:
            perf += improvement / denom

    return float(np.clip(perf, 0.0, 1.0))


def evaluate_single(
    agent,
    problem_fn,
    optimizer_class,
    dim: int,
    bounds: np.ndarray,
    pop_size: int,
    T: int,
    n_runs: int,
    f_star: float,
    seed_base: int = 42,
) -> Tuple[float, float, List[float]]:
    """Evaluate agent on a single problem.

    Returns: (mean_performance, mean_final_best, mean_convergence)
    """
    action_names = _get_action_names(getattr(optimizer_class, 'K', 3))
    K = len(action_names)
    run_perfs = []
    run_finals = []
    run_convergences = []

    for run in range(n_runs):
        run_seed = seed_base + run * 1000
        opt = optimizer_class(dim=dim, bounds=bounds, pop_size=pop_size, seed=run_seed)
        env = MambaEnvWrapper(opt, problem_fn, dim, bounds, pop_size, run_seed, T=T)

        state = env.reset()
        best_so_far = float('inf')
        convergence = []

        for t in range(T):
            action_bins = agent.act(state)
            actions_dict = {action_names[i]: int(action_bins[i]) for i in range(K)}
            state, curr_best, _, done = env.step(actions_dict)
            if curr_best < best_so_far:
                best_so_far = curr_best
            convergence.append(best_so_far)
            if done:
                break

        run_perfs.append(_compute_performance(convergence, f_star))
        run_finals.append(best_so_far)
        run_convergences.append(convergence)

    # Average convergence curves (pad to same length)
    max_len = max(len(c) for c in run_convergences)
    padded = np.full((n_runs, max_len), np.nan)
    for i, c in enumerate(run_convergences):
        padded[i, :len(c)] = c
    mean_conv = np.nanmean(padded, axis=0).tolist()

    return float(np.mean(run_perfs)), float(np.mean(run_finals)), mean_conv


# ──────────────────────────── ID Test ────────────────────────────

def run_id_test(
    agents: Dict[str, object],
    optimizer_class,
    bbob_suite,
    n_runs: int = 19,
    pop_size: int = 20,
    T: int = 500,
    verbose: bool = True,
) -> Dict[str, EvalResult]:
    """In-distribution test on BBOB test functions.

    Uses known f* from function.optimum (0 for most BBOB functions).
    """
    test_functions = bbob_suite.get_test_functions()
    K = getattr(optimizer_class, 'K', 3)
    dim = bbob_suite.dim
    results = {}

    if verbose:
        print(f"\n{'='*60}")
        print(f"  ID Test: BBOB Test Functions")
        print(f"  Functions: {len(test_functions)}, Runs: {n_runs}")
        print(f"  pop_size={pop_size}, T={T}, K={K}")
        print(f"{'='*60}")

    for agent_name, agent in agents.items():
        if verbose:
            print(f"\n  Evaluating: {agent_name}")

        per_fn_perf = {}
        per_fn_best = {}
        all_convergences = []
        total_time = 0.0

        for fn_name, fn in test_functions:
            # Get f* from function definition
            f_star = getattr(fn, 'optimum', 0.0)
            bounds = bbob_suite.get_bounds(fn_name.split('_')[0])

            if verbose:
                print(f"    {fn_name} (f*={f_star:.2f})...", end=" ", flush=True)

            start = time.time()
            perf, final_best, conv = evaluate_single(
                agent, fn, optimizer_class, dim, bounds,
                pop_size, T, n_runs, f_star
            )
            elapsed = time.time() - start
            total_time += elapsed

            per_fn_perf[fn_name] = perf
            per_fn_best[fn_name] = final_best
            all_convergences.append(conv)

            if verbose:
                print(f"Perf={perf:.4f}, Best={final_best:.4f}, Time={elapsed:.1f}s")

        # Average across functions
        mean_perf = float(np.mean(list(per_fn_perf.values())))
        std_perf = float(np.std(list(per_fn_perf.values())))
        mean_final = float(np.mean(list(per_fn_best.values())))

        # Average convergence curve
        max_len = max(len(c) for c in all_convergences) if all_convergences else 0
        if max_len > 0:
            padded = np.full((len(all_convergences), max_len), np.nan)
            for i, c in enumerate(all_convergences):
                padded[i, :len(c)] = c
            mean_conv = np.nanmean(padded, axis=0).tolist()
        else:
            mean_conv = []

        results[agent_name] = EvalResult(
            name=agent_name,
            test_type='id',
            mean_performance=mean_perf,
            std_performance=std_perf,
            final_best=mean_final,
            total_time=total_time,
            convergence_curve=mean_conv,
            per_function_performance=per_fn_perf,
            per_function_bests=per_fn_best,
        )

        if verbose:
            print(f"  {agent_name} ID Performance: {mean_perf:.4f} ± {std_perf:.4f}")

    return results


# ──────────────────────────── OOD Test ────────────────────────────

def run_ood_test(
    agents: Dict[str, object],
    optimizer_class,
    n_problems: int = 10,
    dim: int = 50,
    n_runs: int = 5,
    pop_size: int = 10,
    T: int = 50,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, EvalResult]:
    """Out-of-distribution test on neural evolution tasks.

    Small population (10) and few steps (50) to test zero-shot transfer.
    Uses proxy f* = best observed across all agents and runs.
    """
    K = getattr(optimizer_class, 'K', 3)
    ood_env = NeuralEvolutionEnv(dim=dim, pop_size=pop_size, seed=seed)
    problems = ood_env.get_test_problems(n_problems)
    bounds = ood_env.get_bounds()

    if verbose:
        print(f"\n{'='*60}")
        print(f"  OOD Test: Neural Evolution Tasks")
        print(f"  Problems: {n_problems}, dim={dim}, Runs: {n_runs}")
        print(f"  pop_size={pop_size}, T={T} (zero-shot transfer)")
        print(f"{'='*60}")

    results = {}

    for agent_name, agent in agents.items():
        if verbose:
            print(f"\n  Evaluating: {agent_name}")

        per_fn_perf = {}
        per_fn_best = {}
        all_convergences = []
        total_time = 0.0

        for fn_name, fn in problems:
            if verbose:
                print(f"    {fn_name}...", end=" ", flush=True)

            start = time.time()
            perf, final_best, conv = evaluate_single(
                agent, fn, optimizer_class, dim, bounds,
                pop_size, T, n_runs, f_star=0.0, seed_base=seed
            )
            elapsed = time.time() - start
            total_time += elapsed

            per_fn_perf[fn_name] = perf
            per_fn_best[fn_name] = final_best
            all_convergences.append(conv)

            if verbose:
                print(f"Perf={perf:.4f}, Best={final_best:.4f}")

        mean_perf = float(np.mean(list(per_fn_perf.values())))
        std_perf = float(np.std(list(per_fn_perf.values())))
        mean_final = float(np.mean(list(per_fn_best.values())))

        max_len = max(len(c) for c in all_convergences) if all_convergences else 0
        if max_len > 0:
            padded = np.full((len(all_convergences), max_len), np.nan)
            for i, c in enumerate(all_convergences):
                padded[i, :len(c)] = c
            mean_conv = np.nanmean(padded, axis=0).tolist()
        else:
            mean_conv = []

        results[agent_name] = EvalResult(
            name=agent_name,
            test_type='ood',
            mean_performance=mean_perf,
            std_performance=std_perf,
            final_best=mean_final,
            total_time=total_time,
            convergence_curve=mean_conv,
            per_function_performance=per_fn_perf,
            per_function_bests=per_fn_best,
        )

        if verbose:
            print(f"  {agent_name} OOD Performance: {mean_perf:.4f} ± {std_perf:.4f}")

    return results


# ──────────────────────────── Display ────────────────────────────

def _print_summary(results: Dict[str, EvalResult], title: str):
    """Print summary table."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    print(f"{'Agent':<20} {'Performance':<14} {'Std':<10} {'Final Best':<14} {'Time (s)':<10}")
    print("-" * 70)
    for name, r in sorted(results.items(), key=lambda x: -x[1].mean_performance):
        print(f"{name:<20} {r.mean_performance:<14.4f} {r.std_performance:<10.4f} "
              f"{r.final_best:<14.6f} {r.total_time:<10.1f}")
    print(f"{'='*70}")


def _print_per_function(results: Dict[str, EvalResult]):
    """Print per-function comparison table."""
    valid = {k: v for k, v in results.items() if v.per_function_performance}
    if len(valid) < 2:
        return

    all_fns = set()
    for r in valid.values():
        all_fns.update(r.per_function_performance.keys())
    all_fns = sorted(all_fns)

    print(f"\n  Per-function Performance:")
    header = f"  {'Function':<18}"
    for name in valid:
        header += f"{name:<16}"
    print(header)
    print(f"  {'-' * (18 + 16 * len(valid))}")

    wins = {name: 0 for name in valid}
    for fn in all_fns:
        row = f"  {fn:<18}"
        best_val, best_name = float('-inf'), None
        for name, r in valid.items():
            p = r.per_function_performance.get(fn, 0.0)
            row += f"{p:<16.4f}"
            if p > best_val:
                best_val, best_name = p, name
        if best_name:
            wins[best_name] += 1
        print(row)

    print(f"  {'-' * (18 + 16 * len(valid))}")
    print(f"\n  Win counts:")
    for name, w in sorted(wins.items(), key=lambda x: -x[1]):
        print(f"    {name}: {w}/{len(all_fns)}")


# ──────────────────────────── Main entry ────────────────────────────

def run_evaluation(
    trained_model_dir: str = './Trained_model',
    mode: str = 'both',  # 'id', 'ood', or 'both'
    n_runs_id: int = 19,
    n_runs_ood: int = 5,
    pop_size_id: Optional[int] = None,
    pop_size_ood: int = 10,
    T_id: int = 500,
    T_ood: int = 50,
    dim_ood: int = 50,
    n_ood_problems: int = 10,
    device: str = 'cpu',
    verbose: bool = True,
):
    """Run ID and/or OOD evaluation."""

    # Determine algorithm (with default pop sizes)
    alg_dir = os.path.basename(trained_model_dir)
    if 'Alg2' in alg_dir:
        optimizer_class, K, default_pop = Alg2Optimizer, 16, 500
    elif 'Alg1' in alg_dir:
        optimizer_class, K, default_pop = Alg1Optimizer, 10, 250
    else:
        optimizer_class, K, default_pop = Alg0Optimizer, 3, 100

    if pop_size_id is None:
        pop_size_id = default_pop

    # Build agents
    agents = {}

    # Trained model
    ckpt_path = os.path.join(trained_model_dir, 'best.pth')
    if os.path.exists(ckpt_path):
        try:
            trained = QMAgent.from_checkpoint(ckpt_path, device=device)
            agents['Trained'] = trained
            if verbose:
                print(f"  Loaded trained model from {ckpt_path}")
        except Exception as e:
            print(f"  [Warning] Failed to load model: {e}")

    if verbose:
        print(f"\n  Agents: {list(agents.keys())}")
        print(f"  Algorithm: {optimizer_class.__name__} (K={K})")

    all_results = {}

    # ID Test — always use project's BBOB suite (full 24-function support)
    if mode in ('id', 'both'):
        bbob_suite = BBOBSuite(dim=5, n_test_instances=1)

        id_results = run_id_test(
            agents, optimizer_class, bbob_suite,
            n_runs=n_runs_id, pop_size=pop_size_id, T=T_id, verbose=verbose,
        )
        _print_summary(id_results, "ID Test: BBOB Test Functions")
        _print_per_function(id_results)
        all_results['id'] = {k: asdict(v) for k, v in id_results.items()}

    # OOD Test
    if mode in ('ood', 'both'):
        ood_results = run_ood_test(
            agents, optimizer_class,
            n_problems=n_ood_problems, dim=dim_ood,
            n_runs=n_runs_ood, pop_size=pop_size_ood, T=T_ood,
            verbose=verbose,
        )
        _print_summary(ood_results, "OOD Test: Neural Evolution Tasks")
        _print_per_function(ood_results)
        all_results['ood'] = {k: asdict(v) for k, v in ood_results.items()}

    # Save
    save_path = os.path.join(trained_model_dir, 'evaluation_results.json')
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    if verbose:
        print(f"\nResults saved to: {save_path}")

    return all_results


# ──────────────────────────── CLI ────────────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Q-Mamba Evaluation')
    parser.add_argument('--model', type=str, default=None,
                        help='Model directory (e.g., ./Trained_model/Alg0_ACQL)')
    parser.add_argument('--mode', type=str, default='both',
                        choices=['id', 'ood', 'both'],
                        help='Test mode: id, ood, or both')
    parser.add_argument('--n_runs', type=int, default=19,
                        help='Runs per function (ID test)')
    parser.add_argument('--n_runs_ood', type=int, default=5,
                        help='Runs per problem (OOD test)')
    parser.add_argument('--pop_size', type=int, default=None,
                        help='Population size (ID test, auto-detected from algorithm)')
    parser.add_argument('--pop_size_ood', type=int, default=10,
                        help='Population size (OOD test)')
    parser.add_argument('--T', type=int, default=500,
                        help='Optimization steps (ID test)')
    parser.add_argument('--T_ood', type=int, default=50,
                        help='Optimization steps (OOD test)')
    parser.add_argument('--dim_ood', type=int, default=50,
                        help='Problem dimension (OOD test)')
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()

    if args.model:
        run_evaluation(
            trained_model_dir=args.model, mode=args.mode,
            n_runs_id=args.n_runs, n_runs_ood=args.n_runs_ood,
            pop_size_id=args.pop_size, pop_size_ood=args.pop_size_ood,
            T_id=args.T, T_ood=args.T_ood, dim_ood=args.dim_ood,
            device=args.device,
        )
    else:
        print("Usage: python evaluate_comparison.py --model <path> [--mode id|ood|both]")
        print("Example: python evaluate_comparison.py --model ./Trained_model/Alg0_CQL --mode both")

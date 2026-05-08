"""
Random / Exploit baseline test for BBOB functions.
Matches evaluate_comparison.py methodology.
"""
import argparse, json, time, sys, os
import numpy as np
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from algorithms.alg0 import Alg0Optimizer
from algorithms.alg1 import Alg1Optimizer
from algorithms.alg2 import Alg2Optimizer
from env.state import StateExtractor
from data.bbob_suite import BBOBSuite


def _get_action_names(K):
    if K == 3:   return ['F1', 'F2', 'Cr']
    elif K == 10: return ['Cr1', 'Xrmpx', 'sigma', 'bc1', 'cm1', 'F1', 'F2', 'Cr2', 'bc2', 'cm2']
    elif K == 16: return ['Cr1', 'Xrmpx', 'eta_m', 'eta_c', 'Xrsbx', 'sigma',
                          'F13', 'F23', 'Cr3', 'F14', 'F24', 'Cr4', 'cm1', 'cm2', 'cm3', 'cm4']
    return [f'p{i}' for i in range(K)]


def _compute_performance(fitness_history: List[float], f_star: float) -> float:
    if len(fitness_history) < 2:
        return 0.0
    f0 = fitness_history[0]
    denom = f0 - f_star
    if abs(denom) < 1e-10:
        return 1.0 if fitness_history[-1] <= f_star + 1e-10 else 0.0
    perf = 0.0
    for t in range(1, len(fitness_history)):
        imp = fitness_history[t - 1] - fitness_history[t]
        if imp > 0:
            perf += imp / denom
    return float(np.clip(perf, 0.0, 1.0))


class RandomAgent:
    def __init__(self, K, M=16, seed=42):
        self.K = K; self.M = M
        self.rng = np.random.RandomState(seed)
    def act(self, state):
        return self.rng.randint(0, self.M, size=self.K)


class ExploitAgent:
    def __init__(self, K, M=16, seed=42):
        self.K = K; self.M = M
        self.rng = np.random.RandomState(seed)
    def act(self, state):
        return np.full(self.K, self.M // 2, dtype=np.int64)


def evaluate_agent(agent, fn, optimizer_class, dim, bounds, pop_size, T, n_runs, f_star, seed_base=42):
    K = len(_get_action_names(getattr(optimizer_class, 'K', 3)))
    action_names = _get_action_names(K)
    run_perfs, run_finals = [], []
    extractor_cls = StateExtractor
    for run in range(n_runs):
        rs = seed_base + run * 1000
        opt = optimizer_class(dim=dim, bounds=bounds, pop_size=pop_size, seed=rs)
        pop = opt.initialize()
        fitness = np.array([fn(x) for x in pop])
        se = extractor_cls()
        best = float(fitness.min())
        conv = [best]
        for t in range(T):
            state = se.compute(pop, fitness, t, T)
            action_bins = agent.act(state)
            action_bins = np.clip(action_bins, 0, 15)
            actions_dict = {action_names[i]: int(action_bins[i]) for i in range(K)}
            pop, fitness = opt.step(pop, fitness, tuple(actions_dict.values()), fn, t, T)
            cb = float(fitness.min())
            if cb < best: best = cb
            conv.append(best)
        run_perfs.append(_compute_performance(conv, f_star))
        run_finals.append(best)
    return float(np.mean(run_perfs)), float(np.mean(run_finals))


def main():
    p = argparse.ArgumentParser(description='Random/Exploit baseline BBOB test')
    p.add_argument('--algorithm', default='Alg0', choices=['Alg0', 'Alg1', 'Alg2'])
    p.add_argument('--strategy', default='both', choices=['random', 'exploit', 'both'])
    p.add_argument('--pop_size', type=int, default=None, help='Override default pop')
    p.add_argument('--K', type=int, default=None, help='Override default K')
    p.add_argument('--T', type=int, default=500)
    p.add_argument('--n_runs', type=int, default=19)
    p.add_argument('--output', type=str, default=None)
    args = p.parse_args()

    opt_map = {'Alg0': (Alg0Optimizer, 100, 3), 'Alg1': (Alg1Optimizer, 250, 10), 'Alg2': (Alg2Optimizer, 500, 16)}
    opt_cls, def_pop, def_K = opt_map[args.algorithm]
    pop_size = args.pop_size or def_pop
    K = args.K or def_K

    bbob = BBOBSuite(dim=5, n_test_instances=1)
    test_fns = bbob.get_test_functions()

    strategies = ['random', 'exploit'] if args.strategy == 'both' else [args.strategy]
    results = {}

    for strat in strategies:
        agent = RandomAgent(K) if strat == 'random' else ExploitAgent(K)
        print(f"\n{'='*60}")
        print(f"  {strat.upper()}  |  {args.algorithm}  K={K}  pop={pop_size}  T={args.T}  runs={args.n_runs}")
        print(f"{'='*60}")

        per_fn = {}
        for fn_name, fn in test_fns:
            f_star = getattr(fn, 'optimum', 0.0)
            bounds = bbob.get_bounds(fn_name.split('_')[0])
            perf, final = evaluate_agent(agent, fn, opt_cls, 5, bounds, pop_size, args.T, args.n_runs, f_star)
            per_fn[fn_name] = (perf, final)
            print(f"  {fn_name:<8}  perf={perf:.4f}  final_best={final:.6f}")

        means = [v[0] for v in per_fn.values()]
        results[strat] = {'mean_perf': float(np.mean(means)), 'std_perf': float(np.std(means)), 'per_function': {k: v[0] for k, v in per_fn.items()}}
        print(f"  {'─'*50}")
        print(f"  MEAN: {results[strat]['mean_perf']:.4f} ± {results[strat]['std_perf']:.4f}")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved → {args.output}")


if __name__ == '__main__':
    main()

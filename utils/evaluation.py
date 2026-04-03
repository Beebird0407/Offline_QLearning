"""Evaluation module for Q-Mamba and baselines."""

import numpy as np
import time
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass
import json

from data.bbob_suite import BBOBSuite
from model.agent import QMAgent
from algorithms.alg0 import Alg0Optimizer


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    name: str
    performance: float  # Cumulative performance
    mean_fitness: float
    std_fitness: float
    training_time: float
    inference_time: float
    n_evaluations: int
    convergence_curve: List[float]


class Evaluator:
    """
    Comprehensive evaluation for Q-Mamba and baselines.
    """

    def __init__(
        self,
        agent: QMAgent,
        pop_size: int = 20,
        T: int = 500,
        n_runs: int = 19,
        seed: int = 42
    ):
        """
        Args:
            agent: Q-Mamba agent
            pop_size: Population size
            T: Number of generations
            n_runs: Number of independent runs per test
            seed: Base random seed
        """
        self.agent = agent
        self.pop_size = pop_size
        self.T = T
        self.n_runs = n_runs
        self.seed = seed

    def _compute_reward(
        self,
        prev_fitness: float,
        curr_fitness: float,
        prev_best: float,
        curr_best: float,
        optimum: float = 0.0
    ) -> float:
        """
        Compute reward: (f^{t-1} - f^t) / (f^{t-1} - f^*)
        """
        if prev_best <= curr_best:
            return 0.0

        denom = prev_best - optimum if prev_best != optimum else 1.0
        return float(np.clip((prev_best - curr_best) / denom, -1.0, 1.0))

    def evaluate_on_problem(
        self,
        problem: Callable,
        dim: int,
        bounds: np.ndarray,
        optimum: float = 0.0,
        seed: Optional[int] = None
    ) -> Tuple[float, float, float, List[float], float, float]:
        """
        Evaluate agent on a single problem.

        Returns:
            (best_fitness, mean_fitness, std_fitness, convergence_curve, train_time, inf_time)
        """
        rng = np.random.RandomState(seed or self.seed)

        all_best_fitness = []
        all_convergence = []
        total_inf_time = 0.0

        for run in range(self.n_runs):
            run_seed = rng.randint(1e6)

            # Track optimization
            fitness_history = []
            start_time = time.time()

            # Initialize optimizer
            from env.state import StateExtractor
            state_extractor = StateExtractor()

            K = self.agent.model.K
            M = self.agent.model.M
            param_ranges = [
                Alg0Optimizer.F1_range,
                Alg0Optimizer.F2_range,
                Alg0Optimizer.Cr_range
            ][:K]

            from env.action import ActionSpace
            action_space = ActionSpace(K, M, param_ranges)

            opt = Alg0Optimizer(
                dim=dim,
                bounds=bounds,
                pop_size=self.pop_size,
                seed=run_seed
            )

            pop = opt.initialize()
            fitness = np.array([problem(x) for x in pop])

            best_fitness = float(fitness.min())
            prev_best = best_fitness
            fitness_history.append(best_fitness)

            for t in range(self.T):
                # Compute state
                state = state_extractor.compute(pop, fitness, t, self.T)

                # Agent action
                inf_start = time.time()
                action_bins = self.agent.select_action(state)
                inf_time = time.time() - inf_start
                total_inf_time += inf_time

                # Execute step
                pop, fitness = opt.step(
                    pop, fitness,
                    tuple(int(b) for b in action_bins),
                    problem, t, self.T
                )

                curr_best = float(fitness.min())
                if curr_best < best_fitness:
                    best_fitness = curr_best
                fitness_history.append(best_fitness)
                prev_best = curr_best

            all_best_fitness.append(best_fitness)
            all_convergence.append(fitness_history)

        train_time = 0.0

        return (
            np.min(all_best_fitness),
            np.mean(all_best_fitness),
            np.std(all_best_fitness),
            mean_conv.tolist(),
            train_time,
            total_inf_time / self.n_runs
        )

    def evaluate_bbob(
        self,
        bbob_suite: BBOBSuite,
        verbose: bool = True
    ) -> Dict[str, EvaluationResult]:
        """
        Evaluate on BBOB test suite.

        Returns:
            Dict of {function_name: EvaluationResult}
        """
        results = {}

        test_functions = bbob_suite.get_test_functions()

        for name, fn in test_functions:
            bounds = bbob_suite.get_bounds(name.split('_')[0])

            if verbose:
                print(f"  Evaluating on {name}...")

            (
                best_f,
                mean_f,
                std_f,
                conv_curve,
                train_time,
                inf_time
            ) = self.evaluate_on_problem(
                problem=fn,
                dim=bbob_suite.dim,
                bounds=bounds,
                optimum=fn.optimum
            )

            # Compute actual performance (cumulative improvement)
            mean_conv = np.array(conv_curve)
            perf = 0.0
            for i in range(1, len(mean_conv)):
                if mean_conv[i-1] > mean_conv[i]:
                    perf += (mean_conv[i-1] - mean_conv[i]) / (mean_conv[0] - fn.optimum + 1e-8)

            results[name] = EvaluationResult(
                name=name,
                performance=perf,
                mean_fitness=mean_f,
                std_fitness=std_f,
                training_time=train_time,
                inference_time=inf_time,
                n_evaluations=self.T * self.pop_size,
                convergence_curve=conv_curve
            )

        return results


def benchmark_in_distribution(
    agent: QMAgent,
    bbob_suite: BBOBSuite,
    n_runs: int = 19,
    verbose: bool = True
) -> Dict:
    """
    In-distribution generalization test.

    Tests on 8 unseen BBOB instances.
    """
    evaluator = Evaluator(
        agent=agent,
        n_runs=n_runs
    )

    results = evaluator.evaluate_bbob(bbob_suite, verbose=verbose)

    # Aggregate
    all_perfs = [r.performance for r in results.values()]
    all_mean_f = [r.mean_fitness for r in results.values()]

    if verbose:
        print(f"\nIn-Distribution Results:")
        print(f"  Mean Performance: {np.mean(all_perfs):.4f} ± {np.std(all_perfs):.4f}")
        print(f"  Mean Best Fitness: {np.mean(all_mean_f):.4f} ± {np.std(all_mean_f):.4f}")

    return {
        'results': {k: vars(v) for k, v in results.items()},
        'summary': {
            'mean_performance': float(np.mean(all_perfs)),
            'std_performance': float(np.std(all_perfs)),
            'mean_best_fitness': float(np.mean(all_mean_f)),
            'std_best_fitness': float(np.std(all_mean_f))
        }
    }


def benchmark_out_of_distribution(
    agent: QMAgent,
    task_name: str = 'Hopper',
    n_runs: int = 10,
    T: int = 50,
    pop_size: int = 10,
    verbose: bool = True
) -> Dict:
    """
    Out-of-distribution: Neuroevolution on MuJoCo tasks.

    Note: Requires mujoco-py or brax to be installed.
    This is a placeholder implementation.
    """
    try:
        # Try to import mujoco
        import mujoco
        HAS_MUJOCO = True
    except ImportError:
        HAS_MUJOCO = False

    if not HAS_MUJOCO:
        if verbose:
            print(f"  MuJoCo not available, skipping OOD test")
        return {'error': 'MuJoCo not installed'}

    if verbose:
        print(f"  Running OOD test on {task_name}...")

    # Placeholder: In practice, this would:
    # 1. Create a 2-layer MLP policy network
    # 2. Use the agent to configure DE parameters for evolving the policy
    # 3. Evaluate on MuJoCo task

    return {
        'task': task_name,
        'n_runs': n_runs,
        'note': 'Placeholder - requires full MuJoCo integration'
    }


def ablation_lambda_beta(
    agent_class,
    train_loader,
    val_loader,
    lambdas: List[float] = [0.0, 1.0, 10.0],
    betas: List[float] = [1.0, 10.0],
    n_epochs: int = 50,
    verbose: bool = True
) -> Dict:
    """
    Ablation study on λ and β parameters.
    """
    results = {}

    for lam in lambdas:
        for beta in betas:
            if verbose:
                print(f"  Testing λ={lam}, β={beta}...")

            # Create model and trainer with these params
            from model.qmamba import QMamba
            from model.trainer import QMTrainer, TrainingConfig

            model = QMamba()
            config = TrainingConfig(beta=beta, lam=lam)
            trainer = QMTrainer(model, config)

            # Train briefly
            history = trainer.fit(train_loader, val_loader, n_epochs=n_epochs, verbose=False)

            results[f'lambda_{lam}_beta_{beta}'] = {
                'final_loss': float(history['total_loss'][-1]) if history['total_loss'] else float('inf'),
                'final_td': float(history['td_loss'][-1]) if history['td_loss'] else float('inf'),
                'final_cql': float(history['cql_loss'][-1]) if history['cql_loss'] else float('inf')
            }

    return results


def ablation_mix_ratio(
    dataset_builder,
    mu_values: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0],
    n_total: int = 5000,
    verbose: bool = True
) -> Dict:
    """
    Ablation study on mix ratio μ.
    """
    results = {}

    for mu in mu_values:
        if verbose:
            print(f"  Testing μ={mu}...")

        builder = dataset_builder.__class__(
            bbob_suite=dataset_builder.bbob_suite,
            optimizer_class=dataset_builder.optimizer_class,
            mu=mu,
            seed=dataset_builder.seed
        )

        train_trajs, val_trajs = builder.build(
            n_total=n_total,
            verbose=False
        )

        results[f'mu_{mu}'] = {
            'n_train': len(train_trajs),
            'n_val': len(val_trajs),
            'mean_reward': float(np.mean([t.total_reward for t in train_trajs])),
            'std_reward': float(np.std([t.total_reward for t in train_trajs]))
        }

    return results
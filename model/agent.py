"""
Q-Mamba Agent for Inference

Wraps trained Q-Mamba model for use in optimization.
"""

import numpy as np
import torch
from typing import Dict, Optional, Callable, List

from .qmamba import QMamba


class QMAgent:
    def __init__(
        self,
        model: QMamba,
        device: str = 'cpu',
        deterministic: bool = True
    ):
        self.model = model.to(device)
        self.device = device
        self.deterministic = deterministic
        self.model.eval()

    @classmethod
    def from_checkpoint(cls, path: str, device: str = 'cpu', **model_kwargs) -> 'QMAgent':
        """Load agent from checkpoint."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        config = checkpoint.get('config', {})
        model_kwargs.setdefault('state_dim', config.get('state_dim', 9))
        model_kwargs.setdefault('K', config.get('K', 3))
        model_kwargs.setdefault('M', config.get('M', 16))
        model_kwargs.setdefault('d_model', config.get('d_model', 128))

        model = QMamba(**model_kwargs)
        model.load_state_dict(checkpoint['model_state_dict'])

        return cls(model, device=device)

    def select_action(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device)
            acts, _, _ = self.model.act(s, deterministic=self.deterministic)
            return acts.cpu().numpy()[0]

    def run_optimization(
        self,
        problem: Callable,
        dim: int,
        bounds: np.ndarray,
        pop_size: int = 20,
        T: int = 500,
        optimizer_class=None,
        seed: Optional[int] = None,
        track_history: bool = True
    ) -> Dict:
        if optimizer_class is None:
            from algorithms.alg0 import Alg0Optimizer as optimizer_class

        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()

        # Initialize optimizer
        from env.state import StateExtractor
        from env.action import ActionSpace

        state_extractor = StateExtractor()
        K = self.model.K
        M = self.model.M

        # Get parameter ranges based on optimizer
        if optimizer_class.__name__ == 'Alg0Optimizer':
            param_ranges = [
                optimizer_class.F1_range,
                optimizer_class.F2_range,
                optimizer_class.Cr_range
            ]
        else:
            param_ranges = [(0.0, 1.0)] * K

        action_space = ActionSpace(K, M, param_ranges)

        opt = optimizer_class(dim=dim, bounds=bounds, pop_size=pop_size, seed=seed or rng.randint(1e6))
        pop = opt.initialize()
        fitness = np.array([problem(x) for x in pop])

        # Track history
        best_fitness = float(fitness.min())
        best_x = pop[np.argmin(fitness)].copy()

        if track_history:
            fitness_history = [best_fitness]
            config_history = []

        # Initial state
        prev_best = best_fitness

        for t in range(T):
            # Compute state
            state = state_extractor.compute(pop, fitness, t, T)

            # Get action from agent
            action_bins = self.select_action(state)
            action_bins = np.clip(action_bins, 0, M - 1)

            # Convert bins to parameters
            params = action_space.undiscretize_bins(action_bins)

            # Execute step
            prev_pop = pop.copy()
            prev_fitness = fitness.copy()
            pop, fitness = opt.step(pop, fitness, tuple(int(b) for b in action_bins), problem, t, T)

            # Update best
            curr_best = float(fitness.min())
            if curr_best < best_fitness:
                best_fitness = curr_best
                best_x = pop[np.argmin(fitness)].copy()

            if track_history:
                fitness_history.append(best_fitness)
                config_history.append({
                    't': t,
                    'action_bins': action_bins.tolist(),
                    'params': params.tolist() if hasattr(params, 'tolist') else list(params),
                    'best_fitness': best_fitness
                })

            prev_best = curr_best

        result = {
            'best_fitness': best_fitness,
            'best_x': best_x,
            'n_evaluations': problem.n_evaluations if hasattr(problem, 'n_evaluations') else T * pop_size
        }

        if track_history:
            result['fitness_history'] = fitness_history
            result['config_history'] = config_history

        return result

    @property
    def uses_mamba(self) -> bool:
        """Check if model uses Mamba."""
        return self.model.uses_mamba

    @property
    def num_parameters(self) -> int:
        """Get number of parameters."""
        return self.model.num_parameters
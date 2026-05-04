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
    def from_checkpoint(cls, path: str, device: str = 'cpu', force_cpu: bool = False, **model_kwargs) -> 'QMAgent':
        """Load agent from checkpoint."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        config = checkpoint.get('config', {})
        model_kwargs.setdefault('state_dim', config.get('state_dim', 9))
        model_kwargs.setdefault('K', config.get('K', 3))
        model_kwargs.setdefault('M', config.get('M', 16))
        model_kwargs.setdefault('d_model', config.get('d_model', 14))
        model_kwargs.setdefault('d_state', config.get('d_state', 32))
        model_kwargs.setdefault('n_layers', config.get('n_layers', 1))
        model_kwargs.setdefault('num_hidden_mlp', config.get('num_hidden_mlp', 32))

        # Force CPU fallback for Mamba if requested
        if force_cpu:
            model_kwargs['force_cpu'] = True

        # Create model and load weights
        model = QMamba(**model_kwargs)

        # Load state dict - handles CPU/CUDA transfer automatically
        state_dict = checkpoint['model_state_dict']
        # Remove 'module.' prefix if present (from DataParallel)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)

        # Move to specified device
        model = model.to(device)

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

        rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
        from env.state import StateExtractor
        from env.action import ActionSpace

        state_extractor = StateExtractor()
        K, M = self.model.K, self.model.M

        if optimizer_class.__name__ == 'Alg0Optimizer':
            param_ranges = [optimizer_class.F1_range, optimizer_class.F2_range, optimizer_class.Cr_range]
        else:
            param_ranges = [(0.0, 1.0)] * K

        action_space = ActionSpace(K, M, param_ranges)
        opt = optimizer_class(dim=dim, bounds=bounds, pop_size=pop_size, seed=seed or rng.randint(1e6))
        pop = opt.initialize()
        fitness = np.array([problem(x) for x in pop])

        best_fitness = float(fitness.min())
        best_x = pop[np.argmin(fitness)].copy()
        fitness_history = [best_fitness] if track_history else None
        config_history = [] if track_history else None

        for t in range(T):
            state = state_extractor.compute(pop, fitness, t, T)
            action_bins = np.clip(self.select_action(state), 0, M - 1)
            params = action_space.undiscretize_bins(action_bins)

            pop, fitness = opt.step(pop, fitness, tuple(int(b) for b in action_bins), problem, t, T)

            curr_best = float(fitness.min())
            if curr_best < best_fitness:
                best_fitness = curr_best
                best_x = pop[np.argmin(fitness)].copy()

            if track_history:
                fitness_history.append(best_fitness)
                config_history.append({
                    't': t, 'action_bins': action_bins.tolist(),
                    'params': params.tolist() if hasattr(params, 'tolist') else list(params),
                    'best_fitness': best_fitness
                })

        result = {
            'best_fitness': best_fitness, 'best_x': best_x,
            'n_evaluations': problem.n_evaluations if hasattr(problem, 'n_evaluations') else T * pop_size
        }
        if track_history:
            result['fitness_history'] = fitness_history
            result['config_history'] = config_history
        return result

    @property
    def uses_mamba(self) -> bool:
        return self.model.uses_mamba

    @property
    def num_parameters(self) -> int:
        return self.model.num_parameters
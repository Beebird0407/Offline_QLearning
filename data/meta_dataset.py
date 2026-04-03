"""E&E Dataset Builder and Meta Data Loader."""

import numpy as np
import pickle
from typing import List, Dict, Optional, Tuple, Callable
from pathlib import Path

from .bbob_suite import BBOBSuite
from .trajectory import Trajectory, TrajectoryCollector


class EEDatasetBuilder:
    """
    Builds E&E (Explore & Exploit) trajectory dataset.

    Mixes trajectories from:
    1. Pretrained MetaBBO baselines (exploitation) - μ fraction
    2. Random policy (exploration) - (1-μ) fraction
    """

    def __init__(
        self,
        bbob_suite: BBOBSuite,
        optimizer_class,
        state_dim: int = 9,
        K: int = 3,
        M: int = 16,
        pop_size: int = 20,
        T: int = 500,
        mu: float = 0.5,
        seed: int = 42
    ):
        """
        Args:
            bbob_suite: BBOBSuite instance
            optimizer_class: Optimizer class to use
            state_dim: State dimension
            K: Number of action parameters
            M: Number of bins per parameter
            pop_size: Population size
            T: Trajectory length (generations)
            mu: Mix ratio (fraction from pretrained baselines)
            seed: Random seed
        """
        self.bbob_suite = bbob_suite
        self.optimizer_class = optimizer_class
        self.state_dim = state_dim
        self.K = K
        self.M = M
        self.pop_size = pop_size
        self.T = T
        self.mu = mu
        self.seed = seed

        # Setup state extractor and action space
        from env.state import StateExtractor
        from env.action import ActionSpace

        self.state_extractor = StateExtractor()
        self.action_space = ActionSpace(K, M)

        # Create collector
        self.collector = TrajectoryCollector(
            optimizer_class=optimizer_class,
            state_extractor=self.state_extractor,
            action_space=self.action_space,
            pop_size=pop_size,
            T=T,
            seed=seed
        )

    def _get_pretrained_baselines(self) -> Dict[str, Callable]:
        """
        Get pretrained MetaBBO baselines (RLPSO, LDE, GLEET).

        For now, returns exploit strategy functions.
        In practice, these would be trained neural networks.
        """
        # Placeholder for actual pretrained baselines
        # In real implementation, load trained models here
        return {
            'rlpso': 'exploit',   # RLPSO - exploit-oriented
            'lde': 'exploit',     # LDE - exploit-oriented
            'gleet': 'exploit',   # GLEET - exploit-oriented
        }

    def build(
        self,
        n_total: int = 10000,
        n_train_tasks: Optional[int] = None,
        meta_agents: Optional[Dict[str, object]] = None,
        save_path: Optional[str] = None,
        verbose: bool = True
    ) -> Tuple[List[Trajectory], List[Trajectory]]:
        """
        Build the E&E dataset.

        Args:
            n_total: Total number of trajectories (D=10K default)
            n_train_tasks: Number of training tasks to use (default: all)
            meta_agents: Dict of trained meta agents for 'meta_alg' strategy
            save_path: Optional path to save dataset
            verbose: Print progress

        Returns:
            (train_trajectories, val_trajectories) - 80/20 split
        """
        rng = np.random.RandomState(self.seed)

        # Calculate trajectory counts
        n_exploit = int(n_total * self.mu)
        n_explore = n_total - n_exploit

        if verbose:
            print(f"  Building E&E Dataset:")
            print(f"    Total trajectories: {n_total}")
            print(f"    Exploit (μ={self.mu}): {n_exploit}")
            print(f"    Explore (1-μ): {n_explore}")
            print(f"    Trajectory length: T={self.T}")

        # Get training functions
        train_functions = self.bbob_suite.get_train_functions()
        if n_train_tasks is not None and n_train_tasks < len(train_functions):
            train_functions = train_functions[:n_train_tasks]

        all_trajectories = []

        # === Collect Exploitation Trajectories ===
        if verbose:
            print(f"\n  Collecting {n_exploit} exploit trajectories...")

        # Determine exploitation strategy
        use_meta_agents = meta_agents is not None and len(meta_agents) > 0

        if use_meta_agents:
            # Get list of available baseline names
            baseline_names = list(meta_agents.keys())
            if verbose:
                print(f"    Using pretrained baselines: {baseline_names}")

        for i in range(n_exploit):
            # Select task (cycle through available tasks)
            task_name, task_fn = train_functions[i % len(train_functions)]
            bounds = self.bbob_suite.get_bounds(task_name.split('_')[0])

            if use_meta_agents:
                # Rotate through available baselines
                baseline_name = baseline_names[i % len(baseline_names)]
                agent = meta_agents[baseline_name]
                traj = self.collector.collect_trajectory(
                    problem=task_fn,
                    dim=self.bbob_suite.dim,
                    bounds=bounds,
                    strategy='meta_alg',
                    task_id=f"{task_name}_{baseline_name}_{i}",
                    meta_agent=agent,
                    seed=self.seed + i
                )
            else:
                # Use built-in exploit strategy
                traj = self.collector.collect_trajectory(
                    problem=task_fn,
                    dim=self.bbob_suite.dim,
                    bounds=bounds,
                    strategy='exploit',
                    task_id=f"{task_name}_exploit_{i}",
                    meta_agent=None,
                    seed=self.seed + i
                )
            all_trajectories.append(traj)

            if verbose and (i + 1) % 500 == 0:
                print(f"    Exploit: {i+1}/{n_exploit}")

        # === Collect Exploration Trajectories ===
        if verbose:
            print(f"\n  Collecting {n_explore} explore trajectories...")

        for i in range(n_explore):
            task_name, task_fn = train_functions[i % len(train_functions)]
            bounds = self.bbob_suite.get_bounds(task_name.split('_')[0])

            traj = self.collector.collect_trajectory(
                problem=task_fn,
                dim=self.bbob_suite.dim,
                bounds=bounds,
                strategy='random',
                task_id=f"{task_name}_explore_{i}",
                meta_agent=None,
                seed=self.seed + n_exploit + i
            )
            all_trajectories.append(traj)

            if verbose and (i + 1) % 500 == 0:
                print(f"    Explore: {i+1}/{n_explore}")

        # === Shuffle and Split ===
        rng.shuffle(all_trajectories)
        n_val = max(1, int(len(all_trajectories) * 0.2))
        train_trajs = all_trajectories[:-n_val]
        val_trajs = all_trajectories[-n_val:]

        if verbose:
            rewards = [t.total_reward for t in all_trajectories]
            print(f"\n  Dataset built successfully!")
            print(f"    Train trajectories: {len(train_trajs)}")
            print(f"    Val trajectories: {len(val_trajs)}")
            print(f"    Avg reward: {np.mean(rewards):.4f} ± {np.std(rewards):.4f}")

        # === Save if requested ===
        if save_path:
            self._save_dataset(train_trajs, val_trajs, save_path)
            if verbose:
                print(f"    Saved to: {save_path}")

        return train_trajs, val_trajs

    def _save_dataset(
        self,
        train_trajs: List[Trajectory],
        val_trajs: List[Trajectory],
        path: str
    ):
        """Save dataset to pickle file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        dataset = {
            'train': [t.to_dict() for t in train_trajs],
            'val': [t.to_dict() for t in val_trajs],
            'config': {
                'state_dim': self.state_dim,
                'K': self.K,
                'M': self.M,
                'pop_size': self.pop_size,
                'T': self.T,
                'mu': self.mu,
                'dim': self.bbob_suite.dim,
            }
        }

        with open(path, 'wb') as f:
            pickle.dump(dataset, f)

    @classmethod
    def load_dataset(cls, path: str) -> Tuple[List[Trajectory], List[Trajectory], dict]:
        """Load dataset from pickle file."""
        with open(path, 'rb') as f:
            dataset = pickle.load(f)

        train_trajs = [Trajectory.from_dict(d) for d in dataset['train']]
        val_trajs = [Trajectory.from_dict(d) for d in dataset['val']]

        return train_trajs, val_trajs, dataset['config']


class MetaDataLoader:
    """
    Data loader for meta-learning batches.

    Samples trajectory fragments for offline RL training.
    """

    def __init__(
        self,
        trajectories: List[Trajectory],
        batch_size: int = 64,
        state_dim: int = 9,
        K: int = 3,
        T_max: Optional[int] = None
    ):
        """
        Args:
            trajectories: List of trajectories
            batch_size: Batch size
            state_dim: State dimension
            K: Number of action parameters
            T_max: Maximum trajectory length (for padding)
        """
        self.trajectories = trajectories
        self.batch_size = batch_size
        self.state_dim = state_dim
        self.K = K

        # Determine T_max
        if T_max is None:
            T_max = max(t.length for t in trajectories)
        self.T_max = T_max

        # Precompute trajectory lengths
        self.lengths = [t.length for t in trajectories]

    def sample_batch(self) -> Dict:
        """
        Sample a batch of trajectory fragments.

        Returns:
            dict with keys: states, actions, rewards, next_states, dones, mask
        """
        # Sample trajectory indices
        indices = np.random.choice(len(self.trajectories), self.batch_size, replace=True)

        # Allocate arrays
        states = np.zeros((self.batch_size, self.T_max, self.state_dim), dtype=np.float32)
        actions = np.zeros((self.batch_size, self.T_max, self.K), dtype=np.int64)
        rewards = np.zeros((self.batch_size, self.T_max), dtype=np.float32)
        next_states = np.zeros((self.batch_size, self.T_max, self.state_dim), dtype=np.float32)
        dones = np.zeros((self.batch_size, self.T_max), dtype=np.float32)
        mask = np.zeros((self.batch_size, self.T_max), dtype=np.float32)

        for b, idx in enumerate(indices):
            traj = self.trajectories[idx]
            T = traj.length

            states[b, :T] = traj.get_states()
            actions[b, :T] = traj.get_actions()
            rewards[b, :T] = traj.get_rewards()
            next_states[b, :T] = traj.get_next_states()
            dones[b, :T] = traj.get_dones()
            mask[b, :T] = 1.0

        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones,
            'mask': mask
        }

    def __len__(self) -> int:
        """Number of batches per epoch (at least 1)."""
        # Each batch samples with replacement, so we can always generate batches
        return max(1, len(self.trajectories))

    def __iter__(self):
        """Iterate over fixed number of batches (one epoch)."""
        for _ in range(len(self)):
            yield self.sample_batch()

    def iterate_batches(self, steps_per_epoch: int = 100) -> Dict:
        """Iterate over random batches."""
        for _ in range(steps_per_epoch):
            yield self.sample_batch()
"""E&E Dataset Builder and Meta Data Loader."""

import numpy as np
import pickle
from typing import List, Dict, Optional, Tuple, Callable
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from .bbob_suite import BBOBSuite
from .trajectory import Trajectory, TrajectoryCollector


# ──────────────────────────── Worker function for multiprocessing ────────────────────────────

def _collect_trajectory_worker(args: dict) -> dict:
    """Worker function for parallel trajectory collection.

    Reconstructs everything from serializable args to avoid pickling issues.
    Returns trajectory as a dict (serializable).
    """
    from env.state import StateExtractor
    from env.action import ActionSpace
    from algorithms.alg0 import Alg0Optimizer
    from algorithms.alg1 import Alg1Optimizer
    from algorithms.alg2 import Alg2Optimizer
    from data.bbob_suite import BBOBSuite

    alg_map = {'Alg0': Alg0Optimizer, 'Alg1': Alg1Optimizer, 'Alg2': Alg2Optimizer,
               'Alg0Optimizer': Alg0Optimizer, 'Alg1Optimizer': Alg1Optimizer, 'Alg2Optimizer': Alg2Optimizer}

    optimizer_class = alg_map[args['optimizer_name']]
    strategy = args['strategy']
    seed = args['seed']
    dim = args['dim']
    pop_size = args['pop_size']
    T = args['T']
    use_lpsr = args.get('use_lpsr', True)
    min_pop_size = args.get('min_pop_size', 4)
    K = args['K']
    M = args['M']
    task_id = args.get('task_id', '')
    cpu_throttle = args.get('cpu_throttle', 0.0)

    # Reconstruct BBOB function
    bbob_suite = BBOBSuite(dim=dim, n_train_instances=1, n_test_instances=1, seed=42)
    fid = args.get('fid', 1)
    inst = args.get('inst', 0)
    fn_class = bbob_suite.FUNCTION_CLASSES[fid]
    rng_shift = np.random.RandomState(42 + fid * 1000 + inst)
    shift = rng_shift.uniform(-4.0, 4.0, dim)
    try:
        fn = fn_class(dim, shift=shift)
    except TypeError:
        fn = fn_class(dim, shift=shift)

    bounds = bbob_suite.get_bounds(fid)

    # Collect trajectory
    state_extractor = StateExtractor()
    action_space = ActionSpace(K, M)
    collector = TrajectoryCollector(
        optimizer_class=optimizer_class,
        state_extractor=state_extractor,
        action_space=action_space,
        pop_size=pop_size,
        T=T,
        seed=seed,
        use_lpsr=use_lpsr,
        min_pop_size=min_pop_size,
        cpu_throttle=cpu_throttle,
    )

    traj = collector.collect_trajectory(
        problem=fn,
        dim=dim,
        bounds=bounds,
        strategy=strategy,
        task_id=task_id,
        meta_agent=None,
        seed=seed,
    )

    return traj.to_dict()


class EEDatasetBuilder:
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
        seed: int = 42,
        use_lpsr: bool = True,
        min_pop_size: int = 4,
        cpu_throttle: float = 0.0,
    ):
        self.bbob_suite = bbob_suite
        self.optimizer_class = optimizer_class
        self.state_dim = state_dim
        self.K = K
        self.M = M
        self.pop_size = pop_size
        self.T = T
        self.mu = mu
        self.seed = seed
        self.use_lpsr = use_lpsr
        self.min_pop_size = min_pop_size
        self.cpu_throttle = cpu_throttle

        from env.state import StateExtractor
        from env.action import ActionSpace

        self.state_extractor = StateExtractor()
        self.action_space = ActionSpace(K, M)

        self.collector = TrajectoryCollector(
            optimizer_class=optimizer_class,
            state_extractor=self.state_extractor,
            action_space=self.action_space,
            pop_size=pop_size,
            T=T,
            seed=seed,
            use_lpsr=use_lpsr,
            min_pop_size=min_pop_size
        )

    def _get_pretrained_baselines(self) -> Dict[str, Callable]:
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
        verbose: bool = True,
        n_workers: Optional[int] = None,
    ) -> Tuple[List[Trajectory], List[Trajectory]]:
        rng = np.random.RandomState(self.seed)

        n_exploit = int(n_total * self.mu)
        n_explore = n_total - n_exploit

        if verbose:
            print(f"  Building E&E Dataset:")
            print(f"    Total trajectories: {n_total}")
            print(f"    Exploit (μ={self.mu}): {n_exploit}")
            print(f"    Explore (1-μ): {n_explore}")
            print(f"    Trajectory length: T={self.T}")

        # Get training function IDs
        train_fn_ids = list(self.bbob_suite.FUNCTION_CLASSES.keys())
        if hasattr(self.bbob_suite, 'TRAIN_IDS'):
            train_fn_ids = self.bbob_suite.TRAIN_IDS
        if n_train_tasks is not None and n_train_tasks < len(train_fn_ids):
            train_fn_ids = train_fn_ids[:n_train_tasks]

        use_meta_agents = meta_agents is not None and len(meta_agents) > 0

        # Build task list (all serializable args)
        task_args = []

        for i in range(n_exploit):
            fid = train_fn_ids[i % len(train_fn_ids)]
            if use_meta_agents:
                baseline_names = list(meta_agents.keys())
                baseline_name = baseline_names[i % len(baseline_names)]
                strategy = 'meta_alg'
                task_id = f"F{fid}_{baseline_name}_exploit_{i}"
            else:
                strategy = 'exploit'
                task_id = f"F{fid}_exploit_{i}"

            task_args.append({
                'optimizer_name': self.optimizer_class.__name__,
                'strategy': strategy,
                'seed': self.seed + i,
                'dim': self.bbob_suite.dim,
                'pop_size': self.pop_size,
                'T': self.T,
                'use_lpsr': self.use_lpsr,
                'min_pop_size': self.min_pop_size,
                'cpu_throttle': self.cpu_throttle,
                'K': self.K,
                'M': self.M,
                'fid': fid,
                'inst': i,
                'task_id': task_id,
            })

        for i in range(n_explore):
            fid = train_fn_ids[i % len(train_fn_ids)]
            task_args.append({
                'optimizer_name': self.optimizer_class.__name__,
                'strategy': 'random',
                'seed': self.seed + n_exploit + i,
                'dim': self.bbob_suite.dim,
                'pop_size': self.pop_size,
                'T': self.T,
                'use_lpsr': self.use_lpsr,
                'min_pop_size': self.min_pop_size,
                'cpu_throttle': self.cpu_throttle,
                'K': self.K,
                'M': self.M,
                'fid': fid,
                'inst': n_exploit + i,
                'task_id': f"F{fid}_explore_{i}",
            })

        # Collect trajectories — stream to disk to avoid OOM
        import os, tempfile, pickle as pk
        tmp_dir = tempfile.mkdtemp(prefix='ee_dataset_')

        if n_workers is None:
            n_workers = min(os.cpu_count() or 1, 8)

        if n_workers <= 1:
            if verbose:
                print(f"\n  Collecting {len(task_args)} trajectories (single process)...")
            for idx, args in enumerate(task_args):
                traj_dict = _collect_trajectory_worker(args)
                with open(os.path.join(tmp_dir, f'{idx}.pkl'), 'wb') as f:
                    pk.dump(traj_dict, f)
                if verbose and (idx + 1) % 500 == 0:
                    print(f"    Progress: {idx+1}/{len(task_args)}")
        else:
            if verbose:
                print(f"\n  Collecting {len(task_args)} trajectories ({n_workers} workers)...")

            batch_size = n_workers * 50
            n_batches = (len(task_args) + batch_size - 1) // batch_size
            completed = 0

            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                for batch_idx in range(n_batches):
                    start = batch_idx * batch_size
                    end = min(start + batch_size, len(task_args))
                    batch_tasks = task_args[start:end]
                    futures = {executor.submit(_collect_trajectory_worker, args): i
                               for i, args in enumerate(batch_tasks)}
                    for future in as_completed(futures):
                        idx = futures[future]
                        try:
                            traj_dict = future.result()
                            with open(os.path.join(tmp_dir, f'{start + idx}.pkl'), 'wb') as f:
                                pk.dump(traj_dict, f)
                        except Exception as e:
                            if verbose:
                                print(f"  [Warning] Worker failed: {e}")
                        completed += 1
                    if verbose:
                        print(f"    Progress: {completed}/{len(task_args)} "
                              f"(batch {batch_idx + 1}/{n_batches})")

        # Shuffle file indices, split, then load train/val in streaming fashion
        saved = sorted(os.listdir(tmp_dir))
        indices = list(range(len(saved)))
        rng.shuffle(indices)
        n_val = max(1, int(len(indices) * 0.2))
        train_idx = set(indices[:-n_val])

        train_trajs, val_trajs = [], []
        for i, fn in enumerate(saved):
            with open(os.path.join(tmp_dir, fn), 'rb') as f:
                traj = Trajectory.from_dict(pk.load(f))
            if i in train_idx:
                train_trajs.append(traj)
            else:
                val_trajs.append(traj)

        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

        if verbose:
            all_rewards = [t.total_reward for t in train_trajs] + [t.total_reward for t in val_trajs]
            print(f"\n  Dataset built successfully!")
            print(f"    Train trajectories: {len(train_trajs)}")
            print(f"    Val trajectories: {len(val_trajs)}")
            print(f"    Avg reward: {np.mean(all_rewards):.4f} ± {np.std(all_rewards):.4f}")

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

        # Get algorithm name from class
        alg_name = self.optimizer_class.__name__ if hasattr(self.optimizer_class, '__name__') else str(self.optimizer_class)

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
                'algorithm': alg_name,
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
            T_max: Maximum trajectory length (for padding). If set, truncates
                   trajectories to this length. If None, uses the longest trajectory.
        """
        self.trajectories = trajectories
        self.batch_size = batch_size
        self.state_dim = state_dim
        self.K = K

        # Determine T_max (cap trajectory length)
        data_max = max(t.length for t in trajectories)
        if T_max is None:
            T_max = data_max
        self.T_max = min(T_max, data_max)

        # Precompute trajectory lengths (capped by T_max)
        self.lengths = [min(t.length, self.T_max) for t in trajectories]

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
            T = min(traj.length, self.T_max)

            states[b, :T] = traj.get_states()[:T]
            actions[b, :T] = traj.get_actions()[:T]
            rewards[b, :T] = traj.get_rewards()[:T]
            next_states[b, :T] = traj.get_next_states()[:T]
            dones[b, :T] = traj.get_dones()[:T]
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
        """Number of batches per epoch."""
        return max(1, len(self.trajectories) // self.batch_size)

    def __iter__(self):
        """Iterate over fixed number of batches (one epoch)."""
        for _ in range(len(self)):
            yield self.sample_batch()

    def iterate_batches(self, steps_per_epoch: int = 100) -> Dict:
        """Iterate over random batches."""
        for _ in range(steps_per_epoch):
            yield self.sample_batch()
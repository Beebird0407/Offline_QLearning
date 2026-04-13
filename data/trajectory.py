"""
Trajectory Collection for E&E Dataset

Records state sequences, action sequences, and reward sequences
for offline RL training.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Callable
import pickle


@dataclass
class Transition:
    """Single transition: (state, action, reward, next_state, done)."""
    state: np.ndarray
    action: np.ndarray  # Bin indices for each hyperparameter
    reward: float
    next_state: np.ndarray
    done: bool

    def to_dict(self) -> dict:
        return {
            'state': self.state,
            'action': self.action,
            'reward': self.reward,
            'next_state': self.next_state,
            'done': self.done
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'Transition':
        return cls(
            state=d['state'],
            action=d['action'],
            reward=d['reward'],
            next_state=d['next_state'],
            done=d['done']
        )


@dataclass
class Trajectory:
    """Complete optimization trajectory."""
    transitions: List[Transition] = field(default_factory=list)
    task_id: str = ""
    strategy: str = ""  # 'exploit', 'explore', or 'meta_alg'
    total_reward: float = 0.0
    initial_fitness: float = 0.0
    final_fitness: float = 0.0

    def __post_init__(self):
        if self.transitions and self.total_reward == 0.0:
            self.total_reward = sum(t.reward for t in self.transitions)
        if self.transitions and self.initial_fitness == 0.0:
            # For minimization: higher initial fitness is worse
            self.initial_fitness = self.transitions[0].state[1] if len(self.transitions[0].state) > 1 else 0.0
            self.final_fitness = self.transitions[-1].next_state[1] if len(self.transitions[-1].next_state) > 1 else 0.0

    def append(self, transition: Transition):
        self.transitions.append(transition)
        self.total_reward += transition.reward

    @property
    def length(self) -> int:
        return len(self.transitions)

    def get_states(self) -> np.ndarray:
        """Get all states as (T, state_dim) array."""
        if not self.transitions:
            return np.array([])
        return np.array([t.state for t in self.transitions])

    def get_actions(self) -> np.ndarray:
        """Get all actions as (T, K) array."""
        if not self.transitions:
            return np.array([])
        return np.array([t.action for t in self.transitions])

    def get_rewards(self) -> np.ndarray:
        """Get all rewards as (T,) array."""
        if not self.transitions:
            return np.array([])
        return np.array([t.reward for t in self.transitions])

    def get_next_states(self) -> np.ndarray:
        """Get all next_states as (T, state_dim) array."""
        if not self.transitions:
            return np.array([])
        return np.array([t.next_state for t in self.transitions])

    def get_dones(self) -> np.ndarray:
        """Get all dones as (T,) array."""
        if not self.transitions:
            return np.array([])
        return np.array([t.done for t in self.transitions])

    def to_dict(self) -> dict:
        return {
            'task_id': self.task_id,
            'strategy': self.strategy,
            'total_reward': self.total_reward,
            'initial_fitness': self.initial_fitness,
            'final_fitness': self.final_fitness,
            'states': self.get_states(),
            'actions': self.get_actions(),
            'rewards': self.get_rewards(),
            'next_states': self.get_next_states(),
            'dones': self.get_dones()
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'Trajectory':
        transitions = []
        for i in range(len(d['states'])):
            transitions.append(Transition(
                state=d['states'][i],
                action=d['actions'][i],
                reward=d['rewards'][i],
                next_state=d['next_states'][i],
                done=d['dones'][i]
            ))
        traj = cls(
            task_id=d['task_id'],
            strategy=d['strategy'],
            total_reward=d['total_reward'],
            initial_fitness=d.get('initial_fitness', 0.0),
            final_fitness=d.get('final_fitness', 0.0)
        )
        traj.transitions = transitions
        return traj

    def save(self, path: str):
        """Save trajectory to file."""
        with open(path, 'wb') as f:
            pickle.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str) -> 'Trajectory':
        """Load trajectory from file."""
        with open(path, 'rb') as f:
            d = pickle.load(f)
        return cls.from_dict(d)


class TrajectoryCollector:
    """
    Collects optimization trajectories using specified optimizer and strategy.
    """

    def __init__(
        self,
        optimizer_class,
        state_extractor,
        action_space,
        pop_size: int = 20,
        T: int = 500,
        seed: int = 42,
        use_lpsr: bool = True,
        min_pop_size: int = 4
    ):
        self.optimizer_class = optimizer_class
        self.state_extractor = state_extractor
        self.action_space = action_space
        self.pop_size = pop_size
        self.T = T
        self.rng = np.random.RandomState(seed)
        self.use_lpsr = use_lpsr
        self.min_pop_size = min_pop_size

    def _compute_reward(
        self,
        prev_fitness: np.ndarray,
        curr_fitness: np.ndarray,
        prev_best: float,
        curr_best: float,
        y_range: float = 1.0
    ) -> float:
        if prev_best <= curr_best:
            return 0.0  # No improvement

        # Normalized improvement
        improvement = prev_best - curr_best
        if y_range > 1e-8:
            return float(np.clip(improvement / y_range, -1.0, 1.0))
        return float(np.clip(improvement, -1.0, 1.0))

    def collect_trajectory(
        self,
        problem: Callable,
        dim: int,
        bounds: np.ndarray,
        strategy: str = 'random',
        task_id: str = "",
        meta_agent=None,
        seed: Optional[int] = None
    ) -> Trajectory:
        rng = self.rng if seed is None else np.random.RandomState(seed)

        # Initialize optimizer
        opt = self.optimizer_class(
            dim=dim,
            bounds=bounds,
            pop_size=self.pop_size,
            seed=seed or rng.randint(1e6),
            use_lpsr=self.use_lpsr,
            min_pop_size=self.min_pop_size
        )
        pop = opt.initialize()
        fitness = np.array([problem(x) for x in pop])

        # Initialize state
        state_extractor = self.state_extractor.__class__()  # Fresh extractor
        prev_best = float(fitness.min())
        best_so_far = prev_best

        trajectory = Trajectory(task_id=task_id, strategy=strategy)

        # Estimate y_range for reward normalization
        y_range = max(1.0, float(fitness.max() - fitness.min()))

        for t in range(self.T):
            # Compute state
            state = state_extractor.compute(pop, fitness, t, self.T)

            # Select action based on strategy
            if strategy == 'random':
                action_bins = rng.randint(0, self.action_space.M, size=self.action_space.K)
            elif strategy == 'exploit':
                # Exploitation-oriented action selection
                prog = t / self.T
                action_bins = np.array([
                    rng.randint(int(8 - prog * 4), int(12 + prog * 4)) % self.action_space.M
                    for _ in range(self.action_space.K)
                ])
            elif strategy == 'meta_alg' and meta_agent is not None:
                # Use trained meta-agent
                with np.no_grad():
                    action_bins = meta_agent.predict(state, rng=rng)
            else:
                action_bins = rng.randint(0, self.action_space.M, size=self.action_space.K)

            # Ensure valid bins
            action_bins = np.clip(action_bins, 0, self.action_space.M - 1)

            # Execute optimization step
            prev_pop = pop.copy()
            prev_fitness = fitness.copy()
            prev_best = best_so_far

            pop, fitness = opt.step(pop, fitness, tuple(action_bins), problem, t, self.T)

            # Update best so far
            curr_best = float(fitness.min())
            if curr_best < best_so_far:
                best_so_far = curr_best

            # Compute reward
            reward = self._compute_reward(prev_fitness, fitness, prev_best, best_so_far, y_range)

            # Compute next state
            next_state = state_extractor.compute(pop, fitness, t + 1, self.T)

            # Check termination
            done = (t == self.T - 1)

            # Add transition
            trajectory.append(Transition(
                state=state,
                action=action_bins.astype(np.int64),
                reward=reward,
                next_state=next_state,
                done=done
            ))

        trajectory.initial_fitness = prev_best
        trajectory.final_fitness = best_so_far

        return trajectory

    def collect_batch(
        self,
        problems: List[Tuple[str, Callable]],
        bounds: np.ndarray,
        strategy: str = 'random',
        meta_agent=None,
        verbose: bool = True
    ) -> List[Trajectory]:
        trajectories = []
        for i, (name, prob) in enumerate(problems):
            dim = len(bounds)
            traj = self.collect_trajectory(
                problem=prob,
                dim=dim,
                bounds=bounds,
                strategy=strategy,
                task_id=name,
                meta_agent=meta_agent,
                seed=42 + i
            )
            trajectories.append(traj)

            if verbose and (i + 1) % 100 == 0:
                print(f"  Collected {i+1}/{len(problems)} trajectories")

        return trajectories
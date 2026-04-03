"""
CoCo BBOB-style Benchmark Suite

Implements standard BBOB test functions with configurable dimensions and shifts.
Training: 16 instances, Testing: 8 instances
Supports dimensions: 5, 10, 20, 50
"""

import numpy as np
from typing import Dict, Tuple, Callable, Optional, List


class BBOBFunction:
    """Base class for BBOB test functions."""

    def __init__(
        self,
        dim: int,
        shift: Optional[np.ndarray] = None,
        rotation: Optional[np.ndarray] = None,
        optimum: float = 0.0
    ):
        self.dim = dim
        self.shift = shift if shift is not None else np.zeros(dim)
        self.rotation = rotation
        self.optimum = optimum
        self.n_evaluations = 0

        # For rotated functions
        if rotation is not None:
            self.rotation_inv = np.linalg.inv(rotation)

    def __call__(self, x: np.ndarray) -> float:
        self.n_evaluations += 1
        return self._evaluate(x)

    def _evaluate(self, x: np.ndarray) -> float:
        raise NotImplementedError

    def _rotate(self, z: np.ndarray) -> np.ndarray:
        """Apply rotation if available."""
        if self.rotation is not None:
            return np.dot(self.rotation, z)
        return z

    def _rotate_inv(self, z: np.ndarray) -> np.ndarray:
        """Apply inverse rotation if available."""
        if self.rotation_inv is not None:
            return np.dot(self.rotation_inv, z)
        return z


class Sphere(BBOBFunction):
    """Sphere function: f(x) = sum((x - shift)^2)"""

    def __init__(self, dim: int, shift: Optional[np.ndarray] = None):
        super().__init__(dim, shift=shift, optimum=0.0)

    def _evaluate(self, x: np.ndarray) -> float:
        z = x - self.shift
        return float(np.sum(z ** 2))


class Ellipsoid(BBOBFunction):
    """Ellipsoid function with conditioning."""

    def __init__(self, dim: int, shift: Optional[np.ndarray] = None, cond: float = 1e6):
        super().__init__(dim, shift=shift, optimum=0.0)
        self.cond = cond

    def _evaluate(self, x: np.ndarray) -> float:
        z = x - self.shift
        coef = [self.cond ** (i / max(self.dim - 1, 1)) for i in range(self.dim)]
        return float(sum(c * z[i] ** 2 for i, c in enumerate(coef)))


class Rastrigin(BBOBFunction):
    """Rastrigin function: highly multimodal."""

    def __init__(self, dim: int, shift: Optional[np.ndarray] = None, scale: float = 1.0):
        super().__init__(dim, shift=shift, optimum=0.0)
        self.scale = scale

    def _evaluate(self, x: np.ndarray) -> float:
        z = (x - self.shift) / self.scale
        return float(10 * self.dim + np.sum(z ** 2 - 10 * np.cos(2 * np.pi * z)))


class Rosenbrock(BBOBFunction):
    """Rosenbrock function (banana function)."""

    def __init__(self, dim: int, shift: Optional[np.ndarray] = None):
        super().__init__(dim, shift=shift, optimum=0.0)

    def _evaluate(self, x: np.ndarray) -> float:
        z = x - self.shift
        return float(sum(100 * (z[i+1] - z[i]**2)**2 + (1 - z[i])**2 for i in range(self.dim - 1)))


class Ackley(BBOBFunction):
    """Ackley function: globally misleading with many local minima."""

    def __init__(self, dim: int, shift: Optional[np.ndarray] = None):
        super().__init__(dim, shift=shift, optimum=0.0)

    def _evaluate(self, x: np.ndarray) -> float:
        z = x - self.shift
        a, b, c = 20.0, 0.2, 2 * np.pi
        term1 = -a * np.exp(-b * np.sqrt(np.mean(z ** 2)))
        term2 = -np.exp(np.mean(np.cos(c * z)))
        return float(-a + np.e + term1 + term2)


class Griewank(BBOBFunction):
    """Griewank function: multi-modal with product term."""

    def __init__(self, dim: int, shift: Optional[np.ndarray] = None):
        super().__init__(dim, shift=shift, optimum=0.0)

    def _evaluate(self, x: np.ndarray) -> float:
        z = x - self.shift
        return float(np.sum(z ** 2) / 4000.0 - np.prod(np.cos(z / np.sqrt(np.arange(1, self.dim + 1)))) + 1)


class Schwefel(BBOBFunction):
    """Schwefel function: deceptive global structure."""

    def __init__(self, dim: int, shift: Optional[np.ndarray] = None):
        super().__init__(dim, shift=shift, optimum=418.9829 * dim)

    def _evaluate(self, x: np.ndarray) -> float:
        z = x - self.shift
        return float(np.sum(z * np.sin(np.sqrt(np.abs(z)))))


class Levy(BBOBFunction):
    """Levy function: irregular landscape."""

    def __init__(self, dim: int, shift: Optional[np.ndarray] = None):
        super().__init__(dim, shift=shift, optimum=0.0)

    def _evaluate(self, x: np.ndarray) -> float:
        z = (x - self.shift) / 4.0
        w = 1.0 + (z - 1.0) / 4.0

        term1 = np.sin(np.pi * w[0]) ** 2
        term2 = sum((w[i] - 1)**2 * (1 + 10 * np.sin(np.pi * w[i] + 1)**2) for i in range(self.dim - 1))
        term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)

        return float(term1 + term2 + term3)


class BBOBSuite:
    FUNCTION_CLASSES = {
        'sphere': Sphere,
        'ellipsoid': Ellipsoid,
        'rastrigin': Rastrigin,
        'rosenbrock': Rosenbrock,
        'ackley': Ackley,
        'griewank': Griewank,
        'schwefel': Schwefel,
        'levy': Levy,
    }

    def __init__(
        self,
        dim: int = 5,
        train_instances: int = 16,
        test_instances: int = 8,
        seed: int = 42
    ):
        self.dim = dim
        self.train_instances = train_instances
        self.test_instances = test_instances
        self.rng = np.random.RandomState(seed)

        # Generate shifts for all instances
        self.train_shifts = self._generate_shifts(train_instances)
        self.test_shifts = self._generate_shifts(test_instances, offset=train_instances)

        # Build function dictionaries
        self.train_fns, self.test_fns = self._build_functions()

    def _generate_shifts(self, n_instances: int, offset: int = 0) -> Dict[str, List[np.ndarray]]:
        """Generate random shifts for each function."""
        shifts = {}
        for fn_name in self.FUNCTION_CLASSES.keys():
            shifts[fn_name] = []
            for i in range(n_instances):
                # Use different seed for reproducibility
                rng = np.random.RandomState(42 + offset + i)
                shift = rng.uniform(-2, 2, self.dim)
                # Scale shift based on function
                if fn_name == 'schwefel':
                    shift *= 50
                elif fn_name == 'ackley':
                    shift *= 3
                elif fn_name == 'griewank':
                    shift *= 100
                shifts[fn_name].append(shift)
        return shifts

    def _build_functions(self) -> Tuple[Dict, Dict]:
        """Build train and test function dictionaries."""
        train_fns = {}
        test_fns = {}

        for fn_name, fn_class in self.FUNCTION_CLASSES.items():
            train_fns[fn_name] = []
            test_fns[fn_name] = []

            for shift in self.train_shifts[fn_name]:
                fn = fn_class(self.dim, shift=shift.copy())
                fn.rng = np.random.RandomState(42 + hash(shift.tobytes()) % (2**31))
                train_fns[fn_name].append(fn)

            for shift in self.test_shifts[fn_name]:
                fn = fn_class(self.dim, shift=shift.copy())
                fn.rng = np.random.RandomState(42 + hash(shift.tobytes()) % (2**31))
                test_fns[fn_name].append(fn)

        return train_fns, test_fns

    def get_train_functions(self) -> List[Tuple[str, BBOBFunction]]:
        """Get all training functions as (name, function) pairs."""
        result = []
        for fn_name, fns in self.train_fns.items():
            for i, fn in enumerate(fns):
                result.append((f"{fn_name}_{i}", fn))
        return result

    def get_test_functions(self) -> List[Tuple[str, BBOBFunction]]:
        """Get all testing functions as (name, function) pairs."""
        result = []
        for fn_name, fns in self.test_fns.items():
            for i, fn in enumerate(fns):
                result.append((f"{fn_name}_{i}", fn))
        return result

    def get_bounds(self, fn_name: str) -> np.ndarray:
        """Get recommended bounds for a function."""
        bounds_map = {
            'sphere': (-5.0, 5.0),
            'ellipsoid': (-5.0, 5.0),
            'rastrigin': (-5.12, 5.12),
            'rosenbrock': (-2.0, 2.0),
            'ackley': (-32.768, 32.768),
            'griewank': (-600.0, 600.0),
            'schwefel': (-500.0, 500.0),
            'levy': (-10.0, 10.0),
        }
        lo, hi = bounds_map.get(fn_name, (-5.0, 5.0))
        return np.array([[lo, hi]] * self.dim)

    def __len__(self) -> int:
        return len(self.FUNCTION_CLASSES)

    def function_names(self) -> List[str]:
        return list(self.FUNCTION_CLASSES.keys())
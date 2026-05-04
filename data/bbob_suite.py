"""
CoCo BBOB Benchmark Suite — Full 24-function implementation.

Implements all 24 BBOB functions (F1–F24) from the COCO benchmark,
categorized into 5 groups per the paper's Table 5:

  Group 1  Separable:            F1, F2, F3*, F4, F5, F6
  Group 2  Low/moderate cond.:   F7, F8*, F9*, F10, F11, F12
  Group 3  High conditioning:    F13*, F14*, F15, F16, F17
  Group 4  Multi-modal:          F18*, F19, F20*, F21*, F22, F23, F24

  Train (16): F1, F2, F4, F5, F6, F7, F10, F11, F12, F15, F16, F17, F19, F22, F23, F24
  Test  (8):  F3, F8, F9, F13, F14, F18, F20, F21

Each function uses standard BBOB transformations: shift, rotation,
oscillatory (T_osz), asymmetric (T_asy), and conditioning (T_lambda).
"""

import numpy as np
from typing import Dict, Tuple, Callable, Optional, List
from scipy.stats import ortho_group


def _T_osz(x: np.ndarray) -> np.ndarray:
    """Oscillatory transformation applied element-wise (COCO BBOB)."""
    out = np.zeros_like(x)
    for i, xi in enumerate(x):
        if xi == 0:
            out[i] = 0.0
        else:
            si = np.sign(xi)
            li = np.log(np.abs(xi))
            out[i] = si * np.exp(li + 0.49 * (np.sin(li) + np.sin(0.79 * li)))
    return out


def _T_asy(x: np.ndarray, beta: float = 0.2) -> np.ndarray:
    """Asymmetric transformation (COCO BBOB)."""
    n = len(x)
    out = np.copy(x)
    for i, xi in enumerate(x):
        if xi > 0:
            out[i] = xi ** (1.0 + beta * (i / max(n - 1, 1)) * np.sqrt(xi))
    return out


def _T_lambda(x: np.ndarray, alpha: float) -> np.ndarray:
    """Lambda (conditioning) transformation: alpha^((i-1)/(n-1)) * x_i."""
    n = len(x)
    if n == 1:
        return x.copy()
    coeffs = np.array([alpha ** ((i) / (n - 1)) for i in range(n)])
    return coeffs * x


def _rotation_matrix(dim: int, rng: np.random.RandomState) -> np.ndarray:
    """Generate a random orthogonal rotation matrix."""
    return ortho_group.rvs(dim, random_state=rng)


def _xopt(dim: int, rng: np.random.RandomState, range_: float = 5.0) -> np.ndarray:
    """Generate a random optimum location in [-range_, range_]."""
    return rng.uniform(-range_, range_, dim)


class BBOBFunction:
    """Base class for BBOB test functions."""

    def __init__(
        self,
        dim: int,
        fid: int,
        shift: Optional[np.ndarray] = None,
        rotation: Optional[np.ndarray] = None,
        rotation2: Optional[np.ndarray] = None,
        optimum: float = 0.0
    ):
        self.dim = dim
        self.fid = fid
        self.shift = shift if shift is not None else np.zeros(dim)
        self.rotation = rotation
        self.rotation2 = rotation2
        self.optimum = optimum
        self.n_evaluations = 0

        if rotation is not None:
            self.rotation_inv = np.linalg.inv(rotation)
        else:
            self.rotation_inv = None

        if rotation2 is not None:
            self.rotation2_inv = np.linalg.inv(rotation2)
        else:
            self.rotation2_inv = None

    def __call__(self, x: np.ndarray) -> float:
        self.n_evaluations += 1
        return self._evaluate(x)

    def _evaluate(self, x: np.ndarray) -> float:
        raise NotImplementedError

    def _rotate(self, z: np.ndarray, R: Optional[np.ndarray] = None) -> np.ndarray:
        R = R if R is not None else self.rotation
        if R is not None:
            return R @ z
        return z


class F1_Sphere(BBOBFunction):
    """F1: Sphere — f(x) = sum(z_i^2)"""
    def __init__(self, dim, shift=None, rotation=None):
        super().__init__(dim, 1, shift=shift, rotation=rotation, optimum=0.0)

    def _evaluate(self, x):
        z = self._rotate(x - self.shift)
        return float(np.sum(z ** 2))


class F2_Ellipsoidal(BBOBFunction):
    """F2: Separable ellipsoidal — f(x) = sum(10^(6*i/(n-1)) * z_i^2)"""
    def __init__(self, dim, shift=None, rotation=None):
        super().__init__(dim, 2, shift=shift, rotation=rotation, optimum=0.0)

    def _evaluate(self, x):
        z = _T_osz(x - self.shift)
        n = len(z)
        if n == 1:
            return float(z[0] ** 2)
        coeffs = np.array([10 ** (6.0 * i / (n - 1)) for i in range(n)])
        return float(np.sum(coeffs * z ** 2))


class F3_Rastrigin(BBOBFunction):
    """F3: Rastrigin — highly multimodal, separable.
       f(x) = 10*n + sum(z_i^2 - 10*cos(2*pi*z_i))"""
    def __init__(self, dim, shift=None, rotation=None, rotation2=None):
        super().__init__(dim, 3, shift=shift, rotation=rotation, rotation2=rotation2, optimum=0.0)

    def _evaluate(self, x):
        z = x - self.shift
        z = _T_asy(z, 0.2)
        z = _T_lambda(z, 10.0)
        z = self._rotate(z)
        return float(10.0 * len(z) + np.sum(z ** 2 - 10.0 * np.cos(2.0 * np.pi * z)))


class F4_BuecheRastrigin(BBOBFunction):
    """F4: Büche-Rastrigin — separable, asymmetric oscillations.
       f(x) = 10*(n-sum(cos(2*pi*zi))) + sum(zi^2) + 10*fpen"""
    def __init__(self, dim, shift=None, rotation=None):
        super().__init__(dim, 4, shift=shift, rotation=rotation, optimum=0.0)

    def _evaluate(self, x):
        z = x - self.shift
        # Asymmetric oscillation: multiply by sign for odd-indexed >0
        for i in range(len(z)):
            if z[i] > 0 and i % 2 == 0:
                z[i] = np.sqrt(z[i])
                z[i] *= 10.0
        z = _T_osz(z)
        n = len(z)
        term1 = 10.0 * (n - np.sum(np.cos(2.0 * np.pi * z)))
        term2 = np.sum(z ** 2)
        # Boundary penalty
        penalty = np.sum(np.maximum(0, np.abs(x - self.shift) - 5.0) ** 2)
        return float(term1 + term2 + 10.0 * penalty)


class F5_LinearSlope(BBOBFunction):
    """F5: Linear slope — f(x) = sum(5*|si| - si*xi)
       where si = sign(xi_opt), optimum at boundary."""
    def __init__(self, dim, shift=None, rotation=None, xopt=None):
        super().__init__(dim, 5, shift=shift, rotation=rotation, optimum=0.0)
        self.rng = np.random.RandomState(42 + dim)
        self.xopt = xopt if xopt is not None else _xopt(dim, self.rng)
        self.s = np.sign(self.xopt)
        self.s[self.s == 0] = 1.0

    def _evaluate(self, x):
        z = x - self.shift
        result = 0.0
        for i in range(len(z)):
            if self.s[i] * self.xopt[i] > 0:
                result += 5.0 * np.abs(self.s[i]) - self.s[i] * z[i]
            else:
                result += 5.0 * np.abs(self.s[i]) + self.s[i] * z[i]
        return float(result)


class F6_AttractiveSector(BBOBFunction):
    """F6: Attractive sector — f(x) = T_osz(sum(100^((i-1)/(n-1)) * z_i^2))
       where z = T_asy(T_osz(R*(x-xopt)))"""
    def __init__(self, dim, shift=None, rotation=None, xopt=None):
        super().__init__(dim, 6, shift=shift, rotation=rotation, optimum=0.0)
        self.rng = np.random.RandomState(42 + dim + 100)
        self.xopt = xopt if xopt is not None else _xopt(dim, self.rng)

    def _evaluate(self, x):
        z = x - self.xopt - self.shift
        z = self._rotate(z)
        z = _T_osz(z)
        z = _T_asy(z, 0.2)
        n = len(z)
        if n == 1:
            coeffs = np.array([1.0])
        else:
            coeffs = np.array([100.0 ** (i / (n - 1)) for i in range(n)])
        return float(_T_osz(np.array([np.sum(coeffs * z ** 2)]))[0])


class F7_StepEllipsoidal(BBOBFunction):
    """F7: Step ellipsoidal — f(x) = 100*max(|x0|/10^4, sum(10^(6i/(n-1)) * round(zi+0.5)^2)) + ||z||^2"""
    def __init__(self, dim, shift=None, rotation=None, xopt=None):
        super().__init__(dim, 7, shift=shift, rotation=rotation, optimum=0.0)
        self.rng = np.random.RandomState(42 + dim + 200)
        self.xopt = xopt if xopt is not None else _xopt(dim, self.rng)

    def _evaluate(self, x):
        z = x - self.xopt - self.shift
        z = self._rotate(z)
        z_hat = _T_osz(z)
        n = len(z)
        if n == 1:
            coeffs = np.array([1.0])
        else:
            coeffs = np.array([10.0 ** (6.0 * i / (n - 1)) for i in range(n)])
        z_round = np.round(z_hat + 0.5)
        z_round[0] = z_hat[0]
        ellip = np.sum(coeffs * z_round ** 2)
        penalty = np.sum(z ** 2)
        return float(100.0 * max(np.abs(z_hat[0]) / 1e4, ellip) + penalty)


class F8_Rosenbrock(BBOBFunction):
    """F8: Rosenbrock original — f(x) = sum(100*(x_{i+1}-x_i^2)^2 + (1-x_i)^2)"""
    def __init__(self, dim, shift=None, rotation=None):
        super().__init__(dim, 8, shift=shift, rotation=rotation, optimum=0.0)

    def _evaluate(self, x):
        z = x - self.shift
        z = z * 2.5  # Standard BBOB scaling: x → 2.5*x + 4.5
        z = z + 4.5
        n = len(z)
        result = 0.0
        for i in range(n - 1):
            result += 100.0 * (z[i + 1] - z[i] ** 2) ** 2 + (1.0 - z[i]) ** 2
        return float(result)


class F9_RosenbrockRotated(BBOBFunction):
    """F9: Rosenbrock, rotated — same as F8 but with rotation."""
    def __init__(self, dim, shift=None, rotation=None):
        super().__init__(dim, 9, shift=shift, rotation=rotation, optimum=0.0)

    def _evaluate(self, x):
        z = x - self.shift
        z = self._rotate(z)
        z = z * 2.5
        z = z + 4.5
        n = len(z)
        result = 0.0
        for i in range(n - 1):
            result += 100.0 * (z[i + 1] - z[i] ** 2) ** 2 + (1.0 - z[i]) ** 2
        return float(result)


class F10_EllipsoidalRotated(BBOBFunction):
    """F10: Ellipsoidal, rotated — f(x) = sum(10^(6*i/(n-1)) * z_i^2)"""
    def __init__(self, dim, shift=None, rotation=None):
        super().__init__(dim, 10, shift=shift, rotation=rotation, optimum=0.0)

    def _evaluate(self, x):
        z = x - self.shift
        z = self._rotate(z)
        z = _T_osz(z)
        n = len(z)
        if n == 1:
            coeffs = np.array([1.0])
        else:
            coeffs = np.array([10.0 ** (6.0 * i / (n - 1)) for i in range(n)])
        return float(np.sum(coeffs * z ** 2))


class F11_Discus(BBOBFunction):
    """F11: Discus — f(x) = 10^6 * z_0^2 + sum(z_i^2), i>0"""
    def __init__(self, dim, shift=None, rotation=None):
        super().__init__(dim, 11, shift=shift, rotation=rotation, optimum=0.0)

    def _evaluate(self, x):
        z = x - self.shift
        z = self._rotate(z)
        z = _T_osz(z)
        result = 1e6 * z[0] ** 2 + np.sum(z[1:] ** 2) if len(z) > 1 else z[0] ** 2
        return float(result)


class F12_BentCigar(BBOBFunction):
    """F12: Bent cigar — f(x) = z_0^2 + 10^6 * sum(z_i^2) with asymmetric transform"""
    def __init__(self, dim, shift=None, rotation=None):
        super().__init__(dim, 12, shift=shift, rotation=rotation, optimum=0.0)

    def _evaluate(self, x):
        z = x - self.shift
        z = self._rotate(z)
        z = _T_asy(z, 0.5)
        z = self._rotate(z) if self.rotation_inv is not None else z
        result = z[0] ** 2 + 1e6 * np.sum(z[1:] ** 2) if len(z) > 1 else z[0] ** 2
        return float(result)


class F13_SharpRidge(BBOBFunction):
    """F13: Sharp ridge — f(x) = z_0^2 + 100*sqrt(sum(z_i^2, i>=2))"""
    def __init__(self, dim, shift=None, rotation=None):
        super().__init__(dim, 13, shift=shift, rotation=rotation, optimum=0.0)

    def _evaluate(self, x):
        z = x - self.shift
        z = self._rotate(z)
        z = _T_lambda(z, 10.0)
        z = self._rotate(z) if self.rotation_inv is not None else z
        n = len(z)
        if n <= 2:
            return float(z[0] ** 2 + 100.0 * np.sqrt(np.sum(z[1:] ** 2)) if n > 1 else z[0] ** 2)
        ridge = 100.0 * np.sqrt(np.sum(z[2:] ** 2))
        return float(z[0] ** 2 + ridge)


class F14_SumPowers(BBOBFunction):
    """F14: Sum of different powers — f(x) = sum(|z_i|^(2+4*i/(n-1)))"""
    def __init__(self, dim, shift=None, rotation=None):
        super().__init__(dim, 14, shift=shift, rotation=rotation, optimum=0.0)

    def _evaluate(self, x):
        z = x - self.shift
        z = self._rotate(z)
        z = _T_osz(z)
        n = len(z)
        if n == 1:
            return float(np.abs(z[0]) ** 2)
        exponents = 2.0 + 4.0 * np.arange(n) / (n - 1)
        return float(np.sum(np.abs(z) ** exponents))


class F15_RastriginRotated(BBOBFunction):
    """F15: Rastrigin, moderately conditioned — f(x) = 10*(n-sum(cos(2*pi*z_i))) + sum(z_i^2)"""
    def __init__(self, dim, shift=None, rotation=None, rotation2=None):
        super().__init__(dim, 15, shift=shift, rotation=rotation, rotation2=rotation2, optimum=0.0)

    def _evaluate(self, x):
        z = x - self.shift
        z = _T_lambda(z, 10.0)
        z = self._rotate(z, self.rotation)
        z = _T_osz(z)
        z = _T_asy(z, 0.2)
        z = self._rotate(z, self.rotation2) if self.rotation2 is not None else z
        n = len(z)
        return float(10.0 * (n - np.sum(np.cos(2.0 * np.pi * z))) + np.sum(z ** 2))


class F16_Weierstrass(BBOBFunction):
    """F16: Weierstrass — f(x) = sum(sum(0.5^k*cos(2*pi*3^k*(z_i+0.5)))) - n*f0"""
    def __init__(self, dim, shift=None, rotation=None, rotation2=None):
        super().__init__(dim, 16, shift=shift, rotation=rotation, rotation2=rotation2, optimum=0.0)
        self.a = 0.5
        self.b = 3.0
        self.kmax = 20

    def _evaluate(self, x):
        z = x - self.shift
        z = _T_lambda(z, 0.01)
        z = self._rotate(z, self.rotation)
        z = _T_osz(z)
        z = _T_asy(z, 0.1)
        z = self._rotate(z, self.rotation2) if self.rotation2 is not None else z
        n = len(z)
        f0 = sum(self.a ** k * np.cos(np.pi * self.b ** k) for k in range(self.kmax + 1))
        result = 0.0
        for i in range(n):
            for k in range(self.kmax + 1):
                result += self.a ** k * np.cos(2.0 * np.pi * self.b ** k * (z[i] + 0.5))
        return float(result - n * f0)


class F17_SchaffersF7(BBOBFunction):
    """F17: Schaffers F7 — f(x) = (1/n*sum(s_i^2))^0.75 * (1+0.25*sin^2(50*s_i^0.2))"""
    def __init__(self, dim, shift=None, rotation=None, rotation2=None):
        super().__init__(dim, 17, shift=shift, rotation=rotation, rotation2=rotation2, optimum=0.0)

    def _evaluate(self, x):
        z = x - self.shift
        z = _T_asy(z, 0.5)
        z = self._rotate(z, self.rotation)
        z = _T_lambda(z, 10.0)
        z = self._rotate(z, self.rotation2) if self.rotation2 is not None else z
        n = len(z)
        if n < 2:
            return 0.0
        s = np.sqrt(z[:-1] ** 2 + z[1:] ** 2)
        avg_s2 = np.mean(s ** 2)
        result = (avg_s2 ** 0.75) * np.mean(1.0 + 0.25 * np.sin(50.0 * s ** 0.2) ** 2)
        return float(result)


class F18_SchaffersF7Ill(BBOBFunction):
    """F18: Schaffers F7, moderately ill-conditioned"""
    def __init__(self, dim, shift=None, rotation=None, rotation2=None):
        super().__init__(dim, 18, shift=shift, rotation=rotation, rotation2=rotation2, optimum=0.0)

    def _evaluate(self, x):
        z = x - self.shift
        z = _T_asy(z, 0.5)
        z = self._rotate(z, self.rotation)
        z = _T_lambda(z, 1000.0)
        z = self._rotate(z, self.rotation2) if self.rotation2 is not None else z
        n = len(z)
        if n < 2:
            return 0.0
        s = np.sqrt(z[:-1] ** 2 + z[1:] ** 2)
        avg_s2 = np.mean(s ** 2)
        result = (avg_s2 ** 0.75) * np.mean(1.0 + 0.25 * np.sin(50.0 * s ** 0.2) ** 2)
        return float(result)


class F19_GriewankRosenbrock(BBOBFunction):
    """F19: Composite Griewank-Rosenbrock (F8F2) — 10/(n-1)*sum(f8(z_i,z_{i+1})) + 1"""
    def __init__(self, dim, shift=None, rotation=None):
        super().__init__(dim, 19, shift=shift, rotation=rotation, optimum=0.0)

    def _evaluate(self, x):
        z = x - self.shift
        z = self._rotate(z)
        z = z * 2.5 + 4.5
        n = len(z)
        if n < 2:
            return 0.0
        f8_vals = np.zeros(n - 1)
        for i in range(n - 1):
            f8_vals[i] = 100.0 * (z[i + 1] - z[i] ** 2) ** 2 + (1.0 - z[i]) ** 2
        result = 10.0 / (n - 1) * np.sum(f8_vals)
        return float(10.0 * (result / (n - 1)) + 1.0)


class F20_Schwefel(BBOBFunction):
    """F20: Schwefel — f(x) = -sum(x_i*sin(sqrt(|x_i|))) + 418.9829*n
       with sign alternation for multi-modal structure."""
    def __init__(self, dim, shift=None, rotation=None):
        super().__init__(dim, 20, shift=shift, rotation=rotation, optimum=418.9829 * dim)

    def _evaluate(self, x):
        z = x - self.shift
        n = len(z)
        # Sign alternation: multiply even indices by -1
        signs = np.ones(n)
        signs[::2] = -1.0
        z = z * signs
        # Schwefel transformation
        z = 4.209687462275036e2 + z
        # Boundary penalty
        result = 0.0
        for i in range(n):
            if np.abs(z[i]) > 500:
                result += 0.01 * (np.abs(z[i]) - 500) ** 2
            else:
                result += z[i] * np.sin(np.sqrt(np.abs(z[i])))
        return float(418.9829 * n - result)


class F21_GallagherGauss21hi(BBOBFunction):
    """F21: Gallagher's Gaussian 21-hi peaks (TEST) — 10 local optima."""
    def __init__(self, dim, shift=None, rotation=None):
        super().__init__(dim, 21, shift=shift, rotation=rotation, optimum=0.0)
        self.rng = np.random.RandomState(42 + 2100)
        self.n_peaks = 21
        self._build_peaks()

    def _build_peaks(self):
        n = self.dim
        # Generate peak centers
        self.centers = self.rng.uniform(-5.0, 5.0, (self.n_peaks, n))
        self.centers[0] = np.zeros(n)  # Global optimum at origin
        # Generate widths (alpha values)
        self.alphas = np.zeros(self.n_peaks)
        self.alphas[0] = 1000.0  # Global optimum is narrow
        for i in range(1, self.n_peaks):
            self.alphas[i] = self.rng.uniform(10.0, 1000.0)
        # Weights
        self.weights = np.array([1.1 ** (0.5 * (i - 10)) for i in range(self.n_peaks)])

    def _evaluate(self, x):
        z = x - self.shift
        n = len(z)
        # Find maximum over all peaks
        max_val = -np.inf
        for i in range(self.n_peaks):
            diff = z - self.centers[i]
            val = self.weights[i] * np.exp(-0.5 * np.sum(self.alphas[i] * diff ** 2) / n)
            if val > max_val:
                max_val = val
        return float(10.0 - max_val)


class F22_GallagherGauss21lo(BBOBFunction):
    """F22: Gallagher's Gaussian 21-lo peaks — 100 local optima."""
    def __init__(self, dim, shift=None, rotation=None):
        super().__init__(dim, 22, shift=shift, rotation=rotation, optimum=0.0)
        self.rng = np.random.RandomState(42 + 2200)
        self.n_peaks = 21
        self._build_peaks()

    def _build_peaks(self):
        n = self.dim
        self.centers = self.rng.uniform(-5.0, 5.0, (self.n_peaks, n))
        self.centers[0] = np.zeros(n)
        self.alphas = np.zeros(self.n_peaks)
        self.alphas[0] = 1000.0
        for i in range(1, self.n_peaks):
            self.alphas[i] = self.rng.uniform(1.0, 100.0)
        self.weights = np.array([1.1 ** (0.5 * (i - 10)) for i in range(self.n_peaks)])

    def _evaluate(self, x):
        z = x - self.shift
        n = len(z)
        max_val = -np.inf
        for i in range(self.n_peaks):
            diff = z - self.centers[i]
            val = self.weights[i] * np.exp(-0.5 * np.sum(self.alphas[i] * diff ** 2) / n)
            if val > max_val:
                max_val = val
        return float(10.0 - max_val)


class F23_Katsuura(BBOBFunction):
    """F23: Katsuura — f(x) = prod(1 + i*sum(floor(2^j*x_ij)*2^(-j)))^(10/n^1.2)"""
    def __init__(self, dim, shift=None, rotation=None, rotation2=None):
        super().__init__(dim, 23, shift=shift, rotation=rotation, rotation2=rotation2, optimum=0.0)

    def _evaluate(self, x):
        z = x - self.shift
        z = self._rotate(z, self.rotation)
        z = _T_lambda(z, 100.0)
        z = self._rotate(z, self.rotation2) if self.rotation2 is not None else z
        n = len(z)
        D = 32  # Number of terms in sum
        result = 1.0
        for i in range(n):
            s = 0.0
            for j in range(1, D + 1):
                s += np.abs(2.0 ** j * z[i] - np.round(2.0 ** j * z[i])) * 2.0 ** (-j)
            result *= (1.0 + (i + 1) * s) ** (10.0 / n ** 1.2)
        return float(result - 1.0)


class F24_LunacekBiRastrigin(BBOBFunction):
    """F24: Lunacek bi-Rastrigin — double Rastrigin structure."""
    def __init__(self, dim, shift=None, rotation=None, rotation2=None):
        super().__init__(dim, 24, shift=shift, rotation=rotation, rotation2=rotation2, optimum=0.0)
        self.rng = np.random.RandomState(42 + 2400)
        self.mu0 = 2.5
        self.s = 1.0 - 0.5 / (np.sqrt(dim + 20.0) - 4.1)
        self.d = 1.0
        self.mu1 = -np.sqrt((self.mu0 ** 2 - self.d) / self.s)

    def _evaluate(self, x):
        z = x - self.shift
        n = len(z)
        # Sign alternation
        signs = np.ones(n)
        self.rng_state = np.random.RandomState(42 + 2400)
        for i in range(n):
            if self.rng_state.rand() < 0.5:
                signs[i] = -1.0
        z = z * signs
        z = z * 0.1  # Standard scaling
        z = self._rotate(z, self.rotation)
        # Double Rastrigin
        x_hat = 2.0 * np.ones(n) * np.sign(self.mu0)
        z = z - x_hat
        s1 = self.mu0 ** 2 * n
        s2 = self.d * n + self.s * np.sum((z - self.mu1) ** 2)
        s3 = np.sum((z - self.mu0) ** 2 - np.cos(2.0 * np.pi * (z - self.mu0)))
        return float(min(s1, s2) + 10.0 * s3)


class BBOBSuite:
    """
    Full 24-function BBOB benchmark suite with paper-specified train/test split.

    Train (16): F1, F2, F4, F5, F6, F7, F10, F11, F12, F15, F16, F17, F19, F22, F23, F24
    Test  (8):  F3, F8, F9, F13, F14, F18, F20, F21
    """

    FUNCTION_CLASSES = {
        1:  F1_Sphere,
        2:  F2_Ellipsoidal,
        3:  F3_Rastrigin,
        4:  F4_BuecheRastrigin,
        5:  F5_LinearSlope,
        6:  F6_AttractiveSector,
        7:  F7_StepEllipsoidal,
        8:  F8_Rosenbrock,
        9:  F9_RosenbrockRotated,
        10: F10_EllipsoidalRotated,
        11: F11_Discus,
        12: F12_BentCigar,
        13: F13_SharpRidge,
        14: F14_SumPowers,
        15: F15_RastriginRotated,
        16: F16_Weierstrass,
        17: F17_SchaffersF7,
        18: F18_SchaffersF7Ill,
        19: F19_GriewankRosenbrock,
        20: F20_Schwefel,
        21: F21_GallagherGauss21hi,
        22: F22_GallagherGauss21lo,
        23: F23_Katsuura,
        24: F24_LunacekBiRastrigin,
    }

    TRAIN_IDS = [1, 2, 4, 5, 6, 7, 10, 11, 12, 15, 16, 17, 19, 22, 23, 24]  # 16 functions
    TEST_IDS  = [3, 8, 9, 13, 14, 18, 20, 21]  # 8 functions

    FUNCTION_GROUPS = {
        'Separable':         [1, 2, 3, 4, 5, 6],
        'Low_conditioning':  [7, 8, 9, 10, 11, 12],
        'High_conditioning': [13, 14, 15, 16, 17],
        'Multi_modal':       [18, 19, 20, 21, 22, 23, 24],
    }

    def __init__(
        self,
        dim: int = 5,
        n_train_instances: int = 1,
        n_test_instances: int = 1,
        seed: int = 42
    ):
        self.dim = dim
        self.n_train_instances = n_train_instances
        self.n_test_instances = n_test_instances
        self.seed = seed

        self._train_fns = {}
        self._test_fns = {}
        self._build_functions()

    def _make_rotation(self, dim: int, seed: int) -> np.ndarray:
        rng = np.random.RandomState(seed)
        return _rotation_matrix(dim, rng)

    def _build_functions(self):
        """Build train and test function instances with proper rotations/shifts."""
        for fid in self.TRAIN_IDS:
            self._train_fns[fid] = []
            for inst in range(self.n_train_instances):
                seed_shift = self.seed + fid * 1000 + inst
                rng = np.random.RandomState(seed_shift)
                shift = rng.uniform(-4.0, 4.0, self.dim)
                rot = self._make_rotation(self.dim, seed_shift + 1)
                rot2 = self._make_rotation(self.dim, seed_shift + 2)
                fn_class = self.FUNCTION_CLASSES[fid]
                # Some functions take rotation2, some don't
                try:
                    fn = fn_class(self.dim, shift=shift, rotation=rot, rotation2=rot2)
                except TypeError:
                    fn = fn_class(self.dim, shift=shift, rotation=rot)
                self._train_fns[fid].append(fn)

        for fid in self.TEST_IDS:
            self._test_fns[fid] = []
            for inst in range(self.n_test_instances):
                seed_shift = self.seed + fid * 1000 + inst + 500
                rng = np.random.RandomState(seed_shift)
                shift = rng.uniform(-4.0, 4.0, self.dim)
                rot = self._make_rotation(self.dim, seed_shift + 1)
                rot2 = self._make_rotation(self.dim, seed_shift + 2)
                fn_class = self.FUNCTION_CLASSES[fid]
                try:
                    fn = fn_class(self.dim, shift=shift, rotation=rot, rotation2=rot2)
                except TypeError:
                    fn = fn_class(self.dim, shift=shift, rotation=rot)
                self._test_fns[fid].append(fn)

    def get_train_functions(self) -> List[Tuple[str, Callable]]:
        """Get all training functions as (name, function) pairs."""
        result = []
        for fid in self.TRAIN_IDS:
            for inst, fn in enumerate(self._train_fns[fid]):
                result.append((f"F{fid}_{inst}", fn))
        return result

    def get_test_functions(self) -> List[Tuple[str, Callable]]:
        """Get all testing functions as (name, function) pairs."""
        result = []
        for fid in self.TEST_IDS:
            for inst, fn in enumerate(self._test_fns[fid]):
                result.append((f"F{fid}_{inst}", fn))
        return result

    def get_bounds(self, fid) -> np.ndarray:
        """Get recommended bounds by function ID (int) or name (str like 'F20')."""
        if isinstance(fid, str):
            fid = int(fid.lstrip('Ff').split('_')[0])
        bounds_map = {
            1:  (-5.0, 5.0),
            2:  (-5.0, 5.0),
            3:  (-5.0, 5.0),
            4:  (-5.0, 5.0),
            5:  (-5.0, 5.0),
            6:  (-5.0, 5.0),
            7:  (-5.0, 5.0),
            8:  (-5.0, 5.0),
            9:  (-5.0, 5.0),
            10: (-5.0, 5.0),
            11: (-5.0, 5.0),
            12: (-5.0, 5.0),
            13: (-5.0, 5.0),
            14: (-5.0, 5.0),
            15: (-5.0, 5.0),
            16: (-5.0, 5.0),
            17: (-5.0, 5.0),
            18: (-5.0, 5.0),
            19: (-5.0, 5.0),
            20: (-500.0, 500.0),  # Schwefel has larger range
            21: (-5.0, 5.0),
            22: (-5.0, 5.0),
            23: (-5.0, 5.0),
            24: (-5.0, 5.0),
        }
        lo, hi = bounds_map.get(fid, (-5.0, 5.0))
        return np.array([[lo, hi]] * self.dim)

    def get_bounds_by_name(self, name: str) -> np.ndarray:
        """Get bounds by function name (e.g., 'F20' or 'F20_0')."""
        fid = int(name.split('_')[0][1:])
        return self.get_bounds(fid)

    def __len__(self) -> int:
        return 24

    def function_names(self) -> List[str]:
        return [f"F{i}" for i in range(1, 25)]

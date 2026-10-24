"""Operator-learning datasets from Lu et al. (DeepONet), Section 4.

Input functions u are sampled from a mean-zero Gaussian random field (GRF) with RBF kernel
k(x1, x2) = exp(-||x1 - x2||^2 / (2 l^2)) on [0, 1]. Each u is represented by its values at m
equispaced sensors; outputs are G(u)(y) at random locations y.

Implemented operators
* ``antiderivative``: ds/dx = u(x), s(0) = 0                     (Section 4.1.1, linear case)
* ``nonlinear_ode``:  ds/dx = -s(x)^2 + u(x), s(0) = 0           (Section 4.1.1, nonlinear case)
* ``pendulum``:       ds1/dt = s2, ds2/dt = -k sin(s1) + u(t)    (Section 4.1.2, k = 1)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def grf_samples(n: int, length_scale: float, n_grid: int, rng: np.random.Generator, T: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Sample n GRF functions on a fine grid via Cholesky of the RBF covariance."""
    x = np.linspace(0.0, T, n_grid)
    K = np.exp(-0.5 * (x[:, None] - x[None, :]) ** 2 / length_scale**2)
    L = np.linalg.cholesky(K + 1e-10 * np.eye(n_grid) * n_grid)
    return x, (L @ rng.standard_normal((n_grid, n))).T  # [n, n_grid]


def _rk4(f, s0: np.ndarray, x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Integrate ds/dx = f(s, u(x)) on grid x with u linearly interpolated at half steps."""
    s = np.zeros((len(x),) + s0.shape)
    s[0] = s0
    for i in range(len(x) - 1):
        h = x[i + 1] - x[i]
        um = 0.5 * (u[i] + u[i + 1])
        k1 = f(s[i], u[i])
        k2 = f(s[i] + 0.5 * h * k1, um)
        k3 = f(s[i] + 0.5 * h * k2, um)
        k4 = f(s[i] + h * k3, u[i + 1])
        s[i + 1] = s[i] + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return s


def solve(operator: str, x: np.ndarray, u: np.ndarray, k: float = 1.0) -> np.ndarray:
    """Return G(u) evaluated on the fine grid x (first state component for the pendulum)."""
    if operator == "antiderivative":
        return np.concatenate([[0.0], np.cumsum(0.5 * (u[1:] + u[:-1]) * np.diff(x))])
    if operator == "nonlinear_ode":
        return _rk4(lambda s, ui: -s**2 + ui, np.zeros(()), x, u)
    if operator == "pendulum":
        return _rk4(lambda s, ui: np.array([s[1], -k * np.sin(s[0]) + ui]), np.zeros(2), x, u)[:, 0]
    raise ValueError(f"unknown operator {operator!r}")


@dataclass
class OperatorData:
    u_sensors: np.ndarray  # [N, m]
    y: np.ndarray          # [N, 1]
    s: np.ndarray          # [N, 1]


def make_dataset(operator: str, n_functions: int, points_per_function: int, m: int = 100,
                 length_scale: float = 0.2, n_grid: int = 1001, T: float = 1.0, seed: int = 0) -> OperatorData:
    """Build (u(x_1..x_m), y, G(u)(y)) triplets; P random output locations per input function."""
    rng = np.random.default_rng(seed)
    x, U = grf_samples(n_functions, length_scale, n_grid, rng, T)
    sensors = np.linspace(0.0, T, m)
    us, ys, ss = [], [], []
    for u in U:
        sol = solve(operator, x, u)
        yq = rng.uniform(0.0, T, points_per_function)
        us.append(np.repeat(np.interp(sensors, x, u)[None], points_per_function, 0))
        ys.append(yq[:, None])
        ss.append(np.interp(yq, x, sol)[:, None])
    return OperatorData(np.concatenate(us).astype(np.float32), np.concatenate(ys).astype(np.float32),
                        np.concatenate(ss).astype(np.float32))

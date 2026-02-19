# bandits_project/algos/etc_bound.py
from __future__ import annotations
import numpy as np


def etc_regret_bound(m: int, deltas: np.ndarray, n: int, K: int, sigma: float = 1.0) -> float:
    """
    Theorem 1.3.2 bound:
      B(m) = m * sum_a Δ_a + (n - mK) * sum_a Δ_a * exp( - m Δ_a^2 / (4σ^2) )
    Constraint: mK <= n
    """
    if m < 0:
        raise ValueError("m must be non-negative.")
    if m * K > n:
        return float("inf")

    deltas = np.asarray(deltas, dtype=float)
    sum_d = float(np.sum(deltas))
    exp_term = np.exp(-(m * deltas**2) / (4.0 * sigma**2))
    sum_exp = float(np.sum(deltas * exp_term))
    return m * sum_d + (n - m * K) * sum_exp


def optimal_m_by_bound(deltas: np.ndarray, n: int, K: int, sigma: float = 1.0) -> int:
    """
    Minimize the bound over integer m in {0,1,..., floor(n/K)}.
    Returns argmin m.
    """
    m_max = n // K
    best_m = 0
    best_val = float("inf")

    for m in range(m_max + 1):
        val = etc_regret_bound(m, deltas=deltas, n=n, K=K, sigma=sigma)
        if val < best_val:
            best_val = val
            best_m = m

    return best_m
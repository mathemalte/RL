# bandits_project/algos/ucb.py
from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence
import numpy as np

PullFn = Callable[[int], float]


def _random_argmax(values: np.ndarray, rng: np.random.Generator) -> int:
    """Argmax with uniform tie-breaking."""
    m = np.max(values)
    idx = np.flatnonzero(values == m)
    return int(rng.choice(idx))


def _update_running_mean(q_hat: np.ndarray, counts: np.ndarray, a: int, x: float) -> None:
    """Qhat update: Q <- Q + (1/N) (x - Q). (Incremental mean)"""
    counts[a] += 1
    n = counts[a]
    q_hat[a] = q_hat[a] + (x - q_hat[a]) / n


def run_ucb_hoeffding(
    pull: PullFn,
    K: int,
    n_steps: int,
    *,
    delta: Optional[float] = None,
    q0: Optional[Sequence[float]] = None,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Standard UCB with parameter delta (Algorithm 4).  [oai_citation:3‡RL_Vorlesung.pdf](sediment://file_000000000a6c720aa8941aeeb29732ab)
      UCB_a(t,delta) = inf if T_a(t)=0 else Qhat_a(t) + sqrt( 2 log(1/delta) / T_a(t) )

    Lecture note suggests e.g. delta = 1/n^2 if horizon n is known.  [oai_citation:4‡RL_Vorlesung.pdf](sediment://file_000000000a6c720aa8941aeeb29732ab)
    If delta is None we default to delta = 1 / n_steps^2.
    """
    rng = np.random.default_rng(seed)

    if delta is None:
        delta = 1.0 / (n_steps * n_steps)
    if not (0.0 < float(delta) < 1.0):
        raise ValueError("delta must be in (0,1).")

    q_hat = np.zeros(K, dtype=float) if q0 is None else np.array(q0, dtype=float).copy()
    if q_hat.shape != (K,):
        raise ValueError(f"q0 must have shape ({K},), got {q_hat.shape}.")

    counts = np.zeros(K, dtype=int)
    actions = np.zeros(n_steps, dtype=int)
    rewards = np.zeros(n_steps, dtype=float)

    log_term = np.log(1.0 / float(delta))

    for t in range(1, n_steps + 1):
        # compute UCB using T_a(t-1) = counts
        ucb = np.empty(K, dtype=float)
        for a in range(K):
            if counts[a] == 0:
                ucb[a] = np.inf  # forces exploring each arm at least once  [oai_citation:5‡RL_Vorlesung.pdf](sediment://file_000000000a6c720aa8941aeeb29732ab)
            else:
                bonus = np.sqrt(2.0 * log_term / counts[a])
                ucb[a] = q_hat[a] + bonus

        a_t = _random_argmax(ucb, rng)
        x_t = float(pull(a_t))

        actions[t - 1] = a_t
        rewards[t - 1] = x_t
        _update_running_mean(q_hat, counts, a_t, x_t)

    return {"actions": actions, "rewards": rewards, "q_hat": q_hat, "counts": counts, "delta": np.array([delta])}


def run_ucb_subgaussian(
    pull: PullFn,
    K: int,
    n_steps: int,
    *,
    sigma: float,
    q0: Optional[Sequence[float]] = None,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    σ-subgaussian UCB version from the lecture:
      UCB_a(t) = inf if T_a(t)=0 else Qhat_a(t) + sqrt( 4 σ^2 log(n) / T_a(t) )  [oai_citation:6‡RL_Vorlesung.pdf](sediment://file_000000000a6c720aa8941aeeb29732ab)

    Note: uses log(n_steps) exactly as in the lecture statement.  [oai_citation:7‡RL_Vorlesung.pdf](sediment://file_000000000a6c720aa8941aeeb29732ab)
    """
    rng = np.random.default_rng(seed)
    sigma = float(sigma)
    if sigma <= 0:
        raise ValueError("sigma must be > 0.")

    q_hat = np.zeros(K, dtype=float) if q0 is None else np.array(q0, dtype=float).copy()
    if q_hat.shape != (K,):
        raise ValueError(f"q0 must have shape ({K},), got {q_hat.shape}.")

    counts = np.zeros(K, dtype=int)
    actions = np.zeros(n_steps, dtype=int)
    rewards = np.zeros(n_steps, dtype=float)

    logn = np.log(float(n_steps))

    for t in range(1, n_steps + 1):
        ucb = np.empty(K, dtype=float)
        for a in range(K):
            if counts[a] == 0:
                ucb[a] = np.inf
            else:
                bonus = np.sqrt((4.0 * sigma * sigma * logn) / counts[a])
                ucb[a] = q_hat[a] + bonus

        a_t = _random_argmax(ucb, rng)
        x_t = float(pull(a_t))

        actions[t - 1] = a_t
        rewards[t - 1] = x_t
        _update_running_mean(q_hat, counts, a_t, x_t)

    return {"actions": actions, "rewards": rewards, "q_hat": q_hat, "counts": counts, "sigma": np.array([sigma])}
# bandits_project/algos/boltzmann.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np

PullFn = Callable[[int], float]


def _update_running_mean(q_hat: np.ndarray, counts: np.ndarray, a: int, x: float) -> None:
    counts[a] += 1
    n = counts[a]
    q_hat[a] = q_hat[a] + (x - q_hat[a]) / n


def _random_argmax(values: np.ndarray, rng: np.random.Generator) -> int:
    m = np.max(values)
    idx = np.flatnonzero(values == m)
    return int(rng.choice(idx))


def _softmax_logits(logits: np.ndarray) -> np.ndarray:
    # stable softmax
    z = logits - np.max(logits)
    ez = np.exp(z)
    return ez / np.sum(ez)


def _sample_categorical(probs: np.ndarray, rng: np.random.Generator) -> int:
    # probs must sum to 1
    return int(rng.choice(len(probs), p=probs))


@dataclass(frozen=True)
class NoiseSpec:
    """
    Generic i.i.d. noise source for the 'argmax(Q + noise)' exploration family.

    name:
      - "gumbel": uses numpy's rng.gumbel(loc=0, scale=1)
      - otherwise: uses scipy.stats.<name>.rvs(...)

    kwargs:
      passed to scipy.stats distribution or to gumbel (loc/scale).
    """
    name: str
    kwargs: Dict[str, Any] = None

    def sample(self, size: int, rng: np.random.Generator) -> np.ndarray:
        name = self.name.lower()
        kw = {} if self.kwargs is None else dict(self.kwargs)

        if name == "gumbel":
            loc = float(kw.pop("loc", 0.0))
            scale = float(kw.pop("scale", 1.0))
            if kw:
                raise ValueError(f"Unused kwargs for gumbel: {kw}")
            return rng.gumbel(loc=loc, scale=scale, size=size).astype(float)

        # scipy.stats distributions
        try:
            import scipy.stats as st  # type: ignore
        except ImportError as e:
            raise ImportError(
                "scipy is required for non-gumbel noise distributions. "
                "Install it via: pip install scipy"
            ) from e

        if not hasattr(st, name):
            raise ValueError(f"Unknown scipy.stats distribution: {name}")

        dist = getattr(st, name)
        # scipy supports random_state as numpy Generator in recent versions;
        # otherwise it will still work with RandomState-like objects.
        return dist.rvs(size=size, random_state=rng, **kw).astype(float)


# -----------------------------
# 1) Simple Boltzmann (softmax)
# -----------------------------
def run_boltzmann_softmax(
    pull: PullFn,
    K: int,
    n_steps: int,
    *,
    theta: float,
    q0: Optional[Sequence[float]] = None,
    seed: Optional[int] = None,
    track_probs: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Algorithm 5: Simple Boltzmann exploration:
      Sample A_t ~ SM(theta, Q_hat(t-1)) and update Q_hat by incremental mean.  [oai_citation:2‡RL_Vorlesung.pdf](sediment://file_000000000a6c720aa8941aeeb29732ab)
    """
    theta = float(theta)
    if theta <= 0:
        raise ValueError("theta must be > 0 (inverse temperature).")

    rng = np.random.default_rng(seed)
    q_hat = np.zeros(K, dtype=float) if q0 is None else np.array(q0, dtype=float).copy()
    if q_hat.shape != (K,):
        raise ValueError(f"q0 must have shape ({K},), got {q_hat.shape}.")

    counts = np.zeros(K, dtype=int)
    actions = np.zeros(n_steps, dtype=int)
    rewards = np.zeros(n_steps, dtype=float)
    probs_hist = np.zeros((n_steps, K), dtype=float) if track_probs else None

    for t in range(1, n_steps + 1):
        probs = _softmax_logits(theta * q_hat)
        if track_probs:
            probs_hist[t - 1] = probs
        a = _sample_categorical(probs, rng)
        x = float(pull(a))

        actions[t - 1] = a
        rewards[t - 1] = x
        _update_running_mean(q_hat, counts, a, x)

    out = {"actions": actions, "rewards": rewards, "q_hat": q_hat, "counts": counts, "theta": np.array([theta])}
    if track_probs:
        out["probs"] = probs_hist
    return out


# -----------------------------------------
# 2) Boltzmann via Gumbel trick (argmax)
# -----------------------------------------
def run_boltzmann_gumbel_trick(
    pull: PullFn,
    K: int,
    n_steps: int,
    *,
    theta: float,
    q0: Optional[Sequence[float]] = None,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Uses the lemma:
      SM(theta, x) (d)= argmax_a { theta*x_a + g_a }
    where g_a are iid standard Gumbel. 

    Equivalent form:
      argmax_a { Q_hat_a + g_a/theta }  [oai_citation:3‡RL_Vorlesung.pdf](sediment://file_000000000a6c720aa8941aeeb29732ab)
    """
    theta = float(theta)
    if theta <= 0:
        raise ValueError("theta must be > 0.")

    rng = np.random.default_rng(seed)
    q_hat = np.zeros(K, dtype=float) if q0 is None else np.array(q0, dtype=float).copy()
    if q_hat.shape != (K,):
        raise ValueError(f"q0 must have shape ({K},), got {q_hat.shape}.")

    counts = np.zeros(K, dtype=int)
    actions = np.zeros(n_steps, dtype=int)
    rewards = np.zeros(n_steps, dtype=float)

    for t in range(1, n_steps + 1):
        g = rng.gumbel(loc=0.0, scale=1.0, size=K)
        scores = q_hat + (g / theta)
        a = _random_argmax(scores, rng)
        x = float(pull(a))

        actions[t - 1] = a
        rewards[t - 1] = x
        _update_running_mean(q_hat, counts, a, x)

    return {"actions": actions, "rewards": rewards, "q_hat": q_hat, "counts": counts, "theta": np.array([theta])}


# ----------------------------------------------------
# 3) Argmax with arbitrary i.i.d. noise distribution
# ----------------------------------------------------
def run_argmax_with_noise(
    pull: PullFn,
    K: int,
    n_steps: int,
    *,
    q0: Optional[Sequence[float]] = None,
    seed: Optional[int] = None,
    base_scale: float = 1.0,
    noise: Union[NoiseSpec, str] = NoiseSpec("gumbel"),
    noise_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, np.ndarray]:
    """
    'Generalised Gumbel trick' (not softmax anymore):
      A_t = argmax_a { Q_hat_a(t-1) + base_scale * Z_a }
    where Z_a are i.i.d. from a chosen distribution.

    The lecture explicitly suggests experimenting with replacing Gumbel by other distributions
    (e.g. non-negative ones).  [oai_citation:4‡RL_Vorlesung.pdf](sediment://file_000000000a6c720aa8941aeeb29732ab)

    Examples:
      noise="cauchy"
      noise="beta", noise_kwargs={"a": 2.0, "b": 2.0}
      noise="betaprime", noise_kwargs={"a": 2.0, "b": 3.0}
      noise="chi", noise_kwargs={"df": 3.0}
    """
    rng = np.random.default_rng(seed)
    base_scale = float(base_scale)
    if base_scale <= 0:
        raise ValueError("base_scale must be > 0.")

    if isinstance(noise, str):
        noise = NoiseSpec(noise, kwargs=noise_kwargs or {})
    else:
        # allow overriding kwargs
        if noise_kwargs is not None:
            noise = NoiseSpec(noise.name, kwargs=noise_kwargs)

    q_hat = np.zeros(K, dtype=float) if q0 is None else np.array(q0, dtype=float).copy()
    if q_hat.shape != (K,):
        raise ValueError(f"q0 must have shape ({K},), got {q_hat.shape}.")

    counts = np.zeros(K, dtype=int)
    actions = np.zeros(n_steps, dtype=int)
    rewards = np.zeros(n_steps, dtype=float)

    for t in range(1, n_steps + 1):
        z = noise.sample(size=K, rng=rng)
        scores = q_hat + base_scale * z
        a = _random_argmax(scores, rng)
        x = float(pull(a))

        actions[t - 1] = a
        rewards[t - 1] = x
        _update_running_mean(q_hat, counts, a, x)

    return {
        "actions": actions,
        "rewards": rewards,
        "q_hat": q_hat,
        "counts": counts,
        "noise_name": np.array([noise.name]),
        "base_scale": np.array([base_scale]),
    }


# ---------------------------------------------------------
# 4) UCB-like Gumbel bonus: Q_hat + sqrt(C/T_a) * Z_a
# ---------------------------------------------------------
def run_gumbel_ucb_style(
    pull: PullFn,
    K: int,
    n_steps: int,
    *,
    C: float,
    q0: Optional[Sequence[float]] = None,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Implements:
      A_t = argmax_a { Q_hat_a(t-1) + sqrt(C / T_a(t-1)) * Z_a }
    where Z_a are i.i.d. standard Gumbel.

    Motivation appears in the lecture right after the Gumbel trick:
      theta^{-1} should be arm-dependent and sqrt(C/T_a) 'might be a good idea'. 

    Convention: if T_a(t-1)=0, score is +inf to force at least one pull.
    """
    rng = np.random.default_rng(seed)
    C = float(C)
    if C <= 0:
        raise ValueError("C must be > 0.")

    q_hat = np.zeros(K, dtype=float) if q0 is None else np.array(q0, dtype=float).copy()
    if q_hat.shape != (K,):
        raise ValueError(f"q0 must have shape ({K},), got {q_hat.shape}.")

    counts = np.zeros(K, dtype=int)
    actions = np.zeros(n_steps, dtype=int)
    rewards = np.zeros(n_steps, dtype=float)

    for t in range(1, n_steps + 1):
        z = rng.gumbel(loc=0.0, scale=1.0, size=K)
        scores = np.empty(K, dtype=float)
        for a in range(K):
            if counts[a] == 0:
                scores[a] = np.inf
            else:
                bonus = np.sqrt(C / counts[a]) * z[a]
                scores[a] = q_hat[a] + bonus

        a_t = _random_argmax(scores, rng)
        x_t = float(pull(a_t))

        actions[t - 1] = a_t
        rewards[t - 1] = x_t
        _update_running_mean(q_hat, counts, a_t, x_t)

    return {"actions": actions, "rewards": rewards, "q_hat": q_hat, "counts": counts, "C": np.array([C])}
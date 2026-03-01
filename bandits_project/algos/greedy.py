# bandits_project/algos/greedy.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


PullFn = Callable[[int], float] # function that takes arm index and returns reward (stochastic bandit environment)
EpsSchedule = Callable[[int], float]  # input: t (1-based), output: epsilon_t in [0,1]


def _random_argmax(values: np.ndarray, rng: np.random.Generator) -> int:
    """Argmax with uniform tie-breaking.""" # if multiple arms have the same max value, choose among them uniformly at random
    m = np.max(values)
    candidates = np.flatnonzero(values == m)
    return int(rng.choice(candidates))


def _update_running_mean(q_hat: np.ndarray, counts: np.ndarray, a: int, x: float) -> None:
    """
    Incremental mean update:
      Q_n = Q_{n-1} + (1/n) (R_n - Q_{n-1})
    See lecture equation (1.3).  [oai_citation:4‡RL_Vorlesung.pdf](sediment://file_000000000a6c720aa8941aeeb29732ab)
    """
    counts[a] += 1
    n = counts[a] #zählt wie oft arm a schon gespielt wurde
    q_hat[a] = q_hat[a] + (x - q_hat[a]) / n #q_hat ist die aktuelle schätzung des mittelwerts von arm a


@dataclass(frozen=True)
class DecreasingEpsilonByBound:
    """
    Lecture's decreasing exploration rate:
      epsilon_t = min{1, C*K / (d^2 * t)}   [oai_citation:5‡RL_Vorlesung.pdf](sediment://file_000000000a6c720aa8941aeeb29732ab)

    Notes:
    - t is 1-based time index.
    - You must provide K, C, d (with d < min gap, as in the theorem statement).
    """
    K: int
    C: float #constant
    d: float #gap parameter

    def __call__(self, t: int) -> float: #macht aus der Formel eine funktion die man benutzen kann
        if t <= 0:
            raise ValueError("t must be >= 1 (1-based).")
        eps = (self.C * self.K) / (self.d * self.d * t)
        return float(min(1.0, max(0.0, eps))) #epsilon_t muss in [0,1] liegen


def run_pure_greedy(
    pull: PullFn,
    K: int,
    n_steps: int,
    *,
    q0: Optional[Sequence[float]] = None,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Pure greedy bandit algorithm (Algorithm 2).  [oai_citation:6‡RL_Vorlesung.pdf](sediment://file_000000000a6c720aa8941aeeb29732ab)
      - At = argmax_a Q_hat[a] with random tie-break
      - Update Q_hat by incremental mean

    Returns dict with:
      actions (n_steps,), rewards (n_steps,), q_hat (K,), counts (K,)
    """
    rng = np.random.default_rng(seed)

    q_hat = np.zeros(K, dtype=float) if q0 is None else np.array(q0, dtype=float).copy()
    if q_hat.shape != (K,):
        raise ValueError(f"q0 must have shape ({K},), got {q_hat.shape}.")

    counts = np.zeros(K, dtype=int)
    actions = np.zeros(n_steps, dtype=int)
    rewards = np.zeros(n_steps, dtype=float)

    for t in range(1, n_steps + 1):
        a = _random_argmax(q_hat, rng) #wähle den arm mit dem höchsten geschätzten Mittelwert, bei Gleichstand zufällig
        x = float(pull(a)) #hole den reward für den gezogenen arm a

        actions[t - 1] = a
        rewards[t - 1] = x
        _update_running_mean(q_hat, counts, a, x) #aktualisiere die Schätzung des Mittelwerts für arm a basierend auf dem neuen reward x

    return {"actions": actions, "rewards": rewards, "q_hat": q_hat, "counts": counts}


def run_epsilon_greedy(
    pull: PullFn,
    K: int,
    n_steps: int,
    *,
    epsilon: float,
    q0: Optional[Sequence[float]] = None,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    ε-greedy with fixed epsilon (Algorithm 3).  [oai_citation:7‡RL_Vorlesung.pdf](sediment://file_000000000a6c720aa8941aeeb29732ab)
      - With prob ε: explore uniformly random arm
      - With prob 1-ε: exploit argmax Q_hat
      - Update Q_hat by incremental mean

    Returns dict with:
      actions, rewards, q_hat, counts
    """
    if not (0.0 <= epsilon <= 1.0):
        raise ValueError("epsilon must be in [0,1].")

    rng = np.random.default_rng(seed)

    q_hat = np.zeros(K, dtype=float) if q0 is None else np.array(q0, dtype=float).copy()
    if q_hat.shape != (K,):
        raise ValueError(f"q0 must have shape ({K},), got {q_hat.shape}.")

    counts = np.zeros(K, dtype=int)
    actions = np.zeros(n_steps, dtype=int)
    rewards = np.zeros(n_steps, dtype=float)

    for t in range(1, n_steps + 1):
        u = rng.random() #ziehe u aus uniform(0,1)
        if u < epsilon: #dann wähle einen arm uniform zufällig
            a = int(rng.integers(low=0, high=K))  # uniform exploration
        else:
            a = _random_argmax(q_hat, rng)        # greedy exploitation

        x = float(pull(a))

        actions[t - 1] = a
        rewards[t - 1] = x
        _update_running_mean(q_hat, counts, a, x)

    return {"actions": actions, "rewards": rewards, "q_hat": q_hat, "counts": counts}


def run_epsilon_greedy_decreasing(
    pull: PullFn,
    K: int,
    n_steps: int,
    *,
    epsilon_t: Union[EpsSchedule, DecreasingEpsilonByBound],
    q0: Optional[Sequence[float]] = None,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    ε-greedy with time-dependent epsilon_t.
    In the lecture, an example is ε_t = min{1, C*K/(d^2 t)}.  [oai_citation:8‡RL_Vorlesung.pdf](sediment://file_000000000a6c720aa8941aeeb29732ab)

      - At time t (1-based):
          with prob ε_t(t): explore uniformly
          else: exploit argmax Q_hat
      - Update Q_hat by incremental mean

    Returns dict with:
      actions, rewards, q_hat, counts, epsilons
    """
    rng = np.random.default_rng(seed)

    q_hat = np.zeros(K, dtype=float) if q0 is None else np.array(q0, dtype=float).copy()
    if q_hat.shape != (K,):
        raise ValueError(f"q0 must have shape ({K},), got {q_hat.shape}.")

    counts = np.zeros(K, dtype=int)
    actions = np.zeros(n_steps, dtype=int)
    rewards = np.zeros(n_steps, dtype=float)
    epsilons = np.zeros(n_steps, dtype=float)

    for t in range(1, n_steps + 1):
        eps = float(epsilon_t(t))
        eps = min(1.0, max(0.0, eps))
        epsilons[t - 1] = eps

        u = rng.random()
        if u < eps:
            a = int(rng.integers(low=0, high=K))
        else:
            a = _random_argmax(q_hat, rng)

        x = float(pull(a))

        actions[t - 1] = a
        rewards[t - 1] = x
        _update_running_mean(q_hat, counts, a, x)

    return {
        "actions": actions,
        "rewards": rewards,
        "q_hat": q_hat,
        "counts": counts,
        "epsilons": epsilons,
    }
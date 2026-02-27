# bandits_project/algos/policy_gradient.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Union

import numpy as np

PullFn = Callable[[int], float]
StepSchedule = Callable[[int], float]  # input: t (1-based), output: alpha_t


def _softmax(theta: np.ndarray) -> np.ndarray:
    # stable softmax
    z = theta - np.max(theta)
    ez = np.exp(z)
    return ez / np.sum(ez)


def _sample_from_probs(probs: np.ndarray, rng: np.random.Generator) -> int:
    return int(rng.choice(len(probs), p=probs))


@dataclass(frozen=True)
class ConstantStepsize:
    alpha: float
    def __call__(self, t: int) -> float:
        return float(self.alpha)


@dataclass(frozen=True)
class DecayingStepsizeSqrt:
    alpha0: float
    def __call__(self, t: int) -> float:
        # alpha_t = alpha0 / sqrt(t)
        return float(self.alpha0 / np.sqrt(t))


def run_policy_gradient(
    pull: PullFn,
    K: int,
    n_steps: int,
    *,
    stepsize: Union[float, StepSchedule] = 0.1,
    baseline: bool = False,
    theta0: Optional[Sequence[float]] = None,
    seed: Optional[int] = None,
    track_probs: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Policy Gradient (REINFORCE) for stochastic bandits using softmax policy:
        pi_theta(a) = exp(theta_a) / sum_b exp(theta_b)

    Update at time t (1-based):
      - sample A_t ~ pi_theta
      - observe reward R_t
      - grad log pi(A_t) = e_{A_t} - pi_theta
      - theta <- theta + alpha_t * (R_t - b_t) * (e_{A_t} - pi_theta)
        where baseline b_t is optional (running mean reward).

    Returns:
      actions, rewards, theta (final), probs_hist (optional), baseline_hist (if baseline)
    """
    rng = np.random.default_rng(seed)

    # stepsize schedule
    if isinstance(stepsize, (int, float)):
        stepsize_fn: StepSchedule = ConstantStepsize(float(stepsize))
    else:
        stepsize_fn = stepsize

    theta = np.zeros(K, dtype=float) if theta0 is None else np.array(theta0, dtype=float).copy()
    if theta.shape != (K,):
        raise ValueError(f"theta0 must have shape ({K},), got {theta.shape}.")

    actions = np.zeros(n_steps, dtype=int)
    rewards = np.zeros(n_steps, dtype=float)
    probs_hist = np.zeros((n_steps, K), dtype=float) if track_probs else None

    # baseline as running mean of rewards (simple, common, variance reduction)
    b = 0.0
    baseline_hist = np.zeros(n_steps, dtype=float) if baseline else None

    for t in range(1, n_steps + 1):
        pi = _softmax(theta)
        if track_probs:
            probs_hist[t - 1] = pi

        a = _sample_from_probs(pi, rng)
        r = float(pull(a))

        actions[t - 1] = a
        rewards[t - 1] = r

        # baseline update (running mean)
        if baseline:
            # use baseline from previous steps for advantage; then update baseline with current reward
            b_old = b
            baseline_hist[t - 1] = b_old
            # incremental mean of rewards
            b = b_old + (r - b_old) / t
            adv = r - b_old
        else:
            adv = r

        alpha_t = float(stepsize_fn(t))
        if alpha_t <= 0:
            raise ValueError("stepsize must be > 0")

        # grad log pi(a) = e_a - pi
        grad_logp = -pi
        grad_logp = grad_logp.copy()
        grad_logp[a] += 1.0

        theta += alpha_t * adv * grad_logp

    out: Dict[str, np.ndarray] = {
        "actions": actions,
        "rewards": rewards,
        "theta": theta,
    }
    if track_probs:
        out["probs"] = probs_hist
    if baseline:
        out["baseline"] = baseline_hist
    return out
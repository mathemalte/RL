# bandits_project/bandits/stochastic_bandit.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence
import numpy as np

DistMode = Literal["bernoulli", "gaussian"]
MeanMode = Literal["manual", "random"]


@dataclass(frozen=True)
class StochasticBanditConfig:
    n_arms: int
    dist: DistMode = "bernoulli"
    mean_mode: MeanMode = "random"
    means: Optional[Sequence[float]] = None
    gap_delta: Optional[float] = None
    seed: Optional[int] = None
    bernoulli_clip: bool = True  # clip p into [0,1]; and in gap-mode clip negatives to 0


class StochasticBandit:
    """
    Stochastic multi-armed bandit.

    Supports:
      - Bernoulli bandits: reward ~ Bernoulli(p_i) where p_i == mean_i
      - Gaussian bandits:  reward ~ Normal(mean_i, 1)

    Mean generation modes:
      - manual: user-provided means (must have length n_arms)
      - random:
          * gaussian: means i.i.d. N(0,1)
          * bernoulli: means i.i.d. Uniform(0,1)

    Optional gap-mode (gap_delta = Δ):
      After drawing means, find the best arm mean mu* and then replace arm means
      in descending order by: mu*, mu*-Δ, mu*-2Δ, ...
      For Bernoulli: any negative means are clipped to 0 (and values >1 to 1 if bernoulli_clip=True).
    """

    def __init__(self, cfg: StochasticBanditConfig):
        if cfg.n_arms <= 0:
            raise ValueError("n_arms must be positive.")
        if cfg.mean_mode not in ("manual", "random"):
            raise ValueError("mean_mode must be 'manual' or 'random'.")
        if cfg.dist not in ("bernoulli", "gaussian"):
            raise ValueError("dist must be 'bernoulli' or 'gaussian'.")

        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        self.means = self._init_means()
        self._apply_gap_mode_if_needed()

        self.opt_mean = float(np.max(self.means))
        self.opt_arms = np.flatnonzero(self.means == self.opt_mean).tolist()

    def _init_means(self) -> np.ndarray:
        n = self.cfg.n_arms

        if self.cfg.mean_mode == "manual":
            if self.cfg.means is None:
                raise ValueError("manual mean_mode requires 'means' to be provided.")
            if len(self.cfg.means) != n:
                raise ValueError(f"means must have length {n}.")
            means = np.asarray(self.cfg.means, dtype=float)

        else:  # random
            if self.cfg.dist == "gaussian":
                means = self.rng.standard_normal(n)  # N(0,1)
            else:
                means = self.rng.uniform(0.0, 1.0, size=n)  # Uniform(0,1)

            # Allow overriding random with explicit means if provided (optional)
            if self.cfg.means is not None:
                if len(self.cfg.means) != n:
                    raise ValueError(f"means must have length {n}.")
                means = np.asarray(self.cfg.means, dtype=float)

        if self.cfg.dist == "bernoulli" and self.cfg.bernoulli_clip:
            means = np.clip(means, 0.0, 1.0)

        return means

    def _apply_gap_mode_if_needed(self) -> None:
        delta = self.cfg.gap_delta
        if delta is None:
            return
        if delta < 0:
            raise ValueError("gap_delta must be non-negative.")
        # Apply gap after drawing random means
        if self.cfg.mean_mode != "random":
            raise ValueError("gap_delta mode requires mean_mode='random' (per assignment).")
        order = np.argsort(-self.means)  # indices sorted by mean desc
        mu_star = float(self.means[order[0]])

        new_means = np.empty_like(self.means, dtype=float)
        for rank, arm in enumerate(order):
            new_means[arm] = mu_star - rank * delta #mu*, mu*-Δ, mu*-2Δ, ...

        if self.cfg.dist == "bernoulli" and self.cfg.bernoulli_clip:
            # Assignment note: set negative means to 0 for Bernoulli (also clip >1 to 1)
            new_means = np.clip(new_means, 0.0, 1.0)

        self.means = new_means

    def pull(self, arm: int) -> float:
        """Draw a reward from the selected arm."""
        if arm < 0 or arm >= self.cfg.n_arms:
            raise IndexError("arm index out of range.")

        m = float(self.means[arm])

        if self.cfg.dist == "gaussian":
            return float(self.rng.normal(loc=m, scale=1.0))

        # bernoulli
        p = np.clip(m, 0.0, 1.0) if self.cfg.bernoulli_clip else m
        return float(self.rng.binomial(n=1, p=p))

    def expected_regret(self, arm: int) -> float:
        """Instantaneous expected regret: mu* - mu_arm."""
        if arm < 0 or arm >= self.cfg.n_arms:
            raise IndexError("arm index out of range.")
        return float(self.opt_mean - float(self.means[arm]))

    def info(self) -> dict:
        """Convenient debug info."""
        return {
            "n_arms": self.cfg.n_arms,
            "dist": self.cfg.dist,
            "mean_mode": self.cfg.mean_mode,
            "means": self.means.copy(),
            "gap_delta": self.cfg.gap_delta,
            "opt_mean": self.opt_mean,
            "opt_arms": list(self.opt_arms),
        }
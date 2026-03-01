# bandits_project/algos/etc.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class ETCConfig:
    exploration_rounds: int                 # total exploration steps T_explore
    seed: Optional[int] = None              # for tie-breaking/random arm choice have same means


class ETC:
    """
    Explore-Then-Commit (ETC) for stochastic multi-armed bandits.

    - Explore: for the first `exploration_rounds` steps, pull arms in a round-robin schedule.
    - Commit: afterwards, always pull the arm with the highest empirical mean.

    Interface required by assignment:
      - takes (bandit, exploration_rounds)
      - keeps track of played rounds internally
      - step() returns (chosen_arm, reward)
    """

    def __init__(self, bandit, config: ETCConfig):
        if config.exploration_rounds < 0:
            raise ValueError("exploration_rounds must be non-negative.")
        if not hasattr(bandit, "pull"):
            raise TypeError("bandit must have a pull(arm) method.")
        if not hasattr(bandit, "cfg") or not hasattr(bandit.cfg, "n_arms"):
            raise TypeError("bandit must expose number of arms as bandit.cfg.n_arms (per your setup).")

        self.bandit = bandit
        self.cfg = config
        self.n_arms = int(bandit.cfg.n_arms)

        self.rng = np.random.default_rng(config.seed)

        # internal state
        self.t = 0  # number of steps played so far

        # statistics for empirical means (compute on the fly or keep running sums[i]/counts[i])
        self.counts = np.zeros(self.n_arms, dtype=int) #how often arm i was played
        self.sums = np.zeros(self.n_arms, dtype=float) #sum of rewards observed from arm i

        # commit state
        self._committed_arm: Optional[int] = None # once we enter commit phase, this is the arm we always pull; None if not yet committed

    def _empirical_means(self) -> np.ndarray:
        means = np.zeros(self.n_arms, dtype=float)
        played = self.counts > 0
        means[played] = self.sums[played] / self.counts[played]
        return means

    def _pick_best_arm(self) -> int:
        means = self._empirical_means()
        best_val = np.max(means)
        best_arms = np.flatnonzero(means == best_val) #finds all arms with the same best empirical mean (could be more than 1 due to ties)
        # tie-break randomly (or deterministically if you set a seed)
        return int(self.rng.choice(best_arms)) # randomly pick one of the best arms if there are ties

    def step(self) -> Tuple[int, float]:
        """
        Execute one step of ETC.
        Returns (chosen_arm, observed_reward).
        """
        # Decide arm
        if self.t < self.cfg.exploration_rounds:
            # round-robin exploration phase: just cycle through arms in order
            arm = self.t % self.n_arms
        else:
            # commit phase
            if self._committed_arm is None:
                self._committed_arm = self._pick_best_arm()
            arm = self._committed_arm

        # Pull arm, observe reward
        reward = float(self.bandit.pull(arm))

        # Update stats
        self.counts[arm] += 1
        self.sums[arm] += reward
        self.t += 1

        return arm, reward

    def info(self) -> dict:
        """Convenient debug info."""
        return {
            "t": self.t,
            "exploration_rounds": self.cfg.exploration_rounds,
            "counts": self.counts.copy(),
            "emp_means": self._empirical_means(),
            "committed_arm": self._committed_arm,
        }
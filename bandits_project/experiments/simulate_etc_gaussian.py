# This file simulates ETC on Gaussian bandits and compares different choices of m (exploration pulls per arm).
# It includes the theory m* from Theorem 1.3.2 bound as one of the options.
# It logs and plots cumulative regret, correct action rate, mean estimates, and action probabilities over time for each m choice.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import matplotlib.pyplot as plt

from bandits_project.bandits.stochastic_bandit import StochasticBandit, StochasticBanditConfig
from bandits_project.algos.etc import ETC, ETCConfig
from bandits_project.algos.etc_bound import optimal_m_by_bound


@dataclass
class OnlineMoments: #object to track the mean and variance over many runs
    """Welford online mean/variance for arrays of fixed shape."""
    n: int
    mean: np.ndarray
    m2: np.ndarray

    @staticmethod
    def create(shape: Tuple[int, ...], dtype=float) -> "OnlineMoments":
        return OnlineMoments(n=0, mean=np.zeros(shape, dtype=dtype), m2=np.zeros(shape, dtype=dtype))

    def update(self, x: np.ndarray) -> None: #welford method for online mean/variance
        x = np.asarray(x, dtype=float)
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

    def variance(self) -> np.ndarray:
        if self.n < 2:
            return np.zeros_like(self.mean)
        return self.m2 / (self.n - 1)


def run_one_etc_gaussian(
    K: int,
    n_steps: int,
    m_per_arm: Optional[int],   # None => use theory m*
    seed: int,
    sigma: float = 1.0
) -> Dict[str, np.ndarray]:
    """
    One experiment run:
      - sample Gaussian means ~ N(0,1)
      - choose m (fixed or theory m* from Theorem 1.3.2 bound)
      - run ETC with exploration_rounds = m*K
      - return time-series metrics
    """
    rng = np.random.default_rng(seed)

    bandit = StochasticBandit(
        StochasticBanditConfig(
            n_arms=K,
            dist="gaussian",
            mean_mode="random",
            seed=int(rng.integers(0, 2**31 - 1)), #seed creates other random means, rewards and decisions but is same across different m values for fair comparison
        )
    )

    true_means = bandit.means.copy()
    best_arm = int(np.argmax(true_means))
    mu_star = float(np.max(true_means))
    deltas = mu_star - true_means  # Δ_a

    # choose m (m_used is exploration pulls per arm; total exploration is m_used*K)
    if m_per_arm is None:
        m_used = int(optimal_m_by_bound(deltas, n=n_steps, K=K, sigma=sigma))
    else:
        m_used = int(m_per_arm)

    # ETC needs total exploration rounds = m*K
    etc = ETC(
        bandit,
        ETCConfig(
            exploration_rounds=m_used * K,
            seed=int(rng.integers(0, 2**31 - 1)),  # tie-break seed
        )
    )

    # Logs
    cum_regret = np.zeros(n_steps, dtype=float)
    correct = np.zeros(n_steps, dtype=float)            # 1 if best arm chosen at t
    est_means = np.zeros((n_steps, K), dtype=float)     # hat_mu[t, arm]
    action_onehot = np.zeros((n_steps, K), dtype=float) # one-hot chosen arms

    # Track estimates ourselves
    counts = np.zeros(K, dtype=int)
    sums = np.zeros(K, dtype=float)

    reg = 0.0
    #Mainloop
    for t in range(n_steps):
        arm, reward = etc.step()
        arm = int(arm)
        reward = float(reward)

        counts[arm] += 1
        sums[arm] += reward

        hat = np.zeros(K, dtype=float)
        played = counts > 0
        hat[played] = sums[played] / counts[played]

        reg += mu_star - float(true_means[arm]) #regret is expected instantaneaous regret = mu* - mu_arm, cumulative regret is sum of that over time
        cum_regret[t] = reg
        correct[t] = 1.0 if arm == best_arm else 0.0
        est_means[t] = hat
        action_onehot[t, arm] = 1.0

    return {
        "true_means": true_means,                # (K,)
        "cum_regret": cum_regret,                # (n_steps,)
        "correct": correct,                      # (n_steps,)
        "est_means": est_means,                  # (n_steps, K)
        "action_onehot": action_onehot,          # (n_steps, K)
        "best_arm": np.array([best_arm], int),   # (1,)
        "m_used": np.array([m_used], int),       # (1,)
    }


MType = Union[int, None]


def label_for_m(m: MType) -> str:
    return "theory m*" if m is None else f"m={m}"


def evaluate_m_grid( #test for many m values and aggregate results
    K: int,
    n_steps: int,
    N: int,
    m_values: List[MType],
    base_seed: int = 0,
    sigma: float = 1.0,
    progress_every: int = 50
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    For each m, run N experiments and aggregate means + variances of all required metrics.
    Also logs mean/var of m_used (useful for theory m*).
    """
    results: Dict[str, Dict[str, np.ndarray]] = {}

    for mi, m in enumerate(m_values):
        lbl = label_for_m(m)

        mom_regret = OnlineMoments.create((n_steps,))
        mom_correct = OnlineMoments.create((n_steps,))
        mom_est = OnlineMoments.create((n_steps, K))
        mom_probs = OnlineMoments.create((n_steps, K))
        mom_true_means = OnlineMoments.create((K,))
        mom_m_used = OnlineMoments.create((1,))

        for r in range(N):
            if progress_every and (r % progress_every == 0):
                print(f"[{lbl}] run {r}/{N}")

            seed = base_seed + 100000 * mi + r # every run has a unique seed, but same m has same seeds across runs for fair comparison
            out = run_one_etc_gaussian(K=K, n_steps=n_steps, m_per_arm=m, seed=seed, sigma=sigma)

            mom_regret.update(out["cum_regret"])
            mom_correct.update(out["correct"])
            mom_est.update(out["est_means"])
            mom_probs.update(out["action_onehot"])
            mom_true_means.update(out["true_means"])
            mom_m_used.update(out["m_used"])

        results[lbl] = { #per m one resultblock with all metrics
            "regret_mean": mom_regret.mean,
            "regret_var": mom_regret.variance(),
            "correct_mean": mom_correct.mean,
            "correct_var": mom_correct.variance(),
            "est_mean": mom_est.mean,
            "est_var": mom_est.variance(),
            "prob_mean": mom_probs.mean,     # avg one-hot = prob of choosing each arm
            "prob_var": mom_probs.variance(),
            "true_means_mean": mom_true_means.mean,
            "true_means_var": mom_true_means.variance(),
            "m_used_mean": mom_m_used.mean,
            "m_used_var": mom_m_used.variance(),
        }

    return results


def plot_results(
    K: int,
    n_steps: int,
    results: Dict[str, Dict[str, np.ndarray]],
    m_values: List[MType],
    est_plot_m: MType = None,   # which curve to use for (c) and (d)
):
    t = np.arange(1, n_steps + 1)
    order = [label_for_m(m) for m in m_values]

    # (a) cumulative regret over time
    plt.figure()
    for lbl in order:
        plt.plot(t, results[lbl]["regret_mean"], label=lbl)
    plt.title("ETC: cumulative regret over time (Gaussian, K=10)")
    plt.xlabel("t")
    plt.ylabel("E[cumulative regret]")
    plt.legend()

    # (b) correct action rate over time
    plt.figure()
    for lbl in order:
        plt.plot(t, results[lbl]["correct_mean"], label=lbl)
    plt.title("ETC: correct action rate over time")
    plt.xlabel("t")
    plt.ylabel("P(play best arm)")
    plt.legend()

    # choose which run/curve to use for plots (c) and (d) to keep them readable
    base_lbl = label_for_m(est_plot_m)
    if base_lbl not in results:
        base_lbl = order[0]

    # (c) estimated means vs true means over time (for one chosen m curve)
    plt.figure()
    for arm in range(K):
        plt.plot(t, results[base_lbl]["est_mean"][:, arm], label=f"hat_mu arm {arm}")
    true_avg = results[base_lbl]["true_means_mean"]
    for arm in range(K):
        plt.hlines(true_avg[arm], xmin=1, xmax=n_steps, linestyles="dashed")
    plt.title(f"ETC: mean estimates over time (using {base_lbl}) vs avg true means (dashed)")
    plt.xlabel("t")
    plt.ylabel("mean")
    plt.legend(ncol=2, fontsize=8)

    # (d) average probability of choosing each arm over time (for one chosen m curve)
    plt.figure()
    for arm in range(K):
        plt.plot(t, results[base_lbl]["prob_mean"][:, arm], label=f"arm {arm}")
    plt.title(f"ETC: P(choose arm) over time (using {base_lbl})")
    plt.xlabel("t")
    plt.ylabel("probability")
    plt.legend(ncol=2, fontsize=8)

    plt.show()


def best_label_by_final_regret(results: Dict[str, Dict[str, np.ndarray]]) -> str:
    return min(results.keys(), key=lambda lbl: float(results[lbl]["regret_mean"][-1]))


def main():
    K = 10
    n_steps = 10_000
    N = 1_000

    # m is exploration pulls PER ARM; total exploration is m*K.
    # Add None to include theory m* from Theorem 1.3.2 bound.
    m_values: List[MType] = [1, 2, 5, 10, 20, 50, 100, None]

    # Gaussian with variance 1 is 1-subgaussian -> sigma=1 is standard here.
    results = evaluate_m_grid(
        K=K,
        n_steps=n_steps,
        N=N,
        m_values=m_values,
        base_seed=0,
        sigma=1.0,
        progress_every=50
    )

    print("\n=== Summary (final mean cumulative regret) ===")
    for m in m_values:
        lbl = label_for_m(m)
        final_reg = results[lbl]["regret_mean"][-1]
        m_used_mean = results[lbl]["m_used_mean"][0]
        print(f"{lbl:10s}  final_regret={final_reg:10.3f}   avg_m_used={m_used_mean:8.3f}")

    best_lbl = best_label_by_final_regret(results)
    print("\nBest (empirical) by final regret:", best_lbl)

    # for (c) and (d), pick the theory curve by default (readable); change if you like.
    plot_results(K, n_steps, results, m_values, est_plot_m=None)


if __name__ == "__main__":
    main()
    #zusammenfassend: für jedes m mit den verschiedenen werten spielt es 1000 mal den gaussian banditen
    #ein run erzeugt neuen gaussian banditen mit neuen means, berechnet bestes mean und wählt m oder über bound. das läuft dann 1000 mal. speichert das alles und wertet es für alle verschiedenen m aus.
    
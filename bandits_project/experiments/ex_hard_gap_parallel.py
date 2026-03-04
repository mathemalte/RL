from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from bandits_project.algos.greedy import run_pure_greedy, run_epsilon_greedy
from bandits_project.algos.ucb import run_ucb_hoeffding, run_ucb_subgaussian


# -----------------------------
# Bandit: Bernoulli K-armed
# -----------------------------
@dataclass
class BernoulliBandit:
    means: np.ndarray  # (K,)
    rng: np.random.Generator

    @property
    def K(self) -> int:
        return int(self.means.shape[0])

    def pull(self, a: int) -> float:
        return float(self.rng.random() < float(self.means[a]))


def sample_small_gap_means(K: int, rng: np.random.Generator, delta: float = 0.01) -> np.ndarray:
    """
    Hard instances: best and second-best are separated by a tiny gap delta.
    Remaining arms are worse by at least ~delta to keep problem structured.
    """
    mu_star = rng.uniform(0.5, 0.9)
    means = np.empty(K, dtype=float)

    means[0] = mu_star
    means[1] = mu_star - delta

    for a in range(2, K):
        means[a] = (mu_star - delta) - rng.uniform(delta, 0.2)

    means = np.clip(means, 0.0, 1.0)
    rng.shuffle(means)  # best arm not always arm0
    return means


def cumulative_pseudo_regret(means: np.ndarray, actions: np.ndarray) -> np.ndarray:
    mu_star = float(np.max(means))
    inst_regret = mu_star - means[actions]
    return np.cumsum(inst_regret)


# -----------------------------
# Online curve stats on sampled points
# -----------------------------
@dataclass
class OnlineCurveStats:
    n_points: int
    count: int = 0
    mean: np.ndarray = None
    M2: np.ndarray = None

    def __post_init__(self):
        self.mean = np.zeros(self.n_points, dtype=float)
        self.M2 = np.zeros(self.n_points, dtype=float)

    def update(self, x: np.ndarray) -> None:
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def std(self) -> np.ndarray:
        if self.count < 2:
            return np.zeros(self.n_points, dtype=float)
        return np.sqrt(self.M2 / (self.count - 1))


# -----------------------------
# Run one algo
# -----------------------------
def run_one_algo(
    algo_name: str,
    bandit: BernoulliBandit,
    n_steps: int,
    params: Dict,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    K = bandit.K

    if algo_name == "greedy":
        out = run_pure_greedy(pull=bandit.pull, K=K, n_steps=n_steps, seed=seed)
        return out["actions"], out["rewards"]

    if algo_name.startswith("eps_fixed"):
        out = run_epsilon_greedy(
            pull=bandit.pull, K=K, n_steps=n_steps, epsilon=float(params["epsilon"]), seed=seed
        )
        return out["actions"], out["rewards"]

    if algo_name == "ucb_hoeffding":
        out = run_ucb_hoeffding(pull=bandit.pull, K=K, n_steps=n_steps, delta=params.get("delta", None), seed=seed)
        return out["actions"], out["rewards"]

    if algo_name == "ucb_subg":
        out = run_ucb_subgaussian(pull=bandit.pull, K=K, n_steps=n_steps, sigma=float(params["sigma"]), seed=seed)
        return out["actions"], out["rewards"]

    raise ValueError(f"Unknown algo: {algo_name}")


# -----------------------------
# Worker: one instance (parallel)
# -----------------------------
def run_single_instance(
    i: int,
    K: int,
    n_steps: int,
    sample_idx: np.ndarray,
    algos: List[Tuple[str, Dict]],
    master_seed: int,
    gap_delta: float,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, float], Dict[str, float]]:
    """
    Returns:
      means (K,)
      regret_samples[name] (len(sample_idx),)
      popt_samples[name]   (len(sample_idx),)
      final_regret[name]   float
      prob_opt_end[name]   float
    """
    rng_i = np.random.default_rng(master_seed + 100_000 * i)

    means = sample_small_gap_means(K, rng_i, delta=gap_delta)
    opt_arm = int(np.argmax(means))

    # for popt curve efficiently
    t = np.arange(1, n_steps + 1, dtype=float)

    regret_samples: Dict[str, np.ndarray] = {}
    popt_samples: Dict[str, np.ndarray] = {}
    final_regret: Dict[str, float] = {}
    prob_opt_end: Dict[str, float] = {}

    for algo_name, params in algos:
        # important: independent RNG per algo so ordering doesn't change results
        bandit_rng = np.random.default_rng(int(rng_i.integers(0, 2**32 - 1)))
        bandit = BernoulliBandit(means=means, rng=bandit_rng)

        seed_run = int(rng_i.integers(0, 2**32 - 1))
        actions, _rewards = run_one_algo(algo_name, bandit, n_steps, params, seed_run)

        reg_curve = cumulative_pseudo_regret(means, actions)
        popt_curve = np.cumsum(actions == opt_arm) / t

        regret_samples[algo_name] = reg_curve[sample_idx]
        popt_samples[algo_name] = popt_curve[sample_idx]

        final_regret[algo_name] = float(reg_curve[-1])
        prob_opt_end[algo_name] = float(popt_curve[-1])

    return means, regret_samples, popt_samples, final_regret, prob_opt_end


# -----------------------------
# Plot helpers
# -----------------------------
def plot_curve_stats(
    stats_by_algo: Dict[str, OnlineCurveStats],
    t_points: np.ndarray,
    N: int,
    ylabel: str,
    title: str,
    outpath: str,
) -> None:
    plt.figure(figsize=(10, 6))
    for name, stats in stats_by_algo.items():
        mean = stats.mean
        std = stats.std()
        ci = 1.96 * std / math.sqrt(N)
        plt.plot(t_points, mean, label=name)
        plt.fill_between(t_points, mean - ci, mean + ci, alpha=0.2)
    plt.xlabel("t")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def boxplot_metric(metric_by_algo: Dict[str, np.ndarray], ylabel: str, title: str, outpath: str) -> None:
    labels = list(metric_by_algo.keys())
    data = [metric_by_algo[k] for k in labels]
    plt.figure(figsize=(max(10, 0.6 * len(labels)), 6))
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main():
    # --- experiment knobs ---
    K = 5
    n_steps = 200_000          # <<< mehr steps
    N = 400                    # <<< bei so großem n_steps lieber N moderat halten
    gap_delta = 0.01           # <<< kleine gaps (hard)
    master_seed = 123

    # downsample curve points (log-spaced): keeps memory and plotting fast
    n_plot_points = 600
    sample_idx = np.unique(np.clip(np.round(np.logspace(0, np.log10(n_steps), n_plot_points)).astype(int) - 1, 0, n_steps - 1))
    t_points = sample_idx + 1

    # choose algos: include "bad" epsilons
    algos: List[Tuple[str, Dict]] = [
        ("greedy", {}),
        ("eps_fixed_001", {"epsilon": 0.01}),
        ("eps_fixed_01", {"epsilon": 0.10}),
        ("eps_fixed_03", {"epsilon": 0.30}),
        ("ucb_hoeffding", {"delta": None}),
        ("ucb_subg", {"sigma": 0.5}),
    ]
    algo_names = [n for (n, _) in algos]

    # stats containers
    regret_stats = {name: OnlineCurveStats(len(sample_idx)) for name in algo_names}
    popt_stats = {name: OnlineCurveStats(len(sample_idx)) for name in algo_names}

    final_regrets = {name: np.zeros(N, dtype=float) for name in algo_names}
    prob_opt_end = {name: np.zeros(N, dtype=float) for name in algo_names}

    # parallel execution
    n_jobs = min(10, os.cpu_count() or 10)
    results = Parallel(n_jobs=n_jobs, prefer="processes", verbose=5)(
        delayed(run_single_instance)(
            i=i,
            K=K,
            n_steps=n_steps,
            sample_idx=sample_idx,
            algos=algos,
            master_seed=master_seed,
            gap_delta=gap_delta,
        )
        for i in range(N)
    )

    # aggregate
    for i, (_means, regret_s, popt_s, final_r, prob_end) in enumerate(results):
        for name in algo_names:
            regret_stats[name].update(regret_s[name])
            popt_stats[name].update(popt_s[name])
            final_regrets[name][i] = final_r[name]
            prob_opt_end[name][i] = prob_end[name]

    # plots
    plot_curve_stats(
        regret_stats, t_points=t_points, N=N,
        ylabel="Cumulative pseudo-regret (sampled)",
        title=f"Hard-gap regret over time (Δ={gap_delta}, n={n_steps}, N={N})",
        outpath="hardgap_regret_curves.png",
    )

    plot_curve_stats(
        popt_stats, t_points=t_points, N=N,
        ylabel="P(A_t = a*) (empirical, sampled)",
        title=f"Hard-gap P(optimal arm) over time (Δ={gap_delta}, n={n_steps}, N={N})",
        outpath="hardgap_popt_curves.png",
    )

    boxplot_metric(
        final_regrets,
        ylabel="Final cumulative pseudo-regret",
        title=f"Final regrets (Δ={gap_delta}, n={n_steps}, N={N})",
        outpath="hardgap_box_final_regrets.png",
    )

    boxplot_metric(
        prob_opt_end,
        ylabel="P(play optimal arm) at horizon n",
        title=f"P(optimal arm) at end (Δ={gap_delta}, n={n_steps}, N={N})",
        outpath="hardgap_box_prob_opt.png",
    )

    print("\nSaved figures:")
    print("  hardgap_regret_curves.png")
    print("  hardgap_popt_curves.png")
    print("  hardgap_box_final_regrets.png")
    print("  hardgap_box_prob_opt.png")


if __name__ == "__main__":
    main()
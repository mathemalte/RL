from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from bandits_project.algos.greedy import (
    run_pure_greedy,
    run_epsilon_greedy,
    run_epsilon_greedy_decreasing,
    DecreasingEpsilonByBound,
)
from bandits_project.algos.ucb import run_ucb_hoeffding, run_ucb_subgaussian
from bandits_project.algos.boltzmann import (
    run_boltzmann_softmax,
    run_boltzmann_gumbel_trick,
    run_argmax_with_noise,
    run_gumbel_ucb_style,
)
from bandits_project.algos.policy_gradient import run_policy_gradient, DecayingStepsizeSqrt


# -----------------------------
# Bandit: Bernoulli K-armed
# -----------------------------
@dataclass
class BernoulliBandit:
    means: np.ndarray  # shape (K,)
    rng: np.random.Generator

    @property
    def K(self) -> int:
        return int(self.means.shape[0])

    def pull(self, a: int) -> float:
        p = float(self.means[a])
        return float(self.rng.random() < p)


def sample_random_means(K: int, rng: np.random.Generator) -> np.ndarray:
    return rng.random(K)


# -----------------------------
# Online stats (Welford) for curves
# -----------------------------
@dataclass
class OnlineCurveStats:
    n_points: int
    count: int = 0
    mean: Optional[np.ndarray] = None
    M2: Optional[np.ndarray] = None

    def __post_init__(self):
        self.mean = np.zeros(self.n_points, dtype=float)
        self.M2 = np.zeros(self.n_points, dtype=float)

    def update(self, x: np.ndarray) -> None:
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def variance(self) -> np.ndarray:
        if self.count < 2:
            return np.zeros(self.n_points, dtype=float)
        return self.M2 / (self.count - 1)

    def std(self) -> np.ndarray:
        return np.sqrt(self.variance())


def cumulative_pseudo_regret(means: np.ndarray, actions: np.ndarray) -> np.ndarray:
    mu_star = float(np.max(means))
    inst_regret = mu_star - means[actions]
    return np.cumsum(inst_regret)


def empirical_arm_stats(K: int, actions: np.ndarray, rewards: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(actions, minlength=K).astype(int)
    sums = np.zeros(K, dtype=float)
    np.add.at(sums, actions, rewards)
    est = np.zeros(K, dtype=float)
    mask = counts > 0
    est[mask] = sums[mask] / counts[mask]
    return counts, est


# -----------------------------
# Parameter tuning (compute-efficient)
# -----------------------------
def tune_grid(
    algo_name: str,
    param_grid: List[Dict],
    K: int,
    n_tune: int,
    N_tune: int,
    seed: int,
) -> Dict:
    rng = np.random.default_rng(seed)
    best_params = param_grid[0]
    best_score = float("inf")

    for params in param_grid:
        scores = []
        for _i in range(N_tune):
            means = sample_random_means(K, rng)
            bandit = BernoulliBandit(means=means, rng=np.random.default_rng(rng.integers(0, 2**32 - 1)))
            actions, _rewards, _extra = run_one_algo(
                algo_name, bandit, n_tune, params, seed=int(rng.integers(0, 2**32 - 1))
            )
            reg = cumulative_pseudo_regret(means, actions)
            scores.append(float(reg[-1]))
        score = float(np.mean(scores))
        if score < best_score:
            best_score = score
            best_params = params

    print(f"[tune] {algo_name}: best avg final regret={best_score:.4f} with params={best_params}")
    return best_params


# -----------------------------
# Registry runner for one algorithm
# -----------------------------
def run_one_algo(
    algo_name: str,
    bandit: BernoulliBandit,
    n_steps: int,
    params: Dict,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    K = bandit.K

    if algo_name == "greedy":
        out = run_pure_greedy(pull=bandit.pull, K=K, n_steps=n_steps, seed=seed)
        return out["actions"], out["rewards"], {}

    if algo_name == "eps_fixed":
        out = run_epsilon_greedy(pull=bandit.pull, K=K, n_steps=n_steps, epsilon=params["epsilon"], seed=seed)
        return out["actions"], out["rewards"], {}

    if algo_name == "eps_decay":
        d = float(params["d"])
        C = float(params["C"])
        sched = DecreasingEpsilonByBound(K=K, C=C, d=d)
        out = run_epsilon_greedy_decreasing(pull=bandit.pull, K=K, n_steps=n_steps, epsilon_t=sched, seed=seed)
        return out["actions"], out["rewards"], {"epsilons": out["epsilons"]}

    if algo_name == "ucb_hoeffding":
        out = run_ucb_hoeffding(pull=bandit.pull, K=K, n_steps=n_steps, delta=params.get("delta", None), seed=seed)
        return out["actions"], out["rewards"], {}

    if algo_name == "ucb_subg":
        out = run_ucb_subgaussian(pull=bandit.pull, K=K, n_steps=n_steps, sigma=params["sigma"], seed=seed)
        return out["actions"], out["rewards"], {}

    if algo_name == "boltz_softmax":
        out = run_boltzmann_softmax(pull=bandit.pull, K=K, n_steps=n_steps, theta=params["theta"], seed=seed)
        return out["actions"], out["rewards"], {}

    if algo_name == "boltz_gumbel":
        out = run_boltzmann_gumbel_trick(pull=bandit.pull, K=K, n_steps=n_steps, theta=params["theta"], seed=seed)
        return out["actions"], out["rewards"], {}

    if algo_name == "boltz_noise_cauchy":
        out = run_argmax_with_noise(
            pull=bandit.pull, K=K, n_steps=n_steps, noise="cauchy", base_scale=params["base_scale"], seed=seed
        )
        return out["actions"], out["rewards"], {}

    if algo_name == "gumbel_ucb_style":
        out = run_gumbel_ucb_style(pull=bandit.pull, K=K, n_steps=n_steps, C=params["C"], seed=seed)
        return out["actions"], out["rewards"], {}

    if algo_name == "pg_no_baseline":
        out = run_policy_gradient(
            pull=bandit.pull,
            K=K,
            n_steps=n_steps,
            stepsize=DecayingStepsizeSqrt(alpha0=params["alpha0"]),
            baseline=False,
            seed=seed,
            track_probs=True,
        )
        return out["actions"], out["rewards"], {"last_probs": out["probs"][-1]}

    if algo_name == "pg_with_baseline":
        out = run_policy_gradient(
            pull=bandit.pull,
            K=K,
            n_steps=n_steps,
            stepsize=DecayingStepsizeSqrt(alpha0=params["alpha0"]),
            baseline=True,
            seed=seed,
            track_probs=True,
        )
        return out["actions"], out["rewards"], {"last_probs": out["probs"][-1]}

    raise ValueError(f"Unknown algo: {algo_name}")


# -----------------------------
# Plot helpers
# -----------------------------
def plot_regret_curves(curves: Dict[str, OnlineCurveStats], n_steps: int, N: int, outpath: str) -> None:
    plt.figure(figsize=(10, 6))
    t = np.arange(1, n_steps + 1)

    for name, stats in curves.items():
        mean = stats.mean
        std = stats.std()
        ci = 1.96 * std / math.sqrt(N)
        plt.plot(t, mean, label=name)
        plt.fill_between(t, mean - ci, mean + ci, alpha=0.2)

    plt.xlabel("t")
    plt.ylabel("Cumulative pseudo-regret")
    plt.title("Regret over time (mean ± 95% CI)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_popt_curves(popt_stats: Dict[str, OnlineCurveStats], n_steps: int, N: int, outpath: str) -> None:
    plt.figure(figsize=(10, 6))
    t = np.arange(1, n_steps + 1)

    for name, stats in popt_stats.items():
        mean = stats.mean
        std = stats.std()
        ci = 1.96 * std / math.sqrt(N)
        plt.plot(t, mean, label=name)
        plt.fill_between(t, mean - ci, mean + ci, alpha=0.2)

    plt.xlabel("t")
    plt.ylabel("P(A_t = a*) (empirical)")
    plt.title("Probability of choosing the optimal arm over time (mean ± 95% CI)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def boxplot_estimates(true_means: np.ndarray, est_means_by_algo: Dict[str, np.ndarray], outpath: str) -> None:
    _N, K = true_means.shape
    labels = []
    data = []

    for a in range(K):
        labels.append(f"true arm{a}")
        data.append(true_means[:, a])
        for name, est in est_means_by_algo.items():
            labels.append(f"{name} arm{a}")
            data.append(est[:, a])

    plt.figure(figsize=(max(12, 0.35 * len(labels)), 6))
    plt.boxplot(data, showfliers=True, whis=(0, 100))
    plt.xticks(np.arange(1, len(labels) + 1), labels, rotation=90)
    plt.ylabel("Mean / estimate")
    plt.title("True arm means vs algorithm estimates (end of horizon)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def boxplot_prob_opt(play_prob_opt_all: Dict[str, np.ndarray], outpath: str) -> None:
    labels = list(play_prob_opt_all.keys())
    data = [play_prob_opt_all[k] for k in labels]

    plt.figure(figsize=(max(10, 0.6 * len(labels)), 6))
    plt.boxplot(data, labels=labels, showfliers=True, whis=(0, 100))
    plt.ylabel("Empirical P(play optimal arm) at horizon n")
    plt.title("Probability of playing the optimal arm (end of horizon)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def boxplot_final_regrets(final_regrets: Dict[str, np.ndarray], outpath: str) -> None:
    labels = list(final_regrets.keys())
    data = [final_regrets[k] for k in labels]
    plt.figure(figsize=(max(10, 0.6 * len(labels)), 6))
    plt.boxplot(data, labels=labels, showfliers=True, whis=(0, 100))
    plt.ylabel("Final cumulative pseudo-regret")
    plt.title("Final regrets at horizon n")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# -----------------------------
# One parallel instance
# -----------------------------
def run_single_instance(
    i: int,
    K: int,
    n_steps: int,
    algos: List[Tuple[str, Dict]],
    master_seed: int,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, float], Dict[str, np.ndarray], Dict[str, float]]:
    """
    Returns:
      means: (K,)
      reg_curves_i[name]: (n_steps,)
      final_regrets_i[name]: float
      est_means_i[name]: (K,)
      play_prob_opt_i[name]: float
    """
    # deterministischer RNG pro Instanz (parallel-safe)
    rng_i = np.random.default_rng(master_seed + 10_000 * i)

    means = sample_random_means(K, rng_i)
    opt_arm = int(np.argmax(means))

    # model-dependent parameter for eps_decay
    mu_star = float(np.max(means))
    gaps = mu_star - means
    positive_gaps = gaps[gaps > 0]
    realized_min_gap = float(np.min(positive_gaps)) if positive_gaps.size > 0 else 1.0
    d_model = 0.5 * realized_min_gap

    reg_curves_i: Dict[str, np.ndarray] = {}
    final_regrets_i: Dict[str, float] = {}
    est_means_i: Dict[str, np.ndarray] = {}
    play_prob_opt_i: Dict[str, float] = {}

    for algo_name, params in algos:
        params_run = dict(params)
        if algo_name == "eps_decay":
            params_run["d"] = d_model

        # Wichtig: eigener RNG/Bandit pro Algorithmus, damit Algorithmen sich nicht über RNG-Verbrauch beeinflussen
        bandit_rng = np.random.default_rng(int(rng_i.integers(0, 2**32 - 1)))
        bandit = BernoulliBandit(means=means, rng=bandit_rng)

        seed_run = int(rng_i.integers(0, 2**32 - 1))
        actions, rewards, _extra = run_one_algo(algo_name, bandit, n_steps, params_run, seed=seed_run)

        reg_curve = cumulative_pseudo_regret(means, actions)
        reg_curves_i[algo_name] = reg_curve
        final_regrets_i[algo_name] = float(reg_curve[-1])

        counts, est = empirical_arm_stats(K, actions, rewards)
        est_means_i[algo_name] = est
        play_prob_opt_i[algo_name] = float(counts[opt_arm] / n_steps)

    return means, reg_curves_i, final_regrets_i, est_means_i, play_prob_opt_i


# -----------------------------
# Main experiment
# -----------------------------
def run_experiment(prefix: str, do_tune: bool) -> None:
    K = 5
    n_steps = 10_000
    N = 1_000
    master_seed = 123
    rng = np.random.default_rng(master_seed)

    algos = [
        ("greedy", {}),
        ("eps_fixed", {"epsilon": 0.1}),
        ("eps_decay", {"C": 2.0, "d": 0.05}),
        ("ucb_hoeffding", {"delta": None}),
        ("ucb_subg", {"sigma": 0.5}),
        ("boltz_softmax", {"theta": 2.0}),
        ("boltz_gumbel", {"theta": 2.0}),
        ("boltz_noise_cauchy", {"base_scale": 0.2}),
        ("gumbel_ucb_style", {"C": 2.0}),
        ("pg_no_baseline", {"alpha0": 0.2}),
        ("pg_with_baseline", {"alpha0": 0.2}),
    ]

    # ---------- TUNING ----------
    if do_tune:
        n_tune = 2_000
        N_tune = 50

        best = tune_grid(
            "eps_fixed",
            [{"epsilon": e} for e in [0.01, 0.02, 0.05, 0.1, 0.2]],
            K, n_tune, N_tune, seed=master_seed + 1
        )
        algos = [(n, (best if n == "eps_fixed" else p)) for (n, p) in algos]

    algo_names = [n for (n, _) in algos]

    regret_stats: Dict[str, OnlineCurveStats] = {name: OnlineCurveStats(n_steps) for name in algo_names}
    popt_stats: Dict[str, OnlineCurveStats] = {name: OnlineCurveStats(n_steps) for name in algo_names}
    final_regrets: Dict[str, np.ndarray] = {name: np.zeros(N, dtype=float) for name in algo_names}
    play_prob_opt_all: Dict[str, np.ndarray] = {name: np.zeros(N, dtype=float) for name in algo_names}

    t = np.arange(1, n_steps + 1)

    for i in range(N):
        means = sample_random_means(K, rng)
        opt_arm = int(np.argmax(means))

        for algo_name, params in algos:
            bandit_rng = np.random.default_rng(int(rng.integers(0, 2**32 - 1)))
            bandit = BernoulliBandit(means=means, rng=bandit_rng)

            seed_run = int(rng.integers(0, 2**32 - 1))
            actions, rewards, _extra = run_one_algo(algo_name, bandit, n_steps, params, seed=seed_run)

            reg_curve = cumulative_pseudo_regret(means, actions)
            regret_stats[algo_name].update(reg_curve)
            final_regrets[algo_name][i] = reg_curve[-1]

            popt_curve = np.cumsum(actions == opt_arm) / t
            popt_stats[algo_name].update(popt_curve)

            counts, _ = empirical_arm_stats(K, actions, rewards)
            play_prob_opt_all[algo_name][i] = counts[opt_arm] / n_steps

        if (i + 1) % 50 == 0:
            print(f"[{prefix}] {i+1}/{N}")

    plot_regret_curves(regret_stats, n_steps, N, outpath=f"{prefix}_ex4_regret_curves.png")
    plot_popt_curves(popt_stats, n_steps, N, outpath=f"{prefix}_ex4_popt_curves.png")
    boxplot_prob_opt(play_prob_opt_all, outpath=f"{prefix}_ex4_box_prob_opt.png")
    boxplot_final_regrets(final_regrets, outpath=f"{prefix}_ex4_box_final_regrets.png")

    print(f"\nFinished run: {prefix}")

if __name__ == "__main__":
    run_experiment(prefix="notuned", do_tune=False)
    run_experiment(prefix="tuned", do_tune=True)
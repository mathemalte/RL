from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

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

    @property
    def opt_mean(self) -> float:
        return float(np.max(self.means))

    @property
    def opt_arms(self) -> np.ndarray:
        return np.flatnonzero(self.means == np.max(self.means))

    def pull(self, a: int) -> float:
        p = float(self.means[a])
        return float(self.rng.random() < p)


def sample_random_means(K: int, rng: np.random.Generator) -> np.ndarray:
    # "random means": uniform in [0,1]
    return rng.random(K)


# -----------------------------
# Online stats (Welford) for regret curves
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
        # x shape (n_points,)
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
    # pseudo-regret: sum_{t<=T} (mu* - mu_{A_t})
    mu_star = float(np.max(means))
    inst_regret = mu_star - means[actions]
    return np.cumsum(inst_regret)


def empirical_arm_stats(K: int, actions: np.ndarray, rewards: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # returns: counts (K,), est_means (K,), with est_mean=0 for never-pulled arms
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
    """
    Very lightweight tuning:
      - sample N_tune random bandits
      - run each candidate for n_tune steps
      - choose parameter set minimizing average final pseudo-regret
    """
    rng = np.random.default_rng(seed)
    best_params = param_grid[0]
    best_score = float("inf")

    for params in param_grid:
        scores = []
        for i in range(N_tune):
            means = sample_random_means(K, rng)
            bandit = BernoulliBandit(means=means, rng=np.random.default_rng(rng.integers(0, 2**32 - 1)))

            actions, rewards, _extra = run_one_algo(algo_name, bandit, n_tune, params, seed=int(rng.integers(0, 2**32 - 1)))
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
        # needs model-dependent d (gap lower bound) in lecture. For random means we approximate using true gaps.
        # Here we set d as a fraction of the realized gap (model-dependent).
        d = float(params["d"])
        C = float(params["C"])
        sched = DecreasingEpsilonByBound(K=K, C=C, d=d)
        out = run_epsilon_greedy_decreasing(pull=bandit.pull, K=K, n_steps=n_steps, epsilon_t=sched, seed=seed)
        return out["actions"], out["rewards"], {"epsilons": out["epsilons"]}

    if algo_name == "ucb_hoeffding":
        out = run_ucb_hoeffding(pull=bandit.pull, K=K, n_steps=n_steps, delta=params.get("delta", None), seed=seed)
        return out["actions"], out["rewards"], {}

    if algo_name == "ucb_subg":
        # For Bernoulli in [0,1], a common subgaussian proxy is sigma=0.5 (bounded -> Hoeffding-type).
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


def boxplot_estimates(true_means: np.ndarray, est_means_by_algo: Dict[str, np.ndarray], outpath: str) -> None:
    """
    true_means: shape (N, K)
    est_means_by_algo[name]: shape (N, K)
    """
    N, K = true_means.shape
    labels = []
    data = []

    # for each arm: add true + each algo estimate
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


def boxplot_probs(prob_by_algo: Dict[str, np.ndarray], outpath: str) -> None:
    """
    prob_by_algo[name]: shape (N, K) with empirical frequencies (#pulls/n)
    """
    labels = []
    data = []
    for name, probs in prob_by_algo.items():
        K = probs.shape[1]
        for a in range(K):
            labels.append(f"{name} arm{a}")
            data.append(probs[:, a])

    plt.figure(figsize=(max(12, 0.35 * len(labels)), 6))
    plt.boxplot(data, showfliers=True, whis=(0, 100))
    plt.xticks(np.arange(1, len(labels) + 1), labels, rotation=90)
    plt.ylabel("Empirical play probability")
    plt.title("Play probabilities per arm (empirical, end of horizon)")
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
# Main experiment
# -----------------------------
def main():
    K = 5
    n_steps = 10_000
    N = 1_000

    master_seed = 123
    rng = np.random.default_rng(master_seed)

    # ----- define algorithms (with some lecture-optimal defaults) -----
    # Where lecture gives "optimal" model-dependent params, we pick them from the instance (see eps_decay below).
    algos = [
        ("greedy", {}),
        ("eps_fixed", {"epsilon": 0.1}),  # will be tuned if you want
        ("eps_decay", {"C": 2.0, "d": 0.05}),  # will be adjusted per instance below
        ("ucb_hoeffding", {"delta": None}),  # default 1/n^2 inside runner
        ("ucb_subg", {"sigma": 0.5}),  # Bernoulli bounded proxy
        ("boltz_softmax", {"theta": 2.0}),  # tune
        ("boltz_gumbel", {"theta": 2.0}),   # tune
        ("boltz_noise_cauchy", {"base_scale": 0.2}),  # tune
        ("gumbel_ucb_style", {"C": 2.0}),   # tune
        ("pg_no_baseline", {"alpha0": 0.2}),  # tune
        ("pg_with_baseline", {"alpha0": 0.2}),# tune
    ]

    # ----- OPTIONAL tuning for params not fixed by lecture -----
    # "compute-efficient": tune on smaller horizon & fewer instances
    do_tune = True
    if do_tune:
        n_tune = 2_000
        N_tune = 50

        # eps_fixed epsilon grid
        best = tune_grid(
            "eps_fixed",
            [{"epsilon": e} for e in [0.01, 0.02, 0.05, 0.1, 0.2]],
            K, n_tune, N_tune, seed=master_seed + 1
        )
        algos = [(n, (best if n == "eps_fixed" else p)) for (n, p) in algos]

        # boltz theta
        best_theta = tune_grid(
            "boltz_softmax",
            [{"theta": th} for th in [0.5, 1.0, 2.0, 4.0, 8.0]],
            K, n_tune, N_tune, seed=master_seed + 2
        )
        algos = [(n, (best_theta if n == "boltz_softmax" else p)) for (n, p) in algos]

        best_theta2 = tune_grid(
            "boltz_gumbel",
            [{"theta": th} for th in [0.5, 1.0, 2.0, 4.0, 8.0]],
            K, n_tune, N_tune, seed=master_seed + 3
        )
        algos = [(n, (best_theta2 if n == "boltz_gumbel" else p)) for (n, p) in algos]

        # cauchy base_scale
        best_scale = tune_grid(
            "boltz_noise_cauchy",
            [{"base_scale": s} for s in [0.05, 0.1, 0.2, 0.4]],
            K, n_tune, N_tune, seed=master_seed + 4
        )
        algos = [(n, (best_scale if n == "boltz_noise_cauchy" else p)) for (n, p) in algos]

        # gumbel ucb style C
        best_C = tune_grid(
            "gumbel_ucb_style",
            [{"C": c} for c in [0.5, 1.0, 2.0, 4.0]],
            K, n_tune, N_tune, seed=master_seed + 5
        )
        algos = [(n, (best_C if n == "gumbel_ucb_style" else p)) for (n, p) in algos]

        # policy gradient alpha0
        best_alpha = tune_grid(
            "pg_no_baseline",
            [{"alpha0": a} for a in [0.05, 0.1, 0.2, 0.4]],
            K, n_tune, N_tune, seed=master_seed + 6
        )
        algos = [(n, (best_alpha if n == "pg_no_baseline" else p)) for (n, p) in algos]

        best_alpha2 = tune_grid(
            "pg_with_baseline",
            [{"alpha0": a} for a in [0.05, 0.1, 0.2, 0.4]],
            K, n_tune, N_tune, seed=master_seed + 7
        )
        algos = [(n, (best_alpha2 if n == "pg_with_baseline" else p)) for (n, p) in algos]

    algo_names = [n for (n, _) in algos]

    # ----- accumulators for outputs -----
    regret_stats: Dict[str, OnlineCurveStats] = {name: OnlineCurveStats(n_steps) for name in algo_names}
    final_regrets: Dict[str, np.ndarray] = {name: np.zeros(N, dtype=float) for name in algo_names}
    true_means_all = np.zeros((N, K), dtype=float)
    est_means_all: Dict[str, np.ndarray] = {name: np.zeros((N, K), dtype=float) for name in algo_names}
    play_probs_all: Dict[str, np.ndarray] = {name: np.zeros((N, K), dtype=float) for name in algo_names}

    # ----- main Monte Carlo loop -----
    for i in range(N):
        means = sample_random_means(K, rng)
        true_means_all[i] = means

        # new reward RNG per iteration (common setting)
        bandit_rng = np.random.default_rng(int(rng.integers(0, 2**32 - 1)))
        bandit = BernoulliBandit(means=means, rng=bandit_rng)

        # model-dependent parameter for eps_decay:
        # lecture requires d < min gap; use a fraction of realized min gap.
        mu_star = float(np.max(means))
        gaps = mu_star - means
        positive_gaps = gaps[gaps > 0]
        realized_min_gap = float(np.min(positive_gaps)) if positive_gaps.size > 0 else 1.0
        d_model = 0.5 * realized_min_gap  # ensure d < min gap

        for algo_name, params in algos:
            params_run = dict(params)
            if algo_name == "eps_decay":
                params_run["d"] = d_model

            # independent seed per (iteration,algo) for internal tie-breaking etc.
            seed_run = int(rng.integers(0, 2**32 - 1))
            actions, rewards, _extra = run_one_algo(algo_name, bandit, n_steps, params_run, seed=seed_run)

            # regret curve
            reg_curve = cumulative_pseudo_regret(means, actions)
            regret_stats[algo_name].update(reg_curve)
            final_regrets[algo_name][i] = float(reg_curve[-1])

            # end-of-horizon estimate means + play probs (empirical)
            counts, est = empirical_arm_stats(K, actions, rewards)
            est_means_all[algo_name][i] = est
            play_probs_all[algo_name][i] = counts / n_steps

        if (i + 1) % 50 == 0:
            print(f"[progress] finished {i+1}/{N}")

    # ----- plots -----
    plot_regret_curves(regret_stats, n_steps, N, outpath="ex4_regret_curves.png")
    boxplot_estimates(true_means_all, est_means_all, outpath="ex4_box_estimates.png")
    boxplot_probs(play_probs_all, outpath="ex4_box_play_probs.png")
    boxplot_final_regrets(final_regrets, outpath="ex4_box_final_regrets.png")

    print("\nSaved figures:")
    print("  ex4_regret_curves.png")
    print("  ex4_box_estimates.png")
    print("  ex4_box_play_probs.png")
    print("  ex4_box_final_regrets.png")


if __name__ == "__main__":
    main()
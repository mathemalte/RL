from bandits_project.bandits.stochastic_bandit import StochasticBandit, StochasticBanditConfig
from bandits_project.algos.ucb import run_ucb_hoeffding, run_ucb_subgaussian


def main():
    cfg = StochasticBanditConfig(
        n_arms=5,
        dist="gaussian",
        mean_mode="random",
        gap_delta=0.5,
        seed=2
    )

    # WICHTIG: für fairen Vergleich: pro Algorithmus ein frischer Bandit mit gleichem cfg/seed
    T = 10_000
    K = cfg.n_arms

    print("=== True instance ===")
    bandit0 = StochasticBandit(cfg)
    print("True means:", bandit0.means)
    print("Optimal mean:", bandit0.opt_mean, "Optimal arms:", bandit0.opt_arms)

    print("\n=== UCB Hoeffding (delta default = 1/n^2) ===")
    bandit1 = StochasticBandit(cfg)
    out1 = run_ucb_hoeffding(pull=bandit1.pull, K=K, n_steps=T, seed=0)  # delta=None -> 1/n^2
    print("Estimated means:", out1["q_hat"])
    print("Pull counts:", out1["counts"])

    print("\n=== UCB sigma-subgaussian ===")
    bandit2 = StochasticBandit(cfg)
    out2 = run_ucb_subgaussian(pull=bandit2.pull, K=K, n_steps=T, sigma=1.0, seed=0)
    print("Estimated means:", out2["q_hat"])
    print("Pull counts:", out2["counts"])


if __name__ == "__main__":
    main()
    
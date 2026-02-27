from bandits_project.bandits.stochastic_bandit import StochasticBandit, StochasticBanditConfig
from bandits_project.algos.boltzmann import (
    run_boltzmann_softmax,
    run_boltzmann_gumbel_trick,
    run_argmax_with_noise,
    run_gumbel_ucb_style,
)

def main():
    cfg = StochasticBanditConfig(
        n_arms=5,
        dist="gaussian",
        mean_mode="random",
        gap_delta=0.5,
        seed=2
    )
    T = 10_000
    K = cfg.n_arms

    bandit0 = StochasticBandit(cfg)
    print("True means:", bandit0.means)
    print("Optimal mean:", bandit0.opt_mean, "Optimal arms:", bandit0.opt_arms)

    print("\n=== Boltzmann softmax ===")
    bandit1 = StochasticBandit(cfg)
    out1 = run_boltzmann_softmax(pull=bandit1.pull, K=K, n_steps=T, theta=2.0, seed=0)
    print("q_hat:", out1["q_hat"])
    print("counts:", out1["counts"])

    print("\n=== Boltzmann via Gumbel trick (should behave similarly) ===")
    bandit2 = StochasticBandit(cfg)
    out2 = run_boltzmann_gumbel_trick(pull=bandit2.pull, K=K, n_steps=T, theta=2.0, seed=0)
    print("q_hat:", out2["q_hat"])
    print("counts:", out2["counts"])

    print("\n=== Argmax with arbitrary noise (example: Cauchy) ===")
    bandit3 = StochasticBandit(cfg)
    out3 = run_argmax_with_noise(pull=bandit3.pull, K=K, n_steps=T, noise="cauchy", base_scale=0.2, seed=0)
    print("q_hat:", out3["q_hat"])
    print("counts:", out3["counts"])

    print("\n=== UCB-like Gumbel bonus: Q_hat + sqrt(C/T) * Z ===")
    bandit4 = StochasticBandit(cfg)
    out4 = run_gumbel_ucb_style(pull=bandit4.pull, K=K, n_steps=T, C=2.0, seed=0)
    print("q_hat:", out4["q_hat"])
    print("counts:", out4["counts"])

if __name__ == "__main__":
    main()
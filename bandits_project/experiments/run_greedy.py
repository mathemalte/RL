from bandits_project.bandits.stochastic_bandit import StochasticBandit, StochasticBanditConfig
from bandits_project.algos.greedy import (
    run_pure_greedy,
    run_epsilon_greedy,
    run_epsilon_greedy_decreasing,
    DecreasingEpsilonByBound
)

def main():
    cfg = StochasticBanditConfig(
        n_arms=5,
        dist="gaussian",
        mean_mode="random",
        gap_delta=0.5,
        seed=2
    )
    bandit = StochasticBandit(cfg)

    print("True means:", bandit.means)
    print("Optimal mean:", bandit.opt_mean)
    print("Optimal arms:", bandit.opt_arms)

    K = cfg.n_arms
    T = 10_000

    # ------------------------
    # 1) Pure Greedy
    # ------------------------
    out_greedy = run_pure_greedy(
        pull=bandit.pull,
        K=K,
        n_steps=T,
        seed=0
    )

    print("\nPure Greedy estimated means:", out_greedy["q_hat"])


    # ------------------------
    # 2) Fixed ε-greedy
    # ------------------------
    out_eps = run_epsilon_greedy(
        pull=bandit.pull,
        K=K,
        n_steps=T,
        epsilon=0.1,
        seed=0
    )

    print("\nEpsilon-Greedy estimated means:", out_eps["q_hat"])


    # ------------------------
    # 3) Decreasing ε_t
    # ε_t = min{1, C*K/(d^2 t)}  (aus Vorlesung)
    # ------------------------
    d = 0.4        # muss < echte Gap sein
    C = 2.0
    sched = DecreasingEpsilonByBound(K=K, C=C, d=d)

    out_dec = run_epsilon_greedy_decreasing(
        pull=bandit.pull,
        K=K,
        n_steps=T,
        epsilon_t=sched,
        seed=0
    )

    print("\nDecreasing ε-Greedy estimated means:", out_dec["q_hat"])


if __name__ == "__main__":
    main()
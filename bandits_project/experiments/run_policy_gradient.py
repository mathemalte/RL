from bandits_project.bandits.stochastic_bandit import StochasticBandit, StochasticBanditConfig
from bandits_project.algos.policy_gradient import run_policy_gradient, DecayingStepsizeSqrt


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

    # Ohne Baseline
    print("\n=== Policy Gradient (no baseline) ===")
    bandit1 = StochasticBandit(cfg)
    out1 = run_policy_gradient(
        pull=bandit1.pull,
        K=K,
        n_steps=T,
        stepsize=DecayingStepsizeSqrt(alpha0=0.2),
        baseline=False,
        seed=0,
    )
    print("final theta:", out1["theta"])
    print("last probs:", out1["probs"][-1])

    # Mit Baseline
    print("\n=== Policy Gradient (with baseline) ===")
    bandit2 = StochasticBandit(cfg)
    out2 = run_policy_gradient(
        pull=bandit2.pull,
        K=K,
        n_steps=T,
        stepsize=DecayingStepsizeSqrt(alpha0=0.2),
        baseline=True,
        seed=0,
    )
    print("final theta:", out2["theta"])
    print("last probs:", out2["probs"][-1])


if __name__ == "__main__":
    main()
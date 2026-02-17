from bandits_project.bandits.stochastic_bandit import StochasticBandit, StochasticBanditConfig
#wow hat der das schön gemacht
def main():
    cfg = StochasticBanditConfig(
        n_arms=5,
        dist="bernoulli",
        mean_mode="random",
        gap_delta=0.3,
        seed=1
    )
    bandit = StochasticBandit(cfg)

    print("Means with gap:", bandit.means)
    print("Best mean:", bandit.opt_mean, "Best arms:", bandit.opt_arms)

    # pull each arm a few times
    for a in range(cfg.n_arms):
        rewards = [bandit.pull(a) for _ in range(10)]
        print(f"arm {a} mean={bandit.means[a]:.3f} rewards={rewards}")

if __name__ == "__main__":
    main()
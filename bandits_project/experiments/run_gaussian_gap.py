from bandits_project.bandits.stochastic_bandit import StochasticBandit, StochasticBanditConfig

def main():
    cfg = StochasticBanditConfig(
        n_arms=5,
        dist="gaussian",
        mean_mode="random",
        gap_delta=0.5,
        seed=2
    )
    bandit = StochasticBandit(cfg)

    print("Means with gap:", bandit.means)
    print("Best mean:", bandit.opt_mean, "Best arms:", bandit.opt_arms)

    for _ in range(5):
        print("pull best arm:", bandit.pull(bandit.opt_arms[0]))

if __name__ == "__main__":
    main()
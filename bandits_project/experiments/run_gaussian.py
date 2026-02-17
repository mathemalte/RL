from bandits_project.bandits.stochastic_bandit import (
    StochasticBandit,
    StochasticBanditConfig
)

def main():
    config = StochasticBanditConfig(
        n_arms=3,
        dist="gaussian",
        mean_mode="random",   # random means ~ N(0,1)
        seed=42
    )

    bandit = StochasticBandit(config)

    print("Gaussian means:", bandit.means)

    # ziehe Arm 0 fünfmal (Rewards ~ Normal(mean, 1))
    for i in range(5):
        r = bandit.pull(0)
        print("Reward:", r)

if __name__ == "__main__":
    main()
from bandits_project.bandits.stochastic_bandit import (
    StochasticBandit,
    StochasticBanditConfig
)

# Bernoulli Bandit mit 3 Armen
config = StochasticBanditConfig(
    n_arms=5,
    dist="bernoulli",
    mean_mode="random",
    #mean_mode="manual",
    #means=[0.1, 0.5, 0.9],
    gap_delta=0.2,
    seed=42
)

bandit = StochasticBandit(config)

print("Means:", bandit.means)

# Ziehe Arm 0 fünfmal
for i in range(5):
    reward = bandit.pull(0)
    print("Reward:", reward)
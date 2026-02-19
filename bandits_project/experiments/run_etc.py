# bandits_project/experiments/run_etc.py

from bandits_project.bandits.stochastic_bandit import StochasticBandit, StochasticBanditConfig
from bandits_project.algos.etc import ETC, ETCConfig


def main():
    bandit = StochasticBandit(
        StochasticBanditConfig(
            n_arms=5,
            dist="bernoulli",
            mean_mode="random",
            gap_delta=0.2,
            seed=1,
        )
    )
    print("True means:", bandit.means)

    etc = ETC(bandit, ETCConfig(exploration_rounds=10, seed=0))

    total_reward = 0.0
    for _ in range(30):
        arm, r = etc.step()
        total_reward += r
        print(f"t={etc.t:02d} arm={arm} r={r}")

    print("ETC info:", etc.info())
    print("Total reward:", total_reward)


if __name__ == "__main__":
    main()
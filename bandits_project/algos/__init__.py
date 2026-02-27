from .etc import ETC, ETCConfig
from .etc_bound import etc_regret_bound, optimal_m_by_bound
# bandits_project/algos/__init__.py
from .greedy import (
    run_pure_greedy,
    run_epsilon_greedy,
    run_epsilon_greedy_decreasing,
    DecreasingEpsilonByBound,
)
from .ucb import run_ucb_hoeffding, run_ucb_subgaussian
from .boltzmann import (
    run_boltzmann_softmax,
    run_boltzmann_gumbel_trick,
    run_argmax_with_noise,
    run_gumbel_ucb_style,
    NoiseSpec,
)
import pytest
import numpy as np
import random
import matplotlib.pyplot as plt
from test_import_tsp_graph import config # Do not delete, this imports the config fixture!
from TSPOptimizer import TSPOptimizer


@pytest.fixture
def target(config):
    target = TSPOptimizer(config = config, chain_length=200, inner_chain_length=100)
    return target


def test_swap2opt(target):
    route1 = [0, 1, 2, 3, 4]
    actual1 = target.swap2opt(route1, 1, 2)
    expected1 = route1
    assert actual1 == expected1
    route2 = [8, 3, 11, 10, 2, 4, 9, 6, 5, 1, 7, 0]
    actual2 = target.swap2opt(route2, 3, 8)
    expected2 = [8, 3, 11, 10, 5, 6, 9, 4, 2, 1, 7, 0]
    assert actual2 == expected2
    actual3 = target.swap2opt(route1, 4,3)
    assert actual3 == expected1


def test_control_parameters(target):
    route = list(np.arange(target.problem_size))
    T_start = 10000
    num_it = 100
    data = np.zeros(num_it)
    for b in range(num_it):
        accept = 0
        for i in range(num_it):
            new_route, diff = target.anneal(route, T_start)
            if new_route != route and diff >= 0:
                accept += 1
            route = new_route
        data[b] = accept/num_it
    print(f"Percentage of acceptions: ", np.mean(data))
    # Initial acceptance rate should be between 50% and 80%
    assert 0.5 < np.mean(data) < 0.8


def test_cooling(target):
    gen0 = target.exponential_cooling(10000, 0.9)
    gen4 = target.exponential_cooling(10000, 0.95)
    gen1 = target.boltzmann_cooling(10000)
    gen2 = target.cauchy_cooling(10000)
    gen3 = target.fast_cooling(10000)
    plt.semilogy(np.arange(0,201), list(gen0), linestyle="-", label="Exponential, $c = 0.9$")
    plt.semilogy(np.arange(0,201), list(gen4), linestyle="-", label="Exponential, $c = 0.95$")
    plt.semilogy(np.arange(0,201), list(gen1), linestyle="dashed", label="Boltzmann")
    plt.semilogy(np.arange(0,201), list(gen2), linestyle="-.",label="Cauchy")
    plt.semilogy(np.arange(0,201), list(gen3), linestyle="dotted", label="Fast")
    plt.xlabel("Temperature level (k)")
    plt.ylabel("Temperature")
    plt.title("Cooling Schedules")
    plt.legend()
    plt.savefig("../figures/cooling-schedules.png")
    plt.show()


def test_exponential_cooling_gen(target):
    gen = target.exponential_cooling(1000, 0.95)
    x = 1000/0.95
    for i in range(target.chain_length):
        k = x
        x = next(gen)
        assert np.isclose(x, 0.95 * k )



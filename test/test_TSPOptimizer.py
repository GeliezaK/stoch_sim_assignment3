import pytest
import numpy as np
import random
import matplotlib.pyplot as plt
from test_import_tsp_graph import config # Do not delete, this imports the config fixture!
from TSPOptimizer import TSPOptimizer


@pytest.fixture
def target(config):
    target = TSPOptimizer(config, 100, 100)
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
    T_start = 1000
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
    t = np.arange(0.01, 1, 0.01)
    accept = lambda diff: np.exp(-np.abs(diff)/t)
    plt.plot(t, accept(5), label='5')
    plt.plot(t, accept(10), label='10')
    plt.plot(t, accept(30), label='30')
    plt.plot(t, accept(50), label='50')
    plt.xlabel('Temperature level (k)')
    plt.ylabel('Probability of accepting')
    plt.xlim((1, 0))
    plt.legend(loc='lower right')
    plt.show()

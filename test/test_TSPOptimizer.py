import pytest
import numpy as np
import matplotlib.pyplot as plt
from test_import_tsp_graph import config # Do not delete, this imports the config fixture!
from TSPOptimizer import TSPOptimizer


@pytest.fixture
def target(config):
    target = TSPOptimizer(config.graph, 100, 10)
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


def test_cooling(target):
    k = np.arange(1, 10000, 5)
    temp = target.T(k)
    accept = lambda diff: np.exp(-np.abs(diff)/temp)
    plt.plot(k, temp)
    plt.show()
    plt.plot(k, accept(10), label='10')
    plt.plot(k, accept(30), label='30')
    plt.plot(k, accept(70), label='70')
    plt.plot(k, accept(100), label='100')
    plt.xlabel('Number of iteration (k)')
    plt.ylabel('Probability of accepting')
    plt.legend()
    plt.show()

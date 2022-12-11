from configuration import TSPConfiguration as target
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pytest


@pytest.fixture(scope='session')
def config():
    filepath = '../tsp_configurations/eil51.tsp.txt'
    config = target(filepath).init_config()
    return config


def test_construct_table(config):
    assert (config.table[0, :] == [37, 52]).all()
    assert (config.table[10, :] == [42, 41]).all()
    assert (config.table[-1, :] == [30, 40]).all()


def test_construct_input_graph(config):
    K = config.graph
    assert np.isclose(K[6][9]['distance'], 54.037, 0.01)
    # Draw a subgraph of K
    res = [5, 6, 7, 8, 9]
    subgraph = K.subgraph(res)
    pos = nx.spring_layout(subgraph, weight='distance')
    nx.draw(subgraph, pos=pos, with_labels=True)
    plt.show()


def test_get_distance(config):
    a = [0, 5]
    p = [3, 1]
    q = [5, 1]
    assert config.get_distance(p, q) == 2.0
    assert config.get_distance(a, p) == 5.0

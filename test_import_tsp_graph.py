import import_tsp_graph as target
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pytest


@pytest.fixture
def init_table():
    filepath = './tsp_configurations/eil51.tsp.txt'
    table = target.construct_coordinate_table(filepath)
    return table


def test_construct_table(init_table):
    assert (init_table[0, :] == [37, 52]).all()
    assert (init_table[10, :] == [42, 41]).all()
    assert (init_table[-1, :] == [30, 40]).all()


def test_construct_input_graph(init_table):
    K = target.construct_input_graph(init_table)
    assert np.isclose(K[6][9]['distance'], 54.037, 0.01)
    # Draw a subgraph of K
    res = [5, 6, 7, 8, 9]
    subgraph = K.subgraph(res)
    pos = nx.spring_layout(subgraph, weight='distance')
    nx.draw(subgraph, pos=pos, with_labels=True)
    plt.show()


def test_get_distance():
    a = [0, 5]
    p = [3, 1]
    q = [5, 1]
    assert target.get_distance(p, q) == 2.0
    assert target.get_distance(a, p) == 5.0

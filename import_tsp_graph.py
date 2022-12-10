import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import re
from math import dist

def construct_coordinate_table(filepath):
    """
    Read the coordinates from the file into a numpy-array.
    :param filepath: (string) Path to the file with the city coordinates.
    :return: (n x 2 numpy.array) A numpy-array that contains the two coordinates
    for each of the n cities.

    Example:
    table = construct_coordinate_table(./tsp_configurations/eil51.tsp.txt)
    assert table[0, 0] == 37 #City 1 has coordinates [37, 52]
    assert table[0, 1] == 52
    """
    f = open(filepath, "r")
    dim = 0
    while True:
        line = f.readline()
        if "DIMENSION" in line:
            dim = int(re.search(r'\d+', line).group())
        if "NODE_COORD_SECTION" in line:
            break
    # Construct numpy array to store the city coordinates
    coord_table= np.zeros((dim, 2))
    while True:
        # Read coordinates to table
        line = f.readline()
        values = list(map(int, re.findall(r'\d+', line)))
        if len(values)==3:
            coord_table[values[0]-1, :] = np.array([values[1], values[2]])
        if "EOF" in line:
            break
    return coord_table

def construct_input_graph(table):
    """
    Return the graph that contains all cities from table and their respective distances.
    :param table: (numpy.array) with 2D coordinates of each city
    :return: (networkx.Graph) A completely connected graph with a node for every city. The edges
    have one attribute 'distance' which records the euclidean distance between the two connected cities.
    """
    K = nx.complete_graph(len(table))
    for (a, b) in K.edges:
        K[a][b]['distance'] = get_distance(table[a, :], table[b,:])
    return K


def get_distance(coord_a, coord_b):
    """
    Calculate the distance between two coordinates A and B in a 2D plane.
    :param coord_b: (array of length 2) [x, y] - Coordinate of point B
    :param coord_a: (array of length 2) [x, y] - Coordinate of point A
    :return: (float) distance between A and B
    """
    return dist(coord_a, coord_b)

if __name__ == '__main__':
    filepath = './tsp_configurations/eil51.tsp.txt'
    table = construct_coordinate_table(filepath)
    K = construct_input_graph(table)
    res = [5, 6, 7, 8, 9]
    subgraph = K.subgraph(res)
    pos = nx.spring_layout(subgraph, weight='distance')
    nx.draw(subgraph, pos=pos, with_labels=True)
    plt.show()


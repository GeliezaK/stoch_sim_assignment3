import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import re
from math import dist


class TSPConfiguration:
    def __init__(self, filepath):
        self.filepath = filepath
        self.table = None
        self.graph = None

    def construct_coordinate_table(self):
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
        f = open(self.filepath, "r")
        dim = 0
        while True:
            line = f.readline()
            if "DIMENSION" in line:
                dim = int(re.search(r'\d+', line).group())
            if "NODE_COORD_SECTION" in line:
                break
        # Construct numpy array to store the city coordinates
        coord_table = np.zeros((dim, 2))
        while True:
            # Read coordinates to table
            line = f.readline()
            if "EOF" in line:
                break
            value_strings = line.split()
            values = [int(float(x)) for x in value_strings]
            # values = list(map(int, re.findall(r'\d+', line)))
            if len(values) == 3:
                coord_table[values[0] - 1, :] = np.array([values[1], values[2]])
        return coord_table

    def construct_input_graph(self):
        """
        Return the graph that contains all cities from table and their respective distances.
        :param table: (numpy.array) with 2D coordinates of each city
        :return: (networkx.Graph) A completely connected graph with a node for every city. The edges
        have one attribute 'distance' which records the euclidean distance between the two connected cities.
        """
        K = nx.complete_graph(len(self.table))
        for (a, b) in K.edges:
            K[a][b]['distance'] = self.get_distance(self.table[a, :], self.table[b, :])
        return K

    def get_distance(self, coord_a, coord_b):
        """
        Calculate the distance between two coordinates A and B in a 2D plane.
        :param coord_b: (array of length 2) [x, y] - Coordinate of point B
        :param coord_a: (array of length 2) [x, y] - Coordinate of point A
        :return: (float) distance between A and B
        """
        return dist(coord_a, coord_b)

    def init_config(self):
        self.table = self.construct_coordinate_table()
        self.graph = self.construct_input_graph()
        print("Initialized config!")
        return self

    def plot_2D(self, route=[]):
        plt.plot(self.table[:,0], self.table[:,1], 'o')
        plt.axis('equal')
        plt.title("Best solution found for a280 with Cauchy cooling\nwith a chain length of 500")
        plt.xlabel("X-Coordinate")
        plt.ylabel("Y-Coordinate")
        if route != []:
            for i in range(0, len(route)):
                city_i = route[i]
                city_j = route[i + 1] if i + 1 < len(route) else route[0]
                x1, x2 = self.table[city_i, 0], self.table[city_j, 0]
                y1, y2 = self.table[city_i, 1], self.table[city_j, 1]
                plt.plot([x1, x2], [y1, y2], 'r-')
        plt.savefig("figures/cauchy_numit25_chain500_a280_bestsolution.png")
        plt.show()

    def length(self, route):
        """Determine the total length of a route in this configuration."""
        length = 0
        for ind in range(len(route)):
            city_i = route[ind]
            city_j = route[ind +1] if ind+1 < len(route) else route[0]
            length += self.graph[city_i][city_j]['distance']
        return length

if __name__ == '__main__':
    filepath = './tsp_configurations/a280.tsp.txt'
    config = TSPConfiguration(filepath).init_config()
    route = []
    with open("data/cauchy_numit25_chain500_a280.csv_min_route") as f:
        for line in f:
            route.append(int(float(line)))
    # route = np.random.permutation(len(config.table))
    # print(min(route))
    config.plot_2D(route)
    print(config.length(route))

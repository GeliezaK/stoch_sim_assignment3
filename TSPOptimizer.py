import numpy as np
import random
import re


def read_optimal_route(filepath):
    """Returns the optimal route as a list"""
    f = open(filepath, "r")
    while True:
        line = f.readline()
        if "TOUR_SECTION" in line:
            break
    # Construct list
    opt_route = []
    while True:
        # Read cities to list
        line = f.readline()
        if "EOF" in line:
            break
        node = int(re.search(r'\d+', line).group())
        if node > 0:
            opt_route.append(node)
    return opt_route


class TSPOptimizer:

    def __init__(self, input_graph, chain_length, discard_index, cooling_schedule='log'):
        """

        :param input_graph: The TSP problem as a networkx.graph where each node is a city and each edge between nodes is
        a connection between cities.
        :param cooling_schedule: The cooling schedule T(i) for the simulated annealing. A function that returns the
        temperature given the current iteration i.
        :param chain_length: The length of the Markov chain n.
        :param discard_index: The index of the iteration k until when results are discarded. The Markov chain from index
        k+1 to n is kept.
        """
        self.input_graph = input_graph
        self.problem_size = len(input_graph)
        self.cooling_schedule = cooling_schedule
        self.chain_length = chain_length
        self.discard_index = discard_index

    def find_optimum(self):
        """"""
        # Step 1: choose random permutation of nodes as start path
        route = np.random.permutation(self.problem_size)
        print(route)
        # Step 2: for n which is the length of the markov chain: Perform a step of simulated annealing
        for k in range(self.chain_length):
            route = self.anneal(route, k)
        return route

    def anneal(self, route, k):
        # Step 1: Choose two random cities i and j, such that i != j and i, j != 0
        available_indices = list(np.arange(1, self.problem_size))
        city_i, city_j = random.sample(available_indices, k=2)
        # Sort indices ascending
        a, b = [city_i, city_j] if city_i < city_j else [city_j, city_i]
        # Calculate cost difference between old and new route
        diff = self.cost_difference(route, a, b)
        if diff < 0:
            # Step 3a: if length(new route) < length(old route): always accept
            route = self.swap2opt(route, city_i, city_j)
        else:
            # Step 3b: else draw random number U from (0,1). If U < e^{-|cost difference|/T(k)}, accept new route anyways
            rand = np.random.random()
            if rand < np.exp(-np.abs(diff) / self.T(k)):
                print("Cost difference: ", diff)
                print(f'Accept new route! Rand = {np.round(rand, 2)} '
                      f'< {np.round(np.exp(-np.abs(diff) / self.T(k)), 2)}')
                route = self.swap2opt(route, city_i, city_j)
        return route

    def swap2opt(self, route, a, b):
        """Perform a 2-opt swap of the route. """
        route = list(route)
        new_route = [None] * len(route)
        # Take route[start] to route[a] and add to new route in order
        new_route[0:a + 1] = route[0:a + 1]
        # Take route[a + 1] to route[b] and add to new route in reverse order
        segment = route[a + 1:b + 1]
        segment.reverse()
        new_route[a + 1:b + 1] = segment
        # Take route[b + 1] to route[end] and add to new route in order
        new_route[b + 1:] = route[b + 1:]
        return new_route

    def cost_difference(self, route, v1, v2):
        # New distance:  dist(city_i, city_j) + dist(city_i+1, city_j+1)
        # Old distance: dist(city_i, city_i+1) + dist(city_j, city_j+1)
        assert v1 < v2
        v2 = 0 if v2 + 1 == len(route) else v2
        diff = - self.dist(route[v1], route[v1 + 1]) - self.dist(route[v2], route[v2 + 1]) + self.dist(route[v1 + 1],
                                                                                                       route[
                                                                                                           v2 + 1]) + self.dist(
            route[v1], route[v2])
        return diff

    def dist(self, a, b):
        """Return the distance between node a and b in the input graph. """
        return self.input_graph[a][b]['distance']

    def T(self, k):
        """Cooling schedule"""
        T0 = 1000
        return T0 / (1 + np.log(1 + k))

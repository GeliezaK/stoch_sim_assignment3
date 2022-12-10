

class TSPOptimizer:

    def __init__(self, input_graph, cooling_schedule, chain_length, discard_index):
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
        self.cooling_schedule = cooling_schedule
        self.chain_length = chain_length
        self.discard_index = discard_index

    # TODO: find the global optimum with simulated annealing


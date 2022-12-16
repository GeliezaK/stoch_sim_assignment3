import numpy as np
import random


class TSPOptimizer:

    def __init__(self, config, chain_length, inner_chain_length):
        """
        Initialize the TSP-Optimizer.
        :param config: The TSP Configuration
        :param cooling_schedule: The cooling schedule T(i) for the simulated annealing. A function that returns the
        temperature given the current iteration i.
        :param chain_length: The length of the Markov chain n.
        :param inner_chain_length: The index of the iteration k until when results are discarded. The Markov chain from index
        k+1 to n is kept.
        """
        self.config = config
        self.problem_size = len(config.graph)
        self.chain_length = chain_length
        self.inner_chain_length = inner_chain_length

    def find_optimum(self, cooling_schedule='fast', Tstart=10000,
                     Tstep=None):
        """

        :param cooling_schedule: (string) the name of the cooling schedule. Possible values: 'exponential', 'boltzmann',
    'cauchy' or 'fast'. Default = 'fast'. If 'exponential' is specified, the parameter Tstep must be given additionally!
        :param Tstart: (float) Start temperature. Default=10000
        :param Tstep: (float) Scaling parameter for 'exponential' cooling. Must be 0 < Tstep < 1. In the
    exponential cooling, the temperature at level k is calculated as T(k) = Tstart*(Tstep**k).
        :return:
        """
        genTemp = self.init_cooling_schedule(Tstart, Tstep, cooling_schedule)

        # Step 1: choose random permutation of nodes as start path
        route = np.random.permutation(self.problem_size)
        # Step 2: for n which is the length of the markov chain: Perform a step of simulated annealing
        T_k = next(genTemp)
        chain_ind = 0
        while chain_ind < self.chain_length:
            num_reductions = 0
            print(f"Temperature: {np.round(T_k, 4)}, k: {chain_ind}")
            for i in range(self.inner_chain_length * self.problem_size):
                # Arbitrarily set the number of changes per temperature level to 100 times the number of cities
                route, diff = self.anneal(route, T_k)
                if diff < 0:
                    num_reductions += 1
                if num_reductions >= 10 * self.problem_size:
                    # After 10 times the number of length reductions as the number of cities are found, continue with
                    # next temperature level
                    print(f"Decrease temperature after {i} iterations.")
                    break

            yield route
            # Increase temperature level and chain length and repeat
            T_k = next(genTemp)
            chain_ind += 1

    def init_cooling_schedule(self, Tstart, Tstep, cooling_schedule):
        """
        Initialize the temperature generator with the specified cooling schedule.
        :param Tstart: (float) Start temperature
        :param Tstep: (float) Scaling parameter for 'exponential' cooling. Must be 0 < Tstep < 1. In the
    exponential cooling, the temperature at level k is calculated as T(k) = Tstart*(Tstep**k).
        :param cooling_schedule: (string) the name of the cooling schedule. Possible values: 'exponential', 'boltzmann',
    'cauchy' or 'fast'. Default = 'fast'. If 'exponential' is specified, the parameter Tstep must be given additionally!
        :return: generator object that gives the current temperature.
        """
        if cooling_schedule == 'exponential':
            assert Tstep != None, 'If exponential cooling schedule is chosen, you have to provide Tstep!'
            assert 0 < Tstep != 1, 'Tstep must be between 0 and 1: 0 < Tstep < 1. '
            genTemp = self.exponential_cooling(Tstart, Tstep)
        elif cooling_schedule == 'boltzmann':
            genTemp = self.boltzmann_cooling(Tstart)
        elif cooling_schedule == 'cauchy':
            genTemp = self.cauchy_cooling(Tstart)
        else:
            # Choose fast cooling
            genTemp = self.fast_cooling(Tstart)
        return genTemp

    def anneal(self, route, T_k):
        """
        Perform one step of the Metropolis algorithm
        :param route: (list) the current solution
        :param T_k: (float) the current temperature
        :return: route, diff. route is a list with the new solution. diff is a float that specifies the difference
        between the last solution and the new solution. It may be positive.
        """
        # Step 1: Choose two random cities i and j, such that i != j and i, j != 0
        available_indices = list(np.arange(1, self.problem_size))
        while True:
            city_i, city_j = random.sample(available_indices, k=2)
            if np.abs(city_i - city_j) > 2:
                # Do not swap if the resulting route would be the same
                break
        # Sort indices ascending
        a, b = [city_i, city_j] if city_i < city_j else [city_j, city_i]
        # Calculate cost difference between old and new route
        diff = self.cost_difference(route, a, b)
        if diff < 0:
            # Step 3a: if length(new route) < length(old route): always accept
            route = self.swap2opt(route, a, b)
        else:
            # Step 3b: else draw random number U from (0,1). If U < e^{-|cost difference|/T(k)}, accept new route anyways
            rand = np.random.random()
            if rand < np.exp(-np.abs(diff) / T_k):
                route = self.swap2opt(route, a, b)
        return route, diff

    def swap2opt(self, route, a, b):
        """
        Perform a 2-opt swap of the route.
        :param route: (list) the current solution
        :param a: (int) Start index of the path to be reversed.
        :param b: (int) End index of the path to be reversed.
        :return: new_route: (list) the new route after reversing the path from a to b
        """
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
        """
        Determine the cost difference after performing 2-opt.
        :param route:
        :param v1:
        :param v2:
        :return:
        """
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
        return self.config.graph[a][b]['distance']

    def exponential_cooling(self, Tstart, cool_step):
        """Exponential cooling schedule generator. """
        Tprev = Tstart
        k = 0
        yield Tstart
        while k < self.chain_length:
            new_T = cool_step * Tprev
            yield new_T
            Tprev = new_T
            k += 1

    def boltzmann_cooling(self, Tstart):
        """Logarithmic cooling (Boltzmann cooling)"""
        yield Tstart
        k = 1
        while k <= self.chain_length:
            yield Tstart / np.log(k)
            k += 1

    def cauchy_cooling(self, Tstart):
        """Cauchy cooling"""
        yield Tstart
        k = 1
        while k <= self.chain_length:
            yield Tstart / k
            k += 1

    def fast_cooling(self, Tstart):
        """From Ingber(1989): fast exponential cooling."""
        yield Tstart
        k = 1
        Tfinal = 10e-4
        D = 2
        m = - np.log(Tfinal / Tstart)
        n = np.log(self.chain_length)
        c = m * np.exp(-(n / D))
        while k <= self.chain_length:
            cool_step = np.exp(-c * (k ** (1 / D)))
            yield Tstart * cool_step
            k += 1

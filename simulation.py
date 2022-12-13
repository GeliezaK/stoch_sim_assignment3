from configuration import TSPConfiguration
from TSPOptimizer import TSPOptimizer
import re
import numpy as np
import time

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

def simulate(num_it, chain_length = 200, inner_chain_length = 100, cooling_schedule = 'exponential', T_start=1000, Tstep=0.95, C = 1000):
    # Optimize route in configuration
    Opt = TSPOptimizer(config, chain_length=chain_length, inner_chain_length=inner_chain_length)
    print(f"----------------------------- Cooling schedule: {cooling_schedule} --------------------------------- ")

    lengths = np.zeros(num_it)
    for i in range(num_it):
        route = Opt.find_optimum(cooling_schedule=cooling_schedule, Tstart=T_start, Tstep=Tstep, C=C)
        if route == opt_route:
            print("Found the global optimum!")
        config.plot_2D(route)
        lengths[i] = config.length(route)
        print("Length of final route: ", lengths[i])
    return lengths



if __name__ == '__main__':
    # Configure problem
    filepath = './tsp_configurations/eil51.tsp.txt'
    config = TSPConfiguration(filepath).init_config()

    # Read solution
    global opt_route
    optimal_route_path = './tsp_configurations/eil51.opt.tour.txt'
    opt_route = read_optimal_route(optimal_route_path)
    opt_route = [value - 1 for value in opt_route]
    # print(opt_route)
    # config.plot_2D(opt_route)
    # print("Length of optimal route: ", config.length(opt_route[:-1]))

    # Read from csv
    #my_data = np.genfromtxt('data.csv', delimiter=',')
    #print(my_data)

    #lengths = simulate(num_it = 5, chain_length=200, inner_chain_length=100, cooling_schedule='log', C=10000)
    lengths = simulate(num_it = 5, chain_length=200, inner_chain_length=100, cooling_schedule='exponential', T_start=10000, Tstep=0.9)
    print("Minimum: ", np.min(lengths))


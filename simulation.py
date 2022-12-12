from configuration import TSPConfiguration
from TSPOptimizer import TSPOptimizer, read_optimal_route
import numpy as np

def simulate(num_it, T_start, Tstep):
    # Optimize route in configuration
    Opt = TSPOptimizer(config, chain_length=200, inner_chain_length=100, Tstep = Tstep)
    print(f"------------------------- Start temp: {T_start}, Temp step: {Tstep} --------------------------------- ")

    lengths = np.zeros(num_it)
    for i in range(num_it):
        route = Opt.find_optimum(T_start)
        config.plot_2D(route)
        lengths[i] = config.length(route)
        print("Length of final route: ", lengths[i])
    return lengths


if __name__ == '__main__':
    filepath = './tsp_configurations/eil51.tsp.txt'
    config = TSPConfiguration(filepath).init_config()


    # Read from csv
    my_data = np.genfromtxt('data.csv', delimiter=',')
    print(my_data)

    lengths = simulate(5, 1000, 0.95)
    print("Minimum: ", np.min(lengths))

    # Plot the solution
    # optimal_route_path = './tsp_configurations/eil51.opt.tour.txt'
    # opt_route = read_optimal_route(optimal_route_path)
    # opt_route = [value - 1 for value in opt_route]
    # print(opt_route)
    # config.plot_2D(opt_route)
    # print("Length of optimal route: ", config.length(opt_route[:-1]))
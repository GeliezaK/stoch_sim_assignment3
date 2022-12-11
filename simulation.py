from configuration import TSPConfiguration
from TSPOptimizer import TSPOptimizer, read_optimal_route



if __name__ == '__main__':
    filepath = './tsp_configurations/eil51.tsp.txt'
    config = TSPConfiguration(filepath).init_config()

    # Optimize route in configuration
    Opt = TSPOptimizer(config.graph, 10000, 100)
    route = Opt.find_optimum()
    print('Final route: ', route)
    config.plot_2D(route)
    print("Length of final route: ", config.length(route))

    # Plot the solution
    optimal_route_path = './tsp_configurations/eil51.opt.tour.txt'
    opt_route = read_optimal_route(optimal_route_path)
    opt_route = [value - 1 for value in opt_route]
    print(opt_route)
    config.plot_2D(opt_route)
    print("Length of optimal route: ", config.length(opt_route[:-1]))
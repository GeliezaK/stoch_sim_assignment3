from configuration import TSPConfiguration
from TSPOptimizer import TSPOptimizer
import matplotlib.pyplot as plt
import re
import numpy as np
import time

def plot_single_config(data, label, linestyle, color):
    """
    Plots the length of the solution against the temperature level (length of markov chain).
    :param data: (chain_length x num_it numpy.array) Data matrix: the rows correspond to the temperature levels, the
    columns are the individual simulations (one column per simulation)
    :param label: The label to appear in the legend
    :param linestyle: the linestyle of the line to be plotted (e.g. 'dashed', '-', ':',...)
    :param color: the color of the line to be plotted (e.g. 'blue', 'g', 'k', ...)
    """
    chain_length = np.size(data, axis=0)
    num_it = np.size(data, axis=1)
    x = np.arange(1, chain_length +1)
    var = np.var(data, axis=1)
    error = calculate_error(var, num_it)
    assert chain_length == len(x)
    plt.plot(x, np.mean(data, axis=1), linestyle = linestyle, color=color, label=label)
    plt.fill_between(x, np.mean(data, axis=1) - error, np.mean(data, axis=1) + error, color=color, alpha=0.3)


def calculate_error(S, n):
    """
    Returns the length of half the 95% confidence interval.
    :param S: variance
    :param n: number of measurements
    :return: length of half of the 95% confidence interval
    """
    return 1.96 * (np.sqrt(S) / np.sqrt(n))

def read_optimal_route(filepath):
    """
    Read in the optimal route from the txt-file.

    :param filepath: path to the txt-file with the optimal route.
    :return: (list) the route as a list of integers.
    """
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
        if ("EOF" in line) or ("-1" in line):
            break
        node = int(re.search(r'\d+', line).group())
        if node > 0:
            opt_route.append(node)
    return opt_route


def experiment(chain_length=200, inner_chain_length=100, cooling_schedule='exponential', T_start=10000, Tstep=0.95):
    """
    Perform a full simulated annealing optimization.
    :param chain_length: (int) the number of temperature levels (= length of the outer chain)
    :param inner_chain_length: (int) the number of repititions per temperature level. E.g. if inner_chain_lengths=100,
    then 100*(number of cities) path changes are performed per temperature level.
    :param cooling_schedule: (string) the name of the cooling schedule to be used. (default: 'exponential').
    Possible values: 'exponential', 'boltzmann',
    'cauchy' or 'fast'. If 'exponential' is specified, the parameter Tstep must be given additionally!
    :param T_start: (float) The first temperature (default: 10000).
    :param Tstep: (float) A scaling value between 0 and 1. This parameter is only needed if the 'exponential' cooling
     schedule is chosen. In the
    exponential cooling, the temperature at level k is calculated as T(k) = Tstart*(Tstep**k).
    :return: lengths, min_route. Lengths is a (chain_length x 1) numpy.array that records the length of the solution
    route for each temperature level. min_route is a python list of the route with minimal length that was found in this
    simulation.
    """
    # Optimize route in configuration
    Opt = TSPOptimizer(config, chain_length=chain_length, inner_chain_length=inner_chain_length)
    gen = Opt.find_optimum(cooling_schedule=cooling_schedule, Tstart=T_start, Tstep=Tstep)

    # Take random route as initial minimum
    min_route = []
    min_length = np.infty
    lengths = np.zeros(chain_length)
    for i in range(chain_length):
        route = next(gen)
        lengths[i] = config.length(route)
        if lengths[i] < min_length:
            min_route = route
            min_length = lengths[i]
    return lengths, min_route


def simulate_config(chain_length, num_it, cooling_schedule_name, filepath, Tstep=None):
    """
    Performs num_it simulations of one configuration and saves the data to a csv file.

    :param chain_length: (int) the length of the markov chain (= the number of temperature levels)
    :param num_it: (int) the number of simulations to be performed per configuration
    :param cooling_schedule_name: (string) the name of the cooling schedule. Possible values: 'exponential', 'boltzmann',
    'cauchy' or 'fast'. If 'exponential' is specified, the parameter Tstep must be given additionally!
    :param filepath: (string) the path where to store the simulated data.
    :param Tstep: (float) scaling parameter between 0 and 1, only necessary if  'exponential' cooling schedule is chosen. In the
    exponential cooling, the temperature at level k is calculated as T(k) = Tstart*(Tstep**k).
    """
    min_route = []
    min_length = np.infty
    data = np.zeros((chain_length, num_it))
    for j in range(num_it):
        print(f"---------------------- Cooling schedule: {cooling_schedule_name}, Iteration: {j} ---------------------")
        lengths, route = experiment(chain_length=chain_length, inner_chain_length=100, cooling_schedule=cooling_schedule_name,
                             T_start=10000, Tstep=Tstep)
        data[:, j] = np.array(lengths)
        if np.min(lengths) < min_length:
            min_route = route
            min_length = np.min(lengths)
    print("-------------------------------Minimum route length: ", min_length, " ------------------------------")
    config.plot_2D(min_route)
    if min_route == opt_route:
        print("*********************************** FOUND GLOBAL OPTIMUM! *********************************************")
    np.savetxt(filepath+'_min_route', np.array(min_route), delimiter=',')
    np.savetxt(filepath, data, delimiter=',')


def run_simulations():
    global config
    # Configure problem
    filepath = './tsp_configurations/a280.tsp.txt'
    config = TSPConfiguration(filepath).init_config()
    # Read solution
    global opt_route
    optimal_route_path = './tsp_configurations/a280.opt.tour.txt'
    opt_route = read_optimal_route(optimal_route_path)
    opt_route = [value - 1 for value in opt_route]
    # print(opt_route)
    # config.plot_2D(opt_route)
    print("Length of optimal route: ", config.length(opt_route[:-1]))

    # Call this function from console with different cooling schedules:
    # Keep chain length = 200 and number of iterations = 25 for all runs
    # 1. 'exponential', Tstep = 0.95, filepath='exponential095_numit25_a280.csv'
    # 2. 'exponential', Tstep =0.9, filepath='exponential09_numit25_a280.csv'
    # 3. 'boltzmann', Tstep = none, filepath='boltzmann_numit25_a280.csv'
    # 4. 'cauchy', Tstep = none, filepath='cauchy_numit25_a280.csv'
    # 5. 'fast', Tstep = none, filepath='fast_numit25_a280.csv'
    # The data is automatically saved to csv files.
    # simulate_config(chain_length=500, num_it=25, cooling_schedule_name='cauchy',
    #                 filepath='cauchy_numit25_chain500_a280.csv', Tstep=None)


if __name__ == '__main__':
    start = time.time()
    
    run_simulations()

    end = time.time()
    print(f"Simulation took {(end - start) / 60} minutes.")

    # Read from csv
    # In order to plot all data, read the files of each configuration and pass to function plot_single_config()
    # data_exp05 = np.genfromtxt('exponential05_numit25_a280.csv', delimiter=',')
    # data_exp095 = np.genfromtxt('exponential095_numit25_a280.csv', delimiter=',')
    # data_exp09 = np.genfromtxt('exponential09_numit25_a280.csv', delimiter=',')
    # data_boltz = np.genfromtxt('boltzmann_numit25_a280.csv', delimiter=',')
    # data_cauchy = np.genfromtxt('cauchy_numit25_a280.csv', delimiter=',')
    # data_fast = np.genfromtxt('fast_numit25_a280.csv', delimiter=',')

    data_cauchy = np.genfromtxt('cauchy_numit25_chain500_a280.csv', delimiter=',')

    # data_fast = np.genfromtxt('fast_numit25_pcb442.csv', delimiter=',')
    # data_exp095 = np.genfromtxt('exponential095_numit25_pcb442.csv', delimiter=',')

    plt.xlabel("Chain length (k)")
    plt.ylabel("Length of route")

    # plot_single_config(data_exp095, 'Exponential, $c=0.95$', '-', 'blue')
    # plot_single_config(data_exp09, 'Exponential, $c=0.9$', '-', 'cyan')
    # plot_single_config(data_exp05, 'Exponential, $c=0.5$', '-', 'yellow')
    # plot_single_config(data_boltz, 'Boltzmann', '-', 'red')
    plot_single_config(data_cauchy, 'Cauchy', '-', 'green')
    # plot_single_config(data_fast, 'Fast', '-', 'purple')

    plt.axhline(2583, label='Optimal solution', color='black', linestyle='--')

    # plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    # plt.xlim(170, 200)
    # plt.ylim(2500, 4100)

    plt.legend()
    plt.savefig("a280_fast_chain500.png")
    plt.show()
# Assignment 3
This is a discription of the directories and files of this project: 

## data/
The simulated data (lengths of the routes for each cooling schedule and each temperature level). 

## test/ 
Some functions to test the code and especially the initial parameters for the cooling schedules. 

## tsp_configurations/
The .txt TSP configurations that were provided for this assignment.

## simulation.py
Run the main method of this file to run a SA-simulation and plot the convergence over the number of temperature levels. 

## statistical_tests.py
Run the main method of this file to get the results of the statistical (Welch) tests.

## TSPOptimizer.py
Class that implements the Simulated Annealing. The constructor takes a configuration (see next section). The class has a method find_optimum() which takes a cooling schedule and a start temperature. Call this method to perform the SA on this configuration with the desired cooling schedule. 

## configuration.py
A class that implements a configuration object for a TSP problem. It takes a filepath to a .txt file that contains a TSP configuration provided for this assignment. Initialize the configuration by calling init_config(). Now the configuration object has an attribute "table" which is a numpy array that stores the coordinates for each city and an attribute "graph" which is a networkx-Graph that serves as a lookup table for the distances between each pair of cities. 



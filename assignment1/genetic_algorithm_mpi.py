from mpi4py import MPI
import numpy as np
import time
import genetic_algorithms_functions as gaf  # Import your functions

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def parallel_fitness_evaluation(population, distance_matrix):
    """
    Evaluate the fitness of a population in parallel using MPI.
    Each process computes the fitness for a portion of the population.
    """
    n = len(population)
    # Determine each process's share of the population.
    chunk_size = n // size
    start = rank * chunk_size
    # Ensure the last process takes any remainder.
    end = n if rank == size - 1 else start + chunk_size
    local_population = population[start:end]
    
    # Compute local fitness values.
    local_fitness = [gaf.calculate_fitness(route, distance_matrix) for route in local_population]
    
    # Gather all local fitness results at the root process.
    all_fitness = comm.gather(local_fitness, root=0)
    
    if rank == 0:
        # Flatten the list of lists.
        fitness = [f for sublist in all_fitness for f in sublist]
        return fitness
    else:
        return None

if __name__ == '__main__':
    if rank == 0:
        # For example, load the city distance matrix from CSV.
        distance_matrix = np.loadtxt("city_distances.csv", delimiter=",")
        num_nodes = distance_matrix.shape[0]
        # Generate an initial population of candidate routes.
        population = gaf.generate_unique_population(population_size=100, num_nodes=num_nodes)
        start_time = time.time()
    else:
        population = None
        distance_matrix = None
        start_time = None

    # Broadcast the population and the distance matrix to all processes.
    population = comm.bcast(population, root=0)
    distance_matrix = comm.bcast(distance_matrix, root=0)
    
    # Perform parallel fitness evaluation.
    fitness = parallel_fitness_evaluation(population, distance_matrix)
    
    if rank == 0:
        end_time = time.time()
        print("Parallel fitness evaluation took:", end_time - start_time, "seconds")
        print("Fitness values:", fitness)
